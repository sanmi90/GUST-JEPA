"""TCN proxy learner for sensor screening at small K.

Companion to :mod:`src.evaluation.conditional_structural_information` (TCSI).
The ridge learner there is the screening default; this module provides a
~50k-param dilated TCN as a stronger drop-in learner for confirmation runs
at small ``K`` (typically ``K in {2, 3, 4}``). Same input-output contract:
fit a sensor-subset window ``(N, K, W)`` to a target, evaluate on a
held-out fold, report a scalar regression score.

We keep this module separate from ``conditional_structural_information``
on purpose:

* The TCSI module is intentionally pure-NumPy / scikit-learn so it stays
  fast and dependency-light for the 192 * 3 = 576 per-sensor screening
  fits that dominate the pilot.
* The TCN is only needed for the small handful of confirmation fits, and
  carries the much heavier torch dependency.

Architecture summary
====================
Input ``(N, K, W)``: ``N`` samples, ``K`` sensors, ``W=17`` time lags.

* 1x1 channel projection: ``Conv1d(K -> base_channels=32, kernel=1)``.
* Three residual blocks, dilations ``1, 2, 4``::

      x -> Conv1d(kernel=3, dilation=d, pad=d, channels=32) -> GELU -> Dropout
        -> Conv1d(kernel=3, dilation=d, pad=d, channels=32) -> GELU -> Dropout
        -> + x  (identity skip)

* Global average pool over the time axis -> ``(N, 32)``.
* Linear head -> ``(N, out_dim)``.

Total params (K = 2, out_dim = 64): roughly 50k. Train with AdamW
``lr=1e-3``, batch size 32, ``MSE`` loss, configurable epoch count
(default 200, scaled to 100 for the four-follow-ups driver).

Public API
==========
:class:`TCNProxyLearner` mirrors ``sklearn``-style ``fit`` / ``predict``.
Centring is applied in-class (no implicit broadcast surprises) so the
training stats are derived purely from the training fold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TCNConfig:
    """Hyper-parameters for :class:`TCNProxyLearner`.

    Attributes:
        base_channels: Channel width inside the TCN trunk.
        kernel_size: Conv1d kernel size for the dilated blocks.
        dilations: Tuple of per-block dilations; one residual block per entry.
        dropout: Dropout probability inside each block (post-GELU).
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        batch_size: Mini-batch size.
        epochs: Number of training epochs.
        device: PyTorch device string for fit / predict.
        seed: Seed for ``torch.manual_seed`` (deterministic init).
    """

    base_channels: int = 32
    kernel_size: int = 3
    dilations: Tuple[int, ...] = (1, 2, 4)
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    epochs: int = 200
    device: str = "cpu"
    seed: int = 0


class _DilatedResBlock(nn.Module):
    """Single dilated residual block (Conv -> GELU -> Drop -> Conv -> GELU -> Drop -> +x)."""

    def __init__(self, ch: int, kernel: int, dilation: int, dropout: float) -> None:
        super().__init__()
        pad = ((kernel - 1) // 2) * dilation
        self.conv1 = nn.Conv1d(ch, ch, kernel_size=kernel, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(ch, ch, kernel_size=kernel, padding=pad, dilation=dilation)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.act(self.conv1(x)))
        h = self.drop(self.act(self.conv2(h)))
        return x + h


class TCNModule(nn.Module):
    """Tiny dilated TCN regressor.

    Input: ``(B, K_sensors, T_window)``. Output: ``(B, out_dim)``.

    The trunk is a stack of dilated residual blocks at ``base_channels``
    width, followed by global average pooling over time and a linear head.
    """

    def __init__(
        self,
        in_channels: int,
        out_dim: int,
        config: TCNConfig,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Conv1d(in_channels, config.base_channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                _DilatedResBlock(
                    ch=config.base_channels,
                    kernel=config.kernel_size,
                    dilation=d,
                    dropout=config.dropout,
                )
                for d in config.dilations
            ]
        )
        self.head = nn.Linear(config.base_channels, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h)
        h = h.mean(dim=-1)
        return self.head(h)


# ---------------------------------------------------------------------------
# Learner
# ---------------------------------------------------------------------------


class TCNProxyLearner:
    """Drop-in TCN replacement for the screening Ridge learner.

    Same lifecycle as :class:`RidgeProxyLearner` from the TCSI module:
    ``fit(X, y)`` once, then ``predict(X)`` to score a held-out fold.
    Centring is performed internally so callers can pass raw windows.

    Args:
        out_dim: Output dimensionality (e.g. 64 for the encoder latent ``z``,
            1 for ``C_L`` or phase).
        config: Optional override of the default :class:`TCNConfig`. Pass
            ``TCNConfig(epochs=100)`` for the scaled-down 100-epoch variant.
    """

    def __init__(
        self,
        out_dim: int,
        config: Optional[TCNConfig] = None,
    ) -> None:
        if out_dim <= 0:
            raise ValueError(f"out_dim must be positive, got {out_dim}")
        self.out_dim = int(out_dim)
        self.config = config if config is not None else TCNConfig()
        self._model: Optional[TCNModule] = None
        self._y_mean: Optional[np.ndarray] = None
        self._y_std: Optional[np.ndarray] = None
        self._x_mean: Optional[np.ndarray] = None
        self._x_std: Optional[np.ndarray] = None
        self._k_sensors: Optional[int] = None
        self._w_window: Optional[int] = None

    def _normalize_inputs(self, X: np.ndarray) -> np.ndarray:
        if self._x_mean is None or self._x_std is None:
            raise RuntimeError("TCNProxyLearner.fit must be called before _normalize_inputs")
        return (X - self._x_mean) / (self._x_std + 1e-6)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the TCN.

        Args:
            X: Input array of shape ``(N, K_sensors, W)``.
            y: Target array of shape ``(N,)`` or ``(N, out_dim)``.
        """
        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.float32)
        if X_np.ndim != 3:
            raise ValueError(f"X must be (N, K, W), got shape {X_np.shape}")
        if y_np.ndim == 1:
            y_np = y_np[:, None]
        if y_np.ndim != 2:
            raise ValueError(f"y must be 1D or 2D, got shape {y_np.shape}")
        if y_np.shape[1] != self.out_dim:
            raise ValueError(
                f"y last-axis ({y_np.shape[1]}) does not match out_dim ({self.out_dim})"
            )

        n, k, w = X_np.shape
        self._k_sensors = k
        self._w_window = w

        # Per-channel-per-lag standardisation. With K=2..4 sensors and W=17
        # lags this is a 34..68-dim normaliser; cheap and avoids the conv
        # trunk having to learn its own input scale.
        self._x_mean = X_np.mean(axis=0, keepdims=True)
        self._x_std = X_np.std(axis=0, keepdims=True)
        self._x_std = np.where(self._x_std < 1e-8, 1.0, self._x_std)
        Xc = self._normalize_inputs(X_np)

        self._y_mean = y_np.mean(axis=0, keepdims=True)
        self._y_std = y_np.std(axis=0, keepdims=True)
        self._y_std = np.where(self._y_std < 1e-8, 1.0, self._y_std)
        yc = (y_np - self._y_mean) / (self._y_std + 1e-6)

        torch.manual_seed(self.config.seed)
        device = torch.device(self.config.device)
        self._model = TCNModule(
            in_channels=k, out_dim=self.out_dim, config=self.config
        ).to(device)
        optim = torch.optim.AdamW(
            self._model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        X_t = torch.from_numpy(Xc).to(device)
        y_t = torch.from_numpy(yc).to(device)
        bs = min(self.config.batch_size, n)
        rng = np.random.default_rng(self.config.seed)

        self._model.train()
        for _ep in range(self.config.epochs):
            perm = rng.permutation(n)
            for i in range(0, n, bs):
                idx = perm[i : i + bs]
                if idx.size == 0:
                    continue
                xb = X_t[idx]
                yb = y_t[idx]
                optim.zero_grad(set_to_none=True)
                pred = self._model(xb)
                loss = ((pred - yb) ** 2).mean()
                loss.backward()
                optim.step()

        self._model.eval()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict targets.

        Args:
            X: Input array of shape ``(N, K_sensors, W)``.

        Returns:
            Predictions of shape ``(N,)`` if the trained ``out_dim == 1``,
            else ``(N, out_dim)``. Matches the conventional 1D vs 2D
            distinction used by the downstream R^2 helpers.
        """
        if (
            self._model is None
            or self._x_mean is None
            or self._x_std is None
            or self._y_mean is None
            or self._y_std is None
        ):
            raise RuntimeError("TCNProxyLearner.fit must be called before predict")
        X_np = np.asarray(X, dtype=np.float32)
        if X_np.ndim != 3:
            raise ValueError(f"X must be (N, K, W), got shape {X_np.shape}")
        if X_np.shape[1] != self._k_sensors or X_np.shape[2] != self._w_window:
            raise ValueError(
                f"X must be (N, {self._k_sensors}, {self._w_window}), got {X_np.shape}"
            )
        Xc = self._normalize_inputs(X_np)
        device = torch.device(self.config.device)
        X_t = torch.from_numpy(Xc).to(device)
        with torch.no_grad():
            pred = self._model(X_t).cpu().numpy()
        y_hat = pred * (self._y_std + 1e-6) + self._y_mean
        if self.out_dim == 1:
            y_hat = y_hat.reshape(-1)
        return y_hat


__all__ = ["TCNConfig", "TCNModule", "TCNProxyLearner"]
