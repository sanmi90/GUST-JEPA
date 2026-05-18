"""Auxiliary observable head for the F-OBS factorial variant (Session 6).

The head maps each per-frame latent ``z_t`` to a small vector of future
lift-coefficient values ``CL(t + delta)`` for ``delta`` in
``cl_future_deltas`` (default ``(8, 16, 24)`` at ``dt_eff = 0.05``, so
``0.4 / 0.8 / 1.2`` convective times into the future).

Motivation (SESSION6_FACTORIAL_DIAGNOSTIC.md Step 3, F-OBS):
    Fukami and Taira's lift-augmented autoencoder (JFM 2023, arXiv:2305.18394)
    showed that adding a small CL prediction head to an otherwise self-
    supervised encoder pressures the latent to retain aerodynamically
    meaningful information without overwhelming the self-supervised
    objective. Solera-Rico et al. (Nat. Commun. 2024, arXiv:2304.03571)
    use a similar weak observable signal on the gust-airfoil dataset.

The head is a small two-layer MLP and is intentionally light: it is the
"weak guidance" branch of the F-OBS hypothesis (D37). All weights live
inside the head; the encoder remains otherwise unconditional.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ObservableHead(nn.Module):
    """Predicts ``CL(t + delta)`` for each frame from the latent ``z_t``.

    Args:
        latent_dim: Encoder latent dimension d (default 32 per HANDOFF D2).
        hidden_dim: Width of the MLP hidden layer (default 64).
        n_deltas: Number of future-CL offsets to predict (default 3 for
            ``(8, 16, 24)`` frames; see ``EpisodeDataset.cl_future_deltas``).
    """

    def __init__(self, latent_dim: int = 32, hidden_dim: int = 64, n_deltas: int = 3) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive; got {latent_dim}")
        if n_deltas <= 0:
            raise ValueError(f"n_deltas must be positive; got {n_deltas}")
        self.latent_dim = int(latent_dim)
        self.n_deltas = int(n_deltas)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_deltas),
        )

    def forward(self, z: Tensor) -> Tensor:
        """Predicts CL futures from a latent sequence.

        Args:
            z: ``(B, T, d)`` per-frame latents. Also accepts ``(N, d)``;
                the leading axes are preserved by the MLP.

        Returns:
            ``(B, T, n_deltas)`` (or ``(N, n_deltas)``) float32 predictions
            of ``CL(t + delta)`` for each delta in the configured offsets.
        """
        if z.shape[-1] != self.latent_dim:
            raise ValueError(
                f"last dim must be latent_dim={self.latent_dim}; got {tuple(z.shape)}"
            )
        return self.net(z)


def observable_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Mean-squared error reduction across all (B, T, n_deltas) entries.

    Args:
        pred: ``(B, T, n_deltas)`` predicted CL futures.
        target: same shape as ``pred``; ground-truth ``CL(t + delta)``.

    Returns:
        Scalar loss (fp32).
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target shapes must match; got {tuple(pred.shape)} vs {tuple(target.shape)}"
        )
    diff = pred.float() - target.float()
    return diff.pow(2).mean()
