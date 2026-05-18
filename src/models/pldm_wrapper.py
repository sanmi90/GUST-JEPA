"""PLDM training wrapper composing encoder + predictor + 5-term loss.

Parallel to :class:`src.models.jepa.JEPA`. The architectural difference
is the loss composition: JEPA uses
``L_pred + 0.5 * L_roll + lambda * L_sigreg`` (HANDOFF.md D21); PLDM
uses the 5-term loss defined in :mod:`src.baselines.pldm`
(HANDOFF.md D8 -> D30).

The encoder and predictor are the same modules used by JEPA, so the
SIGReg-vs-PLDM comparison isolates the LOSS as the only difference
(HANDOFF.md D29). The static episode descriptor ``c = (G, D, Y)``
substitutes for the per-step action that the original PLDM paper's
IDM term consumes.

Reference:
    Sobal et al., "Learning from Reward-Free Offline Data: A Case for
    Planning with Latent Dynamics Models." arXiv:2502.14819,
    February 2025. Section 3.3 (L_sim, Eq. 3); Appendix D.1.1
    (collapse-prevention block and combined objective).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.baselines.pldm import PLDMLoss


class PLDMWrapper(nn.Module):
    """Composes encoder + predictor + 5-term PLDM loss into one forward pass.

    The wrapper:

    1. Encodes ``omega`` to ``z`` once.
    2. Rolls out the predictor from ``z[:, :1, :]`` for ``prediction_horizon``
       steps to produce ``z_hat`` of shape ``(B, prediction_horizon + 1, d)``.
    3. Calls :class:`PLDMLoss` on ``z[:, :prediction_horizon + 1, :]`` and
       ``z_hat`` plus the static descriptor ``c``.
    4. Returns the five PLDM terms, ``L_total``, and the cached ``z``.

    Args:
        encoder: ``HybridCNNViTEncoder`` or equivalent
            ``(B, T, 1, H, W) -> (B, T, d)``.
        predictor: ``AutoregressivePredictor`` with a ``rollout(z_init, cond,
            steps)`` method.
        loss: :class:`PLDMLoss` instance carrying all five PLDM weights
            and the inverse-dynamics MLP.
        prediction_horizon: Number of rollout steps from the seed frame.
            The L_sim window covers ``prediction_horizon + 1`` frames
            (the seed plus ``prediction_horizon`` rolled-out steps).
    """

    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        loss: PLDMLoss,
        prediction_horizon: int = 8,
    ) -> None:
        super().__init__()
        if prediction_horizon < 1:
            raise ValueError(f"prediction_horizon must be >= 1, got {prediction_horizon}")
        self.encoder = encoder
        self.predictor = predictor
        self.loss = loss
        self.prediction_horizon = int(prediction_horizon)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Computes the five-term PLDM loss for one training batch.

        Args:
            batch: ``{'omega': (B, T, 1, H, W), 'c': (B, c_dim)}`` with
                ``T >= prediction_horizon + 1``.

        Returns:
            Dict with keys ``L_total``, ``L_sim``, ``L_var``, ``L_cov``,
            ``L_time_sim``, ``L_idm`` (zero-dim fp32 scalars) and ``z``
            (the encoder output ``(B, T, d)`` with autograd attached).
        """
        if "omega" not in batch or "c" not in batch:
            raise KeyError(f"batch must have keys 'omega' and 'c'; got {list(batch.keys())}")
        omega = batch["omega"]
        cond = batch["c"]
        if omega.dim() != 5:
            raise ValueError(f"omega must be (B, T, C, H, W), got {tuple(omega.shape)}")

        B, T = omega.shape[:2]
        H = self.prediction_horizon
        if T < H + 1:
            raise ValueError(
                f"sub-trajectory T={T} too short for prediction_horizon={H}; need T >= H + 1"
            )

        z = self.encoder(omega)
        z_hat = self.predictor.rollout(z[:, :1, :], cond, steps=H)
        loss_out = self.loss(z[:, : H + 1, :], z_hat, cond)
        loss_out["z"] = z
        return loss_out
