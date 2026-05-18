"""End-to-end JEPA wrapper composing encoder, predictor, and anti-collapse loss.

Reference (HANDOFF.md D5, D21):
    Maes, Le Lidec, Scieur, LeCun, Balestriero. "LeWorldModel: Stable
    End-to-End Joint-Embedding Predictive Architecture from Pixels."
    arXiv:2603.19312, 2026, Section 3.1 (two-term `L = L_pred + lambda * L_sigreg` objective).
    Assran et al. "V-JEPA 2." arXiv:2506.09985, 2025 (rollout extension).

Loss composition (CLAUDE.md "Locked decisions, Training"):
    L_total = L_pred + 0.5 * L_roll + lambda * L_anticollapse + eta * L_obs

The anti-collapse term is SIGReg by default and switches to VICReg via
``set_anticollapse`` when the auto-fallback controller fires. The wrapper
itself owns no logic for that decision; it only exposes the swap entrypoint.

Session 6 extensions (SESSION6_FACTORIAL_DIAGNOSTIC.md Step 3):
    - ``c_dropout_prob`` (F-CD variant): per-batch probability of zeroing
      out ``c`` before it reaches the predictor. The encoder is unchanged.
    - ``observable_head`` + ``observable_weight`` (F-OBS variant): an
      auxiliary head that predicts ``CL(t + delta)`` from each ``z_t``,
      added to ``L_total`` with weight ``eta`` (default 0.01).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.models.observable_head import observable_loss
from src.training.scheduled_sampling import (
    open_loop_rollout_loss,
    teacher_forced_prediction_loss,
)


_VALID_ROLLOUT_STRATEGIES = ("fixed_zero", "uniform_random", "impact_aware")


class JEPA(nn.Module):
    """Composes encoder + predictor + anti-collapse loss into one forward pass.

    The wrapper:
        1. Encodes ``omega`` to ``z`` once (shared by all three loss terms).
        2. Computes teacher-forced one-step MSE on ``z``.
        3. Picks a rollout start ``t0`` per ``rollout_start_strategy`` and
           computes the open-loop ``H_roll``-step rollout MSE.
        4. Applies the anti-collapse loss to ``z.flatten(0, 1)``.
        5. Returns the four scalar losses plus the cached ``z`` (autograd-attached;
           detach in the caller if used for downstream diagnostics).

    Args:
        encoder: ``HybridCNNViTEncoder`` (or equivalent ``(B, T, 1, H, W) -> (B, T, d)``).
        predictor: ``AutoregressivePredictor`` (or anything with the same
            ``(z, cond) -> z_hat`` and ``rollout(z_init, cond, steps)`` API).
        anticollapse: ``SIGReg`` or ``VICReg`` (any ``(N, d) -> scalar`` module).
        lambda_anticollapse: Weight on the anti-collapse term. Default 0.1.
        rollout_weight: Weight on the rollout MSE. Default 0.5.
        H_roll: Open-loop rollout horizon. Default 8 (CLAUDE.md).
        rollout_start_strategy: One of ``fixed_zero``, ``uniform_random``,
            ``impact_aware``. Default ``uniform_random``.
    """

    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        anticollapse: nn.Module,
        lambda_anticollapse: float = 0.1,
        rollout_weight: float = 0.5,
        H_roll: int = 8,
        rollout_start_strategy: str = "uniform_random",
        c_dropout_prob: float = 0.0,
        observable_head: nn.Module | None = None,
        observable_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if rollout_start_strategy not in _VALID_ROLLOUT_STRATEGIES:
            raise ValueError(
                f"rollout_start_strategy must be one of {_VALID_ROLLOUT_STRATEGIES}, "
                f"got {rollout_start_strategy!r}"
            )
        if H_roll < 1:
            raise ValueError(f"H_roll must be >= 1, got {H_roll}")
        if not 0.0 <= c_dropout_prob <= 1.0:
            raise ValueError(f"c_dropout_prob must be in [0, 1]; got {c_dropout_prob}")

        self.encoder = encoder
        self.predictor = predictor
        self.anticollapse = anticollapse
        self.lambda_anticollapse = float(lambda_anticollapse)
        self.rollout_weight = float(rollout_weight)
        self.H_roll = int(H_roll)
        self.rollout_start_strategy = rollout_start_strategy
        self.c_dropout_prob = float(c_dropout_prob)
        self.observable_head = observable_head
        self.observable_weight = float(observable_weight)

    def _sample_t0(self, T: int) -> int:
        """Pick the rollout start position according to the configured strategy."""
        t0_max = T - 1 - self.H_roll
        if t0_max < 0:
            raise ValueError(
                f"sub-trajectory T={T} too short for H_roll={self.H_roll}; need T >= H_roll + 2"
            )
        if self.rollout_start_strategy == "fixed_zero":
            return 0
        if self.rollout_start_strategy == "uniform_random":
            return int(torch.randint(low=0, high=t0_max + 1, size=(1,)).item())
        impact_lo = max(0, 24 - self.H_roll)
        impact_hi = min(t0_max, 24)
        if impact_lo > impact_hi or torch.rand(1).item() >= 0.7:
            return int(torch.randint(low=0, high=t0_max + 1, size=(1,)).item())
        return int(torch.randint(low=impact_lo, high=impact_hi + 1, size=(1,)).item())

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Computes the three-term (or four-term, with observable) loss.

        Args:
            batch: ``{'omega': (B, T, 1, H, W), 'c': (B, cond_dim)}`` and
                optionally ``'cl_future': (B, T, n_deltas)`` if the wrapper
                was built with an ``observable_head``.

        Returns:
            dict with keys ``loss_total``, ``loss_pred``, ``loss_roll``,
            ``loss_anticollapse``, ``loss_obs``, ``z``. All loss tensors are
            0-dim scalars (fp32). ``loss_obs`` is zero when no observable
            head is configured. ``z`` is ``(B, T, d)`` with autograd attached.
        """
        if "omega" not in batch or "c" not in batch:
            raise KeyError(f"batch must have keys 'omega' and 'c'; got {list(batch.keys())}")
        omega = batch["omega"]
        cond = batch["c"]
        if omega.dim() != 5:
            raise ValueError(f"omega must be (B, T, C, H, W), got {tuple(omega.shape)}")
        B, T = omega.shape[:2]

        if self.training and self.c_dropout_prob > 0.0:
            if float(torch.rand((), device=cond.device).item()) < self.c_dropout_prob:
                cond = torch.zeros_like(cond)

        z = self.encoder(omega)
        z_hat_tf = self.predictor(z, cond)
        loss_pred = teacher_forced_prediction_loss(z, z_hat_tf)

        if self.rollout_weight == 0.0:
            loss_roll = torch.zeros((), device=z.device, dtype=torch.float32)
        else:
            t0 = self._sample_t0(T)
            loss_roll = open_loop_rollout_loss(
                predictor=self.predictor,
                z_target=z,
                cond=cond,
                start_t=t0,
                horizon=self.H_roll,
            )

        loss_anticollapse = self.anticollapse(z.flatten(0, 1))

        if self.observable_head is not None and self.observable_weight > 0.0:
            if "cl_future" not in batch:
                raise KeyError(
                    "observable head configured but batch has no 'cl_future' tensor"
                )
            cl_pred = self.observable_head(z)
            loss_obs = observable_loss(cl_pred, batch["cl_future"])
        else:
            loss_obs = torch.zeros((), device=z.device, dtype=torch.float32)

        loss_total = (
            loss_pred
            + self.rollout_weight * loss_roll
            + self.lambda_anticollapse * loss_anticollapse
            + self.observable_weight * loss_obs
        )

        return {
            "loss_total": loss_total,
            "loss_pred": loss_pred,
            "loss_roll": loss_roll,
            "loss_anticollapse": loss_anticollapse,
            "loss_obs": loss_obs,
            "z": z,
        }

    def set_anticollapse(self, new_module: nn.Module) -> None:
        """Swap the anti-collapse module in place.

        Used by the auto-fallback controller to switch SIGReg -> VICReg at
        runtime. Both SIGReg and VICReg have no trainable parameters, so
        no optimizer rebuild is needed. The old module's buffers leave the
        module's ``state_dict`` immediately.
        """
        self.anticollapse = new_module
