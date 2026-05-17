"""Scheduled-sampling losses for JEPA training (V-JEPA 2-AC-faithful).

Reference (HANDOFF.md D21):
    Assran, Duval, Saglio, ... LeCun. "V-JEPA 2: Visual Joint-Embedding
    Predictive Architectures for Visual World Models." arXiv:2506.09985,
    2025, Section 6 and appendices.

The V-JEPA 2-AC recipe sums a teacher-forced one-step loss with an
open-loop rollout loss using fixed coefficients (no Bengio probabilistic
teacher-student mixing). Two transpositions to our setting:

- Teacher-forced one-step loss is computed over all ``T - 1 = 31``
  positions of the sub-trajectory (V-JEPA 2-AC uses 15 positions because
  its architecture exposes 16 frame slots at a time; we have a full
  sub-trajectory).
- Rollout horizon is ``H_roll = 8`` (CLAUDE.md "Locked decisions, Training"),
  larger than V-JEPA 2-AC's ``H_roll = 2`` because vortex impact dynamics
  last 5 to 20 t/c, which the 2-step rollout cannot capture.

The two losses are free functions, not methods on the JEPA wrapper, so
they unit-test in isolation against stub predictors.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def teacher_forced_prediction_loss(z_target: Tensor, z_hat: Tensor) -> Tensor:
    """One-step teacher-forced MSE over ``T - 1`` positions.

    Args:
        z_target: ``(B, T, d)`` ground-truth encoder latents.
        z_hat: ``(B, T, d)`` predictor output, where ``z_hat[:, t, :]`` is
            the prediction of position ``t + 1`` from positions ``[0, t]``.

    Returns:
        Scalar = mean over ``(B, t in [0, T - 2], d)`` of
            ``(z_hat[:, t, :] - z_target[:, t + 1, :])^2``.
    """
    if z_target.shape != z_hat.shape:
        raise ValueError(
            f"z_target and z_hat must agree in shape; got "
            f"{tuple(z_target.shape)} vs {tuple(z_hat.shape)}"
        )
    if z_target.dim() != 3:
        raise ValueError(f"expected (B, T, d), got {tuple(z_target.shape)}")
    if z_target.shape[1] < 2:
        raise ValueError(f"sub-trajectory T must be >= 2, got {z_target.shape[1]}")

    pred = z_hat[:, :-1, :].float()
    target = z_target[:, 1:, :].float()
    return F.mse_loss(pred, target, reduction="mean")


def open_loop_rollout_loss(
    predictor: nn.Module,
    z_target: Tensor,
    cond: Tensor,
    start_t: int,
    horizon: int,
) -> Tensor:
    """Open-loop ``horizon``-step rollout MSE against ground-truth latents.

    Seeds the predictor's ``rollout`` with ``z_target[:, :start_t + 1, :]``
    and computes the MSE between the rolled-out next ``horizon`` frames and
    the ground-truth latents at the same time positions.

    Args:
        predictor: An ``nn.Module`` exposing
            ``rollout(z_init: Tensor, cond: Tensor, steps: int) -> Tensor``
            returning ``(B, T_init + steps, d)`` per the Session 3 spec.
        z_target: ``(B, T, d)`` ground-truth encoder latents.
        cond: ``(B, cond_dim)`` static episode descriptor.
        start_t: Index of the last seed frame. The rollout begins predicting
            frame ``start_t + 1``. Must satisfy ``0 <= start_t`` and
            ``start_t + horizon < T``.
        horizon: Number of rollout steps (``H_roll`` in the project spec).

    Returns:
        Scalar = mean MSE between
            ``z_rollout[:, start_t + 1 : start_t + 1 + horizon, :]`` and
            ``z_target [:, start_t + 1 : start_t + 1 + horizon, :]``.
    """
    if z_target.dim() != 3:
        raise ValueError(f"expected z_target of shape (B, T, d), got {tuple(z_target.shape)}")
    T = z_target.shape[1]
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if start_t < 0 or start_t + horizon >= T:
        raise ValueError(
            f"start_t={start_t} with horizon={horizon} exceeds T={T}; need "
            f"0 <= start_t and start_t + horizon < T"
        )

    z_init = z_target[:, : start_t + 1, :]
    z_full = predictor.rollout(z_init, cond, steps=horizon)
    pred = z_full[:, start_t + 1 : start_t + 1 + horizon, :].float()
    target = z_target[:, start_t + 1 : start_t + 1 + horizon, :].float()
    return F.mse_loss(pred, target, reduction="mean")
