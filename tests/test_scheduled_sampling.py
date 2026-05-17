"""Tests for ``src.training.scheduled_sampling``."""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor, nn

from src.training.scheduled_sampling import (
    open_loop_rollout_loss,
    teacher_forced_prediction_loss,
)


class _ConstantStubPredictor(nn.Module):
    """Predictor whose ``rollout`` emits a constant value for every new frame."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = float(value)

    def rollout(self, z_init: Tensor, cond: Tensor, steps: int) -> Tensor:
        B, _, d = z_init.shape
        const = torch.full((B, steps, d), self.value, dtype=z_init.dtype, device=z_init.device)
        return torch.cat([z_init, const], dim=1)


class _OraclePredictor(nn.Module):
    """Predictor with access to the future latents (cheating; for tests)."""

    def __init__(self, z_full: Tensor) -> None:
        super().__init__()
        self.register_buffer("z_full", z_full)

    def rollout(self, z_init: Tensor, cond: Tensor, steps: int) -> Tensor:
        T_init = z_init.shape[1]
        return torch.cat([z_init, self.z_full[:, T_init : T_init + steps, :]], dim=1)


def test_teacher_forced_loss_shape_check() -> None:
    torch.manual_seed(0)
    z_target = torch.randn(2, 8, 32)
    z_hat = torch.randn(2, 8, 32)
    loss = teacher_forced_prediction_loss(z_target, z_hat)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_teacher_forced_loss_zero_on_perfect_prediction() -> None:
    """If ``z_hat[:, t, :] = z_target[:, t + 1, :]`` for all valid t, loss = 0."""
    torch.manual_seed(0)
    z_target = torch.randn(2, 8, 32)
    z_hat = torch.zeros_like(z_target)
    z_hat[:, :-1, :] = z_target[:, 1:, :]
    loss = teacher_forced_prediction_loss(z_target, z_hat)
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_teacher_forced_loss_value_on_known_input() -> None:
    """Hand-construct: z_target = ones, z_hat = ones shifted right.

    Specifically z_target[t] = t and z_hat[t] = t. Then z_hat[t] = t and
    z_target[t+1] = t+1, so the per-position error is always -1, MSE = 1.
    """
    T, B, d = 4, 1, 2
    z_target = torch.arange(T, dtype=torch.float32).view(1, T, 1).expand(B, T, d).contiguous()
    z_hat = z_target.clone()
    loss = teacher_forced_prediction_loss(z_target, z_hat)
    assert loss.item() == pytest.approx(1.0, abs=1e-6)


def test_teacher_forced_loss_raises_on_short_trajectory() -> None:
    """T < 2 has no valid next-step targets; loss should refuse to compute."""
    z_target = torch.randn(2, 1, 32)
    z_hat = torch.randn(2, 1, 32)
    with pytest.raises(ValueError):
        teacher_forced_prediction_loss(z_target, z_hat)


def test_rollout_loss_shape_check() -> None:
    torch.manual_seed(0)
    z_target = torch.randn(2, 8, 32)
    cond = torch.randn(2, 3)
    predictor = _ConstantStubPredictor(value=0.0)
    loss = open_loop_rollout_loss(predictor, z_target, cond, start_t=2, horizon=3)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_rollout_loss_uses_predictions_not_ground_truth() -> None:
    """Constant-output stub: loss depends on the constant, not z_target."""
    torch.manual_seed(0)
    z_target = torch.randn(2, 8, 32)
    cond = torch.randn(2, 3)
    loss_zero = open_loop_rollout_loss(
        _ConstantStubPredictor(value=0.0), z_target, cond, start_t=2, horizon=3
    )
    loss_one = open_loop_rollout_loss(
        _ConstantStubPredictor(value=1.0), z_target, cond, start_t=2, horizon=3
    )
    assert not math.isclose(loss_zero.item(), loss_one.item(), abs_tol=1e-4)


def test_rollout_with_perfect_predictor_gives_zero_loss() -> None:
    """Oracle predictor returning ground-truth future latents has zero MSE."""
    torch.manual_seed(0)
    z_target = torch.randn(2, 8, 32)
    cond = torch.randn(2, 3)
    predictor = _OraclePredictor(z_target)
    loss = open_loop_rollout_loss(predictor, z_target, cond, start_t=2, horizon=5)
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_rollout_loss_raises_on_out_of_range_start() -> None:
    z_target = torch.randn(2, 8, 32)
    cond = torch.randn(2, 3)
    predictor = _ConstantStubPredictor(value=0.0)
    with pytest.raises(ValueError):
        open_loop_rollout_loss(predictor, z_target, cond, start_t=7, horizon=2)
    with pytest.raises(ValueError):
        open_loop_rollout_loss(predictor, z_target, cond, start_t=-1, horizon=2)
