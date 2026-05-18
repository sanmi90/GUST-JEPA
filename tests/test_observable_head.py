"""Tests for ``src/models/observable_head.py`` (Session 6 F-OBS variant)."""

from __future__ import annotations

import pytest
import torch

from src.models.observable_head import ObservableHead, observable_loss


def test_observable_head_shape_contract() -> None:
    """Input (B, T, d) -> output (B, T, n_deltas); also (N, d) -> (N, n_deltas)."""
    head = ObservableHead(latent_dim=32, hidden_dim=64, n_deltas=3)
    z = torch.randn(4, 10, 32)
    out = head(z)
    assert out.shape == (4, 10, 3)
    z_flat = torch.randn(7, 32)
    assert head(z_flat).shape == (7, 3)


def test_observable_head_rejects_wrong_last_dim() -> None:
    head = ObservableHead(latent_dim=32, n_deltas=3)
    with pytest.raises(ValueError, match="latent_dim=32"):
        head(torch.randn(2, 10, 16))


def test_observable_head_init_validation() -> None:
    with pytest.raises(ValueError, match="latent_dim must be positive"):
        ObservableHead(latent_dim=0)
    with pytest.raises(ValueError, match="n_deltas must be positive"):
        ObservableHead(latent_dim=32, n_deltas=0)


def test_observable_head_gradient_flows() -> None:
    """One backward call leaves all head parameters with nonzero grads."""
    head = ObservableHead(latent_dim=32, n_deltas=3)
    z = torch.randn(4, 10, 32, requires_grad=True)
    pred = head(z)
    target = torch.randn_like(pred)
    loss = observable_loss(pred, target)
    loss.backward()
    for name, p in head.named_parameters():
        assert p.grad is not None and p.grad.abs().sum().item() > 0.0, (
            f"no gradient on {name}"
        )
    assert z.grad is not None and z.grad.abs().sum().item() > 0.0


def test_observable_loss_zero_when_perfect() -> None:
    pred = torch.randn(2, 8, 3)
    assert observable_loss(pred, pred).item() == pytest.approx(0.0, abs=1e-7)


def test_observable_loss_shape_mismatch_errors() -> None:
    with pytest.raises(ValueError, match="shapes must match"):
        observable_loss(torch.zeros(2, 8, 3), torch.zeros(2, 8, 4))


def test_observable_loss_bf16_input_returns_finite() -> None:
    """Inputs may arrive as bf16 from autocast; the loss promotes to fp32."""
    pred = torch.randn(2, 8, 3, dtype=torch.bfloat16)
    target = torch.randn(2, 8, 3, dtype=torch.bfloat16)
    loss = observable_loss(pred, target)
    assert torch.isfinite(loss)
    assert loss.dtype == torch.float32
