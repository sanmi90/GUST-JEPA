"""Unit tests for src/models/adaln.py."""

from __future__ import annotations

import torch
from torch import nn

from src.models.adaln import AdaLN


def test_adaln_zero_output_at_init() -> None:
    """shift, scale, gate are all exactly zero at init."""
    torch.manual_seed(0)
    hidden_dim = 64
    cond_dim = 16
    adaln = AdaLN(hidden_dim=hidden_dim, cond_dim=cond_dim)
    cond = torch.randn(8, cond_dim)
    shift, scale, gate = adaln(cond)
    assert torch.equal(shift, torch.zeros_like(shift))
    assert torch.equal(scale, torch.zeros_like(scale))
    assert torch.equal(gate, torch.zeros_like(gate))
    assert shift.shape == (8, hidden_dim)
    assert scale.shape == (8, hidden_dim)
    assert gate.shape == (8, hidden_dim)


def test_adaln_block_is_identity_at_init() -> None:
    """A DiT-style transformer block using AdaLN at init returns x unchanged."""
    torch.manual_seed(0)
    hidden_dim = 64
    cond_dim = 16
    adaln = AdaLN(hidden_dim=hidden_dim, cond_dim=cond_dim)
    layer_norm = nn.LayerNorm(hidden_dim)
    mlp = nn.Sequential(
        nn.Linear(hidden_dim, 4 * hidden_dim),
        nn.GELU(),
        nn.Linear(4 * hidden_dim, hidden_dim),
    )

    x = torch.randn(8, 5, hidden_dim)
    cond = torch.randn(8, cond_dim)
    shift, scale, gate = adaln(cond)
    x_normed = layer_norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    x_out = x + gate.unsqueeze(1) * mlp(x_normed)
    assert torch.allclose(x_out, x, atol=1e-6)


def test_adaln_gradient_nonzero_after_step() -> None:
    """After one optimizer step, the AdaLN linear weights are non-zero."""
    torch.manual_seed(0)
    hidden_dim = 64
    cond_dim = 16
    adaln = AdaLN(hidden_dim=hidden_dim, cond_dim=cond_dim)
    cond = torch.randn(8, cond_dim)
    opt = torch.optim.SGD(adaln.parameters(), lr=1.0)

    weight_pre = adaln.linear.weight.detach().clone()
    bias_pre = adaln.linear.bias.detach().clone()
    assert torch.equal(weight_pre, torch.zeros_like(weight_pre))
    assert torch.equal(bias_pre, torch.zeros_like(bias_pre))

    shift, scale, gate = adaln(cond)
    loss = shift.sum() + scale.sum() + gate.sum()
    loss.backward()
    opt.step()

    assert not torch.equal(adaln.linear.weight, torch.zeros_like(adaln.linear.weight))
    assert not torch.equal(adaln.linear.bias, torch.zeros_like(adaln.linear.bias))


def test_adaln_conditioning_broadcasts_over_time() -> None:
    """cond of shape (B, T, cond_dim) yields (B, T, hidden_dim) for each output."""
    torch.manual_seed(0)
    hidden_dim = 64
    cond_dim = 16
    adaln = AdaLN(hidden_dim=hidden_dim, cond_dim=cond_dim)
    cond = torch.randn(4, 7, cond_dim)
    shift, scale, gate = adaln(cond)
    assert shift.shape == (4, 7, hidden_dim)
    assert scale.shape == (4, 7, hidden_dim)
    assert gate.shape == (4, 7, hidden_dim)
