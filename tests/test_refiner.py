"""Unit tests for :mod:`src.models.refiner`.

Contracts verified:
    1. Shape contract on ``(B, 1, 192, 96)`` input -> ``(B, 1, 192, 96)``
       residual output.
    2. The last conv is zero-initialised so the refiner is exactly
       identity at init (residual == 0 for any input).
    3. Parameter count is ~200k (50% tolerance: 100k - 300k).
    4. Forward accepts the optional wake-mask channel and rejects
       mismatched mask shapes.
    5. Gradients flow into the trunk after the zero-init heads are
       broken (simulating any step > 0 of training).
"""

from __future__ import annotations

import pytest
import torch

from src.evaluation.decoder_metrics import wake_mask as build_wake_mask
from src.models.refiner import WakeRefiner


def _make_refiner(**overrides) -> WakeRefiner:
    """Build a refiner with the production default config."""
    defaults = dict(
        in_channels=1,
        channels=64,
        n_blocks=6,
        use_wake_mask=True,
        n_groups=8,
    )
    defaults.update(overrides)
    torch.manual_seed(0)
    return WakeRefiner(**defaults)


def test_refiner_forward_shape() -> None:
    """(B=2, 1, 192, 96) input returns (2, 1, 192, 96) residual."""
    refiner = _make_refiner()
    x = torch.randn(2, 1, 192, 96)
    mask = torch.from_numpy(build_wake_mask(192, 96))
    out = refiner(x, mask)
    assert out.shape == (
        2,
        1,
        192,
        96,
    ), f"refiner output shape {tuple(out.shape)} != (2, 1, 192, 96)"
    assert torch.isfinite(out).all()


def test_refiner_identity_at_init() -> None:
    """Last conv is zero-initialised so the residual is all-zero at init."""
    refiner = _make_refiner()
    x = torch.randn(4, 1, 192, 96)
    mask = torch.from_numpy(build_wake_mask(192, 96))
    out = refiner(x, mask)
    assert torch.allclose(
        out, torch.zeros_like(out), atol=1e-7
    ), "residual at init must be exactly zero (identity refiner)"
    # Also when no mask is supplied (use_wake_mask=True with mask=None).
    out_no_mask = refiner(x, None)
    assert torch.allclose(out_no_mask, torch.zeros_like(out_no_mask), atol=1e-7)


def test_refiner_parameter_count() -> None:
    """Parameter count is ~200k within 50% tolerance (100k - 300k)."""
    refiner = _make_refiner()
    n_params = sum(p.numel() for p in refiner.parameters())
    target = 200_000
    lower = int(target * 0.5)
    upper = int(target * 1.5)
    assert lower <= n_params <= upper, (
        f"WakeRefiner has {n_params:,} params; expected ~{target:,} "
        f"(window {lower:,}..{upper:,})"
    )


def test_refiner_without_mask_input_runs() -> None:
    """use_wake_mask=False produces a refiner that ignores the mask
    argument and still returns the right shape."""
    refiner = _make_refiner(use_wake_mask=False)
    x = torch.randn(2, 1, 192, 96)
    out = refiner(x, None)
    assert out.shape == (2, 1, 192, 96)
    # With use_wake_mask=False the stem in_channels should be 1, not 2.
    assert refiner.stem.in_channels == 1


def test_refiner_rejects_bad_mask_shape() -> None:
    """A wake_mask whose spatial shape disagrees with x must error out."""
    refiner = _make_refiner(use_wake_mask=True)
    x = torch.randn(2, 1, 192, 96)
    bad_mask = torch.zeros(96, 48, dtype=torch.bool)
    with pytest.raises(ValueError, match="wake_mask shape"):
        refiner(x, bad_mask)
    # 1-D mask also rejected.
    with pytest.raises(ValueError, match="wake_mask must be 2D"):
        refiner(x, torch.zeros(192))


def test_refiner_gradient_flows() -> None:
    """After breaking the zero-init head, backward yields nonzero grads
    on the trunk conv weights.

    The head conv is zero-initialised at construction (LapSRN-style
    stable init); on the very first step gradient only reaches the
    head. We mimic the iter > 0 state by initialising the head to a
    small random value, then check the upstream trunk also sees
    gradient.
    """
    refiner = _make_refiner()
    torch.nn.init.normal_(refiner.head.weight, std=0.02)
    torch.nn.init.normal_(refiner.head.bias, std=0.02)

    x = torch.randn(2, 1, 192, 96, requires_grad=True)
    mask = torch.from_numpy(build_wake_mask(192, 96))
    target = torch.randn(2, 1, 192, 96)
    out = refiner(x, mask)
    loss = (out - target).pow(2).mean()
    loss.backward()

    assert refiner.stem.weight.grad is not None
    assert (refiner.stem.weight.grad != 0).any(), "stem conv should see gradient"
    for k, block in enumerate(refiner.blocks):
        assert block.conv.weight.grad is not None, f"block {k}: no grad"
        assert (block.conv.weight.grad != 0).any(), f"block {k}: zero grad"
    assert refiner.head.weight.grad is not None
    assert (refiner.head.weight.grad != 0).any()
