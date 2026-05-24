"""Unit tests for :mod:`src.models.discriminator`.

Contracts verified:
    1. Shape contract on ``(B, 1, 192, 96)`` field + ``(192, 96)`` mask ->
       ``(B, 1, 24, 12)`` patch decision map.
    2. Parameter count is ~150k (50% tolerance: 75k - 225k).
    3. Every conv layer has ``spectral_norm`` parametrization attached.
    4. Gradients flow into every conv weight after one backward pass.
    5. The discriminator rejects mismatched mask shapes.
"""

from __future__ import annotations

import pytest
import torch
from torch.nn.utils.parametrize import is_parametrized

from src.evaluation.decoder_metrics import wake_mask as build_wake_mask
from src.models.discriminator import PatchGANDiscriminator


def _make_disc(**overrides) -> PatchGANDiscriminator:
    """Build a discriminator with the production default config."""
    defaults = dict(
        in_channels=1,
        mask_channels=1,
        channels=(32, 64, 128),
        leaky_slope=0.2,
    )
    defaults.update(overrides)
    torch.manual_seed(0)
    return PatchGANDiscriminator(**defaults)


def test_patchgan_forward_shape() -> None:
    """(B=2, 1, 192, 96) field + (192, 96) mask -> (2, 1, 24, 12) logits."""
    disc = _make_disc()
    x = torch.randn(2, 1, 192, 96)
    mask = torch.from_numpy(build_wake_mask(192, 96))
    out = disc(x, mask)
    assert out.shape == (
        2,
        1,
        24,
        12,
    ), f"discriminator output shape {tuple(out.shape)} != (2, 1, 24, 12)"
    assert torch.isfinite(out).all()


def test_patchgan_parameter_count() -> None:
    """Parameter count is ~150k within 50% tolerance (75k - 225k)."""
    disc = _make_disc()
    n_params = sum(p.numel() for p in disc.parameters())
    target = 150_000
    lower = int(target * 0.5)
    upper = int(target * 1.5)
    assert lower <= n_params <= upper, (
        f"PatchGANDiscriminator has {n_params:,} params; expected "
        f"~{target:,} (window {lower:,}..{upper:,})"
    )


def test_patchgan_spectral_norm_present() -> None:
    """Every conv layer in the discriminator has spectral_norm attached."""
    disc = _make_disc()
    conv_names = ["conv1", "conv2", "conv3", "conv4"]
    n_parametrized = 0
    for name in conv_names:
        conv = getattr(disc, name)
        assert is_parametrized(
            conv, "weight"
        ), f"{name}: weight is not parametrized (spectral_norm missing)"
        n_parametrized += 1
    assert n_parametrized == len(
        conv_names
    ), f"only {n_parametrized}/{len(conv_names)} conv layers parametrized"


def test_patchgan_gradient_flows() -> None:
    """One backward pass yields nonzero gradients on every conv weight."""
    disc = _make_disc()
    x = torch.randn(2, 1, 192, 96, requires_grad=True)
    mask = torch.from_numpy(build_wake_mask(192, 96))
    out = disc(x, mask)
    # Hinge loss on a synthetic real / fake batch.
    loss = -out.mean()
    loss.backward()
    for name in ["conv1", "conv2", "conv3", "conv4"]:
        conv = getattr(disc, name)
        # spectral_norm wraps the weight; the underlying parameter is
        # accessible as parametrizations.weight.original.
        original = conv.parametrizations.weight.original
        assert original.grad is not None, f"{name}: no grad"
        assert (original.grad != 0).any(), f"{name}: zero grad"


def test_patchgan_rejects_bad_mask_shape() -> None:
    """A wake_mask whose spatial shape disagrees with x must error out."""
    disc = _make_disc()
    x = torch.randn(2, 1, 192, 96)
    bad_mask = torch.zeros(96, 48, dtype=torch.bool)
    with pytest.raises(ValueError, match="wake_mask shape"):
        disc(x, bad_mask)
    with pytest.raises(ValueError, match="wake_mask must be 2D"):
        disc(x, torch.zeros(192))
