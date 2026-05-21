"""Unit tests for :mod:`src.models.lap_film_decoder`.

Contracts verified:
    1. Shape contract on 2D z (B, latent_dim) -> (B, 1, 192, 96) and pyramid
       has 5 levels at expected resolutions.
    2. Shape contract on 3D z (B, T, latent_dim) -> (B, T, 1, 192, 96).
    3. No NaN / Inf at initialisation.
    4. Gradients flow into FiLM linears and conv weights.
    5. ``use_film=False`` produces a working decoder with fewer parameters
       (no FiLM linears).
    6. ``use_airfoil_mask_channel`` is optional and changes the input-projection
       conv channel count predictably.
    7. bf16 autocast forward + backward succeed on RTX 6000 (skipped otherwise).
"""

from __future__ import annotations

import pytest
import torch

from src.models.lap_film_decoder import FiLMResBlock, LapFiLMDecoder
from src.utils.device import NoRTX6000Error, require_rtx6000


def _make_decoder(**overrides) -> LapFiLMDecoder:
    """Build a small LapFiLMDecoder with the production default shape."""
    defaults = dict(
        latent_dim=32,
        base_hw=(12, 6),
        channels=(64, 64, 48, 32, 24),
        resblocks_per_level=2,
        upsample="pixelshuffle",
        fourier_bands=4,
        use_coord_channels=True,
        use_airfoil_mask_channel=False,
        use_film=True,
    )
    defaults.update(overrides)
    torch.manual_seed(0)
    return LapFiLMDecoder(**defaults)


def test_lap_film_decoder_shape_contract() -> None:
    """z (2, 32) -> pred (2, 1, 192, 96) with 5-level pyramid."""
    dec = _make_decoder()
    z = torch.randn(2, 32)
    out = dec(z)
    assert isinstance(out, dict)
    assert "pred" in out and "pyramid" in out
    assert out["pred"].shape == (2, 1, 192, 96)
    expected_sizes = [(12, 6), (24, 12), (48, 24), (96, 48), (192, 96)]
    assert len(out["pyramid"]) == 5
    for k, (h, w) in enumerate(expected_sizes):
        assert out["pyramid"][k].shape == (2, 1, h, w), (
            f"level {k} shape {out['pyramid'][k].shape} != (2, 1, {h}, {w})"
        )


def test_lap_film_decoder_shape_contract_3d_z() -> None:
    """z (B, T, latent_dim) reshapes correctly and emerges as (B, T, 1, H, W)."""
    dec = _make_decoder()
    z = torch.randn(2, 4, 32)
    out = dec(z)
    assert out["pred"].shape == (2, 4, 1, 192, 96)
    assert out["pyramid"][0].shape == (2, 4, 1, 12, 6)
    assert out["pyramid"][-1].shape == (2, 4, 1, 192, 96)


def test_lap_film_decoder_no_nan_at_init() -> None:
    """Forward pass on random z produces finite outputs at init."""
    dec = _make_decoder()
    z = torch.randn(2, 32)
    out = dec(z)
    assert torch.isfinite(out["pred"]).all()
    for k, p in enumerate(out["pyramid"]):
        assert torch.isfinite(p).all(), f"non-finite at pyramid level {k}"


def test_lap_film_decoder_initial_pred_near_zero() -> None:
    """Heads are zero-initialized so the initial prediction is exactly zero.

    This anchors training near the omega=0 freestream, matching the
    Charbonnier-pyramid loss assumptions: the residual that the network
    learns is the omega field itself, not a perturbation around an
    arbitrary init.
    """
    dec = _make_decoder()
    z = torch.randn(2, 32)
    out = dec(z)
    assert torch.allclose(out["pred"], torch.zeros_like(out["pred"]))


def test_lap_film_decoder_gradient_flows() -> None:
    """Backward through MSE on the final prediction yields nonzero grads on
    every FiLM linear and per-level conv at non-zero locations.

    The prediction heads are zero-initialised by design (LapSRN-style
    stable init), which means at the very first step gradients only
    reach the heads themselves and not the deeper layers. We simulate
    a post-first-step state by re-initialising the heads with small
    random values before running the backward pass.
    """
    dec = _make_decoder()
    # Break the zero-init on the heads (simulating any iter > 0 state)
    for head in dec.heads:
        torch.nn.init.normal_(head.weight, std=0.02)
        torch.nn.init.normal_(head.bias, std=0.02)

    z = torch.randn(2, 32, requires_grad=True)
    target = torch.randn(2, 1, 192, 96)
    out = dec(z)
    loss = (out["pred"] - target).pow(2).mean()
    for p in out["pyramid"][:-1]:
        loss = loss + 0.1 * p.pow(2).mean()
    loss.backward()

    for k, head in enumerate(dec.heads):
        assert head.weight.grad is not None, f"head {k}: no grad"
        assert (head.weight.grad != 0).any(), f"head {k}: zero grad"
    n_film = 0
    for level_blocks in dec.blocks:
        for block in level_blocks:
            assert block.use_film
            assert block.film.weight.grad is not None
            assert (block.film.weight.grad != 0).any(), (
                "FiLM weight grad should be non-zero after one backward pass"
            )
            n_film += 1
    assert n_film == len(dec.channels) * dec.resblocks_per_level
    # And the conv1/conv2 weights inside the FiLM blocks should also see grads.
    for level_blocks in dec.blocks:
        for block in level_blocks:
            assert block.conv1.weight.grad is not None
            assert (block.conv1.weight.grad != 0).any()
            assert block.conv2.weight.grad is not None
            assert (block.conv2.weight.grad != 0).any()


def test_lap_film_decoder_no_film_ablation() -> None:
    """use_film=False produces a working decoder with no FiLM linears
    and the latent broadcast as input channels at every level."""
    dec_film = _make_decoder(use_film=True)
    dec_nofilm = _make_decoder(use_film=False)

    z = torch.randn(2, 32)
    out_nofilm = dec_nofilm(z)
    assert out_nofilm["pred"].shape == (2, 1, 192, 96)

    # No FiLM linears in the no_film variant.
    for level_blocks in dec_nofilm.blocks:
        for block in level_blocks:
            assert not block.use_film
            assert not hasattr(block, "film") or block.film is None

    n_film_params = sum(p.numel() for p in dec_film.parameters())
    n_nofilm_params = sum(p.numel() for p in dec_nofilm.parameters())
    assert n_film_params != n_nofilm_params, (
        "FiLM and no-FiLM variants should have different parameter counts"
    )


def test_lap_film_decoder_airfoil_mask_optional() -> None:
    """Toggling use_airfoil_mask_channel changes input-projection in_channels
    by exactly +/- 1 per level."""
    dec_with = _make_decoder(use_airfoil_mask_channel=True)
    dec_without = _make_decoder(use_airfoil_mask_channel=False)

    for k in range(len(dec_with.channels)):
        in_with = dec_with.input_projs[k].in_channels
        in_without = dec_without.input_projs[k].in_channels
        assert in_with - in_without == 1, (
            f"level {k}: mask=True in_channels={in_with}, "
            f"mask=False in_channels={in_without}, diff != 1"
        )

    # Both still produce the correct output shape.
    z = torch.randn(2, 32)
    assert dec_with(z)["pred"].shape == (2, 1, 192, 96)
    assert dec_without(z)["pred"].shape == (2, 1, 192, 96)


def test_lap_film_decoder_fourier_bands_zero() -> None:
    """fourier_bands=0 produces a working decoder with only raw (x, y)
    in the coord channels (and optional mask)."""
    dec = _make_decoder(fourier_bands=0, use_airfoil_mask_channel=False)
    assert dec.coord_extra == 2
    z = torch.randn(2, 32)
    out = dec(z)
    assert out["pred"].shape == (2, 1, 192, 96)


def test_lap_film_decoder_bilinear_conv_upsample() -> None:
    """The bilinear_conv upsample mode produces the same output shape as
    pixelshuffle."""
    dec = _make_decoder(upsample="bilinear_conv")
    z = torch.randn(2, 32)
    out = dec(z)
    assert out["pred"].shape == (2, 1, 192, 96)
    assert out["pyramid"][0].shape == (2, 1, 12, 6)


def test_film_resblock_identity_at_init() -> None:
    """FiLMResBlock with FiLM linears zero-init is identity-on-residual.

    The default zero init of conv2 + film weights means the residual
    branch contributes a tensor of zeros, so output == input.
    """
    block = FiLMResBlock(ch=16, cond_dim=8)
    # Zero out conv2 to get exact identity (Conv1 still random)
    torch.nn.init.zeros_(block.conv2.weight)
    torch.nn.init.zeros_(block.conv2.bias)
    x = torch.randn(2, 16, 8, 8)
    z = torch.randn(2, 8)
    y = block(x, z)
    assert torch.allclose(y, x, atol=1e-6)


def test_film_resblock_no_cond_runs() -> None:
    """FiLMResBlock with cond_dim=None acts as a plain ResBlock."""
    block = FiLMResBlock(ch=16, cond_dim=None)
    assert not block.use_film
    x = torch.randn(2, 16, 8, 8)
    y = block(x, None)
    assert y.shape == x.shape


def test_lap_film_decoder_bf16_autocast() -> None:
    """Forward + backward under bf16 autocast on RTX 6000."""
    try:
        device = require_rtx6000()
    except NoRTX6000Error as e:
        pytest.skip(str(e))
    dec = _make_decoder().to(device)
    z = torch.randn(2, 32, device=device)
    target = torch.randn(2, 1, 192, 96, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = dec(z)
        loss = (out["pred"].float() - target).pow(2).mean()
    loss.backward()
    assert out["pred"].dtype in (torch.bfloat16, torch.float32)
    assert out["pred"].device.type == "cuda"
