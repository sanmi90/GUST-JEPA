"""Unit tests for src/models/decoder.py.

Covers the HybridViTConvDecoder shape contract, the mirror-image
correspondence with the encoder, the 2D vs 3D input handling, the
gradient flow, and the parameter-count bounds.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.models.decoder import HybridViTConvDecoder
from src.models.encoder import HybridCNNViTEncoder


def test_decoder_shape_2d() -> None:
    """Input (B, d) -> output (B, 1, 192, 96)."""
    torch.manual_seed(0)
    dec = HybridViTConvDecoder(latent_dim=32)
    z = torch.randn(4, 32)
    x_hat = dec(z)
    assert x_hat.shape == (4, 1, 192, 96)


def test_decoder_shape_3d() -> None:
    """Input (B, T, d) -> output (B, T, 1, 192, 96)."""
    torch.manual_seed(0)
    dec = HybridViTConvDecoder(latent_dim=32)
    z = torch.randn(2, 8, 32)
    x_hat = dec(z)
    assert x_hat.shape == (2, 8, 1, 192, 96)


def test_decoder_mirror_of_encoder() -> None:
    """Encoder -> decoder roundtrip preserves shape from input to output."""
    torch.manual_seed(0)
    enc = HybridCNNViTEncoder(latent_dim=32)
    dec = HybridViTConvDecoder(latent_dim=32)
    x = torch.randn(2, 4, 1, 192, 96)
    z = enc(x)
    x_hat = dec(z)
    assert x_hat.shape == x.shape


def test_decoder_gradient_flow() -> None:
    """One optimizer step changes every decoder parameter."""
    torch.manual_seed(0)
    dec = HybridViTConvDecoder(latent_dim=32)
    opt = torch.optim.AdamW(dec.parameters(), lr=1e-3)
    z = torch.randn(2, 32, requires_grad=True)
    target = torch.randn(2, 1, 192, 96)

    initial = {n: p.detach().clone() for n, p in dec.named_parameters()}
    pred = dec(z)
    loss = ((pred - target) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    for n, p in dec.named_parameters():
        assert p.grad is not None, f"no grad for {n}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {n}"
    opt.step()
    for n, p in dec.named_parameters():
        # Parameters with non-trivial gradients should have moved.
        if p.grad.abs().mean() > 1e-9:
            assert not torch.equal(initial[n], p.detach()), f"{n} unchanged after step"


def test_decoder_param_count_within_bounds() -> None:
    """Decoder has 5-12M params at default config (mirror of ~7M ViT + 1M conv)."""
    dec = HybridViTConvDecoder(latent_dim=32)
    n = sum(p.numel() for p in dec.parameters())
    assert 5_000_000 <= n <= 12_000_000, f"unexpected param count {n}"


def test_decoder_feature_map_constraint() -> None:
    """Decoder raises if feature_h * 2^n != out_h or feature_w * 2^n != out_w."""
    with pytest.raises(ValueError):
        HybridViTConvDecoder(feature_h=25, feature_w=12, n_upsample_stages=3,
                             out_h=192, out_w=96)  # 25 * 8 != 192
    with pytest.raises(ValueError):
        HybridViTConvDecoder(feature_h=24, feature_w=11, n_upsample_stages=3,
                             out_h=192, out_w=96)  # 11 * 8 != 96
    with pytest.raises(ValueError):
        HybridViTConvDecoder(feature_h=24, feature_w=12, n_upsample_stages=3,
                             c_mid=(128, 64))  # wrong number of stages


def test_decoder_deterministic_init() -> None:
    """Same seed -> identical decoder."""
    torch.manual_seed(0)
    d1 = HybridViTConvDecoder(latent_dim=32)
    torch.manual_seed(0)
    d2 = HybridViTConvDecoder(latent_dim=32)
    for (n1, p1), (n2, p2) in zip(d1.named_parameters(), d2.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1, p2), f"{n1} differs"


def test_decoder_finite_outputs_on_random_input() -> None:
    """Decoder output is finite for random latent vectors."""
    torch.manual_seed(0)
    dec = HybridViTConvDecoder(latent_dim=32).eval()
    with torch.no_grad():
        z = torch.randn(4, 32)
        x_hat = dec(z)
    assert torch.isfinite(x_hat).all()
