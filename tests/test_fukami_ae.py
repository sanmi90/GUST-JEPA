"""Unit tests for src/baselines/fukami_ae.py.

Covers the shape contract of FukamiCNNEncoder, FukamiCNNDecoder, and
FukamiLiftHead (matching Fukami & Taira J. Fluid Mech. 2023 Table S.1
adapted to our (192, 96) input + d=32 latent), the wrapper forward
joint loss, gradient flow through all three components, and the
parameter-count bound.
"""

from __future__ import annotations

import torch
from torch import nn

from src.baselines.fukami_ae import (
    FukamiAEWrapper,
    FukamiCNNDecoder,
    FukamiCNNEncoder,
    FukamiLiftHead,
)


def test_fukami_encoder_shape_4d() -> None:
    """(B, 1, H, W) input -> (B, d) latent."""
    torch.manual_seed(0)
    enc = FukamiCNNEncoder(latent_dim=32)
    x = torch.randn(4, 1, 192, 96)
    z = enc(x)
    assert z.shape == (4, 32)


def test_fukami_encoder_shape_5d() -> None:
    """(B, T, 1, H, W) input -> (B, T, d) latent."""
    torch.manual_seed(0)
    enc = FukamiCNNEncoder(latent_dim=32)
    x = torch.randn(2, 8, 1, 192, 96)
    z = enc(x)
    assert z.shape == (2, 8, 32)


def test_fukami_decoder_shape_2d() -> None:
    """(B, d) latent -> (B, 1, 192, 96) reconstruction."""
    torch.manual_seed(0)
    dec = FukamiCNNDecoder(latent_dim=32)
    z = torch.randn(4, 32)
    x_hat = dec(z)
    assert x_hat.shape == (4, 1, 192, 96)


def test_fukami_decoder_shape_3d() -> None:
    """(B, T, d) latent -> (B, T, 1, 192, 96) reconstruction."""
    torch.manual_seed(0)
    dec = FukamiCNNDecoder(latent_dim=32)
    z = torch.randn(2, 4, 32)
    x_hat = dec(z)
    assert x_hat.shape == (2, 4, 1, 192, 96)


def test_fukami_lift_head_shape() -> None:
    """Lift head: latent (B[, T], d) -> CL (B[, T], n_deltas)."""
    torch.manual_seed(0)
    head = FukamiLiftHead(latent_dim=32, n_deltas=3)
    z2 = torch.randn(4, 32)
    out2 = head(z2)
    assert out2.shape == (4, 3)
    z3 = torch.randn(2, 5, 32)
    out3 = head(z3)
    assert out3.shape == (2, 5, 3)


def test_fukami_encoder_decoder_roundtrip_shape() -> None:
    """Encoder + decoder preserves input shape."""
    torch.manual_seed(0)
    enc = FukamiCNNEncoder(latent_dim=32)
    dec = FukamiCNNDecoder(latent_dim=32)
    x = torch.randn(2, 4, 1, 192, 96)
    z = enc(x)
    x_hat = dec(z)
    assert x_hat.shape == x.shape


def test_fukami_wrapper_forward_returns_loss_dict() -> None:
    """Wrapper.forward(batch) returns L_total, L_recon, L_lift, z."""
    torch.manual_seed(0)
    wrapper = FukamiAEWrapper(latent_dim=32, n_deltas=3,
                              lambda_recon=1.0, lambda_lift=1.0)
    batch = {
        "omega": torch.randn(2, 4, 1, 192, 96),
        "cl_future": torch.randn(2, 4, 3),
    }
    out = wrapper(batch)
    assert "L_total" in out and "L_recon" in out and "L_lift" in out and "z" in out
    assert torch.isfinite(out["L_total"])
    assert out["z"].shape == (2, 4, 32)


def test_fukami_wrapper_omega_alias() -> None:
    """Wrapper accepts batch key 'omega' or 'omega_z'."""
    torch.manual_seed(0)
    wrapper = FukamiAEWrapper(latent_dim=32, n_deltas=3)
    x = torch.randn(2, 4, 1, 192, 96)
    cl = torch.randn(2, 4, 3)
    out_a = wrapper({"omega": x, "cl_future": cl})
    torch.manual_seed(0)
    wrapper2 = FukamiAEWrapper(latent_dim=32, n_deltas=3)
    out_b = wrapper2({"omega_z": x, "cl_future": cl})
    assert torch.allclose(out_a["L_total"], out_b["L_total"])


def test_fukami_wrapper_gradient_flow() -> None:
    """All wrapper parameters get gradients after one backward."""
    torch.manual_seed(0)
    wrapper = FukamiAEWrapper(latent_dim=32, n_deltas=3)
    batch = {
        "omega": torch.randn(2, 4, 1, 192, 96),
        "cl_future": torch.randn(2, 4, 3),
    }
    out = wrapper(batch)
    out["L_total"].backward()
    for n, p in wrapper.named_parameters():
        assert p.grad is not None, f"no grad for {n}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {n}"


def test_fukami_wrapper_param_count() -> None:
    """Fukami AE is intentionally lightweight (~200-300K params at d=32)."""
    wrapper = FukamiAEWrapper(latent_dim=32, n_deltas=3)
    n = sum(p.numel() for p in wrapper.parameters())
    assert 100_000 <= n <= 500_000, f"unexpected param count {n}"


def test_fukami_lift_loss_masking() -> None:
    """Non-finite CL targets are masked out of the lift loss."""
    torch.manual_seed(0)
    wrapper = FukamiAEWrapper(latent_dim=32, n_deltas=3)
    batch = {
        "omega": torch.randn(2, 4, 1, 192, 96),
        "cl_future": torch.full((2, 4, 3), float("nan")),
    }
    out = wrapper(batch)
    # All targets non-finite -> L_lift is zero, L_total = lambda_recon * L_recon
    assert out["L_lift"].item() == 0.0
    assert torch.isfinite(out["L_total"])
