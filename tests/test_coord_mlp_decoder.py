"""Unit tests for :mod:`src.models.coord_mlp_decoder`."""

from __future__ import annotations

import pytest
import torch

from src.models.coord_mlp_decoder import CoordMLPDecoder


def test_coord_mlp_shape_contract() -> None:
    """z (2, 32) -> (2, 1, 192, 96). z (B, T, 32) -> (B, T, 1, 192, 96)."""
    torch.manual_seed(0)
    dec = CoordMLPDecoder(latent_dim=32, hidden=64, layers=3, chunk_pixels=4096)
    z = torch.randn(2, 32)
    out = dec(z)
    assert out.shape == (2, 1, 192, 96)
    z = torch.randn(2, 4, 32)
    out = dec(z)
    assert out.shape == (2, 4, 1, 192, 96)


def test_coord_mlp_chunking_invariant() -> None:
    """Output is (numerically) identical across chunk sizes.

    Each pixel is processed independently (no batchnorm or cross-pixel
    reductions), so the chunk size is a pure memory optimisation. We
    allow a tiny tolerance because non-associativity of floating-point
    addition can give bit-different results when reshapes change the
    underlying memory layout, but the relative error should be at the
    level of single-precision noise.
    """
    torch.manual_seed(0)
    dec = CoordMLPDecoder(latent_dim=32, hidden=64, layers=3,
                          activation="gelu_fourier", fourier_bands=4)
    z = torch.randn(2, 32)
    outs = {}
    for chunk in (1024, 4096, 18432):
        dec.chunk_pixels = chunk
        outs[chunk] = dec(z)
    for c in (4096, 18432):
        max_abs = (outs[1024] - outs[c]).abs().max().item()
        assert max_abs < 1e-5, f"chunk {c} diff {max_abs}"


def test_coord_mlp_siren_vs_gelu_both_train() -> None:
    """Both activation modes produce finite gradients and the parameter
    counts differ in the expected way (Fourier features in gelu_fourier
    expand the input width)."""
    torch.manual_seed(0)
    dec_siren = CoordMLPDecoder(latent_dim=32, hidden=64, layers=3,
                                activation="sine")
    dec_gelu = CoordMLPDecoder(latent_dim=32, hidden=64, layers=3,
                               activation="gelu_fourier", fourier_bands=8)

    for dec in (dec_siren, dec_gelu):
        z = torch.randn(2, 32, requires_grad=True)
        out = dec(z)
        assert torch.isfinite(out).all()
        loss = out.pow(2).mean()
        loss.backward()
        any_param_grad = False
        for p in dec.parameters():
            if p.grad is not None and (p.grad != 0).any():
                any_param_grad = True
                break
        assert any_param_grad

    n_siren = sum(p.numel() for p in dec_siren.parameters())
    n_gelu = sum(p.numel() for p in dec_gelu.parameters())
    # gelu_fourier expands the first layer (more input dims). The two
    # models should NOT have the same parameter count.
    assert n_siren != n_gelu


def test_coord_mlp_custom_coords_path() -> None:
    """Passing a small coords tensor returns ``(B, 1, N)``-like output
    without building the full grid."""
    dec = CoordMLPDecoder(latent_dim=32, hidden=32, layers=3)
    z = torch.randn(2, 32)
    coords = torch.zeros(7, 2)
    out = dec(z, coords=coords)
    # When custom coords are passed we keep them as a flat list.
    assert out.shape == (2, 7, 1)


def test_coord_mlp_high_frequency_capacity() -> None:
    """SIREN should be able to fit a high-frequency sinusoidal target
    (well-known SIREN strength)."""
    torch.manual_seed(42)
    H, W = 32, 32
    xs = torch.linspace(-1, 1, W)
    ys = torch.linspace(-1, 1, H)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    target = torch.sin(6.0 * xx + 4.0 * yy)[None, None]
    dec = CoordMLPDecoder(latent_dim=4, hidden=64, layers=4,
                          activation="sine", H=H, W=W,
                          chunk_pixels=1024)
    z = torch.randn(1, 4)
    z = nn_param(z)  # learnable so the decoder has a meaningful conditioning
    opt = torch.optim.Adam(list(dec.parameters()) + [z], lr=1e-3)
    for _ in range(300):
        opt.zero_grad()
        pred = dec(z)
        loss = (pred - target).pow(2).mean()
        loss.backward()
        opt.step()
    # After 300 steps the SIREN decoder should have moved off the
    # initial loss significantly (initial random init typically
    # ~5e-1; we just check non-trivial progress).
    final = (dec(z) - target).pow(2).mean().item()
    assert final < 0.1, f"SIREN failed to fit a smooth sine: final loss {final}"


def nn_param(t):
    import torch.nn as nn
    return nn.Parameter(t)
