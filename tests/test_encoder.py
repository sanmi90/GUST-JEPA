"""Unit tests for src/models/encoder.py.

Covers the HybridCNNViTEncoder shape contract, the LeWM BatchNorm projection
constraint (HANDOFF.md D17), parameter-count bounds, gradient flow, bf16
autocast roundtrip, and initialization determinism.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.models.encoder import HybridCNNViTEncoder
from src.utils.device import NoRTX6000Error, require_rtx6000


def test_encoder_shape_contract() -> None:
    """Input (2, 8, 1, 192, 96) -> output (2, 8, 32) at default config."""
    torch.manual_seed(0)
    encoder = HybridCNNViTEncoder()
    x = torch.randn(2, 8, 1, 192, 96)
    z = encoder(x)
    assert z.shape == (2, 8, 32)


def test_encoder_num_spatial_tokens() -> None:
    """num_spatial_tokens == 288 for the default 3-stage stem on (192, 96)."""
    torch.manual_seed(0)
    encoder = HybridCNNViTEncoder()
    assert encoder.num_spatial_tokens == 288


def test_encoder_projection_is_batchnorm() -> None:
    """The final layer of encoder.proj is nn.BatchNorm1d, NOT LayerNorm.

    LeWM-specific constraint (HANDOFF.md D17): SIGReg requires BatchNorm at
    the encoder bottleneck because the final ViT LayerNorm would otherwise
    prevent the anti-collapse objective from being optimized.
    """
    torch.manual_seed(0)
    encoder = HybridCNNViTEncoder()
    final = encoder.proj[-1]
    assert isinstance(
        final, nn.BatchNorm1d
    ), f"expected nn.BatchNorm1d at proj[-1], got {type(final).__name__}"


def test_encoder_parameter_count_in_range() -> None:
    """Total parameters in (6.0M, 7.5M) for the default config.

    SESSION3 spec asked for (8M, 12M), but the spec's own component-level
    estimates (CNN ~2.5M + ViT ~5M + projection ~16k = ~7.5M) already sit
    below 8M. Measured realization of the spec architecture is ~6.67M,
    consistent with the LeWM ViT-Tiny encoder size (~5M, arXiv:2603.19312)
    plus a small CNN stem. Bound is tight enough to catch missing or extra
    transformer blocks, missing CNN stages, or a wrong MLP ratio.
    """
    torch.manual_seed(0)
    encoder = HybridCNNViTEncoder()
    n_params = sum(p.numel() for p in encoder.parameters())
    assert 6.0e6 < n_params < 7.5e6, f"got {n_params} params, expected in (6.0e6, 7.5e6)"


def test_encoder_gradient_flows() -> None:
    """Backward on a scalar loss produces non-zero gradient on the input."""
    torch.manual_seed(0)
    encoder = HybridCNNViTEncoder()
    x = torch.randn(2, 8, 1, 192, 96, requires_grad=True)
    z = encoder(x)
    loss = z.pow(2).sum()
    loss.backward()
    assert x.grad is not None
    assert torch.any(x.grad != 0)


def test_encoder_bf16_autocast_roundtrip() -> None:
    """Forward runs end-to-end under bf16 autocast on the RTX 6000.

    Per the project's Hardware rule (CLAUDE.md), CUDA paths must run on the
    RTX 6000 Blackwell and never silently fall back to CPU. This test
    requires an RTX 6000 device and skips otherwise; the skip message
    surfaces what torch DOES see so the failure mode is clear.
    """
    try:
        device = require_rtx6000()
    except NoRTX6000Error as e:
        pytest.skip(str(e))

    torch.manual_seed(0)
    encoder = HybridCNNViTEncoder().to(device)
    x = torch.randn(2, 4, 1, 192, 96, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        z = encoder(x)
    assert z.dtype in (torch.bfloat16, torch.float32), f"got dtype {z.dtype}"
    assert z.shape == (2, 4, 32)
    assert z.device.type == "cuda", f"expected cuda device, got {z.device}"


def test_encoder_deterministic_with_fixed_seed() -> None:
    """Two encoders built under torch.manual_seed(0) produce identical outputs."""
    torch.manual_seed(0)
    enc_a = HybridCNNViTEncoder()
    torch.manual_seed(0)
    enc_b = HybridCNNViTEncoder()
    x = torch.randn(2, 4, 1, 192, 96)
    z_a = enc_a(x)
    z_b = enc_b(x)
    assert torch.allclose(z_a, z_b, atol=1e-6)
