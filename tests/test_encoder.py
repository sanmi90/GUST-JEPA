"""Unit tests for src/models/encoder.py.

Covers the HybridCNNViTEncoder shape contract, the LeWM BatchNorm projection
constraint (HANDOFF.md D17), parameter-count bounds, gradient flow, bf16
autocast roundtrip, and initialization determinism.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.models.encoder import HybridCNNViTEncoder, PatchPoolEncoder
from src.models.encoder import SpatioTemporalCNNViTEncoder
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


def test_encoder_projection_is_batchnorm_by_default() -> None:
    """The default ``projection_norm`` selects ``nn.BatchNorm1d`` at ``proj[-1]``.

    LeWM-specific constraint (HANDOFF.md D17): SIGReg requires BatchNorm at the
    encoder bottleneck because the final ViT LayerNorm would otherwise prevent
    the anti-collapse objective from being optimized. The default stays
    BatchNorm; ``projection_norm='layernorm'`` is the Session 5 Run B
    diagnostic intervention switch (HANDOFF.md D25).
    """
    torch.manual_seed(0)
    encoder = HybridCNNViTEncoder()
    final = encoder.proj[-1]
    assert isinstance(
        final, nn.BatchNorm1d
    ), f"expected nn.BatchNorm1d at proj[-1] by default, got {type(final).__name__}"


def test_encoder_projection_can_be_layernorm() -> None:
    """``projection_norm='layernorm'`` swaps in ``nn.LayerNorm`` at ``proj[-1]``.

    Session 5 Run B (HANDOFF.md D25) needs this code path to test whether the
    LeWM BatchNorm-at-projection choice is responsible for SIGReg collapse
    on low-intrinsic-dim physics data.
    """
    torch.manual_seed(0)
    encoder = HybridCNNViTEncoder(projection_norm="layernorm")
    final = encoder.proj[-1]
    assert isinstance(final, nn.LayerNorm), (
        "expected nn.LayerNorm at proj[-1] with projection_norm='layernorm', "
        f"got {type(final).__name__}"
    )

    x = torch.randn(2, 4, 1, 192, 96)
    z = encoder(x)
    assert z.shape == (2, 4, 32)


def test_encoder_projection_norm_rejects_unknown() -> None:
    """Unknown ``projection_norm`` values raise a clear ``ValueError``."""
    with pytest.raises(ValueError, match="projection_norm"):
        HybridCNNViTEncoder(projection_norm="instancenorm")


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


def test_patch_pool_encoder_shape_5d() -> None:
    """(B, T, 1, 192, 96) -> (B, T, 64*12*6 = 4608) at default config."""
    torch.manual_seed(0)
    enc = PatchPoolEncoder(in_channels=1, out_channels=64, patch_h=16, patch_w=16)
    x = torch.randn(2, 4, 1, 192, 96)
    h = enc(x)
    assert h.shape == (2, 4, 64 * 12 * 6)


def test_patch_pool_encoder_shape_4d() -> None:
    """(B, 1, 192, 96) -> (B, 4608) on 4D input."""
    torch.manual_seed(0)
    enc = PatchPoolEncoder()
    x = torch.randn(3, 1, 192, 96)
    h = enc(x)
    assert h.shape == (3, 64 * 12 * 6)


def test_patch_pool_encoder_gradient_flows() -> None:
    """Backward through PatchPoolEncoder reaches both proj and input."""
    enc = PatchPoolEncoder()
    x = torch.randn(2, 1, 192, 96, requires_grad=True)
    h = enc(x).sum()
    h.backward()
    assert x.grad is not None and x.grad.abs().sum().item() > 0.0
    for name, p in enc.named_parameters():
        assert p.grad is not None, f"missing grad for {name}"
        assert p.grad.abs().sum().item() > 0.0, f"zero grad for {name}"


def test_patch_pool_encoder_rejects_bad_input_shape() -> None:
    enc = PatchPoolEncoder()
    with pytest.raises(ValueError, match="4D"):
        enc(torch.randn(2, 192, 96))


def test_st_encoder_shape_contract() -> None:
    """Input (2, 8, 1, 192, 96) -> output (2, 8, 32) at default config."""
    torch.manual_seed(0)
    enc = SpatioTemporalCNNViTEncoder()
    x = torch.randn(2, 8, 1, 192, 96)
    z = enc(x)
    assert z.shape == (2, 8, 32)


def test_st_encoder_is_causal() -> None:
    """Perturbing a future frame must not change earlier latents.

    With temporal kernel 3 in the stem + 2 downsamples, the causal receptive
    field is 1 + (3-1)*3 = 7 frames, so z_t depends only on frames in
    [t-6, t]. Perturbing frame 7 must leave z_0..z_6 unchanged. Run in eval()
    so BatchNorm uses fixed running stats and the only temporal coupling under
    test is the 3D stem.
    """
    torch.manual_seed(0)
    enc = SpatioTemporalCNNViTEncoder(latent_dim=16).eval()
    x = torch.randn(1, 10, 1, 192, 96)
    with torch.no_grad():
        z = enc(x)
        x2 = x.clone()
        x2[:, 7] = torch.randn(1, 1, 192, 96)
        z2 = enc(x2)
    assert torch.allclose(z[:, :7], z2[:, :7], atol=1e-5)
    assert not torch.allclose(z[:, 7], z2[:, 7], atol=1e-5)


def test_st_encoder_projection_is_batchnorm_by_default() -> None:
    """The default projection_norm selects nn.BatchNorm1d at proj[-1]."""
    enc = SpatioTemporalCNNViTEncoder()
    assert isinstance(enc.proj[-1], nn.BatchNorm1d)


def test_st_encoder_param_count_bound() -> None:
    """ST encoder stays within ~1.5x the per-frame encoder's parameter count."""
    from src.models.encoder import HybridCNNViTEncoder

    base = sum(p.numel() for p in HybridCNNViTEncoder().parameters())
    st = sum(p.numel() for p in SpatioTemporalCNNViTEncoder().parameters())
    assert st < 1.5 * base, f"ST encoder {st} params exceeds 1.5x base {base}"


# ---------------------------------------------------------------------------
# Session 31 Track B.1: spatial-latent encoder tap.
# ---------------------------------------------------------------------------


def test_encoder_default_latent_mode_is_pooled() -> None:
    """The default ``latent_mode`` keeps the prior pooled ``(B, T, d)`` contract."""
    torch.manual_seed(0)
    encoder = HybridCNNViTEncoder()
    assert encoder.latent_mode == "pooled"
    x = torch.randn(2, 4, 1, 192, 96)
    z = encoder(x)
    assert z.shape == (2, 4, 32)


def test_encoder_pooled_mode_matches_prior_behavior() -> None:
    """Explicit ``latent_mode='pooled'`` is bit-identical to the default path.

    Guards against the spatial tap perturbing the pooled CLS path (e.g. by
    consuming RNG draws during construction or rerouting the forward).
    """
    torch.manual_seed(0)
    enc_default = HybridCNNViTEncoder()
    torch.manual_seed(0)
    enc_pooled = HybridCNNViTEncoder(latent_mode="pooled")
    x = torch.randn(2, 4, 1, 192, 96)
    assert torch.allclose(enc_default(x), enc_pooled(x), atol=1e-6)


def test_encoder_spatial_mode_shape() -> None:
    """``latent_mode='spatial'`` returns ``(B, T, d, h, w)`` with ``h, w = 24, 12``."""
    torch.manual_seed(0)
    encoder = HybridCNNViTEncoder(latent_mode="spatial", latent_dim=32)
    x = torch.randn(2, 4, 1, 192, 96)
    z = encoder(x)
    assert z.shape == (2, 4, 32, 24, 12)


def test_encoder_latent_grid_property() -> None:
    """``latent_grid`` exposes the true ViT token grid ``(24, 12)``."""
    encoder = HybridCNNViTEncoder(latent_mode="spatial")
    assert encoder.latent_grid == (24, 12)


def test_encoder_spatial_boundary_is_batchnorm_not_layernorm() -> None:
    """The spatial latent boundary uses ``nn.BatchNorm2d`` (SIGReg invariant).

    CLAUDE.md forbids LayerNorm at the latent boundary because SIGReg requires
    a BatchNorm there. The spatial tap must use a channel-wise ``BatchNorm2d``
    and must NOT introduce a ``LayerNorm`` at the latent boundary.
    """
    encoder = HybridCNNViTEncoder(latent_mode="spatial", latent_dim=32)
    assert isinstance(encoder.spatial_norm, nn.BatchNorm2d)
    assert encoder.spatial_norm.num_features == 32
    assert not isinstance(encoder.spatial_norm, nn.LayerNorm)


def test_encoder_spatial_rejects_unknown_latent_mode() -> None:
    """Unknown ``latent_mode`` values raise a clear ``ValueError``."""
    with pytest.raises(ValueError, match="latent_mode"):
        HybridCNNViTEncoder(latent_mode="tokens")


def test_encoder_spatial_gradient_flows() -> None:
    """Backward on the spatial latent reaches the input."""
    torch.manual_seed(0)
    encoder = HybridCNNViTEncoder(latent_mode="spatial", latent_dim=16)
    x = torch.randn(2, 3, 1, 192, 96, requires_grad=True)
    z = encoder(x)
    z.pow(2).sum().backward()
    assert x.grad is not None and torch.any(x.grad != 0)


def test_encoder_pooled_param_count_unchanged_by_flag() -> None:
    """Pooled mode does not allocate the spatial tap modules."""
    pooled = HybridCNNViTEncoder(latent_mode="pooled")
    assert not hasattr(pooled, "spatial_proj") or pooled.spatial_proj is None
