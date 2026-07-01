"""Tests for the SESSION 31 Track C.1 canonical-model adapter (pure pieces).

The trainer migration routes the autoencoder and JEPA models through ONE shared
loss path on the spatial latent. These tests pin the four PURE pieces of that
migration in isolation, on CPU with tiny stub modules, so they are fast and
deterministic:

1. ``flatten_spatial_latent``: ``[B, T, d, h, w] -> [B*T*h*w, d]`` (anti-collapse
   over the batch across cases/frames/cells, not over time).
2. ``global_average_pool_latent``: ``[B, T, d, h, w] -> [B, T, d]`` (GAP adapter
   feeding the scalar lift / wake heads).
3. ``assemble_spatial_rollout``: single open-loop rollout, detached online
   targets, no teacher forcing.
4. ``build_outputs``: the named-tensor outputs dict per model type.

Plus a check that the kit's ``compute_total_loss`` accepts a pre-built
anti-collapse module and returns the same value as the build-each-call path.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

from src.config.kit_config import load_model_config
from src.losses.kit import build_anticollapse, compute_total_loss
from src.training.canonical_model import (
    assemble_spatial_rollout,
    build_outputs,
    flatten_spatial_latent,
    global_average_pool_latent,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CANONICAL_DIR = REPO_ROOT / "configs" / "canonical"


class _StubPredictor(nn.Module):
    """Spatial-latent predictor with the ResUNet contract but trivial weights."""

    def __init__(self, latent_dim: int, context_length: int = 2) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.context_length = context_length
        self.conv = nn.Conv2d(context_length * latent_dim, latent_dim, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class _StubDecoder(nn.Module):
    """Maps ``[B, T, d, h, w] -> [B, T, 1, H, W]`` with a fixed upsample factor."""

    def __init__(self, latent_dim: int, scale: int = 8) -> None:
        super().__init__()
        self.scale = scale
        self.conv = nn.Conv2d(latent_dim, 1, kernel_size=1)

    def forward(self, z: Tensor) -> Tensor:
        b, t, d, h, w = z.shape
        flat = z.reshape(b * t, d, h, w)
        field = self.conv(flat)
        field = torch.nn.functional.interpolate(field, scale_factor=self.scale, mode="nearest")
        return field.reshape(b, t, 1, h * self.scale, w * self.scale)


class _StubHead(nn.Module):
    """``(..., d) -> (..., out_dim)`` linear head."""

    def __init__(self, latent_dim: int, out_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.linear = nn.Linear(latent_dim, out_dim)

    def forward(self, z: Tensor) -> Tensor:
        return self.linear(z)


# ---------------------------------------------------------------------------
# flatten_spatial_latent
# ---------------------------------------------------------------------------
def test_flatten_spatial_latent_shape() -> None:
    z = torch.randn(2, 3, 5, 4, 6)
    out = flatten_spatial_latent(z)
    assert out.shape == (2 * 3 * 4 * 6, 5)


def test_flatten_spatial_latent_preserves_channel_values() -> None:
    # Build a latent whose channel c is the constant c at every (B, T, h, w).
    b, t, d, h, w = 2, 2, 3, 2, 2
    z = torch.arange(d, dtype=torch.float32).view(1, 1, d, 1, 1).expand(b, t, d, h, w).contiguous()
    out = flatten_spatial_latent(z)
    # Every flattened row must be [0, 1, 2] (d on the trailing axis).
    assert out.shape == (b * t * h * w, d)
    assert torch.allclose(out, torch.arange(d, dtype=torch.float32).expand_as(out))


def test_flatten_spatial_latent_passthrough_2d_and_3d() -> None:
    z2 = torch.randn(7, 4)
    assert torch.equal(flatten_spatial_latent(z2), z2)
    z3 = torch.randn(2, 5, 4)
    assert flatten_spatial_latent(z3).shape == (10, 4)


def test_flatten_spatial_latent_bad_dim_raises() -> None:
    with pytest.raises(ValueError, match="expected z of shape"):
        flatten_spatial_latent(torch.randn(2, 3, 4, 5))


# ---------------------------------------------------------------------------
# global_average_pool_latent
# ---------------------------------------------------------------------------
def test_gap_latent_matches_manual_mean() -> None:
    z = torch.randn(2, 3, 4, 5, 6)
    out = global_average_pool_latent(z)
    assert out.shape == (2, 3, 4)
    assert torch.allclose(out, z.mean(dim=(3, 4)))


def test_gap_latent_constant_map_returns_constant() -> None:
    b, t, d, h, w = 2, 2, 3, 4, 4
    vals = torch.randn(b, t, d)
    z = vals[:, :, :, None, None].expand(b, t, d, h, w)
    assert torch.allclose(global_average_pool_latent(z), vals, atol=1e-6)


# ---------------------------------------------------------------------------
# assemble_spatial_rollout
# ---------------------------------------------------------------------------
def test_rollout_shapes_and_targets() -> None:
    torch.manual_seed(0)
    d, h, w, horizon = 4, 6, 4, 8
    pred = _StubPredictor(d, context_length=2)
    z = torch.randn(2, 12, d, h, w, requires_grad=True)
    pred_seq, target_seq = assemble_spatial_rollout(pred, z, horizon)
    assert pred_seq.shape == (2, horizon, d, h, w)
    assert target_seq.shape == (2, horizon, d, h, w)
    # Targets are exactly the online encoder outputs at frames cl..cl+H-1, detached.
    cl = pred.context_length
    expected = z[:, cl : cl + horizon]  # noqa: E203 (black-formatted slice)
    assert torch.allclose(target_seq, expected.detach())
    assert not target_seq.requires_grad
    assert pred_seq.requires_grad


def test_rollout_is_autoregressive_not_teacher_forced() -> None:
    # Step 0 consumes z_0,z_1; step 1 must consume z_1 and the PREDICTED z_2,
    # never the ground-truth z_2 (no teacher forcing).
    torch.manual_seed(1)
    d, h, w = 3, 4, 4
    pred = _StubPredictor(d, context_length=2)
    z = torch.randn(1, 6, d, h, w)
    pred_seq, _ = assemble_spatial_rollout(pred, z, horizon=3)
    step0 = pred(torch.cat([z[:, 0], z[:, 1]], dim=1))
    step1 = pred(torch.cat([z[:, 1], step0], dim=1))
    step2 = pred(torch.cat([step0, step1], dim=1))
    assert torch.allclose(pred_seq[:, 0], step0)
    assert torch.allclose(pred_seq[:, 1], step1)
    assert torch.allclose(pred_seq[:, 2], step2)


def test_rollout_too_short_raises() -> None:
    pred = _StubPredictor(4, context_length=2)
    z = torch.randn(2, 9, 4, 6, 4)  # need T >= 2 + 8 = 10
    with pytest.raises(ValueError, match="too short"):
        assemble_spatial_rollout(pred, z, horizon=8)


def test_rollout_encoder_gets_gradient_through_context_only() -> None:
    torch.manual_seed(2)
    d, h, w = 3, 4, 4
    horizon = 3
    pred = _StubPredictor(d, context_length=2)
    cl = pred.context_length
    z = torch.randn(1, 6, d, h, w, requires_grad=True)
    pred_seq, target_seq = assemble_spatial_rollout(pred, z, horizon=horizon)
    loss = (pred_seq - target_seq).pow(2).mean()
    loss.backward()
    # Frames 0,1 are the seed context -> must receive gradient.
    assert z.grad[:, 0].abs().sum() > 0
    assert z.grad[:, 1].abs().sum() > 0
    # Target frames cl..cl+H-1 are detached -> must receive EXACTLY zero gradient
    # (this is the "only" in the test name: the encoder is trained through the
    # rollout context, never through the online targets).
    assert z.grad[:, cl : cl + horizon].abs().sum().item() == 0.0  # noqa: E203


# ---------------------------------------------------------------------------
# build_outputs
# ---------------------------------------------------------------------------
def _spatial_batch(b: int = 2, t: int = 12, d: int = 4, h: int = 6, w: int = 4):
    z = torch.randn(b, t, d, h, w, requires_grad=True)
    batch = {
        "omega": torch.randn(b, t, 1, h * 8, w * 8),
        "cl_future": torch.randn(b, t, 1),
        "wake_target": torch.randn(b, t, 5),
    }
    return z, batch


def test_build_outputs_pred_with_lift() -> None:
    d = 4
    z, batch = _spatial_batch(d=d)
    out = build_outputs(
        z,
        batch,
        objective="pred",
        horizon=8,
        predictor=_StubPredictor(d),
        lift_head=_StubHead(d, 1),
    )
    assert set(out) >= {"z", "_z_spatial", "pred_seq", "target_seq", "cl_pred", "cl_true"}
    assert "recon_field" not in out and "wake_pred" not in out
    assert out["z"].shape[-1] == d
    assert out["cl_pred"].shape == (2, 12, 1)


def test_build_outputs_recon_with_lift() -> None:
    d = 4
    z, batch = _spatial_batch(d=d)
    out = build_outputs(
        z,
        batch,
        objective="recon",
        horizon=8,
        decoder=_StubDecoder(d),
        lift_head=_StubHead(d, 1),
    )
    assert set(out) >= {"z", "_z_spatial", "recon_field", "target_field", "cl_pred", "cl_true"}
    assert "pred_seq" not in out
    assert out["recon_field"].shape == out["target_field"].shape


def test_build_outputs_supervised_only() -> None:
    d = 4
    z, batch = _spatial_batch(d=d)
    out = build_outputs(
        z,
        batch,
        objective="none",
        horizon=8,
        lift_head=_StubHead(d, 1),
        wake_head=_StubHead(d, 5),
    )
    assert set(out) >= {"z", "_z_spatial", "cl_pred", "cl_true", "wake_pred", "wake_true"}
    assert "pred_seq" not in out and "recon_field" not in out


def test_build_outputs_missing_target_raises() -> None:
    d = 4
    z, batch = _spatial_batch(d=d)
    del batch["wake_target"]
    with pytest.raises(KeyError, match="wake head configured"):
        build_outputs(z, batch, objective="none", horizon=8, wake_head=_StubHead(d, 5))


# ---------------------------------------------------------------------------
# kit: pre-built anti-collapse module path
# ---------------------------------------------------------------------------
def test_compute_total_loss_prebuilt_module_matches_build_each_call() -> None:
    cfg = load_model_config(CANONICAL_DIR / "jepa_nowake.yaml")
    torch.manual_seed(0)
    d = 32
    z = torch.randn(64, d)
    pred_seq = torch.randn(2, 8, d)
    target_seq = pred_seq + 0.1
    outputs = {
        "z": z,
        "pred_seq": pred_seq,
        "target_seq": target_seq,
        "cl_pred": torch.randn(2, 12, 1),
        "cl_true": torch.randn(2, 12, 1),
    }
    # SIGReg resamples projections each forward; pin the seed before each call so
    # the prebuilt-module path and the build-each-call path draw the same dirs.
    ac = cfg.anti_collapse
    module = build_anticollapse(ac["method"], d, ac["sigreg"], ac["vicreg"])

    torch.manual_seed(123)
    total_prebuilt, comp_prebuilt = compute_total_loss(cfg, outputs, anticollapse_module=module)
    torch.manual_seed(123)
    total_build, comp_build = compute_total_loss(cfg, outputs)

    assert comp_prebuilt["anti_collapse"] == pytest.approx(comp_build["anti_collapse"], rel=1e-5)
    assert total_prebuilt.item() == pytest.approx(total_build.item(), rel=1e-5)
