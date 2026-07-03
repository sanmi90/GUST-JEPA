"""Tests for the D250 native pooled pipeline (vector training predictor).

The Session 33 D250 decision replaces the pooled flagship's training predictor:
instead of tiling the pooled 32-vector onto the 24 x 12 feature grid and rolling
a spatial ResUNet, the model rolls the (B, T, d) vector directly with the v2.1
:class:`~src.models.predictor.AutoregressivePredictor` (cond_dim=0). These tests
pin the new pieces:

    assemble_vector_rollout   open-loop semantics on (B, T, d) sequences
    CanonicalModel            predictor_class='transformer' build + forward
    guard rails               transformer requires pooled latent; bad names raise

All CPU-friendly (tiny batches), matching the suite's conventions.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.config.kit_config import load_model_config
from src.models.predictor import AutoregressivePredictor
from src.training.canonical_model import (
    CanonicalModel,
    assemble_vector_rollout,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
ABLATION_DIR = REPO_ROOT / "configs" / "ablation"

B, T, D, H = 2, 12, 8, 4


def _predictor() -> AutoregressivePredictor:
    torch.manual_seed(0)
    return AutoregressivePredictor(
        latent_dim=D, cond_dim=0, hidden_dim=32, depth=2, heads=4, max_seq_len=32
    ).train()


def _batch(b: int = 2, t: int = 12, d_wake: int = 80) -> dict[str, torch.Tensor]:
    return {
        "omega": torch.randn(b, t, 1, 192, 96),
        "cl_future": torch.randn(b, t, 1),
        "wake_target": torch.randn(b, t, d_wake),
        "c": torch.randn(b, 3),
    }


def test_vector_rollout_shapes_and_grad() -> None:
    pred = _predictor()
    z = torch.randn(B, T, D, requires_grad=True)
    pred_seq, target_seq = assemble_vector_rollout(pred, z, horizon=H)
    assert pred_seq.shape == (B, H, D)
    assert target_seq.shape == (B, H, D)
    # rollout is autograd-attached; targets are detached online-encoder outputs.
    assert pred_seq.requires_grad
    assert not target_seq.requires_grad
    # targets are exactly frames cl..cl+H-1 of z (cl = 2).
    assert torch.equal(target_seq, z[:, 2 : 2 + H].detach())


def test_vector_rollout_is_open_loop() -> None:
    # Prediction at step s must NOT depend on encoder frames cl..cl+s-1 (those
    # are targets, not inputs): perturbing frame cl leaves the rollout unchanged.
    pred = _predictor().eval()
    z = torch.randn(B, T, D)
    with torch.no_grad():
        a, _ = assemble_vector_rollout(pred, z, horizon=H)
        z_perturbed = z.clone()
        z_perturbed[:, 2:] += 100.0  # everything after the 2-frame seed
        b, _ = assemble_vector_rollout(pred, z_perturbed, horizon=H)
    assert torch.allclose(a, b), "rollout consumed post-seed encoder frames (not open loop)"


def test_vector_rollout_backward_reaches_encoder_side() -> None:
    pred = _predictor()
    z = torch.randn(B, T, D, requires_grad=True)
    pred_seq, target_seq = assemble_vector_rollout(pred, z, horizon=H)
    loss = torch.nn.functional.mse_loss(pred_seq, target_seq)
    loss.backward()
    assert z.grad is not None
    # gradient flows through the seed frames (grad-attached context)...
    assert z.grad[:, :2].abs().sum() > 0
    # ...and the detached targets contribute nothing.
    assert torch.all(z.grad[:, 2 + H :] == 0)


def test_vector_rollout_rejects_short_sequences_and_bad_shapes() -> None:
    pred = _predictor()
    with pytest.raises(ValueError):
        assemble_vector_rollout(pred, torch.randn(B, 3, D), horizon=H)  # T < cl + H
    with pytest.raises(ValueError):
        assemble_vector_rollout(pred, torch.randn(B, T, D, 4), horizon=H)  # not (B, T, d)
    with pytest.raises(ValueError):
        assemble_vector_rollout(pred, torch.randn(B, T, D), horizon=0)


def test_canonical_model_transformer_builds_and_forwards() -> None:
    cfg = load_model_config(ABLATION_DIR / "jepa_pool.yaml")
    model = CanonicalModel(cfg, latent_dim=32, predictor_class="transformer").train()
    assert isinstance(model.predictor, AutoregressivePredictor)
    assert model.predictor.cond_dim == 0
    out = model(_batch())
    # rollout is on the raw pooled vector: (B, H_roll, d), no spatial dims.
    assert out["pred_seq"].shape == (2, model.horizon, 32), out["pred_seq"].shape
    assert out["target_seq"].shape == out["pred_seq"].shape
    assert not out["target_seq"].requires_grad
    # anti-collapse still sees the (B*T, d) pooled latent; heads still fire.
    assert out["z"].shape == (2 * 12, 32)
    assert "cl_pred" in out and "wake_pred" in out
    # end-to-end backward through encoder + predictor + heads.
    loss = (
        torch.nn.functional.mse_loss(out["pred_seq"], out["target_seq"])
        + out["cl_pred"].pow(2).mean()
    )
    loss.backward()
    enc_grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    assert len(enc_grads) > 0, "no gradient reached the encoder"


def test_canonical_model_transformer_requires_pooled() -> None:
    cfg = load_model_config(ABLATION_DIR / "jepa_cnn.yaml")  # spatial latent
    with pytest.raises(ValueError, match="pooled"):
        CanonicalModel(cfg, latent_dim=32, predictor_class="transformer")
    cfg_pool = load_model_config(ABLATION_DIR / "jepa_pool.yaml")
    with pytest.raises(ValueError, match="predictor_class"):
        CanonicalModel(cfg_pool, latent_dim=32, predictor_class="lstm")


def test_default_resunet_path_unchanged() -> None:
    # Existing checkpoints must keep loading: the default build is byte-identical.
    cfg = load_model_config(ABLATION_DIR / "jepa_pool.yaml")
    model = CanonicalModel(cfg, latent_dim=32)
    assert model.predictor_class == "resunet"
    assert not isinstance(model.predictor, AutoregressivePredictor)
    out = model.train()(_batch())
    assert out["pred_seq"].dim() == 5  # spatial rollout (B, H, d, h, w)
