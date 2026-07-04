"""Tests for the SESSION 31 Track E ablation builds of :class:`CanonicalModel`.

Each of the five one-axis ablations must build a :class:`CanonicalModel` of the
right KIND (encoder class, latent mode, latent dim) and forward without error,
while the canonical ``cnn_vit`` + ``spatial`` build stays byte-identical to the
spine (an existing checkpoint would still load ``strict=True``; here we assert the
encoder identity and that the anti-collapse latent is unchanged).

The ablations move exactly one axis versus their reference:
    jepa_cnn / ae_cnn   -> encoder cnn_only (ViT removed), spatial latent kept
    st_d64              -> encoder cnn_vit_temporal + d=64, spatial latent kept
    jepa_pool           -> pooled latent (cnn_vit kept)
    jepa_vicreg         -> anti_collapse.method=vicreg (cnn_vit + spatial kept)
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.config.kit_config import load_model_config
from src.losses.kit import build_anticollapse
from src.models.encoder import (
    CNNOnlyEncoder,
    HybridCNNViTEncoder,
    SpatioTemporalCNNViTEncoder,
)
from src.models.sigreg import SIGReg
from src.models.vicreg import VICReg
from src.training.canonical_model import CanonicalModel

REPO_ROOT = Path(__file__).resolve().parent.parent
ABLATION_DIR = REPO_ROOT / "configs" / "ablation"
CANONICAL_DIR = REPO_ROOT / "configs" / "canonical"

# (encoder class, latent mode, latent dim, objective) expected per ablation.
_EXPECT = {
    "jepa_cnn": (CNNOnlyEncoder, "spatial", 32, "pred"),
    "ae_cnn": (CNNOnlyEncoder, "spatial", 32, "recon"),
    "st_d64": (SpatioTemporalCNNViTEncoder, "spatial", 64, "pred"),
    "jepa_pool": (HybridCNNViTEncoder, "pooled", 32, "pred"),
    "jepa_vicreg": (HybridCNNViTEncoder, "spatial", 32, "pred"),
}


def _batch(b: int = 2, t: int = 12, d_wake: int = 80) -> dict[str, torch.Tensor]:
    return {
        "omega": torch.randn(b, t, 1, 192, 96),
        "cl_future": torch.randn(b, t, 1),
        "wake_target": torch.randn(b, t, d_wake),
        "c": torch.randn(b, 3),
    }


def test_each_ablation_builds_the_right_kind() -> None:
    for name, (cls, mode, d, obj) in _EXPECT.items():
        cfg = load_model_config(ABLATION_DIR / f"{name}.yaml")
        model = CanonicalModel(cfg, latent_dim=d)
        assert isinstance(model.encoder, cls), (name, type(model.encoder).__name__)
        assert model.encoder.latent_mode == mode, (name, model.encoder.latent_mode)
        assert model.latent_dim == d, (name, model.latent_dim)
        assert model.objective == obj, (name, model.objective)
        assert model.encoder.latent_grid == (24, 12), (name, model.encoder.latent_grid)


def test_each_ablation_forwards() -> None:
    for name, (_cls, _mode, d, obj) in _EXPECT.items():
        cfg = load_model_config(ABLATION_DIR / f"{name}.yaml")
        model = CanonicalModel(cfg, latent_dim=d).train()
        out = model(_batch())
        # anti-collapse latent is always (N, d); the spatial latent is (B, T, d, h, w).
        assert out["z"].shape[-1] == d, (name, out["z"].shape)
        assert out["_z_spatial"].shape == (2, 12, d, 24, 12), (name, out["_z_spatial"].shape)
        if obj == "pred":
            assert "pred_seq" in out and "target_seq" in out
        elif obj == "recon":
            assert "recon_field" in out and "target_field" in out
        # lift head is on for every ablation.
        assert "cl_pred" in out


def test_pooled_ablation_anticollapse_is_pooled_not_broadcast() -> None:
    # jepa_pool: anti-collapse must see the (B*T, d) pooled latent, NOT the
    # (B*T*h*w, d) broadcast copies fed to the predictor / decoder / heads.
    cfg = load_model_config(ABLATION_DIR / "jepa_pool.yaml")
    model = CanonicalModel(cfg, latent_dim=32).train()
    out = model(_batch())
    assert out["z"].shape == (2 * 12, 32), out["z"].shape
    assert out["_z_spatial"].shape == (2, 12, 32, 24, 12), out["_z_spatial"].shape
    assert model.pooled_adapter is not None


def test_st_d64_uses_d64_latent() -> None:
    cfg = load_model_config(ABLATION_DIR / "st_d64.yaml")
    model = CanonicalModel(cfg, latent_dim=64).train()
    out = model(_batch())
    assert out["_z_spatial"].shape[2] == 64, out["_z_spatial"].shape


def test_jepa_vicreg_builds_vicreg_and_others_sigreg() -> None:
    cfg_v = load_model_config(ABLATION_DIR / "jepa_vicreg.yaml")
    ac = cfg_v.anti_collapse
    assert ac["method"] == "vicreg"
    assert isinstance(build_anticollapse("vicreg", 32, ac["sigreg"], ac["vicreg"]), VICReg)
    cfg_s = load_model_config(ABLATION_DIR / "jepa_cnn.yaml")
    acs = cfg_s.anti_collapse
    assert acs["method"] == "sigreg"
    assert isinstance(build_anticollapse("sigreg", 32, acs["sigreg"], acs["vicreg"]), SIGReg)


def test_canonical_spine_is_unchanged_cnn_vit_spatial() -> None:
    # The canonical build must remain HybridCNNViTEncoder in spatial mode with no
    # pooled adapter, so existing checkpoints load byte-for-byte.
    cfg = load_model_config(CANONICAL_DIR / "jepa_nowake.yaml")
    model = CanonicalModel(cfg, latent_dim=32)
    assert isinstance(model.encoder, HybridCNNViTEncoder)
    assert model.encoder.latent_mode == "spatial"
    assert model.pooled_adapter is None
    assert model.encoder_kind == "cnn_vit" and model.latent_kind == "spatial"


# ---------------------------------------------------------------------------
# Session 34 Track C: conditioning-cube cells build exactly the right modules
# ---------------------------------------------------------------------------
# name -> (lift_head, wake_head, nearbody_head, objective)
_TRACKC_EXPECT = {
    "jepa_pool_c0": (False, False, False, "pred"),
    "jepa_pool_w": (False, True, False, "pred"),
    "jepa_pool_n": (False, False, True, "pred"),
    "jepa_pool_ln": (True, False, True, "pred"),
    "jepa_pool_wn": (False, True, True, "pred"),
    "jepa_pool_lwn": (True, True, True, "pred"),
    "ae_w_pool": (False, True, False, "recon"),
}


def _trackc_batch(b: int = 2, t: int = 12) -> dict[str, torch.Tensor]:
    return {
        "omega": torch.randn(b, t, 1, 192, 96),
        "cl_future": torch.randn(b, t, 1),
        "wake_target": torch.randn(b, t, 80),
        "nearbody_target": torch.randn(b, t, 80),
        "c": torch.randn(b, 3),
    }


def test_trackc_cells_build_the_right_heads() -> None:
    for name, (lift, wake, nearbody, obj) in _TRACKC_EXPECT.items():
        cfg = load_model_config(ABLATION_DIR / f"{name}.yaml")
        model = CanonicalModel(cfg, latent_dim=32)
        assert model.latent_kind == "pooled", name
        assert (model.lift_head is not None) == lift, name
        assert (model.wake_head is not None) == wake, name
        assert (model.nearbody_head is not None) == nearbody, name
        assert model.objective == obj, name
        if nearbody:
            # Byte-identical head capacity to the wake head (80-D, same class).
            assert type(model.nearbody_head).__name__ == "WakeObservableHead"


def test_trackc_cells_forward_emit_the_right_keys() -> None:
    torch.manual_seed(0)
    for name, (lift, wake, nearbody, obj) in _TRACKC_EXPECT.items():
        cfg = load_model_config(ABLATION_DIR / f"{name}.yaml")
        model = CanonicalModel(cfg, latent_dim=32)
        model.eval()
        with torch.no_grad():
            outs = model(_trackc_batch(b=1, t=10))
        assert ("cl_pred" in outs) == lift, name
        assert ("wake_pred" in outs) == wake, name
        assert ("nearbody_pred" in outs) == nearbody, name
        if nearbody:
            assert outs["nearbody_pred"].shape[-1] == 80, name
        if obj == "pred":
            assert "pred_seq" in outs, name
        else:
            assert "recon_field" in outs, name


def test_trackc_nearbody_head_in_downstream_parameters() -> None:
    cfg = load_model_config(ABLATION_DIR / "jepa_pool_n.yaml")
    model = CanonicalModel(cfg, latent_dim=32)
    nb_params = {id(p) for p in model.nearbody_head.parameters()}
    downstream = {id(p) for p in model.downstream_parameters()}
    assert nb_params <= downstream


def test_trackc_existing_pooled_cell_has_no_nearbody_head() -> None:
    # ANCHOR: the kit default keeps the CLW lineage cell byte-compatible
    # (state dict gains no keys; old checkpoints still load strict=True).
    cfg = load_model_config(ABLATION_DIR / "jepa_pool.yaml")
    model = CanonicalModel(cfg, latent_dim=32)
    assert model.nearbody_head is None
    assert not any(k.startswith("nearbody_head") for k in model.state_dict())
