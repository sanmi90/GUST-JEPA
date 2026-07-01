"""Tests for SESSION 31 Track A loss kit and config loader.

The kit is the fairness keystone: the autoencoder and JEPA models must be one
code path that differs only by config switches. These tests pin (1) each loss
term in isolation, (2) that ``compute_total_loss`` sums exactly the active
terms, (3) that every Tier-1 canonical and reference config resolves to the
on/off matrix in the Session 31 plan (Track C), and (4) that the four
fail-loud config rules raise on a crafted violation and pass on a valid file.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from src.config.kit_config import (
    KitConfigError,
    ResolvedKitConfig,
    load_model_config,
)
from src.losses.kit import (
    anti_collapse_loss,
    build_anticollapse,
    compute_total_loss,
    lift_head_loss,
    pred_mse_seq_loss,
    recon_mse_loss,
    wake_head_loss,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
KIT_PATH = REPO_ROOT / "configs" / "_kit.yaml"
CANONICAL_DIR = REPO_ROOT / "configs" / "canonical"
REFERENCE_DIR = REPO_ROOT / "configs" / "reference"
ABLATION_DIR = REPO_ROOT / "configs" / "ablation"


# ---------------------------------------------------------------------------
# Per-term unit tests
# ---------------------------------------------------------------------------
def test_recon_mse_loss_matches_manual() -> None:
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.tensor([[1.0, 0.0], [3.0, 0.0]])
    out = recon_mse_loss(pred, target)
    # squared errors: 0, 4, 0, 16 -> mean = 5
    assert out.item() == pytest.approx(5.0)
    assert out.dim() == 0


def test_recon_mse_loss_zero_when_equal() -> None:
    x = torch.randn(3, 4)
    assert recon_mse_loss(x, x.clone()).item() == pytest.approx(0.0)


def test_recon_mse_loss_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shapes must match"):
        recon_mse_loss(torch.randn(2, 3), torch.randn(2, 4))


def test_pred_mse_seq_loss_reduces_over_horizon() -> None:
    pred = torch.zeros(2, 5, 3)
    target = torch.ones(2, 5, 3)
    # every squared error is 1 -> mean 1
    assert pred_mse_seq_loss(pred, target).item() == pytest.approx(1.0)


def test_pred_mse_seq_loss_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shapes must match"):
        pred_mse_seq_loss(torch.randn(2, 5, 3), torch.randn(2, 4, 3))


def test_lift_head_loss_smooth_l1_small_residual_is_quadratic() -> None:
    # |residual| < beta -> 0.5 * r^2 / beta; with beta=1 and r=0.5 -> 0.125
    pred = torch.tensor([[0.5]])
    true = torch.tensor([[0.0]])
    assert lift_head_loss(pred, true, beta=1.0).item() == pytest.approx(0.125)


def test_wake_head_loss_uses_beta_half() -> None:
    # smooth_l1 with beta=0.5; residual 0.25 (< beta) -> 0.5 * r^2 / beta
    pred = torch.tensor([[0.25, 0.25]])
    true = torch.tensor([[0.0, 0.0]])
    expected = 0.5 * (0.25**2) / 0.5
    assert wake_head_loss(pred, true, beta=0.5).item() == pytest.approx(expected)


def test_build_anticollapse_sigreg_and_vicreg() -> None:
    sig = build_anticollapse(
        "sigreg", dim=8, sigreg={"projections": 16, "knots": 9, "knot_range": [0.2, 4.0]}, vicreg={}
    )
    vic = build_anticollapse("vicreg", dim=8, sigreg={}, vicreg={"std": 25.0, "cov": 1.0})
    z = torch.randn(32, 8)
    assert sig(z).dim() == 0
    assert vic(z).dim() == 0
    # vicreg std/cov mapped to mu/nu
    assert vic.mu == pytest.approx(25.0)
    assert vic.nu == pytest.approx(1.0)


def test_build_anticollapse_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="method"):
        build_anticollapse("bogus", dim=8, sigreg={}, vicreg={})


def test_anti_collapse_loss_flattens_btd() -> None:
    module = build_anticollapse("vicreg", dim=4, sigreg={}, vicreg={"std": 25.0, "cov": 1.0})
    z3 = torch.randn(2, 6, 4)
    out = anti_collapse_loss(z3, module)
    assert out.dim() == 0


# ---------------------------------------------------------------------------
# compute_total_loss sums exactly the active terms
# ---------------------------------------------------------------------------
def _synthetic_outputs(d: int = 8) -> dict[str, torch.Tensor]:
    return {
        "recon_field": torch.randn(3, 1, 12, 12),
        "target_field": torch.randn(3, 1, 12, 12),
        "pred_seq": torch.randn(3, 8, d),
        "target_seq": torch.randn(3, 8, d),
        "z": torch.randn(24, d),
        "cl_pred": torch.randn(3, 4, 1),
        "cl_true": torch.randn(3, 4, 1),
        "wake_pred": torch.randn(3, 4, 80),
        "wake_true": torch.randn(3, 4, 80),
    }


def test_compute_total_loss_jepa_wake_active_terms() -> None:
    cfg = load_model_config(CANONICAL_DIR / "jepa_wake.yaml", kit_path=KIT_PATH)
    outs = _synthetic_outputs()
    total, comps = compute_total_loss(cfg, outs)
    assert set(comps) - {"total"} == {"pred", "anti_collapse", "lift", "wake"}
    summed = sum(v for k, v in comps.items() if k != "total")
    assert summed == pytest.approx(comps["total"], rel=1e-5)
    assert total.item() == pytest.approx(comps["total"], rel=1e-5)
    assert "recon" not in comps


def test_compute_total_loss_ae_nowake_active_terms() -> None:
    cfg = load_model_config(CANONICAL_DIR / "ae_nowake.yaml", kit_path=KIT_PATH)
    outs = _synthetic_outputs()
    total, comps = compute_total_loss(cfg, outs)
    assert set(comps) - {"total"} == {"recon", "anti_collapse", "lift"}
    assert "pred" not in comps
    assert "wake" not in comps


def test_compute_total_loss_supervised_only_no_objective_no_anticollapse() -> None:
    cfg = load_model_config(CANONICAL_DIR / "supervised_only.yaml", kit_path=KIT_PATH)
    outs = _synthetic_outputs()
    _, comps = compute_total_loss(cfg, outs)
    assert set(comps) - {"total"} == {"lift", "wake"}


def test_compute_total_loss_missing_output_key_raises() -> None:
    cfg = load_model_config(CANONICAL_DIR / "ae_nowake.yaml", kit_path=KIT_PATH)
    outs = _synthetic_outputs()
    del outs["recon_field"]
    with pytest.raises(KeyError, match="recon_field"):
        compute_total_loss(cfg, outs)


def test_compute_total_loss_anticollapse_weighted_by_lambda() -> None:
    # Use the VICReg ablation: it is deterministic (no projection RNG), so the
    # weighted component must equal lambda * raw exactly.
    cfg = load_model_config(ABLATION_DIR / "jepa_vicreg.yaml", kit_path=KIT_PATH)
    outs = _synthetic_outputs()
    _, comps = compute_total_loss(cfg, outs)
    lam = cfg.anti_collapse["lambda"]
    assert lam == pytest.approx(0.02)
    z = outs["z"]
    module = build_anticollapse(
        cfg.anti_collapse["method"],
        z.shape[-1],
        cfg.anti_collapse["sigreg"],
        cfg.anti_collapse["vicreg"],
    )
    raw = anti_collapse_loss(z, module).item()
    assert comps["anti_collapse"] == pytest.approx(lam * raw, rel=1e-5)


# ---------------------------------------------------------------------------
# The Track C on/off matrix
# ---------------------------------------------------------------------------
MATRIX = {
    CANONICAL_DIR / "jepa_nowake.yaml": {"pred", "anti_collapse", "lift"},
    CANONICAL_DIR / "jepa_wake.yaml": {"pred", "anti_collapse", "lift", "wake"},
    CANONICAL_DIR / "ae_nowake.yaml": {"recon", "anti_collapse", "lift"},
    CANONICAL_DIR / "ae_wake.yaml": {"recon", "anti_collapse", "lift", "wake"},
    CANONICAL_DIR / "supervised_only.yaml": {"lift", "wake"},
    REFERENCE_DIR / "fukami.yaml": {"recon", "lift"},
    REFERENCE_DIR / "fukami_wake.yaml": {"recon", "lift", "wake"},
}


@pytest.mark.parametrize("path,expected", list(MATRIX.items()), ids=lambda p: getattr(p, "name", p))
def test_matrix_active_terms(path: Path, expected: set) -> None:
    cfg = load_model_config(path, kit_path=KIT_PATH)
    assert cfg.active_terms() == frozenset(expected), f"{path.name} active terms mismatch"


def test_matrix_anticollapse_method_sigreg_on_canonical_jepa_and_ae() -> None:
    for name in ("jepa_nowake", "jepa_wake", "ae_nowake", "ae_wake"):
        cfg = load_model_config(CANONICAL_DIR / f"{name}.yaml", kit_path=KIT_PATH)
        assert cfg.anti_collapse["on"] is True
        assert cfg.anti_collapse["method"] == "sigreg"


def test_jepa_vicreg_ablation_switches_method() -> None:
    cfg = load_model_config(ABLATION_DIR / "jepa_vicreg.yaml", kit_path=KIT_PATH)
    assert cfg.anti_collapse["method"] == "vicreg"
    assert cfg.active_terms() == frozenset({"pred", "anti_collapse", "lift", "wake"})


def test_ablation_passthrough_model_flags_carried() -> None:
    cnn = load_model_config(ABLATION_DIR / "jepa_cnn.yaml", kit_path=KIT_PATH)
    assert cnn.model["encoder"] == "cnn_only"
    pool = load_model_config(ABLATION_DIR / "jepa_pool.yaml", kit_path=KIT_PATH)
    assert pool.model["latent"] == "pooled"


# ---------------------------------------------------------------------------
# The four fail-loud rules
# ---------------------------------------------------------------------------
def _write_model(tmp_path: Path, body: dict, name: str = "m.yaml") -> Path:
    p = tmp_path / name
    p.write_text(yaml.safe_dump(body))
    return p


def test_rule1_both_objectives_on_raises(tmp_path: Path) -> None:
    body = {
        "name": "bad",
        "representation_objective": {"recon": {"on": True}, "pred": {"on": True}},
        "anti_collapse": {"on": True},
        "supervision": {"lift_head": {"on": True}},
    }
    with pytest.raises(KitConfigError, match="at most one"):
        load_model_config(_write_model(tmp_path, body), kit_path=KIT_PATH)


def test_rule1_no_objective_no_head_raises(tmp_path: Path) -> None:
    body = {
        "name": "bad",
        "representation_objective": {"recon": {"on": False}, "pred": {"on": False}},
        "anti_collapse": {"on": False},
        "supervision": {"lift_head": {"on": False}, "wake_head": {"on": False}},
    }
    with pytest.raises(KitConfigError, match="supervision head"):
        load_model_config(_write_model(tmp_path, body), kit_path=KIT_PATH)


def test_rule2_pred_without_anticollapse_raises(tmp_path: Path) -> None:
    body = {
        "name": "bad",
        "representation_objective": {"pred": {"on": True}},
        "anti_collapse": {"on": False},
        "supervision": {"lift_head": {"on": True}},
    }
    with pytest.raises(KitConfigError, match="anti_collapse.*mandatory"):
        load_model_config(_write_model(tmp_path, body), kit_path=KIT_PATH)


def test_rule2_supervised_only_with_anticollapse_raises(tmp_path: Path) -> None:
    body = {
        "name": "bad",
        "representation_objective": {"recon": {"on": False}, "pred": {"on": False}},
        "anti_collapse": {"on": True},
        "supervision": {"lift_head": {"on": True}, "wake_head": {"on": True}},
    }
    with pytest.raises(KitConfigError, match="off for.*supervised"):
        load_model_config(_write_model(tmp_path, body), kit_path=KIT_PATH)


def test_rule3_override_frozen_pred_horizon_raises(tmp_path: Path) -> None:
    body = {
        "name": "bad",
        "representation_objective": {"pred": {"on": True, "horizon": 16}},
        "anti_collapse": {"on": True},
        "supervision": {"lift_head": {"on": True}},
    }
    with pytest.raises(KitConfigError, match="frozen"):
        load_model_config(_write_model(tmp_path, body), kit_path=KIT_PATH)


def test_rule3_override_frozen_recon_field_raises(tmp_path: Path) -> None:
    body = {
        "name": "bad",
        "representation_objective": {"recon": {"on": True, "field": "raw_omega"}},
        "anti_collapse": {"on": True},
        "supervision": {"lift_head": {"on": True}},
    }
    with pytest.raises(KitConfigError, match="frozen"):
        load_model_config(_write_model(tmp_path, body), kit_path=KIT_PATH)


def test_rule3_override_anticollapse_lambda_raises(tmp_path: Path) -> None:
    body = {
        "name": "bad",
        "representation_objective": {"pred": {"on": True}},
        "anti_collapse": {"on": True, "lambda": 0.5},
        "supervision": {"lift_head": {"on": True}},
    }
    with pytest.raises(KitConfigError, match="frozen"):
        load_model_config(_write_model(tmp_path, body), kit_path=KIT_PATH)


def test_rule3_anticollapse_method_override_allowed(tmp_path: Path) -> None:
    body = {
        "name": "ok",
        "representation_objective": {"pred": {"on": True}},
        "anti_collapse": {"on": True, "method": "vicreg"},
        "supervision": {"lift_head": {"on": True}, "wake_head": {"on": True}},
    }
    cfg = load_model_config(_write_model(tmp_path, body), kit_path=KIT_PATH)
    assert cfg.anti_collapse["method"] == "vicreg"


def test_rule4_override_training_raises(tmp_path: Path) -> None:
    body = {
        "name": "bad",
        "representation_objective": {"pred": {"on": True}},
        "anti_collapse": {"on": True},
        "supervision": {"lift_head": {"on": True}},
        "training": {"seed": 7},
    }
    with pytest.raises(KitConfigError, match="training"):
        load_model_config(_write_model(tmp_path, body), kit_path=KIT_PATH)


def test_valid_minimal_config_passes(tmp_path: Path) -> None:
    body = {
        "name": "ok",
        "representation_objective": {"pred": {"on": True}},
        "anti_collapse": {"on": True},
        "supervision": {"lift_head": {"on": True}},
    }
    cfg = load_model_config(_write_model(tmp_path, body), kit_path=KIT_PATH)
    assert isinstance(cfg, ResolvedKitConfig)
    assert cfg.active_terms() == frozenset({"pred", "anti_collapse", "lift"})
    # frozen params inherited from the kit, not the per-model file
    assert cfg.representation_objective["pred"]["horizon"] == 8
    assert cfg.representation_objective["pred"]["loss"] == "mse_seq"
    assert cfg.training["partition"] == "v2p2"


def test_on_flag_string_off_raises(tmp_path: Path) -> None:
    # The on-key-safe loader keeps bare `off` as the string 'off', and
    # bool('off') is True, which would silently turn a term ON. The loader must
    # reject any on-flag that is not a real YAML boolean. Write raw YAML so the
    # bare unquoted `off` is exercised exactly as a hand-written file would be.
    p = tmp_path / "m.yaml"
    p.write_text(
        "name: bad\n"
        "representation_objective:\n"
        "  recon:\n"
        "    on: off\n"
        "anti_collapse:\n"
        "  on: true\n"
        "supervision:\n"
        "  lift_head:\n"
        "    on: true\n"
    )
    with pytest.raises(KitConfigError, match="boolean"):
        load_model_config(p, kit_path=KIT_PATH)


def test_on_flag_string_yes_raises(tmp_path: Path) -> None:
    # Likewise a bare `yes` value must not be accepted as a truthy on-flag.
    p = tmp_path / "m.yaml"
    p.write_text(
        "name: bad\n"
        "representation_objective:\n"
        "  pred:\n"
        "    on: yes\n"
        "anti_collapse:\n"
        "  on: true\n"
        "supervision:\n"
        "  lift_head:\n"
        "    on: true\n"
    )
    with pytest.raises(KitConfigError, match="boolean"):
        load_model_config(p, kit_path=KIT_PATH)


def test_name_defaults_to_filename_stem_when_omitted(tmp_path: Path) -> None:
    # No `name:` key in the per-model file -> resolve to the filename stem, NOT
    # the kit default name (_kit).
    body = {
        "representation_objective": {"pred": {"on": True}},
        "anti_collapse": {"on": True},
        "supervision": {"lift_head": {"on": True}},
    }
    path = _write_model(tmp_path, body, name="my_model.yaml")
    cfg = load_model_config(path, kit_path=KIT_PATH)
    assert cfg.name == "my_model"


def test_pred_target_is_online() -> None:
    # Design reconciliation: the canonical predictive target is the ONLINE
    # encoder (gray-scott / SkyJEPA lineage, CLAUDE.md "No EMA").
    cfg = load_model_config(CANONICAL_DIR / "jepa_wake.yaml", kit_path=KIT_PATH)
    assert cfg.representation_objective["pred"]["target"] == "online"


def test_audit_all_dirs_validate() -> None:
    # Coverage guard for ae_cnn, st_d64, fukami, fukami_wake and the rest: every
    # canonical, reference and ablation config must validate.
    from src.config.audit import audit_dir

    for directory in (CANONICAL_DIR, REFERENCE_DIR, ABLATION_DIR):
        rows = audit_dir(directory)
        assert rows, f"no configs found in {directory}"
        for r in rows:
            assert r["_ok"] == "true", f"{directory.name}/{r['Model']} failed: {r['notes']}"


def test_model_passthrough_keys_allowed(tmp_path: Path) -> None:
    body = {
        "name": "ok",
        "representation_objective": {"pred": {"on": True}},
        "anti_collapse": {"on": True},
        "supervision": {"lift_head": {"on": True}},
        "model": {"encoder": "cnn_only", "latent": "spatial"},
    }
    cfg = load_model_config(_write_model(tmp_path, body), kit_path=KIT_PATH)
    assert cfg.model["encoder"] == "cnn_only"


# ---------------------------------------------------------------------------
# Audit renderer
# ---------------------------------------------------------------------------
def test_audit_canonical_dir_all_ok_and_renders_matrix() -> None:
    from src.config.audit import audit_dir, render_markdown

    rows = audit_dir(CANONICAL_DIR)
    by_name = {r["Model"]: r for r in rows}
    assert all(r["_ok"] == "true" for r in rows)
    assert by_name["jepa_wake"]["pred"] == "on"
    assert by_name["jepa_wake"]["anti_collapse"] == "sigreg"
    assert by_name["jepa_wake"]["wake"] == "on"
    assert by_name["ae_nowake"]["recon"] == "on"
    assert by_name["supervised_only"]["anti_collapse"] == "off"
    md = render_markdown({"canonical": rows})
    assert "| Model | recon | pred | anti_collapse | lift | wake" in md


def test_audit_reference_bvae_flagged_pending() -> None:
    from src.config.audit import audit_dir

    rows = {r["Model"]: r for r in audit_dir(REFERENCE_DIR)}
    assert "PENDING" in rows["bvae"]["notes"]
    assert rows["bvae"]["wake"] == "TBC"
