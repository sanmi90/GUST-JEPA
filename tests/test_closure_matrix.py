"""Synthetic-data unit tests for scripts/session28/closure_matrix.py (master plan B2).

Everything below runs on constructed NPZ trees in tmp_path; no real artifacts
are needed. The single test that touches real v2.1 artifacts is guarded with
skipif-missing, mirroring tests/test_stats_lib.py. Probe regime under test:
STATE-descriptor probes on PER-frame z (see the module docstring of
closure_matrix.py).

Run targeted (the repo-wide suite is slow):

    timeout 300 pytest tests/test_closure_matrix.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "session28"))

import closure_matrix as cm  # noqa: E402

RNG = np.random.default_rng(11)

REAL_LATENTS = REPO / "outputs/session28/latents/jepa_tf_noc_d64_s42"
REAL_DNS = REPO / "outputs/session28/exp2/dns_physical_metrics.npz"
REAL_SPLIT = REPO / "configs/splits/split_v2p1.json"
real_artifacts_present = (
    (REAL_LATENTS / "train.npz").exists() and REAL_DNS.exists() and REAL_SPLIT.exists()
)


# ------------------------------------------------------------------ pure estimators


def test_r2_matches_sklearn():
    """The protocol R^2 estimator agrees with sklearn on known data."""
    from sklearn.metrics import r2_score

    y_true = RNG.normal(size=200)
    y_pred = y_true + 0.3 * RNG.normal(size=200)
    assert cm.r2_heldout(y_pred, y_true) == pytest.approx(r2_score(y_true, y_pred), abs=1e-12)
    # perfect prediction and mean prediction sanity anchors
    assert cm.r2_heldout(y_true, y_true) == pytest.approx(1.0)
    assert cm.r2_heldout(np.full_like(y_true, y_true.mean()), y_true) == pytest.approx(0.0)


def test_case_folds_no_case_straddles():
    """5-fold case-level construction: each case lives in exactly one fold."""
    cases = np.repeat([f"case_{i:02d}" for i in range(17)], 3)
    folds = cm.make_case_folds(cases, n_folds=5, seed=0)
    assert folds.shape == cases.shape
    for c in set(cases.tolist()):
        assert len(set(folds[cases == c].tolist())) == 1, f"{c} straddles folds"
    assert set(folds.tolist()) == set(range(5))
    # deterministic for a given seed; encounter order must not matter
    assert np.array_equal(folds, cm.make_case_folds(cases, n_folds=5, seed=0))
    perm = RNG.permutation(cases.size)
    refolded = cm.make_case_folds(cases[perm], n_folds=5, seed=0)
    assert np.array_equal(refolded, folds[perm])


def test_horizon_indexing_off_by_one_guard():
    """H counts frames past impact: H = 16 reads frame impact + 16 exactly."""
    n, T, d = 3, 120, 2
    z = np.zeros((n, T, d), dtype=np.float32)
    z[..., 0] = np.arange(T)[None, :]  # frame index baked into channel 0
    impact = np.array([40, 87, 88])
    z_h, valid = cm.gather_at_horizons(z, impact, [4, 16, 32])
    assert z_h.shape == (3, 3, 2)
    assert z_h[0, 1, 0] == 56  # impact 40 + H 16, not 55/57
    assert z_h[0, 0, 0] == 44
    assert z_h[1, 2, 0] == 119  # 87 + 32 = 119 is the last valid frame
    assert valid[1, 2]
    assert not valid[2, 2]  # 88 + 32 = 120 >= T must be excluded
    assert valid[2, 1]  # 88 + 16 = 104 still fine


def test_npz_key_convention_tolerance(tmp_path):
    """Singular and plural case/encounter key conventions both load."""
    n, T, d = 4, 10, 3
    meta = dict(
        z_full=RNG.normal(size=(n, T, d)).astype(np.float32),
        impact_frame=np.full(n, 2, dtype=np.int32),
    )
    cids = np.array(["caseA", "caseA", "caseB", "caseB"])
    eis = np.array([0, 1, 0, 1], dtype=np.int32)

    lat = tmp_path / "lat"
    lat.mkdir()
    np.savez(lat / "test_b.npz", case_id=cids, encounter_index=eis, **meta)  # singular
    np.savez(lat / "test_a.npz", case_ids=cids, encounter_indices=eis, **meta)  # plural

    dns_keys = {}
    for split in ("test_b", "test_a"):
        dns_keys[f"{split}_case_id"] = cids
        dns_keys[f"{split}_encounter_index"] = eis
        dns_keys[f"{split}_impact_frame"] = meta["impact_frame"]
        dns_keys[f"{split}_C_L"] = RNG.normal(size=(n, T))
    np.savez(tmp_path / "dns.npz", **dns_keys)
    dns = np.load(tmp_path / "dns.npz", allow_pickle=True)

    got = cm.load_split_data(lat, "test_b", dns, ["C_L"], "tol")
    assert got is not None and got.z.shape == (n, T, d)
    # the val alias must resolve to the on-disk test_a artifacts
    got_val = cm.load_split_data(lat, "val", dns, ["C_L"], "tol")
    assert got_val is not None and got_val.case_ids.tolist() == cids.tolist()
    assert cm.load_split_data(lat, "test_c", dns, ["C_L"], "tol") is None


def test_cell_uncertainty_bootstrap_wiring():
    """stats_lib-backed CIs: affine u -> R^2 mapping, wake-only case clustering."""
    y_true = np.repeat(RNG.normal(size=6), 4) + 0.1 * RNG.normal(size=24)
    y_pred = y_true + 0.2 * RNG.normal(size=24)
    cases = np.repeat([f"c{i}" for i in range(6)], 4)

    r2, ci, cc = cm.cell_uncertainty(
        y_pred, y_true, cases, with_case_ci=True, n_boot_enc=64, n_boot_case=64, boot_seed=0
    )
    assert r2 == pytest.approx(cm.r2_heldout(y_pred, y_true), abs=1e-12)
    assert ci[0] <= ci[1] and np.isfinite(ci).all()
    assert cc[0] <= cc[1] and np.isfinite(cc).all()
    assert ci[0] <= r2 <= ci[1]
    # reproducible per cell regardless of call order
    r2b, cib, ccb = cm.cell_uncertainty(
        y_pred, y_true, cases, with_case_ci=True, n_boot_enc=64, n_boot_case=64, boot_seed=0
    )
    assert (r2, ci, cc) == (r2b, cib, ccb)
    # non-wake rows carry no case-clustered CI
    _, _, cc_off = cm.cell_uncertainty(
        y_pred, y_true, cases, with_case_ci=False, n_boot_enc=16, n_boot_case=16, boot_seed=0
    )
    assert np.isnan(cc_off).all()


# ------------------------------------------------------------------ synthetic matrix


def _make_split(case_ids, n_enc_per_case, T, d, informative, seed):
    """One synthetic split: targets + latents that (optionally) encode them."""
    rng = np.random.default_rng(seed)
    rows_cid, rows_ei, wake, cl = [], [], [], []
    t = np.arange(T)
    for ci, cid in enumerate(case_ids):
        for ei in range(n_enc_per_case):
            rows_cid.append(cid)
            rows_ei.append(ei)
            wake.append(0.5 * ci + np.sin(2 * np.pi * t / 19.0) + 0.03 * ei)
            cl.append(-0.3 * ci + np.cos(2 * np.pi * t / 13.0))
    n = len(rows_cid)
    wake = np.asarray(wake)
    cl = np.asarray(cl)
    z = rng.normal(size=(n, T, d)).astype(np.float32)
    if informative:
        z[..., 0] = wake + 0.01 * rng.normal(size=(n, T))
        z[..., 1] = cl + 0.01 * rng.normal(size=(n, T))
    return {
        "case_ids": np.array(rows_cid),
        "encounter_indices": np.array(rows_ei, dtype=np.int32),
        "impact_frame": np.full(n, 20, dtype=np.int32),
        "z_full": z,
        "wake_enstrophy": wake,
        "C_L": cl,
    }


@pytest.fixture()
def toy_tree(tmp_path):
    """Families manifest + latents/rollouts/DNS/split-manifest tree in tmp_path."""
    T, d = 60, 6
    split_cases = {
        "train": [f"trn_{i}" for i in range(8)],
        "test_a": ["vala", "valb"],
        "test_b": ["tbi_1", "tbi_2", "tbb_1", "tbb_2"],
    }
    enc_per = {"train": 3, "test_a": 2, "test_b": 2}

    members = {"predfam_d64_s0": True, "predfam_d64_s1": True, "reconfam_d64_s0": False}
    dns_keys = {}
    lat_root = tmp_path / "latents"
    roll_root = tmp_path / "rollouts"
    for mi, (tag, informative) in enumerate(members.items()):
        lat_root.joinpath(tag).mkdir(parents=True, exist_ok=True)
        for split, cases in split_cases.items():
            blob = _make_split(cases, enc_per[split], T, d, informative, seed=100 + mi)
            np.savez(
                lat_root.joinpath(tag, f"{split}.npz"),
                z_full=blob["z_full"],
                case_ids=blob["case_ids"],
                encounter_indices=blob["encounter_indices"],
                impact_frame=blob["impact_frame"],
            )
            if mi == 0:  # DNS targets are model-independent; write once
                dns_keys[f"{split}_case_id"] = blob["case_ids"]
                dns_keys[f"{split}_encounter_index"] = blob["encounter_indices"]
                dns_keys[f"{split}_impact_frame"] = blob["impact_frame"]
                dns_keys[f"{split}_wake_enstrophy"] = blob["wake_enstrophy"]
                dns_keys[f"{split}_C_L"] = blob["C_L"]
            if tag == "predfam_d64_s0" and split == "test_b":
                rdir = roll_root / "pred_d64"
                rdir.mkdir(parents=True, exist_ok=True)
                rng = np.random.default_rng(7)
                np.savez(
                    rdir / "test_b.npz",
                    z_full=blob["z_full"] + 0.02 * rng.normal(size=blob["z_full"].shape),
                    z_markov=np.zeros_like(blob["z_full"]),  # retired; must never be read
                    case_ids=blob["case_ids"],
                    encounter_indices=blob["encounter_indices"],
                    impact_frame=blob["impact_frame"],
                )
    dns_path = tmp_path / "dns.npz"
    np.savez(dns_path, **dns_keys)

    manifest = {
        "cases": {
            **{c: {"split": "train", "tier": None} for c in split_cases["train"]},
            **{c: {"split": "train", "tier": None} for c in split_cases["test_a"]},
            "tbi_1": {"split": "test_b", "tier": "interior"},
            "tbi_2": {"split": "test_b", "tier": "interior"},
            "tbb_1": {"split": "test_b", "tier": "boundary"},
            "tbb_2": {"split": "test_b", "tier": "boundary"},
        }
    }
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(manifest))

    fam_yaml = tmp_path / "families.yaml"
    fam_yaml.write_text("""
families:
  predfam:
    objective: predictive
    macro_family: PredFam
    dims: {64: [0, 1]}
    rollouts: {seed: 0, keys: {64: pred_d64}}
  reconfam:
    objective: reconstructive
    macro_family: ReconFam
    dims: {64: [0]}
  ghost:
    objective: reconstructive
    macro_family: Ghost
    dims: {64: [0]}
""")
    cfg = cm.Config(
        families_manifest=fam_yaml,
        protocol_path=fam_yaml,  # provenance-only field, unused by run_matrix
        dns_metrics=dns_path,
        latents_root=lat_root,
        rollouts_root=roll_root,
        split_manifest=split_path,
        out_dir=tmp_path / "out",
        numbers_part=tmp_path / "out" / "closure_headline.json",
        observables=["wake_enstrophy", "C_L"],
        probe_classes=["ridge"],
        splits=["train", "val", "test_b"],
        horizons=[4, 16],
        test_b_tiers=["interior", "boundary", "pooled"],
        primary_horizon=16,
        primary_observable="wake_enstrophy",
        n_boot_enc=40,
        n_boot_case=40,
    )
    return cfg


def _cell(rows, **sel):
    got = [r for r in rows if all(r[k] == v for k, v in sel.items())]
    assert len(got) == 1, f"expected 1 row for {sel}, found {len(got)}"
    return got[0]


def test_run_matrix_end_to_end_synthetic(toy_tree):
    """Full pipeline on a constructed tree: skips, tiers, endpoints, OOF train rows."""
    cfg = toy_tree
    result = cm.run_matrix(cfg)
    rows = result["rows"]

    assert any("ghost" in s for s in result["skips"]), "missing family must be logged, not crash"
    assert set(result["members"]) == {"predfam_d64_s0", "predfam_d64_s1", "reconfam_d64_s0"}

    base = dict(endpoint="representational", observable="wake_enstrophy", probe="ridge", H=16)
    informative = _cell(rows, tag="predfam_d64_s0", split="test_b", tier="pooled", **base)
    noise = _cell(rows, tag="reconfam_d64_s0", split="test_b", tier="pooled", **base)
    assert informative["value"] > 0.9
    assert noise["value"] < 0.5
    assert informative["n"] == 8 and noise["n"] == 8

    interior = _cell(rows, tag="predfam_d64_s0", split="test_b", tier="interior", **base)
    boundary = _cell(rows, tag="predfam_d64_s0", split="test_b", tier="boundary", **base)
    assert interior["n"] == 4 and boundary["n"] == 4

    # wake rows carry case-clustered CIs; other observables do not
    assert np.isfinite(informative["cc_ci_lo"]) and np.isfinite(informative["cc_ci_hi"])
    cl_row = _cell(
        rows,
        tag="predfam_d64_s0",
        split="test_b",
        tier="pooled",
        endpoint="representational",
        observable="C_L",
        probe="ridge",
        H=16,
    )
    assert np.isnan(cl_row["cc_ci_lo"])

    # val rows resolved through the on-disk test_a alias
    val_row = _cell(rows, tag="predfam_d64_s0", split="val", tier="all", **base)
    assert val_row["n"] == 4 and val_row["value"] > 0.9

    # train rows are out-of-fold and still recover the informative latent
    train_row = _cell(rows, tag="predfam_d64_s0", split="train", tier="all", **base)
    assert train_row["n"] == 24 and train_row["value"] > 0.8

    # forecast endpoint only where the manifest declares a rollout key
    fc = [r for r in rows if r["endpoint"] == "forecast"]
    assert fc and {r["tag"] for r in fc} == {"predfam_d64_s0"}
    assert {r["split"] for r in fc} == {"test_b"}
    fc_row = _cell(
        rows,
        tag="predfam_d64_s0",
        split="test_b",
        tier="pooled",
        endpoint="forecast",
        observable="wake_enstrophy",
        probe="ridge",
        H=16,
    )
    assert fc_row["value"] > 0.8


def test_headline_part_and_gate_from_synthetic(toy_tree):
    """Headline numbers part (eval_all.py contract) + Gate GD on the toy matrix."""
    cfg = toy_tree
    rows = cm.run_matrix(cfg)["rows"]
    members = cm.load_families(cfg.families_manifest, cfg.latents_root, cfg.rollouts_root)
    part = cm.emit_headline_part(rows, members, cfg, cfg.numbers_part)

    assert part["part"] == "closure_headline"
    key = "closure_repr_wake_H16_testb_predfam"
    rec = part["numbers"][key]
    assert rec["macro"] == "NumReprWakePredFam" and rec["macro"].isalpha()
    assert rec["n"] == 2 and rec["seed_sd"] >= 0.0
    assert rec["value"] == pytest.approx(rec["seed_mean"])
    assert rec["ci_lo"] <= rec["value"] + 0.2  # lead-seed case-clustered CI present
    assert set(rec["run_tags"]) == {"predfam_d64_s0", "predfam_d64_s1"}
    assert "ghost" not in json.dumps(part)  # skipped family contributes nothing
    assert cfg.numbers_part.exists()

    verdict = cm.gate_gd_verdict(rows, cfg)
    assert verdict["branch"] == "incomplete"  # ridge only: all three classes required
    ridge = verdict["per_probe"]["ridge"]
    assert ridge["predictive_above_reconstructive"]
    assert ridge["ranking"][0] == "predfam"


def _gate_row(family, objective, probe, value, seed=0, cc_lo=None, cc_hi=None):
    nan = float("nan")
    return {
        "family": family,
        "tag": f"{family}_d64_s{seed}",
        "objective": objective,
        "d": 64,
        "seed": seed,
        "observable": "wake_enstrophy",
        "endpoint": "representational",
        "probe": probe,
        "split": "test_b",
        "tier": "pooled",
        "H": 16,
        "n": 42,
        "value": value,
        "ci_lo": value - 0.1,
        "ci_hi": value + 0.1,
        "cc_ci_lo": nan if cc_lo is None else cc_lo,
        "cc_ci_hi": nan if cc_hi is None else cc_hi,
        "in_gate": True,
    }


def _gate_cfg():
    dummy = Path(".")
    return cm.Config(
        families_manifest=dummy,
        protocol_path=dummy,
        dns_metrics=dummy,
        latents_root=dummy,
        rollouts_root=dummy,
        split_manifest=dummy,
        out_dir=dummy,
        numbers_part=dummy,
        observables=["wake_enstrophy"],
        probe_classes=["ridge", "krr_rbf", "mlp_reg"],
        splits=["test_b"],
        horizons=[16],
        test_b_tiers=["interior", "boundary", "pooled"],
    )


def test_gate_gd_strong_and_weak_branches():
    """Gate GD logic: strong when ordering + CI separation hold, weak on probe rescue."""
    cfg = _gate_cfg()
    strong_rows, weak_rows = [], []
    for probe in ("ridge", "krr_rbf", "mlp_reg"):
        strong_rows += [
            _gate_row("jepa", "predictive", probe, 0.8, cc_lo=0.7, cc_hi=0.9),
            _gate_row("fukami", "reconstructive", probe, 0.1, cc_lo=0.0, cc_hi=0.2),
        ]
        recon_val = 0.7 if probe == "mlp_reg" else 0.1  # nonlinear probe rescues the AE
        weak_rows += [
            _gate_row("jepa", "predictive", probe, 0.8, cc_lo=0.7, cc_hi=0.9),
            _gate_row(
                "fukami",
                "reconstructive",
                probe,
                recon_val,
                cc_lo=recon_val - 0.1,
                cc_hi=recon_val + 0.1,
            ),
        ]
    strong = cm.gate_gd_verdict(strong_rows, cfg)
    assert strong["branch"] == "strong"
    assert strong["rankings_identical"]

    weak = cm.gate_gd_verdict(weak_rows, cfg)
    assert weak["branch"] == "weak"
    assert weak["per_probe"]["mlp_reg"]["weak_trigger"]

    # excluded reference families must not influence the verdict
    excl = [dict(r) for r in strong_rows]
    excl.append({**_gate_row("condref", "predictive", "ridge", -0.5), "in_gate": False})
    assert cm.gate_gd_verdict(excl, cfg)["branch"] == "strong"


# --------------------------------------------------------------------- real artifacts


@pytest.mark.skipif(not real_artifacts_present, reason="v2.1 artifacts not on disk")
def test_real_jepa_tf_noc_member_smoke():
    """Representational test_b harvest on the real lead-seed latents (cheap subset)."""
    member = cm.Member(
        family="jepa_tf_noc",
        objective="predictive",
        macro_family="JepaTf",
        d=64,
        seed=42,
        tag="jepa_tf_noc_d64_s42",
        latents_dir=REAL_LATENTS,
        rollouts_dir=None,
    )
    dummy = Path(".")
    cfg = cm.Config(
        families_manifest=dummy,
        protocol_path=dummy,
        dns_metrics=REAL_DNS,
        latents_root=REAL_LATENTS.parent,
        rollouts_root=dummy,
        split_manifest=REAL_SPLIT,
        out_dir=dummy,
        numbers_part=dummy,
        observables=["C_L", "C_D", "wake_enstrophy"],
        probe_classes=["ridge"],
        splits=["test_b"],
        horizons=[16],
        test_b_tiers=["interior", "boundary", "pooled"],
        n_boot_enc=50,
        n_boot_case=100,
    )
    dns = np.load(REAL_DNS, allow_pickle=True)
    rows, _ = cm.evaluate_member(member, dns, cm.load_tier_map(REAL_SPLIT), cfg)
    pooled = [r for r in rows if r["tier"] == "pooled" and r["observable"] == "wake_enstrophy"]
    assert len(pooled) == 1 and pooled[0]["n"] == 42
    assert np.isfinite(pooled[0]["value"])
    assert np.isfinite(pooled[0]["cc_ci_lo"])  # wake row: case-clustered CI mandatory
    tiers = {r["tier"] for r in rows}
    assert tiers == {"interior", "boundary", "pooled"}
