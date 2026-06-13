"""Unit + regression tests for scripts/session28/stats_harvest.py (master plan B6).

Synthetic, fast tests that pin the harvest's load-bearing behaviour without the
real per-encounter dump:
    - the Holm wrapper orders and rejects against a hand-computed example;
    - the case-permutation paired-difference statistic and case-clustering group by
      case_id (synthetic data with a known per-case sign);
    - the abs-error / paired-difference sign convention (delta > 0 = predictive better);
    - the dump cell-key reproduces closure_matrix.dump_cell_key exactly.

A final skipif-missing integration test runs the real harvest against the on-disk
dump when present (closure_matrix.py --dump-per-encounter), checking the numbers
part validates and the macro-bound headline records exist.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "session28"))

import stats_harvest as sh  # noqa: E402
import stats_lib  # noqa: E402

# --------------------------------------------------------------------------- Holm wrapper


def test_holm_orders_and_rejects_hand_computed():
    """Hand-computed Holm step-down: 4 p-values, alpha = 0.05.

    Sorted ascending p = [0.001, 0.009, 0.02, 0.30] with m = 4:
        rank 0: (4-0)*0.001 = 0.004      <= 0.05 -> reject
        rank 1: (4-1)*0.009 = 0.027      <= 0.05 -> reject
        rank 2: (4-2)*0.02  = 0.040      <= 0.05 -> reject
        rank 3: (4-3)*0.30  = 0.30       >  0.05 -> retain
    Monotone enforcement does not raise any earlier value here.
    """
    pvals = {"a": 0.001, "b": 0.009, "c": 0.02, "d": 0.30}
    adj, survive = stats_lib.holm(pvals, alpha=0.05)
    assert adj["a"] == pytest.approx(0.004, abs=1e-12)
    assert adj["b"] == pytest.approx(0.027, abs=1e-12)
    assert adj["c"] == pytest.approx(0.040, abs=1e-12)
    assert adj["d"] == pytest.approx(0.30, abs=1e-12)
    assert survive == {"a": True, "b": True, "c": True, "d": False}


def test_holm_monotone_enforcement():
    """When a smaller raw p produces a larger step value, Holm carries it forward.

    p = [0.04, 0.041] with m = 2: rank0 = 2*0.04 = 0.08; rank1 = 1*0.041 = 0.041,
    but the running max keeps it at 0.08 (monotone non-decreasing). Neither survives.
    """
    adj, survive = stats_lib.holm({"x": 0.04, "y": 0.041}, alpha=0.05)
    assert adj["x"] == pytest.approx(0.08, abs=1e-12)
    assert adj["y"] == pytest.approx(0.08, abs=1e-12)  # carried forward, not 0.041
    assert survive == {"x": False, "y": False}


# ------------------------------------------------------------- paired-difference convention


def _cell(y_pred, y_true, case_id, enc):
    return sh.DumpCell(
        y_pred=np.asarray(y_pred, dtype=np.float64),
        y_true=np.asarray(y_true, dtype=np.float64),
        case_id=np.asarray(case_id, dtype=object).astype(str),
        encounter_index=np.asarray(enc, dtype=np.int64),
    )


def test_abs_error_and_match_sign_convention():
    """delta = err_recon - err_pred > 0 when the predictive cell tracks the target better."""
    # predictive: perfect; reconstructive: off by 2 everywhere
    pred = _cell([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], ["c0", "c0", "c1", "c1"], [0, 1, 0, 1])
    recon = _cell(
        [3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0], ["c0", "c0", "c1", "c1"], [0, 1, 0, 1]
    )
    err_pred, err_recon, case = sh.match_cells(pred, recon)
    assert np.allclose(err_pred, 0.0)
    assert np.allclose(err_recon, 2.0)
    delta = err_recon - err_pred
    assert np.all(delta > 0)  # predictive better everywhere
    assert float(delta.mean()) == pytest.approx(2.0)


def test_match_cells_reorders_on_key():
    """match_cells aligns on (case_id, encounter_index) even when row order differs."""
    pred = _cell([0.0, 1.0], [0.0, 0.0], ["c0", "c1"], [0, 0])
    # reconstructive rows in reversed order
    recon = _cell([5.0, 4.0], [0.0, 0.0], ["c1", "c0"], [0, 0])
    err_pred, err_recon, case = sh.match_cells(pred, recon)
    # pred row 0 (c0) must pair with recon's c0 row (err 4), pred row 1 (c1) with recon c1 (err 5)
    assert list(case) == ["c0", "c1"]
    assert np.allclose(err_pred, [0.0, 1.0])
    assert np.allclose(err_recon, [4.0, 5.0])


# ---------------------------------------------------------------- case-clustering grouping


def test_case_cluster_bootstrap_groups_by_case():
    """case_cluster_bootstrap resamples whole cases: per-case means are the unit."""
    rng = np.random.default_rng(0)
    # two cases, tight within-case spread, very different case means
    delta = np.concatenate([np.full(20, 1.0), np.full(20, 5.0)]) + 0.01 * rng.normal(size=40)
    case = np.array(["cA"] * 20 + ["cB"] * 20)
    cc = stats_lib.case_cluster_bootstrap(delta, case, n_boot=2000, rng=np.random.default_rng(1))
    assert cc["n_cases"] == 2
    assert set(cc["per_case_mean_delta"]) == {"cA", "cB"}
    assert cc["per_case_mean_delta"]["cA"] == pytest.approx(1.0, abs=0.05)
    assert cc["per_case_mean_delta"]["cB"] == pytest.approx(5.0, abs=0.05)
    # with only two cases the case-clustered CI is wide (resampling 2 cases): it must
    # span well beyond the encounter-mean point estimate of ~3.0
    lo, hi = cc["enc_mean_ci"]
    assert lo <= 3.0 <= hi
    assert (hi - lo) > 1.0


def test_paired_difference_block_known_sign():
    """paired_difference_block reports the right sign and small one-sided p-values."""
    # 5 cases x 4 enc; predictive uniformly closer -> all deltas > 0
    cases = np.repeat([f"c{i}" for i in range(5)], 4)
    enc = np.tile(np.arange(4), 5)
    yt = np.arange(20, dtype=float)
    pred = _cell(yt + 0.1, yt, cases, enc)
    recon = _cell(yt + 1.0, yt, cases, enc)
    blk = sh.paired_difference_block(pred, recon, boot_seed=0)
    assert blk["n_encounters"] == 20
    assert blk["n_cases"] == 5
    assert blk["enc_mean_delta"] == pytest.approx(0.9, abs=1e-9)
    assert blk["sign_k"] == 20  # predictive better on every encounter
    assert blk["sign_p_one_sided"] < 1e-4
    # case-respecting sign p: all 5 case means > 0 -> binomtest(5,5) one-sided
    assert blk["case_level_sign_p"] == pytest.approx(1.0 / 32.0, abs=1e-9)
    assert blk["case_clustered_ci"][0] > 0  # CI excludes zero on the predictive side


# --------------------------------------------------------------------------- dump cell key


def test_dump_cell_key_matches_closure_matrix():
    """sh.PerEncounterDump.key reproduces closure_matrix.dump_cell_key byte-for-byte."""
    import closure_matrix as cm

    rec = {
        "tag": "jepa_tf_noc_d64_s42",
        "observable": "wake_enstrophy",
        "endpoint": "representational",
        "probe": "ridge",
        "split": "test_b",
        "tier": "pooled",
        "H": 16,
    }
    expected = cm.dump_cell_key(rec)
    got = sh.PerEncounterDump.key(
        rec["tag"],
        rec["observable"],
        rec["endpoint"],
        rec["probe"],
        rec["split"],
        rec["tier"],
        rec["H"],
    )
    assert got == expected
    assert got == "jepa_tf_noc_d64_s42|wake_enstrophy|representational|ridge|test_b|pooled|H16"


# --------------------------------------------------------------------- Fig-8 case-id parse


def test_spearman_stat_is_scalar():
    rng = np.random.default_rng(3)
    a = rng.normal(size=30)
    b = a + 0.1 * rng.normal(size=30)
    rho = sh.spearman(a, b)
    assert isinstance(rho, float)
    assert rho > 0.8


# ------------------------------------------------------------------ integration (real dump)

DUMP = REPO / "outputs" / "session28" / "closure_matrix" / "per_encounter" / "dump.npz"
MATRIX = REPO / "outputs" / "session28" / "closure_matrix" / "matrix.npz"
artifacts_present = DUMP.exists() and MATRIX.exists()


@pytest.mark.skipif(not artifacts_present, reason="per-encounter dump / matrix not on disk")
def test_real_harvest_runs_and_validates():
    """End-to-end on the committed dump: blocks compute, numbers part is eval_all-valid."""
    import argparse

    cfg = argparse.Namespace(
        closure_dir=DUMP.parent.parent,
        boot_seed=0,
        perm_seed=0,
        n_perm=2000,
    )
    res = sh.run_harvest(cfg)
    holm = res["holm"]
    assert holm["n_tests"] >= 1
    # the primary paired cell (repr wake) must exist
    assert "repr/wake_enstrophy" in holm["tests"]
    prim = holm["tests"]["repr/wake_enstrophy"]
    assert prim["n_cases"] == 10  # test_b pooled = 10 cases
    # numbers part validates with eval_all's record validator
    sys.path.insert(0, str(REPO / "scripts" / "session28"))
    import eval_all  # noqa: E402

    part = sh.build_numbers_part(holm, res["verdicts"], res["gd"], res["fig8"])
    errs: list[str] = []
    for name, rec in part["numbers"].items():
        errs.extend(eval_all.validate_record("stats_harvest", name, rec))
    assert errs == [], errs
    # the two macro-bound headline records exist
    macros = {rec.get("macro") for rec in part["numbers"].values()}
    assert "NumPairedWakeRepr" in macros
    assert "NumHolmSurvivors" in macros
