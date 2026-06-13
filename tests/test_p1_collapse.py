"""Tests for Physics Track P1 similarity collapse (scripts/session28/p1_collapse.py).

CPU-only, fast. The synthetic tests exercise the pure fit/score primitives:
    - a known power law recovers its exponent and held-out R^2 ~ 1;
    - the variance-reduction ratio separates a true collapse from pure scatter;
    - the case-clustered CI wiring runs and brackets the point estimate;
    - the GP1 gate logic returns the right branch on constructed scores.
The skipif-real tests load the actual v2.1 artefacts iff present.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "session28"))

import p1_collapse as p1  # noqa: E402

RTOL = 1e-6


# ---------------------------------------------------------------- power-law recovery
def test_powerlaw_recovers_known_exponent():
    rng = np.random.default_rng(0)
    s = rng.uniform(0.5, 4.0, size=200)
    k_true, p_true = 1.7, 1.35
    y = k_true * np.power(s, p_true)  # exact, no noise
    fit = p1.fit_powerlaw(s, y)
    assert fit["p"] == pytest.approx(p_true, rel=1e-4)
    assert fit["k"] == pytest.approx(k_true, rel=1e-4)
    # held-out R^2 on an independent draw from the SAME law must be ~1
    s_te = rng.uniform(0.5, 4.0, size=80)
    y_te = k_true * np.power(s_te, p_true)
    fr = p1.fit_and_score(s, y, s_te, y_te, "powerlaw")
    assert fr.exponent == pytest.approx(p_true, rel=1e-4)
    assert fr.r2_heldout == pytest.approx(1.0, abs=1e-6)
    assert fr.vrr_heldout == pytest.approx(1.0, abs=1e-6)


def test_powerlaw_uses_abs_of_signed_candidate():
    # signed candidate (G<0 allowed); response is a non-negative magnitude
    rng = np.random.default_rng(1)
    s = rng.uniform(-4.0, 4.0, size=300)
    s = s[np.abs(s) > 0.3]
    y = 2.0 * np.power(np.abs(s), 1.5)
    fit = p1.fit_powerlaw(s, y)
    assert fit["p"] == pytest.approx(1.5, rel=1e-4)
    assert fit["k"] == pytest.approx(2.0, rel=1e-4)


def test_zero_strength_predicts_zero_response():
    params = {"k": 3.0, "p": 1.2}
    pred = p1.predict_powerlaw(params, np.array([0.0, 2.0]))
    assert pred[0] == 0.0
    assert pred[1] == pytest.approx(3.0 * 2.0**1.2, rel=RTOL)


# ---------------------------------------------------------------- VRR discrimination
def test_vrr_high_on_collapse_low_on_scatter():
    rng = np.random.default_rng(2)
    s_tr = rng.uniform(0.5, 4.0, size=150)
    y_collapse_tr = 1.0 * np.power(s_tr, 1.0)
    s_te = rng.uniform(0.5, 4.0, size=60)
    y_collapse_te = 1.0 * np.power(s_te, 1.0)
    fr = p1.fit_and_score(s_tr, y_collapse_tr, s_te, y_collapse_te, "powerlaw")
    assert fr.vrr_heldout > 0.99  # clean collapse removes nearly all scatter

    # pure scatter: response independent of the candidate -> VRR <= 0
    y_scatter_tr = rng.normal(5.0, 2.0, size=150)
    y_scatter_te = rng.normal(5.0, 2.0, size=60)
    fr2 = p1.fit_and_score(s_tr, y_scatter_tr, s_te, y_scatter_te, "powerlaw")
    assert fr2.vrr_heldout < 0.3  # almost no scatter is explained


def test_vrr_geq_r2_and_close_on_real_fit():
    rng = np.random.default_rng(3)
    s_tr = rng.uniform(0.5, 4.0, size=120)
    y_tr = 1.2 * np.power(s_tr, 0.9) + rng.normal(0, 0.3, size=120)
    s_te = rng.uniform(0.5, 4.0, size=50)
    y_te = 1.2 * np.power(s_te, 0.9) + rng.normal(0, 0.3, size=50)
    fr = p1.fit_and_score(s_tr, y_tr, s_te, y_te, "powerlaw")
    # VRR absorbs the held-out constant bias into the residual mean, R^2 charges it,
    # so VRR >= R^2 and the gap (the held-out calibration bias) is small here.
    assert fr.vrr_heldout >= fr.r2_heldout - 1e-12
    assert fr.vrr_heldout == pytest.approx(fr.r2_heldout, abs=0.02)


def test_vrr_equals_r2_when_residual_mean_zero():
    # exact collapse: held-out residual is identically zero (mean zero) => VRR == R^2
    s = np.linspace(0.5, 4.0, 40)
    y_pred = 1.0 * np.power(s, 1.0)
    vrr = p1.variance_reduction_ratio(y_pred, y_pred)
    from closure_matrix import r2_heldout

    r2 = r2_heldout(y_pred, y_pred)
    assert vrr == pytest.approx(1.0, abs=1e-12)
    assert vrr == pytest.approx(r2, abs=1e-12)


def test_linear_fit_recovers_slope():
    rng = np.random.default_rng(4)
    s = rng.uniform(-3, 3, size=100)
    y = 2.5 * np.abs(s) + 0.7
    fit = p1.fit_linear(s, y)
    assert fit["a"] == pytest.approx(2.5, rel=1e-6)
    assert fit["b"] == pytest.approx(0.7, abs=1e-6)


# ---------------------------------------------------------------- case-clustered CI
def test_heldout_r2_ci_brackets_point_estimate():
    rng = np.random.default_rng(5)
    # train
    s_tr = rng.uniform(0.5, 4.0, size=200)
    y_tr = 1.0 * np.power(s_tr, 1.1) + rng.normal(0, 0.2, size=200)
    # held-out with 6 cases, several encounters each
    cases = np.repeat([f"c{i}" for i in range(6)], 8)
    s_te = rng.uniform(0.5, 4.0, size=cases.size)
    y_te = 1.0 * np.power(s_te, 1.1) + rng.normal(0, 0.2, size=cases.size)
    fr = p1.fit_and_score(s_tr, y_tr, s_te, y_te, "powerlaw")
    lo, hi = p1.heldout_r2_ci(s_tr, y_tr, s_te, y_te, cases, "powerlaw", n_boot=400, seed=7)
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo <= hi
    # point estimate should sit inside a sane CI for a real collapse
    assert lo <= fr.r2_heldout <= hi + 1e-9


def test_exponent_ci_and_overlap_logic():
    # tight SE => narrow CI; identical exponents must overlap
    ci_a = p1.exponent_ci({"p": 1.0, "se_p": 0.05})
    ci_b = p1.exponent_ci({"p": 1.02, "se_p": 0.05})
    assert p1.cis_overlap(ci_a, ci_b)
    # well-separated exponents with tiny SE must NOT overlap
    ci_c = p1.exponent_ci({"p": 1.0, "se_p": 0.01})
    ci_d = p1.exponent_ci({"p": 2.0, "se_p": 0.01})
    assert not p1.cis_overlap(ci_c, ci_d)
    # NaN SE => no overlap (conservative)
    assert not p1.cis_overlap((float("nan"), float("nan")), ci_a)


def test_se_p_zero_residual_powerlaw():
    # exact law => zero residual => se_p == 0 => degenerate but finite CI
    s = np.linspace(0.5, 4.0, 50)
    y = 1.0 * np.power(s, 1.3)
    fit = p1.fit_powerlaw(s, y)
    assert fit["se_p"] == pytest.approx(0.0, abs=1e-9)
    lo, hi = p1.exponent_ci(fit)
    assert lo == pytest.approx(1.3, abs=1e-6) and hi == pytest.approx(1.3, abs=1e-6)


# ---------------------------------------------------------------- GP1 gate logic
def _mk_pooled(force_r2, force_exp, force_exp_ci, latent_r2, latent_exp, latent_exp_ci):
    """Build a minimal results dict that decide_gp1 can read (winner = s2_GD)."""
    res = {r: {c: {"pooled": {}} for c in p1.CANDIDATES} for r in p1.RESPONSES}
    res["force_dcl"]["s2_GD"]["pooled"]["powerlaw"] = {
        "exponent": force_exp,
        "exponent_ci": list(force_exp_ci),
        "r2_heldout": force_r2,
    }
    res["latent_maha"]["s2_GD"]["pooled"]["powerlaw"] = {
        "exponent": latent_exp,
        "exponent_ci": list(latent_exp_ci),
        "r2_heldout": latent_r2,
    }
    # make s2_GD the clear force winner
    for c in p1.CANDIDATES:
        if c != "s2_GD":
            res["force_dcl"][c]["pooled"]["powerlaw"] = {
                "exponent": 1.0,
                "exponent_ci": [0.9, 1.1],
                "r2_heldout": 0.1,
            }
    return res


def test_gp1_strong_branch():
    res = _mk_pooled(0.85, 1.3, (1.2, 1.4), 0.83, 1.32, (1.2, 1.45))
    gate = p1.decide_gp1(res)
    assert gate["branch"] == "STRONG"
    assert gate["winner"] == "s2_GD"
    assert gate["exponents_same_within_ci"]


def test_gp1_medium_branch_different_exponent():
    res = _mk_pooled(0.85, 1.3, (1.25, 1.35), 0.82, 2.4, (2.3, 2.5))
    gate = p1.decide_gp1(res)
    assert gate["branch"] == "MEDIUM"
    assert not gate["exponents_same_within_ci"]


def test_gp1_weak_branch():
    res = _mk_pooled(0.55, 1.3, (1.1, 1.5), 0.6, 1.3, (1.1, 1.5))
    gate = p1.decide_gp1(res)
    assert gate["branch"] == "WEAK"


def test_gp1_medium_when_latent_r2_below_threshold():
    # force collapses but latent R^2 is poor (different image strength)
    res = _mk_pooled(0.82, 1.3, (1.2, 1.4), 0.3, 1.31, (1.2, 1.42))
    gate = p1.decide_gp1(res)
    assert gate["branch"] == "MEDIUM"


# ---------------------------------------------------------------- skipif-real
REAL = p1.PHYSICS_NPZ.exists() and (p1.PRED_LATENT_DIR / "train.npz").exists()


@pytest.mark.skipif(not REAL, reason="v2.1 physics/latent artefacts not present")
def test_real_data_assembly_and_gate():
    data, meta = p1.load_p1_data()
    # candidates s1..s4 present and finite where defined
    for c in p1.CANDIDATES:
        assert c in data.candidates
        assert data.candidates[c].shape == data.split.shape
    # responses populated
    assert np.isfinite(data.responses["force_dcl"]).any()
    assert np.isfinite(data.responses["latent_maha"]).any()
    # s3 is proportional to s2 (Gamma_g = const * G*D) on nonzero rows
    s2, s3 = data.candidates["s2_GD"], data.candidates["s3_Gamma_g"]
    nz = (np.abs(s2) > 0) & np.isfinite(s2) & np.isfinite(s3)
    ratio = s3[nz] / s2[nz]
    assert np.allclose(ratio, ratio[0], rtol=1e-3)
    # orbit caveat metadata present
    assert meta["orbit_effective_dim"] < meta["latent_dim"]
    # gate runs and returns a valid branch
    results = p1.run_scoring(data, seed=0)
    gate = p1.decide_gp1(results)
    assert gate["branch"] in {"STRONG", "MEDIUM", "WEAK"}


@pytest.mark.skipif(not REAL, reason="v2.1 physics/latent artefacts not present")
def test_real_s3_s2_exponents_identical():
    # Gamma_g (s3) proportional to G*D (s2) => identical power-law exponent + R^2
    data, _ = p1.load_p1_data()
    smask = np.ones(data.split.shape, dtype=bool)
    r2 = p1.score_response_candidate(data, "force_dcl", "s2_GD", smask)
    r3 = p1.score_response_candidate(data, "force_dcl", "s3_Gamma_g", smask)
    if "powerlaw" in r2 and "powerlaw" in r3:
        assert r2["powerlaw"]["exponent"] == pytest.approx(r3["powerlaw"]["exponent"], abs=1e-6)
        assert r2["powerlaw"]["r2_heldout"] == pytest.approx(r3["powerlaw"]["r2_heldout"], abs=1e-6)
