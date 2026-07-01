"""TDD for the Session 31 Track F report assembly + case-clustered bootstrap.

No GPU, no cache, no encoder: pure math and dict plumbing. Covers
* the aggregators (VRMSE num/den, R^2 SS_res/SS_tot, mean) and that the full-sample
  bootstrap point equals the direct aggregate;
* case-clustered vs i.i.d. row bootstrap on a synthetic case-structured dataset
  with one outlier case (block resampling must give the WIDER CI);
* percentile CI ordering and NaN handling;
* headline-record merge: required ``value`` key, no duplicate number names, no
  duplicate macro names, alphabetic macro constraint;
* macro emission format (``\\providecommand`` + ``lo``/``hi`` companions).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.report_session31 import (
    agg_mean,
    agg_r2,
    agg_vrmse,
    apply_cis,
    bootstrap_ci,
    build_headline_records,
    case_clustered_bootstrap,
    emit_macros,
    merge_and_validate,
    percentile_ci,
    r2_contribs,
    random_row_bootstrap,
    validate_record,
    vrmse_contribs,
)


# --------------------------------------------------------------------------- aggregators
def test_agg_vrmse_matches_ratio_of_sums() -> None:
    true = np.array([[0.0, 2.0], [4.0, 6.0]])  # mean 3, den = 9+1+1+9 = 20
    pred = true + 1.0  # num = 4
    c = vrmse_contribs(pred, true)
    assert agg_vrmse(c) == pytest.approx(np.sqrt(4.0 / 20.0))
    # per-row num/den sum back to the aggregate
    assert c["num"].sum() == pytest.approx(4.0)
    assert c["den"].sum() == pytest.approx(20.0)


def test_agg_r2_matches_direct() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    yhat = np.array([1.1, 1.9, 3.2, 3.7])
    c = r2_contribs(y, yhat)
    ybar = y.mean()
    expect = 1.0 - ((y - yhat) ** 2).sum() / ((y - ybar) ** 2).sum()
    assert agg_r2(c) == pytest.approx(expect)


def test_agg_mean() -> None:
    assert agg_mean({"val": np.array([0.2, 0.4, 0.6])}) == pytest.approx(0.4)


def test_full_sample_bootstrap_point_equals_aggregate() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    yhat = y + rng.normal(scale=0.3, size=200)
    case_ids = np.repeat(np.arange(10), 20)
    c = r2_contribs(y, yhat)
    rec = bootstrap_ci(case_ids, c, agg_r2, n_boot=200, seed=1)
    assert rec["value"] == pytest.approx(agg_r2(c))
    assert rec["ci_low"] <= rec["value"] <= rec["ci_high"]
    assert rec["n"] == 200


# --------------------------------------------------------------------------- clustered vs iid
def test_case_clustered_wider_than_iid_with_outlier_case() -> None:
    """One outlier case dominates the variance: block resampling gives a wider CI.

    Build 6 cases: five 'ordinary' cases with small residuals and one 'outlier'
    case with large residuals. Under i.i.d. row resampling the outlier's rows are
    diluted (always ~1/6 of the draw); under case resampling the outlier is
    present 0,1,2,... times, so the aggregate swings much harder.
    """
    rng = np.random.default_rng(3)
    ss_res_ord = 0.01  # per-row squared error, ordinary cases
    ss_res_out = 4.0  # per-row squared error, outlier case
    ss_tot = 1.0  # per-row total (fixed)
    rows_per_case = 40
    case_ids = []
    ss_res = []
    ss_tot_arr = []
    for ci in range(6):
        case_ids += [ci] * rows_per_case
        base = ss_res_out if ci == 5 else ss_res_ord
        ss_res += list(base + 0.001 * rng.normal(size=rows_per_case) ** 2)
        ss_tot_arr += [ss_tot] * rows_per_case
    case_ids = np.array(case_ids)
    contribs = {"ss_res": np.array(ss_res), "ss_tot": np.array(ss_tot_arr)}

    clustered = case_clustered_bootstrap(case_ids, contribs, agg_r2, n_boot=2000, seed=7)
    iid = random_row_bootstrap(contribs, agg_r2, n_boot=2000, seed=7)
    w_clustered = np.subtract(*percentile_ci(clustered)[::-1])
    w_iid = np.subtract(*percentile_ci(iid)[::-1])
    assert w_clustered > 1.5 * w_iid


def test_case_drawn_multiple_times_counts_rows_multiply() -> None:
    """A degenerate single-case dataset: every resample gathers that case's rows.

    With one case the block bootstrap point never moves (all draws identical),
    so the CI collapses to the point estimate.
    """
    case_ids = np.zeros(50, dtype=int)
    y = np.linspace(0, 1, 50)
    c = r2_contribs(y, y * 0.9)
    samples = case_clustered_bootstrap(case_ids, c, agg_r2, n_boot=100, seed=0)
    assert np.allclose(samples, samples[0])


def test_vrmse_num_den_aggregation_over_cases() -> None:
    """Aggregated-VRMSE bootstrap ratios sum num/den per case, not per-row VRMSE."""
    rng = np.random.default_rng(11)
    case_ids = np.repeat(np.arange(5), 30)
    true = rng.normal(size=(150, 8))
    pred = true + rng.normal(scale=0.5, size=(150, 8))
    c = vrmse_contribs(pred, true)
    samples = case_clustered_bootstrap(case_ids, c, agg_vrmse, n_boot=500, seed=2)
    # every sample is a valid sqrt(ratio-of-sums): finite, positive
    assert np.all(np.isfinite(samples))
    assert np.all(samples > 0)
    # point matches the whole-set aggregate
    assert agg_vrmse(c) == pytest.approx(np.sqrt(c["num"].sum() / c["den"].sum()))


# --------------------------------------------------------------------------- percentile CI
def test_percentile_ci_ordering_and_nan() -> None:
    s = np.arange(100.0)
    lo, hi = percentile_ci(s, 2.5, 97.5)
    assert lo < hi
    s2 = np.concatenate([s, [np.nan, np.nan]])
    lo2, hi2 = percentile_ci(s2)
    assert lo2 == pytest.approx(lo) and hi2 == pytest.approx(hi)


# --------------------------------------------------------------------------- merge / validate
def test_validate_record_requires_value_and_alpha_macro() -> None:
    assert validate_record("x", {"value": 1.0, "macro": "GoodName"}) == []
    assert any("missing" in e for e in validate_record("x", {"macro": "M"}))
    assert any("alphabetic" in e for e in validate_record("x", {"value": 1.0, "macro": "Bad2"}))
    assert any("unknown" in e for e in validate_record("x", {"value": 1.0, "junk": 3}))


def test_merge_rejects_duplicate_names() -> None:
    p1 = {"numbers": {"a": {"value": 1.0, "macro": "Aa"}}}
    p2 = {"numbers": {"a": {"value": 2.0, "macro": "Bb"}}}
    with pytest.raises(ValueError, match="duplicate number name"):
        merge_and_validate([p1, p2])


def test_merge_rejects_duplicate_macros() -> None:
    p1 = {"numbers": {"a": {"value": 1.0, "macro": "Same"}}}
    p2 = {"numbers": {"b": {"value": 2.0, "macro": "Same"}}}
    with pytest.raises(ValueError, match="duplicate macro"):
        merge_and_validate([p1, p2])


def test_merge_accepts_clean_parts() -> None:
    p1 = {"numbers": {"a": {"value": 1.0, "macro": "Aa"}}}
    p2 = {"numbers": {"b": {"value": 2.0, "macro": "Bb"}}}
    merged = merge_and_validate([p1, p2])
    assert set(merged) == {"a", "b"}


# --------------------------------------------------------------------------- emit
def test_emit_macros_format_with_ci() -> None:
    numbers = {
        "n1": {
            "value": 0.712,
            "macro": "ReprFloorVrmseJepaNoWake",
            "fmt": "%.3f",
            "ci_low": 0.68,
            "ci_high": 0.74,
        },
        "n2": {"value": 0.53, "macro": "ForeMeritJepaNoWakeHiEight"},
    }
    tex = emit_macros(numbers, header=["% test"])
    assert "\\providecommand{\\ReprFloorVrmseJepaNoWake}{0.712}" in tex
    assert "\\providecommand{\\ReprFloorVrmseJepaNoWakelo}{0.680}" in tex
    assert "\\providecommand{\\ReprFloorVrmseJepaNoWakehi}{0.740}" in tex
    assert "\\providecommand{\\ForeMeritJepaNoWakeHiEight}{0.53}" in tex
    # no CI companion when absent
    assert "ForeMeritJepaNoWakeHiEightlo" not in tex


# --------------------------------------------------------------------------- headline build
def _tiny_payloads():
    q1 = {"models": {}}
    q2 = {"models": {}}
    q3 = {"models": {}}
    targets = ("C_L", "C_D", "wake_enstrophy", "circulation_pos", "circulation_neg")
    for m in ("jepa_nowake", "regAE"):
        q1["models"][m] = {
            "decode_floor": {"windowed": {"vrmse": 0.7, "ssim": 0.9, "n": 2016}},
            "probes": {
                "windowed": {t: {"linear_r2": 0.5, "mlp_r2": 0.6, "n_eval": 2016} for t in targets}
            },
        }
        hz = list(range(1, 17))
        q2["models"][m] = {
            "predictors": {
                "resunet_matched": {
                    "field_vrmse": {
                        "horizons": hz,
                        "model": [0.7] * 16,
                        "floor": [0.71] * 16,
                        "persistence": [0.9] * 16,
                        "n": [2016] * 16,
                    },
                    "observable_closure": {
                        t: {"horizons": hz, "model_mlp_r2": [0.5] * 16, "n": [2016] * 16}
                        for t in targets
                    },
                }
            }
        }
        q3["models"][m] = {
            "pressure_to_obs": {
                "latent_r2": 0.4,
                "per_observable": {t: {"pressure_linear_r2": 0.2, "n": 2016} for t in targets},
            },
            "pressure_to_field": {"pressure_vrmse": 0.99, "n": 2016},
        }
    return q1, q2, q3


def test_build_headline_records_merges_and_validates() -> None:
    q1, q2, q3 = _tiny_payloads()
    recs = build_headline_records(q1, q2, q3, models=("jepa_nowake", "regAE"))
    # merge must pass validation (unique names + macros, required value)
    numbers = merge_and_validate([{"numbers": recs}])
    assert numbers  # non-empty
    # spot-check a couple of headline cells and their macros
    assert numbers["q1.floor.vrmse.jepa_nowake"]["macro"] == "ReprFloorVrmseJepaNoWake"
    assert numbers["q2.merit.mean_obs.h8.regAE"]["value"] == pytest.approx(0.5)
    assert numbers["q1.floor.vrmse.jepa_nowake"]["value"] == pytest.approx(0.7)
    gap = numbers["q2.field_gap.h8.jepa_nowake"]["value"]
    assert gap == pytest.approx(0.7 - 0.71)


def test_apply_cis_overlays_bounds() -> None:
    q1, q2, q3 = _tiny_payloads()
    recs = build_headline_records(q1, q2, q3, models=("jepa_nowake", "regAE"))
    ci = {
        "q1.floor.vrmse.jepa_nowake": {
            "ci_low": 0.66,
            "ci_high": 0.75,
            "n": 2000,
            "recompute_value": 0.70,
        }
    }
    out = apply_cis(recs, ci)
    r = out["q1.floor.vrmse.jepa_nowake"]
    assert r["ci_low"] == 0.66 and r["ci_high"] == 0.75 and r["n"] == 2000
    # value preserved; recompute matched within tol so no note
    assert r["value"] == pytest.approx(0.7)
    assert "note" not in r
