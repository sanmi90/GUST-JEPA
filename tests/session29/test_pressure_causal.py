"""Minimal tests for SESSION29 Track I (strictly causal pressure -> state recovery).

Covers the pure, real-latent-free helpers:
  - the window-index builder returns strictly pre-impact indices for the causal
    window and ends-at/past-impact indices for the other two,
  - is_strictly_preimpact classifies the three windows correctly,
  - the variance-weighted R^2 (re-exported from sensing_cf) on synthetic data,
  - the STRONG/WEAK verdict logic on synthetic state-R^2 tables.
No cache files or real latents are touched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "session29"))
import pressure_causal as pc  # noqa: E402


def test_causal_window_is_strictly_preimpact():
    idx = pc.window_indices(40, "preimpact_m30_to_m1")
    # frames [10, 40): 30 frames, last frame 39 < impact 40.
    assert idx.tolist() == list(range(10, 40))
    assert idx[-1] < 40
    assert pc.is_strictly_preimpact("preimpact_m30_to_m1") is True


def test_through_impact_ends_at_impact_not_strictly_pre():
    idx = pc.window_indices(40, "through_impact_m29_to_0")
    # frames [11, 41): 30 frames, last frame 40 == impact (not strictly before).
    assert idx.tolist() == list(range(11, 41))
    assert idx[-1] == 40
    assert pc.is_strictly_preimpact("through_impact_m29_to_0") is False


def test_readout_window_extends_past_impact():
    idx = pc.window_indices(40, "readout_m13_to_p16")
    # frames [26, 56): 30 frames spanning impact-14..impact+15 (NON-causal baseline).
    assert idx.tolist() == list(range(26, 56))
    assert idx[-1] > 40
    assert pc.is_strictly_preimpact("readout_m13_to_p16") is False


def test_all_windows_have_length_W():
    for w in pc.WINDOWS:
        assert len(pc.window_indices(40, w)) == pc.W


def test_window_out_of_bounds_fails_loud():
    with pytest.raises(ValueError):
        pc.window_indices(5, "preimpact_m30_to_m1")  # impact-30 = -25 < 0
    with pytest.raises(ValueError):
        pc.window_indices(115, "readout_m13_to_p16", n_frames=120)  # 115+16 = 131 > 120


def test_unknown_window_raises():
    with pytest.raises(ValueError):
        pc.window_indices(40, "not_a_window")


def test_variance_weighted_r2_perfect_and_null():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(50, 4))
    # perfect prediction -> R^2 = 1.
    assert pc.scf.variance_weighted_r2(y, y) == pytest.approx(1.0)
    # predicting the per-dimension mean -> R^2 = 0 (SSE == SST).
    ybar = np.broadcast_to(y.mean(axis=0, keepdims=True), y.shape)
    assert pc.scf.variance_weighted_r2(ybar, y) == pytest.approx(0.0, abs=1e-9)


def test_vw_r2_ci_brackets_point():
    rng = np.random.default_rng(1)
    y = rng.normal(size=(40, 3))
    yhat = y + 0.1 * rng.normal(size=y.shape)
    cases = np.repeat(np.arange(8), 5)  # 8 cases, 5 encounters each
    point, lo, hi = pc.vw_r2_ci(yhat, y, cases, seed=0)
    assert lo <= point <= hi
    assert point > 0.5  # near-perfect recovery


def test_verdict_strong_when_predictive_wins_causal_window():
    table = {
        "preimpact_m30_to_m1": {"jepa_tf_noc": 0.70, "fukami": 0.40, "pod": 0.20},
        "readout_m13_to_p16": {"jepa_tf_noc": 0.78, "fukami": 0.55, "pod": 0.30},
    }
    v = pc.decide_verdict(table, causal_window="preimpact_m30_to_m1")
    assert v["branch"] == "STRONG"
    assert v["ordering_holds_under_causal_window"] is True
    assert v["ordering_holds_per_window"]["preimpact_m30_to_m1"] is True


def test_verdict_weak_when_ordering_only_holds_noncausal():
    table = {
        "preimpact_m30_to_m1": {"jepa_tf_noc": 0.30, "fukami": 0.45, "pod": 0.20},
        "readout_m13_to_p16": {"jepa_tf_noc": 0.78, "fukami": 0.55, "pod": 0.30},
    }
    v = pc.decide_verdict(table, causal_window="preimpact_m30_to_m1")
    assert v["branch"] == "WEAK"
    assert v["ordering_holds_under_causal_window"] is False
    assert v["ordering_holds_per_window"]["readout_m13_to_p16"] is True
    assert "APPENDIX" in v["recommendation"]


def test_verdict_weak_on_tie_under_causal_window():
    # An exact tie (predictive not STRICTLY highest) is WEAK.
    table = {"preimpact_m30_to_m1": {"jepa_tf_noc": 0.50, "fukami": 0.50, "pod": 0.10}}
    v = pc.decide_verdict(table)
    assert v["branch"] == "WEAK"


def test_verdict_requires_causal_window_present():
    with pytest.raises(ValueError):
        pc.decide_verdict({"readout_m13_to_p16": {"jepa_tf_noc": 0.5}})
