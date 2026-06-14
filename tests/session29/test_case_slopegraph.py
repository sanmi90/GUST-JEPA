"""Minimal tests for SESSION29 Track C-min (per-case wake slopegraph).

The load-bearing claim of the figure is the case-level paired test on the 10
case-mean deltas. We exercise the two pure-stats steps the script relies on
(cm.stats_lib.case_means grouping and case_level_paired_stats sign/signed-rank)
on synthetic vectors with a known sign, and check the compute() verdict logic
on a constructed paired dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "session29"))
sys.path.insert(0, str(REPO / "scripts" / "session21"))
import _s29_common as cm  # noqa: E402
import make_case_slopegraph as mcs  # noqa: E402


def test_sign_test_on_known_positive_delta():
    # 9 positive, 1 negative -> one-sided sign test (k=9, n=10) is significant.
    delta = np.array([1.0, 2.0, 0.5, 3.0, 0.2, 1.5, 4.0, 0.1, 2.2, -0.3])
    stats = cm.stats_lib.case_level_paired_stats(delta)
    assert stats["n_cases"] == 10
    assert stats["cases_jepa_better"] == 9
    assert stats["sign_p_one_sided"] < 0.05
    assert stats["wilcoxon_p_one_sided"] < 0.05
    assert stats["median_case_mean_delta"] > 0


def test_sign_test_on_known_negative_delta():
    # All-negative delta: predictive never wins -> one-sided p must be ~1.
    delta = -np.arange(1.0, 11.0)
    stats = cm.stats_lib.case_level_paired_stats(delta)
    assert stats["cases_jepa_better"] == 0
    assert stats["sign_p_one_sided"] > 0.5


def test_case_means_groups_by_case():
    vals = np.array([1.0, 3.0, 10.0, 20.0])
    cids = np.array(["a", "a", "b", "b"])
    uc, means = cm.stats_lib.case_means(vals, cids)
    assert list(uc) == ["a", "b"]
    assert np.allclose(means, [2.0, 15.0])


def test_compute_verdict_strong_when_case_level_confirms():
    # Build a synthetic paired dataset where JEPA is uniformly far better in
    # every one of 10 cases (4 encounters each): sign + signed-rank both fire.
    rng = np.random.default_rng(0)
    cases = [f"case_{i:02d}" for i in range(10)]
    cid, yt, yp_j, yp_f = [], [], [], []
    for c in cases:
        truth = rng.normal(0, 1, size=4)
        cid += [c] * 4
        yt += truth.tolist()
        yp_j += (truth + rng.normal(0, 0.1, size=4)).tolist()  # tight
        yp_f += (truth + rng.normal(0, 2.0, size=4)).tolist()  # loose
    data = {
        "case_id": np.array(cid),
        "y_true": np.array(yt),
        "y_pred_jepa": np.array(yp_j),
        "y_pred_fukami": np.array(yp_f),
    }
    result = mcs.compute(data)
    assert result["verdict"] == "STRONG"
    assert result["case_level_paired"]["cases_jepa_better"] == 10
    assert len(result["per_case"]) == 10


def test_compute_delta_sign_convention():
    # delta = err_fukami - err_jepa must be positive when JEPA error is smaller.
    data = {
        "case_id": np.array(["a", "a", "b", "b"]),
        "y_true": np.array([0.0, 0.0, 0.0, 0.0]),
        "y_pred_jepa": np.array([0.1, 0.1, 0.1, 0.1]),  # small error
        "y_pred_fukami": np.array([1.0, 1.0, 1.0, 1.0]),  # large error
    }
    result = mcs.compute(data)
    for pc in result["per_case"]:
        assert pc["jepa_wins"] is True
        assert pc["delta_fukami_minus_jepa"] > 0
