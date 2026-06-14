"""Minimal tests for SESSION29 Track D (probe-class sweep) helpers.

Covers the contract items that do not need the heavy real latents: metric
definition, case-clustered CI behaviour, GroupKFold split hygiene (no case_id
leakage across a fold), and the provenance schema.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "session29"))
import _s29_common as cm  # noqa: E402


def test_r2_perfect_prediction():
    rng = np.random.default_rng(0)
    y = rng.normal(size=60)
    cases = np.repeat([f"c{i}" for i in range(6)], 10)
    r2, lo, hi = cm.case_clustered_r2_ci(y, y.copy(), cases, n_boot=200, seed=0)
    assert abs(r2 - 1.0) < 1e-9
    assert lo <= 1.0 <= hi + 1e-9


def test_r2_mean_prediction_is_zero():
    rng = np.random.default_rng(1)
    y = rng.normal(size=60)
    cases = np.repeat([f"c{i}" for i in range(6)], 10)
    r2, _, _ = cm.case_clustered_r2_ci(
        y, np.full_like(y, y.mean()), cases, n_boot=50, seed=0
    )
    assert abs(r2) < 1e-9


def test_groupkfold_no_case_leakage():
    groups = np.repeat([f"c{i}" for i in range(10)], 4)
    X = np.zeros((40, 3))
    for tr, te in GroupKFold(n_splits=5).split(X, X[:, 0], groups):
        assert set(groups[tr]).isdisjoint(set(groups[te]))


def test_provenance_schema():
    prov = cm.provenance([], seed=7)
    for key in ("git_sha", "command", "utc", "versions", "seed", "split"):
        assert key in prov
    assert prov["seed"] == 7
    assert prov["split"] == "v2p1"
