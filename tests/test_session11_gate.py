"""Tests for the Session 11 pre-decoder gate logic.

The gate compares a candidate's wake-probe r2 + PR against the Session 9
baseline reference and emits a single PASS/FAIL signal per the criteria in
``SESSION11_WAKE_RESULTS_FIRST.md``. We test the gate by constructing
synthetic baseline + candidate dicts that hit each criterion in turn.

The function under test is ``scripts.session11_summarize_track1.gate_verdict``,
which the summary script invokes for each Track 1 / Track 2 candidate.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_gate_module():
    """Dynamically import the script; it's outside ``src/`` so we use spec_from_file_location."""
    spec = importlib.util.spec_from_file_location(
        "session11_summarize_track1",
        REPO / "scripts" / "session11_summarize_track1.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_BASELINE = {
    "r2_patch_signed": {"r2_overall": 0.30, "out_dim": 64},
    "r2_patch_signed_spectrum": {"r2_overall": 0.35, "out_dim": 80},
    "r2_wake_coarse_pool": {"r2_overall": 0.27, "out_dim": 288},
    "r2_enstrophy_scalar": {"r2_overall": 0.80, "out_dim": 1},
    "r2_cl": {"r2_overall": 0.79},
    "r2_GDY": {"r2_G": 0.95, "r2_D": 0.85, "r2_Y": 0.86, "r2_overall": 0.885},
    "pr": 2.30,
}


def _candidate_perfect():
    """A candidate that passes every gate criterion comfortably."""
    return {
        "r2_patch_signed": {"r2_overall": 0.45, "out_dim": 64},
        "r2_patch_signed_spectrum": {"r2_overall": 0.55, "out_dim": 80},
        "r2_wake_coarse_pool": {"r2_overall": 0.40, "out_dim": 288},
        "r2_enstrophy_scalar": {"r2_overall": 0.85, "out_dim": 1},
        "r2_cl": {"r2_overall": 0.80},
        "r2_GDY": {"r2_G": 0.96, "r2_D": 0.86, "r2_Y": 0.86, "r2_overall": 0.89},
        "pr": 2.40,
    }


def test_gate_passes_perfect_candidate():
    mod = _load_gate_module()
    v = mod.gate_verdict(_BASELINE, _candidate_perfect())
    assert v["GATE"] is True
    assert v["pass_patch"] and v["pass_spectrum"] and v["pass_gdy"]
    assert v["pass_cl"] and v["pass_pr"]


def test_gate_fails_when_patch_improvement_insufficient():
    mod = _load_gate_module()
    cand = _candidate_perfect()
    cand["r2_patch_signed"]["r2_overall"] = 0.35  # +0.05, below 0.10 threshold
    v = mod.gate_verdict(_BASELINE, cand)
    assert v["pass_patch"] is False
    assert v["GATE"] is False


def test_gate_fails_when_spectrum_improvement_insufficient():
    mod = _load_gate_module()
    cand = _candidate_perfect()
    cand["r2_patch_signed_spectrum"]["r2_overall"] = 0.37  # +0.02, below 0.05 threshold
    v = mod.gate_verdict(_BASELINE, cand)
    assert v["pass_spectrum"] is False
    assert v["GATE"] is False


def test_gate_fails_when_gdy_axis_drops_too_much():
    mod = _load_gate_module()
    cand = _candidate_perfect()
    cand["r2_GDY"]["r2_G"] = 0.92  # drop of 0.03 > 0.02 threshold
    v = mod.gate_verdict(_BASELINE, cand)
    assert v["pass_gdy"] is False
    assert v["GATE"] is False


def test_gate_passes_when_gdy_axis_drops_within_tolerance():
    mod = _load_gate_module()
    cand = _candidate_perfect()
    cand["r2_GDY"]["r2_G"] = 0.94  # drop of 0.01, within tolerance
    v = mod.gate_verdict(_BASELINE, cand)
    assert v["pass_gdy"] is True


def test_gate_fails_when_cl_drops_more_than_5_percent():
    mod = _load_gate_module()
    cand = _candidate_perfect()
    cand["r2_cl"]["r2_overall"] = 0.74  # drop of 0.05 from 0.79 = ~6.3% > 5%
    v = mod.gate_verdict(_BASELINE, cand)
    assert v["pass_cl"] is False
    assert v["GATE"] is False


def test_gate_passes_when_cl_drops_within_5_percent():
    mod = _load_gate_module()
    cand = _candidate_perfect()
    cand["r2_cl"]["r2_overall"] = 0.76  # drop of 0.03 from 0.79 = ~3.8% < 5%
    v = mod.gate_verdict(_BASELINE, cand)
    assert v["pass_cl"] is True


def test_gate_fails_when_pr_collapses():
    mod = _load_gate_module()
    cand = _candidate_perfect()
    cand["pr"] = 2.10  # 2.30 * 0.95 = 2.185; pr=2.10 is below
    v = mod.gate_verdict(_BASELINE, cand)
    assert v["pass_pr"] is False
    assert v["GATE"] is False


def test_gate_passes_when_pr_slightly_below_baseline():
    mod = _load_gate_module()
    cand = _candidate_perfect()
    cand["pr"] = 2.20  # > 2.30 * 0.95 = 2.185
    v = mod.gate_verdict(_BASELINE, cand)
    assert v["pass_pr"] is True


def test_gate_delta_signs_correct():
    """deltas should be positive for improvements, negative for regressions."""
    mod = _load_gate_module()
    v = mod.gate_verdict(_BASELINE, _candidate_perfect())
    assert v["dp_patch"] == pytest.approx(0.15)
    assert v["ds_spectrum"] == pytest.approx(0.20)
    assert v["dcl"] == pytest.approx(0.01)
    assert v["dpr"] == pytest.approx(0.10)
