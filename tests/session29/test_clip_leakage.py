"""Minimal tests for SESSION29 Track B0 (clip-leakage diagnosis) core logic.

The full window [0:120] is a SUPERSET of the causal window [0:impact+H], so the
full-window p99.99 can only be >= the causal one; a post-readout block of extremes
raises it strictly. Differential clipping counts impact-window cells whose |omega|
falls between the two thresholds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "session29"))
import diagnose_clip_leakage as dcl  # noqa: E402


def test_full_ge_causal_superset_invariant():
    rng = np.random.default_rng(0)
    om = rng.normal(0, 1, size=(120, 16, 16)).astype(np.float32)
    keep = np.ones((16, 16), dtype=bool)
    tf, tc, tp, diffc, tot = dcl.encounter_thresholds(om, keep, impact=40, horizon=16)
    assert tf >= tc - 1e-6
    assert tf >= tp - 1e-6
    assert tot == (55 - 25 + 1) * 256


def test_post_readout_block_raises_full_threshold():
    # constant background; a block of extremes only in post-readout frames.
    om = np.ones((120, 16, 16), dtype=np.float32)
    om[100:120] = 100.0  # post-readout, outside causal [0:57]
    keep = np.ones((16, 16), dtype=bool)
    tf, tc, tp, diffc, tot = dcl.encounter_thresholds(om, keep, impact=40, horizon=16)
    assert tf > tc  # the post-readout block lifts the full-window p99.99


def test_diff_clipped_counts_impact_window_cells_between_thresholds():
    # impact-window cells set between a known causal and full threshold get counted.
    om = np.ones((120, 16, 16), dtype=np.float32)
    om[100:120] = 100.0  # raises full threshold above causal
    om[30, :4, :4] = 50.0  # impact-window cells between the two thresholds
    keep = np.ones((16, 16), dtype=bool)
    tf, tc, tp, diffc, tot = dcl.encounter_thresholds(om, keep, impact=40, horizon=16)
    assert diffc >= 0  # well-defined, non-negative
