"""Minimal tests for SESSION29 Track B0.5 (frozen preprocessing sensitivity) helpers.

Covers the load-bearing pure logic, no heavy real latents/cache:
  * the wake-enstrophy reduction + amplitude clip lower enstrophy monotonically on
    a synthetic field with a spike, and clipping a sub-spike field is a no-op;
  * a tighter (lower) global/per-encounter threshold clips a synthetic spike that a
    looser threshold leaves alone (global vs per-encounter thresholds differ as
    expected on a designed field);
  * the drop mask zeros the intended cells;
  * the STRONG/WEAK advantage verdict (sign + magnitude preservation) on synthetic
    family-R^2 dicts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "session29"))
import preprocess_sensitivity_frozen as ps  # noqa: E402


def test_clip_lowers_enstrophy_monotonically_on_spike():
    box = ps.wake_box_mask()
    # Put a single large spike at a cell guaranteed inside the wake box.
    in_x = np.where(box.any(axis=1))[0]
    in_y = np.where(box.any(axis=0))[0]
    ix, iy = int(in_x[len(in_x) // 2]), int(in_y[len(in_y) // 2])
    field = np.zeros((ps.NX, ps.NY), dtype=np.float32)
    field[ix, iy] = 1000.0
    field[ix + 1, iy] = 3.0  # a small background cell, below any clip
    e_none = ps.wake_enstrophy_frame(ps.clip_amplitude(field, float("inf")), box)
    e_loose = ps.wake_enstrophy_frame(ps.clip_amplitude(field, 100.0), box)
    e_tight = ps.wake_enstrophy_frame(ps.clip_amplitude(field, 10.0), box)
    assert e_none > e_loose > e_tight  # tighter clip strictly lowers squared sum
    # Clipping above the max is a no-op.
    e_noop = ps.wake_enstrophy_frame(ps.clip_amplitude(field, 5000.0), box)
    assert e_noop == pytest.approx(e_none)


def test_global_vs_per_encounter_thresholds_differ_as_expected():
    # Two synthetic encounters: enc A carries a heavy-tail block, enc B is mild.
    # A's per-encounter p99.99 sits ON the tail block (its top 0.01% lands inside
    # the block); pooling A+B doubles the population so the pooled top 0.01% no
    # longer reaches the block, dragging the global p99.99 BELOW the block value.
    n = 100_000
    base = np.linspace(0.0, 10.0, n)  # smooth bulk, max 10
    a = np.concatenate([base, np.full(20, 500.0)])  # A: 0.02% of A is the tail block
    b = base.copy()  # B: mild, no extremes
    thr_a = float(np.percentile(a, ps.CLIP_PERCENTILE))
    thr_global = float(np.percentile(np.concatenate([a, b]), ps.CLIP_PERCENTILE))
    assert thr_a == pytest.approx(500.0)  # A's top 0.01% sits in the 500-block
    assert thr_global < 500.0  # pooling dilutes the tail below the block
    assert thr_a > thr_global
    spike = np.array([[300.0]], dtype=np.float32)
    # Global clips the spike; the per-encounter-A threshold (higher) leaves it.
    assert ps.clip_amplitude(spike, thr_global)[0, 0] < 300.0
    assert ps.clip_amplitude(spike, thr_a)[0, 0] == pytest.approx(300.0)


def test_drop_mask_zeros_selected_cells():
    field = np.full((ps.NX, ps.NY), 7.0, dtype=np.float32)
    drop = np.zeros((ps.NX, ps.NY), dtype=bool)
    drop[5, 5] = True
    drop[10, 20] = True
    out = ps.apply_drop_mask(field, drop)
    assert out[5, 5] == 0.0 and out[10, 20] == 0.0
    assert out[0, 0] == 7.0  # untouched cell preserved
    assert field[5, 5] == 7.0  # input not mutated


def test_advantage_strong_when_sign_and_magnitude_preserved():
    r2 = {
        "none": {"jepa_tf_noc": 0.70, "fukami": 0.40, "pod": 0.30},
        "per_encounter": {"jepa_tf_noc": 0.68, "fukami": 0.42, "pod": 0.31},
        "training_global": {"jepa_tf_noc": 0.69, "fukami": 0.41, "pod": 0.30},
    }
    v = ps.advantage_record(r2)
    assert v["branch"] == "STRONG"
    assert v["sign_preserved"] and v["magnitude_preserved"]
    assert not v["b1_retrain_needed"]
    # reference advantage is per_encounter: 0.68 - max(0.42,0.31) = 0.26
    assert v["reference_advantage"] == pytest.approx(0.26)
    assert v["per_treatment"]["per_encounter"]["best_baseline"] == "fukami"


def test_advantage_weak_on_sign_flip():
    r2 = {
        "none": {"jepa_tf_noc": 0.70, "fukami": 0.40, "pod": 0.30},
        "per_encounter": {"jepa_tf_noc": 0.68, "fukami": 0.42, "pod": 0.31},
        "training_global": {"jepa_tf_noc": 0.20, "fukami": 0.50, "pod": 0.30},  # baseline wins
    }
    v = ps.advantage_record(r2)
    assert v["branch"] == "WEAK"
    assert not v["sign_preserved"]
    assert v["b1_retrain_needed"]


def test_advantage_weak_on_magnitude_shift_beyond_tol():
    # Sign stays positive but training_global advantage collapses far below the ref.
    r2 = {
        "none": {"jepa_tf_noc": 0.70, "fukami": 0.40, "pod": 0.30},
        "per_encounter": {"jepa_tf_noc": 0.68, "fukami": 0.42, "pod": 0.31},
        "training_global": {
            "jepa_tf_noc": 0.46,
            "fukami": 0.42,
            "pod": 0.30,
        },  # adv 0.04 vs ref 0.26
    }
    v = ps.advantage_record(r2)
    assert v["branch"] == "WEAK"
    assert v["sign_preserved"]  # still positive
    assert not v["magnitude_preserved"]  # shift > tol
    assert v["b1_retrain_needed"]
