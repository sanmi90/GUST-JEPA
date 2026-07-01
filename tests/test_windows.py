"""Tests for the impact-window definition (SESSION 31 Track 0.B).

The window math is the load-bearing logic that gates every temporal (Q2/Q3)
metric, so it is pinned here before the cache-reading CLI is built on top of it.
All functions under test are pure (numpy in, plain values out).
"""

import numpy as np

from src.evaluation.windows import (
    impact_frame,
    impact_frame_anchored,
    peak_clarity,
    build_windows,
    is_well_separated,
    window_masks,
)


def _logistic_step(n: int, center: float, scale: float = 2.0) -> np.ndarray:
    """A smooth C_L-like step whose |dC_L/dt| peaks exactly at ``center``."""
    i = np.arange(n, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-(i - center) / scale))


def test_impact_frame_is_argmax_of_abs_dcl_dt():
    # A logistic step centred at frame 40: its derivative peaks at 40.
    cl = _logistic_step(120, center=40.0)
    assert impact_frame(cl) == 40


def test_impact_frame_picks_the_steepest_of_two_unequal_ramps():
    # Gentle ramp at 25, sharp ramp at 70; the sharp one wins.
    cl = _logistic_step(120, center=25.0, scale=6.0) + _logistic_step(120, center=70.0, scale=1.0)
    assert impact_frame(cl) == 70


def test_impact_frame_anchored_restricts_to_physics_window():
    # Global steepest slope is a shedding event at frame 70 (outside [25, 55]);
    # the in-window lift response peaks at 42. The anchored trigger must ignore 70.
    cl = _logistic_step(120, center=70.0, scale=1.0) + 0.3 * _logistic_step(
        120, center=42.0, scale=2.0
    )
    assert impact_frame(cl) == 70  # naive global trigger wanders out
    assert impact_frame_anchored(cl, 25, 55) == 42  # anchored stays in the window


def test_impact_frame_anchored_degenerate_single_frame_window():
    cl = _logistic_step(120, center=40.0)
    assert impact_frame_anchored(cl, 33, 33) == 33  # lo == hi -> that frame


def test_build_windows_half_open_bounds_interior():
    w = build_windows(t_impact=40, n_frames=120, w_in=8, w_imp=16, w_relax=48)
    assert w["lead_in"] == (32, 40)
    assert w["impact"] == (40, 56)
    assert w["relaxation"] == (56, 88)
    assert w["t_impact"] == 40
    assert w["n_frames"] == 120


def test_build_windows_clamps_at_start():
    w = build_windows(t_impact=4, n_frames=120, w_in=8, w_imp=16, w_relax=48)
    assert w["lead_in"] == (0, 4)  # clamped from -4
    assert w["impact"] == (4, 20)
    assert w["relaxation"] == (20, 52)


def test_build_windows_clamps_at_end():
    w = build_windows(t_impact=100, n_frames=120, w_in=8, w_imp=16, w_relax=48)
    assert w["lead_in"] == (92, 100)
    assert w["impact"] == (100, 116)
    assert w["relaxation"] == (116, 120)  # clamped from 148


def test_is_well_separated_true_for_interior_impact():
    assert is_well_separated(t_impact=40, n_frames=120, w_in=8, w_relax=48) is True


def test_is_well_separated_false_when_lead_in_underflows():
    assert is_well_separated(t_impact=4, n_frames=120, w_in=8, w_relax=48) is False


def test_is_well_separated_false_when_relaxation_overflows():
    assert is_well_separated(t_impact=100, n_frames=120, w_in=8, w_relax=48) is False


def test_peak_clarity_higher_for_unimodal_than_bimodal():
    unimodal = _logistic_step(120, center=40.0, scale=2.0)
    bimodal = _logistic_step(120, center=30.0, scale=2.0) + _logistic_step(
        120, center=75.0, scale=2.0
    )
    c_uni = peak_clarity(unimodal)
    c_bi = peak_clarity(bimodal)
    assert c_uni > c_bi
    assert c_bi < 2.0  # two comparable peaks -> ratio near 1
    assert c_uni > 3.0  # one dominant peak -> ratio large


def test_window_masks_are_disjoint_and_match_bounds():
    w = build_windows(t_impact=40, n_frames=120, w_in=8, w_imp=16, w_relax=48)
    masks = window_masks(w, n_frames=120)
    assert masks["lead_in"].dtype == bool
    assert masks["lead_in"].sum() == 8
    assert masks["impact"].sum() == 16
    assert masks["relaxation"].sum() == 32
    # frame 35 is in lead_in only; frame 48 in impact only; frame 70 in relaxation only
    assert masks["lead_in"][35] and not masks["impact"][35] and not masks["relaxation"][35]
    assert masks["impact"][48] and not masks["lead_in"][48]
    assert masks["relaxation"][70] and not masks["impact"][70]
    # pairwise disjoint
    assert not (masks["lead_in"] & masks["impact"]).any()
    assert not (masks["impact"] & masks["relaxation"]).any()
