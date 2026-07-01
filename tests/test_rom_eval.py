"""TDD for the pure pieces of the Session 31 Track D Q1 ROM eval.

These exercise the metric math and the latent/target alignment helpers with no
GPU, no encoder and no cache: the aggregated VRMSE formula (hand-checked on a
tiny array), the R^2 computation, per-frame window masking, the latent/target
row-alignment selector, and the physical-observable target computation shape and
sign. The full GPU encode + decoder-probe fit is exercised by the runner, not
here.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.represent import (
    aggregated_vrmse,
    frame_window_mask,
    r2_score_np,
    select_window,
)
from src.evaluation.rom_eval import physical_observables


# --------------------------------------------------------------------------- VRMSE
def test_aggregated_vrmse_hand_checked() -> None:
    """sqrt(sum(err^2) / sum((true - mean_true)^2)) aggregated over the whole set."""
    true = np.array([[0.0, 2.0], [4.0, 6.0]])
    # mean_true = 3; den = 9 + 1 + 1 + 9 = 20.
    pred = true + 1.0  # num = 4 * 1 = 4.
    got = aggregated_vrmse(pred, true)
    assert got == pytest.approx(np.sqrt(4.0 / 20.0))  # 0.4472...


def test_aggregated_vrmse_perfect_is_zero() -> None:
    true = np.linspace(-3.0, 3.0, 24).reshape(2, 3, 4)
    assert aggregated_vrmse(true.copy(), true) == pytest.approx(0.0)


def test_aggregated_vrmse_aggregates_before_dividing() -> None:
    """A near-uniform frame must not blow the ratio up: aggregate num/den globally.

    Two frames: one with real variance, one nearly uniform. A per-sample ratio
    would explode on the uniform frame; the aggregated ratio stays bounded and
    equals sqrt(total_sse / total_sst).
    """
    true = np.stack([np.array([0.0, 10.0, -10.0, 0.0]), np.full(4, 5.0) + 1e-9])
    pred = true + np.array([[1.0, -1.0, 1.0, -1.0], [0.5, 0.5, 0.5, 0.5]])
    mean_true = true.mean()
    exp = np.sqrt(((pred - true) ** 2).sum() / ((true - mean_true) ** 2).sum())
    assert aggregated_vrmse(pred, true) == pytest.approx(exp)


def test_aggregated_vrmse_constant_target_is_nan() -> None:
    true = np.full((3, 5), 2.0)
    assert np.isnan(aggregated_vrmse(true + 1.0, true))


def test_aggregated_vrmse_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        aggregated_vrmse(np.zeros((2, 3)), np.zeros((2, 4)))


# --------------------------------------------------------------------------- R^2
def test_r2_perfect_prediction() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2_score_np(y, y.copy()) == pytest.approx(1.0)


def test_r2_mean_prediction_is_zero() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2_score_np(y, np.full_like(y, y.mean())) == pytest.approx(0.0)


def test_r2_hand_checked_negative() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])  # mean 2.5, SST = 5.0
    pred = np.array([2.0, 2.0, 2.0, 2.0])  # SSE = 1 + 0 + 1 + 4 = 6
    assert r2_score_np(y, pred) == pytest.approx(1.0 - 6.0 / 5.0)


def test_r2_constant_target_is_nan() -> None:
    y = np.full(5, 3.0)
    assert np.isnan(r2_score_np(y, y + 1.0))


# --------------------------------------------------------------------------- windows
def test_frame_window_mask_impact_and_relaxation() -> None:
    entry = {"impact": [2, 5], "relaxation": [5, 8]}
    mask = frame_window_mask(entry, n_frames=10)
    assert mask.dtype == bool and mask.shape == (10,)
    assert mask.sum() == 6  # frames 2,3,4,5,6,7
    assert np.array_equal(np.where(mask)[0], np.array([2, 3, 4, 5, 6, 7]))


def test_frame_window_mask_clamps_out_of_range() -> None:
    entry = {"impact": [8, 15], "relaxation": [15, 20]}
    mask = frame_window_mask(entry, n_frames=10)
    assert np.array_equal(np.where(mask)[0], np.array([8, 9]))


def test_frame_window_mask_missing_part_is_ignored() -> None:
    entry = {"impact": [1, 3]}  # no relaxation key
    mask = frame_window_mask(entry, n_frames=6)
    assert mask.sum() == 2 and np.array_equal(np.where(mask)[0], np.array([1, 2]))


def test_frame_window_mask_none_entry_all_false() -> None:
    mask = frame_window_mask(None, n_frames=7)
    assert mask.shape == (7,) and not mask.any()


# --------------------------------------------------------------------------- alignment
def test_select_window_keeps_rows_aligned() -> None:
    n, d = 6, 4
    z = np.arange(n * d).reshape(n, d).astype(float)
    y = np.arange(n).astype(float)
    mask = np.array([True, False, True, True, False, False])
    z_sel, y_sel = select_window(z, y, mask)
    assert z_sel.shape[0] == y_sel.shape[0] == int(mask.sum()) == 3
    assert np.array_equal(z_sel, z[mask])
    assert np.array_equal(y_sel, y[mask])
    # the same rows are selected on both sides.
    assert np.array_equal(z_sel[:, 0].astype(int) // d, y_sel.astype(int))


def test_select_window_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        select_window(np.zeros((5, 2)), np.zeros(4), np.ones(5, dtype=bool))


# --------------------------------------------------------------------------- physics
def test_physical_observables_shapes_and_signs() -> None:
    """A positive blob inside the wake box gives circ_pos>0, circ_neg==0, ens>0."""
    T, H, W = 2, 192, 96
    field = np.zeros((T, H, W), dtype=np.float32)
    # Wake box is x in [0.5, 4.0], |y| <= 1.0 on the (-1.5,4.5) x (-1.5,1.5) grid.
    # Put a +5 patch near the domain centre, comfortably inside the wake box.
    field[:, 80:120, 40:56] = 5.0
    obs = physical_observables(field)
    assert set(obs) == {"wake_enstrophy", "circulation_pos", "circulation_neg"}
    for v in obs.values():
        assert v.shape == (T,)
    assert np.all(obs["wake_enstrophy"] > 0.0)
    assert np.all(obs["circulation_pos"] > 0.0)
    assert np.all(obs["circulation_neg"] == 0.0)


def test_physical_observables_negative_blob() -> None:
    T, H, W = 1, 192, 96
    field = np.zeros((T, H, W), dtype=np.float32)
    field[:, 80:120, 40:56] = -5.0
    obs = physical_observables(field)
    assert np.all(obs["circulation_neg"] < 0.0)
    assert np.all(obs["circulation_pos"] == 0.0)
