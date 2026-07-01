"""Unit tests for the pure Q2 (Track D temporal forecast) rollout helpers.

Covers the framework-agnostic pieces of :mod:`src.evaluation.rollout`:
autoregressive rolling (shape + feedback), three-curve assembly + horizon
indexing (identity predictor -> model == persistence; truth predictor ->
model == floor, both in latent space), drift rel-L2, and window-restricted
horizon sample selection. The heavy GPU orchestration is not exercised here.
"""

from __future__ import annotations

import numpy as np

from src.evaluation.rollout import (
    assemble_three_curve,
    drift_rel_l2,
    roll_autoregressive,
    window_horizon_samples,
)


# --------------------------------------------------------------------------- roll
def test_roll_autoregressive_shape_and_length():
    # context of cl=2 scalars; predictor returns a fixed shape.
    ctx = [np.zeros((3, 4), dtype=np.float32), np.ones((3, 4), dtype=np.float32)]

    def predict(context):
        # mean of the two context frames
        return 0.5 * (context[-1] + context[-2])

    preds = roll_autoregressive(predict, ctx, steps=5)
    assert len(preds) == 5
    for p in preds:
        assert p.shape == (3, 4)


def test_roll_autoregressive_identity_feedback():
    # identity predictor (returns the last context frame): every prediction is
    # the last context frame, unchanged by feedback.
    last = np.array([2.0, 7.0, -1.0], dtype=np.float64)
    ctx = [np.array([0.0, 0.0, 0.0]), last.copy()]

    def predict(context):
        return context[-1].copy()

    preds = roll_autoregressive(predict, ctx, steps=4)
    for p in preds:
        assert np.allclose(p, last)


def test_roll_autoregressive_linear_progression():
    # predictor extrapolating a constant velocity: next = curr + (curr - prev).
    # Starting from prev=0, curr=1 (velocity 1) the rollout is 2, 3, 4, ...
    prev = np.array([0.0])
    curr = np.array([1.0])

    def predict(context):
        return context[-1] + (context[-1] - context[-2])

    preds = roll_autoregressive(predict, [prev, curr], steps=4)
    vals = np.array([p[0] for p in preds])
    assert np.allclose(vals, [2.0, 3.0, 4.0, 5.0])


# --------------------------------------------------------------------------- assembly
def _make_traj(L=20, d=3, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((L, d)).astype(np.float64)


def test_assemble_three_curve_truth_predictor_matches_floor():
    z_true = _make_traj()
    horizons = [1, 2, 3, 4]
    anchors = np.array([2, 5, 9, 12])
    # rolled == the true future latent (perfect predictor).
    rolled = np.stack([np.stack([z_true[a + h] for h in horizons]) for a in anchors])  # [A, H, d]
    model, floor, persist, target = assemble_three_curve(rolled, z_true, anchors, horizons)
    assert np.array_equal(target, anchors[:, None] + np.array(horizons)[None, :])
    assert np.allclose(model, floor)


def test_assemble_three_curve_identity_predictor_matches_persistence():
    z_true = _make_traj(seed=1)
    horizons = [1, 2, 3]
    anchors = np.array([3, 6, 10])
    # rolled == the anchor latent held constant (identity/persistence predictor).
    rolled = np.stack([np.stack([z_true[a] for _ in horizons]) for a in anchors])  # [A, H, d]
    model, floor, persist, _ = assemble_three_curve(rolled, z_true, anchors, horizons)
    assert np.allclose(model, persist)
    # persistence holds the anchor value across every horizon.
    for ai, a in enumerate(anchors):
        for hi in range(len(horizons)):
            assert np.allclose(persist[ai, hi], z_true[a])


def test_assemble_three_curve_indexing():
    z_true = _make_traj(seed=2)
    horizons = [1, 5]
    anchors = np.array([4, 8])
    rolled = np.zeros((2, 2, z_true.shape[1]))
    _, floor, persist, target = assemble_three_curve(rolled, z_true, anchors, horizons)
    # floor at (anchor a, horizon h) is z_true[a + h].
    assert np.allclose(floor[0, 0], z_true[5])
    assert np.allclose(floor[0, 1], z_true[9])
    assert np.allclose(floor[1, 1], z_true[13])
    assert target[1, 0] == 9


# --------------------------------------------------------------------------- drift
def test_drift_rel_l2_zero_when_equal():
    a = _make_traj(seed=3)
    assert drift_rel_l2(a, a) == 0.0


def test_drift_rel_l2_known_value():
    true = np.array([[3.0, 4.0]])  # ||true|| = 5
    rolled = np.array([[0.0, 0.0]])  # diff = -true, ||diff|| = 5
    assert np.isclose(drift_rel_l2(rolled, true), 1.0)


def test_drift_rel_l2_aggregated_over_samples():
    true = np.array([[3.0, 4.0], [0.0, 0.0]])
    rolled = np.array([[3.0, 4.0], [1.0, 0.0]])
    # num = 0 + 1 = 1; den = 25 + 0 = 25; sqrt(1/25) = 0.2
    assert np.isclose(drift_rel_l2(rolled, true), 0.2)


# --------------------------------------------------------------------------- windows
def test_window_horizon_samples_selects_in_window_targets():
    L = 12
    window_mask = np.zeros(L, dtype=bool)
    window_mask[6:10] = True  # frames 6,7,8,9 in-window
    anchors = np.array([3, 4, 5, 6, 7, 8])
    horizons = [1, 2, 3]
    sel = window_horizon_samples(window_mask, anchors, horizons, context_length=2, n_frames=L)
    # horizon 1: target = anchor+1 in {6,7,8,9} -> anchors 5,6,7,8 (indices 2,3,4,5)
    assert set(anchors[sel[1]]) == {5, 6, 7, 8}
    # horizon 3: target = anchor+3 in {6,7,8,9} -> anchors 3,4,5,6 (indices 0,1,2,3)
    assert set(anchors[sel[3]]) == {3, 4, 5, 6}


def test_window_horizon_samples_drops_out_of_range_targets():
    L = 10
    window_mask = np.ones(L, dtype=bool)
    anchors = np.array([7, 8])
    horizons = [1, 3]
    sel = window_horizon_samples(window_mask, anchors, horizons, context_length=2, n_frames=L)
    # horizon 3: target = 10, 11 both out of range -> empty.
    assert sel[3].size == 0
    # horizon 1: target 8, 9 valid.
    assert set(anchors[sel[1]]) == {7, 8}


def test_window_horizon_samples_respects_context_validity():
    L = 12
    window_mask = np.ones(L, dtype=bool)
    anchors = np.array([0, 1, 2])  # anchor 0 has no room for cl=2 context
    horizons = [1]
    sel = window_horizon_samples(window_mask, anchors, horizons, context_length=2, n_frames=L)
    # anchor 0 -> needs frame -1 for context, dropped; anchors 1,2 kept.
    assert set(anchors[sel[1]]) == {1, 2}


# --------------------------------------------------------------------------- gatherers
def test_gather_spatial_reads_the_horizon_step_not_the_list_position():
    """The h=H bucket must hold the step-H rolled latent regardless of the horizons list.

    Regression for the ``rolled_np[idx, hi]`` bug: indexing the rolled tensor by the
    enumerate position ``hi`` only equals the step ``h-1`` when ``horizons == [1,2,..]``.
    Passing a single horizon ``[3]`` (as the closure-CI path did) made hi=0 read step 1,
    scoring a barely-rolled latent against the h=3 target. Both ``[1,2,3]`` and ``[3]``
    use H=max=3 (identical anchors/rollout), so the h=3 bucket must be identical.
    """
    import torch

    from src.evaluation.rollout import Encounter, _gather_spatial
    from src.models.resunet_predictor import ResUNetPredictor

    L, d, h, w = 12, 4, 8, 8
    rng = np.random.default_rng(0)
    z_spatial = rng.standard_normal((L, d, h, w)).astype(np.float32)
    window_mask = np.zeros(L, dtype=bool)
    window_mask[4:9] = True
    enc = Encounter(
        key="case/00",
        z_spatial=z_spatial,
        z_gap=z_spatial.mean(axis=(2, 3)),
        window_mask=window_mask,
        global_rows=np.arange(L),
        frames=np.arange(L),
    )
    torch.manual_seed(0)
    predictor = ResUNetPredictor(latent_dim=d, context_length=2).eval()
    device = torch.device("cpu")

    b_full = _gather_spatial(predictor, [enc], [1, 2, 3], 2, device)
    b_single = _gather_spatial(predictor, [enc], [3], 2, device)

    rolled_full = np.concatenate(b_full[3].rolled_gap)
    rolled_single = np.concatenate(b_single[3].rolled_gap)
    assert rolled_full.shape == rolled_single.shape
    # The h=3 rolled latent is deterministic; it must not depend on the horizons list.
    assert np.allclose(rolled_full, rolled_single)
