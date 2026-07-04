"""Tests for the Track C lift-tracking metrics (Session 34)."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.lift_metrics import lift_phase_lag, peak_region_r2, peak_window_mask


class TestPeakWindowMask:
    def test_window_centered_on_abs_peak(self):
        cl = np.zeros(100)
        cl[60] = -5.0  # negative peak; |C_L| governs
        m = peak_window_mask(cl, half_width=8)
        assert m.sum() == 17
        assert m[52] and m[68] and not m[51] and not m[69]

    def test_window_clipped_at_bounds(self):
        cl = np.zeros(20)
        cl[1] = 3.0
        m = peak_window_mask(cl, half_width=8)
        assert m[0] and m[9] and not m[10]

    def test_rejects_2d(self):
        with pytest.raises(ValueError):
            peak_window_mask(np.zeros((4, 4)))


class TestPeakRegionR2:
    def test_perfect_prediction_gives_one(self):
        rng = np.random.default_rng(0)
        traces = [rng.normal(size=120) for _ in range(5)]
        assert peak_region_r2(traces, [t.copy() for t in traces]) == pytest.approx(1.0)

    def test_mean_prediction_gives_zero(self):
        rng = np.random.default_rng(1)
        traces = [rng.normal(size=120) + 2.0 for _ in range(6)]
        # Predicting the pooled peak-window mean gives R^2 = 0 by construction.
        masks = [peak_window_mask(t) for t in traces]
        pooled_mean = np.concatenate([t[m] for t, m in zip(traces, masks)]).mean()
        preds = [np.full_like(t, pooled_mean) for t in traces]
        assert peak_region_r2(traces, preds) == pytest.approx(0.0, abs=1e-9)

    def test_bad_prediction_negative(self):
        rng = np.random.default_rng(2)
        traces = [rng.normal(size=120) for _ in range(4)]
        preds = [t + 10.0 for t in traces]
        assert peak_region_r2(traces, preds) < 0.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            peak_region_r2([np.zeros(10)], [np.zeros(11)])


class TestLiftPhaseLag:
    def test_zero_lag_for_identical_signals(self):
        t = np.linspace(0, 6 * np.pi, 240)
        s = np.sin(t) + 0.3 * np.sin(2.7 * t)
        lag, corr = lift_phase_lag(s, s)
        assert lag == pytest.approx(0.0, abs=1e-6)
        assert corr == pytest.approx(1.0, rel=1e-6)

    @pytest.mark.parametrize("shift", [3, 7, -5])
    def test_integer_shift_recovered(self, shift):
        rng = np.random.default_rng(3)
        base = np.cumsum(rng.normal(size=300))  # smooth-ish random walk
        base -= base.mean()
        if shift >= 0:
            pred = np.concatenate([np.full(shift, base[0]), base[:-shift or None]])
        else:
            pred = np.concatenate([base[-shift:], np.full(-shift, base[-1])])
        # pred is base delayed by `shift` frames -> pred trails truth by shift.
        lag, corr = lift_phase_lag(pred, base, dt=0.05, max_lag=20)
        assert lag == pytest.approx(shift * 0.05, abs=0.05 * 0.6)
        assert corr > 0.9

    def test_subframe_refinement_between_integers(self):
        # A sinusoid shifted by 2.5 frames: the parabolic refinement should
        # land strictly between the integer lags.
        n = np.arange(400, dtype=np.float64)
        period = 40.0
        true = np.sin(2 * np.pi * n / period)
        pred = np.sin(2 * np.pi * (n - 2.5) / period)
        lag, _ = lift_phase_lag(pred, true, dt=0.05, max_lag=20)
        assert lag == pytest.approx(2.5 * 0.05, abs=0.01)

    def test_positive_lag_means_pred_trails(self):
        n = np.arange(300, dtype=np.float64)
        true = np.sin(2 * np.pi * n / 60.0)
        pred_trailing = np.sin(2 * np.pi * (n - 4.0) / 60.0)
        lag, _ = lift_phase_lag(pred_trailing, true, dt=0.05, max_lag=20)
        assert lag > 0
