"""Tests for the Track T3 Grassberger-Procaccia correlation-dimension estimator.

CPU-only, seeded (project convention: unit tests stay CPU-friendly). The
estimator lives in scripts/session33/track_t3_effective_dimension.py with
numpy-only module-level imports, so importing it here is cheap.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.session33.track_t3_effective_dimension import (
    correlation_integral,
    gp_dimension,
)


def _distinct_ids(n):
    """Every point its own 'encounter' -> the Theiler window excludes nothing."""
    return np.arange(n), np.zeros(n)


def test_circle_dimension_near_one():
    rng = np.random.default_rng(0)
    n = 1500
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    pts = np.stack([np.cos(theta), np.sin(theta)], axis=1).astype(np.float32)
    pts += rng.normal(scale=1e-3, size=pts.shape).astype(np.float32)
    enc_ids, times = _distinct_ids(n)
    d_eff, diag = gp_dimension(pts, enc_ids, times, seed=0)
    assert diag["n_allowed_pairs"] == n * (n - 1) // 2
    assert 0.8 <= d_eff <= 1.3, f"circle d_eff={d_eff}"


def test_gaussian_ball_dimension_near_three():
    rng = np.random.default_rng(1)
    n = 1500
    pts = rng.normal(size=(n, 3)).astype(np.float32)
    enc_ids, times = _distinct_ids(n)
    d_eff, _ = gp_dimension(pts, enc_ids, times, seed=1)
    assert 2.4 <= d_eff <= 3.6, f"ball d_eff={d_eff}"


def test_embedded_plane_ignores_ambient_dimension():
    """A 2D plane embedded in 32-d ambient space must read ~2, not ~32."""
    rng = np.random.default_rng(2)
    n = 1500
    uv = rng.uniform(-1.0, 1.0, size=(n, 2)).astype(np.float32)
    basis = np.linalg.qr(rng.normal(size=(32, 2)))[0].astype(np.float32)  # (32, 2)
    pts = uv @ basis.T
    enc_ids, times = _distinct_ids(n)
    d_eff, _ = gp_dimension(pts, enc_ids, times, seed=2)
    assert 1.6 <= d_eff <= 2.5, f"plane d_eff={d_eff}"


def test_theiler_window_excludes_same_encounter_close_pairs():
    # 2 encounters x 4 frames each; theiler_dt=2 excludes |dt|<2 within encounter.
    pts = np.arange(8, dtype=np.float32)[:, None]
    enc_ids = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    times = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    r_grid = np.array([100.0])  # every allowed pair counts
    C, n_allowed = correlation_integral(pts, enc_ids, times, r_grid, theiler_dt=2)
    # total pairs 28; same-encounter |dt|<2 pairs: 3 per encounter -> 6 excluded.
    assert n_allowed == 22
    assert C[0] == pytest.approx(1.0)


def test_theiler_window_off_with_distinct_ids():
    pts = np.arange(6, dtype=np.float32)[:, None]
    enc_ids, times = _distinct_ids(6)
    C, n_allowed = correlation_integral(pts, enc_ids, times, np.array([100.0]), theiler_dt=30)
    assert n_allowed == 15
