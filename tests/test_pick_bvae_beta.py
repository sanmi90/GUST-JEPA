"""Tests for the T8 beta-VAE L-curve knee picker (scripts/session28/pick_bvae_beta.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "session28"))

from pick_bvae_beta import knee_index  # noqa: E402


class TestKneeIndex:
    def test_obvious_corner_is_picked(self) -> None:
        # L-shaped curve ordered by ascending beta: KL falls fast then
        # flattens, recon flat then degrades; corner at index 2.
        rate = np.array([80.0, 40.0, 15.0, 8.0, 6.0])
        dist = np.array([0.100, 0.101, 0.105, 0.140, 0.200])
        assert knee_index(rate, dist) == 2

    def test_ties_break_toward_larger_beta(self) -> None:
        # Symmetric V around the chord: indices 1 and 3 equidistant.
        rate = np.array([1.0, 0.5, 0.5, 0.5, 0.0])
        dist = np.array([0.0, 0.0, 0.5, 1.0, 1.0])
        d = knee_index(rate, dist)
        assert d >= 2  # never the earlier of two equally good points

    def test_degenerate_flat_curve_returns_last(self) -> None:
        rate = np.array([1.0, 1.0, 1.0])
        dist = np.array([2.0, 2.0, 2.0])
        # all points coincide after normalisation; tie-break gives the
        # largest beta (last index)
        assert knee_index(rate, dist) == 2

    def test_rejects_short_sweeps(self) -> None:
        with pytest.raises(ValueError):
            knee_index(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
