"""Minimal tests for SESSION29 Track G (physical floors) pure helpers.

Covers the verdict logic (STRONG iff the predictive latent strictly clears every
floor; WEAK if any floor matches or beats it) on synthetic R^2 dicts, plus the
empty-floor guard. No heavy real latents are touched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "session29"))
import physics_floors as pf  # noqa: E402


def test_strong_when_latent_clears_every_floor():
    v = pf.decide_verdict(0.80, {"gdy": 0.40, "gdy_history": 0.55, "persistence": 0.62})
    assert v["branch"] == "STRONG"
    assert v["best_floor"] == "persistence"
    assert v["best_floor_r2"] == pytest.approx(0.62)
    assert v["latent_advantage_over_best_floor"] == pytest.approx(0.18)
    assert set(v["floors_cleared"]) == {"gdy", "gdy_history", "persistence"}
    assert v["floors_not_cleared"] == []


def test_weak_when_a_floor_beats_the_latent():
    v = pf.decide_verdict(0.50, {"gdy": 0.30, "gdy_history": 0.65, "persistence": 0.20})
    assert v["branch"] == "WEAK"
    assert v["best_floor"] == "gdy_history"
    assert "gdy_history" in v["floors_not_cleared"]
    assert set(v["floors_cleared"]) == {"gdy", "persistence"}


def test_weak_on_exact_tie():
    # A floor that exactly matches the latent (latent_r2 <= floor_r2) is WEAK.
    v = pf.decide_verdict(0.42, {"persistence": 0.42})
    assert v["branch"] == "WEAK"
    assert v["floors_not_cleared"] == ["persistence"]
    assert v["latent_advantage_over_best_floor"] == pytest.approx(0.0)


def test_empty_floors_raises():
    with pytest.raises(ValueError):
        pf.decide_verdict(0.5, {})
