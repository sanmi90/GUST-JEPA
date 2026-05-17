"""Tests for ``src.training.auto_fallback.AutoFallbackController``."""

from __future__ import annotations

from src.training.auto_fallback import AutoFallbackController


def test_fallback_does_not_fire_before_threshold() -> None:
    ctrl = AutoFallbackController(d=32)
    fired = ctrl.step(iteration=1_000, pr=0.0, probe_r2=0.0)
    assert fired is False
    assert ctrl.fired is False
    assert ctrl.fired_at_iter is None


def test_fallback_does_not_fire_when_pr_healthy() -> None:
    ctrl = AutoFallbackController(d=32)
    fired = ctrl.step(iteration=20_000, pr=0.5 * 32, probe_r2=0.0)
    assert fired is False
    assert ctrl.fired is False


def test_fallback_does_not_fire_when_r2_healthy() -> None:
    ctrl = AutoFallbackController(d=32)
    fired = ctrl.step(iteration=20_000, pr=0.1 * 32, probe_r2=0.9)
    assert fired is False
    assert ctrl.fired is False


def test_fallback_fires_on_both_conditions_met() -> None:
    ctrl = AutoFallbackController(d=32)
    fired = ctrl.step(iteration=20_000, pr=0.1 * 32, probe_r2=0.3)
    assert fired is True
    assert ctrl.fired is True
    assert ctrl.fired_at_iter == 20_000


def test_fallback_is_idempotent_once_fired() -> None:
    ctrl = AutoFallbackController(d=32)
    first = ctrl.step(iteration=20_000, pr=0.1 * 32, probe_r2=0.3)
    second = ctrl.step(iteration=21_000, pr=0.1 * 32, probe_r2=0.3)
    third = ctrl.step(iteration=22_000, pr=0.05 * 32, probe_r2=0.1)
    assert first is True
    assert second is False
    assert third is False
    assert ctrl.fired is True
    assert ctrl.fired_at_iter == 20_000


def test_fallback_threshold_at_exactly_20k() -> None:
    """Iteration boundary: 19_999 does not fire, 20_000 does."""
    ctrl_a = AutoFallbackController(d=32)
    assert ctrl_a.step(iteration=19_999, pr=0.1 * 32, probe_r2=0.3) is False
    assert ctrl_a.fired is False

    ctrl_b = AutoFallbackController(d=32)
    assert ctrl_b.step(iteration=20_000, pr=0.1 * 32, probe_r2=0.3) is True
    assert ctrl_b.fired is True


def test_fallback_records_history() -> None:
    """Every step appends ``(iter, pr, probe_r2)`` to ``history``."""
    ctrl = AutoFallbackController(d=32)
    ctrl.step(iteration=1000, pr=20.0, probe_r2=0.9)
    ctrl.step(iteration=2000, pr=10.0, probe_r2=0.5)
    assert ctrl.history == [(1000, 20.0, 0.9), (2000, 10.0, 0.5)]
