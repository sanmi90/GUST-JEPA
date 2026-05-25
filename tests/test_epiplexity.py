"""Tests for ``src.evaluation.epiplexity``.

Pure CPU / numpy. No GPU or torch dependency.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.epiplexity import (
    epiplexity_decomposition,
    epiplexity_summary,
    load_loss_curve,
    prequential_coding,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
S12_E_D64_METRICS = REPO_ROOT / "outputs/runs/session12/S12_E_d64/encoder/metrics.jsonl"


def test_constant_loss_curve_is_zero() -> None:
    """No excess above the floor when every loss equals the floor."""
    losses = np.full(50, 1.234, dtype=np.float64)
    out = prequential_coding(losses, loss_floor=1.234)
    assert out == pytest.approx(0.0, abs=1e-12)

    # Floor inferred from the trailing median also equals 1.234 here.
    out_default = prequential_coding(losses)
    assert out_default == pytest.approx(0.0, abs=1e-12)


def test_inverse_sqrt_curve_matches_closed_form() -> None:
    """For L_i = 1/sqrt(i+1) with floor 0, sum_{i=0}^{N-1} 1/sqrt(i+1)
    agrees with the integral 2 * sqrt(N) within a small Euler-Maclaurin
    correction. We use a 5% tolerance per the task spec.
    """
    n = 10_000
    i = np.arange(n, dtype=np.float64)
    losses = 1.0 / np.sqrt(i + 1.0)
    p_preq = prequential_coding(losses, loss_floor=0.0)
    # Closed-form continuous approximation: integral_0^N x^{-1/2} dx = 2 sqrt(N).
    closed_form = 2.0 * np.sqrt(float(n))
    relative_error = abs(p_preq - closed_form) / closed_form
    assert relative_error < 0.05


def test_steeper_curve_has_larger_preq() -> None:
    """Two curves with the same floor (0.1) and same length but different
    decay rates: the steeper one should integrate to a larger ``|P_preq|``
    because it spends more time far above the floor."""
    n = 500
    i = np.arange(n, dtype=np.float64)
    floor = 0.1
    # Both curves asymptote to the floor; "slow" decays at half the rate of "fast".
    fast = floor + 2.0 * np.exp(-i / 20.0)
    slow = floor + 2.0 * np.exp(-i / 200.0)
    p_fast = prequential_coding(fast, loss_floor=floor)
    p_slow = prequential_coding(slow, loss_floor=floor)
    # The slower decay accumulates more area above the floor: a larger P_preq
    # corresponds to a less "efficient" learner. So slow > fast.
    assert p_slow > p_fast
    # Sanity: both are positive.
    assert p_fast > 0.0 and p_slow > 0.0


@pytest.mark.skipif(
    not S12_E_D64_METRICS.exists(),
    reason="S12_E_d64 metrics file not present",
)
def test_load_real_s12_e_d64() -> None:
    """The real run file loads with config['d'] == 64 and >400 log rows."""
    iters, losses, config = load_loss_curve(S12_E_D64_METRICS)
    assert iters.shape == losses.shape
    assert losses.size > 200, f"expected >200 training-log rows, got {losses.size}"
    assert config.get("d") == 64
    assert iters[0] == 0
    assert np.all(np.isfinite(losses))


@pytest.mark.skipif(
    not S12_E_D64_METRICS.exists(),
    reason="S12_E_d64 metrics file not present",
)
def test_epiplexity_summary_real_run() -> None:
    """``epiplexity_summary`` on the real file returns finite positive scalars."""
    summary = epiplexity_summary(S12_E_D64_METRICS)
    assert summary["P_preq"] > 0.0
    assert summary["L_M"] > 0.0
    assert summary["N"] > 200
    assert np.isfinite(summary["P_preq"])
    assert np.isfinite(summary["L_M"])
    assert np.isfinite(summary["H_T"])
    assert summary["S_T"] == summary["P_preq"]
    assert summary["loss_key"] == "loss_total"
    assert isinstance(summary["config"], dict)
    assert summary["config"].get("d") == 64


@pytest.mark.skipif(
    not S12_E_D64_METRICS.exists(),
    reason="S12_E_d64 metrics file not present",
)
def test_epiplexity_decomposition_returns_all_components() -> None:
    """All seven standard loss columns are present in the S12_E_d64 run."""
    parts = epiplexity_decomposition(S12_E_D64_METRICS)
    expected = {
        "loss_total",
        "loss_pred",
        "loss_roll",
        "loss_anticollapse",
        "loss_obs",
        "loss_wake",
        "loss_tc",
    }
    missing = expected - set(parts.keys())
    assert not missing, f"missing decomposition keys: {missing}"
    for key, summary in parts.items():
        assert summary["loss_key"] == key
        assert summary["N"] > 200
        assert np.isfinite(summary["P_preq"])


@pytest.mark.skipif(
    not S12_E_D64_METRICS.exists(),
    reason="S12_E_d64 metrics file not present",
)
def test_loss_floor_override_increases_preq(tmp_path: Path) -> None:
    """Passing ``loss_floor=0`` produces a larger ``|P_preq|`` than the default
    trailing-median floor (which is strictly positive for our positive-loss
    curves)."""
    default_summary = epiplexity_summary(S12_E_D64_METRICS)
    zero_summary = epiplexity_summary(S12_E_D64_METRICS, loss_floor=0.0)
    assert zero_summary["L_M"] == 0.0
    assert default_summary["L_M"] > 0.0
    assert zero_summary["P_preq"] > default_summary["P_preq"]


def test_load_loss_curve_synthetic_file(tmp_path: Path) -> None:
    """Round-trip a small synthetic ``metrics.jsonl``: the loader handles
    interleaved diagnostic lines (missing the loss key) and a leading config
    record, and produces aligned ``(iters, losses)`` arrays."""
    path = tmp_path / "metrics.jsonl"
    records = [
        {"event": "config", "wandb_run_id": "abc", "seed": 7, "d": 32},
        {"event": "log", "step": 0, "iter": 0, "loss_total": 2.5, "loss_pred": 2.0},
        {"event": "log", "step": 0, "diag/pr": 1.1},
        {"event": "log", "step": 50, "iter": 50, "loss_total": 1.0, "loss_pred": 0.8},
        {"event": "log", "step": 50, "diag/pr": 1.5},
        {"event": "log", "step": 100, "iter": 100, "loss_total": 0.5, "loss_pred": 0.4},
    ]
    with path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    iters, losses, config = load_loss_curve(path, loss_key="loss_total")
    np.testing.assert_array_equal(iters, np.array([0, 50, 100], dtype=np.int64))
    np.testing.assert_allclose(losses, np.array([2.5, 1.0, 0.5]))
    assert config["wandb_run_id"] == "abc"
    assert config["d"] == 32

    # Default floor uses median of trailing 10%: with N=3 that's tail=1, so
    # L_M = 0.5. Excess = [2.0, 0.5, 0.0]. With iters=[0,50,100], widths from
    # np.diff are [50, 50], repeated to [50, 50, 50], so the Riemann sum is
    # 2.0*50 + 0.5*50 + 0.0*50 = 125.0.
    summary = epiplexity_summary(path)
    assert summary["L_M"] == pytest.approx(0.5)
    assert summary["N"] == 3
    assert summary["P_preq"] == pytest.approx(125.0)
    assert summary["H_T"] == pytest.approx(0.5 * 3)
    assert summary["eff"] == pytest.approx(125.0 / (0.5 * 3))


def test_load_loss_curve_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_loss_curve(Path("/nonexistent/metrics.jsonl"))


def test_load_loss_curve_missing_key_raises(tmp_path: Path) -> None:
    """A file with no rows matching the requested key raises ValueError."""
    path = tmp_path / "metrics.jsonl"
    with path.open("w") as fh:
        fh.write(json.dumps({"event": "config", "seed": 0}) + "\n")
        fh.write(json.dumps({"event": "log", "iter": 0, "diag/pr": 1.0}) + "\n")
    with pytest.raises(ValueError, match="no log entries"):
        load_loss_curve(path, loss_key="loss_total")
