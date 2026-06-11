"""Tests for scripts/session28/undisturbed_validation.py (plan A2, referee M13).

Properties under test, fixed after the 2026-06-11 root-cause session:
(1) spectral_peaks returns only true local maxima of the one-sided spectrum,
    with the log-parabolic shift clamped to half a bin, so St >= 0 always
    (the original picker emitted St = -0.030 from a zeroed-window edge).
(2) lift_spectrum_summary separates the shedding line (above a stated St
    floor) from the low-frequency modulation (below it) and detects the
    subharmonic line at half the shedding frequency.
(3) window_stats computes moment statistics over a stated stationary window.

A real-data integration test runs when the PREVENT Baseline.h5 is present.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "session28"))

import undisturbed_validation as uv  # noqa: E402

DT = 0.05


def _synthetic(n: int = 700, seed: int = 0) -> np.ndarray:
    """Three known tones (shedding 0.69, subharmonic 0.345, modulation 0.04)."""
    t = np.arange(n) * DT
    rng = np.random.default_rng(seed)
    return (
        1.00 * np.sin(2 * np.pi * 0.690 * t)
        + 0.35 * np.sin(2 * np.pi * 0.345 * t + 0.7)
        + 0.80 * np.sin(2 * np.pi * 0.040 * t + 0.2)
        + 0.02 * rng.standard_normal(n)
    )


def _red_noise_plus_tone(n: int = 700, seed: int = 1) -> np.ndarray:
    """Integrated noise (smoothly decaying spectrum) + one tone: the structure
    that made the original picker interpolate on a non-local-max slope bin."""
    rng = np.random.default_rng(seed)
    drift = np.cumsum(rng.standard_normal(n)) * 0.02
    t = np.arange(n) * DT
    return drift + 0.5 * np.sin(2 * np.pi * 0.690 * t)


def _spectrum(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xf = x - x.mean()
    win = np.hanning(xf.size)
    P = np.abs(np.fft.rfft(xf * win)) ** 2 / (np.sum(win**2) / 2.0)
    f = np.fft.rfftfreq(xf.size, d=DT)
    return f, P


def _assert_peaks_are_clamped_local_maxima(x: np.ndarray, peaks: list[dict]) -> None:
    f, P = _spectrum(x)
    df = f[1] - f[0]
    interior = np.arange(1, P.size - 1)
    locmax = interior[(P[interior] >= P[interior - 1]) & (P[interior] >= P[interior + 1])]
    for p in peaks:
        assert p["St"] >= 0.0, f"negative frequency emitted: {p}"
        nearest = locmax[np.argmin(np.abs(f[locmax] - p["St"]))]
        assert (
            abs(f[nearest] - p["St"]) <= 0.5 * df + 1e-12
        ), f"peak {p} is not within half a bin of a local maximum"


class TestSpectralPeaks:
    def test_three_tones_recovered_in_power_order(self) -> None:
        x = _synthetic()
        peaks = uv.spectral_peaks(x, DT, n_peaks=3)
        assert len(peaks) == 3
        df = 1.0 / (x.size * DT)
        got = [p["St"] for p in peaks]
        # power order: amp 1.0 tone, amp 0.8 tone, amp 0.35 tone
        for st, true in zip(got, (0.690, 0.040, 0.345)):
            assert abs(st - true) <= df, f"St {st} vs true {true}"

    def test_peaks_are_local_maxima_with_clamped_shift_synthetic(self) -> None:
        x = _synthetic()
        _assert_peaks_are_clamped_local_maxima(x, uv.spectral_peaks(x, DT, n_peaks=5))

    def test_no_negative_or_slope_peaks_on_red_noise(self) -> None:
        x = _red_noise_plus_tone()
        peaks = uv.spectral_peaks(x, DT, n_peaks=4)
        _assert_peaks_are_clamped_local_maxima(x, peaks)


class TestLiftSpectrumSummary:
    def test_shedding_modulation_subharmonic_split(self) -> None:
        x = _synthetic()
        s = uv.lift_spectrum_summary(x, DT, st_floor=0.15)
        df = 1.0 / (x.size * DT)
        assert abs(s["St_shedding"]["St"] - 0.690) <= df
        assert abs(s["St_modulation"]["St"] - 0.040) <= df
        assert s["St_subharmonic"] is not None
        assert abs(s["St_subharmonic"]["St"] - 0.345) <= 1.5 * df

    def test_no_subharmonic_reported_when_absent(self) -> None:
        t = np.arange(700) * DT
        rng = np.random.default_rng(2)
        x = np.sin(2 * np.pi * 0.690 * t) + 0.02 * rng.standard_normal(700)
        s = uv.lift_spectrum_summary(x, DT, st_floor=0.15)
        assert s["St_subharmonic"] is None


class TestWindowStats:
    def test_moments_over_stated_window(self) -> None:
        n = 800
        t = np.arange(n) * DT
        cl = 0.75 + 0.10 * np.sin(2 * np.pi * 0.69 * t)
        cd = 0.25 + 0.02 * np.sin(2 * np.pi * 1.38 * t)
        w = uv.window_stats(cl, cd, DT, lo_tc=20.0, hi_tc=40.0)
        assert w["n_frames"] == 400
        assert abs(w["CL_mean"] - 0.75) < 5e-3
        assert abs(w["CL_rms"] - 0.10 / np.sqrt(2)) < 5e-3
        assert abs(w["CD_mean"] - 0.25) < 5e-3
        assert w["window_tc"] == [20.0, 40.0]


@pytest.mark.skipif(
    not (
        Path(os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT")))
        / "data/raw/periodic/Baseline.h5"
    ).exists(),
    reason="PREVENT raw Baseline.h5 not available",
)
class TestRealBaseline:
    def test_baseline_lift_lines(self) -> None:
        import h5py

        raw = (
            Path(os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT")))
            / "data/raw/periodic/Baseline.h5"
        )
        with h5py.File(raw, "r") as g:
            cl = np.asarray(g["forces/CL"], dtype=np.float64).squeeze()
        x = cl[100:]
        peaks = uv.spectral_peaks(x, DT, n_peaks=4)
        _assert_peaks_are_clamped_local_maxima(x, peaks)
        s = uv.lift_spectrum_summary(x, DT, st_floor=0.15)
        # measured 2026-06-11: dominant lift line 0.69, subharmonic 0.34,
        # low-frequency modulation 0.04
        assert 0.60 <= s["St_shedding"]["St"] <= 0.78
        assert s["St_modulation"]["St"] < 0.10
        assert s["St_subharmonic"] is not None
        assert 0.30 <= s["St_subharmonic"]["St"] <= 0.40
