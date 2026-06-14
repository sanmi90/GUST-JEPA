"""Minimal tests for SESSION29 Track A (baseline external validation) helpers.

The spectral-peak helper must recover the frequency of a clean synthetic
sinusoid (the Strouhal estimator); the rms helper must return the amplitude/
sqrt(2) of a zero-mean sinusoid and be mean-invariant; the within-band envelope
must bracket the available references.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "session29"))
import validate_baseline as vb  # noqa: E402


def test_spectral_peak_recovers_synthetic_strouhal():
    # convective-time series at dt = 0.05; inject a clean line at St = 0.675.
    dt = 0.05
    t = np.arange(2048) * dt
    st_true = 0.675
    x = 0.5 + 0.2 * np.sin(2 * np.pi * st_true * t)
    st = vb.spectral_peak_st(x, dt)
    df = 1.0 / (t.size * dt)
    assert abs(st - st_true) <= 2 * df  # within a couple of spectral bins


def test_rms_fluctuation_of_sinusoid_is_amplitude_over_sqrt2():
    t = np.linspace(0, 100, 20000, endpoint=False)
    amp = 0.13
    x = 7.0 + amp * np.sin(2 * np.pi * 0.34 * t)  # nonzero mean
    rms = vb.rms_fluctuation(x)
    assert abs(rms - amp / np.sqrt(2)) < 1e-3  # mean removed, amplitude/sqrt2


def test_frame_wake_enstrophy_only_counts_wake_box():
    mask = vb.wake_mask()
    omega = np.zeros((vb.NX, vb.NY), dtype=np.float64)
    # a unit-magnitude cell well inside the wake box contributes; one upstream
    # of x=0.5 (outside the box) must not.
    x, y = vb._coord_grid()
    in_x = int(np.argmin(np.abs(x - 2.0)))
    in_y = int(np.argmin(np.abs(y - 0.0)))
    out_x = int(np.argmin(np.abs(x - (-1.0))))  # upstream, outside box
    omega[out_x, in_y] = 1000.0
    e_outside_only = vb.frame_wake_enstrophy(omega, mask)
    assert e_outside_only == 0.0
    omega[in_x, in_y] = 5.0
    e_inside = vb.frame_wake_enstrophy(omega, mask)
    assert abs(e_inside - 25.0 * vb.DX * vb.DY) < 1e-9


def test_within_band_envelope_brackets_available_refs():
    # ours equal to a Fukami-like value should land inside the 0.734-0.763 band.
    ours = {
        "CL_mean": 0.74,
        "CL_rms": 0.12,
        "CD_mean": 0.25,
        "CD_rms": 0.02,
        "St": 0.675,
        "enstrophy": 1.0,
    }
    table, availability = vb.build_table(ours)
    verdict = vb.within_band(table)
    assert verdict["CL_mean"]["checked"]
    assert verdict["CL_mean"]["in_band"]
    assert verdict["CL_mean"]["ref_band"] == [0.734, 0.763]
    # Rolandi/Gupta have no rms -> only Fukami available for CL_rms.
    assert availability["Rolandi2025"] == ["CL_mean", "CD_mean"]
    assert "CL_rms" not in availability["Rolandi2025"]
    # enstrophy cell is the no-external-reference marker for every reference.
    assert table["enstrophy"]["Fukami2025"]["status"] == vb.NO_EXT
