"""Tests for Physics Track P4 (LEV circulation budget + gust-sign asymmetry).

CPU-only. Synthetic-data unit tests for the load-bearing primitives, plus a
skipif-real smoke test that the real v2.1 cache + decode + physics artefacts are
wired correctly (skipped when they are absent so the suite stays portable).

The LEV-ID machinery is reused verbatim from scripts/session23/exp_lev_tracking;
these tests exercise the P4 layer on top of it (the per-frame Gamma_LEV series,
the peak detection, the normalisation/scale convention, and the stats wiring).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "session28"))
sys.path.insert(0, str(REPO / "scripts" / "session23"))

import exp_lev_tracking as elt  # noqa: E402
import p4_lev_budget as p4  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers: build a synthetic field with a known negative LEV blob.
# ---------------------------------------------------------------------------
def _neg_blob_field(amp: float, cx_px: int = 55, cy_px: int = 55, rad_px: int = 8):
    """A (192, 96) field with one Gaussian NEGATIVE lobe in the x/c<1 region.

    cx_px in [48, 80) so it lands inside the leading-edge / wing search region.
    Returns a NORMALISED-scale field (so the elt OMEGA_THR=0.5 convention bites).
    """
    H, W = elt.H_PX, elt.W_PX
    yy, xx = np.meshgrid(np.arange(W), np.arange(H))
    r2 = (xx - cx_px) ** 2 + (yy - cy_px) ** 2
    field = -amp * np.exp(-r2 / (2.0 * rad_px**2))
    return field.astype(np.float64)


def _analytic_circulation(field2d_L, airfoil):
    """Ground-truth Gamma over the detected lobe via elt.lev_from_field."""
    lobe = elt.lev_from_field(field2d_L, airfoil)
    return None if lobe is None else lobe["Gamma"]


# ---------------------------------------------------------------------------
# 1. Gamma_LEV of a known negative blob == analytic circulation (sum*dx*dy).
# ---------------------------------------------------------------------------
def test_gamma_lev_known_blob_matches_analytic():
    from scipy.ndimage import label

    airfoil = elt.load_masks()[0]
    raw = _neg_blob_field(amp=3.0)  # well above OMEGA_THR=0.5
    fL = elt.large_scale(raw)
    # Independent analytic circulation: Gamma = sum_lobe omega * dx * dy over the
    # STRONGEST connected NEGATIVE component (exactly the lev_from_field rule), so
    # the test is a true closed-form check of the integral, not a re-run of the fn.
    reg = fL.copy()
    reg[elt.TE_X_INDEX :, :] = 0.0
    reg[airfoil] = 0.0
    lab, ncomp = label(reg < -elt.OMEGA_THR)
    assert ncomp >= 1
    strengths = [float((-reg[lab == i]).sum()) for i in range(1, ncomp + 1)]
    k = int(np.argmax(strengths)) + 1
    analytic = float(reg[lab == k].sum() * elt.DX * elt.DY)

    # Via the P4 series wrapper on a 1-frame window.
    ser = p4.lev_series(raw[None, :, :], airfoil)
    assert ser["n_detected"] == 1
    assert ser["gamma"][0] == pytest.approx(analytic, rel=1e-9, abs=1e-9)
    # circulation of a negative lobe is negative; peak |Gamma| is its magnitude.
    assert ser["gamma"][0] < 0
    assert ser["peak_gamma_abs"] == pytest.approx(abs(analytic), rel=1e-9)
    # The Gaussian lobe integral is O(amp * area); sanity that it is non-trivial.
    assert abs(ser["gamma"][0]) > 0.1


# ---------------------------------------------------------------------------
# 2. The sigma/c=0.05 Gaussian band suppresses fine scale.
# ---------------------------------------------------------------------------
def test_large_scale_band_suppresses_fine_scale():
    rng = np.random.default_rng(0)
    # A smooth large lobe plus high-frequency speckle.
    base = _neg_blob_field(amp=2.0, rad_px=12)
    speckle = rng.standard_normal((elt.H_PX, elt.W_PX)) * 1.5
    noisy = base + speckle
    fL = elt.large_scale(noisy)
    # The filter must remove most of the per-pixel speckle variance.
    resid_full = noisy - base
    resid_filt = fL - elt.large_scale(base)
    assert resid_filt.std() < 0.5 * resid_full.std()
    # And the large lobe must survive (peak |omega_L| still well above threshold).
    assert (-fL).max() > elt.OMEGA_THR


# ---------------------------------------------------------------------------
# 3. Peak detection on a known time-varying circulation curve.
# ---------------------------------------------------------------------------
def test_peak_detection_on_known_curve():
    airfoil = elt.load_masks()[0]
    # Amplitude ramps up then down; peak at window index 5 (frame 30).
    amps = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    n = len(amps)
    win = np.zeros((n, elt.H_PX, elt.W_PX))
    for t, a in enumerate(amps):
        if a > 0:
            win[t] = _neg_blob_field(amp=a)
    # Patch WINDOW_FRAMES so the absolute-frame mapping is deterministic here.
    saved = p4.WINDOW_FRAMES
    p4.WINDOW_FRAMES = np.arange(25, 25 + n)
    try:
        ser = p4.lev_series(win, airfoil)
    finally:
        p4.WINDOW_FRAMES = saved
    # |Gamma| should peak at the max-amplitude frame (index 5 -> frame 30).
    assert int(np.argmax(np.abs(ser["gamma"]))) == 5
    assert ser["peak_gamma_frame"] == 30
    # Detection only where amp>0 (9 frames).
    assert ser["n_detected"] == sum(a > 0 for a in amps)


# ---------------------------------------------------------------------------
# 4. No lobe above threshold -> not detected, zero circulation, NaN centroid.
# ---------------------------------------------------------------------------
def test_no_lobe_not_detected():
    airfoil = elt.load_masks()[0]
    weak = _neg_blob_field(amp=0.05)  # below OMEGA_THR after filtering
    ser = p4.lev_series(weak[None, :, :], airfoil)
    assert ser["n_detected"] == 0
    assert ser["gamma"][0] == 0.0
    assert np.isnan(ser["x"][0])
    assert ser["peak_gamma_abs"] == 0.0
    assert ser["peak_gamma_frame"] == -1


# ---------------------------------------------------------------------------
# 5. Correlation + case-clustering wiring (GP4 branch logic).
# ---------------------------------------------------------------------------
def test_gp4_correlation_wiring_and_branch():
    rng = np.random.default_rng(1)
    # Build rows where peak_gamma is a near-linear function of dcl => |r| high.
    rows = []
    cases = [f"C{i}" for i in range(6)]
    for ci, c in enumerate(cases):
        for e in range(4):
            dcl = 0.1 + 0.05 * ci + 0.01 * e
            gamma = 2.0 * dcl + 0.001 * rng.standard_normal()
            rows.append(
                {
                    "case_id": c,
                    "enc": e,
                    "dcl_peak_simple": dcl,
                    "peak_gamma_abs": gamma,
                    "n_detected": 5,
                }
            )
    gp4 = p4.gp4_correlation(rows, rng)
    assert gp4["n"] == len(rows)
    assert gp4["n_cases"] == len(cases)
    assert gp4["pearson_r"] > 0.9  # strong by construction
    assert gp4["branch"] == "main_text"  # |r| >= 0.6
    # CI must be a valid 2-element interval bracketing the point estimate-ish.
    lo, hi = gp4["pearson_ci"]
    assert lo <= hi
    assert np.isfinite(lo) and np.isfinite(hi)
    # case-permutation p is a probability in (0, 1].
    assert 0.0 < gp4["pearson_perm_p"] <= 1.0

    # A noise-only relationship should land in the appendix branch.
    rows_noise = []
    for ci, c in enumerate(cases):
        for e in range(4):
            rows_noise.append(
                {
                    "case_id": c,
                    "enc": e,
                    "dcl_peak_simple": float(rng.standard_normal()),
                    "peak_gamma_abs": float(rng.standard_normal()),
                    "n_detected": 5,
                }
            )
    gp4n = p4.gp4_correlation(rows_noise, rng)
    assert gp4n["branch"] == "appendix"
    assert abs(gp4n["pearson_r"]) < p4.GP4_THR


# ---------------------------------------------------------------------------
# 6. G-sign asymmetry: matched-|G| pairing + sign of the reported difference.
# ---------------------------------------------------------------------------
def test_gsign_asymmetry_matched_and_sign():
    rng = np.random.default_rng(2)
    rows = []
    # matched |G| in {1, 2}; G>0 gets a LARGER peak gamma by construction.
    for absG in (1.0, 2.0):
        for e in range(3):
            rows.append(
                {
                    "case_id": f"P{absG}",
                    "enc": e,
                    "G": absG,
                    "D": 0.5,
                    "Y": 0.1,
                    "peak_gamma_abs": 1.0 + absG + 0.01 * e,
                    "detach_tc": 0.5,
                    "n_detected": 5,
                }
            )
            rows.append(
                {
                    "case_id": f"N{absG}",
                    "enc": e,
                    "G": -absG,
                    "D": 0.5,
                    "Y": 0.1,
                    "peak_gamma_abs": 0.5 + 0.01 * e,
                    "detach_tc": 0.6,
                    "n_detected": 5,
                }
            )
    res = p4.gsign_asymmetry(rows, rng)
    assert set(res["matched_absG"]) == {1.0, 2.0}
    pg = res["peak_gamma_abs"]
    # G>0 mean must exceed G<0 mean -> positive asymmetry.
    assert pg["asymmetry_pos_minus_neg"] > 0
    assert pg["pos_mean"] > pg["neg_mean"]
    # paired-bucket CI is a valid interval.
    lo, hi = res["matched_paired_peak_gamma_diff_ci"]
    assert lo <= hi
    assert res["n_matched_buckets"] == 2


# ---------------------------------------------------------------------------
# 7. Key-convention tolerance: normalisation divisor brings raw -> ~unit scale.
# ---------------------------------------------------------------------------
def test_norm_divisor_scale_convention():
    # A raw negative lobe of amplitude ~ 300 (typical core) divided by 3*train_std
    # (~10.9 for v2p1) lands a few units negative -> above OMEGA_THR after filter.
    divisor = 3.0 * 3.6336832437749322  # v2p1 train_std
    raw_amp = 300.0
    norm = _neg_blob_field(amp=raw_amp) / divisor
    fL = elt.large_scale(norm)
    # Peak of the filtered normalised lobe should be O(1-10), comfortably > thr.
    assert (-fL).max() > elt.OMEGA_THR
    assert (-fL).max() < 50.0  # not absurd


# ---------------------------------------------------------------------------
# 8. Detachment clock (circulation half-life): a lobe whose |Gamma| ramps down
#    below 50% of its peak detaches; a steady lobe does not.
# ---------------------------------------------------------------------------
def test_detachment_clock():
    airfoil = elt.load_masks()[0]
    n = 8
    saved = p4.WINDOW_FRAMES
    p4.WINDOW_FRAMES = np.arange(p4.IMPACT_FRAME, p4.IMPACT_FRAME + n)
    try:
        # |Gamma| ramps up to a peak then decays well below half: should detach.
        amps = [1.0, 2.0, 4.0, 5.0, 2.0, 1.0, 0.5, 0.0]  # peak at t=3, half=2.5
        win = np.zeros((n, elt.H_PX, elt.W_PX))
        for t, a in enumerate(amps):
            if a > 0:
                win[t] = _neg_blob_field(amp=a)
        ser = p4.lev_series(win, airfoil)
        assert ser["peak_gamma_frame"] == p4.IMPACT_FRAME + 3
        assert np.isfinite(ser["detach_tc"])
        # First frame after the peak (t=3) where |Gamma| < 0.5 * peak is t=4
        # (amp 2.0 -> Gamma well below the amp-5.0 peak's half).
        assert ser["detach_tc"] == pytest.approx((p4.IMPACT_FRAME + 4 - p4.IMPACT_FRAME) * p4.DT_TC)

        # A steady lobe (constant amplitude) never drops below half -> no detach.
        win2 = np.stack([_neg_blob_field(amp=3.0, cx_px=55) for _ in range(n)])
        ser2 = p4.lev_series(win2, airfoil)
        assert not np.isfinite(ser2["detach_tc"])
    finally:
        p4.WINDOW_FRAMES = saved


# ---------------------------------------------------------------------------
# 9. skipif-real: the real v2.1 artefacts are wired and produce sane numbers.
# ---------------------------------------------------------------------------
_REAL_PRESENT = (
    p4.PHYSICS_NPZ.exists()
    and (p4.DECODE_ROOT / "tf_noc" / "recon_test_b.npz").exists()
    and p4.SPLIT_PATH.exists()
    and p4.PIPELINE_MANIFEST.exists()
    and (p4.CACHE_ROOT).exists()
)


@pytest.mark.skipif(not _REAL_PRESENT, reason="v2.1 cache/decode/physics artefacts absent")
def test_real_artefacts_load_and_one_encounter():
    divisor = p4.load_pipeline_norm_divisor()
    assert divisor == pytest.approx(3.0 * 3.6336832437749322, rel=1e-6)
    peak = p4.load_physics_peak_dcl()
    assert len(peak) > 0
    blob = p4.load_decode("tf_noc", "recon", "test_b")
    assert blob is not None
    assert blob["omega"].shape[1:] == (31, 192, 96)
    # The decoded fields are raw scale (O(100s) max), not normalised.
    assert blob["omega"].max() > 20.0
    # Window frames are [25, 55].
    assert list(blob["window_frames"][[0, -1]]) == [25, 55]
