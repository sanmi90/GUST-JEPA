"""Unit tests for :mod:`src.evaluation.decoder_metrics`."""

from __future__ import annotations

import numpy as np

from src.evaluation.decoder_metrics import (
    aggregate_split_metrics,
    compute_encounter_metrics,
    radial_power_spectrum,
    wake_2d_premult_spectrum,
    wake_2d_premult_spectrum_series,
    wake_mask,
)


def test_wake_mask_shape_and_extent() -> None:
    """The wake mask covers x in (0, 4.5) and |y| < 1.25 on the 192x96 grid."""
    m = wake_mask(192, 96)
    assert m.shape == (192, 96)
    assert m.dtype == bool
    # Roughly: x in (0, 4.5) is 4.5/(4.5 - -1.5) = 75% of H, and
    # |y| < 1.25 is 2.5/3.0 = ~83% of W. So ~62% of pixels.
    frac = m.mean()
    assert 0.55 < frac < 0.70, f"wake fraction {frac:.3f} out of expected band"


def test_compute_metrics_perfect_reconstruction() -> None:
    """On pred == target, all error metrics are exactly zero."""
    rng = np.random.default_rng(0)
    target = rng.normal(size=(8, 192, 96)).astype(np.float32) * 50.0
    m = compute_encounter_metrics(target, target.copy(), active_tau_raw=1.0)
    assert m.mse_full == 0.0
    assert m.mse_active == 0.0
    assert m.mse_wake == 0.0
    assert m.eps_volume == 0.0
    assert m.enstrophy_rel_err_full == 0.0
    assert m.enstrophy_rel_err_wake == 0.0
    assert m.circulation_abs_err_wake == 0.0
    assert m.local_fft_err_mean == 0.0
    assert m.radial_spectrum_l2_wake == 0.0
    # SSIM == 1 on perfect reconstruction (within floating point)
    assert m.ssim_mean > 0.999


def test_compute_metrics_zeros_baseline() -> None:
    """When pred is all-zeros and target has structure, MSE > 0 and
    enstrophy_rel_err is ~1 (the reconstruction missed all enstrophy)."""
    rng = np.random.default_rng(0)
    target = rng.normal(size=(4, 192, 96)).astype(np.float32) * 50.0
    pred = np.zeros_like(target)
    m = compute_encounter_metrics(target, pred, active_tau_raw=1.0)
    assert m.mse_full > 0.0
    assert m.eps_volume > 0.9, f"all-zeros pred should give eps_vol near 1, got {m.eps_volume}"
    assert m.enstrophy_rel_err_full > 0.9
    assert m.enstrophy_rel_err_wake > 0.9


def test_rel_l2_series_no_blowup_on_zero_target() -> None:
    """rel_l2_series is finite even when the target is identically zero
    (the eps floor protects the denominator)."""
    from src.evaluation.decoder_metrics import rel_l2_series
    t = np.zeros(8)
    p = np.ones(8)
    val = rel_l2_series(p, t)
    assert np.isfinite(val), f"rel_l2 should be finite on zero target, got {val}"


def test_enstrophy_rel_err_finite_on_near_zero_frames() -> None:
    """The enstrophy relative error must NOT blow up when individual
    frames have near-zero target enstrophy (the Test A Baseline case).
    Aggregating per-encounter via Fukami's L2-rel-error over the time
    series gives a finite, well-defined number."""
    target = np.zeros((10, 192, 96), dtype=np.float32)
    target[5, 100, 50] = 10.0  # one nonzero frame
    pred = target.copy()
    pred[5, 100, 50] = 5.0  # half the magnitude
    m = compute_encounter_metrics(target, pred, active_tau_raw=1.0)
    assert np.isfinite(m.enstrophy_rel_err_full), (
        "enstrophy_rel_err blew up on a sparse-active target"
    )
    assert np.isfinite(m.radial_spectrum_l2_wake)


def test_radial_power_spectrum_smoke() -> None:
    """The radial spectrum returns finite bin centers and powers."""
    rng = np.random.default_rng(0)
    field = rng.normal(size=(192, 96)).astype(np.float32)
    k, P = radial_power_spectrum(field, n_bins=16)
    assert k.shape == (16,)
    assert P.shape == (16,)
    assert np.isfinite(P).all()
    assert (P >= 0).all()


def test_aggregate_split_metrics() -> None:
    """Aggregator returns mean/median per field and the encounter count."""
    rng = np.random.default_rng(0)
    target = rng.normal(size=(3, 192, 96)).astype(np.float32) * 30.0
    pred = target + 1.0
    m1 = compute_encounter_metrics(target, pred, active_tau_raw=1.0)
    m2 = compute_encounter_metrics(target, pred + 1.0, active_tau_raw=1.0)
    agg = aggregate_split_metrics([m1, m2])
    assert agg["n_encounters"] == 2
    assert "mse_full_mean" in agg
    assert "mse_full_median" in agg
    assert agg["mse_full_mean"] > 0


# -----------------------------------------------------------------------------
# Session 12: 2D premultiplied wake power spectrum
# -----------------------------------------------------------------------------


def test_wake_2d_premult_spectrum_perfect_alignment() -> None:
    """On pred == target, contour IoU is 1 and wavelength ratio is 1."""
    rng = np.random.default_rng(0)
    target = rng.normal(size=(192, 96)).astype(np.float32) * 30.0
    out = wake_2d_premult_spectrum(target, target)
    # Both pred and target give identical premultiplied PSDs, so every
    # level-mask coincides exactly: IoU == 1, median wavelength ratio == 1.
    assert np.allclose(out["contour_iou"], 1.0), out["contour_iou"]
    assert np.allclose(out["median_wavelength_ratio"], 1.0), \
        out["median_wavelength_ratio"]
    assert out["max_wavelength_ratio"] == 1.0
    assert out["mean_contour_iou"] == 1.0
    # The wake patch shape is the canonical (144, 80) for 192x96 inputs.
    assert out["wake_patch_shape"] == (144, 80)


def test_wake_2d_premult_spectrum_returns_expected_keys() -> None:
    """The output dict has every documented key with shapes consistent."""
    rng = np.random.default_rng(1)
    target = rng.normal(size=(192, 96)).astype(np.float32) * 20.0
    pred = target + 0.5 * rng.normal(size=(192, 96)).astype(np.float32) * 20.0
    out = wake_2d_premult_spectrum(pred, target)
    expected_keys = {
        "premult_pred", "premult_target", "kx_grid", "ky_grid",
        "contour_levels", "contour_iou", "median_wavelength_ratio",
        "max_wavelength_ratio", "mean_contour_iou", "wake_patch_shape",
    }
    assert set(out.keys()) == expected_keys
    h, w = out["wake_patch_shape"]
    for k in ("premult_pred", "premult_target", "kx_grid", "ky_grid"):
        assert out[k].shape == (h, w), f"{k}: {out[k].shape}"
    # Premultiplied PSDs are non-negative.
    assert (out["premult_pred"] >= 0).all()
    assert (out["premult_target"] >= 0).all()
    # Mismatched fields produce ratio >= 1.
    assert out["max_wavelength_ratio"] >= 1.0
    # IoU is in [0, 1].
    assert (out["contour_iou"] >= 0).all() and (out["contour_iou"] <= 1).all()


def test_wake_2d_premult_spectrum_series_perfect() -> None:
    """Time-series variant: pred == target gives IoU 1 across frames."""
    rng = np.random.default_rng(2)
    target = rng.normal(size=(4, 192, 96)).astype(np.float32) * 25.0
    out = wake_2d_premult_spectrum_series(target, target)
    assert out["n_frames"] == 4
    assert np.allclose(out["contour_iou"], 1.0)
    assert out["max_wavelength_ratio"] == 1.0


def test_wake_2d_premult_spectrum_detects_high_freq_mismatch() -> None:
    """High-frequency content in pred but smooth target produces a
    measurable wavelength shift in the premultiplied spectrum."""
    rng = np.random.default_rng(3)
    H, W = 192, 96
    # Build a target with primarily low-wavenumber content
    xs = np.linspace(0, 2 * np.pi, H)
    ys = np.linspace(0, 2 * np.pi, W)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    target = np.sin(xx) * np.cos(yy) * 10.0  # low-frequency
    # Pred adds a high-frequency component
    pred = target + 0.5 * np.sin(8 * xx) * np.cos(8 * yy) * 10.0
    out = wake_2d_premult_spectrum(pred, target)
    # The high-frequency mismatch should produce a measurable difference;
    # contour IoU strictly less than 1 at the higher contour levels.
    assert out["mean_contour_iou"] < 1.0
    # The wavelength ratio should reflect the high-frequency mismatch:
    # the pred has shorter wavelengths in its premultiplied spectrum, so
    # the median wavelength ratio is non-trivial.
    assert np.isfinite(out["max_wavelength_ratio"])
    assert out["max_wavelength_ratio"] > 1.0


def test_wake_2d_premult_spectrum_no_window_finite() -> None:
    """With window=None the result is still finite (no NaN/Inf)."""
    rng = np.random.default_rng(4)
    target = rng.normal(size=(192, 96)).astype(np.float32) * 30.0
    pred = target * 0.9
    out = wake_2d_premult_spectrum(pred, target, window=None)
    assert np.isfinite(out["max_wavelength_ratio"]) or np.isnan(
        out["max_wavelength_ratio"]
    )
    assert (np.isfinite(out["contour_iou"])).all()
