"""Per-encounter metrics for the JEPA visualisation decoder (Session 10).

The Session 9 ``decoder_summary.json`` reported MSE, SSIM, and
``eps_volume`` (Fukami's L2 relative error). Session 10 adds five
physics-grounded metrics so the paper can argue not just that the
decoded field looks visually similar, but that it reproduces the
right enstrophy, circulation, and wavenumber content in the wake.

All metrics here take RAW-SCALE omega fields ``(T, H, W)`` and return
floats. The training loop computes losses in normalised space; we
unnormalise back to raw scale before evaluating these metrics so the
numbers are comparable across configurations and to Fukami's
publication.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


_PHYSICAL_X = (-1.5, 4.5)
_PHYSICAL_Y = (-1.5, 1.5)
_WAKE_X = (0.0, 4.5)
_WAKE_Y_ABS = 1.25


def _build_pixel_grids(H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    """Return physical (x, y) grids of shape (H, W) for the cache geometry.

    Pixel index (h, w) corresponds to physical coords
        x = x_min + h * (x_max - x_min) / (H - 1)
        y = y_min + w * (y_max - y_min) / (W - 1)
    matching the visualisation convention used in
    ``scripts/session9_decoder_fig3_pipeline.py``.
    """
    x_min, x_max = _PHYSICAL_X
    y_min, y_max = _PHYSICAL_Y
    xs = np.linspace(x_min, x_max, H)
    ys = np.linspace(y_min, y_max, W)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    return xx, yy


def wake_mask(H: int, W: int) -> np.ndarray:
    """Boolean wake-ROI mask of shape (H, W): True inside the wake region."""
    xx, yy = _build_pixel_grids(H, W)
    return (xx > _WAKE_X[0]) & (xx < _WAKE_X[1]) & (np.abs(yy) < _WAKE_Y_ABS)


def active_mask(target: np.ndarray, tau: float) -> np.ndarray:
    """Per-frame active-pixel mask in RAW scale (|omega| > tau)."""
    return np.abs(target) > tau


def _safe_div(a: float, b: float, fallback: float = 0.0) -> float:
    if abs(b) < 1e-12:
        return fallback
    return a / b


def radial_power_spectrum(field: np.ndarray, mask: Optional[np.ndarray] = None,
                          n_bins: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Compute 1D radial power spectrum of a 2D field.

    Returns ``(k_bins, power)`` where ``k_bins`` is the central
    wavenumber of each radial bin and ``power`` is the average
    ``|FFT|^2`` value of pixels in that bin. The wavenumber bins are
    linearly spaced from 0 to the Nyquist limit; the masked input is
    multiplied by a Hann window before the FFT to suppress edge
    artifacts.

    If ``mask`` is provided, the field is restricted to ``mask=True``
    pixels (others zeroed) before windowing -- this lets the caller
    compute the spectrum of, e.g., the wake region only.
    """
    H, W = field.shape
    work = field.copy()
    if mask is not None:
        work[~mask] = 0.0
    # 2D Hann window
    wx = np.hanning(H)[:, None]
    wy = np.hanning(W)[None, :]
    work = work * (wx * wy)
    F = np.fft.fftshift(np.fft.fft2(work))
    P = np.abs(F) ** 2 / (H * W)
    # Radial wavenumber grid in pixel-frequency units
    kx = np.fft.fftshift(np.fft.fftfreq(H) * H)[:, None]
    ky = np.fft.fftshift(np.fft.fftfreq(W) * W)[None, :]
    k = np.sqrt(kx ** 2 + ky ** 2)
    k_max = min(H, W) // 2
    k_bins = np.linspace(0, k_max, n_bins + 1)
    radial = np.zeros(n_bins, dtype=np.float64)
    for i in range(n_bins):
        m = (k >= k_bins[i]) & (k < k_bins[i + 1])
        if m.any():
            radial[i] = P[m].mean()
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    return k_centers, radial


@dataclass
class EncounterMetrics:
    """Per-encounter metric bundle (raw-scale)."""

    mse_full: float
    mse_active: float
    mse_inactive: float
    mse_wake: float
    ssim_mean: float
    eps_per_frame: float
    eps_volume: float
    enstrophy_rel_err_full: float
    enstrophy_rel_err_wake: float
    circulation_abs_err_wake: float
    local_fft_err_mean: float
    radial_spectrum_l2_wake: float
    n_frames: int

    def to_dict(self) -> dict:
        return {k: float(v) for k, v in self.__dict__.items()}


def _ssim_single(x: np.ndarray, y: np.ndarray,
                 c1: float = 0.16, c2: float = 1.44) -> float:
    """Fukami's SSIM definition on a single (H, W) pair."""
    mu_x, mu_y = x.mean(), y.mean()
    var_x, var_y = x.var(), y.var()
    cov_xy = ((x - mu_x) * (y - mu_y)).mean()
    num = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
    return float(num / max(den, 1e-12))


def _l2_relative_error(q: np.ndarray, q_hat: np.ndarray,
                       eps: float = 1.0) -> float:
    """Fukami's L_2 relative reconstruction error.

    Uses an eps floor of 1.0 in raw vorticity units to prevent the
    metric from exploding on near-zero baseline frames where the
    denominator's ``||q||_2`` collapses. The floor is a coarse
    physical-scale safety net (typical |omega| is in the tens to
    hundreds for this dataset).
    """
    num = float(np.sqrt(((q - q_hat) ** 2).sum()))
    den = float(np.sqrt((q ** 2).sum()))
    return num / max(den, eps)


def rel_l2_series(p_series: np.ndarray, t_series: np.ndarray,
                  eps: float = 1e-6) -> float:
    """Aggregate L2 relative error on a time-series-like signal.

    Standard Fukami-style rel-L2 with one ratio per encounter (not per
    frame), computed by summing over the entire input. This avoids the
    per-frame blowup that occurs when some frames have near-zero
    denominator. The eps floor only catches the degenerate
    target == 0 case; for typical inputs the denominator is large and
    eps is inactive.

    Use this for time-aggregated physics metrics (enstrophy series,
    spectrum bins, etc.) where the per-frame relative error is
    physically meaningless when the target itself is small.
    """
    p = np.asarray(p_series).ravel()
    t = np.asarray(t_series).ravel()
    num = float(np.sqrt(((p - t) ** 2).sum()))
    den = float(np.sqrt((t ** 2).sum()))
    return num / max(den, eps)


def _local_fft_err(pred: np.ndarray, target: np.ndarray, patch: int = 32) -> float:
    """Mean absolute FFT magnitude error over non-overlapping patches.

    The mean is taken first across frequency components within a patch
    and then across patches and frames.
    """
    T, H, W = pred.shape
    if H % patch or W % patch:
        # No FFT comparison if the grid doesn't tile.
        return float("nan")
    nH, nW = H // patch, W // patch
    errs = []
    for t in range(T):
        for ih in range(nH):
            for iw in range(nW):
                pp = pred[t, ih * patch: (ih + 1) * patch,
                            iw * patch: (iw + 1) * patch]
                tp = target[t, ih * patch: (ih + 1) * patch,
                              iw * patch: (iw + 1) * patch]
                Fp = np.fft.fft2(pp)
                Ft = np.fft.fft2(tp)
                errs.append(np.abs(Fp - Ft).mean())
    return float(np.mean(errs)) if errs else float("nan")


def compute_encounter_metrics(
    target: np.ndarray,
    pred: np.ndarray,
    active_tau_raw: float = 1.0,
    radial_n_bins: int = 32,
) -> EncounterMetrics:
    """Compute the Session 10 extended metric bundle for one encounter.

    Args:
        target: ``(T, H, W)`` ground-truth omega (raw scale).
        pred: ``(T, H, W)`` reconstruction (raw scale).
        active_tau_raw: Raw-omega threshold separating active vs
            inactive pixels. Default 1.0 (units of 1 / convective time)
            -- a coarse heuristic; the wake-ROI and freestream regions
            are not solely separated by this.
        radial_n_bins: Number of radial bins in the spectrum.
    """
    assert target.shape == pred.shape, (
        f"shape mismatch: target {target.shape} vs pred {pred.shape}"
    )
    T, H, W = target.shape

    diff = pred - target
    mse_full = float((diff ** 2).mean())

    active = active_mask(target, active_tau_raw)
    inactive = ~active
    mse_active = float((diff[active] ** 2).mean()) if active.any() else float("nan")
    mse_inactive = float((diff[inactive] ** 2).mean()) if inactive.any() else float("nan")

    wm = wake_mask(H, W)
    wake_3d = np.broadcast_to(wm, target.shape)
    mse_wake = float((diff[wake_3d] ** 2).mean()) if wake_3d.any() else float("nan")

    ssim_mean = float(np.mean([_ssim_single(target[t], pred[t]) for t in range(T)]))
    # Per-frame eps: median across frames within the encounter (the mean was
    # outlier-dominated on Baseline pre-impact frames where the per-frame
    # denominator ||q_t||_2 drops to the eps=1.0 floor and the ratio explodes).
    eps_pf_series = [_l2_relative_error(target[t], pred[t]) for t in range(T)]
    eps_pf = float(np.median(eps_pf_series))
    eps_vol = _l2_relative_error(target, pred)

    # Enstrophy: spatial integral of omega^2 per frame. Relative error is
    # aggregated as one Fukami-style L2 ratio over the (T,) time series
    # to avoid per-frame blowup on near-zero-target frames.
    enstrophy_t = (target ** 2).sum(axis=(-2, -1))
    enstrophy_p = (pred ** 2).sum(axis=(-2, -1))
    enstrophy_full = rel_l2_series(enstrophy_p, enstrophy_t)

    enstrophy_t_w = (target ** 2 * wm).sum(axis=(-2, -1))
    enstrophy_p_w = (pred ** 2 * wm).sum(axis=(-2, -1))
    enstrophy_wake = rel_l2_series(enstrophy_p_w, enstrophy_t_w)

    # Circulation (signed): integral of omega per frame inside the wake.
    # Absolute, not relative -- the magnitude is interpretable directly.
    circ_t = (target * wm).sum(axis=(-2, -1))
    circ_p = (pred * wm).sum(axis=(-2, -1))
    circ_abs_err = float(np.mean(np.abs(circ_p - circ_t)))

    local_fft = _local_fft_err(pred, target, patch=32)

    # Radial power spectrum on the wake ROI: one L2 relative error per
    # encounter over the full (T, n_bins) spectrum, not a mean of
    # per-frame ratios. This keeps the metric finite even on frames
    # where some wavenumber bins have near-zero target power.
    sp_t_all = []
    sp_p_all = []
    for t in range(T):
        _, P_t = radial_power_spectrum(target[t], mask=wm, n_bins=radial_n_bins)
        _, P_p = radial_power_spectrum(pred[t], mask=wm, n_bins=radial_n_bins)
        sp_t_all.append(P_t)
        sp_p_all.append(P_p)
    sp_t_all = np.asarray(sp_t_all)
    sp_p_all = np.asarray(sp_p_all)
    radial_spectrum_l2 = rel_l2_series(sp_p_all, sp_t_all)

    return EncounterMetrics(
        mse_full=mse_full,
        mse_active=mse_active,
        mse_inactive=mse_inactive,
        mse_wake=mse_wake,
        ssim_mean=ssim_mean,
        eps_per_frame=eps_pf,
        eps_volume=eps_vol,
        enstrophy_rel_err_full=enstrophy_full,
        enstrophy_rel_err_wake=enstrophy_wake,
        circulation_abs_err_wake=circ_abs_err,
        local_fft_err_mean=local_fft,
        radial_spectrum_l2_wake=radial_spectrum_l2,
        n_frames=T,
    )


def aggregate_split_metrics(per_encounter: list[EncounterMetrics]) -> dict:
    """Aggregate per-encounter metrics into mean / median per metric."""
    if not per_encounter:
        return {"n_encounters": 0}
    keys = [k for k in per_encounter[0].__dict__.keys() if k != "n_frames"]
    out: dict[str, float] = {}
    for k in keys:
        vals = np.array([getattr(m, k) for m in per_encounter])
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            out[f"{k}_mean"] = float("nan")
            out[f"{k}_median"] = float("nan")
        else:
            out[f"{k}_mean"] = float(np.mean(vals))
            out[f"{k}_median"] = float(np.median(vals))
    out["n_encounters"] = len(per_encounter)
    return out


# -----------------------------------------------------------------------------
# 2D premultiplied wake power spectrum (Session 12, PRF 2026 Figs 5-6 style)
# -----------------------------------------------------------------------------


def _hann_window_2d_np(h: int, w: int) -> np.ndarray:
    """Separable 2D Hann window of shape (h, w)."""
    wx = np.hanning(h)[:, None]
    wy = np.hanning(w)[None, :]
    return wx * wy


def wake_2d_premult_spectrum(
    pred: np.ndarray,
    target: np.ndarray,
    physical_extent: tuple[float, float, float, float] = (-1.5, 4.5, -1.5, 1.5),
    wake_x_min: float = 0.0,
    wake_x_max: float = 4.5,
    wake_y_max: float = 1.25,
    window: Optional[str] = "hann",
    contour_levels: tuple[float, ...] = (0.10, 0.50, 0.90),
) -> dict:
    """2D premultiplied wake power spectrum agreement (PRF 2026 style).

    Following Balasubramanian, Cremades, Vinuesa, Tammisola (Phys. Rev.
    Fluids 11, 044907, 2026; their Figs. 5-6) the premultiplied power
    spectral density is computed as

        E_premult(k_x, k_y) = |k_x| * |k_y| * phi_omega(k_x, k_y),

    where ``phi_omega`` is the 2D PSD of the (wake-cropped, windowed)
    omega field. The premultiplication emphasises the energy-bearing
    intermediate wavenumbers; contours of normalised E_premult trace
    out which (lambda_x, lambda_y) wavelengths carry the small-scale
    wake content the MSL-trained decoders smooth out.

    For our non-periodic, airfoil-masked domain the spectrum is
    restricted to the wake ROI (rows [row_lo, row_hi) span
    ``(wake_x_min, wake_x_max)`` along the streamwise H axis; columns
    [col_lo, col_hi) span ``(-wake_y_max, wake_y_max)`` along the
    cross-stream W axis -- canonical convention from :func:`wake_mask`).
    A 2D Hann window is applied before FFT to suppress edge artifacts.

    Agreement metrics returned:

    - ``contour_iou``: per-level intersection-over-union of the
      ``E_premult / E_premult.max() >= level`` masks for pred and
      target. 1.0 means the contour regions coincide perfectly.
    - ``median_wavelength_ratio``: per-level ``max(med_pred / med_target,
      med_target / med_pred)`` where ``med_X`` is the median wavelength
      ``2*pi/|k|`` inside the contour mask of X. 1.0 means perfect
      wavelength agreement; the PRF paper's "within factor 2" rule
      corresponds to <= 2.0.
    - ``max_wavelength_ratio``: ``max`` of ``median_wavelength_ratio``
      across the requested contour levels. The Session 12 success
      criterion uses this scalar.
    - ``premult_pred`` / ``premult_target``: the 2D premultiplied PSD
      fields (shifted so kx=0, ky=0 is at the centre).
    - ``kx_grid`` / ``ky_grid``: physical-unit wavenumber grids.

    Args:
        pred, target: ``(H, W)`` raw-scale omega fields (single frame).
            For multiple frames, average :func:`wake_2d_premult_spectrum`
            outputs across frames before reporting the headline ratio.
        physical_extent: ``(x_min, x_max, y_min, y_max)`` of the full
            field; defaults match partition v1.
        wake_x_min, wake_x_max, wake_y_max: Wake ROI in physical units.
        window: ``"hann"`` (default) or ``None``.
        contour_levels: Fractional levels of the normalised E_premult.

    Returns:
        Dict with the keys described above.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"shape mismatch: pred {pred.shape} vs target {target.shape}"
        )
    if pred.ndim != 2:
        raise ValueError(
            f"expected 2D (H, W) inputs; got pred shape {pred.shape}"
        )
    H, W = pred.shape

    x_min, x_max, y_min, y_max = physical_extent
    row_lo = int(round((wake_x_min - x_min) / max(x_max - x_min, 1e-12) * H))
    row_hi = int(round((wake_x_max - x_min) / max(x_max - x_min, 1e-12) * H))
    col_lo = int(round((-wake_y_max - y_min) / max(y_max - y_min, 1e-12) * W))
    col_hi = int(round((wake_y_max - y_min) / max(y_max - y_min, 1e-12) * W))
    row_lo = max(0, min(row_lo, H))
    row_hi = max(row_lo + 1, min(row_hi, H))
    col_lo = max(0, min(col_lo, W))
    col_hi = max(col_lo + 1, min(col_hi, W))
    pred_w = pred[row_lo:row_hi, col_lo:col_hi].astype(np.float64)
    tgt_w = target[row_lo:row_hi, col_lo:col_hi].astype(np.float64)

    h, w = pred_w.shape

    if window == "hann":
        win = _hann_window_2d_np(h, w)
        pred_w = pred_w * win
        tgt_w = tgt_w * win
    elif window is None:
        pass
    else:
        raise ValueError(f"unknown window {window!r}; use 'hann' or None")

    f_pred = np.fft.fftshift(np.fft.fft2(pred_w))
    f_target = np.fft.fftshift(np.fft.fft2(tgt_w))
    psd_pred = (np.abs(f_pred) ** 2) / (h * w)
    psd_target = (np.abs(f_target) ** 2) / (h * w)

    dx = (x_max - x_min) / max(H - 1, 1)
    dy = (y_max - y_min) / max(W - 1, 1)
    kx = np.fft.fftshift(np.fft.fftfreq(h, d=dx)) * 2 * math.pi
    ky = np.fft.fftshift(np.fft.fftfreq(w, d=dy)) * 2 * math.pi
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    abs_kx = np.abs(kx_grid)
    abs_ky = np.abs(ky_grid)

    premult_pred = abs_kx * abs_ky * psd_pred
    premult_target = abs_kx * abs_ky * psd_target

    pp_norm = premult_pred / max(premult_pred.max(), 1e-30)
    pt_norm = premult_target / max(premult_target.max(), 1e-30)

    contour_iou = np.zeros(len(contour_levels), dtype=np.float64)
    wavelength_ratio = np.full(len(contour_levels), np.nan, dtype=np.float64)
    k_mag = np.sqrt(kx_grid ** 2 + ky_grid ** 2)
    valid_k = k_mag > 1e-12

    for i, level in enumerate(contour_levels):
        pred_region = pp_norm >= level
        tgt_region = pt_norm >= level
        if pred_region.any() or tgt_region.any():
            inter = float((pred_region & tgt_region).sum())
            union = float((pred_region | tgt_region).sum())
            contour_iou[i] = inter / max(union, 1.0)
        else:
            contour_iou[i] = 1.0  # both empty -> trivially aligned

        pred_mask_k = pred_region & valid_k
        tgt_mask_k = tgt_region & valid_k
        if pred_mask_k.any() and tgt_mask_k.any():
            lam_pred = 2 * math.pi / k_mag[pred_mask_k]
            lam_target = 2 * math.pi / k_mag[tgt_mask_k]
            med_p = float(np.median(lam_pred))
            med_t = float(np.median(lam_target))
            if med_t > 0 and med_p > 0:
                ratio = med_p / med_t
                wavelength_ratio[i] = max(ratio, 1.0 / ratio)

    finite_ratio = wavelength_ratio[np.isfinite(wavelength_ratio)]
    max_ratio = float(np.max(finite_ratio)) if finite_ratio.size else float("nan")
    mean_iou = float(np.mean(contour_iou)) if contour_iou.size else float("nan")

    return {
        "premult_pred": premult_pred,
        "premult_target": premult_target,
        "kx_grid": kx_grid,
        "ky_grid": ky_grid,
        "contour_levels": np.asarray(contour_levels, dtype=np.float64),
        "contour_iou": contour_iou,
        "median_wavelength_ratio": wavelength_ratio,
        "max_wavelength_ratio": max_ratio,
        "mean_contour_iou": mean_iou,
        "wake_patch_shape": (h, w),
    }


def wake_2d_premult_spectrum_series(
    pred: np.ndarray,
    target: np.ndarray,
    physical_extent: tuple[float, float, float, float] = (-1.5, 4.5, -1.5, 1.5),
    wake_x_min: float = 0.0,
    wake_x_max: float = 4.5,
    wake_y_max: float = 1.25,
    window: Optional[str] = "hann",
    contour_levels: tuple[float, ...] = (0.10, 0.50, 0.90),
) -> dict:
    """Wake 2D premultiplied power spectrum averaged across frames.

    Args:
        pred, target: ``(T, H, W)`` arrays of single-channel omega over
            a time series (per-encounter or per-batch).

    Computes :func:`wake_2d_premult_spectrum` per frame and averages
    the premultiplied PSD fields, then re-derives the agreement
    metrics on the time-averaged spectra. This is the PRF 2026
    methodology (their Figs. 5-6 are computed on test-set time-averaged
    spectra) and the headline number for the paper.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"shape mismatch: pred {pred.shape} vs target {target.shape}"
        )
    if pred.ndim != 3:
        raise ValueError(
            f"expected (T, H, W) inputs; got pred shape {pred.shape}"
        )
    T = pred.shape[0]
    psd_pred_avg = None
    psd_target_avg = None
    kx_grid = None
    ky_grid = None
    for t in range(T):
        out = wake_2d_premult_spectrum(
            pred[t], target[t],
            physical_extent=physical_extent,
            wake_x_min=wake_x_min, wake_x_max=wake_x_max,
            wake_y_max=wake_y_max,
            window=window, contour_levels=contour_levels,
        )
        if psd_pred_avg is None:
            psd_pred_avg = out["premult_pred"].copy()
            psd_target_avg = out["premult_target"].copy()
            kx_grid = out["kx_grid"]
            ky_grid = out["ky_grid"]
        else:
            psd_pred_avg += out["premult_pred"]
            psd_target_avg += out["premult_target"]
    psd_pred_avg /= T
    psd_target_avg /= T

    pp_norm = psd_pred_avg / max(psd_pred_avg.max(), 1e-30)
    pt_norm = psd_target_avg / max(psd_target_avg.max(), 1e-30)

    k_mag = np.sqrt(kx_grid ** 2 + ky_grid ** 2)
    valid_k = k_mag > 1e-12

    contour_iou = np.zeros(len(contour_levels), dtype=np.float64)
    wavelength_ratio = np.full(len(contour_levels), np.nan, dtype=np.float64)
    for i, level in enumerate(contour_levels):
        pred_region = pp_norm >= level
        tgt_region = pt_norm >= level
        if pred_region.any() or tgt_region.any():
            inter = float((pred_region & tgt_region).sum())
            union = float((pred_region | tgt_region).sum())
            contour_iou[i] = inter / max(union, 1.0)
        else:
            contour_iou[i] = 1.0
        pred_mask_k = pred_region & valid_k
        tgt_mask_k = tgt_region & valid_k
        if pred_mask_k.any() and tgt_mask_k.any():
            lam_pred = 2 * math.pi / k_mag[pred_mask_k]
            lam_target = 2 * math.pi / k_mag[tgt_mask_k]
            med_p = float(np.median(lam_pred))
            med_t = float(np.median(lam_target))
            if med_t > 0 and med_p > 0:
                ratio = med_p / med_t
                wavelength_ratio[i] = max(ratio, 1.0 / ratio)

    finite_ratio = wavelength_ratio[np.isfinite(wavelength_ratio)]
    max_ratio = float(np.max(finite_ratio)) if finite_ratio.size else float("nan")
    mean_iou = float(np.mean(contour_iou)) if contour_iou.size else float("nan")

    return {
        "premult_pred": psd_pred_avg,
        "premult_target": psd_target_avg,
        "kx_grid": kx_grid,
        "ky_grid": ky_grid,
        "contour_levels": np.asarray(contour_levels, dtype=np.float64),
        "contour_iou": contour_iou,
        "median_wavelength_ratio": wavelength_ratio,
        "max_wavelength_ratio": max_ratio,
        "mean_contour_iou": mean_iou,
        "n_frames": T,
    }
