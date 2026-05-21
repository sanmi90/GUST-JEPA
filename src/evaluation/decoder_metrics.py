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
