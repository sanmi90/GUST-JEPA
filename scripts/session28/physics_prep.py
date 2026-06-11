#!/usr/bin/env python
"""Model-free physics prep for Tracks P1 / P2 / P3 (Session 28 master plan, v2.1 UNCOND).

Computes, per encounter over the v2p1 splits (train, val, test_b, test_c), using ONLY the
preprocessed cache (omega_z, C_L) and the split manifest (NO trained model, NO torch, NO GPU):

P3 coverage audit (impact phase)
    Shedding phase at the impact frame, phi_imp in [0, 2pi), from the Hilbert transform
    (scipy.signal.hilbert) of the encounter's OWN pre-impact C_L (frames 0..impact-1,
    ~40 frames). Filter documentation: the slow lift modulation (St ~ 0.044, period
    ~455 frames, measured in outputs/session28/undisturbed_stats.json) is locally LINEAR
    over the 40-frame pre-impact window, so we remove it with a least-squares linear
    detrend before the Hilbert transform. A band-pass at the carrier (St ~ 0.675, period
    ~29.6 frames, i.e. only ~1.35 carrier cycles inside the window) would ring at the
    window edges worse than detrend+Hilbert, so band-passing is deliberately NOT used.
    Hilbert edge handling: the unwrapped phase is fit linearly over the interior of the
    window (15 percent trimmed at each edge) and extrapolated to the impact frame; the
    fit R^2 and the fitted period are stored as quality diagnostics (re-release
    encounters recovering from the previous gust can have a corrupted pre-impact cycle).
    The direct end-sample Hilbert phase propagated one frame is stored as
    phi_imp_endpoint for robustness comparison. The audit reports occupied-bin coverage
    (12 bins), arc coverage (1 - largest gap / 2pi), and circular statistics, pooled,
    per split, and per encounter index; POOR coverage is expected because gusts release
    on a fixed 120-frame cadence (~2.14 subharmonic shedding periods, see HANDOFF D148).

P1 response amplitudes (physical side only; NO fitting, NO collapse scoring here)
    (i) dcl_peak_phase_matched: peak |Delta C_L| within frames [impact, impact+40], where
    the reference is the encounter's OWN pre-impact cycle extrapolated as a
    single-harmonic phase-matched model: C_L_ref(t) = trend(impact) + A cos(phi_fit(t)),
    with phi_fit the linear Hilbert-phase fit above, A the median Hilbert envelope over
    the window interior, and the linear trend FROZEN at its impact-frame value (the slow
    modulation is not extrapolated to avoid overshoot over the 2 t/c response window).
    Known limitation: the subharmonic line (St ~ 0.34, amplitude ~1/4 of the carrier) is
    not modeled by the single harmonic; the simple variant brackets it from the other
    side. (ii) dcl_peak_simple: peak |C_L - mean(pre-impact C_L)| over the same window
    (conflates the undisturbed oscillation amplitude with the gust response; for weak
    gusts it floors at the pre-impact oscillation amplitude, ~0.18 for the baseline).
    (iii) Peak large-scale wake-enstrophy excursion: omega_z Gaussian-filtered per frame
    at sigma/c = 0.05 (= 1.6 px at dx = dy = 0.03125 c; scipy.ndimage.gaussian_filter,
    sigma = (0, 1.6, 1.6) on the (T, x, y) stack), enstrophy integrated over the wake
    region x in [0.5, 4], |y| <= 1 (E(t) = sum omega_f^2 dA, RAW physical omega_z, not
    the pipeline-normalised field), excursion measured against the encounter's own
    pre-impact mean. Both the [impact, impact+40] peak and the full post-impact peak are
    stored.

P2 physical recovery clock tau_rec
    First post-impact frame at which the large-scale wake enstrophy re-enters the
    pre-impact envelope (mean +- 2 sd of the encounter's OWN pre-impact large-scale wake
    enstrophy, frames 0..impact-1) AND stays inside for >= one full shedding period
    (default dwell 56 frames = the subharmonic clock, parameterized). Encounters that do
    not recover within the 120-frame window are CENSORED and reported as such (no silent
    exclusion). A third status, reentered_unconfirmed, marks encounters whose enstrophy
    re-enters and stays inside until the record ends but for fewer than dwell frames
    (the dwell cannot be certified); these count as censored in the strict fractions and
    are reported separately. tau_rec is stored both as the absolute frame index and as
    (frame - impact) * dt_tc in convective time.

Pre-registered scaling candidates (per encounter; values are case-level functions of
(G, D, Y) repeated per encounter)
    s1 = G                       (Kussner-like gust ratio)
    s2 = G * D                   (proportional to the Taylor-profile gust circulation)
    s3 = Gamma_g(G, D) / (u_inf c), the Taylor-vortex CORE circulation integrated
         numerically from the implemented gust profile. Profile source: the DNS (SOD2D,
         PREVENT campaign) reproduce the configuration of Fukami, Smith and Taira, Phys.
         Rev. Fluids 10, 084703 (2025), whose Eq. (1) defines the gust as a Taylor
         vortex (Taylor 1918) with rotational velocity profile
             u_theta(r) = u_theta,max * (r / R) * exp((1 - r^2 / R^2) / 2),
         where R is the radius of maximum tangential velocity, u_theta,max = G u_inf and
         D = 2R/c. Confirmed against the PRF text at /tmp/fukami_prf.txt (Sec. II,
         Eq. (1)); the inventory parser (data_manifest/raw_cases_inventory.yaml) encodes
         only (G, D, Y), consistent with this two-parameter profile. The Taylor vortex
         is shielded (net circulation -> 0 as r -> infinity), so the meaningful strength
         measure is the core circulation: Gamma_g = 2 pi * integral_0^{r0} omega(r) r dr
         up to the first zero crossing of the vorticity (r0 = sqrt(2) R analytically),
         evaluated here by numerical integration of the profile (analytic value
         2 pi e^{-1/2} G D ~= 3.8104 G D, used only as a cross-check in the tests; s3 is
         exactly proportional to s2, with the honest prefactor, as the master plan
         anticipates). If the profile could not have been pinned, s3 would be NaN with a
         note; it IS pinned, so no NaN is emitted.
    s4 = the Martinez-Muriel and Flores (2020) induced-vertical-velocity ratio: the
         maximum vertical velocity induced by the vortex over the airfoil chord, divided
         by u_inf, computed from the SAME Taylor profile at miss distance |Y|.
         Assumptions (documented, model-free): the chord is idealised as the horizontal
         line the vortex path passes at vertical offset Y (flat-chord approximation; the
         14 deg incidence geometry is ignored), and the vortex is frozen while sweeping
         past, so the maximum over time and chord position equals the maximum over
         horizontal separation dx of v(dx) = u_theta(r) * dx / r with r =
         sqrt(dx^2 + Y^2). Signed by G; at Y = 0 it reduces to s4 = G exactly.

Outputs
    outputs/session28/physics/per_encounter_physics.npz   (one row per encounter)
    outputs/session28/physics/physics_summary.json        (P3 coverage, P2 fractions)

CLI
    nice -n 10 python scripts/session28/physics_prep.py \\
        --split-manifest configs/splits/split_v2p1.json \\
        --output-dir outputs/session28/physics

All heavy work is plain numpy/scipy on CPU; ~382 encounters take a few minutes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import gaussian_filter
from scipy.signal import hilbert

# --------------------------------------------------------------------------------------
# Grid and clock constants (frozen; see CLAUDE.md "Dataset layout").
# --------------------------------------------------------------------------------------
DT_TC = 0.05
NX, NY = 192, 96
X_MIN, X_MAX = -1.5, 4.5
Y_MIN, Y_MAX = -1.5, 1.5
DX = (X_MAX - X_MIN) / NX  # 0.03125 c
DY = (Y_MAX - Y_MIN) / NY  # 0.03125 c
CELL_AREA = DX * DY

WAKE_X = (0.5, 4.0)
WAKE_ABS_Y = 1.0

SIGMA_C_DEFAULT = 0.05  # Gaussian filter scale in chord units
DWELL_FRAMES_DEFAULT = 56  # one full shedding period at the subharmonic clock (St ~ 0.34)
ENVELOPE_NSD_DEFAULT = 2.0
RESPONSE_WINDOW_DEFAULT = 40  # frames after impact for the P1 response peaks
EDGE_TRIM_FRAC_DEFAULT = 0.15
PHASE_R2_MIN_DEFAULT = 0.90  # "clean pre-impact cycle" threshold for the coverage audit
PHASE_AMP_MAX_DEFAULT = 0.45  # ~2.5x the baseline pre-impact lift amplitude (0.04-0.19)
N_PHASE_BINS_DEFAULT = 12

TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------------------
# P3: pre-impact Hilbert phase
# --------------------------------------------------------------------------------------
@dataclass
class PreImpactCycleFit:
    """Single-harmonic description of the pre-impact C_L cycle.

    Attributes:
        n_pre: number of pre-impact frames used.
        trend_intercept, trend_slope: linear trend (slow-modulation proxy) in frame units.
        omega_rad_per_frame: fitted phase rate from the interior linear phase fit.
        phi_intercept: fitted phase at frame 0 (radians, unwrapped convention).
        phase_fit_r2: R^2 of the linear fit to the unwrapped Hilbert phase (interior).
        period_frames: 2 pi / omega (NaN if omega <= 0).
        amplitude: median Hilbert envelope over the window interior.
        phi_imp: phase extrapolated to the impact frame, in [0, 2pi).
        phi_imp_endpoint: direct end-sample Hilbert phase propagated one frame, [0, 2pi).
        pre_mean: mean of the raw pre-impact C_L.
    """

    n_pre: int
    trend_intercept: float
    trend_slope: float
    omega_rad_per_frame: float
    phi_intercept: float
    phase_fit_r2: float
    period_frames: float
    amplitude: float
    phi_imp: float
    phi_imp_endpoint: float
    pre_mean: float


def fit_pre_impact_cycle(
    cl_pre: np.ndarray, edge_trim_frac: float = EDGE_TRIM_FRAC_DEFAULT
) -> PreImpactCycleFit:
    """Fit the pre-impact lift cycle: linear detrend + Hilbert + interior linear phase fit.

    The linear detrend removes the slow modulation (St ~ 0.044; locally linear over the
    ~40-frame window). The unwrapped Hilbert phase is fit linearly over the interior
    (edge_trim_frac trimmed at each end) to suppress Hilbert edge artifacts, then
    extrapolated to the impact frame (= len(cl_pre), one past the window).
    """
    cl_pre = np.asarray(cl_pre, dtype=np.float64)
    n = cl_pre.size
    if n < 8:
        nan = float("nan")
        return PreImpactCycleFit(n, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan)
    t = np.arange(n, dtype=np.float64)
    trend_slope, trend_intercept = np.polyfit(t, cl_pre, 1)
    detrended = cl_pre - (trend_intercept + trend_slope * t)

    analytic = hilbert(detrended)
    phase = np.unwrap(np.angle(analytic))
    envelope = np.abs(analytic)

    trim = max(2, int(round(edge_trim_frac * n)))
    interior = slice(trim, n - trim)
    omega, phi0 = np.polyfit(t[interior], phase[interior], 1)
    resid = phase[interior] - (phi0 + omega * t[interior])
    ss_tot = float(np.sum((phase[interior] - phase[interior].mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0 else float("nan")

    period = TWO_PI / omega if omega > 0 else float("nan")
    phi_imp = float((phi0 + omega * n) % TWO_PI)
    phi_imp_endpoint = float((phase[-1] + omega) % TWO_PI)
    return PreImpactCycleFit(
        n_pre=n,
        trend_intercept=float(trend_intercept),
        trend_slope=float(trend_slope),
        omega_rad_per_frame=float(omega),
        phi_intercept=float(phi0),
        phase_fit_r2=float(r2),
        period_frames=float(period),
        amplitude=float(np.median(envelope[interior])),
        phi_imp=phi_imp,
        phi_imp_endpoint=phi_imp_endpoint,
        pre_mean=float(cl_pre.mean()),
    )


# --------------------------------------------------------------------------------------
# P1: response amplitudes
# --------------------------------------------------------------------------------------
def phase_matched_reference(fit: PreImpactCycleFit, frames: np.ndarray) -> np.ndarray:
    """Extrapolate the pre-impact cycle to `frames` as a single-harmonic phase-matched model.

    C_L_ref(t) = trend(impact) + A cos(phi0 + omega t), with the linear trend FROZEN at
    its value at the impact frame (= fit.n_pre) so the slow modulation is not
    extrapolated across the response window.
    """
    frames = np.asarray(frames, dtype=np.float64)
    trend_at_impact = fit.trend_intercept + fit.trend_slope * fit.n_pre
    return trend_at_impact + fit.amplitude * np.cos(
        fit.phi_intercept + fit.omega_rad_per_frame * frames
    )


def delta_cl_peaks(
    cl: np.ndarray, impact: int, fit: PreImpactCycleFit, window: int = RESPONSE_WINDOW_DEFAULT
) -> tuple[float, float]:
    """Peak |Delta C_L| in [impact, impact+window]: phase-matched and simple variants."""
    cl = np.asarray(cl, dtype=np.float64)
    t_end = min(impact + window, cl.size - 1)
    frames = np.arange(impact, t_end + 1)
    ref = phase_matched_reference(fit, frames)
    pm = float(np.max(np.abs(cl[impact : t_end + 1] - ref)))
    simple = float(np.max(np.abs(cl[impact : t_end + 1] - fit.pre_mean)))
    return pm, simple


def wake_mask() -> np.ndarray:
    """Boolean (NX, NY) mask of the wake region x in [0.5, 4], |y| <= 1 (cell centers)."""
    xc = X_MIN + (np.arange(NX) + 0.5) * DX
    yc = Y_MIN + (np.arange(NY) + 0.5) * DY
    return ((xc >= WAKE_X[0]) & (xc <= WAKE_X[1]))[:, None] & (np.abs(yc) <= WAKE_ABS_Y)[None, :]


def large_scale_wake_enstrophy(
    omega: np.ndarray, sigma_px: float, mask: np.ndarray | None = None
) -> np.ndarray:
    """Per-frame large-scale wake enstrophy E(t) = sum_wake omega_f(t)^2 dA.

    omega is the RAW (T, NX, NY) omega_z stack; the Gaussian filter (sigma_px in pixels,
    sigma/c = 0.05 -> 1.6 px at dx = 0.03125 c) acts per frame in the two spatial axes.
    """
    if mask is None:
        mask = wake_mask()
    om = gaussian_filter(np.asarray(omega, dtype=np.float64), sigma=(0.0, sigma_px, sigma_px))
    return (om**2 * mask[None, :, :]).sum(axis=(1, 2)) * CELL_AREA


# --------------------------------------------------------------------------------------
# P2: recovery clock
# --------------------------------------------------------------------------------------
def envelope_bounds(pre: np.ndarray, nsd: float = ENVELOPE_NSD_DEFAULT) -> tuple[float, float]:
    """(lo, hi) = mean -+ nsd * sd (ddof=1) of the encounter's own pre-impact trace."""
    pre = np.asarray(pre, dtype=np.float64)
    m = float(pre.mean())
    s = float(pre.std(ddof=1))
    return m - nsd * s, m + nsd * s


def first_sustained_reentry(
    trace: np.ndarray, start: int, lo: float, hi: float, dwell: int
) -> tuple[str, int | None]:
    """First t* >= start with trace inside [lo, hi] for >= dwell consecutive frames.

    Returns (status, tau):
        ("recovered", t*)               a full dwell run was observed starting at t*;
        ("reentered_unconfirmed", t*)   the trace re-enters at t* and stays inside until
                                        the record ends, but fewer than dwell frames were
                                        observed (dwell cannot be certified);
        ("censored", None)              no qualifying run.
    """
    trace = np.asarray(trace, dtype=np.float64)
    n = trace.size
    inside = (trace >= lo) & (trace <= hi)
    unconfirmed: int | None = None
    t = start
    while t < n:
        if not inside[t]:
            t += 1
            continue
        e = t
        while e < n and inside[e]:
            e += 1
        if e - t >= dwell:
            return "recovered", t
        if e == n and unconfirmed is None:
            unconfirmed = t
        t = e + 1
    if unconfirmed is not None:
        return "reentered_unconfirmed", unconfirmed
    return "censored", None


# --------------------------------------------------------------------------------------
# Pre-registered scaling candidates
# --------------------------------------------------------------------------------------
def taylor_u_theta(r: np.ndarray, G: float, D: float) -> np.ndarray:
    """Taylor-vortex tangential velocity, Fukami et al. PRF 2025 Eq. (1); u_inf = c = 1.

    u_theta(r) = G (r/R) exp((1 - r^2/R^2)/2) with R = D/2; peak velocity G at r = R.
    """
    R = 0.5 * D
    rr = np.asarray(r, dtype=np.float64) / R
    return G * rr * np.exp(0.5 * (1.0 - rr**2))


def taylor_circulation(G: float, D: float, n: int = 20001, r_max_over_r0: float = 8.0) -> float:
    """Core circulation Gamma_g / (u_inf c), integrated numerically from the profile.

    Gamma(r) = 2 pi integral_0^r omega(r') r' dr' with omega r = d(r u_theta)/dr, so the
    integrand is q(r) = d(r u_theta)/dr evaluated by numerical differentiation; the core
    circulation is the signed extremum of the cumulative integral (reached at the first
    vorticity zero crossing, r0 = sqrt(2) R analytically). The Taylor vortex is shielded
    (Gamma -> 0 as r -> inf), so the extremum, not the total, is the strength measure.
    Analytic cross-check: Gamma_g = 2 pi e^{-1/2} G D.
    """
    if G == 0.0 or D == 0.0:
        return 0.0
    R = 0.5 * D
    r = np.linspace(0.0, r_max_over_r0 * R, n)
    q = np.gradient(r * taylor_u_theta(r, G, D), r)
    gamma = TWO_PI * cumulative_trapezoid(q, r, initial=0.0)
    return float(gamma[np.argmax(np.abs(gamma))])


def mmf_velocity_ratio(G: float, D: float, Y: float, n: int = 20001) -> float:
    """Martinez-Muriel and Flores (2020) induced-vertical-velocity ratio (signed by G).

    Maximum vertical velocity induced by the Taylor vortex over the airfoil chord,
    divided by u_inf, at miss distance |Y|: extremum over horizontal separation dx of
    v(dx) = u_theta(r) dx / r, r = sqrt(dx^2 + Y^2). Flat-chord, frozen-vortex
    approximation (see module docstring). At Y = 0 this reduces to G exactly.
    """
    if G == 0.0 or D == 0.0:
        return 0.0
    R = 0.5 * D
    dx = np.linspace(0.0, 10.0 * R + 5.0 * abs(Y), n)
    r = np.hypot(dx, Y)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.where(r > 0, taylor_u_theta(r, G, D) * dx / np.maximum(r, 1e-300), 0.0)
    return float(v[np.argmax(np.abs(v))])


# --------------------------------------------------------------------------------------
# Per-encounter driver
# --------------------------------------------------------------------------------------
@dataclass
class PrepParams:
    sigma_c: float = SIGMA_C_DEFAULT
    dwell_frames: int = DWELL_FRAMES_DEFAULT
    envelope_nsd: float = ENVELOPE_NSD_DEFAULT
    response_window: int = RESPONSE_WINDOW_DEFAULT
    edge_trim_frac: float = EDGE_TRIM_FRAC_DEFAULT
    phase_r2_min: float = PHASE_R2_MIN_DEFAULT
    phase_amp_max: float = PHASE_AMP_MAX_DEFAULT
    n_phase_bins: int = N_PHASE_BINS_DEFAULT

    @property
    def sigma_px(self) -> float:
        return self.sigma_c / DX


def process_encounter(
    h5_path: Path | str,
    split: str,
    params: PrepParams,
    mask: np.ndarray | None = None,
) -> dict:
    """Compute the full per-encounter physics row from one cache file."""
    import h5py  # local import keeps the synthetic-data unit tests free of h5py

    if mask is None:
        mask = wake_mask()
    with h5py.File(h5_path, "r") as f:
        cl = f["C_L"][:].astype(np.float64)
        omega = f["omega_z"][:]
        attrs = dict(f.attrs)
    impact = int(attrs["impact_frame_estimate"])

    fit = fit_pre_impact_cycle(cl[:impact], edge_trim_frac=params.edge_trim_frac)
    dcl_pm, dcl_simple = delta_cl_peaks(cl, impact, fit, window=params.response_window)

    ens = large_scale_wake_enstrophy(omega, params.sigma_px, mask)
    pre = ens[:impact]
    pre_mean = float(pre.mean())
    pre_sd = float(pre.std(ddof=1))
    lo, hi = envelope_bounds(pre, params.envelope_nsd)

    d_ens = ens - pre_mean
    w_end = min(impact + params.response_window, ens.size - 1)
    den_imp_win = float(np.max(np.abs(d_ens[impact : w_end + 1])))
    post_abs = np.abs(d_ens[impact:])
    i_pk = int(np.argmax(post_abs))
    den_post = float(post_abs[i_pk])
    den_post_signed = float(d_ens[impact:][i_pk])

    status, tau = first_sustained_reentry(ens, impact, lo, hi, params.dwell_frames)
    tau_frame = float("nan") if tau is None else float(tau)
    tau_tc = float("nan") if tau is None else (tau - impact) * DT_TC

    G = float(attrs["G"])
    D = float(attrs["D"])
    Y = float(attrs["Y"])
    return {
        "case_id": str(attrs["case_id"]),
        "encounter_index": int(attrs["encounter_index"]),
        "split": split,
        "source_group": str(attrs.get("source_group", "")),
        "G": G,
        "D": D,
        "Y": Y,
        "impact_frame": impact,
        "phi_imp": fit.phi_imp,
        "phi_imp_endpoint": fit.phi_imp_endpoint,
        "phase_fit_r2": fit.phase_fit_r2,
        "phase_period_frames": fit.period_frames,
        "pre_cl_mean": fit.pre_mean,
        "pre_cl_amp": fit.amplitude,
        "dcl_peak_phase_matched": dcl_pm,
        "dcl_peak_simple": dcl_simple,
        "enstrophy_pre_mean": pre_mean,
        "enstrophy_pre_sd": pre_sd,
        "denstrophy_peak_imp40": den_imp_win,
        "denstrophy_peak_post": den_post,
        "denstrophy_peak_post_signed": den_post_signed,
        "tau_rec_frame": tau_frame,
        "tau_rec_tc": tau_tc,
        "recovery_status": status,
        "censored": status != "recovered",
        "s1": G,
        "s2": G * D,
        "s3": taylor_circulation(G, D),
        "s4": mmf_velocity_ratio(G, D, Y),
    }


def iter_encounters(manifest: dict) -> Iterator[tuple[str, int, str]]:
    """Yield (case_id, encounter_index, split) for every encounter in the v2p1 splits.

    Train cases: train_encounter_indices -> "train", val_encounter_indices -> "val".
    test_b / test_c cases: every listed encounter belongs to the case split.
    """
    for case_id in sorted(manifest["cases"]):
        case = manifest["cases"][case_id]
        if case["split"] == "train":
            for k in case["train_encounter_indices"]:
                yield case_id, int(k), "train"
            for k in case["val_encounter_indices"]:
                yield case_id, int(k), "val"
        else:
            for k in case["train_encounter_indices"] + case["val_encounter_indices"]:
                yield case_id, int(k), case["split"]


# --------------------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------------------
def circular_stats(phi: np.ndarray) -> dict:
    """Circular mean, resultant length R, and circular std of phases in radians."""
    phi = np.asarray(phi, dtype=np.float64)
    phi = phi[np.isfinite(phi)]
    if phi.size == 0:
        return {"n": 0, "mean": float("nan"), "R": float("nan"), "circ_std": float("nan")}
    z = np.exp(1j * phi).mean()
    r = float(np.abs(z))
    return {
        "n": int(phi.size),
        "mean": float(np.angle(z) % TWO_PI),
        "R": r,
        "circ_std": float(math.sqrt(-2.0 * math.log(r))) if r > 0 else float("inf"),
    }


def coverage_stats(phi: np.ndarray, n_bins: int = N_PHASE_BINS_DEFAULT) -> dict:
    """Cycle-coverage measures for a set of impact phases in [0, 2pi).

    occupied_bin_fraction: fraction of n_bins phase bins containing >= 1 encounter.
    arc_coverage_fraction: 1 - (largest empty angular gap) / 2pi.
    """
    phi = np.asarray(phi, dtype=np.float64)
    phi = np.sort(phi[np.isfinite(phi)] % TWO_PI)
    base = circular_stats(phi)
    if phi.size == 0:
        return {
            **base,
            "occupied_bin_fraction": 0.0,
            "arc_coverage_fraction": 0.0,
            "n_bins": n_bins,
            "occupied_bins": 0,
            "largest_gap_rad": TWO_PI,
        }
    bin_counts = np.bincount(
        np.floor(phi / TWO_PI * n_bins).astype(int).clip(0, n_bins - 1), minlength=n_bins
    )
    occupied = int((bin_counts > 0).sum())
    if phi.size == 1:
        largest_gap = TWO_PI
    else:
        gaps = np.diff(phi)
        wrap_gap = TWO_PI - (phi[-1] - phi[0])
        largest_gap = float(max(gaps.max(), wrap_gap))
    return {
        **base,
        "n_bins": n_bins,
        "bin_counts": bin_counts.tolist(),
        "occupied_bins": occupied,
        "occupied_bin_fraction": occupied / n_bins,
        "largest_gap_rad": largest_gap,
        "arc_coverage_fraction": 1.0 - largest_gap / TWO_PI,
    }


def _circ_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = (np.asarray(a) - np.asarray(b)) % TWO_PI
    return np.minimum(d, TWO_PI - d)


def _recovery_block(rows: list[dict]) -> dict:
    n = len(rows)
    rec = [r for r in rows if r["recovery_status"] == "recovered"]
    unc = [r for r in rows if r["recovery_status"] == "reentered_unconfirmed"]
    cen = [r for r in rows if r["recovery_status"] == "censored"]
    taus = np.array([r["tau_rec_tc"] for r in rec], dtype=np.float64)
    return {
        "n": n,
        "recovered": len(rec),
        "reentered_unconfirmed": len(unc),
        "censored": len(cen),
        "frac_recovered": len(rec) / n if n else float("nan"),
        "frac_censored_strict": (len(unc) + len(cen)) / n if n else float("nan"),
        "tau_rec_tc_median_recovered": float(np.median(taus)) if taus.size else float("nan"),
        "tau_rec_tc_max_recovered": float(np.max(taus)) if taus.size else float("nan"),
    }


def build_summary(rows: list[dict], params: PrepParams, manifest_path: str) -> dict:
    splits = sorted({r["split"] for r in rows})
    phi_all = np.array([r["phi_imp"] for r in rows])
    r2_all = np.array([r["phase_fit_r2"] for r in rows])
    amp_all = np.array([r["pre_cl_amp"] for r in rows])
    clean = r2_all >= params.phase_r2_min
    clean_key = f"clean_r2_ge_{params.phase_r2_min:.2f}"
    # Pre-impact windows of strong gusts are contaminated by the incoming vortex's
    # induced-lift ramp; the Hilbert envelope then far exceeds the baseline oscillation
    # amplitude (~0.1-0.2), so phi_imp is no longer a shedding phase. The amp-gated
    # subset is the honest coverage measure for the PRC ambition.
    clean_amp = clean & (amp_all <= params.phase_amp_max)
    clean_amp_key = f"clean_amp_le_{params.phase_amp_max:.2f}_r2_ge_{params.phase_r2_min:.2f}"

    p3: dict = {
        "pooled": {
            "all": coverage_stats(phi_all, params.n_phase_bins),
            clean_key: coverage_stats(phi_all[clean], params.n_phase_bins),
            clean_amp_key: coverage_stats(phi_all[clean_amp], params.n_phase_bins),
        },
        "per_split": {},
        "per_encounter_index": {},
        "by_release": {},
        "phase_fit_r2_quantiles": {
            q: float(np.nanquantile(r2_all, float(q)))
            for q in ("0.05", "0.25", "0.5", "0.75", "0.95")
        },
        "pre_cl_amp_quantiles": {
            q: float(np.nanquantile(amp_all, float(q)))
            for q in ("0.05", "0.25", "0.5", "0.75", "0.95")
        },
        "phi_imp_vs_endpoint_circdiff_rad": {},
        "phase_period_frames_median": float(np.nanmedian([r["phase_period_frames"] for r in rows])),
    }
    for s in splits:
        m = np.array([r["split"] == s for r in rows])
        p3["per_split"][s] = {
            "all": coverage_stats(phi_all[m], params.n_phase_bins),
            clean_key: coverage_stats(phi_all[m & clean], params.n_phase_bins),
            clean_amp_key: coverage_stats(phi_all[m & clean_amp], params.n_phase_bins),
        }
    enc_idx = np.array([r["encounter_index"] for r in rows])
    for k in sorted(set(enc_idx.tolist())):
        p3["per_encounter_index"][str(k)] = circular_stats(phi_all[enc_idx == k])
    p3["by_release"] = {
        "first_release_all": coverage_stats(phi_all[enc_idx == 0], params.n_phase_bins),
        "first_release_clean_amp": coverage_stats(
            phi_all[clean_amp & (enc_idx == 0)], params.n_phase_bins
        ),
        "re_release_all": coverage_stats(phi_all[enc_idx > 0], params.n_phase_bins),
        "re_release_clean_amp": coverage_stats(
            phi_all[clean_amp & (enc_idx > 0)], params.n_phase_bins
        ),
    }
    dphi = _circ_diff(phi_all, np.array([r["phi_imp_endpoint"] for r in rows]))
    p3["phi_imp_vs_endpoint_circdiff_rad"] = {
        "median": float(np.nanmedian(dphi)),
        "p90": float(np.nanquantile(dphi, 0.9)),
    }

    p2: dict = {"per_split": {}, "by_G": {}, "by_D": {}, "by_absY": {}, "per_case": {}}
    for s in splits:
        p2["per_split"][s] = _recovery_block([r for r in rows if r["split"] == s])
    for key, fn in (
        ("by_G", lambda r: f"{r['G']:+.2f}"),
        ("by_D", lambda r: f"{r['D']:.2f}"),
        ("by_absY", lambda r: f"{abs(r['Y']):.2f}"),
    ):
        for lvl in sorted({fn(r) for r in rows}):
            p2[key][lvl] = _recovery_block([r for r in rows if fn(r) == lvl])
    for cid in sorted({r["case_id"] for r in rows}):
        p2["per_case"][cid] = _recovery_block([r for r in rows if r["case_id"] == cid])
    # The undisturbed Baseline case calibrates the envelope rule: any Baseline encounter
    # not classified "recovered" is a false negative of the frozen rule (the slow lift
    # modulation walks the enstrophy level out of the narrow 40-frame pre-impact band).
    if "Baseline" in p2["per_case"]:
        blk = dict(p2["per_case"]["Baseline"])
        blk["false_negative_fraction"] = 1.0 - blk["frac_recovered"]
        p2["calibration_reference_baseline"] = blk
    # Re-release encounters (k >= 1) of strong gusts have an inflated pre-impact
    # envelope (the previous gust's wake is still transiting the box), which makes
    # immediate "recovery" easier; quantify the envelope width by encounter index.
    p2["pre_envelope_relwidth_by_encounter_index"] = {}
    for k in sorted(set(enc_idx.tolist())):
        rel = np.array(
            [
                r["enstrophy_pre_sd"] / r["enstrophy_pre_mean"]
                for r in rows
                if r["encounter_index"] == k and r["enstrophy_pre_mean"] > 0
            ]
        )
        p2["pre_envelope_relwidth_by_encounter_index"][str(k)] = {
            "median": float(np.median(rel)),
            "p90": float(np.quantile(rel, 0.9)),
        }

    def _split_quants(name: str) -> dict:
        out = {}
        for s in splits:
            v = np.array([r[name] for r in rows if r["split"] == s], dtype=np.float64)
            out[s] = {
                "median": float(np.nanmedian(v)),
                "p90": float(np.nanquantile(v, 0.9)),
                "max": float(np.nanmax(v)),
            }
        return out

    p1 = {
        "dcl_peak_phase_matched": _split_quants("dcl_peak_phase_matched"),
        "dcl_peak_simple": _split_quants("dcl_peak_simple"),
        "denstrophy_peak_post": _split_quants("denstrophy_peak_post"),
    }

    s2 = np.array([r["s2"] for r in rows])
    s3 = np.array([r["s3"] for r in rows])
    nz = s2 != 0
    prefactor = float(np.median(s3[nz] / s2[nz])) if nz.any() else float("nan")
    return {
        "_doc": (
            "Model-free P1/P2/P3 prep (Session 28). Definitions frozen in the master plan; "
            "see scripts/session28/physics_prep.py docstring. No fitting, no collapse "
            "scoring here (gated Phase C work)."
        ),
        "split_manifest": manifest_path,
        "n_encounters": len(rows),
        "params": {
            "sigma_c": params.sigma_c,
            "sigma_px": params.sigma_px,
            "dwell_frames": params.dwell_frames,
            "envelope_nsd": params.envelope_nsd,
            "response_window_frames": params.response_window,
            "edge_trim_frac": params.edge_trim_frac,
            "phase_r2_min": params.phase_r2_min,
            "phase_amp_max": params.phase_amp_max,
            "n_phase_bins": params.n_phase_bins,
            "dt_tc": DT_TC,
            "wake_region": {"x": list(WAKE_X), "abs_y_max": WAKE_ABS_Y},
            "grid": {
                "nx": NX,
                "ny": NY,
                "dx": DX,
                "dy": DY,
                "x_extent": [X_MIN, X_MAX],
                "y_extent": [Y_MIN, Y_MAX],
            },
        },
        "p3_phase_coverage": p3,
        "p2_recovery": p2,
        "p1_response_quantiles": p1,
        "scaling_candidates": {
            "s1": "G",
            "s2": "G * D",
            "s3": "Taylor-vortex core circulation, numeric (Fukami PRF 2025 Eq. (1))",
            "s4": "Martinez-Muriel & Flores (2020) max induced-vertical-velocity ratio",
            "s3_over_s2_prefactor_numeric": prefactor,
            "s3_over_s2_prefactor_analytic": TWO_PI * math.exp(-0.5),
        },
    }


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
NPZ_COLUMNS = [
    "case_id",
    "encounter_index",
    "split",
    "source_group",
    "G",
    "D",
    "Y",
    "impact_frame",
    "phi_imp",
    "phi_imp_endpoint",
    "phase_fit_r2",
    "phase_period_frames",
    "pre_cl_mean",
    "pre_cl_amp",
    "dcl_peak_phase_matched",
    "dcl_peak_simple",
    "enstrophy_pre_mean",
    "enstrophy_pre_sd",
    "denstrophy_peak_imp40",
    "denstrophy_peak_post",
    "denstrophy_peak_post_signed",
    "tau_rec_frame",
    "tau_rec_tc",
    "recovery_status",
    "censored",
    "s1",
    "s2",
    "s3",
    "s4",
]


def default_cache_root() -> Path:
    prevent = os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT"))
    cache = os.environ.get("VORTEX_JEPA_CACHE", str(Path(prevent) / "data/processed/vortex-jepa"))
    return Path(cache) / "v2p1"


def main(argv: list[str] | None = None) -> int:
    repo = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split-manifest", default=str(repo / "configs/splits/split_v2p1.json"))
    ap.add_argument(
        "--cache-root",
        default=str(default_cache_root()),
        help="v2p1 per-encounter cache root (default from PREVENT_ROOT)",
    )
    ap.add_argument("--output-dir", default=str(repo / "outputs/session28/physics"))
    ap.add_argument("--sigma-c", type=float, default=SIGMA_C_DEFAULT)
    ap.add_argument("--dwell-frames", type=int, default=DWELL_FRAMES_DEFAULT)
    ap.add_argument("--envelope-nsd", type=float, default=ENVELOPE_NSD_DEFAULT)
    ap.add_argument("--response-window", type=int, default=RESPONSE_WINDOW_DEFAULT)
    ap.add_argument("--phase-r2-min", type=float, default=PHASE_R2_MIN_DEFAULT)
    ap.add_argument("--phase-amp-max", type=float, default=PHASE_AMP_MAX_DEFAULT)
    ap.add_argument("--limit", type=int, default=0, help="process only the first N encounters")
    args = ap.parse_args(argv)

    params = PrepParams(
        sigma_c=args.sigma_c,
        dwell_frames=args.dwell_frames,
        envelope_nsd=args.envelope_nsd,
        response_window=args.response_window,
        phase_r2_min=args.phase_r2_min,
        phase_amp_max=args.phase_amp_max,
    )
    manifest = json.loads(Path(args.split_manifest).read_text())
    cache_root = Path(args.cache_root)
    if not cache_root.is_dir():
        sys.exit(f"cache root not found: {cache_root} (set PREVENT_ROOT)")

    mask = wake_mask()
    rows: list[dict] = []
    todo = list(iter_encounters(manifest))
    if args.limit:
        todo = todo[: args.limit]
    t0 = time.time()
    for i, (case_id, k, split) in enumerate(todo):
        path = cache_root / case_id / f"encounter_{k:02d}.h5"
        if not path.is_file():
            sys.exit(f"missing cache file: {path}")
        rows.append(process_encounter(path, split, params, mask))
        if (i + 1) % 50 == 0 or i + 1 == len(todo):
            print(
                f"  [{i + 1}/{len(todo)}] {case_id} enc {k} ({time.time() - t0:.0f} s)", flush=True
            )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for col in NPZ_COLUMNS:
        vals = [r[col] for r in rows]
        if col in ("case_id", "split", "source_group", "recovery_status"):
            arrays[col] = np.array(vals)
        elif col in ("encounter_index", "impact_frame"):
            arrays[col] = np.array(vals, dtype=np.int64)
        elif col == "censored":
            arrays[col] = np.array(vals, dtype=bool)
        else:
            arrays[col] = np.array(vals, dtype=np.float64)
    npz_path = out_dir / "per_encounter_physics.npz"
    np.savez(npz_path, **arrays)

    summary = build_summary(rows, params, args.split_manifest)
    json_path = out_dir / "physics_summary.json"
    json_path.write_text(json.dumps(summary, indent=1))

    print(f"\nwrote {npz_path} ({len(rows)} encounters) and {json_path}")
    for key, blk in summary["p3_phase_coverage"]["pooled"].items():
        print(
            f"P3 coverage (pooled, {key}): n={blk['n']} occupied "
            f"{blk['occupied_bins']}/{blk['n_bins']} bins ({blk['occupied_bin_fraction']:.2f}), "
            f"arc {blk['arc_coverage_fraction']:.2f}, R = {blk['R']:.3f}"
        )
    for s, blk in summary["p2_recovery"]["per_split"].items():
        print(
            f"P2 {s}: n={blk['n']} recovered {blk['recovered']} "
            f"({blk['frac_recovered']:.2f}), unconfirmed {blk['reentered_unconfirmed']}, "
            f"censored {blk['censored']}; median tau_rec "
            f"{blk['tau_rec_tc_median_recovered']:.2f} t/c"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
