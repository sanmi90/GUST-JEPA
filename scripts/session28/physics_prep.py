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

P2 physical recovery clock tau_rec (v1; kept for continuity, superseded by v2 below)
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

P2 recovery clock v2 (HEADLINE rule; 2026-06-11)
    The v1 rule is miscalibrated in both directions: judged against its OWN 40-frame
    pre-impact envelope, the undisturbed Baseline passes only 2/6 (the St ~ 0.044 slow
    modulation walks the enstrophy level out of the narrow band), while strong-gust
    re-releases pass at tau = 0 (their pre-window is inflated by the previous gust's
    transiting wake). v2 replaces the per-encounter reference with a FIXED null band:
    the [q01, q99] interval (widened to [q005, q995] only if the calibration below
    demands it) of the large-scale wake enstrophy over the SETTLED undisturbed Baseline
    record, global raw frame >= 400 (t/c >= 20, HANDOFF D180; 320 frames = the last
    ~2.7 Baseline encounters). The band is phase-agnostic by design: the full shedding
    cycle and the St ~ 0.04 modulation live inside it by construction. Recovery:
    tau_rec_v2 = first post-impact frame t* that is itself inside the band AND whose
    following W = 56-frame window [t*, t* + W) has occupancy (fraction of frames inside
    the band) >= theta; the inside-at-t* requirement keeps tau interpretable as a
    recovery instant (without it, theta < 1 lets t* sit on an excursion frame). If
    fewer than W frames remain after t* but >= W/2 = 28, the remainder is evaluated and
    a pass is flagged recovered_short_window; otherwise the encounter is censored.
    theta is calibrated ONCE on the null case (no per-gust-case tuning): the largest
    theta in {0.95, 0.9, 0.85, 0.8} for which ALL 6 Baseline encounters are declared
    recovered (full-window status) with tau_rec_v2 <= 8 frames under their own
    impact_frame attr; if no theta passes at [q01, q99] the band widens to
    [q005, q995] and the scan repeats. Every calibration step is recorded in the
    summary JSON and the CLI asserts the final Baseline 6/6. Columns tau_rec_v2_frames
    (absolute frame index, NaN if censored) and tau_rec_v2_tc ((frame - impact) *
    dt_tc) mirror the v1 convention.

Within-case phase ladder (PRC prep for Phase C)
    Within a case the 4-6 encounters share (G, D, Y) and differ ONLY in the gust
    release instant (every 120 frames = 6 t/c = 4.05 dominant lift periods at
    St 0.675), so consecutive encounters sample shifted shedding phases. Per case the
    ladder compares measured phi_imp against the cadence prediction
    phi_pred(k) = phi_meas(k_anchor) + (k - k_anchor) * dphi, anchored at the case's
    cleanest encounter: lowest pre_cl_amp among encounters with phase-fit R^2 >= 0.9,
    fitted period within [0.7, 1.4] x the dominant period, and pre_cl_amp <= 0.45
    (cases with no qualifying encounter are flagged unanchored). The period gate is
    load-bearing: Baseline encounter 2 has R^2 = 0.94 but a 16.5-frame fitted period
    (a wrong-line lock) and is also the lowest-amp Baseline encounter, so without the
    gate it would anchor and corrupt the whole reference ladder. Three cadence steps
    are carried: the dominant spectral clock (St 0.675 -> +0.322 rad/enc, the author's
    measured +0.3), the subharmonic spectral clock expressed in carrier phase via the
    2:1 lock (2 x St 0.338 -> +0.374 rad/enc), and the settled-Baseline ladder fit
    (resultant-maximizing step over Baseline encounters k >= 3; the spectral lines
    carry +-0.029 St resolution = +-1.1 rad/enc of cadence-step uncertainty, so the
    ladder fit is the precision measurement and is carried as a third, bonus clock).
    Validation: signed circular residuals wrap(phi_meas - phi_pred) in [-pi, pi) over
    CLEAN non-anchor encounters, pooled circular sd per clock; the ladder is declared
    usable for phase ASSIGNMENT if the pooled sd <= 0.5 rad at some clock. The
    offset-1 consecutive-clean-pair steps grouped by |G| are reported as the
    phase-response (PRC) precursor: on Baseline they reproduce the cadence to ~0.06
    rad, so any extra dispersion in gust cases is the gust's own phase response.

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
    outputs/session28/physics/phase_ladder.npz            (per-encounter ladder rows)
    outputs/session28/physics/physics_summary.json        (P3 coverage, P2 v1+v2
                                                           fractions, calibration
                                                           provenance, ladder verdict)

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
# P2 v2: settled-Baseline reference band + occupancy recovery rule (headline)
# --------------------------------------------------------------------------------------
SETTLED_GLOBAL_FRAME_MIN = 400  # HANDOFF D180: Baseline raw frames < 400 (t/c < 20) are transient
BAND_QUANTILES_PRIMARY = (0.01, 0.99)
BAND_QUANTILES_WIDE = (0.005, 0.995)
THETA_GRID = (0.95, 0.90, 0.85, 0.80)  # scanned descending: largest passing theta wins
OCC_WINDOW_DEFAULT = 56  # W: occupancy window, frames (= the v1 dwell, one subharmonic period)
OCC_MIN_WINDOW_DEFAULT = 28  # shortest remainder window that can still certify recovery
CALIB_MAX_TAU_FRAMES = 8  # null requirement: Baseline must recover within 8 frames of impact


@dataclass
class RecoveryRuleV2:
    """Frozen v2 recovery rule: fixed settled-Baseline band + occupancy threshold.

    Attributes:
        band_lo, band_hi: enstrophy band from the settled Baseline distribution.
        theta: occupancy threshold chosen by the null-case calibration.
        band_quantiles: the quantile pair that produced (band_lo, band_hi).
        occ_window: occupancy window W in frames.
        occ_min_window: minimum remainder window for recovered_short_window.
        calibration: full provenance dict (every step tried, per-encounter results).
    """

    band_lo: float
    band_hi: float
    theta: float
    band_quantiles: tuple[float, float]
    occ_window: int = OCC_WINDOW_DEFAULT
    occ_min_window: int = OCC_MIN_WINDOW_DEFAULT
    calibration: dict | None = None


def occupancy_recovery(
    trace: np.ndarray,
    start: int,
    lo: float,
    hi: float,
    theta: float,
    window: int = OCC_WINDOW_DEFAULT,
    min_window: int = OCC_MIN_WINDOW_DEFAULT,
) -> tuple[str, int | None]:
    """First t* >= start, inside [lo, hi], whose window [t*, t*+window) has occupancy >= theta.

    Occupancy = fraction of window frames inside the band, so isolated single-frame
    excursions inside the window are tolerated up to floor((1 - theta) * window) frames
    (3 at theta = 0.95, W = 56). t* itself must be inside the band: without this, a
    theta < 1 window starting ON an excursion frame would be declared the recovery
    instant. Returns (status, t*):
        ("recovered", t*)              full window observed and passed;
        ("recovered_short_window", t*) fewer than `window` but >= `min_window` frames
                                       remained after t* and the remainder passed;
        ("censored", None)             no qualifying t* (including t* with fewer than
                                       min_window frames left, which cannot be judged).
    Scanning is in ascending t*, so full-window verdicts always precede short-window
    ones (short windows only exist within `window` frames of the record end).
    """
    trace = np.asarray(trace, dtype=np.float64)
    n = trace.size
    inside = (trace >= lo) & (trace <= hi)
    for t in range(start, n):
        n_avail = n - t
        if n_avail < min_window:
            break
        if not inside[t]:
            continue
        occ = float(inside[t : t + window].mean())
        if occ >= theta:
            return ("recovered", t) if n_avail >= window else ("recovered_short_window", t)
    return "censored", None


def calibration_reference_case(manifest: dict) -> tuple[str, list[int]]:
    """(case_id, sorted encounter indices) of the is_calibration_reference case."""
    for case_id in sorted(manifest["cases"]):
        case = manifest["cases"][case_id]
        if case.get("is_calibration_reference"):
            ks = sorted(case["train_encounter_indices"] + case["val_encounter_indices"])
            return case_id, ks
    raise KeyError("no case with is_calibration_reference=true in the split manifest")


def collect_calibration_records(
    cache_root: Path,
    case_id: str,
    encounter_indices: list[int],
    params: "PrepParams",
    mask: np.ndarray | None = None,
) -> list[dict]:
    """Enstrophy traces + attrs for the calibration (Baseline) encounters."""
    import h5py  # local import keeps the synthetic-data unit tests free of h5py

    if mask is None:
        mask = wake_mask()
    records = []
    for k in encounter_indices:
        path = Path(cache_root) / case_id / f"encounter_{k:02d}.h5"
        with h5py.File(path, "r") as f:
            omega = f["omega_z"][:]
            impact = int(f.attrs["impact_frame_estimate"])
            frame_start = int(f.attrs["frame_start"])
        records.append(
            {
                "encounter_index": int(k),
                "impact_frame": impact,
                "frame_start": frame_start,
                "ens": large_scale_wake_enstrophy(omega, params.sigma_px, mask),
            }
        )
    return records


def settled_reference_band(
    records: list[dict],
    quantiles: tuple[float, float],
    settled_min_global_frame: int = SETTLED_GLOBAL_FRAME_MIN,
) -> tuple[float, float, int]:
    """(lo, hi, n_frames) band from the settled subset of the calibration traces.

    A frame at local index t of an encounter starting at global raw frame `frame_start`
    is settled iff frame_start + t >= settled_min_global_frame (HANDOFF D180).
    """
    parts = []
    for rec in records:
        offset = max(0, settled_min_global_frame - rec["frame_start"])
        if offset < rec["ens"].size:
            parts.append(rec["ens"][offset:])
    if not parts:
        raise ValueError("no settled frames in the calibration records")
    settled = np.concatenate(parts)
    lo, hi = np.quantile(settled, quantiles)
    return float(lo), float(hi), int(settled.size)


def calibrate_recovery_v2(
    records: list[dict],
    occ_window: int = OCC_WINDOW_DEFAULT,
    occ_min_window: int = OCC_MIN_WINDOW_DEFAULT,
    theta_grid: tuple[float, ...] = THETA_GRID,
    band_ladder: tuple[tuple[float, float], ...] = (BAND_QUANTILES_PRIMARY, BAND_QUANTILES_WIDE),
    max_tau_frames: int = CALIB_MAX_TAU_FRAMES,
) -> RecoveryRuleV2:
    """Calibrate (band, theta) on the null case; no per-gust-case tuning.

    For each band (primary first, widened second) the theta grid is scanned descending;
    the first (= largest) theta for which EVERY calibration encounter is declared
    "recovered" (full window) with tau <= max_tau_frames freezes the rule. Every step is
    recorded in the returned rule's `calibration` dict. Raises RuntimeError (with the
    full step record in args) if no step passes: the hard requirement is null 6/6.
    """
    steps: list[dict] = []
    for quantiles in band_ladder:
        lo, hi, n_settled = settled_reference_band(records, quantiles)
        for theta in theta_grid:
            per_enc = []
            for rec in records:
                status, tau = occupancy_recovery(
                    rec["ens"], rec["impact_frame"], lo, hi, theta, occ_window, occ_min_window
                )
                per_enc.append(
                    {
                        "encounter_index": rec["encounter_index"],
                        "status": status,
                        "tau_frames": None if tau is None else int(tau - rec["impact_frame"]),
                    }
                )
            ok = all(
                e["status"] == "recovered" and e["tau_frames"] <= max_tau_frames for e in per_enc
            )
            steps.append(
                {
                    "band_quantiles": list(quantiles),
                    "band_lo": lo,
                    "band_hi": hi,
                    "n_settled_frames": n_settled,
                    "theta": theta,
                    "per_encounter": per_enc,
                    "pass": ok,
                }
            )
            if ok:
                calibration = {
                    "settled_min_global_frame": SETTLED_GLOBAL_FRAME_MIN,
                    "n_settled_frames": n_settled,
                    "max_tau_frames": max_tau_frames,
                    "theta_grid": list(theta_grid),
                    "band_ladder": [list(q) for q in band_ladder],
                    "steps": steps,
                    "n_null_encounters": len(records),
                    "null_6of6": True,
                }
                return RecoveryRuleV2(
                    band_lo=lo,
                    band_hi=hi,
                    theta=theta,
                    band_quantiles=quantiles,
                    occ_window=occ_window,
                    occ_min_window=occ_min_window,
                    calibration=calibration,
                )
    raise RuntimeError(
        "v2 recovery calibration failed: no (band, theta) recovers the null case "
        f"{len(records)}/{len(records)} within {max_tau_frames} frames",
        steps,
    )


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
    v2_rule: RecoveryRuleV2 | None = None,
) -> dict:
    """Compute the full per-encounter physics row from one cache file.

    When `v2_rule` is None (unit tests, exploratory calls) the v2 recovery columns are
    emitted as status "unscored" with NaN taus; the CLI always passes a calibrated rule.
    """
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

    if v2_rule is None:
        status_v2, tau_v2 = "unscored", None
    else:
        status_v2, tau_v2 = occupancy_recovery(
            ens,
            impact,
            v2_rule.band_lo,
            v2_rule.band_hi,
            v2_rule.theta,
            v2_rule.occ_window,
            v2_rule.occ_min_window,
        )
    tau_v2_frame = float("nan") if tau_v2 is None else float(tau_v2)
    tau_v2_tc = float("nan") if tau_v2 is None else (tau_v2 - impact) * DT_TC

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
        "tau_rec_v2_frames": tau_v2_frame,
        "tau_rec_v2_tc": tau_v2_tc,
        "recovery_status_v2": status_v2,
        "censored_v2": status_v2 == "censored",
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
# Within-case phase ladder (PRC prep)
# --------------------------------------------------------------------------------------
RELEASE_PERIOD_FRAMES = 120  # gust released every 120 frames
RELEASE_PERIOD_TC = RELEASE_PERIOD_FRAMES * DT_TC  # 6.0 t/c
ST_DOM_FALLBACK = 0.6752056821661404  # Baseline lift spectrum (undisturbed_stats.json, D180)
ST_SUB_FALLBACK = 0.3382904869675207
PERIOD_GATE_REL = (0.7, 1.4)  # fitted-period window around the dominant period (wrong-line gate)
LADDER_USABLE_SD_RAD = 0.5  # pooled residual circ-sd bar for "usable to assign phases"
LADDER_FIT_GRID_N = 72000  # cadence-step grid resolution for the resultant fit (~0.09 mrad)


def wrap_pm_pi(x: np.ndarray | float) -> np.ndarray | float:
    """Wrap angle(s) to [-pi, pi)."""
    return (np.asarray(x, dtype=np.float64) + math.pi) % TWO_PI - math.pi


def circ_residual(measured: np.ndarray | float, predicted: np.ndarray | float):
    """Signed circular residual measured - predicted, wrapped to [-pi, pi)."""
    return wrap_pm_pi(np.asarray(measured, dtype=np.float64) - np.asarray(predicted))


def circ_residual_stats(res: np.ndarray) -> dict:
    """n, circular mean (wrapped to [-pi, pi)), resultant R, circular sd of residuals."""
    base = circular_stats(np.asarray(res, dtype=np.float64))
    if base["n"]:
        base["mean"] = float(wrap_pm_pi(base["mean"]))
    return base


@dataclass
class CadenceClocks:
    """Cadence-step hypotheses for the within-case phase ladder.

    phi_imp is the DOMINANT-carrier phase, so every step is expressed in carrier
    radians per encounter (release interval 120 frames = 6 t/c):
        step_dom: wrap(2 pi * St_dom * 6)        -- the dominant spectral line as clock;
        step_sub: wrap(2 pi * (2 St_sub) * 6)    -- the subharmonic line as clock, mapped
                  into carrier phase through the 2:1 lock (the carrier completes two
                  cycles per subharmonic cycle).
    The spectral resolution of the 35 t/c Baseline record is 0.029 St = 1.08 rad/enc of
    step uncertainty, so these are point hypotheses, not precision measurements; the
    settled-Baseline ladder fit (fit_ladder_cadence) is the precision instrument.
    """

    st_dom: float = ST_DOM_FALLBACK
    st_sub: float = ST_SUB_FALLBACK
    source: str = "frozen fallback (module constants, measured in D180)"

    @property
    def t_dom_frames(self) -> float:
        return 1.0 / (self.st_dom * DT_TC)

    @property
    def step_dom(self) -> float:
        return float(wrap_pm_pi(TWO_PI * self.st_dom * RELEASE_PERIOD_TC))

    @property
    def step_sub(self) -> float:
        return float(wrap_pm_pi(TWO_PI * 2.0 * self.st_sub * RELEASE_PERIOD_TC))


def load_clocks(stats_path: Path | str | None) -> CadenceClocks:
    """Clocks from outputs/session28/undisturbed_stats.json, else frozen fallbacks."""
    if stats_path is not None and Path(stats_path).is_file():
        stats = json.loads(Path(stats_path).read_text())
        ours = stats.get("ours", stats)
        if "St_lift_dominant" in ours and "St_lift_subharmonic" in ours:
            return CadenceClocks(
                st_dom=float(ours["St_lift_dominant"]),
                st_sub=float(ours["St_lift_subharmonic"]),
                source=str(stats_path),
            )
    return CadenceClocks()


def fit_ladder_cadence(
    k: np.ndarray, phi: np.ndarray, grid_n: int = LADDER_FIT_GRID_N
) -> tuple[float, float]:
    """(step, resultant R) maximizing |mean_k exp(i (phi_k - k step))| over [-pi, pi).

    The resultant-maximizing wrapped step is the circular least-squares cadence of the
    ladder phi_k ~ phi_0 + k * step (mod 2 pi); R = 1 means a perfect linear ladder.
    """
    k = np.asarray(k, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    good = np.isfinite(phi)
    k, phi = k[good], phi[good]
    if k.size < 3:
        return float("nan"), float("nan")
    grid = np.linspace(-math.pi, math.pi, grid_n, endpoint=False)
    resultant = np.abs(np.exp(1j * (phi[None, :] - np.outer(grid, k))).mean(axis=1))
    i = int(np.argmax(resultant))
    return float(grid[i]), float(resultant[i])


def build_phase_ladder(
    rows: list[dict],
    params: PrepParams,
    clocks: CadenceClocks,
    calib_case_id: str = "Baseline",
) -> tuple[dict, dict]:
    """Per-encounter ladder arrays (for phase_ladder.npz) + ladder summary dict.

    Flags per encounter:
        period_gate_ok: fitted Hilbert period within PERIOD_GATE_REL x dominant period;
        clean: finite phi AND phase_fit_r2 >= phase_r2_min AND period_gate_ok AND
               pre_cl_amp <= phase_amp_max (uncontaminated, well-locked pre-window);
        anchor: the case's clean encounter with the lowest pre_cl_amp (one per anchored
                case; deviation from the bare R^2 >= 0.9 spec is documented in the
                module docstring: the period and amp gates prevent anchoring on a
                wrong-line lock or on a gust-contaminated pre-window).
    Predictions phi_pred_{dom, sub, fit} are emitted for EVERY encounter of an anchored
    case (clean or not; assigning phases to contaminated encounters is the Phase C use
    case) and are NaN for unanchored cases. Residuals are pooled over clean non-anchor
    encounters only.
    """
    n = len(rows)
    case = np.array([r["case_id"] for r in rows])
    k = np.array([r["encounter_index"] for r in rows], dtype=np.int64)
    phi = np.array([r["phi_imp"] for r in rows], dtype=np.float64)
    r2 = np.array([r["phase_fit_r2"] for r in rows], dtype=np.float64)
    amp = np.array([r["pre_cl_amp"] for r in rows], dtype=np.float64)
    period = np.array([r["phase_period_frames"] for r in rows], dtype=np.float64)

    period_lo = PERIOD_GATE_REL[0] * clocks.t_dom_frames
    period_hi = PERIOD_GATE_REL[1] * clocks.t_dom_frames
    period_ok = np.isfinite(period) & (period >= period_lo) & (period <= period_hi)
    clean = (
        np.isfinite(phi) & (r2 >= params.phase_r2_min) & period_ok & (amp <= params.phase_amp_max)
    )

    # Settled-Baseline ladder fit: encounter k overlaps the settled record (global raw
    # frame >= SETTLED_GLOBAL_FRAME_MIN) iff 120 k + 119 >= 400, i.e. k >= 3. Encounter
    # 3's pre-impact window (global 360..399) is the transient tail, but its measured
    # step agrees with the fully settled 4 -> 5 step to ~0.01 rad (see summary).
    settled_min_k = math.ceil(
        (SETTLED_GLOBAL_FRAME_MIN - (RELEASE_PERIOD_FRAMES - 1)) / RELEASE_PERIOD_FRAMES
    )
    fit_mask = (case == calib_case_id) & (k >= settled_min_k) & period_ok & np.isfinite(phi)
    step_fit, fit_resultant = fit_ladder_cadence(k[fit_mask], phi[fit_mask])

    steps = {"dom": clocks.step_dom, "sub": clocks.step_sub, "fit": step_fit}
    anchor = np.zeros(n, dtype=bool)
    case_anchored = np.zeros(n, dtype=bool)
    preds = {name: np.full(n, np.nan) for name in steps}
    n_cases = len(set(case.tolist()))
    n_anchored = 0
    for cid in np.unique(case):
        in_case = case == cid
        eligible = in_case & clean
        if not eligible.any():
            continue
        n_anchored += 1
        k_anchor = k[eligible][int(np.argmin(amp[eligible]))]
        anchor_idx = int(np.flatnonzero(eligible & (k == k_anchor))[0])
        anchor[anchor_idx] = True
        case_anchored[in_case] = True
        for name, step in steps.items():
            if math.isfinite(step):
                preds[name][in_case] = (phi[anchor_idx] + (k[in_case] - k_anchor) * step) % TWO_PI
    resids = {name: circ_residual(phi, preds[name]) for name in steps}

    pool = clean & case_anchored & ~anchor
    pooled = {
        name: circ_residual_stats(resids[name][pool & np.isfinite(resids[name])]) for name in steps
    }

    # PRC precursor: offset-1 steps between consecutive clean encounters, grouped by |G|.
    # On Baseline these reproduce the cadence (dispersion ~0.06 rad); extra dispersion in
    # gust cases is the gust's own phase response between the two releases.
    pair_steps: list[tuple[float, float]] = []
    for cid in np.unique(case):
        idx = np.flatnonzero(case == cid)
        idx = idx[np.argsort(k[idx])]
        for a, b in zip(idx[:-1], idx[1:]):
            if k[b] == k[a] + 1 and clean[a] and clean[b]:
                pair_steps.append((abs(rows[a]["G"]), float(circ_residual(phi[b], phi[a]))))
    prc = {}
    for g in sorted({g for g, _ in pair_steps}):
        vals = np.array([s for gg, s in pair_steps if gg == g])
        stats = circ_residual_stats(vals)
        entry = {
            "n_pairs": stats["n"],
            "step_circ_mean_rad": stats["mean"],
            "step_circ_sd_rad": stats["circ_std"],
        }
        if math.isfinite(step_fit):
            entry["resid_vs_fit_circ_sd_rad"] = circ_residual_stats(circ_residual(vals, step_fit))[
                "circ_std"
            ]
        prc[f"{g:.2f}"] = entry

    sds = {name: pooled[name]["circ_std"] for name in steps if pooled[name]["n"] > 0}
    finite_sds = {name: sd for name, sd in sds.items() if math.isfinite(sd)}
    winner = min(finite_sds, key=finite_sds.get) if finite_sds else None
    usable = winner is not None and finite_sds[winner] <= LADDER_USABLE_SD_RAD

    base_mask = case == calib_case_id
    base_idx = np.flatnonzero(base_mask)
    base_idx = base_idx[np.argsort(k[base_idx])]
    summary = {
        "_doc": (
            "Within-case phase ladder: measured phi_imp vs cadence predictions anchored "
            "at the cleanest encounter. See module docstring for gates and clock "
            "definitions; residuals are signed circular, wrapped to [-pi, pi)."
        ),
        "clocks": {
            "st_dom": clocks.st_dom,
            "st_sub": clocks.st_sub,
            "st_source": clocks.source,
            "release_period_frames": RELEASE_PERIOD_FRAMES,
            "release_period_tc": RELEASE_PERIOD_TC,
            "step_dom_rad_per_enc": clocks.step_dom,
            "step_sub_rad_per_enc": clocks.step_sub,
            "step_sub_note": "2 x St_sub mapped to carrier phase via the 2:1 lock",
            "step_fit_rad_per_enc": step_fit,
            "step_fit_resultant_R": fit_resultant,
            "step_fit_source": (
                f"resultant-max over {calib_case_id} encounters k >= {settled_min_k} "
                f"(settled record, global frame >= {SETTLED_GLOBAL_FRAME_MIN})"
            ),
            "spectral_step_resolution_rad_per_enc": TWO_PI * RELEASE_PERIOD_TC / 35.0,
        },
        "quality_gates": {
            "phase_r2_min": params.phase_r2_min,
            "phase_amp_max": params.phase_amp_max,
            "period_gate_frames": [period_lo, period_hi],
            "period_gate_rel": list(PERIOD_GATE_REL),
        },
        "baseline_ladder": {
            "k": k[base_idx].tolist(),
            "phi_meas": phi[base_idx].tolist(),
            "period_frames": period[base_idx].tolist(),
            "period_gate_ok": period_ok[base_idx].tolist(),
            "consecutive_wrapped_steps": circ_residual(
                phi[base_idx][1:], phi[base_idx][:-1]
            ).tolist(),
            "settled_min_k": settled_min_k,
        },
        "anchoring": {
            "n_cases": n_cases,
            "n_anchored": n_anchored,
            "frac_anchored": n_anchored / n_cases if n_cases else float("nan"),
            "n_clean_encounters": int(clean.sum()),
            "n_pooled_residuals": int(pool.sum()),
        },
        "pooled_residuals": pooled,
        "verdict": {
            "usable_sd_threshold_rad": LADDER_USABLE_SD_RAD,
            "usable_for_phase_assignment": usable,
            "winning_clock": winner,
            "winning_pooled_sd_rad": finite_sds.get(winner, float("nan")),
            "per_clock_pooled_sd_rad": sds,
        },
        "prc_precursor_offset1_steps_by_absG": prc,
    }

    arrays = {
        "case_id": case,
        "encounter_index": k,
        "split": np.array([r["split"] for r in rows]),
        "G": np.array([r["G"] for r in rows], dtype=np.float64),
        "D": np.array([r["D"] for r in rows], dtype=np.float64),
        "Y": np.array([r["Y"] for r in rows], dtype=np.float64),
        "phi_meas": phi,
        "phase_fit_r2": r2,
        "pre_cl_amp": amp,
        "phase_period_frames": period,
        "period_gate_ok": period_ok,
        "clean": clean,
        "anchor": anchor,
        "case_anchored": case_anchored,
        "phi_pred_dom": preds["dom"],
        "phi_pred_sub": preds["sub"],
        "phi_pred_fit": preds["fit"],
        "resid_dom": resids["dom"],
        "resid_sub": resids["sub"],
        "resid_fit": resids["fit"],
    }
    return arrays, summary


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
    # Clamp: a single unit vector (or perfectly aligned set) can give |z| = 1 + O(eps),
    # which would push -2 log r negative and fault the sqrt.
    r = min(float(np.abs(z)), 1.0)
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


def _recovery_block_v2(rows: list[dict]) -> dict:
    n = len(rows)
    full = [r for r in rows if r["recovery_status_v2"] == "recovered"]
    short = [r for r in rows if r["recovery_status_v2"] == "recovered_short_window"]
    cen = [r for r in rows if r["recovery_status_v2"] == "censored"]
    taus = np.array([r["tau_rec_v2_tc"] for r in full + short], dtype=np.float64)
    return {
        "n": n,
        "recovered": len(full),
        "recovered_short_window": len(short),
        "censored": len(cen),
        "frac_recovered_full": len(full) / n if n else float("nan"),
        "frac_recovered_any": (len(full) + len(short)) / n if n else float("nan"),
        "frac_censored": len(cen) / n if n else float("nan"),
        "tau_rec_v2_tc_median_recovered": float(np.median(taus)) if taus.size else float("nan"),
        "tau_rec_v2_tc_max_recovered": float(np.max(taus)) if taus.size else float("nan"),
    }


_V1_RECOVERED = ("recovered",)
_V2_RECOVERED = ("recovered", "recovered_short_window")


def _tau_zero_table(rows: list[dict], status_key: str, tau_key: str, recovered: tuple) -> dict:
    """Fraction of recovered encounters with tau exactly 0 (= still/already in band at
    impact), split by release class and |G|. The v1 artifact under test: re-releases
    (k >= 1) of strong gusts recovering at tau = 0 against their own inflated
    pre-impact envelope."""
    out: dict = {}
    for cls_name, cls_fn in (
        ("first_release_k0", lambda r: r["encounter_index"] == 0),
        ("re_release_k_ge_1", lambda r: r["encounter_index"] >= 1),
    ):
        blk = {}
        for g in sorted({abs(r["G"]) for r in rows}):
            sel = [r for r in rows if abs(r["G"]) == g and cls_fn(r) and r[status_key] in recovered]
            n = len(sel)
            n0 = sum(1 for r in sel if r[tau_key] == r["impact_frame"])
            blk[f"{g:.2f}"] = {
                "n_recovered": n,
                "n_tau0": n0,
                "frac_tau0": n0 / n if n else None,
            }
        out[cls_name] = blk
    return out


def _is_monotone(seq: list[float]) -> bool:
    return all(b <= a for a, b in zip(seq, seq[1:])) or all(b >= a for a, b in zip(seq, seq[1:]))


def build_summary(
    rows: list[dict],
    params: PrepParams,
    manifest_path: str,
    v2_rule: RecoveryRuleV2 | None = None,
) -> dict:
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

    p2_v2: dict | None = None
    if v2_rule is not None:
        p2_v2 = {
            "_doc": (
                "HEADLINE recovery rule (v2): fixed settled-Baseline enstrophy band + "
                "occupancy threshold calibrated on the null case. v1 (per-encounter "
                "envelope) is kept above for continuity; see module docstring."
            ),
            "rule": {
                "band_quantiles": list(v2_rule.band_quantiles),
                "band_lo": v2_rule.band_lo,
                "band_hi": v2_rule.band_hi,
                "theta": v2_rule.theta,
                "occ_window_frames": v2_rule.occ_window,
                "occ_min_window_frames": v2_rule.occ_min_window,
            },
            "calibration": v2_rule.calibration,
            "per_split": {},
            "by_G": {},
            "by_D": {},
            "by_absY": {},
            "per_case": {},
        }
        for s in splits:
            p2_v2["per_split"][s] = _recovery_block_v2([r for r in rows if r["split"] == s])
            p2_v2["per_split"][s]["v1_frac_recovered_strict"] = p2["per_split"][s]["frac_recovered"]
        for key, fn in (
            ("by_G", lambda r: f"{r['G']:+.2f}"),
            ("by_D", lambda r: f"{r['D']:.2f}"),
            ("by_absY", lambda r: f"{abs(r['Y']):.2f}"),
        ):
            for lvl in sorted({fn(r) for r in rows}):
                p2_v2[key][lvl] = _recovery_block_v2([r for r in rows if fn(r) == lvl])
        for cid in sorted({r["case_id"] for r in rows}):
            p2_v2["per_case"][cid] = _recovery_block_v2([r for r in rows if r["case_id"] == cid])
        base_rows = [r for r in rows if r["case_id"] == "Baseline"]
        if base_rows:
            blk = _recovery_block_v2(base_rows)
            blk["all_recovered_within_8_frames"] = all(
                r["recovery_status_v2"] == "recovered"
                and r["tau_rec_v2_frames"] - r["impact_frame"] <= CALIB_MAX_TAU_FRAMES
                for r in base_rows
            )
            blk["tau_frames_after_impact"] = {
                str(r["encounter_index"]): r["tau_rec_v2_frames"] - r["impact_frame"]
                for r in sorted(base_rows, key=lambda r: r["encounter_index"])
            }
            p2_v2["calibration_reference_baseline"] = blk
        # v1 artifact checks. Artifact 1: re-releases recovering at tau = 0 against
        # their own inflated pre-window. Under v2 a tau = 0 of a WEAK gust is honest
        # physics (the gust never pushes the large-scale wake enstrophy outside the
        # undisturbed band), so the table is split by |G|. Artifact 2: v1's recovered
        # fraction was non-monotone in D (0.08 / 0.24 / 0.17); note the D levels are
        # not G-balanced, so monotonicity in D is a soft expectation, not a law.
        gust = [r for r in rows if r["D"] > 0]
        by_d = {}
        for lvl in sorted({r["D"] for r in gust}):
            sub = [r for r in gust if r["D"] == lvl]
            v1b, v2b = _recovery_block(sub), _recovery_block_v2(sub)
            by_d[f"{lvl:.2f}"] = {
                "n": len(sub),
                "v1_frac_recovered": v1b["frac_recovered"],
                "v2_frac_recovered_any": v2b["frac_recovered_any"],
                "v1_tau_tc_median": v1b["tau_rec_tc_median_recovered"],
                "v2_tau_tc_median": v2b["tau_rec_v2_tc_median_recovered"],
            }
        p2_v2["v1_artifact_checks"] = {
            "tau_zero_rate_v1": _tau_zero_table(
                rows, "recovery_status", "tau_rec_frame", _V1_RECOVERED
            ),
            "tau_zero_rate_v2": _tau_zero_table(
                rows, "recovery_status_v2", "tau_rec_v2_frames", _V2_RECOVERED
            ),
            "by_D_gust_only": by_d,
            "v1_frac_recovered_monotone_in_D": _is_monotone(
                [by_d[lvl]["v1_frac_recovered"] for lvl in sorted(by_d)]
            ),
            "v2_frac_recovered_any_monotone_in_D": _is_monotone(
                [by_d[lvl]["v2_frac_recovered_any"] for lvl in sorted(by_d)]
            ),
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
    summary: dict = {
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
    if p2_v2 is not None:
        summary["p2_recovery_v2"] = p2_v2
    return summary


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
    "tau_rec_v2_frames",
    "tau_rec_v2_tc",
    "recovery_status_v2",
    "censored_v2",
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
    ap.add_argument("--occ-window", type=int, default=OCC_WINDOW_DEFAULT)
    ap.add_argument("--occ-min-window", type=int, default=OCC_MIN_WINDOW_DEFAULT)
    ap.add_argument(
        "--undisturbed-stats",
        default=str(repo / "outputs/session28/undisturbed_stats.json"),
        help="A2 stats JSON for the measured Strouhal lines (fallback: frozen constants)",
    )
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

    # Calibrate the v2 recovery rule on the null case BEFORE touching any gust encounter.
    calib_case, calib_ks = calibration_reference_case(manifest)
    print(f"calibrating v2 recovery rule on {calib_case} encounters {calib_ks} ...", flush=True)
    calib_records = collect_calibration_records(cache_root, calib_case, calib_ks, params, mask)
    v2_rule = calibrate_recovery_v2(calib_records, args.occ_window, args.occ_min_window)
    for step in v2_rule.calibration["steps"]:
        taus = [e["tau_frames"] for e in step["per_encounter"]]
        print(
            f"  band q{step['band_quantiles']} = [{step['band_lo']:.3f}, "
            f"{step['band_hi']:.3f}], theta {step['theta']:.2f}: "
            f"{'PASS' if step['pass'] else 'fail'} (taus {taus})"
        )
    print(
        f"v2 rule frozen: band [{v2_rule.band_lo:.3f}, {v2_rule.band_hi:.3f}] "
        f"(q{list(v2_rule.band_quantiles)}), theta {v2_rule.theta:.2f}, "
        f"W {v2_rule.occ_window}, {calib_case} 6/6 confirmed"
    )

    rows: list[dict] = []
    todo = list(iter_encounters(manifest))
    if args.limit:
        todo = todo[: args.limit]
    t0 = time.time()
    for i, (case_id, k, split) in enumerate(todo):
        path = cache_root / case_id / f"encounter_{k:02d}.h5"
        if not path.is_file():
            sys.exit(f"missing cache file: {path}")
        rows.append(process_encounter(path, split, params, mask, v2_rule))
        if (i + 1) % 50 == 0 or i + 1 == len(todo):
            print(
                f"  [{i + 1}/{len(todo)}] {case_id} enc {k} ({time.time() - t0:.0f} s)", flush=True
            )

    # Hard requirement: the frozen rule recovers the null case n/n in the production run.
    base_rows = [r for r in rows if r["case_id"] == calib_case]
    if len(base_rows) == len(calib_ks):
        assert all(
            r["recovery_status_v2"] == "recovered"
            and r["tau_rec_v2_frames"] - r["impact_frame"] <= CALIB_MAX_TAU_FRAMES
            for r in base_rows
        ), f"{calib_case} not recovered {len(calib_ks)}/{len(calib_ks)} by the frozen v2 rule"
        print(f"{calib_case} {len(base_rows)}/{len(calib_ks)} recovered under the frozen rule")
    else:
        print(f"WARNING: --limit cut {calib_case}; null consistency check skipped")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for col in NPZ_COLUMNS:
        vals = [r[col] for r in rows]
        if col in ("case_id", "split", "source_group", "recovery_status", "recovery_status_v2"):
            arrays[col] = np.array(vals)
        elif col in ("encounter_index", "impact_frame"):
            arrays[col] = np.array(vals, dtype=np.int64)
        elif col in ("censored", "censored_v2"):
            arrays[col] = np.array(vals, dtype=bool)
        else:
            arrays[col] = np.array(vals, dtype=np.float64)
    npz_path = out_dir / "per_encounter_physics.npz"
    np.savez(npz_path, **arrays)

    clocks = load_clocks(args.undisturbed_stats)
    ladder_arrays, ladder_summary = build_phase_ladder(rows, params, clocks, calib_case)
    ladder_path = out_dir / "phase_ladder.npz"
    np.savez(ladder_path, **ladder_arrays)

    summary = build_summary(rows, params, args.split_manifest, v2_rule)
    summary["phase_ladder"] = ladder_summary
    json_path = out_dir / "physics_summary.json"
    json_path.write_text(json.dumps(summary, indent=1))

    print(f"\nwrote {npz_path} ({len(rows)} encounters), {ladder_path} and {json_path}")
    for key, blk in summary["p3_phase_coverage"]["pooled"].items():
        print(
            f"P3 coverage (pooled, {key}): n={blk['n']} occupied "
            f"{blk['occupied_bins']}/{blk['n_bins']} bins ({blk['occupied_bin_fraction']:.2f}), "
            f"arc {blk['arc_coverage_fraction']:.2f}, R = {blk['R']:.3f}"
        )
    for s, blk in summary["p2_recovery"]["per_split"].items():
        print(
            f"P2 v1 {s}: n={blk['n']} recovered {blk['recovered']} "
            f"({blk['frac_recovered']:.2f}), unconfirmed {blk['reentered_unconfirmed']}, "
            f"censored {blk['censored']}; median tau_rec "
            f"{blk['tau_rec_tc_median_recovered']:.2f} t/c"
        )
    for s, blk in summary["p2_recovery_v2"]["per_split"].items():
        print(
            f"P2 v2 {s}: n={blk['n']} recovered {blk['recovered']}+{blk['recovered_short_window']}"
            f" short ({blk['frac_recovered_any']:.2f} any), censored {blk['censored']} "
            f"({blk['frac_censored']:.2f}); median tau "
            f"{blk['tau_rec_v2_tc_median_recovered']:.2f} t/c "
            f"(v1 strict frac {blk['v1_frac_recovered_strict']:.2f})"
        )
    checks = summary["p2_recovery_v2"]["v1_artifact_checks"]
    print(
        "artifact checks: v1 monotone-in-D "
        f"{checks['v1_frac_recovered_monotone_in_D']}, v2 monotone-in-D "
        f"{checks['v2_frac_recovered_any_monotone_in_D']}"
    )
    verdict = ladder_summary["verdict"]
    print(
        f"phase ladder: anchored {ladder_summary['anchoring']['n_anchored']}/"
        f"{ladder_summary['anchoring']['n_cases']} cases, pooled residual sd "
        + ", ".join(f"{c}={sd:.3f}" for c, sd in verdict["per_clock_pooled_sd_rad"].items())
        + f" rad; usable={verdict['usable_for_phase_assignment']} "
        f"(bar {verdict['usable_sd_threshold_rad']} rad, winner {verdict['winning_clock']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
