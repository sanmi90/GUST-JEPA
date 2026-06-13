#!/usr/bin/env python
"""Physics Track P2 latent-clock tie-in / Gate GP2 (Session 28 master plan, v2.1 UNCOND).

The PHYSICAL recovery clock tau_rec already exists per encounter in
outputs/session28/physics/per_encounter_physics.npz. This script reuses the v2
(settled-Baseline band, occupancy-dwell) physical clock as the reference and computes a
LATENT recovery clock from the predictive (jepa_tf_noc) latent trajectories, then asks
the held-out question that makes P2 a latent result: does the LATENT clock match the
PHYSICAL clock per encounter?

Physical clock (reference; NOT recomputed here)
    Read directly from per_encounter_physics.npz. We use the v2 redesign columns
    (recovery_status_v2 / tau_rec_v2_frames / censored_v2), the null-calibrated
    "wake enstrophy re-enters the settled-Baseline band and dwells" rule (physics_prep
    P2 v2). tau_rec_v2_frames is the ABSOLUTE frame index (NaN if censored);
    recovery_status_v2 in {recovered, recovered_short_window, censored}. An encounter is
    censored-physical iff censored_v2 is True (recovery_status_v2 == "censored"); the
    recovered_short_window status counts as recovered (a pass was certified on the
    available remainder). CENSORED encounters are carried as censored, never dropped.

Latent clock (computed here)
    The baseline limit-cycle TUBE is the distribution of the predictive latent z over the
    SETTLED undisturbed Baseline orbit (global raw frame >= 400, t/c >= 20; HANDOFF D180),
    pooled over the Baseline encounters present in the family's TRAIN latents. The tube is
    described by the orbit mean + ridged covariance whitening (s46_regen geometry, reused);
    the tube RADIUS is q95 of the baseline orbit's OWN Mahalanobis spread (the 95th
    percentile of the Mahalanobis distance of each settled-Baseline frame to the orbit
    distribution). The latent recovery rule is physics_prep.occupancy_recovery applied to the
    tube-membership signal, MATCHED to the physical v2 enstrophy clock so the gate compares
    like with like (only the underlying signal differs): latent tau_rec = the first
    post-impact frame t* that is itself inside the tube AND whose [t*, t*+W) window has
    occupancy (fraction of frames inside) >= theta. Defaults are matched to the physical
    clock's null-calibrated rule (W = 56 frames ~ the subharmonic shedding period
    1/(St_sub*dt_tc) ~= 59, theta = 0.8, min_window = 28); all parameterized on the CLI.
    Encounters whose latent never returns-and-dwells within the 120-frame window are
    CENSORED, the same convention as the physical clock; if a return is observed but < W but
    >= min_window frames remain, the status is "recovered_short_window" and counts as
    recovered (mirrors the physical rule). Using the SAME occupancy semantics as the physical
    clock (rather than a strict dwell) is load-bearing: a strict every-frame dwell censors
    the whole test_b set because the latent never holds the tight tube for 56 consecutive
    frames in the 80-frame post-impact window, which would make the gate vacuous.

    The orbit cloud is small (~80 frames) and very low-rank (effective dim ~3.6 of 64), so
    the Mahalanobis whitening is ill-conditioned exactly as s46_regen documents. We follow
    the master-plan P2 definition literally (Mahalanobis tube) but the gate statistic is a
    RANK correlation (Spearman), which is insensitive to the magnitude inflation of the
    whitened distances. A Euclidean tube cross-check (tube radius = q95 of the Euclidean
    spread; well-conditioned) is carried alongside so the gate verdict can be confirmed not
    to be a whitening artefact, mirroring the headline-vs-crosscheck split in s46_regen.

Gate GP2 (D191)
    Spearman(latent tau_rec, physical tau_rec) over the test_b encounters recovered (NOT
    censored) under BOTH clocks. The tau used is the convective-time-since-impact
    (tau_rec_tc), so both clocks are on the same physical axis. >= 0.7: "the latent carries
    the recovery clock" enters the flow-physics section (S4.4) with the tau_rec map figure.
    Below 0.7: the physical maps stand as DNS physics (still a contribution) but the
    latent-clock sentence is DROPPED. We report the Spearman as measured, with a
    case-clustered bootstrap CI and a case-permutation p (VALID here: the gate is a
    correlation / trend, not a paired location test, so stats_lib.case_permutation_p is the
    right instrument, unlike s46 Q2's paired distance), the recovered-vs-censored fraction
    for BOTH clocks, and the n that enters the correlation.

    Adversarial reporting: the Spearman is whatever the data give. If it is below 0.7 the
    README and the numbers part say so plainly and record that the latent-clock claim is
    dropped while the physical maps stand. Censoring is always reported, never silently
    excluded.

Map data (for Figure 9)
    The physical tau_rec(G, D, Y) over the envelope (one point per encounter, carried with
    G/D/Y/split/case_id/status) and tau_rec versus the P1-style |G| axis are emitted in
    results.json (and drawn in fig_p2_recovery).

Outputs
    outputs/session28/p2/results.json     full provenance (clocks, gate, fractions, map data)
    outputs/session28/p2/README.md        human-readable, gate verdict stated plainly
    outputs/session28/p2/fig_p2_recovery.{pdf,png}   physical tau_rec map + latent-vs-
                                                     physical clock scatter with the Spearman
    outputs/session28/numbers_parts/p2.json   eval_all part (ALPHABETIC macros only)

CLI (CPU-only; numpy/scipy/matplotlib; reads the two NPZ artefacts, no torch, no GPU)
    export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
    nice -n 10 .venv/bin/python scripts/session28/p2_latent_clock.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO))  # so `from src...` and sibling modules resolve

import physics_prep as pp  # noqa: E402  DT_TC, ST_SUB_FALLBACK, calibration_reference_case
import s46_regen as s46  # noqa: E402  settled_baseline_orbit + OrbitGeometry (verbatim reuse)
import stats_lib  # noqa: E402  case-clustered CI + case-permutation p

# --------------------------------------------------------------------------- defaults
LATENTS_ROOT = REPO / "outputs/session28/latents"
PHYSICS_NPZ = REPO / "outputs/session28/physics/per_encounter_physics.npz"
SPLIT_MANIFEST = REPO / "configs/splits/split_v2p1.json"
OUT_DIR = REPO / "outputs/session28/p2"
NUMBERS_PART = REPO / "outputs/session28/numbers_parts/p2.json"

PRED_FAMILY = "jepa_tf_noc_d64_s42"

# The latent recovery rule MIRRORS the physical v2 clock (physics_prep.occupancy_recovery)
# so the gate compares like with like: same occupancy semantics, only the underlying signal
# differs (tube membership vs enstrophy-band membership). The defaults are matched to the
# physical clock's null-calibrated rule (theta = 0.8, occ_window = 56, occ_min_window = 28;
# physics_summary.json p2_recovery_v2.rule). The subharmonic shedding period is
# 1/(St_sub*dt_tc) ~= 59 frames (St_sub ~ 0.3383, D180); the physical clock rounds this to a
# 56-frame occupancy window, and we keep that window for parity. The window is exposed via
# --latent-dwell-frames; the master plan's "sustained one period" is the occupancy window.
SUBHARMONIC_PERIOD_FRAMES = int(round(1.0 / (pp.ST_SUB_FALLBACK * pp.DT_TC)))  # 59
LATENT_DWELL_FRAMES_DEFAULT = pp.OCC_WINDOW_DEFAULT  # 56, matched to the physical clock
LATENT_OCC_THETA_DEFAULT = 0.8  # matched to the physical v2 null-calibrated theta
LATENT_OCC_MIN_WINDOW_DEFAULT = pp.OCC_MIN_WINDOW_DEFAULT  # 28, matched to the physical clock
TUBE_RADIUS_PCTL = 95.0  # q95 of the baseline orbit's own Mahalanobis (and Euclidean) spread
GP2_THRESHOLD = 0.7  # D191: Spearman >= 0.7 -> latent carries the clock

# Physical-clock status taxonomy (physics_prep v2). recovered_short_window counts as
# recovered (a dwell was certified on the available remainder); only "censored" is censored.
PHYS_RECOVERED_STATUSES = ("recovered", "recovered_short_window")


# --------------------------------------------------------------------------- tube geometry
@dataclass
class LatentTube:
    """Baseline limit-cycle tube: orbit geometry + the Mahalanobis / Euclidean radii.

    Attributes:
        geom: s46_regen.OrbitGeometry (mean, ridged whitening, diameters, effective dim).
        orbit_w: the whitened settled-Baseline orbit cloud (n_pts, d), cached for distances.
        orbit: the raw settled-Baseline orbit cloud (n_pts, d), cached for Euclidean distances.
        maha_radius: q95 of the baseline orbit's OWN Mahalanobis distance to its distribution.
        euclid_radius: q95 of the baseline orbit's OWN min-Euclidean spread (cross-check).
        radius_pctl: the percentile used for both radii (TUBE_RADIUS_PCTL).
    """

    geom: "s46.OrbitGeometry"
    orbit_w: np.ndarray
    orbit: np.ndarray
    maha_radius: float
    euclid_radius: float
    radius_pctl: float = TUBE_RADIUS_PCTL


def mahalanobis_to_distribution(points: np.ndarray, geom: "s46.OrbitGeometry") -> np.ndarray:
    """Mahalanobis distance of each point to the orbit DISTRIBUTION (mean + whitening).

    This is the distance to the orbit's Gaussian model (||W (x - mu)||), NOT the min
    distance to the discrete cloud; the tube is "inside q95 of the orbit's own Mahalanobis
    spread", so the recovery test and the radius calibration use the same distance-to-
    distribution definition for consistency.
    """
    pw = (np.asarray(points, dtype=np.float64) - geom.mean) @ geom.whiten
    return np.sqrt((pw**2).sum(axis=1))


def build_latent_tube(
    train: dict, baseline_cid: str, radius_pctl: float = TUBE_RADIUS_PCTL
) -> LatentTube:
    """Construct the baseline limit-cycle tube from the family's TRAIN latents.

    Reuses s46_regen.settled_baseline_orbit (settled global frame >= 400 D180) and
    s46_regen._orbit_geometry (ridged covariance whitening). The Mahalanobis tube radius is
    the radius_pctl-th percentile of the Mahalanobis distance of each settled-Baseline frame
    to the orbit distribution; the Euclidean tube radius is the radius_pctl-th percentile of
    each settled-Baseline frame's MIN Euclidean distance to the rest of the cloud (a robust
    "thickness" of the cloud), carried as the well-conditioned cross-check.
    """
    orbit = s46.settled_baseline_orbit(train, baseline_cid)
    geom = s46._orbit_geometry(orbit)
    orbit_w = (orbit - geom.mean) @ geom.whiten
    maha_self = mahalanobis_to_distribution(orbit, geom)
    maha_radius = float(np.percentile(maha_self, radius_pctl))
    # Euclidean self-spread: min distance of each orbit point to the OTHER orbit points.
    dmat = np.sqrt(((orbit[:, None, :] - orbit[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(dmat, np.inf)
    euclid_self = dmat.min(axis=1)
    euclid_radius = float(np.percentile(euclid_self, radius_pctl))
    return LatentTube(
        geom=geom,
        orbit_w=orbit_w,
        orbit=orbit,
        maha_radius=max(maha_radius, 1e-12),
        euclid_radius=max(euclid_radius, 1e-12),
        radius_pctl=radius_pctl,
    )


# ------------------------------------------------------------------- tube-return clock rule
def tube_return_clock(inside: np.ndarray, start: int, dwell: int) -> tuple[str, int | None]:
    """STRICT-dwell tube-return reference: first t* inside the tube that dwells >= dwell.

    This is the theta = 1.0 limit (every frame of the dwell window must be inside). The
    PRODUCTION latent clock instead reuses physics_prep.occupancy_recovery (theta < 1) so its
    recovery semantics are byte-identical to the physical v2 enstrophy clock it is compared
    against in the gate; this strict variant is kept as the readable reference and is
    exercised by the unit tests (it equals occupancy_recovery at theta = 1.0).

    `inside` is a boolean per-frame trace (True = latent is inside the tube). t* itself must
    be inside. Returns (status, t*):
        ("recovered", t*)               a full `dwell`-frame run inside was observed at t*;
        ("recovered_short_window", t*)  the latent re-enters at t* and stays inside until the
                                        record ends, but fewer than `dwell` frames remain
                                        (the full dwell cannot be certified, but no excursion
                                        was seen on the available remainder; counts as
                                        recovered, mirroring physics_prep's short-window);
        ("censored", None)              no qualifying t*.
    Scanning is ascending, so full-dwell verdicts always precede short-window ones.
    """
    inside = np.asarray(inside, dtype=bool)
    n = inside.size
    t = start
    short: int | None = None
    while t < n:
        if not inside[t]:
            t += 1
            continue
        e = t
        while e < n and inside[e]:
            e += 1
        run = e - t
        if run >= dwell:
            return "recovered", t
        if e == n and short is None:
            # ran into the record end without an excursion, but < dwell frames available.
            short = t
        t = e + 1
    if short is not None:
        return "recovered_short_window", short
    return "censored", None


def latent_recovery_for_encounter(
    z_traj: np.ndarray,
    impact: int,
    tube: LatentTube,
    metric: str,
    window: int = LATENT_DWELL_FRAMES_DEFAULT,
    theta: float = LATENT_OCC_THETA_DEFAULT,
    min_window: int = LATENT_OCC_MIN_WINDOW_DEFAULT,
) -> tuple[str, int | None, np.ndarray]:
    """Latent recovery status + tau (absolute frame) + per-frame distance for one encounter.

    The recovery rule is physics_prep.occupancy_recovery applied to the tube-membership
    signal, so it is byte-identical in semantics to the physical v2 enstrophy clock: the
    latent has "recovered" at the first post-impact frame t* that is itself inside the tube
    AND whose [t*, t*+window) window has occupancy (fraction of frames inside) >= theta; if
    < window but >= min_window frames remain it is recovered_short_window; else censored. The
    "band" passed to occupancy_recovery is [-inf, radius] on the distance trace, so "inside
    the band" == "inside the tube".

    metric == "maha": distance-to-distribution Mahalanobis, radius = tube.maha_radius.
    metric == "euclid": min Euclidean distance to the orbit cloud, radius = tube.euclid_radius
    (the well-conditioned cross-check). The scan starts at the impact frame.
    """
    z_traj = np.asarray(z_traj, dtype=np.float64)
    if metric == "maha":
        dist = mahalanobis_to_distribution(z_traj, tube.geom)
        radius = tube.maha_radius
    elif metric == "euclid":
        dist = s46._min_dist_to_cloud(z_traj, tube.orbit)
        radius = tube.euclid_radius
    else:
        raise ValueError(f"unknown metric {metric!r}")
    status, t = pp.occupancy_recovery(
        dist, impact, -np.inf, radius, theta, window=window, min_window=min_window
    )
    return status, t, dist


# ------------------------------------------------------------------------- physics loading
def load_physics(physics_npz: Path) -> dict:
    """Per-encounter physics rows keyed by (case_id, encounter_index)."""
    b = np.load(physics_npz, allow_pickle=True)
    cols = {k: b[k] for k in b.files}
    n = cols["case_id"].shape[0]
    rows: dict[tuple[str, int], dict] = {}
    for i in range(n):
        key = (str(cols["case_id"][i]), int(cols["encounter_index"][i]))
        rows[key] = {k: cols[k][i] for k in cols}
    return rows


def phys_is_recovered(status: str) -> bool:
    """True iff the physical v2 status is a recovered (non-censored) status."""
    return str(status) in PHYS_RECOVERED_STATUSES


# ---------------------------------------------------------------- latent clock over a split
def latent_clock_for_split(family: str, split: str, tube: LatentTube, cfg: "Config") -> list[dict]:
    """Per-encounter latent recovery rows for one split (maha headline + euclid cross-check)."""
    data = s46.load_family_split(family, split)
    z = data["z_full"]
    imp = data["impact_frame"]
    rows = []
    for i in range(z.shape[0]):
        impact = int(np.clip(imp[i], 0, z.shape[1] - 1))
        status_m, t_m, _ = latent_recovery_for_encounter(
            z[i], impact, tube, "maha", cfg.latent_dwell, cfg.occ_theta, cfg.occ_min_window
        )
        status_e, t_e, _ = latent_recovery_for_encounter(
            z[i], impact, tube, "euclid", cfg.latent_dwell, cfg.occ_theta, cfg.occ_min_window
        )
        rows.append(
            {
                "case_id": str(data["case_ids"][i]),
                "encounter_index": int(data["encounter_indices"][i]),
                "split": split,
                "impact_frame": impact,
                "latent_status_maha": status_m,
                "latent_tau_frame_maha": None if t_m is None else int(t_m),
                "latent_tau_tc_maha": None if t_m is None else (t_m - impact) * pp.DT_TC,
                "latent_recovered_maha": phys_is_recovered(status_m),
                "latent_status_euclid": status_e,
                "latent_tau_frame_euclid": None if t_e is None else int(t_e),
                "latent_tau_tc_euclid": None if t_e is None else (t_e - impact) * pp.DT_TC,
                "latent_recovered_euclid": phys_is_recovered(status_e),
            }
        )
    return rows


# ------------------------------------------------------------------------------ the gate
def spearman_stat(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho of (x, y); NaN-safe scalar wrapper for case_permutation_p.

    A constant input (e.g. a permutation that collapses one variable) yields an undefined
    rho; we map that to 0.0 (no rank association) rather than NaN so the permutation count
    stays well-defined.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ConstantInputWarning on degenerate permutations
        rho = spearmanr(
            np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        ).correlation
    return float(rho) if np.isfinite(rho) else 0.0


def gate_block(
    phys_tau: np.ndarray,
    lat_tau: np.ndarray,
    case_ids: np.ndarray,
    cfg: "Config",
    metric_name: str,
) -> dict:
    """Spearman + case-clustered CI + case-permutation p over the paired (recovered) taus.

    The CI is a case-clustered bootstrap of the Spearman: cases are resampled with
    replacement (stats_lib draw order) and the Spearman is recomputed on the pooled
    encounters of the resampled cases. The p-value is the case-permutation p
    (stats_lib.case_permutation_p), which permutes the physical-tau case blocks against the
    latent-tau case blocks; this is VALID for a correlation / trend statistic.
    """
    n = phys_tau.size
    rho = spearman_stat(phys_tau, lat_tau)
    n_cases = len(set(case_ids.tolist()))

    # Case-clustered bootstrap CI of the Spearman.
    rng = np.random.default_rng(cfg.boot_seed)
    uc = np.array(sorted(set(case_ids.tolist())))
    idx_by_case = {c: np.where(case_ids == c)[0] for c in uc}
    boot = np.empty(cfg.n_boot_case)
    for b in range(cfg.n_boot_case):
        pick = uc[rng.integers(0, len(uc), size=len(uc))]
        sel = np.concatenate([idx_by_case[c] for c in pick])
        boot[b] = spearman_stat(phys_tau[sel], lat_tau[sel])
    ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    # Case-permutation p (one-sided "greater"; a positive trend is the pre-registered claim).
    rng_perm = np.random.default_rng(cfg.perm_seed)
    perm = stats_lib.case_permutation_p(
        phys_tau,
        lat_tau,
        case_ids,
        spearman_stat,
        n_perm=cfg.n_perm,
        rng=rng_perm,
        alternative="greater",
    )
    return {
        "metric": metric_name,
        "spearman_rho": rho,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "ci_method": "case-clustered bootstrap of the Spearman (n_boot_case)",
        "case_permutation_p_one_sided_greater": perm["p"],
        "n_perm": perm["n_perm"],
        "n_encounters": int(n),
        "n_cases": int(n_cases),
        "passes_gp2": bool(rho >= GP2_THRESHOLD),
    }


# ------------------------------------------------------------------------- assemble results
def fraction_block(statuses: list[str], recovered_predicate) -> dict:
    """Counts + fractions for a status list under a recovered/censored predicate."""
    n = len(statuses)
    from collections import Counter

    counts = Counter(statuses)
    n_rec = sum(1 for s in statuses if recovered_predicate(s))
    return {
        "n": n,
        "n_recovered": int(n_rec),
        "n_censored": int(n - n_rec),
        "frac_recovered": (n_rec / n) if n else float("nan"),
        "frac_censored": ((n - n_rec) / n) if n else float("nan"),
        "status_counts": dict(counts),
    }


def latent_return_diagnostic(family: str, tube: LatentTube, split: str = "test_b") -> dict:
    """Closest-approach diagnostic: how near does the gusted latent get to the tube?

    For each encounter of `split`, the minimum post-impact Euclidean distance to the orbit
    cloud (in orbit-diameter units) and the minimum post-impact Mahalanobis-to-distribution
    are recorded, plus the relaxation ratio (last-frame / impact-frame Euclidean distance).
    This explains a 0-recovery latent clock without hand-waving: if the closest approach is
    many orbit-diameters away, the tube (q95 of the orbit's OWN spread) is simply unreachable
    within the window, which is the honest mechanism, not a coding artefact.
    """
    data = s46.load_family_split(family, split)
    z = data["z_full"]
    imp = data["impact_frame"]
    euclid_diam = tube.geom.euclid_diameter
    min_euclid_diam, min_maha, relax = [], [], []
    for i in range(z.shape[0]):
        impact = int(np.clip(imp[i], 0, z.shape[1] - 1))
        de = s46._min_dist_to_cloud(z[i], tube.orbit)[impact:]
        dm = mahalanobis_to_distribution(z[i], tube.geom)[impact:]
        min_euclid_diam.append(float(de.min()) / euclid_diam)
        min_maha.append(float(dm.min()))
        relax.append(float(de[-1] / max(de[0], 1e-9)))
    min_euclid_diam = np.array(min_euclid_diam)
    min_maha = np.array(min_maha)
    relax = np.array(relax)
    return {
        "split": split,
        "n": int(z.shape[0]),
        "orbit_euclid_diameter": euclid_diam,
        "euclid_tube_radius_in_diam": tube.euclid_radius / euclid_diam,
        "maha_tube_radius": tube.maha_radius,
        "median_min_euclid_dist_in_diam": float(np.median(min_euclid_diam)),
        "min_min_euclid_dist_in_diam": float(min_euclid_diam.min()),
        "median_min_maha_dist": float(np.median(min_maha)),
        "median_relaxation_ratio": float(np.median(relax)),
        "n_reaching_within_1_diam": int((min_euclid_diam <= 1.0).sum()),
        "n_reaching_within_2_diam": int((min_euclid_diam <= 2.0).sum()),
    }


def physical_map_data(phys_rows: dict, split: str) -> dict:
    """tau_rec(G, D, Y) map data + tau_rec vs |G| for one split (physical v2 clock)."""
    recs = [r for r in phys_rows.values() if str(r["split"]) == split]
    recs.sort(key=lambda r: (str(r["case_id"]), int(r["encounter_index"])))
    out = {
        "case_id": [str(r["case_id"]) for r in recs],
        "encounter_index": [int(r["encounter_index"]) for r in recs],
        "G": [float(r["G"]) for r in recs],
        "D": [float(r["D"]) for r in recs],
        "Y": [float(r["Y"]) for r in recs],
        "abs_G": [abs(float(r["G"])) for r in recs],
        "tau_rec_tc": [None if bool(r["censored_v2"]) else float(r["tau_rec_v2_tc"]) for r in recs],
        "recovery_status_v2": [str(r["recovery_status_v2"]) for r in recs],
        "censored": [bool(r["censored_v2"]) for r in recs],
    }
    return out


@dataclass
class Config:
    family: str = PRED_FAMILY
    latent_dwell: int = LATENT_DWELL_FRAMES_DEFAULT  # occupancy window (frames)
    occ_theta: float = LATENT_OCC_THETA_DEFAULT  # occupancy threshold (matched to physical)
    occ_min_window: int = LATENT_OCC_MIN_WINDOW_DEFAULT  # short-window floor
    radius_pctl: float = TUBE_RADIUS_PCTL
    boot_seed: int = 0
    perm_seed: int = 0
    n_boot_case: int = stats_lib.N_BOOT_CASE
    n_perm: int = 10000
    skip: tuple[str, ...] = field(default_factory=tuple)


def run_p2(cfg: Config, physics_npz: Path, split_manifest: Path) -> dict:
    """Compute the full P2 latent-clock tie-in and Gate GP2."""
    manifest = json.loads(split_manifest.read_text())
    baseline_cid = s46.baseline_case_id(manifest)
    phys_rows = load_physics(physics_npz)

    train = s46.load_family_split(cfg.family, "train")
    tube = build_latent_tube(train, baseline_cid, cfg.radius_pctl)

    # Latent clock on test_b (the gate split) and on every latent split present for the
    # record. The latent NPZ uses the v2.1 split filenames (train, test_a, test_b, test_c;
    # test_a is the renamed val per CLAUDE.md); the join to the physical clock is by
    # (case_id, encounter_index), not by split name, so only the present files are read.
    fam_dir = LATENTS_ROOT / cfg.family
    candidate_splits = ("train", "val", "test_a", "test_b", "test_c")
    lat_rows_by_split = {
        split: latent_clock_for_split(cfg.family, split, tube, cfg)
        for split in candidate_splits
        if (fam_dir / f"{split}.npz").is_file()
    }
    if "test_b" not in lat_rows_by_split:
        raise FileNotFoundError(f"no test_b latents under {fam_dir} (the gate split)")

    # ---- Gate: align physical and latent clocks on test_b, recovered-in-BOTH only. ----
    tb_lat = lat_rows_by_split["test_b"]
    phys_status_tb, lat_status_maha_tb, lat_status_euclid_tb = [], [], []
    paired = {"maha": [], "euclid": []}  # (phys_tau_tc, lat_tau_tc, case_id)
    per_encounter = []
    for lr in sorted(tb_lat, key=lambda r: (r["case_id"], r["encounter_index"])):
        key = (lr["case_id"], lr["encounter_index"])
        pr = phys_rows.get(key)
        if pr is None:
            continue
        phys_rec = phys_is_recovered(str(pr["recovery_status_v2"]))
        phys_tau_tc = None if bool(pr["censored_v2"]) else float(pr["tau_rec_v2_tc"])
        phys_status_tb.append(str(pr["recovery_status_v2"]))
        lat_status_maha_tb.append(lr["latent_status_maha"])
        lat_status_euclid_tb.append(lr["latent_status_euclid"])
        per_encounter.append(
            {
                "case_id": lr["case_id"],
                "encounter_index": lr["encounter_index"],
                "G": float(pr["G"]),
                "D": float(pr["D"]),
                "Y": float(pr["Y"]),
                "phys_status": str(pr["recovery_status_v2"]),
                "phys_recovered": bool(phys_rec),
                "phys_tau_tc": phys_tau_tc,
                "latent_status_maha": lr["latent_status_maha"],
                "latent_recovered_maha": lr["latent_recovered_maha"],
                "latent_tau_tc_maha": lr["latent_tau_tc_maha"],
                "latent_status_euclid": lr["latent_status_euclid"],
                "latent_recovered_euclid": lr["latent_recovered_euclid"],
                "latent_tau_tc_euclid": lr["latent_tau_tc_euclid"],
            }
        )
        if phys_rec:
            for m in ("maha", "euclid"):
                if lr[f"latent_recovered_{m}"]:
                    paired[m].append((phys_tau_tc, float(lr[f"latent_tau_tc_{m}"]), lr["case_id"]))

    gates = {}
    for m in ("maha", "euclid"):
        if len(paired[m]) >= 3:
            phys_tau = np.array([p[0] for p in paired[m]], dtype=np.float64)
            lat_tau = np.array([p[1] for p in paired[m]], dtype=np.float64)
            cids = np.array([p[2] for p in paired[m]])
            gates[m] = gate_block(phys_tau, lat_tau, cids, cfg, m)
        else:
            gates[m] = {
                "metric": m,
                "spearman_rho": float("nan"),
                "n_encounters": len(paired[m]),
                "passes_gp2": False,
                "note": "fewer than 3 recovered-in-both pairs; Spearman undefined",
            }

    headline = gates["maha"]
    gp2_pass = bool(headline.get("passes_gp2", False))
    n_pairs = headline["n_encounters"]
    if gp2_pass:
        branch = "latent-clock claim ENTERS S4.4 with the tau_rec map (Spearman >= 0.7)"
    elif n_pairs < 3:
        branch = (
            "latent-clock sentence DROPPED; the latent clock recovers too few encounters for "
            f"a meaningful Spearman (recovered-in-both n={n_pairs}). Physical DNS maps stand "
            "as the contribution."
        )
    else:
        branch = "latent-clock sentence DROPPED; physical DNS maps stand as the contribution"

    # ---- Fractions for both clocks (test_b; the gate split). ----
    fractions_tb = {
        "physical_v2": fraction_block(phys_status_tb, phys_is_recovered),
        "latent_maha": fraction_block(lat_status_maha_tb, phys_is_recovered),
        "latent_euclid": fraction_block(lat_status_euclid_tb, phys_is_recovered),
    }

    # ---- Latent closest-approach diagnostic (explains a 0-recovery latent clock). ----
    return_diag = latent_return_diagnostic(cfg.family, tube, "test_b")

    # ---- Map data (physical clock) over the envelope, per split. ----
    maps = {split: physical_map_data(phys_rows, split) for split in ("test_b", "test_c")}

    return {
        "config": {
            "family": cfg.family,
            "latent_dwell_frames": cfg.latent_dwell,
            "latent_dwell_tc": cfg.latent_dwell * pp.DT_TC,
            "latent_occ_theta": cfg.occ_theta,
            "latent_occ_min_window": cfg.occ_min_window,
            "latent_dwell_note": (
                "Latent recovery uses physics_prep.occupancy_recovery on the tube-membership "
                f"signal, MATCHED to the physical v2 clock (theta={cfg.occ_theta}, "
                f"window={cfg.latent_dwell}, min_window={cfg.occ_min_window}). The subharmonic "
                f"shedding period is 1/(St_sub*dt_tc)~={SUBHARMONIC_PERIOD_FRAMES} frames "
                f"(St_sub={pp.ST_SUB_FALLBACK:.4f}); the physical clock rounds this to a "
                "56-frame occupancy window and we keep that window for parity."
            ),
            "tube_radius_pctl": cfg.radius_pctl,
            "gp2_threshold": GP2_THRESHOLD,
            "n_boot_case": cfg.n_boot_case,
            "n_perm": cfg.n_perm,
            "boot_seed": cfg.boot_seed,
            "perm_seed": cfg.perm_seed,
        },
        "physical_clock_source": {
            "npz": str(physics_npz),
            "columns": ["recovery_status_v2", "tau_rec_v2_frames", "tau_rec_v2_tc", "censored_v2"],
            "rule": (
                "v2 settled-Baseline band + occupancy dwell (physics_prep P2 v2); "
                "recovered_short_window counts as recovered, only 'censored' is censored"
            ),
        },
        "latent_tube": {
            "baseline_case_id": baseline_cid,
            "orbit_n_points": tube.geom.n_pts,
            "orbit_effective_dim": tube.geom.effective_dim,
            "maha_radius_q95": tube.maha_radius,
            "euclid_radius_q95": tube.euclid_radius,
            "definition": (
                "settled undisturbed Baseline predictive-latent orbit (global raw frame "
                ">= 400, t/c >= 20; D180); tube = Mahalanobis-to-distribution <= q95 of the "
                "orbit's own Mahalanobis spread, sustained one subharmonic period. Mahalanobis "
                "is ill-conditioned (orbit effective dim ~3.6 of 64; see s46_regen); the gate "
                "statistic is a RANK correlation and a well-conditioned Euclidean tube is "
                "carried as a cross-check."
            ),
        },
        "gate_gp2": {
            "headline_metric": "maha",
            "headline": headline,
            "euclid_crosscheck": gates["euclid"],
            "passes_gp2": gp2_pass,
            "branch": branch,
        },
        "fractions_test_b": fractions_tb,
        "latent_return_diagnostic": return_diag,
        "per_encounter_test_b": per_encounter,
        "physical_map_data": maps,
    }


# --------------------------------------------------------------------------- numbers part
def build_numbers_part(results: dict) -> dict:
    """eval_all numbers part. Macros are ALPHABETIC only (P2 = 'Ptwo', GP2 = 'GPtwo')."""
    g = results["gate_gp2"]
    h = g["headline"]
    e = g["euclid_crosscheck"]
    fr = results["fractions_test_b"]
    perm_p = h.get("case_permutation_p_one_sided_greater", float("nan"))
    e_rho = e.get("spearman_rho", float("nan"))
    evaluable = h["n_encounters"] >= 3 and np.isfinite(h["spearman_rho"])
    numbers: dict[str, dict] = {}

    # When the gate is not evaluable (latent recovers too few encounters), the Spearman is
    # undefined: emit a clean "n/a" string macro (NaN/None would render badly in LaTeX or
    # fail emit_macros' printf formatting) and omit the CI keys. The branch/fraction macros
    # below carry the honest verdict numerically.
    spearman_rec = {
        "macro": "NumPtwoGPtwoSpearman",
        "n": h["n_encounters"],
        "split": "test_b",
        "observable": "latent tau_rec vs physical tau_rec (Mahalanobis tube)",
        "source": "p2_latent_clock.py",
    }
    if evaluable:
        spearman_rec.update(
            {
                "value": h["spearman_rho"],
                "fmt": "%.2f",
                "ci_lo": h.get("ci_lo", float("nan")),
                "ci_hi": h.get("ci_hi", float("nan")),
                "note": (
                    f"case-permutation one-sided p = {perm_p:.4g}; "
                    f"GP2 threshold {GP2_THRESHOLD}; passes={h['passes_gp2']}; "
                    f"recovered-in-both n={h['n_encounters']} over "
                    f"{h.get('n_cases', 'NA')} cases; Euclidean cross-check rho={e_rho:.2f}"
                ),
            }
        )
    else:
        spearman_rec.update(
            {
                "value": "n/a",
                "fmt": "%s",
                "note": (
                    f"GP2 not evaluable: latent recovers too few test_b encounters "
                    f"(recovered-in-both n={h['n_encounters']}) for a meaningful Spearman; "
                    f"GP2 threshold {GP2_THRESHOLD}; passes=False. Latent-clock claim dropped; "
                    "physical maps stand. See results.json latent_return_diagnostic."
                ),
            }
        )
    numbers["p2_gp2_latent_phys_spearman"] = spearman_rec
    numbers["p2_gp2_branch"] = {
        "macro": "NumPtwoGPtwoBranch",
        "value": 1.0 if g["passes_gp2"] else 0.0,
        "fmt": "%.0f",
        "split": "test_b",
        "source": "p2_latent_clock.py",
        "note": g["branch"],
    }
    numbers["p2_frac_recovered_physical"] = {
        "macro": "NumPtwoFracRecoveredPhysical",
        "value": fr["physical_v2"]["frac_recovered"],
        "fmt": "%.2f",
        "n": fr["physical_v2"]["n"],
        "split": "test_b",
        "observable": "physical v2 recovery clock",
        "source": "p2_latent_clock.py",
        "note": (
            f"{fr['physical_v2']['n_recovered']}/{fr['physical_v2']['n']} recovered, "
            f"{fr['physical_v2']['n_censored']} censored; "
            f"statuses {fr['physical_v2']['status_counts']}"
        ),
    }
    numbers["p2_frac_censored_physical"] = {
        "macro": "NumPtwoFracCensoredPhysical",
        "value": fr["physical_v2"]["frac_censored"],
        "fmt": "%.2f",
        "n": fr["physical_v2"]["n"],
        "split": "test_b",
        "observable": "physical v2 recovery clock",
        "source": "p2_latent_clock.py",
        "note": f"{fr['physical_v2']['n_censored']}/{fr['physical_v2']['n']} censored",
    }
    numbers["p2_frac_recovered_latent"] = {
        "macro": "NumPtwoFracRecoveredLatent",
        "value": fr["latent_maha"]["frac_recovered"],
        "fmt": "%.2f",
        "n": fr["latent_maha"]["n"],
        "split": "test_b",
        "observable": "latent recovery clock (Mahalanobis tube)",
        "source": "p2_latent_clock.py",
        "note": (
            f"{fr['latent_maha']['n_recovered']}/{fr['latent_maha']['n']} recovered, "
            f"{fr['latent_maha']['n_censored']} censored; "
            f"statuses {fr['latent_maha']['status_counts']}"
        ),
    }
    numbers["p2_frac_censored_latent"] = {
        "macro": "NumPtwoFracCensoredLatent",
        "value": fr["latent_maha"]["frac_censored"],
        "fmt": "%.2f",
        "n": fr["latent_maha"]["n"],
        "split": "test_b",
        "observable": "latent recovery clock (Mahalanobis tube)",
        "source": "p2_latent_clock.py",
        "note": f"{fr['latent_maha']['n_censored']}/{fr['latent_maha']['n']} censored",
    }
    return {"part": "p2", "numbers": numbers}


# ------------------------------------------------------------------------------- the figure
def make_figure(results: dict, out_dir: Path) -> None:
    """Physical tau_rec(G, |Y| shade) map + latent-vs-physical clock scatter with Spearman."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(REPO / "scripts" / "session21"))
    try:
        import figstyle

        figstyle.use_style()
        jepa_c = figstyle.FAMILY_COLOR["jepa"]
        oracle_c = figstyle.FAMILY_COLOR["oracle"]
        fig_w, fig_h = figstyle.figure_size(1.0, aspect=0.46)
    except Exception:
        jepa_c, oracle_c = "#1b7837", "#404040"
        fig_w, fig_h = 6.8, 3.1

    fig, (axm, axs) = plt.subplots(1, 2, figsize=(fig_w, fig_h))

    # ---- Left: physical tau_rec over the (|G|, D) envelope, test_b + test_c, |Y| as size.
    sc = None
    for split, marker, label in (("test_b", "o", "test B"), ("test_c", "^", r"test C ($|G|=4$)")):
        m = results["physical_map_data"][split]
        absg = np.array(m["abs_G"], dtype=float)
        d = np.array(m["D"], dtype=float)
        absy = np.abs(np.array(m["Y"], dtype=float))
        tau = np.array([np.nan if t is None else t for t in m["tau_rec_tc"]], dtype=float)
        cens = np.array(m["censored"], dtype=bool)
        sizes = 18 + 90 * absy
        rec = ~cens & np.isfinite(tau)
        if rec.any():
            sc = axm.scatter(
                absg[rec],
                d[rec],
                c=tau[rec],
                s=sizes[rec],
                marker=marker,
                cmap="viridis",
                edgecolors="k",
                linewidths=0.4,
                label=label,
                zorder=3,
            )
        if cens.any():
            axm.scatter(
                absg[cens],
                d[cens],
                facecolors="none",
                s=sizes[cens],
                marker=marker,
                edgecolors="0.4",
                linewidths=0.9,
                label=f"{label} censored",
                zorder=2,
            )
    axm.set_xlabel(r"$|G|$")
    axm.set_ylabel(r"$D$")
    axm.set_title(r"physical $\tau_{\rm rec}$ over the envelope")
    if sc is not None:
        cb = fig.colorbar(sc, ax=axm, fraction=0.046, pad=0.03)
        cb.set_label(r"$\tau_{\rm rec}$ ($t/c$ since impact)")
    axm.legend(loc="best", fontsize=6)

    # ---- Right: latent vs physical clock scatter, recovered-in-both, with the Spearman.
    pe = results["per_encounter_test_b"]
    xp, yl = [], []
    for r in pe:
        if r["phys_recovered"] and r["latent_recovered_maha"]:
            xp.append(r["phys_tau_tc"])
            yl.append(r["latent_tau_tc_maha"])
    xp = np.array(xp, dtype=float)
    yl = np.array(yl, dtype=float)
    h = results["gate_gp2"]["headline"]
    rho = h.get("spearman_rho", float("nan"))
    p = h.get("case_permutation_p_one_sided_greater", float("nan"))
    verdict = _verdict_word(results["gate_gp2"])
    if xp.size:
        axs.scatter(xp, yl, color=jepa_c, s=24, edgecolors="k", linewidths=0.4, zorder=3)
        lims = [min(xp.min(), yl.min()) - 0.2, max(xp.max(), yl.max()) + 0.2]
        axs.plot(lims, lims, color=oracle_c, lw=0.8, ls="--", zorder=1, label=r"$y=x$")
        axs.set_xlim(lims)
        axs.set_ylim(lims)
        axs.set_xlabel(r"physical $\tau_{\rm rec}$ ($t/c$)")
        axs.set_ylabel(r"latent $\tau_{\rm rec}$ ($t/c$)")
        axs.set_title(
            "latent vs physical clock\n"
            + rf"Spearman $\rho={rho:.2f}$ (n={h['n_encounters']}, {verdict})"
        )
        txt = f"GP2 {GP2_THRESHOLD}: {verdict}\np={p:.3g}"
    else:
        # Degenerate gate: the latent recovers no test_b encounter. Show the honest
        # mechanism (closest-approach distribution) instead of an empty scatter.
        rd = results["latent_return_diagnostic"]
        vals = np.array([rd["median_min_euclid_dist_in_diam"], rd["min_min_euclid_dist_in_diam"]])
        axs.bar(["median", "closest"], vals, color=jepa_c, edgecolor="k", linewidth=0.4, width=0.6)
        axs.axhline(
            rd["euclid_tube_radius_in_diam"],
            color=oracle_c,
            ls="--",
            lw=0.9,
            label="tube radius (q95)",
        )
        axs.set_ylabel("min post-impact distance\nto orbit (orbit-diameters)")
        axs.set_title(
            "latent never re-enters the tube\n"
            + rf"GP2 not evaluable (recovered-in-both n={h['n_encounters']})"
        )
        axs.legend(loc="best", fontsize=6)
        txt = (
            f"GP2 {GP2_THRESHOLD}: dropped\nlatent recovers 0/{rd['n']}\n"
            f"closest {rd['min_min_euclid_dist_in_diam']:.1f} diam"
        )
    axs.text(
        0.04,
        0.96,
        txt,
        transform=axs.transAxes,
        va="top",
        ha="left",
        fontsize=6,
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85),
    )

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig_p2_recovery.{ext}")
    plt.close(fig)


# -------------------------------------------------------------------------------- README
def _verdict_word(g: dict) -> str:
    """Verdict word for the README: PASS, BELOW 0.7, or NOT EVALUABLE (too few pairs)."""
    if g["passes_gp2"]:
        return "PASS"
    if g["headline"]["n_encounters"] < 3:
        return "NOT EVALUABLE (too few recovered-in-both encounters)"
    return "BELOW 0.7"


def write_readme(results: dict, out_dir: Path) -> None:
    """Human-readable P2 summary; the gate verdict is stated plainly."""
    g = results["gate_gp2"]
    h = g["headline"]
    e = g["euclid_crosscheck"]
    fr = results["fractions_test_b"]
    cfg = results["config"]
    tube = results["latent_tube"]

    def fmt_p(d):
        return f"{d.get('case_permutation_p_one_sided_greater', float('nan')):.4g}"

    lines = [
        "# Session 28 / Physics Track P2: the latent recovery clock and Gate GP2",
        "",
        "The physical recovery clock tau_rec (v2: wake enstrophy re-enters the settled-",
        "Baseline band and dwells one period) is read from",
        "outputs/session28/physics/per_encounter_physics.npz. This script computes the LATENT",
        "recovery clock from the predictive (jepa_tf_noc) latent trajectories and correlates",
        "it with the physical one (Gate GP2). NOT git-committed; left for review.",
        "",
        "## Latent tube",
        "",
        f"- Baseline case: {tube['baseline_case_id']}; settled orbit points: "
        f"{tube['orbit_n_points']} (effective dim {tube['orbit_effective_dim']:.1f} of 64).",
        f"- Mahalanobis tube radius (q{cfg['tube_radius_pctl']:.0f}): "
        f"{tube['maha_radius_q95']:.3f}.",
        f"- Euclidean cross-check radius (q{cfg['tube_radius_pctl']:.0f}): "
        f"{tube['euclid_radius_q95']:.3f}.",
        f"- Recovery rule: occupancy window {cfg['latent_dwell_frames']} frames "
        f"({cfg['latent_dwell_tc']:.2f} t/c), theta {cfg['latent_occ_theta']}, min-window "
        f"{cfg['latent_occ_min_window']}. {cfg['latent_dwell_note']}",
        "- The orbit is very low-rank, so the Mahalanobis whitening is ill-conditioned (see",
        "  s46_regen). The gate statistic is a RANK correlation (insensitive to whitening",
        "  magnitude) and a well-conditioned Euclidean tube is carried as a cross-check.",
        "",
        "## Gate GP2 (Spearman of latent vs physical tau_rec, recovered-in-both, test_b)",
        "",
        f"- HEADLINE (Mahalanobis tube): Spearman rho = {h['spearman_rho']:.3f} "
        + (
            f"[{h.get('ci_lo', float('nan')):.2f}, {h.get('ci_hi', float('nan')):.2f}] "
            f"(case-clustered CI), case-permutation p = {fmt_p(h)}, "
            f"n = {h['n_encounters']} encounters over {h.get('n_cases', 'NA')} cases."
            if "ci_lo" in h
            else f"(n = {h['n_encounters']}; {h.get('note', '')})."
        ),
        f"- Euclidean cross-check: Spearman rho = {e.get('spearman_rho', float('nan')):.3f}"
        + (
            f" [{e.get('ci_lo', float('nan')):.2f}, {e.get('ci_hi', float('nan')):.2f}], "
            f"p = {fmt_p(e)}, n = {e['n_encounters']}."
            if "ci_lo" in e
            else f" (n = {e['n_encounters']}; {e.get('note', '')})."
        ),
        "",
        f"GP2 threshold = {GP2_THRESHOLD}. **VERDICT: " + _verdict_word(g) + "**.",
        "",
        g["branch"].rstrip(".") + ".",
        "",
    ]
    if not g["passes_gp2"]:
        if h["n_encounters"] < 3:
            lines += [
                "Stated plainly: the latent recovery clock recovers too few test_b encounters",
                f"({h['n_encounters']} recovered-in-both with the physical clock) for a",
                "meaningful Spearman, so GP2 is not evaluable as a correlation. The",
                "latent-clock sentence is DROPPED from S4.4. The physical tau_rec(G, D, Y) maps",
                "stand as DNS physics and remain a contribution. See the diagnostic below for",
                "the mechanism (the latent never re-enters the tight baseline tube).",
                "",
            ]
        else:
            lines += [
                "Stated plainly: the latent recovery clock does NOT track the physical recovery",
                "clock at the GP2 bar on test_b. The latent-clock sentence is DROPPED from S4.4.",
                "The physical tau_rec(G, D, Y) maps stand as DNS physics and remain a",
                "contribution.",
                "",
            ]
    rd = results["latent_return_diagnostic"]
    lines += [
        "## Latent closest-approach diagnostic (test_b)",
        "",
        "Why the latent clock recovers as it does: the settled-Baseline orbit is a thin,",
        f"low-rank tube (Euclidean diameter {rd['orbit_euclid_diameter']:.2f} latent units; the",
        f"q95-self-spread tube radius is {rd['euclid_tube_radius_in_diam']:.2f} orbit-diameters",
        "Euclidean). The gusted test_b latents approach it only weakly:",
        "",
        f"- median minimum post-impact distance to the orbit: "
        f"{rd['median_min_euclid_dist_in_diam']:.2f} orbit-diameters "
        f"(closest any encounter gets: {rd['min_min_euclid_dist_in_diam']:.2f}).",
        f"- median minimum post-impact Mahalanobis-to-distribution: "
        f"{rd['median_min_maha_dist']:.1f} (tube radius {rd['maha_tube_radius']:.1f}); the",
        "  whitening inflates off-orbit directions, so the Mahalanobis tube is unreachable.",
        f"- median relaxation ratio (last-frame / impact-frame distance): "
        f"{rd['median_relaxation_ratio']:.2f} (the latent barely drifts back).",
        f"- encounters reaching within 1 / 2 orbit-diameters at any post-impact frame: "
        f"{rd['n_reaching_within_1_diam']} / {rd['n_reaching_within_2_diam']} of {rd['n']}.",
        "",
        "This is the honest mechanism, not a coding artefact: the latent does not return to",
        "the baseline limit-cycle tube within the 120-frame window, consistent with the short",
        "release cadence (D153) that censors most of the PHYSICAL clock too, and with the s46",
        "Q2 finding that the predictive latent does not sit close to the baseline orbit.",
        "",
    ]
    lines += [
        "## Recovered vs censored fractions (test_b)",
        "",
        "| clock | n | recovered | censored | frac recovered |",
        "|---|---|---|---|---|",
    ]
    for key, label in (
        ("physical_v2", "physical (v2)"),
        ("latent_maha", "latent (Mahalanobis)"),
        ("latent_euclid", "latent (Euclidean)"),
    ):
        b = fr[key]
        lines.append(
            f"| {label} | {b['n']} | {b['n_recovered']} | {b['n_censored']} | "
            f"{b['frac_recovered']:.2f} |"
        )
    lines += [
        "",
        "Status taxonomy: 'recovered_short_window' counts as recovered (a dwell was certified",
        "on the available remainder); only 'censored' is censored. Censoring is never silently",
        "excluded; encounters censored under EITHER clock are simply not in the paired",
        "correlation, and the counts above make that explicit.",
        "",
        "## Map data",
        "",
        "Physical tau_rec(G, D, Y) over test_b/test_c and tau_rec vs |G| are in results.json",
        "(physical_map_data) and drawn in fig_p2_recovery.{pdf,png} (left panel: the envelope",
        "map; right panel: the latent-vs-physical clock scatter with the Spearman).",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> None:
    """Run P2, write results.json + README.md + fig_p2_recovery + the numbers part."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", type=str, default=PRED_FAMILY)
    parser.add_argument("--physics-npz", type=Path, default=PHYSICS_NPZ)
    parser.add_argument("--split-manifest", type=Path, default=SPLIT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--numbers-part", type=Path, default=NUMBERS_PART)
    parser.add_argument(
        "--latent-dwell-frames",
        type=int,
        default=LATENT_DWELL_FRAMES_DEFAULT,
        help="Occupancy window for the latent recovery rule (subharmonic period ~ 59; "
        "default 56, matched to the physical clock).",
    )
    parser.add_argument(
        "--occ-theta",
        type=float,
        default=LATENT_OCC_THETA_DEFAULT,
        help="Occupancy threshold (matched to the physical v2 clock; default 0.8).",
    )
    parser.add_argument(
        "--occ-min-window",
        type=int,
        default=LATENT_OCC_MIN_WINDOW_DEFAULT,
        help="Shortest remainder window that can certify recovery (default 28).",
    )
    parser.add_argument("--radius-pctl", type=float, default=TUBE_RADIUS_PCTL)
    parser.add_argument("--n-boot-case", type=int, default=stats_lib.N_BOOT_CASE)
    parser.add_argument("--n-perm", type=int, default=10000)
    parser.add_argument("--boot-seed", type=int, default=0)
    parser.add_argument("--perm-seed", type=int, default=0)
    parser.add_argument("--no-figure", action="store_true", help="Skip the figure.")
    args = parser.parse_args(argv)

    cfg = Config(
        family=args.family,
        latent_dwell=args.latent_dwell_frames,
        occ_theta=args.occ_theta,
        occ_min_window=args.occ_min_window,
        radius_pctl=args.radius_pctl,
        boot_seed=args.boot_seed,
        perm_seed=args.perm_seed,
        n_boot_case=args.n_boot_case,
        n_perm=args.n_perm,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    results = run_p2(cfg, args.physics_npz, args.split_manifest)
    results["generated_iso"] = datetime.now(timezone.utc).isoformat()
    results["split_manifest"] = str(args.split_manifest)
    results["physics_npz"] = str(args.physics_npz)
    (args.out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"[p2] results.json written ({time.time() - t0:.1f}s)")

    part = build_numbers_part(results)
    args.numbers_part.parent.mkdir(parents=True, exist_ok=True)
    args.numbers_part.write_text(json.dumps(part, indent=2))
    print(f"[p2] numbers part written ({len(part['numbers'])} numbers)")

    write_readme(results, args.out_dir)
    print("[p2] README.md written")

    if not args.no_figure:
        make_figure(results, args.out_dir)
        print("[p2] fig_p2_recovery.{pdf,png} written")

    g = results["gate_gp2"]
    h = g["headline"]
    print(
        f"[p2] GP2: Spearman rho = {h['spearman_rho']:.3f} (n={h['n_encounters']}), "
        f"threshold {GP2_THRESHOLD} -> {'PASS' if g['passes_gp2'] else 'BELOW 0.7'}"
    )
    print(f"[p2] branch: {g['branch']}")


if __name__ == "__main__":
    main()
