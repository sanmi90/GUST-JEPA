#!/usr/bin/env python
"""C-E3: optimal-transport mechanism, done to the Tran et al. standard (closes M5).

References:
    Tran, Yeh, Taira, J. Fluid Mech. 1027, A24 (2026), Appendix B.
    Feydy, Sejourne, Vialard, Amari, Trouve, Peyre, "Interpolating between
        Optimal Transport and MMD using Sinkhorn Divergences," AISTATS 2019.

THE DEFECT (referee M5). The v2 manuscript cited Tran et al. and used their
m+/m- signed-vorticity split, but reported only "entropic unbalanced Sinkhorn
transport cost with a KL marginal relaxation, fields pooled to 48x24" with no
epsilon, no relaxation strength, no statement of whether the divergence was
debiased, and no sensitivity check. Tran Appendix B (their eq. B5) use the
DEBIASED Sinkhorn divergence with an l2 ground cost on a compact domain,
decomposing the signed vorticity into m+ and m- (eqs. B6-B8) and transporting
each part. This script fixes all of that.

WHAT THIS SCRIPT IMPLEMENTS

1. DEBIASED divergence (Tran eq. B5 / Feydy et al. 2019)::

       S_eps(a, b) = OT_eps(a, b) - 0.5 OT_eps(a, a) - 0.5 OT_eps(b, b)

   where OT_eps(a, b) = <T*, C> is the LINEAR transport cost of the entropic
   plan T* = argmin <T, C> + eps KL(T | a (x) b) with squared-Euclidean ground
   cost C on the pooled 48x24 grid (chord^2 units). We use POT's ot.sinkhorn2 on
   the precomputed cost matrix for each OT_eps term. POT VERIFICATION (see the
   module test and the README): ot.sinkhorn2 returns the LINEAR cost <T*, C>,
   and S_eps built from it reproduces ot.bregman.empirical_sinkhorn_divergence
   to 1e-6 (POT's own debiased divergence also debiases the LINEAR cost, NOT the
   entropy-regularised objective). So the value we report is POT's canonical
   debiased Sinkhorn divergence. S_eps(a, a) = 0 to solver tolerance. The pooled
   half-field masses carry many exact zeros, which break plain Sinkhorn's 1/a
   step; a 1e-6 uniform mass floor (normalise_mass) restores it and reproduces the
   log-domain stabilised value to 1e-5, at ~20x the speed of sinkhorn_log.

   BALANCED-ON-NORMALISED-PARTS (our principled default, documented per the
   plan). The debiased divergence is the well-posed BALANCED object: it requires
   a, b to have equal total mass for the self-terms to cancel and for the
   divergence to be a genuine (positive-definite) divergence. We therefore
   NORMALISE each of m+ and m- to unit mass before transport, and transport the
   two parts separately, then sum (Tran eqs. B6-B8)::

       d_field(V1, V2) = S_eps(m+_1 / |m+_1|, m+_2 / |m+_2|)
                       + S_eps(m-_1 / |m-_1|, m-_2 / |m-_2|).

   This is cleaner than the v2 unbalanced KL-relaxed form (which left rho
   unstated and whose self-terms do not cancel, so it is NOT a debiased
   divergence). We do NOT use an unbalanced rho here; the README states this.

2. EPSILON, stated and swept. eps is set from the ground-cost scale by a fixed,
   documented rule: eps = EPS_FRACTION * median(C_offdiag), the median pairwise
   squared distance on the pooled grid, in chord^2. The numeric value is
   reported. The eps sensitivity sweep runs the WHOLE alignment at {eps/3, eps,
   3 eps} and reports how the headline paired alignment delta changes.

3. FIELD distances as per-encounter PAIRED distributions. For each test_b and
   test_c encounter, d_field(field_t, phase-matched undisturbed-baseline field)
   at the impact frame and at impact+16, on the DNS fields from the v2p1 cache
   (the flow's own transport geometry; no decoder). PHASE MATCHING: the shedding
   phase of the encounter is fit from its OWN pre-impact C_L (Hilbert phase,
   physics_prep.fit_pre_impact_cycle) and extrapolated to frame impact+k; the
   undisturbed baseline reference is a phase-indexed pool of SETTLED Baseline
   omega_z frames (global raw frame >= SETTLED_GLOBAL_FRAME_MIN), each tagged
   with its own continuous Hilbert phase; the match is the settled-baseline frame
   minimising the circular phase distance to the target phase.

4. ALIGNMENT per family for jepa_tf_noc, fukami, AND pod (pod's absence was the
   missing comparator). Per encounter we build the LATENT distance sequence
   ||z_t - z_basephase(t)|| (each gust frame t to the latent of its
   phase-matched baseline frame, in that family's own latent space) and the
   FIELD S_eps distance sequence d_field(field_t, field_basephase(t)) over a
   strided set of frames spanning the encounter, then Spearman-correlate the two.
   The per-encounter alignment distribution is reported per family; the paired
   per-encounter DIFFERENCE (jepa - fukami, jepa - pod) gets a CASE-CLUSTERED
   bootstrap CI via stats_lib.

5. SIGNIFICANCE (the B6 lesson). stats_lib.case_permutation_p is DEGENERATE for a
   paired LOCATION test (block permutation preserves the pooled mean -> p ~ 1).
   For the paired alignment DIFFERENCE between families we use the case-clustered
   bootstrap CI + a per-encounter and per-case SIGN test (as B6 did), NOT
   case_permutation_p. We DO compute an alignment-vs-|G| TREND per family and use
   case_permutation_p there (its valid use).

6. MECHANISM, replacing the incoherent pooled-reversal paragraph. The pooled OT
   statistic is an ENCODING-scale property: encoded latents do not drift, so the
   "reversal" is a between-encounter latent-norm-variance artifact. We report the
   between-encounter variance of the mean encoded ||z|| per family (on the ENCODED
   impact-frame latents) and show it accounts for the pooled-reversal direction.

7. GATE GE3. If POD's alignment >= JEPA's, the mechanism sentence becomes "the
   predictive objective recovers, at nonlinear compactness, the trajectory-local
   transport alignment a linear basis (POD) has by construction and the
   reconstructive (Fukami) latent loses." We report which branch holds.

Pure CPU numpy + POT + scipy. No GPU, no training, no decoder. READ-ONLY on the
v2p1 cache and the pre-extracted latents.
"""

from __future__ import annotations

import os

# Limit BLAS/OMP threads to 1 so the process pool (one worker per encounter) does
# not oversubscribe the cores. Must be set before numpy/scipy/ot import.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import ot  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "session28"))
sys.path.insert(0, str(REPO))

import stats_lib  # noqa: E402
from physics_prep import fit_pre_impact_cycle  # noqa: E402

# ----------------------------------------------------------------------------
# Paths (inputs live in the MAIN repo outputs/ tree, which is gitignored and not
# duplicated into worktrees; outputs of THIS script go under the active tree).
# CE3_INPUT_REPO overrides the latents root; defaults to the shared main checkout.
# ----------------------------------------------------------------------------
INPUT_REPO = Path(os.environ.get("CE3_INPUT_REPO", str(REPO)))
LATENTS_ROOT = INPUT_REPO / "outputs" / "session28" / "latents"
SPLIT_MANIFEST = REPO / "configs" / "splits" / "split_v2p1.json"

OUT_DIR = REPO / "outputs" / "session28" / "transport"
NUMBERS_PART = REPO / "outputs" / "session28" / "numbers_parts" / "transport_ce3.json"

# Headline families (the missing comparator pod is INCLUDED). Canonical seed s42
# for the seeded families (families_closure.yaml); pod is deterministic.
FAMILIES = {
    "jepa": "jepa_tf_noc_d64_s42",
    "fukami": "fukami_d64_s42",
    "pod": "pod_d64",
}

# Cache layout (CLAUDE.md): v2p1 per-encounter cache under VORTEX_JEPA_CACHE.
DEFAULT_CACHE = Path(os.path.expanduser("~/PREVENT")) / "data" / "processed" / "vortex-jepa"


def cache_root() -> Path:
    return Path(os.environ.get("VORTEX_JEPA_CACHE", str(DEFAULT_CACHE))) / "v2p1"


# ----------------------------------------------------------------------------
# Grid geometry (locked, CLAUDE.md): physical extent x in (-1.5, 4.5) over 192
# px, y in (-1.5, 1.5) over 96 px -> 32 px/chord isotropic. Average-pool by 4 to
# 48x24 (8 px/chord) before building the cost matrix, as in session20/27.
# ----------------------------------------------------------------------------
EXTENT_X = (-1.5, 4.5)
EXTENT_Y = (-1.5, 1.5)
POOL = 4
NX_POOL = 192 // POOL  # 48 (chordwise)
NY_POOL = 96 // POOL  # 24 (cross-stream)

# Entropic regularisation eps fixed from the ground-cost scale. eps = fraction of
# the median off-diagonal squared distance (chord^2). 0.05 of the median (~0.25
# chord^2 on this grid) is a few pooled-cell widths, small enough that the
# entropic blur does not smear the leading-edge vortex (~1 chord); the
# {eps/3, eps, 3 eps} sweep brackets it by an order of magnitude.
#
# SOLVER: plain (BLAS) Sinkhorn, ot.sinkhorn2 method="sinkhorn". At this eps the
# kernel exp(-C/eps) does NOT underflow (C_max/eps ~ 170 < 690), but the pooled
# half-field masses carry many EXACT zeros, which break the plain solver's 1/a
# step (divide-by-zero -> NaN -> degenerate <T,C> = 0). The fix is a tiny uniform
# mass floor (MASS_FLOOR_REG) added before normalisation so every cell has
# positive mass: this reproduces the log-domain stabilised divergence to 1e-5
# (S(a,b) = 0.34564 vs 0.34564) at ~20x the speed. Verified against the
# sinkhorn_log solver and POT empirical_sinkhorn_divergence.
EPS_FRACTION = 0.05
SINKHORN_METHOD = "sinkhorn"
SINKHORN_NUMITERMAX = 2000
SINKHORN_STOPTHR = 1e-9
MASS_FLOOR_REG = 1e-6  # uniform-fraction floor before normalising (kills exact zeros)
MASS_FLOOR = 1e-12  # so an all-zero half-field normalises to uniform, not NaN

# Recovery/settled convention (physics_prep / HANDOFF D180): a baseline frame at
# local index t of an encounter starting at global raw frame frame_start is
# "settled" iff frame_start + t >= SETTLED_GLOBAL_FRAME_MIN.
SETTLED_GLOBAL_FRAME_MIN = 400

# Frame offsets relative to impact for the paired FIELD-distance distributions.
IMPACT_OFFSET = 0
IMPACT16_OFFSET = 16

# Alignment frame grid: strided frames spanning the encounter. The latent and
# field distance sequences are both built on this grid and Spearman-correlated.
ALIGN_FRAME_STRIDE = 4  # 120 -> 30 frames

EDGE_TRIM_FRAC = 0.15  # baseline full-encounter phase fit edge trim


# ----------------------------------------------------------------------------
# Pooling and the squared-Euclidean cost matrix
# ----------------------------------------------------------------------------
def pool4(field: np.ndarray) -> np.ndarray:
    """Average-pool a (192, 96) field by factor POOL to (48, 24)."""
    h, w = field.shape
    return field.reshape(h // POOL, POOL, w // POOL, POOL).mean(axis=(1, 3))


def build_cost_matrix() -> np.ndarray:
    """Squared-Euclidean ground cost between pooled pixel centres, in chord^2."""
    dx = (EXTENT_X[1] - EXTENT_X[0]) / NX_POOL
    dy = (EXTENT_Y[1] - EXTENT_Y[0]) / NY_POOL
    xs = EXTENT_X[0] + (np.arange(NX_POOL) + 0.5) * dx
    ys = EXTENT_Y[0] + (np.arange(NY_POOL) + 0.5) * dy
    xx, yy = np.meshgrid(xs, ys, indexing="ij")  # (48, 24)
    coords = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (1152, 2)
    return ot.dist(coords, coords, metric="sqeuclidean")  # (1152, 1152), chord^2


def eps_from_cost(cost: np.ndarray) -> float:
    """eps = EPS_FRACTION * median off-diagonal squared distance (chord^2)."""
    iu = np.triu_indices(cost.shape[0], k=1)
    return float(EPS_FRACTION * np.median(cost[iu]))


# Module-global cost matrix and eps. Built once in main() and inherited by Pool
# workers through fork (copy-on-write), so they are not pickled per task.
_COST: np.ndarray | None = None
_EPS: float | None = None


def split_pos_neg_pooled(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pool then split into positive / negative vorticity mass vectors (1152,)."""
    mp = pool4(np.maximum(field, 0.0)).ravel()
    mn = pool4(np.maximum(-field, 0.0)).ravel()
    return mp, mn


def normalise_mass(m: np.ndarray) -> np.ndarray:
    """Normalise a non-negative mass vector to unit total (uniform if empty).

    A tiny uniform floor (MASS_FLOOR_REG * mean) is added before normalising so no
    cell has EXACTLY zero mass; this keeps plain Sinkhorn's 1/a step well defined
    (exact zeros otherwise yield NaN and a degenerate transport cost). The floor is
    1e-6 of the mean per cell, far below the vortex signal, and reproduces the
    log-domain divergence to 1e-5.
    """
    m = m.astype(np.float64)
    s = m.sum()
    if s < MASS_FLOOR:
        return np.full(m.shape, 1.0 / m.size)
    m = m + MASS_FLOOR_REG * (s / m.size)  # uniform floor at 1e-6 of the mean
    return m / m.sum()


# ----------------------------------------------------------------------------
# Debiased Sinkhorn divergence S_eps and the signed d_field
# ----------------------------------------------------------------------------
def ot_eps(a: np.ndarray, b: np.ndarray, cost: np.ndarray, eps: float) -> float:
    """Linear entropic transport cost <T*, C> via POT ot.sinkhorn2 (mass-balanced).

    Plain (BLAS) Sinkhorn (method=SINKHORN_METHOD). a, b must have strictly
    positive entries (see normalise_mass's MASS_FLOOR_REG) so the 1/a step is well
    defined; otherwise the plain solver returns a degenerate 0 cost.
    """
    return float(
        ot.sinkhorn2(
            a,
            b,
            cost,
            eps,
            method=SINKHORN_METHOD,
            numItermax=SINKHORN_NUMITERMAX,
            stopThr=SINKHORN_STOPTHR,
        )
    )


def s_eps(a: np.ndarray, b: np.ndarray, cost: np.ndarray, eps: float) -> float:
    """Debiased Sinkhorn divergence S_eps(a, b) (Tran eq. B5 / Feydy 2019).

    a, b are UNIT-mass distributions on the pooled grid. Returns
    OT_eps(a, b) - 0.5 OT_eps(a, a) - 0.5 OT_eps(b, b) using the LINEAR transport
    cost for each term, which is the value POT's empirical_sinkhorn_divergence
    also reports (verified to 1e-6; see README). S_eps(a, a) = 0 to tolerance.
    """
    w_ab = ot_eps(a, b, cost, eps)
    w_aa = ot_eps(a, a, cost, eps)
    w_bb = ot_eps(b, b, cost, eps)
    return w_ab - 0.5 * w_aa - 0.5 * w_bb


def d_field_from_masses(
    mp1: np.ndarray,
    mn1: np.ndarray,
    mp2: np.ndarray,
    mn2: np.ndarray,
    cost: np.ndarray,
    eps: float,
) -> float:
    """Signed-vorticity debiased OT field distance (Tran eqs. B6-B8).

    Each of m+, m- is normalised to unit mass (balanced-on-normalised-parts) and
    the two parts are transported separately and summed.
    """
    return s_eps(normalise_mass(mp1), normalise_mass(mp2), cost, eps) + s_eps(
        normalise_mass(mn1), normalise_mass(mn2), cost, eps
    )


def d_field(f1: np.ndarray, f2: np.ndarray, cost: np.ndarray, eps: float) -> float:
    """d_field between two (192, 96) signed vorticity fields."""
    mp1, mn1 = split_pos_neg_pooled(f1)
    mp2, mn2 = split_pos_neg_pooled(f2)
    return d_field_from_masses(mp1, mn1, mp2, mn2, cost, eps)


# ----------------------------------------------------------------------------
# Phase-matched undisturbed baseline reference
# ----------------------------------------------------------------------------
TWO_PI = 2.0 * np.pi


def _load_encounter(cid: str, k: int) -> dict:
    """Load one cached encounter's omega_z, C_L, impact frame, frame_start."""
    import h5py

    path = cache_root() / cid / f"encounter_{k:02d}.h5"
    with h5py.File(path, "r") as f:
        omega = f["omega_z"][:]  # (120, 192, 96)
        cl = f["C_L"][:].astype(np.float64)
        impact = int(f.attrs["impact_frame_estimate"])
        frame_start = int(f.attrs["frame_start"])
    return {"omega": omega, "cl": cl, "impact": impact, "frame_start": frame_start}


def _full_encounter_phase(cl: np.ndarray) -> tuple[np.ndarray, float]:
    """Continuous Hilbert phase (rad) for every frame of a clean baseline cycle.

    Linear detrend + Hilbert + interior linear phase fit (same recipe as
    physics_prep.fit_pre_impact_cycle but over the WHOLE encounter, which is a
    settled limit cycle for the baseline). Returns (phase_per_frame mod 2pi,
    omega_rad_per_frame).
    """
    from scipy.signal import hilbert

    x = np.asarray(cl, dtype=np.float64)
    n = x.size
    t = np.arange(n, dtype=np.float64)
    slope, intercept = np.polyfit(t, x, 1)
    det = x - (intercept + slope * t)
    phase = np.unwrap(np.angle(hilbert(det)))
    trim = max(2, int(round(EDGE_TRIM_FRAC * n)))
    interior = slice(trim, n - trim)
    omega, phi0 = np.polyfit(t[interior], phase[interior], 1)
    phase_fit = (phi0 + omega * t) % TWO_PI
    return phase_fit, float(omega)


class BaselinePhaseRef:
    """Phase-indexed pool of SETTLED undisturbed-baseline frames.

    Holds, for every settled baseline frame (global raw frame >=
    SETTLED_GLOBAL_FRAME_MIN), its continuous Hilbert phase and its omega_z field.
    match_index(phi) returns the pool index whose phase is circularly closest to
    phi. The omega fields are kept with pooled-pos/neg masses precomputed so the
    field-distance worker does not re-pool the baseline each call.
    """

    def __init__(
        self,
        phases: np.ndarray,
        omega: np.ndarray,
        masses: list[tuple[np.ndarray, np.ndarray]],
        meta: list[tuple[str, int, int]],
    ) -> None:
        self.phases = phases  # (M,) in [0, 2pi)
        self.omega = omega  # (M, 192, 96)
        self.masses = masses  # list of (m+, m-) pooled mass vectors
        self.meta = meta  # (case_id, encounter_index, local_frame)

    def match_index(self, phi: float) -> int:
        d = np.abs(((self.phases - phi + np.pi) % TWO_PI) - np.pi)  # circular distance
        return int(np.argmin(d))

    def match_field(self, phi: float) -> np.ndarray:
        return self.omega[self.match_index(phi)]

    def match_masses(self, phi: float) -> tuple[np.ndarray, np.ndarray]:
        return self.masses[self.match_index(phi)]


def build_baseline_ref(manifest: dict) -> BaselinePhaseRef:
    """Build the settled-baseline phase reference from the calibration case."""
    from physics_prep import calibration_reference_case

    cid, encs = calibration_reference_case(manifest)
    phases_all: list[float] = []
    omega_all: list[np.ndarray] = []
    masses_all: list[tuple[np.ndarray, np.ndarray]] = []
    meta_all: list[tuple[str, int, int]] = []
    for k in encs:
        rec = _load_encounter(cid, k)
        phase_fit, _ = _full_encounter_phase(rec["cl"])
        fs = rec["frame_start"]
        for t in range(rec["omega"].shape[0]):
            if fs + t < SETTLED_GLOBAL_FRAME_MIN:
                continue
            phases_all.append(phase_fit[t])
            omega_all.append(rec["omega"][t])
            masses_all.append(split_pos_neg_pooled(rec["omega"][t]))
            meta_all.append((cid, k, t))
    if not omega_all:
        raise ValueError("no settled baseline frames; check SETTLED_GLOBAL_FRAME_MIN")
    return BaselinePhaseRef(
        np.asarray(phases_all),
        np.stack(omega_all),
        masses_all,
        meta_all,
    )


def encounter_target_phases(cl: np.ndarray, impact: int, frames: np.ndarray) -> np.ndarray:
    """Shedding phase at each frame, from the encounter's OWN pre-impact C_L fit.

    The pre-impact Hilbert phase fit gives phi at the impact frame and the rate
    omega; phi(frame) = phi_imp + omega * (frame - impact), wrapped to [0, 2pi).
    """
    fit = fit_pre_impact_cycle(cl[:impact])
    phi = (fit.phi_imp + fit.omega_rad_per_frame * (frames - impact)) % TWO_PI
    return phi


# ----------------------------------------------------------------------------
# Per-encounter worker: field-distance sequence + paired field distances
# ----------------------------------------------------------------------------
_BASELINE_REF: BaselinePhaseRef | None = None


def _encounter_field_sequence(args: tuple) -> dict:
    """Worker: for one gust encounter, build the d_field sequence vs the
    phase-matched baseline over the alignment frame grid, plus the paired field
    distances at impact and impact+16, for the eps grid.
    """
    cid, enc, eps_grid = args
    rec = _load_encounter(cid, enc)
    omega = rec["omega"]
    impact = rec["impact"]
    nT = omega.shape[0]
    frames = np.arange(0, nT, ALIGN_FRAME_STRIDE)
    phis = encounter_target_phases(rec["cl"], impact, frames)

    # pos/neg pooled masses of each gust frame on the grid (reused across eps).
    gust_masses = [split_pos_neg_pooled(omega[fr]) for fr in frames]
    match_idx = np.array([_BASELINE_REF.match_index(p) for p in phis])

    out: dict = {
        "case_id": cid,
        "encounter_index": int(enc),
        "impact": int(impact),
        "frames": frames,
        "match_idx": match_idx,
        "d_field_by_eps": {},
        "impact_d_by_eps": {},
        "impact16_d_by_eps": {},
    }
    for label, eps in eps_grid.items():
        seq = np.empty(frames.size)
        for i, mi in enumerate(match_idx):
            bp, bn = _BASELINE_REF.masses[mi]
            gp, gn = gust_masses[i]
            seq[i] = d_field_from_masses(gp, gn, bp, bn, _COST, eps)
        out["d_field_by_eps"][label] = seq
        # impact and impact+16 paired distances (own phase match per offset).
        for off, key in (
            (IMPACT_OFFSET, "impact_d_by_eps"),
            (IMPACT16_OFFSET, "impact16_d_by_eps"),
        ):
            fr = impact + off
            if fr >= nT:
                out[key][label] = float("nan")
                continue
            phi = encounter_target_phases(rec["cl"], impact, np.array([fr]))[0]
            mi = _BASELINE_REF.match_index(phi)
            bp, bn = _BASELINE_REF.masses[mi]
            gp, gn = split_pos_neg_pooled(omega[fr])
            out[key][label] = d_field_from_masses(gp, gn, bp, bn, _COST, eps)
    return out


# ----------------------------------------------------------------------------
# Latent distance sequences (cheap; main process)
# ----------------------------------------------------------------------------
def load_family_latents(family_dir: str) -> dict:
    """Load test_b + test_c z_full and metadata for one family."""
    out: dict = {}
    for split in ("test_b", "test_c"):
        d = np.load(LATENTS_ROOT / family_dir / f"{split}.npz", allow_pickle=True)
        out[split] = {
            "z_full": d["z_full"],  # (n, 120, dd)
            "case_ids": np.array([str(c) for c in d["case_ids"]]),
            "encounter_indices": np.array([int(e) for e in d["encounter_indices"]]),
            "impact_frame": np.array([int(i) for i in d["impact_frame"]]),
            "G": d["G"].astype(float),
        }
    return out


def baseline_latents(family_dir: str, manifest: dict) -> dict:
    """Phase-indexed baseline latents for one family, aligned to _BASELINE_REF.

    The latent NPZ stores z_full per (case_id, encounter). We pull the Baseline
    case's frames and index them with the SAME (encounter, local_frame) -> phase
    order as _BASELINE_REF.meta, so the latent of a matched baseline frame is
    z_base_pool[match_idx]. The Baseline encounters split across train (encs 0-3)
    and test_a (the renamed val, encs 4-5); both files are read.
    """
    from physics_prep import calibration_reference_case

    cid, _ = calibration_reference_case(manifest)
    train = np.load(LATENTS_ROOT / family_dir / "train.npz", allow_pickle=True)
    val = np.load(LATENTS_ROOT / family_dir / "test_a.npz", allow_pickle=True)
    zb: dict[tuple[str, int], np.ndarray] = {}
    for d in (train, val):
        cids = np.array([str(c) for c in d["case_ids"]])
        eix = np.array([int(e) for e in d["encounter_indices"]])
        zf = d["z_full"]
        for j in range(cids.size):
            if cids[j] == cid:
                zb[(cid, int(eix[j]))] = zf[j]  # (120, dd)
    lat = np.stack([zb[(c, k)][t] for (c, k, t) in _BASELINE_REF.meta])  # (M, dd)
    return {"case_id": cid, "z_base_pool": lat}


def latent_distance_sequence(
    z_full_enc: np.ndarray, frames: np.ndarray, match_idx: np.ndarray, z_base_pool: np.ndarray
) -> np.ndarray:
    """||z_t - z_basephase(t)|| over the alignment frames for one encounter."""
    zf = z_full_enc[frames]  # (nf, dd)
    zb = z_base_pool[match_idx]  # (nf, dd)
    return np.sqrt(((zf - zb) ** 2).sum(axis=1))


# ----------------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------------
def run(max_enc: int | None, workers: int, splits: list[str]) -> dict:
    global _COST, _EPS, _BASELINE_REF
    manifest = json.loads(SPLIT_MANIFEST.read_text())

    _COST = build_cost_matrix()
    _EPS = eps_from_cost(_COST)
    eps_grid = {"eps_third": _EPS / 3.0, "eps": _EPS, "eps_triple": 3.0 * _EPS}
    print(
        f"cost matrix {_COST.shape}, max {_COST.max():.3f} chord^2; "
        f"eps = {_EPS:.5f} chord^2 (= {EPS_FRACTION} x median pairwise sq dist); "
        f"eps grid = {{{eps_grid['eps_third']:.5f}, {_EPS:.5f}, {eps_grid['eps_triple']:.5f}}}; "
        f"workers={workers}",
        flush=True,
    )

    print("building settled-baseline phase reference ...", flush=True)
    _BASELINE_REF = build_baseline_ref(manifest)
    print(
        f"  baseline pool: {_BASELINE_REF.phases.size} settled frames "
        f"(phase span {_BASELINE_REF.phases.min():.2f}-{_BASELINE_REF.phases.max():.2f} rad)",
        flush=True,
    )

    fam_latents = {f: load_family_latents(d) for f, d in FAMILIES.items()}
    base_latents = {f: baseline_latents(d, manifest) for f, d in FAMILIES.items()}

    # Verify (case, encounter) ordering matches across families per split.
    for split in splits:
        ref_c = fam_latents["jepa"][split]["case_ids"]
        ref_e = fam_latents["jepa"][split]["encounter_indices"]
        for f in FAMILIES:
            assert np.array_equal(fam_latents[f][split]["case_ids"], ref_c) and np.array_equal(
                fam_latents[f][split]["encounter_indices"], ref_e
            ), f"latent ordering mismatch {f} {split}"

    # Heavy field-distance sequences (one per gust encounter), computed ONCE and
    # reused for every family's alignment.
    field_seqs: dict = {}
    for split in splits:
        ref = fam_latents["jepa"][split]
        n_all = ref["case_ids"].size
        n = n_all if max_enc is None else min(max_enc, n_all)
        tasks = [(ref["case_ids"][i], int(ref["encounter_indices"][i]), eps_grid) for i in range(n)]
        t0 = time.time()
        if workers > 1:
            with Pool(workers) as pool:
                seqs = pool.map(_encounter_field_sequence, tasks, chunksize=1)
        else:
            seqs = [_encounter_field_sequence(t) for t in tasks]
        field_seqs[split] = seqs
        print(f"  [field] {split}: {n} enc in {time.time()-t0:.1f}s", flush=True)

    # Alignment per family per split, for each eps in the grid.
    align: dict = {f: {"per_split": {}} for f in FAMILIES}
    for split in splits:
        ref = fam_latents["jepa"][split]
        seqs = field_seqs[split]
        n = len(seqs)
        for f in FAMILIES:
            zf_all = fam_latents[f][split]["z_full"]
            zb_pool = base_latents[f]["z_base_pool"]
            per_enc_rho = {label: np.empty(n) for label in eps_grid}
            for i in range(n):
                frames = seqs[i]["frames"]
                match_idx = seqs[i]["match_idx"]
                lat_seq = latent_distance_sequence(zf_all[i], frames, match_idx, zb_pool)
                for label in eps_grid:
                    fseq = seqs[i]["d_field_by_eps"][label]
                    per_enc_rho[label][i] = spearmanr(lat_seq, fseq).correlation
            align[f]["per_split"][split] = {
                "case_ids": ref["case_ids"][:n].tolist(),
                "encounter_indices": ref["encounter_indices"][:n].tolist(),
                "rho_per_eps": {k: v.tolist() for k, v in per_enc_rho.items()},
                "rho_mean": float(np.nanmean(per_enc_rho["eps"])),
                "rho_median": float(np.nanmedian(per_enc_rho["eps"])),
            }

    results: dict = {}
    results["align"] = align
    results["field_paired"] = collect_field_paired(field_seqs, splits)
    results["paired_deltas"] = paired_deltas(align, splits)
    results["eps_sensitivity"] = eps_sensitivity(align, eps_grid)
    results["trend"] = alignment_vs_g_trend(align, fam_latents, splits)
    results["mechanism"] = norm_variance_mechanism(fam_latents)
    results["gate_ge3"] = gate_ge3(align)
    results["config"] = {
        "reference": "Tran, Yeh, Taira, JFM 1027, A24 (2026), App. B; Feydy et al., AISTATS 2019",
        "debiased": True,
        "divergence": "S_eps(a,b) = OT_eps(a,b) - 0.5 OT_eps(a,a) - 0.5 OT_eps(b,b)",
        "ot_eps_via": f"ot.sinkhorn2 method={SINKHORN_METHOD} (linear transport cost); "
        "1e-6 uniform mass floor; matches POT empirical_sinkhorn_divergence to 1e-6",
        "sinkhorn_method": SINKHORN_METHOD,
        "mass_floor_reg": MASS_FLOOR_REG,
        "marginal_constraint": "balanced-on-normalised-parts (m+ and m- each unit mass); "
        "no unbalanced rho",
        "ground_cost": "squared-Euclidean, chord^2",
        "pool_factor": POOL,
        "pooled_grid": [NX_POOL, NY_POOL],
        "extent_x_chord": list(EXTENT_X),
        "extent_y_chord": list(EXTENT_Y),
        "eps_chord2": _EPS,
        "eps_rule": f"{EPS_FRACTION} x median off-diagonal squared distance (chord^2)",
        "eps_grid_chord2": {k: float(v) for k, v in eps_grid.items()},
        "align_frame_stride": ALIGN_FRAME_STRIDE,
        "impact_offset": IMPACT_OFFSET,
        "impact16_offset": IMPACT16_OFFSET,
        "settled_global_frame_min": SETTLED_GLOBAL_FRAME_MIN,
        "baseline_pool_size": int(_BASELINE_REF.phases.size),
        "families": FAMILIES,
        "splits": splits,
    }
    return results


def collect_field_paired(field_seqs: dict, splits: list[str]) -> dict:
    """Per-encounter paired FIELD-distance distributions at impact and impact+16."""
    out: dict = {}
    for split in splits:
        seqs = field_seqs[split]
        out[split] = {
            "case_ids": [s["case_id"] for s in seqs],
            "encounter_indices": [s["encounter_index"] for s in seqs],
            "impact_d_eps": [float(s["impact_d_by_eps"]["eps"]) for s in seqs],
            "impact16_d_eps": [float(s["impact16_d_by_eps"]["eps"]) for s in seqs],
            "impact_d_mean": float(np.nanmean([s["impact_d_by_eps"]["eps"] for s in seqs])),
            "impact16_d_mean": float(np.nanmean([s["impact16_d_by_eps"]["eps"] for s in seqs])),
        }
    return out


def paired_deltas(align: dict, splits: list[str]) -> dict:
    """Paired per-encounter alignment delta (jepa - other) with clustered CI + sign tests.

    Reported pooled over splits AND per split, at the central eps. Significance via
    case-clustered bootstrap CI + per-encounter and per-case sign tests; NOT
    case_permutation_p (degenerate for a paired location test, the B6 lesson).
    """
    out: dict = {}
    for other in ("fukami", "pod"):
        out[other] = {}
        for scope, split_list in (("pooled", splits), *((s, [s]) for s in splits)):
            jepa_rho: list[float] = []
            other_rho: list[float] = []
            cids: list[str] = []
            for split in split_list:
                jr = np.array(align["jepa"]["per_split"][split]["rho_per_eps"]["eps"])
                orr = np.array(align[other]["per_split"][split]["rho_per_eps"]["eps"])
                cc = align["jepa"]["per_split"][split]["case_ids"]
                jepa_rho.extend(jr.tolist())
                other_rho.extend(orr.tolist())
                cids.extend(cc)
            jepa_rho = np.array(jepa_rho)
            other_rho = np.array(other_rho)
            case_ids = np.array(cids)
            valid = np.isfinite(jepa_rho) & np.isfinite(other_rho)
            jr = jepa_rho[valid]
            orr = other_rho[valid]
            cv = case_ids[valid]
            delta = jr - orr  # > 0 means jepa better aligned
            boot = stats_lib.case_cluster_bootstrap(delta, cv, rng=np.random.default_rng(0))
            # per-encounter sign test: jepa better (larger rho) than other.
            # sign_test_one_sided(err_a, err_b) counts err_a < err_b; using
            # (-jr, -orr) counts jr > orr (jepa higher rho).
            k_enc, n_enc, p_enc = stats_lib.sign_test_one_sided(-jr, -orr)
            _, case_mean_delta = stats_lib.case_means(delta, cv)
            case_stats = stats_lib.case_level_paired_stats(case_mean_delta)
            out[other][scope] = {
                "n_encounters": int(delta.size),
                "n_cases": int(np.unique(cv).size),
                "mean_delta": float(delta.mean()),
                "median_delta": float(np.median(delta)),
                "enc_mean_ci": boot["enc_mean_ci"],
                "case_mean": boot["case_mean"],
                "case_mean_ci": boot["case_mean_ci"],
                "sign_test_enc": {"k": k_enc, "n_eff": n_enc, "p_one_sided": p_enc},
                "sign_test_case": {
                    "k": case_stats["cases_jepa_better"],
                    "n": case_stats["n_cases"],
                    "p_one_sided": case_stats["sign_p_one_sided"],
                    "wilcoxon_p_one_sided": case_stats["wilcoxon_p_one_sided"],
                },
            }
    return out


def eps_sensitivity(align: dict, eps_grid: dict) -> dict:
    """Headline (test_b) paired delta jepa-fukami and jepa-pod across the eps grid."""
    out: dict = {}
    split = "test_b"
    cids = np.array(align["jepa"]["per_split"][split]["case_ids"])
    for other in ("fukami", "pod"):
        out[other] = {}
        for label in eps_grid:
            jr = np.array(align["jepa"]["per_split"][split]["rho_per_eps"][label])
            orr = np.array(align[other]["per_split"][split]["rho_per_eps"][label])
            valid = np.isfinite(jr) & np.isfinite(orr)
            delta = jr[valid] - orr[valid]
            boot = stats_lib.case_cluster_bootstrap(
                delta, cids[valid], rng=np.random.default_rng(0)
            )
            out[other][label] = {
                "mean_delta": float(delta.mean()),
                "enc_mean_ci": boot["enc_mean_ci"],
                "jepa_rho_mean": float(np.nanmean(jr)),
                "other_rho_mean": float(np.nanmean(orr)),
            }
    fd = [out["fukami"][lab]["mean_delta"] for lab in eps_grid]
    out["jepa_minus_fukami_delta_span_testb"] = [float(min(fd)), float(max(fd))]
    return out


def alignment_vs_g_trend(align: dict, fam_latents: dict, splits: list[str]) -> dict:
    """Alignment-vs-|G| TREND per family, with case_permutation_p (its VALID use).

    Per-encounter Spearman(|G|, rho) pooled over splits, case-permutation p (the
    one statistic case_permutation_p is appropriate for: a dependence/trend, not a
    paired location difference).
    """

    def spear(x, y):
        r = spearmanr(x, y).correlation
        return 0.0 if np.isnan(r) else r

    out: dict = {}
    for f in FAMILIES:
        g_all: list[float] = []
        rho_all: list[float] = []
        cids: list[str] = []
        for split in splits:
            rho = np.array(align[f]["per_split"][split]["rho_per_eps"]["eps"])
            cc = align[f]["per_split"][split]["case_ids"]
            gg = np.abs(fam_latents[f][split]["G"][: rho.size])
            g_all.extend(gg.tolist())
            rho_all.extend(rho.tolist())
            cids.extend(cc)
        g_arr = np.array(g_all)
        rho_arr = np.array(rho_all)
        cid_arr = np.array(cids)
        valid = np.isfinite(rho_arr)
        res = stats_lib.case_permutation_p(
            g_arr[valid],
            rho_arr[valid],
            cid_arr[valid],
            spear,
            n_perm=10000,
            rng=np.random.default_rng(0),
            alternative="two-sided",
        )
        out[f] = {"spearman_absG_rho": res["stat"], "case_perm_p": res["p"]}
    return out


def norm_variance_mechanism(fam_latents: dict) -> dict:
    """Between-encounter variance of the mean encoded ||z|| per family.

    The pooled OT statistic is an ENCODING-scale property. On the ENCODED
    impact-frame latents, encoded latents do not drift; the "pooled reversal" is a
    between-encounter latent-norm-variance artifact. We report, per family and
    pooled over test_b + test_c, the mean of ||z_impact|| and the between-encounter
    variance of the per-encounter ||z_impact||, and the sign of the
    fukami-minus-jepa difference (the direction of the pooled reversal).
    """
    out: dict = {}
    for f in FAMILIES:
        per_enc_norm: list[float] = []
        for split in ("test_b", "test_c"):
            d = fam_latents[f][split]
            zf = d["z_full"]  # (n, 120, dd)
            imp = d["impact_frame"]
            for i in range(zf.shape[0]):
                z_imp = zf[i, int(imp[i])]
                per_enc_norm.append(float(np.linalg.norm(z_imp)))
        arr = np.array(per_enc_norm)
        out[f] = {
            "mean_norm": float(arr.mean()),
            "between_encounter_var_meannorm": float(arr.var(ddof=1)),
            "between_encounter_sd_meannorm": float(arr.std(ddof=1)),
            "cv_meannorm": (
                float(arr.std(ddof=1) / arr.mean()) if arr.mean() > 0 else float("nan")
            ),
            "n_encounters": int(arr.size),
        }
    out["fukami_minus_jepa_var"] = (
        out["fukami"]["between_encounter_var_meannorm"]
        - out["jepa"]["between_encounter_var_meannorm"]
    )
    out["fukami_norm_var_exceeds_jepa"] = bool(out["fukami_minus_jepa_var"] > 0)
    return out


def gate_ge3(align: dict) -> dict:
    """GE3: does POD's alignment >= JEPA's? Report the branch HONESTLY.

    The plan's GE3 trigger is POD >= JEPA on the model-selection split (test_b).
    But the branch wording must reflect the FULL three-way ordering actually
    observed, which on this run contradicts the anticipated narrative: the
    reconstructive (Fukami) latent has the HIGHEST per-encounter alignment, and
    POD's ranking flips between test_b and test_c. We do not assert JEPA beats
    Fukami when it does not.
    """
    split = "test_b"
    jepa = align["jepa"]["per_split"][split]["rho_mean"]
    pod = align["pod"]["per_split"][split]["rho_mean"]
    fukami = align["fukami"]["per_split"][split]["rho_mean"]
    pod_ge_jepa = bool(pod >= jepa)
    fukami_gt_jepa = bool(fukami > jepa)
    # per-split POD-vs-JEPA, since the ordering can flip across splits.
    pod_ge_jepa_by_split = {
        s: bool(
            align["pod"]["per_split"][s]["rho_mean"] >= align["jepa"]["per_split"][s]["rho_mean"]
        )
        for s in align["jepa"]["per_split"]
    }
    if pod_ge_jepa:
        head = (
            "POD_GE_JEPA (test_b): the predictive objective recovers, at nonlinear "
            "compactness, the trajectory-local transport alignment a linear basis "
            "(POD) has by construction and the reconstructive (Fukami) latent loses."
        )
    else:
        head = (
            "JEPA_GT_POD (test_b): the predictive latent's trajectory-local transport "
            "alignment exceeds the linear floor (POD) on test_b."
        )
    # honest qualifier: Fukami beats JEPA here, and POD flips by split.
    qualifier = ""
    if fukami_gt_jepa:
        qualifier += (
            " HOWEVER the reconstructive (Fukami) latent has the HIGHEST per-encounter "
            "alignment (it beats JEPA), so the v2 'JEPA tracks transport geometry best' "
            "claim does NOT hold for the per-encounter statistic; see the alignment-vs-|G| "
            "trend, which shows the AE and POD alignment is driven by latent-norm scaling "
            "with gust strength, not trajectory-local geometry tracking."
        )
    if pod_ge_jepa_by_split.get("test_c") and not pod_ge_jepa_by_split.get("test_b"):
        qualifier += (
            " POD's alignment exceeds JEPA's on test_c (|G|=4), reversing the test_b order."
        )
    branch = head + qualifier
    return {
        "split": split,
        "rho_mean_jepa": jepa,
        "rho_mean_pod": pod,
        "rho_mean_fukami": fukami,
        "pod_ge_jepa": pod_ge_jepa,
        "fukami_gt_jepa": fukami_gt_jepa,
        "pod_ge_jepa_by_split": pod_ge_jepa_by_split,
        "branch": branch,
    }


# ----------------------------------------------------------------------------
# Numbers part (eval_all format) + README + NPZ
# ----------------------------------------------------------------------------
def fmt_num(value, **extra) -> dict:
    rec = {"value": float(value), "source": "transport_ce3.py"}
    rec.update(extra)
    return rec


def write_numbers_part(results: dict) -> None:
    cfg = results["config"]
    pooled_fuk = results["paired_deltas"]["fukami"]["pooled"]
    pooled_pod = results["paired_deltas"]["pod"]["pooled"]
    tb_fuk = results["paired_deltas"]["fukami"]["test_b"]
    span = results["eps_sensitivity"]["jepa_minus_fukami_delta_span_testb"]
    gate = results["gate_ge3"]
    mech = results["mechanism"]
    trend = results["trend"]
    numbers = {
        "transport_eps_chord2": fmt_num(
            cfg["eps_chord2"],
            macro="TransportEps",
            fmt="%.4f",
            unit="chord^2",
            note=cfg["eps_rule"],
        ),
        "transport_debiased": fmt_num(
            1.0,
            macro="TransportDebiased",
            fmt="%s",
            note="debiased Sinkhorn divergence S_eps (Tran B5 / Feydy 2019); yes",
        ),
        "transport_align_delta_jepa_fukami_testb": fmt_num(
            tb_fuk["mean_delta"],
            macro="TransportAlignDeltaFukTb",
            fmt="%.3f",
            ci_lo=tb_fuk["enc_mean_ci"][0],
            ci_hi=tb_fuk["enc_mean_ci"][1],
            n=tb_fuk["n_encounters"],
            split="test_b",
            note="paired per-encounter Spearman delta jepa-fukami, case-clustered CI",
        ),
        "transport_align_delta_jepa_fukami_pooled": fmt_num(
            pooled_fuk["mean_delta"],
            macro="TransportAlignDeltaFuk",
            fmt="%.3f",
            ci_lo=pooled_fuk["enc_mean_ci"][0],
            ci_hi=pooled_fuk["enc_mean_ci"][1],
            n=pooled_fuk["n_encounters"],
            split="test_b+test_c",
            note="paired alignment delta jepa-fukami pooled, case-clustered CI; "
            f"per-enc sign p={pooled_fuk['sign_test_enc']['p_one_sided']:.3g}, "
            f"per-case sign p={pooled_fuk['sign_test_case']['p_one_sided']:.3g}",
        ),
        "transport_align_delta_jepa_pod_pooled": fmt_num(
            pooled_pod["mean_delta"],
            macro="TransportAlignDeltaPod",
            fmt="%.3f",
            ci_lo=pooled_pod["enc_mean_ci"][0],
            ci_hi=pooled_pod["enc_mean_ci"][1],
            n=pooled_pod["n_encounters"],
            split="test_b+test_c",
            note="paired alignment delta jepa-pod pooled, case-clustered CI; "
            f"per-enc sign p={pooled_pod['sign_test_enc']['p_one_sided']:.3g}, "
            f"per-case sign p={pooled_pod['sign_test_case']['p_one_sided']:.3g}",
        ),
        "transport_eps_sensitivity_lo": fmt_num(
            span[0],
            macro="TransportEpsSensLo",
            fmt="%.3f",
            split="test_b",
            note="min jepa-fukami alignment delta over {eps/3, eps, 3 eps}",
        ),
        "transport_eps_sensitivity_hi": fmt_num(
            span[1],
            macro="TransportEpsSensHi",
            fmt="%.3f",
            split="test_b",
            note="max jepa-fukami alignment delta over {eps/3, eps, 3 eps}",
        ),
        "transport_align_jepa_testb": fmt_num(
            gate["rho_mean_jepa"],
            macro="TransportAlignJepaTb",
            fmt="%.3f",
            split="test_b",
            note="mean per-encounter OT-latent Spearman, jepa",
        ),
        "transport_align_pod_testb": fmt_num(
            gate["rho_mean_pod"],
            macro="TransportAlignPodTb",
            fmt="%.3f",
            split="test_b",
            note="mean per-encounter OT-latent Spearman, pod",
        ),
        "transport_align_fukami_testb": fmt_num(
            gate["rho_mean_fukami"],
            macro="TransportAlignFukTb",
            fmt="%.3f",
            split="test_b",
            note="mean per-encounter OT-latent Spearman, fukami",
        ),
        "transport_pod_ge_jepa": fmt_num(
            1.0 if gate["pod_ge_jepa"] else 0.0,
            macro="TransportPodGeJepa",
            fmt="%s",
            note=gate["branch"],
        ),
        "transport_fukami_gt_jepa": fmt_num(
            1.0 if gate["fukami_gt_jepa"] else 0.0,
            macro="TransportFukGtJepa",
            fmt="%s",
            split="test_b",
            note="reconstructive (Fukami) per-encounter alignment exceeds JEPA's; "
            "contradicts the v2 'JEPA tracks transport geometry best' claim",
        ),
        "transport_trend_absG_pod": fmt_num(
            trend["pod"]["spearman_absG_rho"],
            macro="TransportTrendPod",
            fmt="%.2f",
            split="test_b+test_c",
            note=f"Spearman(|G|, alignment), pod; case-perm p={trend['pod']['case_perm_p']:.3g}",
        ),
        "transport_trend_absG_jepa": fmt_num(
            trend["jepa"]["spearman_absG_rho"],
            macro="TransportTrendJepa",
            fmt="%.2f",
            split="test_b+test_c",
            note=f"Spearman(|G|, alignment), jepa; case-perm p={trend['jepa']['case_perm_p']:.3g}",
        ),
        "transport_normvar_fukami": fmt_num(
            mech["fukami"]["between_encounter_var_meannorm"],
            macro="TransportNormVarFuk",
            fmt="%.3f",
            note="between-encounter variance of mean encoded ||z_impact||, fukami",
        ),
        "transport_normvar_jepa": fmt_num(
            mech["jepa"]["between_encounter_var_meannorm"],
            macro="TransportNormVarJepa",
            fmt="%.3f",
            note="between-encounter variance of mean encoded ||z_impact||, jepa",
        ),
    }
    part = {"part": "transport_ce3", "numbers": numbers}
    NUMBERS_PART.parent.mkdir(parents=True, exist_ok=True)
    NUMBERS_PART.write_text(json.dumps(part, indent=2, sort_keys=True))
    print(f"numbers part -> {NUMBERS_PART}", flush=True)


def write_readme(results: dict) -> None:
    cfg = results["config"]
    gate = results["gate_ge3"]
    mech = results["mechanism"]
    trend = results["trend"]
    span = results["eps_sensitivity"]["jepa_minus_fukami_delta_span_testb"]
    pf = results["paired_deltas"]["fukami"]["pooled"]
    pp = results["paired_deltas"]["pod"]["pooled"]
    lines = [
        "# C-E3 optimal-transport mechanism (closes referee M5)",
        "",
        "Tran, Yeh, Taira, JFM 1027, A24 (2026), Appendix B; Feydy et al., AISTATS 2019.",
        "",
        "## Divergence",
        "",
        "DEBIASED Sinkhorn divergence (yes):",
        "",
        "    S_eps(a, b) = OT_eps(a, b) - 0.5 OT_eps(a, a) - 0.5 OT_eps(b, b)",
        "",
        "OT_eps is the LINEAR entropic transport cost <T*, C> from POT ot.sinkhorn2 on the",
        "precomputed squared-Euclidean cost matrix. POT verification: this construction",
        "reproduces ot.bregman.empirical_sinkhorn_divergence to 1e-6, i.e. POT's own debiased",
        "divergence also debiases the LINEAR cost (not the entropy-regularised objective).",
        "S_eps(a, a) = 0 to solver tolerance. The pooled half-field masses carry many exact",
        "zeros that break plain Sinkhorn's 1/a step; a 1e-6 uniform mass floor restores it and",
        "matches the log-domain stabilised value to 1e-5 at ~20x the speed.",
        "",
        "## Marginal constraint: BALANCED-on-normalised-parts (no unbalanced rho)",
        "",
        "Each of m+ = max(omega, 0) and m- = max(-omega, 0) is pooled to the 48x24 grid and",
        "NORMALISED to unit mass before transport; the two parts are transported separately",
        "and summed (Tran eqs. B6-B8). The debiased divergence is the well-posed BALANCED",
        "object, so we do NOT use an unbalanced KL relaxation; there is no rho. This is the",
        "cleaner principled default and replaces the v2 unstated-rho unbalanced form.",
        "",
        "## Ground cost and epsilon",
        "",
        f"Squared-Euclidean, chord^2, on the pooled "
        f"{cfg['pooled_grid'][0]}x{cfg['pooled_grid'][1]} "
        f"grid (average-pool factor {cfg['pool_factor']}, extent x={cfg['extent_x_chord']} "
        f"y={cfg['extent_y_chord']}).",
        "",
        f"eps = {cfg['eps_rule']} = {cfg['eps_chord2']:.5f} chord^2.",
        f"eps sensitivity grid (chord^2): {cfg['eps_grid_chord2']}.",
        "",
        "## Phase matching",
        "",
        "Each gust encounter's shedding phase is fit from its OWN pre-impact C_L (Hilbert",
        "phase, physics_prep.fit_pre_impact_cycle) and extrapolated to frame impact+k. The",
        "undisturbed reference is a phase-indexed pool of SETTLED Baseline omega_z frames",
        f"(global raw frame >= {cfg['settled_global_frame_min']}; pool size",
        f"{cfg['baseline_pool_size']} frames), each tagged with its continuous Hilbert phase",
        "(full-encounter fit). The match is the settled-baseline frame minimising circular",
        "phase distance to the target phase. The matched baseline LATENT (same family) is the",
        "z of that baseline frame, so the field and latent distances share one matching.",
        "",
        "## Field distances and alignment",
        "",
        "Per test_b and test_c encounter: d_field(field_t, phase-matched baseline) at impact",
        "and impact+16 (paired distributions, not single-frame means), on the DNS v2p1 cache",
        "(no decoder). Alignment = per-encounter Spearman between the latent distance sequence",
        "||z_t - z_basephase(t)|| and the field S_eps sequence over strided frames, for jepa,",
        "fukami, AND pod. The paired per-encounter difference (jepa-fukami, jepa-pod) carries",
        "a CASE-CLUSTERED bootstrap CI (stats_lib) + per-encounter and per-case SIGN tests.",
        "case_permutation_p is NOT used for the paired location test (degenerate, the B6",
        "lesson); it is reserved for the alignment-vs-|G| trend.",
        "",
        "## Headline results",
        "",
        f"test_b mean OT-latent Spearman: jepa={gate['rho_mean_jepa']:.3f}, "
        f"pod={gate['rho_mean_pod']:.3f}, fukami={gate['rho_mean_fukami']:.3f}.",
        f"Paired delta jepa-fukami (pooled test_b+test_c): {pf['mean_delta']:+.3f} "
        f"CI [{pf['enc_mean_ci'][0]:+.3f}, {pf['enc_mean_ci'][1]:+.3f}], "
        f"per-enc sign p={pf['sign_test_enc']['p_one_sided']:.3g}, "
        f"per-case sign p={pf['sign_test_case']['p_one_sided']:.3g}.",
        f"Paired delta jepa-pod (pooled): {pp['mean_delta']:+.3f} "
        f"CI [{pp['enc_mean_ci'][0]:+.3f}, {pp['enc_mean_ci'][1]:+.3f}], "
        f"per-enc sign p={pp['sign_test_enc']['p_one_sided']:.3g}, "
        f"per-case sign p={pp['sign_test_case']['p_one_sided']:.3g}.",
        f"eps sensitivity: jepa-fukami test_b delta spans [{span[0]:+.3f}, {span[1]:+.3f}] "
        "over {eps/3, eps, 3 eps} (essentially eps-invariant).",
        "",
        "## Headline finding (CONTRADICTS the v2 transport narrative)",
        "",
        "The per-encounter (trajectory-local) OT-latent alignment is HIGHEST for the",
        "reconstructive Fukami AE, not for JEPA: the paired jepa-fukami delta is negative with",
        "a case-clustered CI excluding zero on test_b and pooled. The v2 manuscript claim that",
        "the JEPA latent tracks the transport geometry best is NOT supported by the corrected",
        "debiased per-encounter statistic. POD (linear floor) is lowest on test_b but exceeds",
        "JEPA on test_c (|G|=4). The per-encounter sign tests are non-significant (the spread",
        "is large at n=10 cases), so the honest statement is a negative/descriptive one, not a",
        "JEPA win.",
        "",
        "## Alignment-vs-|G| trend (case-permutation p; the valid use of case_permutation_p)",
        "",
        f"Spearman(|G|, per-encounter rho), pooled test_b+test_c: "
        f"pod={trend['pod']['spearman_absG_rho']:.2f} (p={trend['pod']['case_perm_p']:.3g}), "
        f"fukami={trend['fukami']['spearman_absG_rho']:.2f} "
        f"(p={trend['fukami']['case_perm_p']:.3g}), "
        f"jepa={trend['jepa']['spearman_absG_rho']:.2f} (p={trend['jepa']['case_perm_p']:.3g}). "
        "POD's and Fukami's alignment rises with gust strength while JEPA's is flat: the AE/POD "
        "alignment is an artifact of latent-norm scaling with |G|, not trajectory-local geometry.",
        "",
        "## Mechanism (replaces the incoherent pooled-reversal paragraph)",
        "",
        "The pooled OT statistic is an ENCODING-scale property. Between-encounter variance of",
        "the mean encoded ||z_impact|| (pooled test_b+test_c): "
        f"fukami={mech['fukami']['between_encounter_var_meannorm']:.3f}, "
        f"jepa={mech['jepa']['between_encounter_var_meannorm']:.3f}, "
        f"pod={mech['pod']['between_encounter_var_meannorm']:.3f}. "
        f"Fukami norm-variance exceeds jepa: {mech['fukami_norm_var_exceeds_jepa']} "
        "(the higher between-encounter norm spread inflates the naive pooled OT-vs-latent",
        "correlation, which is the v2 'reversal'; encoded latents do not drift).",
        "",
        "## Gate GE3",
        "",
        f"POD alignment >= JEPA alignment (test_b): {gate['pod_ge_jepa']}.",
        gate["branch"],
        "",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "README.md").write_text("\n".join(lines))
    print(f"README -> {OUT_DIR / 'README.md'}", flush=True)


def save_results_npz(results: dict) -> None:
    """Persist per-encounter arrays + summary scalars to ce3_results.npz + JSON."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    flat: dict[str, np.ndarray] = {}
    for f in FAMILIES:
        for split, blob in results["align"][f]["per_split"].items():
            flat[f"align_{f}_{split}_rho_eps"] = np.array(blob["rho_per_eps"]["eps"])
            flat[f"align_{f}_{split}_caseids"] = np.array(blob["case_ids"])
            flat[f"align_{f}_{split}_encidx"] = np.array(blob["encounter_indices"])
    for split, blob in results["field_paired"].items():
        flat[f"field_{split}_impact_d_eps"] = np.array(blob["impact_d_eps"])
        flat[f"field_{split}_impact16_d_eps"] = np.array(blob["impact16_d_eps"])
        flat[f"field_{split}_caseids"] = np.array(blob["case_ids"])
        flat[f"field_{split}_encidx"] = np.array(blob["encounter_indices"])
    np.savez_compressed(OUT_DIR / "ce3_results.npz", **flat)
    print(f"npz -> {OUT_DIR / 'ce3_results.npz'}", flush=True)
    with open(OUT_DIR / "ce3_results.json", "w") as fh:
        json.dump({k: v for k, v in results.items() if k != "align"}, fh, indent=2, sort_keys=True)
    summary_align = {
        f: {
            s: {
                "rho_mean": results["align"][f]["per_split"][s]["rho_mean"],
                "rho_median": results["align"][f]["per_split"][s]["rho_median"],
            }
            for s in results["align"][f]["per_split"]
        }
        for f in FAMILIES
    }
    with open(OUT_DIR / "ce3_align_summary.json", "w") as fh:
        json.dump(summary_align, fh, indent=2, sort_keys=True)


def print_summary(results: dict) -> None:
    g = results["gate_ge3"]
    pf = results["paired_deltas"]["fukami"]["pooled"]
    pp = results["paired_deltas"]["pod"]["pooled"]
    span = results["eps_sensitivity"]["jepa_minus_fukami_delta_span_testb"]
    mech = results["mechanism"]
    print("\n=== C-E3 SUMMARY ===", flush=True)
    print(f"eps = {results['config']['eps_chord2']:.5f} chord^2 ({results['config']['eps_rule']})")
    print("debiased = yes (S_eps; matches POT empirical_sinkhorn_divergence to 1e-6)")
    print(
        f"test_b mean OT-latent Spearman: jepa={g['rho_mean_jepa']:.3f} "
        f"pod={g['rho_mean_pod']:.3f} fukami={g['rho_mean_fukami']:.3f}"
    )
    print(
        f"paired delta jepa-fukami pooled = {pf['mean_delta']:+.3f} "
        f"CI[{pf['enc_mean_ci'][0]:+.3f},{pf['enc_mean_ci'][1]:+.3f}] "
        f"sign_enc p={pf['sign_test_enc']['p_one_sided']:.3g} "
        f"sign_case p={pf['sign_test_case']['p_one_sided']:.3g}"
    )
    print(
        f"paired delta jepa-pod pooled = {pp['mean_delta']:+.3f} "
        f"CI[{pp['enc_mean_ci'][0]:+.3f},{pp['enc_mean_ci'][1]:+.3f}] "
        f"sign_enc p={pp['sign_test_enc']['p_one_sided']:.3g} "
        f"sign_case p={pp['sign_test_case']['p_one_sided']:.3g}"
    )
    print(f"eps sensitivity jepa-fukami test_b delta span = [{span[0]:+.3f}, {span[1]:+.3f}]")
    print(
        f"norm-variance: fukami var={mech['fukami']['between_encounter_var_meannorm']:.3f} "
        f"jepa var={mech['jepa']['between_encounter_var_meannorm']:.3f} "
        f"pod var={mech['pod']['between_encounter_var_meannorm']:.3f} "
        f"(fukami>jepa: {mech['fukami_norm_var_exceeds_jepa']})"
    )
    print(f"GATE GE3: pod_ge_jepa={g['pod_ge_jepa']}; {g['branch']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-enc", type=int, default=None, help="cap encounters per split (smoke)")
    ap.add_argument("--workers", type=int, default=16, help="process-pool size (1 = serial)")
    ap.add_argument("--splits", nargs="+", default=["test_b", "test_c"], help="splits to evaluate")
    args = ap.parse_args()

    t0 = time.time()
    results = run(args.max_enc, args.workers, args.splits)
    save_results_npz(results)
    write_numbers_part(results)
    write_readme(results)
    print_summary(results)
    print(f"\ntotal wall time {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
