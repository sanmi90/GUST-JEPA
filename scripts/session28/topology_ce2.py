"""Topology fairness for the latent encounter cycles (master plan C-E2; closes referee M4).

Referee M4: the v2 Vietoris-Rips persistence was computed on RAW latent
coordinates with a noise floor set as a fraction of cloud diameter. The JEPA
and reconstructive latents have, by construction, wildly different metric
structure: the JEPA latent is whitened toward an isotropic Gaussian by its
SIGReg regulariser, while the autoencoder (AE) latent is whatever the decoder
found convenient. Anisotropic per-coordinate scaling ALONE can fragment a clean
loop into multiple short-lived generators (or kill the single persistent
generator outright), so a raw-coordinate persistence comparison between families
is not metric-fair. The fair version recomputes persistence after per-family
standardisation (cheap whitening: z-score each latent dimension by that
family's TRAIN per-dim std) and, optionally, after a full Mahalanobis whitening
by the family's train covariance, and adds the cleanest possible control: the
undisturbed no-gust Baseline limit cycle per family.

The topological object is the per-encounter latent TRAJECTORY point cloud: z
over the 120 frames traces the shedding limit cycle, so a clean cycle is one
persistent H1 generator. Fragmentation shows up as the loss of that single
generator (zero, or many short-lived ones). We count "highly persistent" H0
(connected components) and H1 (loops) generators per encounter at the canonical
5 percent persistence floor (floor = fraction of the cloud diameter) plus the
robustness grid {2, 5, 10, 20} percent.

Gate GE2:
    STRONG if fragmentation survives whitening AND the reconstructive (fukami)
    family fragments even the no-gust Baseline limit cycle: the topology claim
    strengthens.
    WEAK if per-family standardisation HEALS the fukami fragmentation (the loop
    becomes one generator): the figure reframes from "topology" to "metric
    organisation". Either way the abstract attribution is fixed: fragmentation
    is an ENCODING property (it is measured on the simulation-ENCODED latent),
    manifold departure is a ROLLOUT property (Sec 4.3 drift). The v2 abstract
    conflated them by saying the reconstructive ROLLOUT "fragments the encounter
    topology"; fragmentation belongs to the ENCODING.

CLAUDE.md subspace-overlap caveat is respected in spirit: we never read a
near-floor generator-count difference as signal. The headline contrast is the
distribution of the single clean H1 generator (count == 1 vs not), and the gate
decisions use whole-number generator counts, not marginal lifetimes.

Data: per-family ENCODED latents at
    outputs/session28/latents/<family>/{train,test_b,test_c}.npz
with key z_full (n_enc, 120, d) and case_ids / encounter_indices (the loader
tolerates singular and plural key spellings). The no-gust control is the
Baseline case (case_id == "Baseline") in the train split.

CPU only; ripser 0.6.15, persim 0.3.8 (gudhi is NOT installed).

Outputs:
    outputs/session28/topology/ce2_results.npz        per-encounter generator
                                                       counts per family x metric
                                                       x floor (H0 and H1)
    outputs/session28/numbers_parts/topology_ce2.json eval_all part: headline
                                                       H1-count contrast raw vs
                                                       whitened per family, the
                                                       Baseline no-gust generator
                                                       count per family, GE2 branch
    outputs/session28/topology/README.md              metrics, GE2 branch, and
                                                       the encoding-vs-rollout
                                                       attribution fix

Usage:
    python scripts/session28/topology_ce2.py
    python scripts/session28/topology_ce2.py --families jepa_tf_noc_d64_s42 \
        fukami_d64_s42 pod_d64 --no-maha
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
from ripser import ripser as ripser_fn
from scipy.spatial.distance import pdist

REPO = Path(__file__).resolve().parents[2]
LATENTS_DIR = REPO / "outputs" / "session28" / "latents"
OUT_DIR = REPO / "outputs" / "session28" / "topology"
PARTS_DIR = REPO / "outputs" / "session28" / "numbers_parts"

# The three core comparators (master plan C-E2); bvae is optional.
CORE_FAMILIES = ["jepa_tf_noc_d64_s42", "fukami_d64_s42", "pod_d64"]
OPTIONAL_FAMILIES = ["bvae_faith_d64_s42"]

CANONICAL_FLOOR = 0.05
FLOOR_GRID = (0.02, 0.05, 0.10, 0.20)
GUSTED_SPLITS = ("test_b", "test_c")
BASELINE_CASE = "Baseline"

# A short, stable display name per family for macros and the README.
FAMILY_LABEL = {
    "jepa_tf_noc_d64_s42": "Jepa",
    "fukami_d64_s42": "Fukami",
    "pod_d64": "Pod",
    "bvae_faith_d64_s42": "Bvae",
}


# --------------------------------------------------------------------------- #
# NPZ loading (tolerate singular / plural key spellings)
# --------------------------------------------------------------------------- #
def _pick_key(keys: Iterable[str], candidates: Iterable[str]) -> str | None:
    keyset = set(keys)
    for cand in candidates:
        if cand in keyset:
            return cand
    return None


def load_split(family: str, split: str) -> dict[str, np.ndarray]:
    """Load one family/split NPZ, returning z_full, case_ids, encounter_indices.

    Tolerates both singular (case_id, encounter_index) and plural
    (case_ids, encounter_indices) key spellings (M4 deliverable note).
    """
    path = LATENTS_DIR / family / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing latents: {path}")
    data = np.load(path, allow_pickle=True)
    keys = list(data.keys())
    z_key = _pick_key(keys, ["z_full", "z_fulls", "Z_full"])
    cid_key = _pick_key(keys, ["case_ids", "case_id", "caseids", "caseid"])
    enc_key = _pick_key(
        keys, ["encounter_indices", "encounter_index", "encounter_idx", "encounter"]
    )
    if z_key is None or cid_key is None:
        raise KeyError(f"{path}: need z_full and case_ids; have {keys}")
    z_full = np.asarray(data[z_key], dtype=np.float64)
    case_ids = np.asarray(data[cid_key]).astype(str)
    if enc_key is not None:
        enc_idx = np.asarray(data[enc_key]).astype(int)
    else:
        enc_idx = np.arange(z_full.shape[0], dtype=int)
    return {"z_full": z_full, "case_ids": case_ids, "encounter_indices": enc_idx}


# --------------------------------------------------------------------------- #
# Per-family whitening transforms (fit on TRAIN frames, pooled over encounters)
# --------------------------------------------------------------------------- #
def fit_whiteners(family: str) -> dict[str, dict[str, np.ndarray]]:
    """Fit per-dim std and full-covariance (Mahalanobis) whiteners on TRAIN.

    Both are estimated from the pooled (encounter x frame) TRAIN latent matrix,
    i.e. the family's own train per-dim std and train covariance. The point of
    M4 is to remove the family-specific anisotropic per-coordinate scale before
    comparing topology.
    """
    train = load_split(family, "train")
    # (n_enc, 120, d) -> (n_enc * 120, d) pooled frames.
    z = train["z_full"].reshape(-1, train["z_full"].shape[-1])
    mean = z.mean(axis=0)
    std = z.std(axis=0)
    # Guard against dead dimensions (POD tail modes can be ~0 on some splits).
    std_safe = np.where(std > 1e-12, std, 1.0)

    # Full covariance whitening: W = mean-centre then multiply by Sigma^{-1/2}.
    cov = np.cov(z, rowvar=False)
    cov = cov + 1e-8 * np.eye(cov.shape[0])
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, 1e-12, None)
    w_maha = evecs @ np.diag(evals**-0.5) @ evecs.T

    return {
        "stdz": {"mean": mean, "scale": std_safe},
        "maha": {"mean": mean, "matrix": w_maha},
    }


def apply_metric(cloud: np.ndarray, metric: str, whit: dict) -> np.ndarray:
    """Transform a (T, d) trajectory point cloud into the chosen metric space."""
    if metric == "raw":
        return cloud
    if metric == "stdz":
        return (cloud - whit["stdz"]["mean"]) / whit["stdz"]["scale"]
    if metric == "maha":
        return (cloud - whit["maha"]["mean"]) @ whit["maha"]["matrix"]
    raise ValueError(f"unknown metric {metric!r}")


# --------------------------------------------------------------------------- #
# Persistence and generator counting
# --------------------------------------------------------------------------- #
def persistence_counts(cloud: np.ndarray, floors: Iterable[float]) -> dict[float, dict[str, int]]:
    """Vietoris-Rips H0/H1 persistent-generator counts at each floor fraction.

    The noise floor is a fraction of the point cloud diameter (max pairwise
    distance), matching the v2 convention. H0: the one always-present
    infinite-lifetime component plus every finite-death component whose lifetime
    is at least floor * diameter (a clean connected cloud gives 1; separated
    blobs add one each). H1: every loop whose lifetime is at least
    floor * diameter (a clean cycle gives exactly 1; fragmentation gives 0 or
    many short-lived loops).
    """
    diam = float(pdist(cloud).max())
    if diam <= 0.0:
        # Degenerate cloud (all points coincide): one component, no loop.
        return {float(f): {"H0": 1, "H1": 0} for f in floors}
    res = ripser_fn(cloud, maxdim=1)
    dgm0 = res["dgms"][0]
    dgm1 = res["dgms"][1]

    fin0 = dgm0[np.isfinite(dgm0[:, 1])]
    n_inf0 = int((~np.isfinite(dgm0[:, 1])).sum())  # number of infinite components
    life0 = fin0[:, 1] - fin0[:, 0]

    fin1 = dgm1[np.isfinite(dgm1[:, 1])] if dgm1.size else dgm1.reshape(0, 2)
    life1 = fin1[:, 1] - fin1[:, 0] if fin1.size else np.zeros(0)

    out: dict[float, dict[str, int]] = {}
    for f in floors:
        thr = f * diam
        h0 = n_inf0 + int((life0 >= thr).sum())
        h1 = int((life1 >= thr).sum())
        out[float(f)] = {"H0": h0, "H1": h1}
    return out


# --------------------------------------------------------------------------- #
# Shedding-period estimate (for the single-period no-gust control)
# --------------------------------------------------------------------------- #
# The undisturbed Baseline shedding period is ~59 frames (St_full = 0.338,
# dt_tc = 0.05), so one 120-frame Baseline encounter spans ~2 full limit-cycle
# loops. A multi-period trajectory is NOT a single clean H1 generator under
# Vietoris-Rips even for a perfect limit cycle, so the no-gust control is
# reported BOTH on the full 120-frame encounter (continuity with the gusted
# encounters, which are dominated by the single impact event) AND segmented into
# single shedding periods, where "clean cycle -> one generator" is the correct
# null. The gusted encounters are left whole because the impact concentrates the
# trajectory into one dominant excursion.
SHEDDING_PERIOD_FRAMES_ANCHOR = 59


def estimate_period_frames(traj: np.ndarray, anchor: int = SHEDDING_PERIOD_FRAMES_ANCHOR) -> int:
    """Estimate the limit-cycle period (frames) from the PC1 autocorrelation peak.

    Projects the (T, d) trajectory onto its leading principal component and
    finds the first autocorrelation peak in a window around the St-derived
    anchor. Falls back to the anchor if no clean peak is found.
    """
    x = traj - traj.mean(axis=0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    pc1 = x @ vt[0]
    pc1 = pc1 - pc1.mean()
    n = pc1.size
    ac = np.correlate(pc1, pc1, mode="full")[n - 1 :]
    ac = ac / (ac[0] + 1e-12)
    lo, hi = max(8, anchor // 2), min(n - 1, int(anchor * 1.6))
    if hi <= lo + 1:
        return int(anchor)
    window = ac[lo:hi]
    peak = lo + int(np.argmax(window))
    # Require a genuine local maximum, else trust the anchor.
    if 0 < peak < n - 1 and ac[peak] > ac[peak - 1] and ac[peak] >= ac[peak + 1]:
        return int(peak)
    return int(anchor)


def segment_periods(traj: np.ndarray, period: int) -> list[np.ndarray]:
    """Split a (T, d) trajectory into consecutive single-period (period, d) windows."""
    n = traj.shape[0]
    if period < 8 or period > n:
        return [traj]
    segs = [traj[s : s + period] for s in range(0, n - period + 1, period)]
    return [s for s in segs if s.shape[0] >= 8]


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def analyse_family(family: str, metrics: list[str], floors: tuple[float, ...]) -> dict:
    """Compute per-encounter generator counts for one family across metrics."""
    whit = fit_whiteners(family)

    rows = []  # one dict per (encounter, metric)
    for split in GUSTED_SPLITS:
        sp = load_split(family, split)
        for i in range(sp["z_full"].shape[0]):
            cloud = sp["z_full"][i]  # (120, d)
            for metric in metrics:
                tc = apply_metric(cloud, metric, whit)
                counts = persistence_counts(tc, floors)
                for floor, c in counts.items():
                    rows.append(
                        {
                            "split": split,
                            "case_id": sp["case_ids"][i],
                            "encounter_index": int(sp["encounter_indices"][i]),
                            "metric": metric,
                            "floor": floor,
                            "H0": c["H0"],
                            "H1": c["H1"],
                        }
                    )

    # No-gust control: the Baseline encounters live in the TRAIN split. Reported
    # two ways: the full 120-frame encounter (multi-period, like the gusted ones)
    # and segmented into single shedding periods (the clean-limit-cycle null).
    train = load_split(family, "train")
    base_mask = train["case_ids"] == BASELINE_CASE
    base_rows = []
    base_period_rows = []
    base_idx = np.nonzero(base_mask)[0]
    period = estimate_period_frames(train["z_full"][base_idx[0]]) if base_idx.size else 0
    for i in base_idx:
        cloud = train["z_full"][i]
        enc = int(train["encounter_indices"][i])
        for metric in metrics:
            tc = apply_metric(cloud, metric, whit)
            counts = persistence_counts(tc, floors)
            for floor, c in counts.items():
                base_rows.append(
                    {
                        "case_id": BASELINE_CASE,
                        "encounter_index": enc,
                        "metric": metric,
                        "floor": floor,
                        "H0": c["H0"],
                        "H1": c["H1"],
                    }
                )
            for seg_i, seg in enumerate(segment_periods(tc, period)):
                pc = persistence_counts(seg, floors)
                for floor, c in pc.items():
                    base_period_rows.append(
                        {
                            "case_id": BASELINE_CASE,
                            "encounter_index": enc,
                            "period_index": seg_i,
                            "metric": metric,
                            "floor": floor,
                            "H0": c["H0"],
                            "H1": c["H1"],
                        }
                    )

    return {
        "gusted": rows,
        "baseline": base_rows,
        "baseline_period": base_period_rows,
        "n_baseline": int(base_mask.sum()),
        "period_frames": int(period),
    }


def _h1_at(rows: list[dict], metric: str, floor: float) -> np.ndarray:
    sel = [r["H1"] for r in rows if r["metric"] == metric and abs(r["floor"] - floor) < 1e-9]
    return np.asarray(sel, dtype=float)


def _frac_single(rows: list[dict], metric: str, floor: float) -> float:
    """Fraction of clouds that are a single clean H1 generator at this floor."""
    arr = _h1_at(rows, metric, floor)
    if arr.size == 0:
        return float("nan")
    return float((arr == 1).mean())


def decide_gate(per_family: dict, headline_metric: str) -> dict:
    """GE2 branch from the fair fukami-vs-jepa H1 contrast under whitening.

    M4 makes the comparison metric-fair, so the decisive test is COMPARATIVE: a
    single-family "fukami fragments" statement is not enough (the full-encounter
    Baseline traces ~2 shedding periods and so fragments for every family,
    including JEPA; that is reported but is not the discriminator). The decisive
    quantities, all at the 5 percent floor:

      g_jepa, g_fuk   = fraction of GUSTED encounters that are a single clean H1
                        generator, JEPA vs fukami, under the headline whitened
                        metric.
      b_jepa, b_fuk   = same fraction on the SINGLE-PERIOD no-gust control (the
                        clean-limit-cycle null), under the whitened metric.

    STRONG : after whitening the predictive (JEPA) family keeps the single clean
             generator more often than the reconstructive (fukami) family on BOTH
             the gusted encounters AND the single-period no-gust control, i.e.
             fragmentation is a property of the reconstructive ENCODING that
             survives removing the anisotropic per-coordinate scale.
    WEAK   : whitening closes the gusted gap (fukami's single-cycle fraction
             reaches JEPA's, gap <= 0.1) -> reframe Figure 5 from "topology" to
             "metric organisation".
    """
    fuk = per_family["fukami_d64_s42"]
    jep = per_family.get("jepa_tf_noc_d64_s42")

    g_fuk_raw = _frac_single(fuk["gusted"], "raw", CANONICAL_FLOOR)
    g_fuk_wht = _frac_single(fuk["gusted"], headline_metric, CANONICAL_FLOOR)
    g_jep_wht = (
        _frac_single(jep["gusted"], headline_metric, CANONICAL_FLOOR)
        if jep is not None
        else float("nan")
    )

    # Single-period no-gust control (the clean-limit-cycle null).
    b_fuk_wht = _frac_single(fuk["baseline_period"], headline_metric, CANONICAL_FLOOR)
    b_jep_wht = (
        _frac_single(jep["baseline_period"], headline_metric, CANONICAL_FLOOR)
        if jep is not None
        else float("nan")
    )

    gusted_gap = g_jep_wht - g_fuk_wht  # >0 means JEPA keeps the cycle more often
    baseline_gap = b_jep_wht - b_fuk_wht

    # "Healed" if whitening brings fukami's gusted single-cycle fraction up to
    # JEPA's (the metric-organisation explanation): the gap closes or reverses.
    gusted_healed = np.isfinite(gusted_gap) and gusted_gap <= 0.1
    # The single-period no-gust control separates the families in the predictive
    # family's favour (the clean-limit-cycle test the referee asked for).
    nogust_separates = np.isfinite(baseline_gap) and baseline_gap > 0.2

    if gusted_healed and nogust_separates:
        branch = "WEAK-MIXED"
        reason = (
            "split verdict. On the WHOLE gusted encounters per-family "
            f"standardisation closes/reverses the gap (gap = {gusted_gap:+.2f}, "
            "fukami is read as a single cycle at least as often as the "
            "predictive family), so the v2 raw-coordinate gusted fragmentation "
            "claim was an anisotropic-scale artefact and Figure 5 must reframe "
            "from topology to metric organisation. BUT the single-period no-gust "
            f"limit-cycle control DOES separate the families (gap = "
            f"{baseline_gap:+.2f}: the reconstructive encoding fragments the "
            "clean shedding cycle while the predictive encoding keeps it), and "
            "this survives whitening; the clean-cycle topology statement holds "
            "on the no-gust control, scoped to the limit cycle"
        )
    elif gusted_healed:
        branch = "WEAK"
        reason = (
            "per-family standardisation closes the fukami-vs-jepa gusted gap "
            f"(gap = {gusted_gap:+.2f} <= 0.10 single-cycle fraction) and the "
            f"single-period no-gust control does not separate the families "
            f"(gap = {baseline_gap:+.2f}); the fragmentation was an "
            "anisotropic-scale artefact, so reframe Figure 5 from topology to "
            "metric organisation"
        )
    elif nogust_separates:
        branch = "STRONG"
        reason = (
            "after whitening the predictive family keeps the single clean "
            f"generator more often than the reconstructive family on gusted "
            f"encounters (gap = {gusted_gap:+.2f}) AND on the single-period "
            f"no-gust limit cycle (gap = {baseline_gap:+.2f}); fragmentation is "
            "a property of the reconstructive ENCODING that survives whitening"
        )
    else:
        branch = "STRONG-PARTIAL"
        reason = (
            f"the gusted gap survives whitening (gap = {gusted_gap:+.2f}) but the "
            "single-period no-gust control does not separate the families "
            f"(gap = {baseline_gap:+.2f}); report the surviving gusted "
            "fragmentation and qualify the no-gust line"
        )

    return {
        "branch": branch,
        "reason": reason,
        "headline_metric": headline_metric,
        "fukami_gusted_frac_single_raw": g_fuk_raw,
        "fukami_gusted_frac_single_whitened": g_fuk_wht,
        "jepa_gusted_frac_single_whitened": g_jep_wht,
        "gusted_gap_jepa_minus_fukami": float(gusted_gap),
        "fukami_baseline_period_frac_single_whitened": b_fuk_wht,
        "jepa_baseline_period_frac_single_whitened": b_jep_wht,
        "baseline_period_gap_jepa_minus_fukami": float(baseline_gap),
    }


# --------------------------------------------------------------------------- #
# Output assembly
# --------------------------------------------------------------------------- #
def build_part(per_family: dict, metrics: list[str], gate: dict) -> dict:
    """eval_all part: headline raw-vs-whitened H1 contrast, Baseline counts, GE2."""
    headline_metric = gate["headline_metric"]
    numbers: dict[str, dict] = {}

    for fam, res in per_family.items():
        label = FAMILY_LABEL.get(fam, fam.title().replace("_", ""))
        # Median gusted H1 generator count, raw vs headline whitened metric.
        raw = _h1_at(res["gusted"], "raw", CANONICAL_FLOOR)
        wht = _h1_at(res["gusted"], headline_metric, CANONICAL_FLOOR)
        raw_med = float(np.median(raw)) if raw.size else float("nan")
        wht_med = float(np.median(wht)) if wht.size else float("nan")
        raw_single = _frac_single(res["gusted"], "raw", CANONICAL_FLOOR)
        wht_single = _frac_single(res["gusted"], headline_metric, CANONICAL_FLOOR)

        numbers[f"topo_h1_gusted_raw_median_{fam}"] = {
            "macro": f"TopoHoneRawMed{label}",
            "value": raw_med,
            "fmt": "%.1f",
            "n": int(raw.size),
            "split": "test_b+test_c",
            "source": "topology_ce2.py",
            "note": (
                f"median per-encounter persistent H1 generator count, RAW latent "
                f"coordinates, {int(CANONICAL_FLOOR * 100)}% diameter floor; "
                f"frac single-cycle = {raw_single:.2f}"
            ),
        }
        numbers[f"topo_h1_gusted_whitened_median_{fam}"] = {
            "macro": f"TopoHoneWhtMed{label}",
            "value": wht_med,
            "fmt": "%.1f",
            "n": int(wht.size),
            "split": "test_b+test_c",
            "source": "topology_ce2.py",
            "note": (
                f"median per-encounter persistent H1 generator count, "
                f"per-family {headline_metric} whitening, "
                f"{int(CANONICAL_FLOOR * 100)}% diameter floor; "
                f"frac single-cycle = {wht_single:.2f}"
            ),
        }
        # No-gust control, single-period segmentation (the clean-limit-cycle null).
        # The full 120-frame Baseline encounter traces ~2 shedding periods and so
        # fragments for every family; the single-period median is the fair "clean
        # cycle == 1" reference.
        bp_raw = _h1_at(res["baseline_period"], "raw", CANONICAL_FLOOR)
        bp_wht = _h1_at(res["baseline_period"], headline_metric, CANONICAL_FLOOR)
        bp_raw_single = _frac_single(res["baseline_period"], "raw", CANONICAL_FLOOR)
        bp_wht_single = _frac_single(res["baseline_period"], headline_metric, CANONICAL_FLOOR)
        b_full_raw = _h1_at(res["baseline"], "raw", CANONICAL_FLOOR)
        numbers[f"topo_h1_baseline_period_raw_median_{fam}"] = {
            "macro": f"TopoBaseRawMed{label}",
            "value": float(np.median(bp_raw)) if bp_raw.size else float("nan"),
            "fmt": "%.1f",
            "n": int(bp_raw.size),
            "split": "baseline",
            "source": "topology_ce2.py",
            "note": (
                "no-gust Baseline SINGLE-PERIOD limit-cycle median persistent H1 "
                f"count, RAW metric, {int(CANONICAL_FLOOR * 100)}% diameter floor "
                f"(clean cycle == 1); period = {res['period_frames']} frames; "
                f"frac single-cycle = {bp_raw_single:.2f}; "
                f"full-encounter (multi-period) median = "
                f"{float(np.median(b_full_raw)) if b_full_raw.size else float('nan'):.1f}"
            ),
        }
        numbers[f"topo_h1_baseline_period_whitened_median_{fam}"] = {
            "macro": f"TopoBaseWhtMed{label}",
            "value": float(np.median(bp_wht)) if bp_wht.size else float("nan"),
            "fmt": "%.1f",
            "n": int(bp_wht.size),
            "split": "baseline",
            "source": "topology_ce2.py",
            "note": (
                "no-gust Baseline SINGLE-PERIOD limit-cycle median persistent H1 "
                f"count, {headline_metric} whitening, "
                f"{int(CANONICAL_FLOOR * 100)}% diameter floor (clean cycle == 1); "
                f"frac single-cycle = {bp_wht_single:.2f}"
            ),
        }

    # The decisive comparative gap (predictive minus reconstructive single-cycle
    # fraction) under the headline whitened metric, gusted and no-gust control.
    numbers["topo_gusted_gap_jepa_minus_fukami"] = {
        "value": gate["gusted_gap_jepa_minus_fukami"],
        "fmt": "%.2f",
        "split": "test_b+test_c",
        "source": "topology_ce2.py",
        "note": (
            "JEPA minus fukami single-cycle fraction on gusted encounters under "
            f"{headline_metric} whitening at {int(CANONICAL_FLOOR * 100)}% floor "
            f"(jepa = {gate['jepa_gusted_frac_single_whitened']:.2f}, "
            f"fukami = {gate['fukami_gusted_frac_single_whitened']:.2f}); >0 means "
            "the predictive encoding keeps the clean cycle more often"
        ),
    }
    numbers["topo_baseline_period_gap_jepa_minus_fukami"] = {
        "value": gate["baseline_period_gap_jepa_minus_fukami"],
        "fmt": "%.2f",
        "split": "baseline",
        "source": "topology_ce2.py",
        "note": (
            "JEPA minus fukami single-cycle fraction on the SINGLE-PERIOD no-gust "
            f"control under {headline_metric} whitening "
            f"(jepa = {gate['jepa_baseline_period_frac_single_whitened']:.2f}, "
            f"fukami = {gate['fukami_baseline_period_frac_single_whitened']:.2f})"
        ),
    }

    # GE2 branch encoded as an integer macro
    # (1 STRONG, 2 WEAK-MIXED, 0 WEAK, -1 STRONG-PARTIAL), plus a human-readable
    # note carrying the branch string and reason.
    branch_code = {"STRONG": 1, "WEAK-MIXED": 2, "WEAK": 0, "STRONG-PARTIAL": -1}[gate["branch"]]
    numbers["topo_ge2_branch"] = {
        "macro": "TopoGEtwoBranch",
        "value": branch_code,
        "fmt": "%d",
        "source": "topology_ce2.py",
        "note": (
            f"GE2 branch = {gate['branch']} "
            "(1 STRONG / 2 WEAK-MIXED / 0 WEAK / -1 STRONG-PARTIAL): "
            f"{gate['reason']}. Headline whitened metric = {headline_metric}. "
            "Encoding-vs-rollout fix: fragmentation is an ENCODING property "
            "(measured on simulation-encoded latents here); manifold departure "
            "is a ROLLOUT property (Sec 4.3 drift)."
        ),
    }
    return {"part": "topology_ce2", "numbers": numbers}


def write_npz(per_family: dict, metrics: list[str], floors: tuple[float, ...]) -> Path:
    """Flatten every per-encounter row into parallel arrays for the NPZ archive."""
    fams, splits, cids, encs, mets, flrs, h0s, h1s = ([] for _ in range(8))
    for fam, res in per_family.items():
        for r in res["gusted"]:
            fams.append(fam)
            splits.append(r["split"])
            cids.append(r["case_id"])
            encs.append(r["encounter_index"])
            mets.append(r["metric"])
            flrs.append(r["floor"])
            h0s.append(r["H0"])
            h1s.append(r["H1"])
        for r in res["baseline"]:
            fams.append(fam)
            splits.append("baseline_full")
            cids.append(r["case_id"])
            encs.append(r["encounter_index"])
            mets.append(r["metric"])
            flrs.append(r["floor"])
            h0s.append(r["H0"])
            h1s.append(r["H1"])
        for r in res["baseline_period"]:
            fams.append(fam)
            splits.append("baseline_period")
            cids.append(r["case_id"])
            encs.append(r["encounter_index"])
            mets.append(r["metric"])
            flrs.append(r["floor"])
            h0s.append(r["H0"])
            h1s.append(r["H1"])
    out = OUT_DIR / "ce2_results.npz"
    np.savez_compressed(
        out,
        family=np.asarray(fams),
        split=np.asarray(splits),
        case_id=np.asarray(cids),
        encounter_index=np.asarray(encs, dtype=int),
        metric=np.asarray(mets),
        floor=np.asarray(flrs, dtype=float),
        H0=np.asarray(h0s, dtype=int),
        H1=np.asarray(h1s, dtype=int),
        metrics_used=np.asarray(metrics),
        floor_grid=np.asarray(floors, dtype=float),
        canonical_floor=np.asarray(CANONICAL_FLOOR),
        period_families=np.asarray(list(per_family.keys())),
        period_frames=np.asarray([per_family[f]["period_frames"] for f in per_family], dtype=int),
    )
    return out


def write_readme(per_family: dict, metrics: list[str], gate: dict) -> Path:
    headline_metric = gate["headline_metric"]
    lines = []
    lines.append("# Topology fairness (C-E2; closes referee M4)")
    lines.append("")
    lines.append(
        "Vietoris-Rips persistence (ripser, maxdim=1) on the per-encounter latent "
        "trajectory point cloud (z over 120 frames traces the shedding limit cycle; "
        "a clean cycle is one persistent H1 generator). Generators counted at the "
        "canonical 5 percent diameter floor and over the robustness grid "
        f"{[int(f * 100) for f in FLOOR_GRID]} percent."
    )
    lines.append("")
    lines.append("## Metrics (per family)")
    lines.append("")
    lines.append("- raw: learned-latent coordinates as stored (continuity with v2).")
    lines.append(
        "- stdz: per-family standardisation, z-score each latent dim by that "
        "family's TRAIN per-dim std (the cheap whitening; the headline fair metric)."
    )
    if "maha" in metrics:
        lines.append(
            "- maha: full Mahalanobis whitening by the family's TRAIN covariance "
            "(Sigma^{-1/2}); reported for completeness."
        )
    lines.append("")
    lines.append(f"Headline whitened metric for the GE2 decision: {headline_metric}.")
    lines.append("")

    def med(a):
        return f"{np.median(a):.1f}" if a.size else "nan"

    lines.append("## Gusted encounters (test_b + test_c), H1 at the 5 percent floor")
    lines.append("")
    lines.append(
        "Median H1 generator count and the fraction of encounters read as a "
        "single clean cycle (count == 1), raw vs the headline whitened metric. "
        "Gusted encounters are kept whole: the impact event concentrates the "
        "trajectory into one dominant excursion."
    )
    lines.append("")
    lines.append(
        "| family | median (raw) | median (whitened) | "
        "frac single (raw) | frac single (whitened) |"
    )
    lines.append("|---|---|---|---|---|")
    for fam, res in per_family.items():
        raw = _h1_at(res["gusted"], "raw", CANONICAL_FLOOR)
        wht = _h1_at(res["gusted"], headline_metric, CANONICAL_FLOOR)
        lines.append(
            f"| {fam} | {med(raw)} | {med(wht)} | "
            f"{_frac_single(res['gusted'], 'raw', CANONICAL_FLOOR):.2f} | "
            f"{_frac_single(res['gusted'], headline_metric, CANONICAL_FLOOR):.2f} |"
        )
    lines.append("")
    lines.append("## No-gust Baseline control, H1 at the 5 percent floor")
    lines.append("")
    lines.append(
        "The full 120-frame Baseline encounter traces ~2 shedding periods "
        "(St_full = 0.338, dt_tc = 0.05, period ~59 frames), so it fragments for "
        "EVERY family, including the predictive one; that full-encounter number "
        "is reported but is not the clean-cycle null. The fair clean-limit-cycle "
        "reference is the SINGLE-PERIOD segmentation, where a clean cycle is one "
        "generator."
    )
    lines.append("")
    lines.append(
        "| family | period (frames) | full-encounter median (raw) | "
        "single-period median (raw) | single-period median (whitened) | "
        "single-period frac single (whitened) |"
    )
    lines.append("|---|---|---|---|---|---|")
    for fam, res in per_family.items():
        bf_raw = _h1_at(res["baseline"], "raw", CANONICAL_FLOOR)
        bp_raw = _h1_at(res["baseline_period"], "raw", CANONICAL_FLOOR)
        bp_wht = _h1_at(res["baseline_period"], headline_metric, CANONICAL_FLOOR)
        lines.append(
            f"| {fam} | {res['period_frames']} | {med(bf_raw)} | {med(bp_raw)} | "
            f"{med(bp_wht)} | "
            f"{_frac_single(res['baseline_period'], headline_metric, CANONICAL_FLOOR):.2f} |"
        )
    lines.append("")
    lines.append("## Decisive comparative gap (predictive minus reconstructive)")
    lines.append("")
    lines.append(
        f"Single-cycle fraction, JEPA minus fukami, under {headline_metric} "
        "whitening at the 5 percent floor:"
    )
    lines.append("")
    lines.append(
        f"- gusted encounters: {gate['gusted_gap_jepa_minus_fukami']:+.2f} "
        f"(jepa {gate['jepa_gusted_frac_single_whitened']:.2f}, "
        f"fukami {gate['fukami_gusted_frac_single_whitened']:.2f})"
    )
    lines.append(
        f"- single-period no-gust control: "
        f"{gate['baseline_period_gap_jepa_minus_fukami']:+.2f} "
        f"(jepa {gate['jepa_baseline_period_frac_single_whitened']:.2f}, "
        f"fukami {gate['fukami_baseline_period_frac_single_whitened']:.2f})"
    )
    lines.append("")
    lines.append("## GE2 branch")
    lines.append("")
    lines.append(f"Branch: {gate['branch']}")
    lines.append("")
    lines.append(gate["reason"])
    lines.append("")
    lines.append("## Encoding-vs-rollout attribution fix (abstract)")
    lines.append("")
    lines.append(
        "The v2 abstract said the reconstructive ROLLOUT 'fragments the encounter "
        "topology', but Sec 4.3 establishes fragmentation for the simulation-"
        "ENCODED latent and says the rollout lifetime ratio does NOT separate "
        "families. This analysis computes persistence on the simulation-ENCODED "
        "latent trajectories. Fragmentation therefore belongs to the ENCODING; "
        "manifold departure (the drift result, Sec 4.3) belongs to the ROLLOUT. "
        "The abstract must attribute fragmentation to the encoding and manifold "
        "departure to the rollout, regardless of the GE2 branch."
    )
    lines.append("")
    out = OUT_DIR / "README.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--families",
        nargs="+",
        default=None,
        help="Families to analyse (default: 3 core comparators + bvae if present).",
    )
    p.add_argument(
        "--no-maha",
        action="store_true",
        help="Skip the full Mahalanobis metric (keep raw + per-dim stdz only).",
    )
    p.add_argument(
        "--headline-metric",
        default="stdz",
        choices=["stdz", "maha"],
        help="Whitened metric used for the GE2 decision (default: stdz).",
    )
    args = p.parse_args()

    if args.families is not None:
        families = list(args.families)
    else:
        families = list(CORE_FAMILIES)
        for fam in OPTIONAL_FAMILIES:
            if (LATENTS_DIR / fam / "train.npz").exists():
                families.append(fam)

    metrics = ["raw", "stdz"] if args.no_maha else ["raw", "stdz", "maha"]
    if args.headline_metric not in metrics:
        metrics.append(args.headline_metric)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PARTS_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    per_family: dict[str, dict] = {}
    for fam in families:
        f0 = time.time()
        per_family[fam] = analyse_family(fam, metrics, FLOOR_GRID)
        n_g = len(
            {(r["split"], r["case_id"], r["encounter_index"]) for r in per_family[fam]["gusted"]}
        )
        print(
            f"[topology_ce2] {fam}: {n_g} gusted encounters, "
            f"{per_family[fam]['n_baseline']} Baseline encs "
            f"(period {per_family[fam]['period_frames']} frames), "
            f"{time.time() - f0:.1f}s"
        )

    gate = decide_gate(per_family, args.headline_metric)
    part = build_part(per_family, metrics, gate)

    npz_path = write_npz(per_family, metrics, FLOOR_GRID)
    readme_path = write_readme(per_family, metrics, gate)
    part_path = PARTS_DIR / "topology_ce2.json"
    part_path.write_text(json.dumps(part, indent=2) + "\n")

    print(f"[topology_ce2] wrote {npz_path}")
    print(f"[topology_ce2] wrote {part_path}")
    print(f"[topology_ce2] wrote {readme_path}")
    print(f"[topology_ce2] GE2 branch = {gate['branch']}: {gate['reason']}")
    print(f"[topology_ce2] total wall time {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
