"""Session 33 no-gust cycle topology on the pooled v2.2 latents (appendix).

Re-run of scripts/session28/topology_ce2.py for the v2.2 manuscript, per
SESSION_33_MANUSCRIPT_V3.md Section 11 item 6 ("The topology of the no-gust
cycle, appendix, if kept per D223"). The method is kept unchanged:
Vietoris-Rips persistence (ripser, maxdim = 1) on the per-encounter latent
trajectory point cloud, per-family standardisation (stdz) plus full
Mahalanobis whitening (maha) fitted on the family's pooled TRAIN frames, the
canonical 5 percent diameter noise floor with the {2, 5, 10, 20} percent
robustness grid, and the no-gust Baseline limit-cycle control reported both
on the full 120-frame encounter (multi-period) and segmented into single
shedding periods (the clean-cycle null, one persistent H1 generator). The H1
loop of the shedding limit cycle is the target signal.

Adaptations to the pooled v2.2 tier (deviations from topology_ce2.py):
- Inputs are the session31/32 pooled d = 32 per-frame caches (key z_gap);
  trajectories are reassembled by grouping on (case_id, encounter_index) and
  sorting by frame. The no-gust control is the case_id == "Baseline" rows of
  the train split (4 encounters x 120 frames).
- Families are jepa_pool (predictive), supervised_only_pool (supervised
  control), regAE_pool and pod (reconstructive baselines); the v2.1 fukami
  d = 64 family has no pooled v2.2 counterpart in this tier.
- Gusted encounters come from test_b ONLY: the pooled caches carry no test_c
  split (v2.1 used test_b + test_c).
- The GE2-style gate compares jepa_pool vs regAE_pool (the reconstructive
  comparator of the pooled tier) with the topology_ce2 branch logic and
  thresholds unchanged.
- Everything is written to one JSON (no NPZ/README/numbers-part triplet); the
  per-encounter generator-count rows keep the ce2 row schema.

CPU only (numpy/scipy/ripser); no GPU is touched. If ripser is not
importable the script writes {"status": "blocked", ...} and exits.

Output: outputs/session33/topology_v2p2.json.

Usage:
    export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
    taskset -c 16-23 nice -n 10 .venv/bin/python \\
        scripts/session33/topology_v2p2.py
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT_PATH_DEFAULT = REPO / "outputs" / "session33" / "topology_v2p2.json"

# Model tag -> per-frame pooled-latent caches (repo-relative).
MODEL_CACHES = {
    "jepa_pool": "outputs/session31/q1_latents/latents_jepa_pool_{split}.npz",
    "supervised_only_pool": (
        "outputs/session32/q1_pool_latents/latents_supervised_only_pool_{split}.npz"
    ),
    "regAE_pool": "outputs/session32/q1_pool_latents/latents_regAE_pool_{split}.npz",
    "pod": "outputs/session31/q1_latents/latents_pod_{split}.npz",
}

PREDICTIVE_FAMILY = "jepa_pool"
RECON_FAMILY = "regAE_pool"

CANONICAL_FLOOR = 0.05
FLOOR_GRID = (0.02, 0.05, 0.10, 0.20)
GUSTED_SPLITS = ("test_b",)  # the pooled caches carry no test_c (deviation from ce2)
BASELINE_CASE = "Baseline"

# Undisturbed Baseline shedding period anchor: ~59 frames (St_full = 0.338,
# dt_tc = 0.05), so one 120-frame Baseline encounter spans ~2 limit-cycle
# loops. Single-period segmentation is the clean-cycle null (ce2 convention).
SHEDDING_PERIOD_FRAMES_ANCHOR = 59


# --------------------------------------------------------------------------- #
# Loading (pooled per-frame caches -> per-encounter trajectories)
# --------------------------------------------------------------------------- #
def load_trajectories(family: str, split: str) -> dict[tuple[str, int], np.ndarray]:
    """Reassemble per-encounter (T, d) latent trajectories from a pooled cache."""
    path = REPO / MODEL_CACHES[family].format(split=split)
    blob = np.load(path, allow_pickle=True)
    z = blob["z_gap"].astype(np.float64)
    cids = np.array([str(c) for c in blob["case_id"]])
    encs = blob["encounter_index"].astype(np.int64)
    frames = blob["frame"].astype(np.int64)
    trajs: dict[tuple[str, int], np.ndarray] = {}
    for cid in np.unique(cids):
        cmask = cids == cid
        for enc in np.unique(encs[cmask]):
            sel = np.where(cmask & (encs == enc))[0]
            order = np.argsort(frames[sel])
            f = frames[sel][order]
            if not np.all(np.diff(f) == 1):
                raise ValueError(f"non-contiguous frames for ({cid}, {enc}) in {path}")
            trajs[(cid, int(enc))] = z[sel[order]]
    return trajs


# --------------------------------------------------------------------------- #
# Per-family whitening transforms (fit on TRAIN frames, pooled over encounters)
# --------------------------------------------------------------------------- #
def fit_whiteners(family: str) -> dict[str, dict[str, np.ndarray]]:
    """Fit per-dim std (stdz) and full-covariance (maha) whiteners on TRAIN.

    Both are estimated from the pooled (encounter x frame) TRAIN latent matrix
    (the family's own train per-dim std and train covariance), exactly as in
    topology_ce2.fit_whiteners: the point is to remove the family-specific
    anisotropic per-coordinate scale before comparing topology.
    """
    trajs = load_trajectories(family, "train")
    z = np.concatenate([t for t in trajs.values()], axis=0)
    mean = z.mean(axis=0)
    std = z.std(axis=0)
    std_safe = np.where(std > 1e-12, std, 1.0)

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
# Persistence and generator counting (verbatim ce2 logic)
# --------------------------------------------------------------------------- #
def persistence_counts(cloud: np.ndarray, floors: Iterable[float]) -> dict[float, dict[str, int]]:
    """Vietoris-Rips H0/H1 persistent-generator counts at each floor fraction.

    The noise floor is a fraction of the point cloud diameter (max pairwise
    distance). H0: the always-present infinite-lifetime component plus every
    finite-death component whose lifetime is at least floor * diameter. H1:
    every loop whose lifetime is at least floor * diameter (a clean cycle
    gives exactly 1; fragmentation gives 0 or many short-lived loops).
    """
    from ripser import ripser as ripser_fn
    from scipy.spatial.distance import pdist

    diam = float(pdist(cloud).max())
    if diam <= 0.0:
        return {float(f): {"H0": 1, "H1": 0} for f in floors}
    res = ripser_fn(cloud, maxdim=1)
    dgm0 = res["dgms"][0]
    dgm1 = res["dgms"][1]

    fin0 = dgm0[np.isfinite(dgm0[:, 1])]
    n_inf0 = int((~np.isfinite(dgm0[:, 1])).sum())
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


def estimate_period_frames(traj: np.ndarray, anchor: int = SHEDDING_PERIOD_FRAMES_ANCHOR) -> int:
    """Estimate the limit-cycle period (frames) from the PC1 autocorrelation peak."""
    x = traj - traj.mean(axis=0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    pc1 = x @ vt[0]
    pc1 = pc1 - pc1.mean()
    n = pc1.size
    ac = np.correlate(pc1, pc1, mode="full")[n - 1:]
    ac = ac / (ac[0] + 1e-12)
    lo, hi = max(8, anchor // 2), min(n - 1, int(anchor * 1.6))
    if hi <= lo + 1:
        return int(anchor)
    window = ac[lo:hi]
    peak = lo + int(np.argmax(window))
    if 0 < peak < n - 1 and ac[peak] > ac[peak - 1] and ac[peak] >= ac[peak + 1]:
        return int(peak)
    return int(anchor)


def segment_periods(traj: np.ndarray, period: int) -> list[np.ndarray]:
    """Split a (T, d) trajectory into consecutive single-period (period, d) windows."""
    n = traj.shape[0]
    if period < 8 or period > n:
        return [traj]
    segs = [traj[s: s + period] for s in range(0, n - period + 1, period)]
    return [s for s in segs if s.shape[0] >= 8]


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def analyse_family(family: str, metrics: list[str], floors: tuple[float, ...]) -> dict:
    """Compute per-encounter generator counts for one family across metrics."""
    whit = fit_whiteners(family)

    rows = []
    for split in GUSTED_SPLITS:
        trajs = load_trajectories(family, split)
        for (cid, enc), cloud in sorted(trajs.items()):
            for metric in metrics:
                tc = apply_metric(cloud, metric, whit)
                counts = persistence_counts(tc, floors)
                for floor, c in counts.items():
                    rows.append(
                        {
                            "split": split,
                            "case_id": cid,
                            "encounter_index": enc,
                            "metric": metric,
                            "floor": floor,
                            "H0": c["H0"],
                            "H1": c["H1"],
                        }
                    )

    # No-gust control: the Baseline encounters live in the TRAIN split.
    train = load_trajectories(family, "train")
    base_keys = sorted(k for k in train if k[0] == BASELINE_CASE)
    base_rows = []
    base_period_rows = []
    period = estimate_period_frames(train[base_keys[0]]) if base_keys else 0
    for cid, enc in base_keys:
        cloud = train[(cid, enc)]
        for metric in metrics:
            tc = apply_metric(cloud, metric, whit)
            counts = persistence_counts(tc, floors)
            for floor, c in counts.items():
                base_rows.append(
                    {
                        "case_id": cid,
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
                            "case_id": cid,
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
        "n_baseline": len(base_keys),
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
    """GE2-style branch from the fair recon-vs-predictive H1 contrast (ce2 logic).

    Identical branch logic and thresholds to topology_ce2.decide_gate, with the
    reconstructive comparator swapped from fukami (v2.1 d = 64) to regAE_pool
    (the pooled v2.2 tier's reconstructive family). All quantities at the
    canonical 5 percent floor under the headline whitened metric: single-cycle
    fractions on the gusted encounters (test_b only here) and on the
    single-period no-gust control (the clean-limit-cycle null).
    """
    rec = per_family[RECON_FAMILY]
    jep = per_family.get(PREDICTIVE_FAMILY)

    g_rec_raw = _frac_single(rec["gusted"], "raw", CANONICAL_FLOOR)
    g_rec_wht = _frac_single(rec["gusted"], headline_metric, CANONICAL_FLOOR)
    g_jep_wht = (
        _frac_single(jep["gusted"], headline_metric, CANONICAL_FLOOR)
        if jep is not None
        else float("nan")
    )

    b_rec_wht = _frac_single(rec["baseline_period"], headline_metric, CANONICAL_FLOOR)
    b_jep_wht = (
        _frac_single(jep["baseline_period"], headline_metric, CANONICAL_FLOOR)
        if jep is not None
        else float("nan")
    )

    gusted_gap = g_jep_wht - g_rec_wht
    baseline_gap = b_jep_wht - b_rec_wht

    gusted_healed = np.isfinite(gusted_gap) and gusted_gap <= 0.1
    nogust_separates = np.isfinite(baseline_gap) and baseline_gap > 0.2

    if gusted_healed and nogust_separates:
        branch = "WEAK-MIXED"
        reason = (
            "split verdict. On the WHOLE gusted encounters per-family "
            f"standardisation closes/reverses the gap (gap = {gusted_gap:+.2f}), "
            "so a gusted fragmentation claim would be an anisotropic-scale "
            "artefact. BUT the single-period no-gust limit-cycle control DOES "
            f"separate the families (gap = {baseline_gap:+.2f}: the "
            "reconstructive encoding fragments the clean shedding cycle while "
            "the predictive encoding keeps it) and this survives whitening; the "
            "clean-cycle topology statement holds, scoped to the limit cycle"
        )
    elif gusted_healed:
        branch = "WEAK"
        reason = (
            "per-family standardisation closes the recon-vs-predictive gusted "
            f"gap (gap = {gusted_gap:+.2f} <= 0.10 single-cycle fraction) and "
            "the single-period no-gust control does not separate the families "
            f"(gap = {baseline_gap:+.2f}); no topology claim survives whitening"
        )
    elif nogust_separates:
        branch = "STRONG"
        reason = (
            "after whitening the predictive family keeps the single clean "
            "generator more often than the reconstructive family on gusted "
            f"encounters (gap = {gusted_gap:+.2f}) AND on the single-period "
            f"no-gust limit cycle (gap = {baseline_gap:+.2f}); fragmentation "
            "is a property of the reconstructive ENCODING that survives "
            "whitening"
        )
    else:
        branch = "STRONG-PARTIAL"
        reason = (
            f"the gusted gap survives whitening (gap = {gusted_gap:+.2f}) but "
            "the single-period no-gust control does not separate the families "
            f"(gap = {baseline_gap:+.2f}); report the surviving gusted "
            "fragmentation and qualify the no-gust line"
        )

    return {
        "branch": branch,
        "reason": reason,
        "headline_metric": headline_metric,
        "predictive_family": PREDICTIVE_FAMILY,
        "recon_family": RECON_FAMILY,
        "recon_gusted_frac_single_raw": g_rec_raw,
        "recon_gusted_frac_single_whitened": g_rec_wht,
        "predictive_gusted_frac_single_whitened": g_jep_wht,
        "gusted_gap_predictive_minus_recon": float(gusted_gap),
        "recon_baseline_period_frac_single_whitened": b_rec_wht,
        "predictive_baseline_period_frac_single_whitened": b_jep_wht,
        "baseline_period_gap_predictive_minus_recon": float(baseline_gap),
    }


def build_summary(per_family: dict, metrics: list[str]) -> dict:
    """Per-family medians and single-cycle fractions at the canonical floor."""
    summary: dict[str, dict] = {}
    for fam, res in per_family.items():
        fam_sum: dict[str, dict] = {}
        for metric in metrics:
            g = _h1_at(res["gusted"], metric, CANONICAL_FLOOR)
            bf = _h1_at(res["baseline"], metric, CANONICAL_FLOOR)
            bp = _h1_at(res["baseline_period"], metric, CANONICAL_FLOOR)
            fam_sum[metric] = {
                "gusted_h1_median": float(np.median(g)) if g.size else float("nan"),
                "gusted_frac_single": _frac_single(res["gusted"], metric, CANONICAL_FLOOR),
                "baseline_full_h1_median": float(np.median(bf)) if bf.size else float("nan"),
                "baseline_period_h1_median": float(np.median(bp)) if bp.size else float("nan"),
                "baseline_period_frac_single": _frac_single(
                    res["baseline_period"], metric, CANONICAL_FLOOR
                ),
            }
        summary[fam] = {
            "period_frames": res["period_frames"],
            "n_baseline_encounters": res["n_baseline"],
            "by_metric": fam_sum,
        }
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--families",
        nargs="+",
        default=list(MODEL_CACHES),
        choices=list(MODEL_CACHES),
        help="Families to analyse (default: the four pooled d = 32 topology families).",
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
        help="Whitened metric used for the gate decision (default: stdz).",
    )
    p.add_argument("--output", default=str(OUT_PATH_DEFAULT), help="Output JSON path.")
    args = p.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import ripser  # noqa: F401
    except ImportError:
        out_path.write_text(
            json.dumps({"status": "blocked", "reason": "ripser not installed"}, indent=2) + "\n"
        )
        print(f"[topology_v2p2] BLOCKED: ripser not installed; wrote {out_path}")
        return

    metrics = ["raw", "stdz"] if args.no_maha else ["raw", "stdz", "maha"]
    if args.headline_metric not in metrics:
        metrics.append(args.headline_metric)

    t0 = time.time()
    per_family: dict[str, dict] = {}
    for fam in args.families:
        f0 = time.time()
        per_family[fam] = analyse_family(fam, metrics, FLOOR_GRID)
        n_g = len(
            {(r["split"], r["case_id"], r["encounter_index"]) for r in per_family[fam]["gusted"]}
        )
        print(
            f"[topology_v2p2] {fam}: {n_g} gusted encounters, "
            f"{per_family[fam]['n_baseline']} Baseline encs "
            f"(period {per_family[fam]['period_frames']} frames), "
            f"{time.time() - f0:.1f}s"
        )

    gate = decide_gate(per_family, args.headline_metric)
    summary = build_summary(per_family, metrics)

    out = {
        "status": "ok",
        "params": {
            "script": "scripts/session33/topology_v2p2.py",
            "spec": "SESSION_33_MANUSCRIPT_V3.md Section 11 item 6 (appendix per HANDOFF D223)",
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "method": (
                "Vietoris-Rips persistence (ripser, maxdim=1) on per-encounter "
                "latent trajectory point clouds; generator counts at diameter "
                "noise floors; per-family whiteners fit on pooled TRAIN frames "
                "(topology_ce2.py method, unchanged)"
            ),
            "metrics": metrics,
            "headline_metric": args.headline_metric,
            "canonical_floor": CANONICAL_FLOOR,
            "floor_grid": list(FLOOR_GRID),
            "gusted_splits": list(GUSTED_SPLITS),
            "gusted_split_note": (
                "test_b only: the pooled v2.2 caches carry no test_c split "
                "(v2.1 topology_ce2 used test_b + test_c)"
            ),
            "baseline_case": BASELINE_CASE,
            "model_caches": {f: MODEL_CACHES[f] for f in args.families},
        },
        "summary": summary,
        "gate": gate,
        "per_family": per_family,
    }
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[topology_v2p2] wrote {out_path}")

    # ---- console summary ----------------------------------------------------
    hm = args.headline_metric
    print(f"\n=== H1 at the {int(CANONICAL_FLOOR * 100)}% floor "
          f"(median count / frac single-cycle) ===")
    print(
        f"{'family':22s} {'gusted raw':>12s} {'gusted ' + hm:>12s} "
        f"{'base-full raw':>14s} {'base-per raw':>13s} {'base-per ' + hm:>13s}"
    )
    for fam in args.families:
        s = summary[fam]["by_metric"]
        print(
            f"{fam:22s} "
            f"{s['raw']['gusted_h1_median']:4.1f}/{s['raw']['gusted_frac_single']:.2f}   "
            f"{s[hm]['gusted_h1_median']:4.1f}/{s[hm]['gusted_frac_single']:.2f}   "
            f"{s['raw']['baseline_full_h1_median']:6.1f}        "
            f"{s['raw']['baseline_period_h1_median']:4.1f}/"
            f"{s['raw']['baseline_period_frac_single']:.2f}    "
            f"{s[hm]['baseline_period_h1_median']:4.1f}/"
            f"{s[hm]['baseline_period_frac_single']:.2f}"
        )
    print(f"\n[topology_v2p2] gate branch = {gate['branch']}: {gate['reason']}")
    print(f"[topology_v2p2] total wall time {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
