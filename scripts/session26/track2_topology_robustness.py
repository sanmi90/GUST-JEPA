#!/usr/bin/env python3
"""Session 26 Track 2: persistent-homology threshold and sampling robustness.

The manuscript cites the significant-H1 generator count of the simulation-encoded latent as
the decisive topological separation (predictive median 1 clean loop vs reconstructive many
spurious generators, Mann-Whitney p ~ 4e-8). A topology referee from the Smith, Fukami, Sedky,
Jones, Taira (JFM 980, A18, 2024) group will expect the separation to be robust to (i) the noise
floor that declares an H1 generator significant, and (ii) the number of points sampled per
encounter (the convergence protocol of that paper).

The significance rule is floor = NOISE_FRAC * cloud_scale, where cloud_scale is the largest finite
H0 death (the Rips scale at which the 120-frame latent cloud becomes connected). The canonical
analysis uses NOISE_FRAC = 0.05 and all 120 frames. This script sweeps:
  - NOISE_FRAC over {0.02, 0.05, 0.10, 0.15, 0.20};
  - points per encounter over {120, 60, 40, 30} (uniform stride subsampling of the trajectory),
and at each grid point reports the median significant-H1 count for the predictive (JEPA z_dns) and
reconstructive (Fukami z_dns) encodings and the one-sided Mann-Whitney p (JEPA fewer generators),
plus a case-level signed-rank p at the canonical point (the 42 encounters come from 10 cases).

We run ripser once per (encounter, family, stride), store every H1 lifetime and the cloud_scale,
then re-threshold cheaply across NOISE_FRAC. CPU only, no training.

Output: outputs/session26/topology_robustness/{grid.csv, grid.json}.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from ripser import ripser
from scipy.stats import mannwhitneyu, wilcoxon
import warnings

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
ROLLOUTS = REPO / "outputs" / "session18" / "exp_b1_test3"
OUT = REPO / "outputs" / "session26" / "topology_robustness"
OUT.mkdir(parents=True, exist_ok=True)

JEPA_TAG = "jepa_d64_test1_noBN"
FUKAMI_TAG = "fukami_d64_noBN"
KEY = "z_dns"  # simulation-encoded latent (the manuscript's generator-count claim)
NOISE_FRACS = [0.02, 0.05, 0.10, 0.15, 0.20]
STRIDES = [1, 2, 3, 4]  # -> 120, 60, 40, 30 points per encounter
SPLIT = "test_b"


def h1_lifetimes_and_scale(cloud: np.ndarray):
    """All H1 lifetimes and the cloud_scale (max finite H0 death) for one point cloud."""
    res = ripser(cloud.astype(np.float64), maxdim=1)
    h0, h1 = res["dgms"][0], res["dgms"][1]
    deaths = h0[:, 1][np.isfinite(h0[:, 1])]
    scale = float(deaths.max()) if deaths.size else 0.0
    lifetimes = (h1[:, 1] - h1[:, 0]) if h1.size else np.zeros(0)
    return lifetimes, scale


def load_clouds(tag: str):
    blob = np.load(ROLLOUTS / f"rollouts_{tag}" / f"{SPLIT}.npz", allow_pickle=True)
    z = blob[KEY].astype(np.float64)  # (n, 120, d)
    cid = blob["case_ids"] if "case_ids" in blob.files else blob["case_id"]
    return z, np.array([str(c) for c in cid])


def main():
    zj, cj = load_clouds(JEPA_TAG)
    zf, cf = load_clouds(FUKAMI_TAG)
    assert np.array_equal(cj, cf), "JEPA/Fukami encounter order mismatch"
    n_enc, T, _ = zj.shape
    case_ids = cj

    # run ripser once per (encounter, family, stride); store lifetimes + scale
    store = {}  # (stride, family) -> list of (lifetimes, scale)
    for stride in STRIDES:
        idx = np.arange(0, T, stride)
        for fam, z in (("jepa", zj), ("fukami", zf)):
            recs = []
            for i in range(n_enc):
                lt, sc = h1_lifetimes_and_scale(z[i, idx])
                recs.append((lt, sc))
            store[(stride, fam)] = recs

    def nsig_array(stride, fam, frac):
        return np.array([int((lt > frac * sc).sum()) if sc > 0 else 0
                         for lt, sc in store[(stride, fam)]], dtype=float)

    rows = []
    grid = {"noise_fracs": NOISE_FRACS, "strides": STRIDES,
            "points_per_stride": {s: int(len(np.arange(0, T, s))) for s in STRIDES},
            "canonical": {"noise_frac": 0.05, "stride": 1}, "cells": []}
    for stride in STRIDES:
        npts = int(len(np.arange(0, T, stride)))
        for frac in NOISE_FRACS:
            jn = nsig_array(stride, "jepa", frac)
            fn = nsig_array(stride, "fukami", frac)
            try:
                u, p = mannwhitneyu(jn, fn, alternative="less")
                u, p = float(u), float(p)
            except Exception:
                u, p = float("nan"), float("nan")
            # case-level signed-rank on per-case mean counts (same cases)
            uc = sorted(set(case_ids.tolist()))
            jcm = np.array([jn[case_ids == c].mean() for c in uc])
            fcm = np.array([fn[case_ids == c].mean() for c in uc])
            diff = fcm - jcm
            try:
                _, wp = wilcoxon(diff, alternative="greater")
                wp = float(wp)
            except Exception:
                wp = float("nan")
            cell = {"stride": stride, "n_points": npts, "noise_frac": frac,
                    "jepa_median": float(np.median(jn)), "fukami_median": float(np.median(fn)),
                    "jepa_mean": float(jn.mean()), "fukami_mean": float(fn.mean()),
                    "mannwhitney_p": p, "case_wilcoxon_p": wp,
                    "cases_jepa_fewer": int(np.sum(diff > 0))}
            grid["cells"].append(cell)
            rows.append(cell)

    # console
    print(f"{'n_pts':>5} {'frac':>5} {'JEPA med':>8} {'Fuk med':>8} {'MW p':>10} {'case p':>9} {'cases':>6}")
    for r in rows:
        print(f"{r['n_points']:>5} {r['noise_frac']:>5} {r['jepa_median']:>8.1f} "
              f"{r['fukami_median']:>8.1f} {r['mannwhitney_p']:>10.2e} {r['case_wilcoxon_p']:>9.4f} "
              f"{r['cases_jepa_fewer']:>4}/10")
    sep = [r for r in rows if r["jepa_median"] < r["fukami_median"] and r["mannwhitney_p"] < 1e-3]
    grid["summary"] = {
        "n_cells": len(rows),
        "cells_with_separation_and_MW_p_below_1e-3": len(sep),
        "max_MW_p_over_grid": float(max(r["mannwhitney_p"] for r in rows)),
        "all_cells_jepa_median_below_fukami": bool(all(r["jepa_median"] < r["fukami_median"] for r in rows)),
        "max_case_wilcoxon_p_over_grid": float(max(r["case_wilcoxon_p"] for r in rows)),
    }
    print("\nSUMMARY:", json.dumps(grid["summary"], indent=0))

    (OUT / "grid.json").write_text(json.dumps(grid, indent=2))
    with (OUT / "grid.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {OUT}/grid.json and grid.csv")


if __name__ == "__main__":
    main()
