#!/usr/bin/env python3
"""
session25_c0_robustness.py
==========================

Robustness annex for the Track C0 de-risking gate (Session 25). The pre-registered
command in scripts/run_causal_analysis.py runs SURD on ONE impact-frame sample per
training encounter (n ~ 237) over a 6x6x6 lattice (216 cells). The infotheory
README flags that regime as data-starved and synergy-biased, and recommends
estimating on the pooled training trajectories (~237 encounters x ~120 frames).

This script triangulates the gate's headline criterion (is the future WAKE more
synergistic in (G, current wake state) than the future LIFT?) across:

  designs : per-encounter impact frame (the pre-registered command),
            per-frame pooled over the full trajectory,
            per-frame pooled over the post-impact window [impact, T-H);
  bins    : {4, 5, 6, 8} quantile bins (the SURD robustness knob).

It reuses the exact data path and partition map wired into infotheory.io_vortex,
so the per-encounter leg reproduces the run_causal_analysis.py numbers. SURD on a
fixed pmf is deterministic; the only stochastic ingredient (surrogate nulls) is not
used for the discrete decomposition, so no RNG is threaded here beyond the fixed
quantile edges.

Honesty: SURD/IND are observational (HANDOFF D154). Per-frame pooling violates
sample independence (frames within an encounter are autocorrelated), so the pooled
legs over-count effective samples; the per-encounter leg under-counts. The truth is
triangulated by agreement across designs, not by any single number.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infotheory import io_vortex
from infotheory.estimators import quantise, joint_pmf
from infotheory.surd import surd_discrete

HORIZON = 16


def load_train_series(cache_root: Path, split: dict, partition: str = "v1"):
    """Return per-train-encounter (G, enstrophy[t], CL[t], impact) full series."""
    ep_root = io_vortex._episode_path(cache_root, partition)
    wo_root = io_vortex._wake_observables_path(cache_root, partition)
    part_of = io_vortex._build_partition_map(split)
    out = []
    for (case_id, enc), plabel in sorted(part_of.items()):
        if plabel != "train":
            continue
        ep_f = io_vortex._encounter_file(ep_root, case_id, enc)
        wo_f = io_vortex._encounter_file(wo_root, case_id, enc)
        if not ep_f.exists() or not wo_f.exists():
            continue
        with h5py.File(ep_f, "r") as g:
            cl = np.asarray(g["C_L"], float).ravel()
            G = float(g.attrs["G"])
            impact = int(g.attrs.get("impact_frame_estimate", 40))
        with h5py.File(wo_f, "r") as g:
            ens = np.asarray(g["enstrophy_scalar"], float).ravel()
        n = min(len(cl), len(ens))
        out.append({"G": G, "ens": ens[:n], "cl": cl[:n], "impact": impact, "n": n})
    return out


def build_design(series, mode: str):
    """Build (G, wake_now) sources and (wake_future, cl_future) targets per mode."""
    Gs, wake_now, wake_fut, cl_fut = [], [], [], []
    for s in series:
        n, imp = s["n"], s["impact"]
        if mode == "per_encounter":
            ts = [imp] if imp + HORIZON < n else []
        elif mode == "per_frame_full":
            ts = range(0, n - HORIZON)
        elif mode == "per_frame_postimpact":
            ts = range(imp, n - HORIZON)
        else:
            raise ValueError(mode)
        for t in ts:
            Gs.append(s["G"])
            wake_now.append(s["ens"][t])
            wake_fut.append(s["ens"][t + HORIZON])
            cl_fut.append(s["cl"][t + HORIZON])
    return (np.asarray(Gs), np.asarray(wake_now),
            np.asarray(wake_fut), np.asarray(cl_fut))


def surd_fracs(G, wake_now, target, nb):
    """SURD of target from (G, wake_now); return key normalised fractions."""
    s1 = quantise(G, nb)
    s2 = quantise(wake_now, nb)
    t = quantise(target, nb)
    p = joint_pmf(np.column_stack([t, s1, s2]), (nb, nb, nb))
    res = surd_discrete(p)
    norm = res.normalised()
    return {
        "S": norm.get("S(1, 2)", 0.0),
        "U_G": norm.get("U(1,)", 0.0),
        "U_wake": norm.get("U(2,)", 0.0),
        "R": norm.get("R(1, 2)", 0.0),
        "leak": res.info_leak,
        "H": res.h_target,
        "MI": res.mi_total,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="configs/splits/split_v1.json")
    ap.add_argument("--partition", default="v1")
    ap.add_argument("--bins", type=int, nargs="+", default=[4, 5, 6, 8])
    args = ap.parse_args()

    cache = io_vortex.resolve_cache_root()
    split = io_vortex.load_split(args.split)
    series = load_train_series(cache, split, args.partition)
    print(f"[data] {len(series)} train encounters loaded\n")

    designs = ["per_encounter", "per_frame_full", "per_frame_postimpact"]
    hdr = (f"{'design':<22}{'bins':>5}{'n':>8}"
           f"{'S_wake/H':>10}{'S_lift/H':>10}{'ratio':>8}"
           f"{'Uwake_G':>9}{'Ulift_G':>9}{'leak_w':>8}{'leak_l':>8}")
    print(hdr)
    print("-" * len(hdr))
    summary = {}
    for mode in designs:
        G, wake_now, wake_fut, cl_fut = build_design(series, mode)
        n = len(G)
        for nb in args.bins:
            w = surd_fracs(G, wake_now, wake_fut, nb)
            l = surd_fracs(G, wake_now, cl_fut, nb)
            ratio = w["S"] / l["S"] if l["S"] > 1e-9 else float("inf")
            summary[(mode, nb)] = (w, l, ratio)
            print(f"{mode:<22}{nb:>5}{n:>8}"
                  f"{w['S']:>10.3f}{l['S']:>10.3f}{ratio:>8.2f}"
                  f"{w['U_G']:>9.3f}{l['U_G']:>9.3f}{w['leak']:>8.3f}{l['leak']:>8.3f}")
        print()

    # Gate readout, per design (median ratio over the bins sweep).
    print("=" * 72)
    print("GATE READOUT (criterion 1: S_wake/H >= ~2 x S_lift/H)")
    print("=" * 72)
    for mode in designs:
        ratios = [summary[(mode, nb)][2] for nb in args.bins]
        lift_unique_gt_syn = all(summary[(mode, nb)][1]["U_G"] > summary[(mode, nb)][1]["S"]
                                 for nb in args.bins)
        leak_ok = all(summary[(mode, nb)][0]["leak"] < 0.7 for nb in args.bins)
        med = float(np.median(ratios))
        print(f"  {mode:<22} median ratio={med:.2f}  range=[{min(ratios):.2f},{max(ratios):.2f}]  "
              f"lift(U[G]>S) all-bins={lift_unique_gt_syn}  leak<0.7 all-bins={leak_ok}")


if __name__ == "__main__":
    main()
