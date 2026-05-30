"""Quality-control scan of every raw DNS case in periodic/ and run3/.

For each case we read the spanwise vorticity omega_z = curlU[..., 2] over the full
(T, 192, 96, 32) field and report, on the FLUID cells only (inside_solid == 0):

  - nan_in_fluid / inf_in_fluid : non-finite values where the flow is defined.
        NaN/Inf is expected only inside the solid; any in the fluid is a real
        numerical fault and the case should be repeated.
  - max_fluid                   : max |omega_z| over all fluid cells. A high value
        here can be the benign leading-edge finite-difference artifact (the cells
        one layer off the airfoil), so it is not by itself an anomaly.
  - max_off_airfoil             : max |omega_z| EXCLUDING the one-cell layer
        adjacent to the solid. A high value here is away from the LE artifact and
        indicates a genuine vorticity blow-up.
  - p99_9_off_airfoil           : 99.9th percentile of |omega_z| off the airfoil
        (the physical tail, for context).
  - n_gt_{5k,10k,20k}_off       : counts of off-airfoil cells above thresholds.
  - frame_of_max                : the frame index where max_fluid occurs.

Anomaly / repeat logic (a case is flagged ``repeat`` if any holds):
  - nan_in_fluid > 0 or inf_in_fluid > 0           (numerical fault)
  - max_off_airfoil > HARD_MAX (default 10000)     (blow-up beyond the
        cache-integrity bound of CLAUDE.md)
  - max_off_airfoil is a population outlier         (> median + N*MAD across all
        cases, computed after the scan; catches milder anomalies relative to the
        rest of the dataset).

Output: data_manifest/raw_cases_qc.csv (one row per case, sorted worst-first) and
data_manifest/raw_cases_qc.yaml (machine-readable, with the population thresholds
and the repeat list). The auto-generated raw_cases_inventory.yaml is NOT edited;
this QC report sits alongside it.

Usage:
  export PREVENT_ROOT=$HOME/PREVENT
  python scripts/qc_raw_vorticity.py            # scan all
  python scripts/qc_raw_vorticity.py --limit 3  # quick smoke on 3 files
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import time
from pathlib import Path

import h5py
import numpy as np
import yaml
from scipy.ndimage import binary_dilation

REPO = Path(__file__).resolve().parents[1]
PREVENT = Path(os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT")))
RAW = PREVENT / "data" / "raw" / "periodic"
OUT_CSV = REPO / "data_manifest" / "raw_cases_qc.csv"
OUT_YAML = REPO / "data_manifest" / "raw_cases_qc.yaml"

HARD_MAX = 10000.0          # cache-integrity upper bound (CLAUDE.md)
OUTLIER_N_MAD = 6.0         # population-relative anomaly: median + N*MAD
CHUNK = 80                  # frames per read
OMEGA_Z_INDEX = 2           # curlU[..., 2] is omega_z (du/dy - dv/dx)


def load_inventory_meta() -> dict:
    """filename(stem) -> {case_id, G, D, Y} from the parser manifest, for context."""
    inv = REPO / "data_manifest" / "raw_cases_inventory.yaml"
    meta = {}
    if not inv.exists():
        return meta
    with open(inv) as f:
        data = yaml.safe_load(f)
    for c in data.get("cases", []):
        stem = Path(c.get("filename", "")).stem
        meta[stem] = {"case_id": c.get("case_id", ""),
                      "G": c.get("G", ""), "D": c.get("D", ""), "Y": c.get("Y", "")}
    return meta


def list_cases() -> list[dict]:
    meta = load_inventory_meta()
    cases = []
    for f in sorted(glob.glob(str(RAW / "*.h5"))):
        stem = Path(f).stem
        cases.append({"name": stem, "source": "periodic", "path": f, **meta.get(stem, {})})
    for f in sorted(glob.glob(str(RAW / "run3" / "*.h5"))):
        stem = Path(f).stem
        cases.append({"name": stem, "source": "run3", "path": f, **meta.get(stem, {})})
    return cases


def scan_case(path: str) -> dict:
    with h5py.File(path, "r") as h:
        solid = np.asarray(h["inside_solid"]).squeeze(-1).astype(bool)  # (192,96,32)
        fluid = ~solid
        # one-cell layer adjacent to the solid (where the LE finite-difference
        # artifact lives); "off-airfoil" excludes solid AND this adjacent layer.
        adjacent = binary_dilation(solid, iterations=1) & fluid
        off_airfoil = fluid & ~adjacent

        cu = h["curlU"]
        T = int(cu.shape[0])
        max_fluid = 0.0
        max_off = 0.0
        frame_of_max = -1
        nan_fluid = 0
        inf_fluid = 0
        n_gt = {5000: 0, 10000: 0, 20000: 0}
        per_frame_max = np.zeros(T, dtype=np.float64)
        off_sample = []  # subsample of |omega| off-airfoil for the percentile

        for t0 in range(0, T, CHUNK):
            oz = cu[t0:t0 + CHUNK, :, :, :, OMEGA_Z_INDEX]  # (c,192,96,32)
            c = oz.shape[0]
            ozf = oz[:, fluid]                              # (c, n_fluid)
            nan_fluid += int(np.isnan(ozf).sum())
            inf_fluid += int(np.isinf(ozf).sum())
            af = np.abs(ozf)
            af_finite = np.where(np.isfinite(af), af, 0.0)
            # per-frame fluid max
            pf = af_finite.max(axis=1)
            per_frame_max[t0:t0 + c] = pf
            cmax = float(pf.max())
            if cmax > max_fluid:
                max_fluid = cmax
                frame_of_max = int(t0 + int(pf.argmax()))
            # off-airfoil
            ozo = oz[:, off_airfoil]
            ao = np.abs(ozo)
            ao = ao[np.isfinite(ao)]
            if ao.size:
                max_off = max(max_off, float(ao.max()))
                for thr in n_gt:
                    n_gt[thr] += int((ao > thr).sum())
                # subsample every 200th for percentile context
                off_sample.append(ao[::200])

        if off_sample:
            off_cat = np.concatenate(off_sample)
            p99_9 = float(np.percentile(off_cat, 99.9))
            p99_99 = float(np.percentile(off_cat, 99.99))
            mean_off = float(off_cat.mean())
        else:
            p99_9 = p99_99 = mean_off = float("nan")

        return {
            "n_frames": T,
            "fluid_cells": int(fluid.sum()),
            "nan_in_fluid": nan_fluid,
            "inf_in_fluid": inf_fluid,
            "max_fluid": round(max_fluid, 1),
            "max_off_airfoil": round(max_off, 1),
            "frame_of_max": frame_of_max,
            "p99_9_off_airfoil": round(p99_9, 1),
            "p99_99_off_airfoil": round(p99_99, 1),
            "mean_off_airfoil": round(mean_off, 2),
            "n_gt_5k_off": n_gt[5000],
            "n_gt_10k_off": n_gt[10000],
            "n_gt_20k_off": n_gt[20000],
            "n_frames_off_gt_3k": int((per_frame_max > 3000).sum()),
        }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="scan only the first N cases")
    args = ap.parse_args()

    cases = list_cases()
    if args.limit:
        cases = cases[: args.limit]
    print(f"[qc] scanning {len(cases)} cases ({sum(c['source']=='periodic' for c in cases)} "
          f"periodic + {sum(c['source']=='run3' for c in cases)} run3)", flush=True)

    rows = []
    for i, c in enumerate(cases):
        t0 = time.time()
        try:
            stats = scan_case(c["path"])
            err = ""
        except Exception as exc:  # a case that cannot even be opened is itself a fault
            stats = {"n_frames": -1, "nan_in_fluid": -1, "max_off_airfoil": float("nan")}
            err = f"{type(exc).__name__}: {exc}"
        row = {"case": c["name"], "source": c["source"],
               "case_id": c.get("case_id", ""), "G": c.get("G", ""),
               "D": c.get("D", ""), "Y": c.get("Y", ""),
               **stats, "read_error": err}
        rows.append(row)
        print(f"[qc] {i+1}/{len(cases)} {c['source']:8s} {c['name'][:42]:42s} "
              f"max_off={stats.get('max_off_airfoil')} nan={stats.get('nan_in_fluid')} "
              f"({time.time()-t0:.0f}s){' ERR '+err if err else ''}", flush=True)

    # population-relative anomaly threshold on max_off_airfoil
    vals = np.array([r["max_off_airfoil"] for r in rows
                     if isinstance(r.get("max_off_airfoil"), (int, float))
                     and np.isfinite(r["max_off_airfoil"])])
    med = float(np.median(vals)) if vals.size else float("nan")
    mad = float(np.median(np.abs(vals - med))) if vals.size else float("nan")
    outlier_thr = med + OUTLIER_N_MAD * mad if np.isfinite(mad) else float("inf")
    print(f"\n[qc] population max_off_airfoil: median={med:.0f} MAD={mad:.0f} "
          f"outlier_thr(median+{OUTLIER_N_MAD}*MAD)={outlier_thr:.0f}", flush=True)

    for r in rows:
        reasons = []
        if r.get("read_error"):
            reasons.append("read_error")
        if (r.get("nan_in_fluid") or 0) > 0:
            reasons.append("nan_in_fluid")
        if (r.get("inf_in_fluid") or 0) > 0:
            reasons.append("inf_in_fluid")
        mo = r.get("max_off_airfoil")
        if isinstance(mo, (int, float)) and np.isfinite(mo):
            if mo > HARD_MAX:
                reasons.append(f"max_off>{int(HARD_MAX)}")
            elif mo > outlier_thr:
                reasons.append("population_outlier")
        r["flags"] = ";".join(reasons)
        r["repeat"] = bool(reasons)

    # sort worst-first: repeats by max_off desc, then the rest by max_off desc
    rows.sort(key=lambda r: (not r["repeat"],
                             -(r["max_off_airfoil"] if isinstance(r.get("max_off_airfoil"), (int, float))
                               and np.isfinite(r["max_off_airfoil"]) else -1)))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = ["case", "source", "case_id", "G", "D", "Y", "repeat", "flags",
              "max_off_airfoil", "max_fluid", "nan_in_fluid", "inf_in_fluid",
              "frame_of_max", "p99_9_off_airfoil", "p99_99_off_airfoil",
              "mean_off_airfoil", "n_gt_5k_off", "n_gt_10k_off", "n_gt_20k_off",
              "n_frames_off_gt_3k", "n_frames", "fluid_cells", "read_error"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    repeats = [r["case"] for r in rows if r["repeat"]]
    summary = {
        "qc_version": "raw_vorticity_qc_v1",
        "scanned": len(rows),
        "thresholds": {"hard_max_off_airfoil": HARD_MAX,
                       "population_median_max_off": med,
                       "population_mad_max_off": mad,
                       "outlier_threshold": outlier_thr,
                       "outlier_n_mad": OUTLIER_N_MAD},
        "n_repeat": len(repeats),
        "repeat_cases": repeats,
        "per_case": {r["case"]: {k: r[k] for k in fields if k in r} for r in rows},
    }
    with open(OUT_YAML, "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False, default_flow_style=False)

    print(f"\n[qc] {len(repeats)}/{len(rows)} cases flagged for repeat: {repeats}")
    print(f"[qc] wrote {OUT_CSV} and {OUT_YAML}")


if __name__ == "__main__":
    main()
