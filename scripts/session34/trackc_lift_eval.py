"""Track C decoded-C_L eval: peak-region R^2, phase lag, three-curve records.

Consumes the per-frame latent caches written by the C1 represent sweep
(``outputs/session34/trackc_latents/latents_<run>_{train,test_b}.npz``) and,
for every (cell, seed):

1. Fits the FROZEN post-hoc probe pair (RidgeCV linear + small MLP, the
   represent.py convention) on train ``z_gap -> C_L``. The probe protocol is
   identical for every cell -- a cell's own lift head is never used here, so
   C0/CW/CN read out through exactly the same capacity as CL/CLW.
2. Predicts per-frame C_L on test_b, groups rows by (case_id, encounter), and
   emits per-encounter records: peak-window SSE/SST (half_width = 8 frames),
   per-encounter peak R^2, phase lag (max_lag 20 frames, dt 0.05), and the
   persistence-floor reference (pre-impact mean over frames [0, 25) held
   constant).
3. Aggregates pooled peak-region R^2 per (cell, seed, probe).

Output: ``outputs/session34/trackc_lift.json``.

Run (CPU, after the C1 latent sweep):
    taskset -c 0-15 python -m scripts.session34.trackc_lift_eval
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.session34.trackc_cells import CELLS, RUNS_BASE  # noqa: E402
from src.evaluation.lift_metrics import (  # noqa: E402
    lift_phase_lag,
    peak_region_r2,
    peak_window_mask,
)
from src.evaluation.represent import fit_linear_probe, fit_mlp_probe  # noqa: E402

PRE_IMPACT_FRAMES = 25  # persistence floor: mean over frames [0, 25) held constant
DT_TC = 0.05
MAX_LAG_FRAMES = 20
HALF_WIDTH = 8


def load_cache(cache_dir: Path, run_name: str, split: str) -> dict:
    path = cache_dir / f"latents_{run_name}_{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"latent cache missing: {path} (run the C1 sweep first)")
    z = np.load(path, allow_pickle=True)
    return {
        "z_gap": z["z_gap"],
        "case_id": z["case_id"],
        "encounter_index": z["encounter_index"],
        "frame": z["frame"],
        "cl": z["target_C_L"],
    }


def group_encounters(cache: dict) -> list[dict]:
    """Split flat rows into per-encounter dicts ordered by frame."""
    keys = list(zip(cache["case_id"].tolist(), cache["encounter_index"].tolist()))
    out: dict[tuple, list[int]] = {}
    for i, key in enumerate(keys):
        out.setdefault(key, []).append(i)
    encs = []
    for (cid, k), idx in out.items():
        idx = np.asarray(idx)
        order = np.argsort(cache["frame"][idx])
        idx = idx[order]
        encs.append({"case_id": cid, "encounter_index": int(k), "rows": idx})
    encs.sort(key=lambda e: (e["case_id"], e["encounter_index"]))
    return encs


def eval_run(cache_dir: Path, run_name: str, seed: int) -> dict:
    tr = load_cache(cache_dir, run_name, "train")
    tb = load_cache(cache_dir, run_name, "test_b")
    probes = {
        "linear": fit_linear_probe(tr["z_gap"], tr["cl"]),
        "mlp": fit_mlp_probe(tr["z_gap"], tr["cl"], seed=seed),
    }
    encs = group_encounters(tb)
    per_probe: dict[str, dict] = {}
    for pname, probe in probes.items():
        pred_all = probe.predict(tb["z_gap"])
        records = []
        truths, preds = [], []
        for enc in encs:
            rows = enc["rows"]
            truth = tb["cl"][rows].astype(np.float64)
            pred = np.asarray(pred_all[rows], dtype=np.float64)
            persist = np.full_like(truth, truth[:PRE_IMPACT_FRAMES].mean())
            m = peak_window_mask(truth, half_width=HALF_WIDTH)
            sse = float(((truth[m] - pred[m]) ** 2).sum())
            sse_persist = float(((truth[m] - persist[m]) ** 2).sum())
            sst = float(((truth[m] - truth[m].mean()) ** 2).sum())
            lag_tc, corr = lift_phase_lag(pred, truth, dt=DT_TC, max_lag=MAX_LAG_FRAMES)
            records.append({
                "case_id": enc["case_id"],
                "encounter_index": enc["encounter_index"],
                "peak_sse": sse,
                "peak_sse_persistence": sse_persist,
                "peak_sst": sst,
                "peak_r2": 1.0 - sse / max(sst, 1e-12),
                "peak_r2_persistence": 1.0 - sse_persist / max(sst, 1e-12),
                "phase_lag_tc": lag_tc,
                "phase_corr": corr,
            })
            truths.append(truth)
            preds.append(pred)
        pooled = peak_region_r2(truths, preds, half_width=HALF_WIDTH)
        lags = np.asarray([r["phase_lag_tc"] for r in records])
        per_probe[pname] = {
            "pooled_peak_r2": pooled,
            "median_phase_lag_tc": float(np.median(lags)),
            "iqr_phase_lag_tc": [float(np.percentile(lags, 25)), float(np.percentile(lags, 75))],
            "n_encounters": len(records),
            "encounters": records,
        }
    return per_probe


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Track C decoded-C_L eval")
    ap.add_argument("--cache-dir", default="outputs/session34/trackc_latents")
    ap.add_argument("--out", default="outputs/session34/trackc_lift.json")
    ap.add_argument("--cells", nargs="+", default=list(CELLS))
    args = ap.parse_args(argv)

    cache_dir = REPO_ROOT / args.cache_dir
    results: dict[str, dict] = {}
    t0 = time.time()
    for cell in args.cells:
        results[cell] = {}
        for seed, run_name in sorted(CELLS[cell].items()):
            print(f"[trackc-lift] {cell} s{seed} ({run_name})", flush=True)
            results[cell][f"s{seed}"] = {
                "run_name": run_name,
                **eval_run(cache_dir, run_name, seed),
            }
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": {
            "probe": "represent.fit_linear_probe + fit_mlp_probe on train z_gap -> C_L",
            "half_width_frames": HALF_WIDTH,
            "max_lag_frames": MAX_LAG_FRAMES,
            "dt_tc": DT_TC,
            "persistence": f"pre-impact mean over frames [0, {PRE_IMPACT_FRAMES})",
            "runs_base": str(RUNS_BASE),
        },
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=1))
    print(f"[trackc-lift] wrote {out_path} in {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
