"""Track C observable closure: each cell's OWN heads scored on test_b (C4).

For every (cell, seed) run, loads the full CanonicalModel checkpoint, applies
its trained heads to the C1-cached per-frame pooled latents (``z_gap`` -- the
exact tensor the heads see in training, since GAP of the broadcast pooled
latent is the pooled latent), and scores against the training targets:

- lift head: raw current-frame C_L (kit convention, cl_future_deltas=(0,)),
- wake head: standardized ``patch_signed_spectrum`` from the wake cache,
- nearbody head: standardized ``nearbody_lift_element`` from the nearbody cache.

Reports per-head R^2 pooled over all test_b frames (mean over output dims for
the 80-D heads) plus per-encounter mean squared error for paired analyses.

CPU-only. Output: ``outputs/session34/trackc_head_closure.json``.

Run (after the C1 latent sweep):
    taskset -c 0-15 python -m scripts.session34.trackc_head_closure
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.session34.trackc_cells import CELLS, CHECKPOINT, RUNS_BASE  # noqa: E402
from scripts.session34.trackc_lift_eval import group_encounters, load_cache  # noqa: E402
from src.config.kit_config import load_model_config  # noqa: E402
from src.data.wake_observables import WakeObservableStats  # noqa: E402
from src.training.canonical_model import CanonicalModel  # noqa: E402

PREVENT = Path(os.environ.get("PREVENT_ROOT", "/home/carlos/PREVENT"))
CACHE = Path(os.environ.get("VORTEX_JEPA_CACHE", PREVENT / "data" / "processed" / "vortex-jepa"))


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def load_full_model(run_dir: Path) -> CanonicalModel:
    blob = torch.load(run_dir / CHECKPOINT, map_location="cpu", weights_only=False)
    args = blob.get("args", {})
    run_config = blob.get("run_config", {})
    config_path = args.get("config") or run_config.get("config_path")
    cfg = load_model_config(_resolve(config_path))
    model = CanonicalModel(
        cfg,
        latent_dim=int(args.get("d", 32)),
        projection_norm=args.get("projection_norm", "batchnorm"),
        predictor_class=args.get("predictor_class", "resunet"),
    )
    model.load_state_dict(blob["model_state_dict"], strict=True)
    model.eval()
    return model


class ObsStatsCache:
    """Lazy loader for the standardized observable targets of one family."""

    def __init__(self, root: Path, mode: str) -> None:
        self.root = root
        self.mode = mode
        with open(root / "_train_stats.json") as f:
            self.stats = WakeObservableStats.from_dict(json.load(f)[mode])
        self._enc: dict[tuple, np.ndarray] = {}

    def target(self, case_id: str, k: int, frames: np.ndarray) -> np.ndarray:
        key = (case_id, int(k))
        if key not in self._enc:
            with h5py.File(self.root / case_id / f"encounter_{k:02d}.h5", "r") as g:
                arr = g[self.mode][...].astype(np.float32)
            std = self.stats.standardize(torch.from_numpy(arr)).numpy()
            self._enc[key] = std
        return self._enc[key][frames]


def r2_per_dim_mean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean over output dims of per-dim R^2 (pooled over rows)."""
    y_true = np.atleast_2d(y_true.T).T.astype(np.float64)
    y_pred = np.atleast_2d(y_pred.T).T.astype(np.float64)
    sse = ((y_true - y_pred) ** 2).sum(axis=0)
    sst = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - sse / np.maximum(sst, 1e-12)
    return float(r2.mean())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Track C head closure")
    ap.add_argument("--cache-dir", default="outputs/session34/trackc_latents")
    ap.add_argument("--out", default="outputs/session34/trackc_head_closure.json")
    ap.add_argument("--cells", nargs="+", default=list(CELLS))
    ap.add_argument("--partition", default="v2p2")
    args = ap.parse_args(argv)

    cache_dir = REPO_ROOT / args.cache_dir
    wake_stats = ObsStatsCache(CACHE / args.partition / "wake_observables",
                               "patch_signed_spectrum")
    nb_stats = ObsStatsCache(CACHE / args.partition / "nearbody_observables",
                             "nearbody_lift_element")

    results: dict[str, dict] = {}
    t0 = time.time()
    for cell in args.cells:
        results[cell] = {}
        for seed, run_name in sorted(CELLS[cell].items()):
            run_dir = RUNS_BASE / run_name
            model = load_full_model(run_dir)
            tb = load_cache(cache_dir, run_name, "test_b")
            z = torch.from_numpy(tb["z_gap"]).float()
            encs = group_encounters(tb)
            rec: dict = {"run_name": run_name, "heads": {}}
            with torch.no_grad():
                head_preds = {}
                if model.lift_head is not None:
                    head_preds["lift"] = model.lift_head(z).squeeze(-1).numpy()
                if model.wake_head is not None:
                    head_preds["wake"] = model.wake_head(z).numpy()
                if model.nearbody_head is not None:
                    head_preds["nearbody"] = model.nearbody_head(z).numpy()
            for head, pred in head_preds.items():
                if head == "lift":
                    true = tb["cl"].astype(np.float64)
                else:
                    stats = wake_stats if head == "wake" else nb_stats
                    true = np.concatenate([
                        stats.target(e["case_id"], e["encounter_index"],
                                     tb["frame"][e["rows"]])
                        for e in encs
                    ])
                    pred = np.concatenate([pred[e["rows"]] for e in encs])
                    # rows re-ordered per-encounter above; reorder truth-aligned copy
                if head == "lift":
                    per_enc = [
                        {
                            "case_id": e["case_id"],
                            "encounter_index": e["encounter_index"],
                            "mse": float(np.mean(
                                (true[e["rows"]] - pred[e["rows"]]) ** 2)),
                        }
                        for e in encs
                    ]
                    r2 = r2_per_dim_mean(true, pred)
                else:
                    r2 = r2_per_dim_mean(true, pred)
                    per_enc = None
                rec["heads"][head] = {"r2": r2, "per_encounter": per_enc}
                print(f"[closure] {cell} s{seed} {head}: R2={r2:+.3f}", flush=True)
            results[cell][f"s{seed}"] = rec
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "protocol": {
            "latents": str(cache_dir),
            "lift_target": "raw current-frame C_L",
            "wake_target": "standardized patch_signed_spectrum",
            "nearbody_target": "standardized nearbody_lift_element",
            "metric": "mean over dims of pooled per-dim R2 on test_b",
        },
        "results": results,
    }, indent=1))
    print(f"[closure] wrote {out_path} in {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
