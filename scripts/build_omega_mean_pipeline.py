"""Build the omega pipeline manifest for partition v1_mean.

Computes per-encounter p99.99 clip thresholds and pool train_stats (mean, std)
for the spanwise-mean omega cache built by scripts/build_omega_mean_cache.py.

Schema matches outputs/data_pipeline/v1/manifest.json so that
OmegaPipeline.from_manifest loads it identically.

Output: outputs/data_pipeline/v1_mean/manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np


REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=str(REPO / "configs/splits/split_v1.json"))
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--out-dir", default=str(REPO / "outputs/data_pipeline/v1_mean"))
    args = ap.parse_args()

    prevent = Path(os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT")))
    cache_root = Path(args.cache_root) if args.cache_root else prevent / "data/processed/vortex-jepa/v1_mean"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.split) as f:
        split = json.load(f)

    # Collect per-encounter p99.99 thresholds (nested: {case_id: {enc_str: thr}})
    # + per-frame samples for train_stats. The nested schema is what
    # OmegaPipeline.from_manifest expects (matches v1 manifest format).
    thresholds: dict[str, dict[str, float]] = {}
    train_samples = []
    train_cases = [c for c in split["cases"].values() if c["split"] == "train"]
    print(f"[mean-pipeline] processing {len(train_cases)} train cases + all cases for thresholds")

    all_cases = list(split["cases"].values())
    for c in all_cases:
        cid = c["case_id"]
        for k in range(c["n_encounters_full"]):
            cache_p = cache_root / cid / f"encounter_{k:02d}.h5"
            if not cache_p.exists():
                continue
            with h5py.File(cache_p, "r") as f:
                om = f["omega_z"][:].astype(np.float32)
            mask = np.isfinite(om)
            if not mask.any():
                continue
            thr = float(np.percentile(np.abs(om[mask]), 99.99))
            thresholds.setdefault(cid, {})[str(k)] = thr
            if c["split"] == "train" and k in c.get("train_encounter_indices", []):
                # Add a downsampled sample for train_stats (every 8th frame to keep it tractable)
                train_samples.append(om[::8].ravel())

    pool = np.concatenate(train_samples) if train_samples else np.zeros(1, dtype=np.float32)
    train_mean = float(pool.mean())
    train_std = float(pool.std())
    n_pixels = int(pool.size * 8)
    print(f"[mean-pipeline] thresholds: {len(thresholds)} entries; "
          f"train_stats: mean={train_mean:.4f} std={train_std:.4f} n_pixels={n_pixels}")
    print(f"[mean-pipeline] 3*std (sigma): {3 * train_std:.4f}")

    # Load reference v1 manifest for the mask + schema shape
    v1_manifest_path = REPO / "outputs/data_pipeline/v1/manifest.json"
    with v1_manifest_path.open() as f:
        v1_manifest = json.load(f)

    # Copy the mask file to the new dir
    v1_mask_path = REPO / "outputs/data_pipeline/v1/airfoil_adjacent_mask.npy"
    if v1_mask_path.exists():
        shutil.copy(v1_mask_path, out_dir / "airfoil_adjacent_mask.npy")

    # Build the new manifest with same schema
    manifest = dict(v1_manifest)
    manifest["version"] = "1.0.0-mean"
    manifest["partition"] = "v1_mean"
    manifest["note"] = (
        "Spanwise-mean omega_z (averaged across 32 z-stations) instead of single "
        "mid-span slice. Thresholds + train_stats recomputed on the mean cache. "
        "All other fields (mask, schema) inherited from v1 manifest. "
        "Created by scripts/build_omega_mean_pipeline.py for Session 14 Path 2."
    )
    manifest["parent_manifest"] = "outputs/data_pipeline/v1/manifest.json"
    manifest["parent_version"] = "1.0.0"
    manifest["thresholds"] = thresholds
    manifest["train_stats"] = {"mean": train_mean, "std": train_std, "n_pixels": n_pixels}

    out_path = out_dir / "manifest.json"
    with out_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[mean-pipeline] saved -> {out_path}")


if __name__ == "__main__":
    main()
