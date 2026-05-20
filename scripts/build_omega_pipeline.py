"""Build the reusable omega preprocessing pipeline manifest.

Iterates every encounter in the partition's split manifest, computes
its post-mask p99.99 clip threshold, then aggregates train-set
statistics (mean and std) AFTER both the spatial mask and per-encounter
clip have been applied. Saves a single JSON manifest plus the binary
airfoil mask sidecar so the pipeline can be reconstructed via
``OmegaPipeline.from_manifest``.

This is run once at session start (or whenever new cases are added).
Output files are checked into the repo's outputs/data_pipeline/
directory so all downstream training and evaluation pulls from the
same canonical preprocessing.

Usage:
    python scripts/build_omega_pipeline.py
        --partition v1
        --output-dir outputs/data_pipeline/v1
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import binary_dilation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.omega_pipeline import OmegaPipeline, OmegaTrainStats  # noqa: E402


PREVENT = Path(os.environ.get("PREVENT_ROOT", "/home/carlos/PREVENT"))
CACHE = Path(os.environ.get("VORTEX_JEPA_CACHE", PREVENT / "data" / "processed" / "vortex-jepa"))


def build_airfoil_adjacent_mask() -> np.ndarray:
    """Read the canonical inside_solid mask from the raw Baseline file,
    project to the mid-plane (z-index 16), and dilate by 1 cell.

    Returns a ``(192, 96)`` boolean array True where the cell is inside
    the airfoil or in the 1-cell-adjacent stencil layer (where the
    finite-difference vorticity produces artifact spikes).
    """
    raw = PREVENT / "data" / "raw" / "periodic" / "Baseline.h5"
    with h5py.File(raw, "r") as f:
        inside_solid = np.asarray(f["inside_solid"]).squeeze()[:, :, 16]
    return binary_dilation(inside_solid > 0).astype(np.bool_)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build omega preprocessing manifest")
    p.add_argument("--partition", type=str, default="v1")
    p.add_argument("--output-dir", type=str, default="outputs/data_pipeline/v1")
    p.add_argument("--clip-percentile", type=float, default=99.99,
                   help="Per-encounter clip percentile (post-mask). Default 99.99 "
                        "removes the top 0.01%% of pixels which catches the "
                        "leading-edge artifact while preserving every dense "
                        "physical feature.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = REPO / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[build-pipeline] partition={args.partition}", flush=True)
    print(f"[build-pipeline] clip percentile = p{args.clip_percentile}", flush=True)
    print(f"[build-pipeline] output dir = {out_dir}", flush=True)

    # Stage A: build and save the airfoil mask
    print("[build-pipeline] step 1/3: airfoil-adjacent mask", flush=True)
    mask = build_airfoil_adjacent_mask()
    mask_filename = "airfoil_adjacent_mask.npy"
    np.save(out_dir / mask_filename, mask)
    n_masked = int(mask.sum())
    print(f"  saved {out_dir / mask_filename} ({n_masked} cells masked of "
          f"{mask.size} = {100*n_masked/mask.size:.2f}%)",
          flush=True)

    # Stage B: per-encounter p99.99 thresholds (post-mask)
    print("[build-pipeline] step 2/3: per-encounter p"
          f"{args.clip_percentile} thresholds (post-mask)", flush=True)
    with open(REPO / "configs" / "splits" / f"split_{args.partition}.json") as f:
        manifest = json.load(f)

    thresholds: dict[str, dict[str, float]] = {}
    for cid, case in sorted(manifest["cases"].items()):
        case_thresh = {}
        for k in range(case["n_encounters_full"]):
            p = CACHE / args.partition / cid / f"encounter_{k:02d}.h5"
            if not p.exists():
                continue
            with h5py.File(p, "r") as f:
                omega = np.asarray(f["omega_z"], dtype=np.float32)
            # Apply mask first, then compute percentile of |omega|
            omega_masked = omega.copy()
            omega_masked[:, mask] = 0.0
            t = float(np.percentile(np.abs(omega_masked), args.clip_percentile))
            case_thresh[str(k)] = t
        thresholds[cid] = case_thresh
    n_encs = sum(len(v) for v in thresholds.values())
    print(f"  computed {n_encs} encounter thresholds across {len(thresholds)} cases",
          flush=True)
    t_vals = np.array([v for case_t in thresholds.values() for v in case_t.values()])
    print(f"  p{args.clip_percentile} range: min={t_vals.min():.2f}, "
          f"median={np.median(t_vals):.2f}, max={t_vals.max():.2f}",
          flush=True)

    # Stage C: train-set mean and std AFTER mask + clip
    print("[build-pipeline] step 3/3: train-set mean/std (post-mask, post-clip)",
          flush=True)
    sum_x = 0.0
    sum_x2 = 0.0
    n_pixels = 0
    for cid, case in sorted(manifest["cases"].items()):
        if case["split"] != "train":
            continue
        test_a_idx = set(case.get("test_a_encounter_indices", []))
        for k in range(case["n_encounters_full"]):
            if k in test_a_idx:
                continue
            p = CACHE / args.partition / cid / f"encounter_{k:02d}.h5"
            if not p.exists():
                continue
            with h5py.File(p, "r") as f:
                omega = np.asarray(f["omega_z"], dtype=np.float32)
            # Stage 1: spatial mask
            omega_clean = omega.copy()
            omega_clean[:, mask] = 0.0
            # Stage 2: per-encounter clip
            t = thresholds[cid][str(k)]
            omega_clean = np.clip(omega_clean, -t, t)
            # Exclude the masked-zero cells from the statistics; they
            # carry no physical information so should not bias the
            # mean / std toward zero.
            keep = ~mask
            vals = omega_clean[:, keep]
            sum_x += vals.sum()
            sum_x2 += (vals ** 2).sum()
            n_pixels += vals.size

    mean = float(sum_x / n_pixels)
    var = float(sum_x2 / n_pixels) - mean ** 2
    std = float(np.sqrt(max(var, 0.0)))
    print(f"  train pixels (post-mask, unmasked-only): {n_pixels:,}", flush=True)
    print(f"  mean = {mean:.6f}, std = {std:.4f}", flush=True)

    # Save the manifest
    pipeline = OmegaPipeline(
        mask=mask,
        thresholds=thresholds,
        train_stats=OmegaTrainStats(mean=mean, std=std, n_pixels=n_pixels),
        version=args.partition,
    )
    manifest_dict = pipeline.to_dict(mask_path=mask_filename)
    manifest_dict["partition"] = args.partition
    manifest_dict["clip_percentile"] = args.clip_percentile
    manifest_dict["created_at"] = datetime.datetime.now().isoformat(timespec="seconds")

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_dict, f, indent=2)
    print(f"[build-pipeline] saved {manifest_path}", flush=True)

    # Round-trip sanity check
    print("[build-pipeline] round-trip sanity check", flush=True)
    pipe2 = OmegaPipeline.from_manifest(manifest_path)
    test_cid = next(iter(thresholds.keys()))
    test_k = 0
    test_thresh = pipe2.get_threshold(test_cid, test_k)
    print(f"  {test_cid} encounter {test_k}: threshold = {test_thresh:.2f} "
          f"(reload matches: {test_thresh == thresholds[test_cid]['0']})",
          flush=True)
    print(f"  train_stats: mean={pipe2.train_stats.mean:.6f}, "
          f"std={pipe2.train_stats.std:.4f}",
          flush=True)
    print("[build-pipeline] DONE", flush=True)


if __name__ == "__main__":
    main()
