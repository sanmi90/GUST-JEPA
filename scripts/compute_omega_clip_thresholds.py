"""Precompute per-encounter omega clipping thresholds + airfoil mask.

Generates two artifact-suppression artifacts that the training pipeline
can consume directly:

1. `outputs/runs/session9/omega_clip_thresholds.json`:
   `{case_id: {encounter_index: {p99: ..., p99_9: ..., p99_99: ...}}}`
   for every (case, encounter) in train/test_a/test_b/test_c. The p99.99
   is the recommended clip threshold per the Session 9 diagnostic.

2. `outputs/runs/session9/airfoil_adjacent_mask.npy`:
   Shape `(192, 96)` boolean array. True where the cell is INSIDE the
   airfoil OR in the 1-cell-adjacent layer. Setting omega = 0 on this
   mask removes the leading-edge finite-difference artifact in 93-100%
   of |omega| > 500 pixels across all 246 encounters.

Run once before training; both files are cheap to load into the dataset.

Usage:
    python scripts/compute_omega_clip_thresholds.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import binary_dilation

REPO = Path(__file__).resolve().parents[1]
PREVENT = Path(os.environ.get("PREVENT_ROOT", "/home/carlos/PREVENT"))
CACHE = Path(os.environ.get("VORTEX_JEPA_CACHE", PREVENT / "data" / "processed" / "vortex-jepa"))


def build_airfoil_adjacent_mask() -> np.ndarray:
    """Return the boolean (192, 96) mask marking inside-solid + 1-cell-adjacent.

    Read the raw Baseline file to get the canonical inside_solid mask
    at the mid-plane (z-index 16), then dilate it by one cell to get
    the 1-layer-adjacent stencil. Setting omega = 0 on this combined
    mask removes the LE finite-difference artifact.
    """
    raw = PREVENT / "data" / "raw" / "periodic" / "Baseline.h5"
    with h5py.File(raw, "r") as f:
        inside_solid_3d = np.asarray(f["inside_solid"]).squeeze()  # (192, 96, 32)
    solid = (inside_solid_3d[:, :, 16] > 0)
    dilated = binary_dilation(solid)  # 1-cell expansion
    # Return the union (solid + adjacent layer)
    return dilated.astype(np.bool_)


def compute_encounter_thresholds(omega: np.ndarray, mask: np.ndarray) -> dict:
    """Compute p99 / p99.9 / p99.99 of |omega| over the encounter, with
    mask applied (omega set to 0 where mask is True).

    The thresholds are computed AFTER spatial masking so they reflect
    the cleaned distribution rather than the artifact-polluted one.
    """
    # Apply spatial mask: zero out inside-solid + 1-cell-adjacent
    omega_clean = omega.copy()
    omega_clean[:, mask] = 0.0
    a = np.abs(omega_clean)
    ps = np.percentile(a, [99, 99.9, 99.99])
    return {
        "p99": float(ps[0]),
        "p99_9": float(ps[1]),
        "p99_99": float(ps[2]),
        "max_before_mask": float(np.abs(omega).max()),
        "max_after_mask": float(a.max()),
    }


def main() -> None:
    out_dir = REPO / "outputs" / "runs" / "session9"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[clip-thresholds] computing airfoil-adjacent mask")
    mask = build_airfoil_adjacent_mask()
    mask_path = out_dir / "airfoil_adjacent_mask.npy"
    np.save(mask_path, mask)
    print(f"[clip-thresholds] saved {mask_path} ({mask.sum()} masked cells "
          f"of {mask.size} = {100*mask.sum()/mask.size:.2f}%)")

    print("[clip-thresholds] iterating manifest to compute per-encounter thresholds")
    with open(REPO / "configs" / "splits" / "split_v1.json") as f:
        manifest = json.load(f)

    thresholds: dict[str, dict[str, dict[str, float]]] = {}
    for cid, case in sorted(manifest["cases"].items()):
        case_thresh = {}
        for k in range(case["n_encounters_full"]):
            p = CACHE / "v1" / cid / f"encounter_{k:02d}.h5"
            if not p.exists():
                continue
            with h5py.File(p, "r") as f:
                omega = np.asarray(f["omega_z"], dtype=np.float32)
            case_thresh[str(k)] = compute_encounter_thresholds(omega, mask)
        thresholds[cid] = case_thresh

    json_path = out_dir / "omega_clip_thresholds.json"
    with open(json_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"[clip-thresholds] saved {json_path}")

    # Print a quick summary
    p99_99_values = []
    max_before_mask = []
    max_after_mask = []
    for cid, case_thresh in thresholds.items():
        for k, t in case_thresh.items():
            p99_99_values.append(t["p99_99"])
            max_before_mask.append(t["max_before_mask"])
            max_after_mask.append(t["max_after_mask"])
    p99_99_values = np.array(p99_99_values)
    max_before = np.array(max_before_mask)
    max_after = np.array(max_after_mask)
    print()
    print("[clip-thresholds] per-encounter p99.99 (computed AFTER spatial mask):")
    print(f"  min: {p99_99_values.min():.2f}, median: {np.median(p99_99_values):.2f}, "
          f"mean: {p99_99_values.mean():.2f}, max: {p99_99_values.max():.2f}")
    print(f"[clip-thresholds] max |omega| BEFORE mask: min={max_before.min():.1f}, "
          f"median={np.median(max_before):.1f}, max={max_before.max():.1f}")
    print(f"[clip-thresholds] max |omega| AFTER mask:  min={max_after.min():.1f}, "
          f"median={np.median(max_after):.1f}, max={max_after.max():.1f}")


if __name__ == "__main__":
    main()
