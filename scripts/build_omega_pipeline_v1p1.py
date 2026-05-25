"""Build outputs/data_pipeline/v1p1/manifest.json (incremental v1.1).

v1.1 = v1.0 + 28 clip thresholds for the 7 new run3 cases (Gust_048-054)
that landed in the v1.5 split (Session 14 task #10 / D108). The new cases
are NOT in the train pool, so train_stats (mean, std, n_pixels) and the
airfoil mask are kept identical to v1.0 -- only the per-encounter clip
``thresholds`` dict grows by 28 entries.

Unlike ``scripts/build_omega_pipeline.py`` (which iterates the entire
partition's split manifest from scratch), this script merges new thresholds
into the existing v1 manifest without re-touching the v1 stats. That keeps
Session 11-13 numerics reproducible against the v1 manifest while letting
Session 14 v1.5 supplement work pull from v1.1.

Run with no arguments:

    python scripts/build_omega_pipeline_v1p1.py

Outputs:
    outputs/data_pipeline/v1p1/airfoil_adjacent_mask.npy (copy of v1)
    outputs/data_pipeline/v1p1/manifest.json
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.omega_pipeline import OmegaPipeline  # noqa: E402

PREVENT = Path(os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT")))
CACHE = Path(
    os.environ.get("VORTEX_JEPA_CACHE", str(PREVENT / "data" / "processed" / "vortex-jepa"))
)

V1_MANIFEST = REPO / "outputs" / "data_pipeline" / "v1" / "manifest.json"
V1P1_DIR = REPO / "outputs" / "data_pipeline" / "v1p1"
V1P1_MANIFEST = V1P1_DIR / "manifest.json"

# The 7 new run3 cases added in Session 14 task #10 (Gust_048..054).
NEW_CASES = [
    "G+0.50_D0.50_Y-0.20",
    "G+0.50_D1.50_Y+0.00",
    "G+2.00_D1.00_Y+0.20",
    "G+3.00_D0.50_Y+0.00",
    "G+3.00_D1.00_Y-0.20",
    "G+3.00_D1.50_Y-0.10",
    "G-2.00_D1.00_Y+0.00",
]
N_ENCOUNTERS_PER_CASE = 4  # run3 cases all have 4 encounters (480 frames / 120)
CLIP_PERCENTILE = 99.99


def main() -> None:
    V1P1_DIR.mkdir(parents=True, exist_ok=True)

    # Load the v1 manifest and its mask sidecar.
    with open(V1_MANIFEST) as f:
        v1 = json.load(f)
    mask_path_v1 = V1_MANIFEST.parent / v1["mask_path"]
    mask = np.load(mask_path_v1)
    n_v1_thresh = sum(len(v) for v in v1["thresholds"].values())
    print(f"[v1p1] loaded v1 manifest: {n_v1_thresh} thresholds across "
          f"{len(v1['thresholds'])} cases", flush=True)
    print(f"[v1p1] v1 train_stats: mean={v1['train_stats']['mean']:.6f} "
          f"std={v1['train_stats']['std']:.4f} "
          f"n_pixels={v1['train_stats']['n_pixels']:,}", flush=True)

    # Copy the mask sidecar verbatim into v1p1/.
    mask_dst = V1P1_DIR / v1["mask_path"]
    shutil.copy2(mask_path_v1, mask_dst)
    print(f"[v1p1] copied mask sidecar to {mask_dst}", flush=True)

    # Compute new thresholds for the 7 v1.5 supplement cases.
    new_thresholds: dict[str, dict[str, float]] = {}
    n_new = 0
    for cid in sorted(NEW_CASES):
        case_thresh: dict[str, float] = {}
        for k in range(N_ENCOUNTERS_PER_CASE):
            p = CACHE / "v1" / cid / f"encounter_{k:02d}.h5"
            if not p.exists():
                raise FileNotFoundError(f"missing cache encounter: {p}")
            with h5py.File(p, "r") as f:
                omega = np.asarray(f["omega_z"], dtype=np.float32)
            omega_masked = omega.copy()
            omega_masked[:, mask] = 0.0
            t = float(np.percentile(np.abs(omega_masked), CLIP_PERCENTILE))
            case_thresh[str(k)] = t
            n_new += 1
            print(f"  {cid} enc {k:02d}: p99.99 = {t:.2f}", flush=True)
        new_thresholds[cid] = case_thresh
    print(f"[v1p1] computed {n_new} new thresholds across "
          f"{len(new_thresholds)} cases", flush=True)

    # Merge: refuse to silently overwrite an existing case.
    merged: dict[str, dict[str, float]] = dict(v1["thresholds"])
    for cid, t in new_thresholds.items():
        if cid in merged:
            raise RuntimeError(
                f"case {cid} already present in v1 manifest; refusing to overwrite."
            )
        merged[cid] = t
    print(f"[v1p1] merged thresholds: {len(merged)} cases, "
          f"{sum(len(v) for v in merged.values())} total", flush=True)

    manifest = {
        "version": "1.1.0",
        "mask_path": v1["mask_path"],
        "train_stats": dict(v1["train_stats"]),
        "thresholds": merged,
        "partition": "v1.1",
        "clip_percentile": CLIP_PERCENTILE,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "note": (
            "v1.1 = v1.0 + 28 clip thresholds for the 7 new run3 cases "
            "(Gust_048-054). Train stats unchanged (no new training cases). "
            "Created for Session 14 v1.5 split."
        ),
        "parent_manifest": "outputs/data_pipeline/v1/manifest.json",
        "parent_version": v1.get("version", "v1"),
    }
    with open(V1P1_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[v1p1] wrote {V1P1_MANIFEST}", flush=True)

    # Round-trip via OmegaPipeline.from_manifest.
    print("[v1p1] round-trip sanity check via OmegaPipeline.from_manifest",
          flush=True)
    pipe = OmegaPipeline.from_manifest(V1P1_MANIFEST)
    probe_cid = "G+3.00_D1.00_Y-0.20"
    probe_k = 0
    t_probe = pipe.get_threshold(probe_cid, probe_k)
    assert np.isfinite(t_probe), f"expected finite threshold, got {t_probe}"
    assert t_probe == new_thresholds[probe_cid][str(probe_k)], "reload mismatch"
    print(f"  pipe.get_threshold({probe_cid!r}, {probe_k}) = {t_probe:.4f} (finite)",
          flush=True)
    t_v1 = pipe.get_threshold("Baseline", 0)
    assert t_v1 == v1["thresholds"]["Baseline"]["0"], "v1 thresholds not preserved"
    print(f"  pipe.get_threshold('Baseline', 0) = {t_v1:.4f} (matches v1)",
          flush=True)
    print(f"  train_stats: mean={pipe.train_stats.mean:.6f} "
          f"std={pipe.train_stats.std:.4f} "
          f"n_pixels={pipe.train_stats.n_pixels:,}", flush=True)
    print("[v1p1] DONE", flush=True)


if __name__ == "__main__":
    main()
