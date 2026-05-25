"""Build the spanwise-mean omega cache (partition v1_mean).

For each case in the v1 split, read raw curlU at every spanwise z-station
(32 stations), average the omega_z component across stations, and write the
same cache schema as scripts/preprocess.py but with omega_z = spanwise mean
instead of the single mid-span slice.

Outputs: ``${PREVENT_ROOT}/data/processed/vortex-jepa/v1_mean/<case_id>/encounter_{k:02d}.h5``

All other fields (p_wall, C_L, C_D, attrs) are identical to v1. The
``mid_span_index`` attr is replaced with the literal string "mean".

Wall time: ~30-60 min depending on disk speed. Disk usage: ~30 MB per
encounter * 250 encounters = ~7.5 GB.

Used to support Session 14 D113 follow-up: train a JEPA encoder on
spanwise-mean vorticity to test whether the slice-vs-mean preprocessing
choice changes the production metric. The slice-trained encoder already
gave +0.07 GDY R^2 on mean input (D113); this script enables the proper
apples-to-apples retrain.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import yaml


REPO = Path(__file__).resolve().parent.parent


def extract_encounter_mean(raw: h5py.File, k: int, config: dict) -> dict:
    frames_per = config["encounter"]["frames_per_encounter"]
    f0, f1 = k * frames_per, (k + 1) * frames_per
    omega_z_idx = config["raw"]["omega_z_index"]
    nan_fill = config["outputs"]["omega_z"]["nan_fill"]
    # Spanwise mean across all 32 z-stations (NOT a single slice)
    curl_all = raw[config["raw"]["curl_path"]][f0:f1, :, :, :, omega_z_idx]
    curl_all = np.nan_to_num(curl_all, nan=nan_fill)
    omega_z_mean = curl_all.mean(axis=3).astype(np.float32)
    n_surf = config["sensors"]["n_surface_points"]
    n_z = config["sensors"]["n_z_stations"]
    p_raw = raw[config["raw"]["sensors_p_path"]][:, f0:f1]
    p_wall = p_raw.reshape(n_surf, n_z, frames_per).mean(axis=1).T.astype(np.float32)
    cl = raw[config["raw"]["forces_CL_path"]][f0:f1].astype(np.float32)
    cd = raw[config["raw"]["forces_CD_path"]][f0:f1].astype(np.float32)
    return dict(omega_z=omega_z_mean, p_wall=p_wall, C_L=cl, C_D=cd,
                frame_start=f0, frame_end=f1)


def write_encounter(out_path: Path, encoded: dict, case_meta: dict,
                    k: int, config: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    o_cfg = config["outputs"]
    with h5py.File(out_path, "w") as g:
        g.create_dataset("omega_z", data=encoded["omega_z"],
                         chunks=tuple(o_cfg["omega_z"]["chunks"]),
                         compression=o_cfg["omega_z"]["compression"])
        g.create_dataset("p_wall", data=encoded["p_wall"],
                         chunks=tuple(o_cfg["p_wall"]["chunks"]))
        g.create_dataset("C_L", data=encoded["C_L"])
        g.create_dataset("C_D", data=encoded["C_D"])
        g.attrs["case_id"] = case_meta["case_id"]
        g.attrs["G"] = float(case_meta["G"])
        g.attrs["D"] = float(case_meta["D"])
        g.attrs["Y"] = float(case_meta["Y"])
        g.attrs["source_group"] = case_meta["source_group"]
        g.attrs["encounter_index"] = int(k)
        g.attrs["frame_start"] = int(encoded["frame_start"])
        g.attrs["frame_end"] = int(encoded["frame_end"])
        g.attrs["dt_tc"] = float(config["encounter"]["dt_tc"])
        g.attrs["impact_frame_estimate"] = 40
        g.attrs["mid_span_index"] = "mean"
        g.attrs["omega_z_sign_convention"] = config["raw"]["omega_z_sign_convention"]
        g.attrs["preprocessing_version"] = "1.0.0-mean"
        g.attrs["partition_version"] = "v1_mean"
        g.attrs["raw_relative_path"] = case_meta["relative_path"]
        g.attrs["n_frames"] = int(config["encounter"]["frames_per_encounter"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=str(REPO / "configs/splits/split_v1.json"))
    ap.add_argument("--config", default=str(REPO / "configs/preprocessing.yaml"))
    ap.add_argument("--cache-root", default=None,
                    help="Override cache root. Default: $PREVENT_ROOT/data/processed/vortex-jepa/v1_mean")
    ap.add_argument("--cases", nargs="+", default=None,
                    help="Subset of case_ids to build. Default: all.")
    ap.add_argument("--force", action="store_true", help="Overwrite existing.")
    args = ap.parse_args()

    prevent_root = Path(os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT")))
    cache_root = Path(args.cache_root) if args.cache_root else prevent_root / "data/processed/vortex-jepa/v1_mean"
    cache_root.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(args.split) as f:
        split = json.load(f)

    cases = list(split["cases"].values())
    if args.cases:
        wanted = set(args.cases)
        cases = [c for c in cases if c["case_id"] in wanted]
    print(f"[mean-cache] building {len(cases)} cases at {cache_root}")

    import time
    t0 = time.time()
    n_skipped = 0
    n_done = 0
    for c in cases:
        case_id = c["case_id"]
        n_enc = int(c["n_encounters_full"])
        raw_path = prevent_root / c["relative_path"]
        if not raw_path.exists():
            print(f"[mean-cache] SKIP {case_id}: raw missing {raw_path}")
            n_skipped += 1
            continue
        case_dir = cache_root / case_id
        with h5py.File(raw_path, "r") as f:
            for k in range(n_enc):
                out_path = case_dir / f"encounter_{k:02d}.h5"
                if out_path.exists() and not args.force:
                    continue
                encoded = extract_encounter_mean(f, k, config)
                write_encounter(out_path, encoded, c, k, config)
        n_done += 1
        if n_done % 10 == 0:
            elapsed = time.time() - t0
            print(f"[mean-cache] {n_done}/{len(cases)} cases done "
                  f"({elapsed:.1f} s, ~{elapsed / max(n_done, 1):.1f} s/case)")
    elapsed = time.time() - t0
    print(f"[mean-cache] complete: {n_done} cases in {elapsed:.1f} s "
          f"(skipped {n_skipped})")


if __name__ == "__main__":
    main()
