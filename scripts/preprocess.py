"""Preprocess raw PREVENT HDF5 files into the per-encounter JEPA cache.

For each case in data_manifest/raw_cases_inventory.yaml, extract per-encounter:
  - omega_z (mid-plane spanwise vorticity): shape (120, nx, ny)
  - p_wall (spanwise-averaged wall pressure):  shape (120, n_surface_points)
  - C_L, C_D                                    shape (120,)

Output layout:
  $VORTEX_JEPA_CACHE/{partition_version}/{case_id}/encounter_{k:02d}.h5

Defaults are taken from configs/preprocessing.yaml. Raw-file decoding
parameters (omega_z component index, sign convention, mid-span index, sensor
reshape) are baked there and frozen per `preprocessing_version`.

CLI:
    python scripts/preprocess.py --partition v1
    python scripts/preprocess.py --partition v1 --cases Baseline G+4.00_D1.00_Y+0.10
    python scripts/preprocess.py --partition v1 --dry-run
    python scripts/preprocess.py --partition v1 --force        # re-cache existing
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import yaml


REPO = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    with open(REPO / "configs" / "preprocessing.yaml") as f:
        return yaml.safe_load(f)


def load_inventory() -> dict:
    with open(REPO / "data_manifest" / "raw_cases_inventory.yaml") as f:
        return yaml.safe_load(f)


def resolve_cache_root(config: dict) -> Path:
    root = os.environ.get(config["cache"]["root_env"])
    if root:
        return Path(root)
    prevent_root = os.environ.get("PREVENT_ROOT")
    if not prevent_root:
        sys.exit("Neither VORTEX_JEPA_CACHE nor PREVENT_ROOT is set.")
    return Path(prevent_root) / config["cache"]["root_default"]


def encounter_out_path(cache_root: Path, case_id: str, k: int, config: dict) -> Path:
    return (cache_root
            / config["cache"]["partition_subdir"].format(partition_version=config["partition_target"])
            / config["cache"]["case_subdir"].format(case_id=case_id)
            / config["cache"]["encounter_file"].format(encounter_index=k))


def extract_encounter(raw: h5py.File, k: int, config: dict) -> dict:
    frames_per = config["encounter"]["frames_per_encounter"]
    f0, f1 = k * frames_per, (k + 1) * frames_per
    mid = config["grid"]["mid_span_index"]
    omega_z_idx = config["raw"]["omega_z_index"]
    nan_fill = config["outputs"]["omega_z"]["nan_fill"]

    omega_z = raw[config["raw"]["curl_path"]][f0:f1, :, :, mid, omega_z_idx]
    omega_z = np.nan_to_num(omega_z, nan=nan_fill).astype(np.float32)

    n_surf = config["sensors"]["n_surface_points"]
    n_z = config["sensors"]["n_z_stations"]
    p_raw = raw[config["raw"]["sensors_p_path"]][:, f0:f1]
    p_wall = p_raw.reshape(n_surf, n_z, frames_per).mean(axis=1).T.astype(np.float32)

    cl = raw[config["raw"]["forces_CL_path"]][f0:f1].astype(np.float32)
    cd = raw[config["raw"]["forces_CD_path"]][f0:f1].astype(np.float32)

    return dict(omega_z=omega_z, p_wall=p_wall, C_L=cl, C_D=cd,
                frame_start=f0, frame_end=f1)


def write_encounter(out_path: Path, encoded: dict, case_meta: dict,
                    encounter_index: int, config: dict) -> None:
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
        g.attrs["encounter_index"] = int(encounter_index)
        g.attrs["frame_start"] = int(encoded["frame_start"])
        g.attrs["frame_end"] = int(encoded["frame_end"])
        g.attrs["n_frames"] = int(config["encounter"]["frames_per_encounter"])
        g.attrs["dt_tc"] = float(config["encounter"]["dt_tc"])
        g.attrs["impact_frame_estimate"] = int(config["encounter"]["impact_frame_estimate"])
        g.attrs["mid_span_index"] = int(config["grid"]["mid_span_index"])
        g.attrs["omega_z_sign_convention"] = config["raw"]["omega_z_sign_convention"]
        g.attrs["preprocessing_version"] = config["preprocessing_version"]
        g.attrs["partition_version"] = config["partition_target"]
        g.attrs["raw_relative_path"] = case_meta["relative_path"]


def process_case(case_meta: dict, prevent_root: Path, cache_root: Path,
                 config: dict, dry_run: bool, force: bool) -> tuple[int, int]:
    raw_path = prevent_root / case_meta["relative_path"]
    n = int(case_meta["n_encounters_full"])
    written, skipped = 0, 0
    raw = None
    try:
        for k in range(n):
            out_path = encounter_out_path(cache_root, case_meta["case_id"], k, config)
            if out_path.exists() and not force:
                skipped += 1
                continue
            if dry_run:
                written += 1
                continue
            if raw is None:
                raw = h5py.File(raw_path, "r")
            encoded = extract_encounter(raw, k, config)
            write_encounter(out_path, encoded, case_meta, k, config)
            written += 1
    finally:
        if raw is not None:
            raw.close()
    return written, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--partition", default="v1",
                    help="Partition version (must match preprocessing.yaml partition_target).")
    ap.add_argument("--cases", nargs="*",
                    help="Optional list of case_ids to process; default = all in inventory.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be written but do not extract.")
    ap.add_argument("--force", action="store_true",
                    help="Re-cache encounters even if already present.")
    args = ap.parse_args()

    config = load_config()
    if args.partition != config["partition_target"]:
        sys.exit(f"--partition {args.partition} does not match "
                 f"preprocessing.yaml partition_target={config['partition_target']}")

    inv = load_inventory()
    prevent_root = Path(os.environ.get("PREVENT_ROOT", "/home/carlos/PREVENT"))
    cache_root = resolve_cache_root(config)

    print(f"PREVENT_ROOT          = {prevent_root}")
    print(f"cache_root            = {cache_root}")
    print(f"partition             = {config['partition_target']}")
    print(f"preprocessing_version = {config['preprocessing_version']}")
    print(f"dry_run = {args.dry_run},  force = {args.force}")

    cases = inv["cases"]
    if args.cases:
        wanted = set(args.cases)
        cases = [c for c in cases if c["case_id"] in wanted]
        missing = wanted - {c["case_id"] for c in cases}
        if missing:
            sys.exit(f"Unknown case_ids: {sorted(missing)}")

    print(f"\nProcessing {len(cases)} case(s).")
    t0 = time.time()
    total_w, total_s = 0, 0
    for c in cases:
        ts = time.time()
        w, s = process_case(c, prevent_root, cache_root, config,
                            dry_run=args.dry_run, force=args.force)
        total_w += w
        total_s += s
        dt = time.time() - ts
        msg = f"  {c['case_id']:35s}  encounters: +{w:2d} written  /{s:2d} skipped  ({dt:5.1f}s)"
        print(msg)
    dt_total = time.time() - t0
    print(f"\nDone in {dt_total:.1f}s.  Total: written={total_w}, skipped={total_s}.")


if __name__ == "__main__":
    main()
