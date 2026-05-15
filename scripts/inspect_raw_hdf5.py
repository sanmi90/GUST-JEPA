"""Inspect raw PREVENT HDF5 files and report their schema.

Step 0 of SESSION_DATA_PREP.md: before any preprocessing is defined, dump the
on-disk HDF5 layout for one periodic file and one run3 file. Downstream code
(configs/preprocessing.yaml, scripts/preprocess.py) reads schema.yaml.

Emits, into --output (default outputs/schema_inspection/):

    periodic.txt   h5dump-style tree for the periodic sample
    run3.txt       h5dump-style tree for the run3 sample
    schema.yaml    structured schema: detected variables, coordinates,
                   time-axis heuristic, force time series, researcher
                   action items.

Detection is best-effort by name matching. Unmatched fields appear in the .txt
trees so the researcher can fill them in manually before downstream code runs.

CLI:
    python scripts/inspect_raw_hdf5.py \\
        --periodic-sample $PREVENT_ROOT/data/raw/periodic/Baseline.h5 \\
        --run3-sample $PREVENT_ROOT/data/raw/run3/Gust_XXX.h5 \\
        --output outputs/schema_inspection/
"""
from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import yaml


VAR_PATTERNS = {
    "omega_z": [r"omega[_/\-]?z\b", r"vorticity[_/\-]?z\b", r"\bwz\b", r"\bw_z\b"],
    "omega_x": [r"omega[_/\-]?x\b", r"vorticity[_/\-]?x\b", r"\bwx\b", r"\bw_x\b"],
    "omega_y": [r"omega[_/\-]?y\b", r"vorticity[_/\-]?y\b", r"\bwy\b", r"\bw_y\b"],
    "u":       [r"/u$", r"velocity[_/\-]?x\b", r"\bu_x\b", r"u_velocity"],
    "v":       [r"/v$", r"velocity[_/\-]?y\b", r"\bu_y\b", r"v_velocity"],
    "w":       [r"/w$", r"velocity[_/\-]?z\b", r"\bu_z\b", r"w_velocity"],
    "p":       [r"/p$", r"pressure", r"\bpres\b"],
    "C_L":     [r"/cl$", r"/c_l$", r"lift_coef", r"force.*lift"],
    "C_D":     [r"/cd$", r"/c_d$", r"drag_coef", r"force.*drag"],
    "C_M":     [r"/cm$", r"/c_m$", r"moment_coef", r"pitching_moment"],
}

COORD_PATTERNS = {
    "x": [r"^/x$", r"/coords/x$", r"/mesh/x$", r"/grid/x$"],
    "y": [r"^/y$", r"/coords/y$", r"/mesh/y$", r"/grid/y$"],
    "z": [r"^/z$", r"/coords/z$", r"/mesh/z$", r"/grid/z$"],
}


def _match_any(path: str, patterns: list[str]) -> bool:
    p = path.lower()
    return any(re.search(pat, p) for pat in patterns)


def _stringify(v):
    if isinstance(v, np.ndarray):
        if v.ndim == 0:
            return _stringify(v.item())
        if v.size <= 8:
            return v.tolist()
        return f"<array shape={list(v.shape)} dtype={v.dtype}>"
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def walk_hdf5(f: h5py.File) -> list[dict]:
    items = [{"path": "/", "kind": "group",
              "attrs": {k: _stringify(v) for k, v in f.attrs.items()}}]

    def visitor(name, obj):
        path = "/" + name
        attrs = {k: _stringify(v) for k, v in obj.attrs.items()}
        if isinstance(obj, h5py.Dataset):
            items.append({
                "path": path,
                "kind": "dataset",
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "chunks": list(obj.chunks) if obj.chunks else None,
                "compression": obj.compression,
                "compression_opts": obj.compression_opts,
                "attrs": attrs,
            })
        else:
            items.append({"path": path, "kind": "group", "attrs": attrs})

    f.visititems(visitor)
    return items


def detect_variables(items: list[dict]) -> dict:
    detected: dict = {}
    for item in items:
        if item["kind"] != "dataset":
            continue
        for var, patterns in VAR_PATTERNS.items():
            if var in detected:
                continue
            if _match_any(item["path"], patterns):
                detected[var] = {
                    "path": item["path"],
                    "shape": item["shape"],
                    "dtype": item["dtype"],
                }
                break
    return detected


def detect_coordinates(f: h5py.File, items: list[dict]) -> dict:
    coords: dict = {}
    for axis, patterns in COORD_PATTERNS.items():
        for item in items:
            if item["kind"] != "dataset":
                continue
            if not _match_any(item["path"], patterns):
                continue
            ds = f[item["path"]]
            info = {
                "path": item["path"],
                "shape": list(ds.shape),
                "dtype": str(ds.dtype),
            }
            if ds.ndim == 1 and ds.size <= 10000:
                arr = ds[...]
                info["min"] = float(arr.min())
                info["max"] = float(arr.max())
                info["n"] = int(arr.size)
                if axis == "z":
                    info["L_z"] = float(arr.max() - arr.min())
                    info["mid_index_argmin_abs"] = int(np.argmin(np.abs(arr)))
                    info["mid_index_n_over_2"] = arr.size // 2
            coords[axis] = info
            break
    return coords


def detect_time_axis(items: list[dict]) -> dict | None:
    largest = None
    for item in items:
        if item["kind"] != "dataset":
            continue
        if not item.get("shape") or len(item["shape"]) < 3:
            continue
        size = 1
        for s in item["shape"]:
            size *= s
        if largest is None or size > largest["size"]:
            largest = {"size": size, "item": item}
    if largest is None:
        return None
    it = largest["item"]
    return {
        "candidate_dataset": it["path"],
        "shape": it["shape"],
        "assumed_time_axis": 0,
        "assumed_time_length": it["shape"][0],
        "note": "Heuristic only: leading axis of the largest dataset. Verify against expected n_frames (800 periodic, 480 run3).",
    }


def detect_forces(items: list[dict]) -> dict:
    forces: dict = {}
    for var in ("C_L", "C_D", "C_M"):
        for item in items:
            if item["kind"] != "dataset":
                continue
            if _match_any(item["path"], VAR_PATTERNS[var]):
                forces[var] = {
                    "path": item["path"],
                    "shape": item["shape"],
                    "dtype": item["dtype"],
                }
                break
    return forces


def render_text_tree(items: list[dict]) -> str:
    lines: list[str] = []
    for item in items:
        path = item["path"]
        depth = 0 if path == "/" else path.count("/") - 1
        indent = "  " * depth
        if item["kind"] == "group":
            lines.append(f"{indent}{path}{'' if path == '/' else '/'}  (group)")
        else:
            extras = []
            if item["chunks"]:
                extras.append(f"chunks={tuple(item['chunks'])}")
            if item["compression"]:
                extras.append(f"compression={item['compression']}")
            extra_str = (f"  [{', '.join(extras)}]") if extras else ""
            lines.append(
                f"{indent}{path}  shape={tuple(item['shape'])} dtype={item['dtype']}{extra_str}"
            )
        for k, v in (item.get("attrs") or {}).items():
            lines.append(f"{indent}  @{k} = {v!r}")
    return "\n".join(lines) + "\n"


def _human_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PiB"


def inspect_file(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Sample file not found: {path}")
    with h5py.File(path, "r") as f:
        items = walk_hdf5(f)
        variables = detect_variables(items)
        coords = detect_coordinates(f, items)
        time_info = detect_time_axis(items)
        forces = detect_forces(items)
    size = path.stat().st_size
    return {
        "path": str(path),
        "file_size_bytes": size,
        "file_size_human": _human_bytes(size),
        "tree_items": items,
        "variables_detected": variables,
        "coordinates": coords,
        "time_axis_heuristic": time_info,
        "forces_detected": forces,
    }


def write_outputs(periodic: dict, run3: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "periodic.txt").write_text(
        f"# Schema inspection for periodic sample\n"
        f"# File: {periodic['path']}\n"
        f"# Size: {periodic['file_size_human']}\n\n"
        + render_text_tree(periodic["tree_items"])
    )
    (out_dir / "run3.txt").write_text(
        f"# Schema inspection for run3 sample\n"
        f"# File: {run3['path']}\n"
        f"# Size: {run3['file_size_human']}\n\n"
        + render_text_tree(run3["tree_items"])
    )

    schema = {
        "manifest_version": "schema_inspection_v1",
        "created_iso": datetime.now(timezone.utc).isoformat(),
        "source": "scripts/inspect_raw_hdf5.py",
        "inspected_files": {
            "periodic": {k: v for k, v in periodic.items() if k != "tree_items"},
            "run3": {k: v for k, v in run3.items() if k != "tree_items"},
        },
        "researcher_action_items": [
            "Confirm physical variable mapping (which path holds omega_z, pressure, etc.) and fill in any not auto-detected.",
            "Confirm the spanwise axis (axis index in the omega_z array) and the mid-span strategy (argmin(|z|) vs n_z // 2).",
            "Confirm L_z and the spanwise grid resolution.",
            "Decide whether wall pressure is directly available on the airfoil surface or must be interpolated from the volume.",
            "Decide whether C_L/C_D/C_M are stored as time series or must be integrated from surface stress.",
            "Confirm mesh type: body-fitted vs immersed boundary on a Cartesian block.",
            "Confirm time-axis interpretation: which axis of which dataset indexes frames, and whether frame 0 is gust-launch-aligned.",
            "Confirm airfoil orientation: rotated alpha=14 deg in the mesh frame, or freestream tilted.",
        ],
    }
    (out_dir / "schema.yaml").write_text(yaml.safe_dump(schema, sort_keys=False))


def print_summary(label: str, result: dict) -> None:
    print(f"\n=== {label}: {result['path']} ===")
    print(f"  Size: {result['file_size_human']}")
    n_ds = sum(1 for i in result["tree_items"] if i["kind"] == "dataset")
    n_grp = sum(1 for i in result["tree_items"] if i["kind"] == "group")
    print(f"  Tree: {n_grp} groups, {n_ds} datasets")
    if result["variables_detected"]:
        print("  Variables detected:")
        for v, info in result["variables_detected"].items():
            print(f"    {v:8s} -> {info['path']}  shape={tuple(info['shape'])}  dtype={info['dtype']}")
    else:
        print("  Variables detected: (none auto-matched; inspect the .txt tree)")
    if result["coordinates"]:
        print("  Coordinates:")
        for axis, info in result["coordinates"].items():
            extras = []
            if "L_z" in info:
                extras.append(f"L_z={info['L_z']:.4f}")
            if "n" in info:
                extras.append(f"n={info['n']}")
            tail = ("  " + ", ".join(extras)) if extras else ""
            print(f"    {axis} -> {info['path']}  shape={tuple(info['shape'])}{tail}")
    else:
        print("  Coordinates: (none auto-matched)")
    if result["forces_detected"]:
        print("  Forces detected:")
        for v, info in result["forces_detected"].items():
            print(f"    {v} -> {info['path']}  shape={tuple(info['shape'])}")
    else:
        print("  Forces detected: (none auto-matched)")
    ti = result["time_axis_heuristic"]
    if ti:
        print(f"  Time-axis heuristic: {ti['candidate_dataset']} shape={tuple(ti['shape'])} "
              f"(axis 0 -> length {ti['assumed_time_length']})")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--periodic-sample", required=True, type=Path,
                    help="Path to one periodic HDF5 file (e.g., Baseline.h5 or Gust_*.h5 in data/raw/periodic).")
    ap.add_argument("--run3-sample", required=True, type=Path,
                    help="Path to one run3 HDF5 file (Gust_*.h5 in data/raw/run3).")
    ap.add_argument("--output", default=Path("outputs/schema_inspection"), type=Path,
                    help="Directory where periodic.txt, run3.txt, and schema.yaml are written.")
    args = ap.parse_args()

    periodic = inspect_file(args.periodic_sample)
    run3 = inspect_file(args.run3_sample)

    write_outputs(periodic, run3, args.output)

    print_summary("periodic", periodic)
    print_summary("run3", run3)
    print("\nWrote:")
    print(f"  {args.output}/periodic.txt")
    print(f"  {args.output}/run3.txt")
    print(f"  {args.output}/schema.yaml")
    print("\nNext: review schema.yaml, resolve the researcher_action_items, then "
          "write configs/preprocessing.yaml.")


if __name__ == "__main__":
    main()
