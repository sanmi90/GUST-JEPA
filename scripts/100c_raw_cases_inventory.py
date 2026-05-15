"""Raw cases inventory — generates a parser manifest between on-disk
HDF5 filenames and physical conditions (G, D, Y/c).

Inspects two source directories:
  - data/raw/periodic/        → 800-snapshot cases (1 Baseline + N gust)
  - data/raw/periodic/run3/   → 480-snapshot cases (run3 DoE-2 gust)

For every `Gust_*` filename, parses (G, D, Y) via the locked Plan v3.3
parser (Y back-solved from x_file / y_file rotation around α=14°,
snapped to the DoE-2 Y grid). Emits a YAML manifest with:
  - Physical constants header (dt, gust period, encounter length,
    airfoil, Re, parser formulas).
  - Cases list mapping filename ↔ case_id with G, D, Y, source group,
    n_frames.

Re-runnable: new cases dropped into either directory are picked up on
the next run. The output file is overwritten.

CLI:
    PYTHONPATH=src python scripts/periodic_v2/100c_raw_cases_inventory.py
        [--out data_manifest/raw_cases_inventory.yaml]
        [--print-table]
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/home/carlos/PREVENT")
sys.path.insert(0, str(REPO / "scripts" / "periodic_v2"))


# Reuse the locked parser regexes from 100b_dice_filename_parser.py.
# Inlined here (rather than imported) because 100b's module name starts
# with a digit, which makes import awkward; the regex spec is short and
# the locked formulas are mirrored explicitly in the YAML header.
ALPHA_DEG = 14.0
ALPHA_RAD = math.radians(ALPHA_DEG)
COS_A = math.cos(ALPHA_RAD)
SIN_A = math.sin(ALPHA_RAD)

DOE2_Y_GRID = (-0.4, -0.2, -0.1, 0.0, +0.1, +0.2, +0.4)

X_RE = re.compile(r"x(-?\d+\.\d+)")
Y_RE = re.compile(r"y(-?\d+\.\d+)")
G_RE = re.compile(r"s(-?\d+\.\d+)")
D_RE = re.compile(r"d(\d+\.\d)")

PERIODIC_DIR = REPO / "data" / "raw" / "periodic"
RUN3_DIR = PERIODIC_DIR / "run3"

# Per-source frame counts (convention; not verified by HDF5 open).
FRAMES_PER_SOURCE = {
    "periodic": 800,
    "run3": 480,
}

# Encounter length (locked Plan v3.3 / v3.4 convention).
ENCOUNTER_FRAMES = 120

DEFAULT_OUT = REPO / "data_manifest" / "raw_cases_inventory.yaml"


def _snap_to_doe2_grid(y: float) -> float:
    return min(DOE2_Y_GRID, key=lambda gpt: abs(gpt - y))


def parse_filename(name: str) -> dict | None:
    """Parse `Gust_*` filename → {G, D, Y, case_id}. Returns None on
    Baseline or any non-Gust filename.
    """
    if not name.startswith("Gust_"):
        return None
    g_m = G_RE.search(name)
    d_m = D_RE.search(name)
    x_m = X_RE.search(name)
    y_m = Y_RE.search(name)
    if not (g_m and d_m and x_m and y_m):
        return {"parse_error": f"missing token in {name}"}
    g = float(g_m.group(1))
    d = float(d_m.group(1))
    x_file = float(x_m.group(1))
    y_file = float(y_m.group(1))
    y_from_y = (y_file + 0.484) / COS_A
    y_from_x = -(x_file + 1.941) / SIN_A
    if abs(y_from_y - y_from_x) > 1e-2:
        return {
            "parse_error": (
                f"Y_from_y={y_from_y:.6f} vs Y_from_x={y_from_x:.6f} "
                f"inconsistent in {name}"
            )
        }
    y_mean = 0.5 * (y_from_y + y_from_x)
    y_snap = _snap_to_doe2_grid(y_mean)
    case_id = f"G{g:+.2f}_D{d:.2f}_Y{y_snap:+.2f}"
    return {
        "G": g,
        "D": d,
        "Y": y_snap,
        "case_id": case_id,
        "Y_from_y_file": round(y_from_y, 6),
        "Y_from_x_file": round(y_from_x, 6),
    }


def scan_dir(directory: Path, source_group: str) -> list[dict]:
    """Scan a directory for *.h5 files; return list of case entries."""
    entries: list[dict] = []
    if not directory.exists():
        return entries
    n_frames = FRAMES_PER_SOURCE[source_group]
    for path in sorted(directory.glob("*.h5")):
        name = path.name
        parsed = parse_filename(name)
        relative = path.relative_to(REPO).as_posix()
        entry: dict = {
            "filename": name,
            "relative_path": relative,
            "source_group": source_group,
            "n_frames": n_frames,
            "n_encounters_full": n_frames // ENCOUNTER_FRAMES,
            "trailing_partial_frames": n_frames % ENCOUNTER_FRAMES,
        }
        if parsed is None:
            # Baseline / non-Gust file. No gust release ↔ G=0, D=0
            # (zero amplitude, zero duration); Y is moot without a
            # gust trajectory, recorded as 0.0 by convention.
            entry["case_id"] = path.stem  # "Baseline"
            entry["G"] = 0.0
            entry["D"] = 0.0
            entry["Y"] = 0.0
            entry["note"] = "no gust release; periodic shedding baseline"
        elif "parse_error" in parsed:
            entry["case_id"] = None
            entry["G"] = None
            entry["D"] = None
            entry["Y"] = None
            entry["parse_error"] = parsed["parse_error"]
        else:
            entry["case_id"] = parsed["case_id"]
            entry["G"] = parsed["G"]
            entry["D"] = parsed["D"]
            entry["Y"] = parsed["Y"]
            entry["Y_from_y_file"] = parsed["Y_from_y_file"]
            entry["Y_from_x_file"] = parsed["Y_from_x_file"]
        entries.append(entry)
    return entries


def detect_duplicates(entries: list[dict]) -> list[tuple[str, list[str]]]:
    """Return [(case_id, [filenames])] for any case_id appearing in 2+
    files.
    """
    by_cid: dict[str, list[str]] = {}
    for e in entries:
        cid = e.get("case_id")
        if cid is None:
            continue
        by_cid.setdefault(cid, []).append(e["filename"])
    return [(cid, fns) for cid, fns in by_cid.items() if len(fns) > 1]


def write_yaml(entries: list[dict], out_path: Path) -> None:
    """Write the YAML manifest with constants header + cases list."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    iso = datetime.now(timezone.utc).isoformat()
    duplicates = detect_duplicates(entries)
    n_by_src: dict[str, int] = {}
    n_parse_errors = 0
    for e in entries:
        n_by_src[e["source_group"]] = n_by_src.get(e["source_group"], 0) + 1
        if e.get("parse_error"):
            n_parse_errors += 1

    with open(out_path, "w") as f:
        f.write("# Raw cases inventory — parser manifest between on-disk\n")
        f.write("# HDF5 filenames and physical conditions (G, D, Y/c).\n")
        f.write("# Regenerated by scripts/periodic_v2/100c_raw_cases_inventory.py.\n")
        f.write("# Edit-by-hand discouraged; rerun the script to refresh.\n")
        f.write("\n")
        f.write("manifest_version: raw_cases_inventory_v1\n")
        f.write(f"created_iso: '{iso}'\n")
        f.write("source: scripts/periodic_v2/100c_raw_cases_inventory.py\n")
        f.write("\n")

        f.write("# -------------------------------------------------------------\n")
        f.write("# Physical constants (apply to every case below).\n")
        f.write("# -------------------------------------------------------------\n")
        f.write("physical_constants:\n")
        f.write("  airfoil: NACA0012\n")
        f.write("  alpha_deg: 14.0\n")
        f.write("  Re: 5000\n")
        f.write("  dt_tc: 0.05            # convective time per snapshot\n")
        f.write("  gust_period_tc: 6.0    # gust released every 6 t/c\n")
        f.write(f"  encounter_frames: {ENCOUNTER_FRAMES}  "
                f"# = gust_period_tc / dt_tc (locked Plan v3.3 / v3.4)\n")
        f.write("  frames_per_source:\n")
        f.write(f"    periodic: {FRAMES_PER_SOURCE['periodic']}    "
                "# 6 full encounters + 80-frame trailing partial\n")
        f.write(f"    run3: {FRAMES_PER_SOURCE['run3']}        "
                "# 4 full encounters, no trailing partial\n")
        f.write("\n")

        f.write("# -------------------------------------------------------------\n")
        f.write("# Filename parser (locked Plan v3.3 / v3.4 convention).\n")
        f.write("# `s` token encodes G (can be negative); `d` token encodes D.\n")
        f.write("# `x` and `y` tokens encode airfoil leading-edge position\n")
        f.write("# which round-trips to Y/c via the α=14° rotation below.\n")
        f.write("# -------------------------------------------------------------\n")
        f.write("parser:\n")
        f.write("  regex_g: 's(-?\\d+\\.\\d+)'\n")
        f.write("  regex_d: 'd(\\d+\\.\\d)'\n")
        f.write("  regex_x: 'x(-?\\d+\\.\\d+)'\n")
        f.write("  regex_y: 'y(-?\\d+\\.\\d+)'\n")
        f.write(f"  alpha_deg: {ALPHA_DEG}\n")
        f.write(f"  cos_alpha: {COS_A:.10f}\n")
        f.write(f"  sin_alpha: {SIN_A:.10f}\n")
        f.write("  formula_forward:\n")
        f.write("    y_file: '-0.484 + cos(14°) * Y'\n")
        f.write("    x_file: '-1.941 - sin(14°) * Y'\n")
        f.write("  formula_inverse:\n")
        f.write("    Y_from_y_file: '(y_file + 0.484) / cos(14°)'\n")
        f.write("    Y_from_x_file: '-(x_file + 1.941) / sin(14°)'\n")
        f.write("  doe2_y_grid: [-0.4, -0.2, -0.1, 0.0, +0.1, +0.2, +0.4]\n")
        f.write("  case_id_format: 'G{G:+.2f}_D{D:.2f}_Y{Y:+.2f}'\n")
        f.write("\n")

        f.write("# -------------------------------------------------------------\n")
        f.write("# Summary.\n")
        f.write("# -------------------------------------------------------------\n")
        f.write("summary:\n")
        f.write(f"  n_cases_total: {len(entries)}\n")
        for src, n in sorted(n_by_src.items()):
            f.write(f"  n_cases_{src}: {n}\n")
        f.write(f"  n_parse_errors: {n_parse_errors}\n")
        f.write(f"  n_duplicate_case_ids: {len(duplicates)}\n")
        if duplicates:
            f.write("  duplicates:\n")
            for cid, fns in duplicates:
                f.write(f"    {cid}:\n")
                for fn in fns:
                    f.write(f"      - {fn}\n")
        f.write("\n")

        f.write("# -------------------------------------------------------------\n")
        f.write("# Cases (sorted by source_group, then filename).\n")
        f.write("# -------------------------------------------------------------\n")
        f.write("cases:\n")
        for e in sorted(entries, key=lambda r: (r["source_group"], r["filename"])):
            f.write(f"  - filename: {e['filename']}\n")
            f.write(f"    relative_path: {e['relative_path']}\n")
            f.write(f"    source_group: {e['source_group']}\n")
            f.write(f"    n_frames: {e['n_frames']}\n")
            f.write(f"    n_encounters_full: {e['n_encounters_full']}\n")
            f.write(f"    trailing_partial_frames: {e['trailing_partial_frames']}\n")
            if e["case_id"] is not None:
                f.write(f"    case_id: {e['case_id']}\n")
            else:
                f.write("    case_id: null\n")
            if e["G"] is not None:
                f.write(f"    G: {e['G']:+.2f}\n")
                f.write(f"    D: {e['D']:.2f}\n")
                f.write(f"    Y: {e['Y']:+.2f}\n")
                if "Y_from_y_file" in e:
                    f.write(f"    Y_from_y_file: {e['Y_from_y_file']:+.6f}\n")
                if "Y_from_x_file" in e:
                    f.write(f"    Y_from_x_file: {e['Y_from_x_file']:+.6f}\n")
            else:
                f.write("    G: null\n")
                f.write("    D: null\n")
                f.write("    Y: null\n")
            if "note" in e:
                f.write(f"    note: {e['note']!r}\n")
            if "parse_error" in e:
                f.write(f"    parse_error: {e['parse_error']!r}\n")
            f.write("\n")


def print_table(entries: list[dict]) -> None:
    """Print a compact human-readable table to stdout."""
    print(f"{'source':>9s}  {'filename':40s}  {'case_id':24s}  "
          f"{'G':>6s}  {'D':>5s}  {'Y':>6s}  {'n':>4s}")
    print("-" * 110)
    for e in sorted(entries, key=lambda r: (r["source_group"], r["filename"])):
        g = f"{e['G']:+.2f}" if e["G"] is not None else "  -  "
        d = f"{e['D']:.2f}" if e["D"] is not None else "  -  "
        y = f"{e['Y']:+.2f}" if e["Y"] is not None else "  -  "
        cid = e["case_id"] or "(parse_error)"
        print(f"{e['source_group']:>9s}  {e['filename']:40s}  {cid:24s}  "
              f"{g:>6s}  {d:>5s}  {y:>6s}  {e['n_frames']:>4d}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"output YAML path (default: {DEFAULT_OUT})")
    parser.add_argument("--print-table", action="store_true",
                        help="print a human-readable summary table")
    args = parser.parse_args()

    entries: list[dict] = []
    entries.extend(scan_dir(PERIODIC_DIR, "periodic"))
    entries.extend(scan_dir(RUN3_DIR, "run3"))

    if not entries:
        print(f"FATAL: no *.h5 files found in {PERIODIC_DIR} or {RUN3_DIR}",
              file=sys.stderr)
        return 1

    write_yaml(entries, args.out)
    print(f"Wrote {args.out} ({len(entries)} cases)")

    duplicates = detect_duplicates(entries)
    n_errors = sum(1 for e in entries if e.get("parse_error"))
    if duplicates:
        print(f"WARN: {len(duplicates)} duplicate case_ids "
              f"(see summary.duplicates in YAML):")
        for cid, fns in duplicates:
            print(f"  {cid}: {fns}")
    if n_errors:
        print(f"WARN: {n_errors} parse errors")

    if args.print_table:
        print()
        print_table(entries)

    return 0


if __name__ == "__main__":
    sys.exit(main())
