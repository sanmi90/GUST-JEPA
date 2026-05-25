"""Generate split_v1p5.json: v1 plus 7 new run3 cases assigned to test_b.

Session 14 user instruction (2026-05-24): "There are new cases in run3
integrate them but add them in test." The 7 new run3 cases (Gust_048-054)
appeared on disk after the 2026-05-22 inventory regeneration and are NOT
in split_v1.json. To preserve Session 11-13 reproducibility (W&B
``split_sha256`` anchors) we leave split_v1.json untouched and produce a
parallel split_v1p5.json that:

- Includes every v1 case with the same split assignment.
- Adds the 7 new run3 cases parsed directly from filenames, tagged
  ``split: test_b`` per the user's "add in test" instruction. None of
  the 7 new cases are G=+4 so test_c is unaffected.
- Bumps test_b from 6 to 13 cases (180 -> 180 train, 70 -> 70 test_a,
  28 -> 56 test_b, 24 -> 24 test_c encounters).

D89 wake_observable train_stats are NOT affected because no new case
joins train. The Session 12 _train_stats_v1.4 snapshot remains valid.

Run:
    python build_split_manifest_v1p5.py
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent
V1_PATH = REPO / "configs" / "splits" / "split_v1.json"
INVENTORY_PATH = REPO / "data_manifest" / "raw_cases_inventory.yaml"
OUTPUT_PATH = REPO / "configs" / "splits" / "split_v1p5.json"

# The 7 new run3 cases (Gust_048-054) plus any later additions detected.
NEW_RUN3_FILENAMES = [
    "Gust_048_x-1.989_y-0.290_s2.0_d1.0.h5",
    "Gust_049_x-1.941_y-0.484_s-2.0_d1.0.h5",
    "Gust_050_x-1.892_y-0.678_s0.5_d0.5.h5",
    "Gust_051_x-1.941_y-0.484_s0.5_d1.5.h5",
    "Gust_052_x-1.892_y-0.678_s3.0_d1.0.h5",
    "Gust_053_x-1.941_y-0.484_s3.0_d0.5.h5",
    "Gust_054_x-1.916_y-0.581_s3.0_d1.5.h5",
]

ALPHA_DEG = 14.0
ALPHA_RAD = math.radians(ALPHA_DEG)
COS_A = math.cos(ALPHA_RAD)
SIN_A = math.sin(ALPHA_RAD)
DOE2_Y_GRID = (-0.4, -0.2, -0.1, 0.0, +0.1, +0.2, +0.4)

X_RE = re.compile(r"x(-?\d+\.\d+)")
Y_RE = re.compile(r"y(-?\d+\.\d+)")
G_RE = re.compile(r"s(-?\d+\.\d+)")
D_RE = re.compile(r"d(\d+\.\d)")


def _snap_to_doe2_grid(y: float) -> float:
    return min(DOE2_Y_GRID, key=lambda gpt: abs(gpt - y))


def parse_filename(name: str) -> dict:
    """Parse a Gust_*.h5 filename into (G, D, Y, case_id) using the same
    rotation formulas as scripts/100c_raw_cases_inventory.py.
    """
    g = float(G_RE.search(name).group(1))
    d = float(D_RE.search(name).group(1))
    x_file = float(X_RE.search(name).group(1))
    y_file = float(Y_RE.search(name).group(1))
    y_from_y = (y_file + 0.484) / COS_A
    y_from_x = -(x_file + 1.941) / SIN_A
    if abs(y_from_y - y_from_x) > 1e-2:
        raise ValueError(
            f"Y_from_y={y_from_y:.6f} vs Y_from_x={y_from_x:.6f} inconsistent in {name}"
        )
    y_snap = _snap_to_doe2_grid(0.5 * (y_from_y + y_from_x))
    case_id = f"G{g:+.2f}_D{d:.2f}_Y{y_snap:+.2f}"
    return {"G": g, "D": d, "Y": y_snap, "case_id": case_id, "filename": name}


def main() -> None:
    with V1_PATH.open() as f:
        v1 = json.load(f)
    with INVENTORY_PATH.open() as f:
        inv_raw = f.read()
    inv_sha = hashlib.sha256(inv_raw.encode("utf-8")).hexdigest()

    # Copy v1 cases dict and bump test_b assignments for new cases
    cases_out = dict(v1["cases"])

    new_test_b_case_ids: list[str] = []
    for fname in NEW_RUN3_FILENAMES:
        parsed = parse_filename(fname)
        case_id = parsed["case_id"]
        if case_id in cases_out:
            print(f"  WARN: {fname} parses to {case_id} which is already in v1; skipping")
            continue
        relative_path = f"data/raw/periodic/run3/{fname}"
        cases_out[case_id] = {
            "case_id": case_id,
            "G": parsed["G"],
            "D": parsed["D"],
            "Y": parsed["Y"],
            "source_group": "run3",
            "filename": fname,
            "relative_path": relative_path,
            "n_frames": 480,
            "n_encounters_full": 4,
            "trailing_partial_frames": 0,
            "discarded_trailing_partial": False,
            "split": "test_b",
            "train_encounter_indices": [],
            "test_a_encounter_indices": [],
            "is_calibration_reference": False,
        }
        new_test_b_case_ids.append(case_id)

    # Recompute summary
    counts = {"train": 0, "test_b": 0, "test_c": 0, "n_calibration_reference": 0}
    enc_counts = {"train": 0, "test_a": 0, "test_b": 0, "test_c": 0}
    for c in cases_out.values():
        s = c["split"]
        counts[s] = counts.get(s, 0) + 1
        if c.get("is_calibration_reference"):
            counts["n_calibration_reference"] += 1
        if s == "train":
            enc_counts["train"] += len(c["train_encounter_indices"])
            enc_counts["test_a"] += len(c["test_a_encounter_indices"])
        elif s == "test_b":
            enc_counts["test_b"] += c["n_encounters_full"]
        elif s == "test_c":
            enc_counts["test_c"] += c["n_encounters_full"]

    manifest = dict(v1)  # carry forward physical_constants, impact_metadata, etc.
    manifest["manifest_version"] = "split_v1p5"
    manifest["created_iso"] = datetime.now(timezone.utc).isoformat()
    manifest["source_inventory"] = {
        "version": "raw_cases_inventory_v1+v1p5_supplement",
        "created_iso": v1["source_inventory"]["created_iso"],
        "sha256": inv_sha,
        "v1p5_supplement": {
            "added_run3_files": NEW_RUN3_FILENAMES,
            "rationale": (
                "7 new run3 cases (Gust_048-054) appeared on disk after the "
                "2026-05-22 inventory regen. User instructed (2026-05-24, "
                "Session 14): add them to test, not training. Assigned to "
                "test_b per the user instruction. No re-training implied."
            ),
        },
    }
    manifest["test_b_cases"] = sorted(
        [cid for cid, c in cases_out.items() if c["split"] == "test_b"]
    )
    manifest["test_c_cases"] = sorted(
        [cid for cid, c in cases_out.items() if c["split"] == "test_c"]
    )
    manifest["v1p5_added_test_b_cases"] = sorted(new_test_b_case_ids)
    manifest["summary"] = {
        "n_cases_total": len(cases_out),
        "n_cases_train": counts["train"],
        "n_cases_test_b": counts["test_b"],
        "n_cases_test_c": counts["test_c"],
        "n_cases_calibration_reference": counts["n_calibration_reference"],
        "n_encounters_train": enc_counts["train"],
        "n_encounters_test_a": enc_counts["test_a"],
        "n_encounters_test_b": enc_counts["test_b"],
        "n_encounters_test_c": enc_counts["test_c"],
        "n_encounters_total_in_splits": sum(enc_counts.values()),
        "delta_vs_v1": {
            "added_cases": len(new_test_b_case_ids),
            "added_test_b_encounters": enc_counts["test_b"] - v1["summary"]["n_encounters_test_b"],
        },
    }
    manifest["cases"] = cases_out

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=False)

    print(f"Wrote {OUTPUT_PATH}")
    print(f"Inventory SHA256: {inv_sha[:16]}...")
    print(f"Summary:")
    for k, v in manifest["summary"].items():
        if isinstance(v, dict):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v}")
    print(f"\nNewly added test_b cases (run3 supplement):")
    for cid in sorted(new_test_b_case_ids):
        c = cases_out[cid]
        print(f"  + {cid}  (G={c['G']:+.2f} D={c['D']:.2f} Y={c['Y']:+.2f})")


def maybe_run_integrity_audit() -> None:
    """Invoke the standalone integrity audit if its cache + script are present.

    This keeps every v1.5+ split-builder run paired with an up-to-date
    integrity manifest + clean-split companion. User instruction
    (2026-05-25, Session 14): when new cases are integrated, the
    data-quality manifest should refresh automatically.
    """
    audit_script = REPO / "scripts" / "data_integrity_audit.py"
    if not audit_script.exists():
        return
    import subprocess
    print()
    print("[v1.5 builder] running data integrity audit (per user request)")
    try:
        subprocess.run(
            ["python", str(audit_script),
             "--split", str(OUTPUT_PATH),
             "--write-clean-split"],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"[v1.5 builder] integrity audit failed: {e}; clean split not refreshed")


if __name__ == "__main__":
    main()
    maybe_run_integrity_audit()
