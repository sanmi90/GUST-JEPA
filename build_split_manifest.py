"""Generate split_v1.json from raw_cases_inventory.yaml.

Encodes the locked-in decisions:
- Test C = G == +4.0 (4 periodic cases). Never used for selection.
- Test B = 6 manually selected interior cases in (G, D, Y), pooled across source groups.
- Test A = within each training case, last 2 of 6 encounters (periodic)
          or last 1 of 4 encounters (run3).
- Baseline (no gust) is in `train` like any other periodic case, AND is
  flagged `is_calibration_reference: true` so calibration tools can still
  identify it.
- Periodic trailing partials discarded.
- Impact frame ~ 40 (vortex centroid crosses LE at t ~ 1.965 t/c with dt = 0.05).
- Sub-trajectory L = 32 with 70 percent impact-aware, 30 percent uniform sampling.
"""

import yaml
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent
INVENTORY_PATH = REPO / "data_manifest" / "raw_cases_inventory.yaml"
OUTPUT_PATH = REPO / "split_v1.json"

# Test B selection: 6 interior cases, pooled across periodic and run3.
# Picked for diversity in (G, D, Y) with D = 1.0 as primary interior axis,
# allowing two D-boundary cases (5 and 6) to stress-test interpolation at D extremes.
TEST_B_CASE_IDS = {
    "G+1.00_D1.00_Y+0.10",   # periodic, fully interior
    "G+2.00_D1.00_Y-0.10",   # periodic, interior, opposite Y
    "G+0.50_D1.00_Y+0.20",   # run3, interior, weak positive G
    "G-0.50_D1.00_Y+0.00",   # run3, interior, weak negative G, exact midplane
    "G-1.50_D0.50_Y-0.20",   # run3, moderate negative G, D boundary
    "G+1.50_D1.50_Y-0.20",   # run3, moderate positive G, D boundary
}


def case_split(case_id: str, G: float, source: str) -> str:
    if G == 4.0:
        return "test_c"
    if case_id in TEST_B_CASE_IDS:
        return "test_b"
    return "train"


def encounter_assignment(split: str, source: str, n_encounters: int):
    """Return (train_encounter_indices, test_a_encounter_indices)."""
    if split != "train":
        return [], list(range(n_encounters))
    if source == "periodic":
        return [0, 1, 2, 3], [4, 5]
    return [0, 1, 2], [3]


def main():
    with open(INVENTORY_PATH) as f:
        raw = f.read()
    inv = yaml.safe_load(raw)
    inv_sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()

    cases_out = {}
    for c in inv["cases"]:
        case_id = c["case_id"]
        G = float(c.get("G", 0.0))
        D = float(c.get("D", 0.0))
        Y = float(c.get("Y", 0.0))
        source = c["source_group"]
        n_encounters = int(c.get("n_encounters_full", 0))
        trailing = int(c.get("trailing_partial_frames", 0))

        split = case_split(case_id, G, source)
        train_enc, test_a_enc = encounter_assignment(split, source, n_encounters)

        cases_out[case_id] = {
            "case_id": case_id,
            "G": G,
            "D": D,
            "Y": Y,
            "source_group": source,
            "filename": c.get("filename", ""),
            "relative_path": c.get("relative_path", ""),
            "n_frames": int(c.get("n_frames", 0)),
            "n_encounters_full": n_encounters,
            "trailing_partial_frames": trailing,
            "discarded_trailing_partial": trailing > 0,
            "split": split,
            "train_encounter_indices": train_enc,
            "test_a_encounter_indices": test_a_enc,
            "is_calibration_reference": case_id == "Baseline",
        }

    # Verify Test B selections exist
    for tb in TEST_B_CASE_IDS:
        assert tb in cases_out, f"Test B case {tb!r} not present in inventory"
        assert cases_out[tb]["split"] == "test_b", f"{tb!r} not tagged test_b"

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

    manifest = {
        "manifest_version": "split_v1",
        "created_iso": datetime.now(timezone.utc).isoformat(),
        "source_inventory": {
            "version": inv.get("manifest_version", ""),
            "created_iso": inv.get("created_iso", ""),
            "sha256": inv_sha,
        },
        "physical_constants": {
            "airfoil": inv["physical_constants"]["airfoil"],
            "alpha_deg": inv["physical_constants"]["alpha_deg"],
            "Re": inv["physical_constants"]["Re"],
            "dt_tc": inv["physical_constants"]["dt_tc"],
            "gust_period_tc": inv["physical_constants"]["gust_period_tc"],
            "encounter_frames": inv["physical_constants"]["encounter_frames"],
        },
        "impact_metadata": {
            "impact_frame_estimate": 40,
            "impact_window_frames": [25, 55],
            "rationale": (
                "Vortex always launched at x ~ -1.965 (chord-leading-edge frame), "
                "convected by U_inf = 1 with dt_tc = 0.05. Centroid crosses LE at "
                "t ~ 1.965 t/c which is frame ~ 40. Impact window [25, 55] covers "
                "the largest gust radius (D/2 = 0.75, ~15 frames of physical extent) "
                "plus a small pre/post-impact margin."
            ),
        },
        "subtrajectory_sampling": {
            "subtraj_length": 32,
            "impact_aware_fraction": 0.70,
            "impact_overlap_start_range": [8, 40],
            "uniform_start_range": [0, 88],
            "rationale": (
                "With L = 32 and impact window [25, 55], any start in "
                "impact_overlap_start_range = [8, 40] yields a sub-trajectory "
                "[start, start + 32) whose intersection with the impact window contains "
                "at least 7 frames. This is the 'impact-aware' branch of the mixture: "
                "impact_aware_fraction is the mixture weight (probability of drawing "
                "from this branch), impact_overlap_start_range is the qualifying start "
                "range. Note: start = 8 produces sub-traj [8, 40), which does NOT contain "
                "frame 40 itself but does contain the impact-window prefix [25, 40). The "
                "30 percent uniform branch preserves coverage of pre-launch convection "
                "and post-impact recovery."
            ),
        },
        "split_policy": {
            "test_c_criterion": "G == +4.0 (4 periodic cases)",
            "test_b_criterion": "6 manually selected interior cases, pooled across source groups",
            "test_a_criterion": "within training cases: last 2 of 6 encounters (periodic) or last 1 of 4 (run3)",
            "baseline_role": "in train (encounters 0-3) and test_a (encounters 4-5), like any other periodic case; also flagged is_calibration_reference=true for tooling that needs to identify the no-gust reference",
            "trailing_partial_policy": "discarded",
            "evaluation_rollout": (
                "Full-episode rollout at evaluation: initialize predictor from frames [0, 32), "
                "autoregressively predict frames [32, 120). Latent and decoded errors reported."
            ),
        },
        "test_b_cases": sorted(TEST_B_CASE_IDS),
        "test_c_cases": sorted([cid for cid, c in cases_out.items() if c["split"] == "test_c"]),
        "summary": {
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
        },
        "cases": cases_out,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=False)

    print(f"Wrote {OUTPUT_PATH}")
    print(f"Inventory SHA256: {inv_sha[:16]}...")
    print("Summary:")
    for k, v in manifest["summary"].items():
        print(f"  {k}: {v}")
    print()
    print("Test B cases:")
    for cid in manifest["test_b_cases"]:
        c = cases_out[cid]
        print(f"  {cid}  (src={c['source_group']}, n_enc={c['n_encounters_full']})")
    print()
    print("Test C cases:")
    for cid in manifest["test_c_cases"]:
        c = cases_out[cid]
        print(f"  {cid}  (src={c['source_group']}, n_enc={c['n_encounters_full']})")


if __name__ == "__main__":
    main()
