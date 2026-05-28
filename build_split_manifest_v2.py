"""Generate split_v2.json from raw_cases_inventory.yaml.

Differences from v1 (`build_split_manifest.py`):

- **Test B** expanded from 6 to 10 cases under explicit stratification
  criteria C1-C5 (see TEST_B_V2_CRITERIA below). C6 (Manhattan neighbor
  constraint) was investigated and dropped — the run3 design is an offset
  Latin Hypercube, so no negative-G case has a Manhattan-1 neighbor; the
  interpolation-vs-OOD demarcation is carried by Test C (G=+4, strictly
  outside train's G range [-3, +3]).
- **Test A renamed to val.** The encounter-level within-train-case holdout
  is identical to v1 in mechanism (periodic: 4 train + 2 val; run3: 3
  train + 1 val); the rename clarifies that it is the model-selection
  signal monitored during training.
- **Two-tier Test B reporting**. Each test_b case carries a
  `n_train_neighbors_d2` metadata field counting train cases within
  Manhattan distance <= 2 grid steps in (G, D, Y). Cases with >= 4 such
  neighbors are tagged tier `interior` (genuine interpolation); 1-3 are
  `boundary`. Reported as separate aggregates in the paper.
- **Test C** unchanged (G == +4.0, 4 periodic cases).
- **Baseline** unchanged (split=train with is_calibration_reference=true).

The criteria below are the experimental-design section verbatim.
"""

import yaml
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent
INVENTORY_PATH = REPO / "data_manifest" / "raw_cases_inventory.yaml"
OUTPUT_PATH = REPO / "configs" / "splits" / "split_v2.json"

# Test B v2: 10 interior cases under criteria C1-C5.
# Selection deterministic; produced by analysis at /tmp/test_b_v2_proposal.json
# (or configs/splits/test_b_v2_proposal.json archive).
TEST_B_V2_CASE_IDS = {
    "G+0.50_D1.50_Y+0.00",   # run3,    |G|=0.5  Y=0
    "G-0.50_D1.00_Y-0.40",   # run3,    |G|=0.5  Y=-0.4 corner
    "G+1.00_D0.50_Y+0.40",   # run3,    |G|=1.0  Y=+0.4 corner
    "G-1.00_D1.00_Y-0.20",   # run3,    |G|=1.0  Y<0 mid
    "G+1.50_D1.50_Y+0.10",   # run3,    |G|=1.5  Y>0 mid
    "G-1.50_D0.50_Y-0.20",   # run3,    |G|=1.5  Y<0 mid
    "G+2.00_D0.50_Y+0.10",   # periodic |G|=2.0  Y>0 mid
    "G-2.00_D1.00_Y-0.40",   # run3,    |G|=2.0  Y=-0.4 corner
    "G+3.00_D1.00_Y+0.10",   # run3,    |G|=3.0  Y>0 mid
    "G-3.00_D1.50_Y-0.10",   # run3,    |G|=3.0  Y<0 mid
}

TEST_B_V2_CRITERIA = {
    "C1_G_magnitude":
        "At least 2 cases at |G| in {0.5, 1.0, 1.5, 2.0} and at least 1 case at "
        "|G|=3.0, so that every gust-strength regime is probed by the held-out set.",
    "C2_G_sign_balance":
        "Absolute difference between G > 0 and G < 0 case counts is at most 2, "
        "avoiding sign-biased model evaluation.",
    "C3_D_coverage":
        "At least 3 cases per gust-diameter bucket D in {0.5, 1.0, 1.5}; mid-D "
        "(D=1.0) receives one extra slot proportional to its inventory dominance.",
    "C4_Y_span":
        "At least 2 cases at the DoE Y corners (|Y|=0.4), at least 1 case at the "
        "midplane (Y=0), and both Y signs represented; together these probe the "
        "off-centerline impulse direction.",
    "C5_source_pooling":
        "Periodic and run3 cases mixed without per-source quota; the run3 design "
        "is offset Latin Hypercube which constrains the available combinations "
        "and limits the achievable periodic share.",
    "C6_dropped_rationale":
        "An earlier C6 required each test_b case to have at least one train "
        "neighbor at grid-step Manhattan distance <= 1 in (G, D, Y). The run3 "
        "design is offset Latin Hypercube (each (G, D) cell covers only 2 of 7 "
        "Y points, with no Y overlap between adjacent G), so no negative-G case "
        "has a Manhattan-1 train neighbor by construction. C6 was therefore "
        "unworkable. The interpolation-vs-OOD demarcation is carried instead by "
        "Test C: G == +4.0 is strictly outside the train G range [-3, +3].",
}

# Manhattan distance computation (used for the two-tier neighbor-density label).
G_GRID = [-3.0, -2.0, -1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
D_GRID = [0.0, 0.5, 1.0, 1.5]
Y_GRID = [-0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4]


def _grid_idx(grid, v, tol=1e-3):
    for i, g in enumerate(grid):
        if abs(g - v) < tol:
            return i
    return None


def case_split_v2(case_id: str, G: float) -> str:
    if G == 4.0:
        return "test_c"
    if case_id in TEST_B_V2_CASE_IDS:
        return "test_b"
    return "train"


def encounter_assignment_v2(split: str, source: str, n_encounters: int):
    """Return (train_encounter_indices, val_encounter_indices).

    Same mechanism as v1 (test_a -> val rename only)."""
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

    # First pass: build cases_out with split assignment
    cases_out = {}
    for c in inv["cases"]:
        case_id = c["case_id"]
        G = float(c.get("G", 0.0))
        D = float(c.get("D", 0.0))
        Y = float(c.get("Y", 0.0))
        source = c["source_group"]
        n_encounters = int(c.get("n_encounters_full", 0))
        trailing = int(c.get("trailing_partial_frames", 0))

        split = case_split_v2(case_id, G)
        train_enc, val_enc = encounter_assignment_v2(split, source, n_encounters)

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
            "val_encounter_indices": val_enc,
            "is_calibration_reference": case_id == "Baseline",
        }

    # Compute Manhattan-distance neighbor counts for two-tier labelling.
    train_grid_coords = set()
    for cid, c in cases_out.items():
        if c["split"] != "train":
            continue
        gi = _grid_idx(G_GRID, c["G"])
        di = _grid_idx(D_GRID, c["D"])
        yi = _grid_idx(Y_GRID, c["Y"])
        if None not in (gi, di, yi):
            train_grid_coords.add((gi, di, yi))

    for cid, c in cases_out.items():
        if c["split"] != "test_b":
            c["n_train_neighbors_d2"] = None
            c["tier"] = None
            continue
        gi = _grid_idx(G_GRID, c["G"])
        di = _grid_idx(D_GRID, c["D"])
        yi = _grid_idx(Y_GRID, c["Y"])
        if None in (gi, di, yi):
            c["n_train_neighbors_d2"] = None
            c["tier"] = None
            continue
        nbrs = 0
        for gj in range(len(G_GRID)):
            for dj in range(len(D_GRID)):
                for yj in range(len(Y_GRID)):
                    if (gj, dj, yj) == (gi, di, yi):
                        continue
                    md = abs(gj - gi) + abs(dj - di) + abs(yj - yi)
                    if md <= 2 and (gj, dj, yj) in train_grid_coords:
                        nbrs += 1
        c["n_train_neighbors_d2"] = nbrs
        c["tier"] = "interior" if nbrs >= 4 else "boundary"

    # Verify expected sets
    for tb in TEST_B_V2_CASE_IDS:
        assert tb in cases_out, f"Test B v2 case {tb!r} not present in inventory"
        assert cases_out[tb]["split"] == "test_b", f"{tb!r} not tagged test_b"

    counts = {"train": 0, "test_b": 0, "test_c": 0, "n_calibration_reference": 0}
    enc_counts = {"train": 0, "val": 0, "test_b": 0, "test_c": 0}
    for c in cases_out.values():
        s = c["split"]
        counts[s] = counts.get(s, 0) + 1
        if c.get("is_calibration_reference"):
            counts["n_calibration_reference"] += 1
        if s == "train":
            enc_counts["train"] += len(c["train_encounter_indices"])
            enc_counts["val"] += len(c["val_encounter_indices"])
        elif s == "test_b":
            enc_counts["test_b"] += c["n_encounters_full"]
        elif s == "test_c":
            enc_counts["test_c"] += c["n_encounters_full"]

    n_interior = sum(1 for c in cases_out.values()
                     if c.get("tier") == "interior")
    n_boundary = sum(1 for c in cases_out.values()
                     if c.get("tier") == "boundary")

    manifest = {
        "manifest_version": "split_v2",
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
                "at least 7 frames. The 30 percent uniform branch preserves coverage of "
                "pre-launch convection and post-impact recovery."
            ),
        },
        "split_policy": {
            "test_c_criterion": "G == +4.0 (4 periodic cases); never used for selection",
            "test_b_criterion": (
                "10 interior cases under stratification criteria C1-C5 (see "
                "test_b_criteria); cases tagged tier=interior if n_train_neighbors_d2 "
                ">= 4, else tier=boundary"
            ),
            "val_criterion": (
                "within training cases: last 2 of 6 encounters (periodic) or last 1 of 4 "
                "encounters (run3); used as the model-selection signal and reported "
                "alongside train and test metrics to demonstrate no overfitting"
            ),
            "baseline_role": (
                "in train (encounters 0-3) and val (encounters 4-5) like any other "
                "periodic case; flagged is_calibration_reference=true for tooling"
            ),
            "trailing_partial_policy": "discarded",
            "evaluation_rollout": (
                "Full-episode rollout at evaluation: initialize predictor from frames "
                "[0, 32), autoregressively predict frames [32, 120)"
            ),
            "uncertainty_protocol": {
                "bootstrap_n": 2000,
                "bootstrap_resample_unit": "encounter",
                "seed_variance_n": 3,
                "probe_cv_folds": 5,
                "probe_cv_unit": "case",
                "note": (
                    "Headline metrics report bootstrap CI (2000 resamples on test "
                    "encounters) PLUS 3-seed encoder variance PLUS 5-fold probe CV "
                    "on the readout step. Train, val, test_b, test_c reported for "
                    "every headline metric."
                ),
            },
        },
        "test_b_criteria": TEST_B_V2_CRITERIA,
        "test_b_cases": sorted(TEST_B_V2_CASE_IDS),
        "test_c_cases": sorted([cid for cid, c in cases_out.items() if c["split"] == "test_c"]),
        "summary": {
            "n_cases_total": len(cases_out),
            "n_cases_train": counts["train"],
            "n_cases_test_b": counts["test_b"],
            "n_cases_test_b_interior": n_interior,
            "n_cases_test_b_boundary": n_boundary,
            "n_cases_test_c": counts["test_c"],
            "n_cases_calibration_reference": counts["n_calibration_reference"],
            "n_encounters_train": enc_counts["train"],
            "n_encounters_val": enc_counts["val"],
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
    print("Test B v2 cases (10):")
    for cid in manifest["test_b_cases"]:
        c = cases_out[cid]
        print(f"  {cid}  src={c['source_group']:>9s}  n_enc={c['n_encounters_full']}"
              f"  nbrs_d2={c['n_train_neighbors_d2']}  tier={c['tier']}")
    print()
    print("Test C cases:")
    for cid in manifest["test_c_cases"]:
        c = cases_out[cid]
        print(f"  {cid}  src={c['source_group']}  n_enc={c['n_encounters_full']}")


if __name__ == "__main__":
    main()
