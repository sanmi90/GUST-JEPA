"""Tests for SESSION 31 Track 0.A data certification helpers.

The cache-walking CLI is integration; these pin the pure check logic that
decides pass/fail (split disjointness, normalisation tolerance, per-encounter
frame alignment, pressure alignment).
"""

from src.data.verify import (
    check_split_disjoint,
    check_normalisation,
    check_encounter_alignment,
    check_pressure_alignment,
)


def _manifest():
    # train cases carry a within-case val holdout; test_b/test_c are case-disjoint.
    cases = {
        "A": {
            "split": "train",
            "n_encounters_full": 6,
            "train_encounter_indices": [0, 1, 2, 3],
            "val_encounter_indices": [4, 5],
            "G": 1.0,
            "D": 0.5,
            "Y": 0.1,
        },
        "B": {
            "split": "train",
            "n_encounters_full": 4,
            "train_encounter_indices": [0, 1],
            "val_encounter_indices": [2, 3],
            "G": -1.0,
            "D": 1.0,
            "Y": -0.1,
        },
        "C": {"split": "test_b", "n_encounters_full": 6, "G": 2.0, "D": 0.5, "Y": 0.1},
        "D": {"split": "test_c", "n_encounters_full": 6, "G": 4.0, "D": 1.0, "Y": 0.1},
    }
    return {"cases": cases, "test_b_cases": ["C"], "test_c_cases": ["D"]}


def test_split_disjoint_passes_for_consistent_partition():
    res = check_split_disjoint(_manifest())
    assert res["ok"] is True
    assert res["overlaps"]["train|test_b"] == []
    assert res["overlaps"]["test_b|test_c"] == []
    assert res["n_per_split"] == {"train": 2, "test_b": 1, "test_c": 1}
    # val is an intended within-train-case encounter holdout, surfaced not failed
    assert res["val_within_train_cases"] == ["A", "B"]


def test_split_disjoint_fails_when_top_level_list_disagrees_with_split_field():
    m = _manifest()
    m["test_b_cases"] = []  # C is split==test_b but missing from the top-level list
    res = check_split_disjoint(m)
    assert res["ok"] is False
    assert any("test_b" in r for r in res["reasons"])


def test_normalisation_ok_within_tolerance():
    res = check_normalisation(
        train_std=3.539555, ssim_L=8.48680, expected_std=3.5396, expected_L=8.487, tol=1e-3
    )
    assert res["ok"] is True


def test_normalisation_fails_when_std_drifts():
    res = check_normalisation(
        train_std=3.60, ssim_L=8.487, expected_std=3.5396, expected_L=8.487, tol=1e-3
    )
    assert res["ok"] is False
    assert "train_std" in res["reasons"][0]


def test_encounter_alignment_passes_when_frames_match():
    shapes = {"omega_z": (120, 192, 96), "p_wall": (120, 192), "C_L": (120,), "C_D": (120,)}
    attrs = {"case_id": "A", "G": 1.0, "D": 0.5, "Y": 0.1}
    res = check_encounter_alignment(
        shapes,
        attrs,
        expected={"case_id": "A", "G": 1.0, "D": 0.5, "Y": 0.1},
        n_frames_expected=120,
    )
    assert res["ok"] is True


def test_encounter_alignment_fails_on_frame_mismatch():
    shapes = {
        "omega_z": (120, 192, 96),
        "p_wall": (119, 192),  # off by one
        "C_L": (120,),
        "C_D": (120,),
    }
    attrs = {"case_id": "A", "G": 1.0, "D": 0.5, "Y": 0.1}
    res = check_encounter_alignment(
        shapes,
        attrs,
        expected={"case_id": "A", "G": 1.0, "D": 0.5, "Y": 0.1},
        n_frames_expected=120,
    )
    assert res["ok"] is False
    assert any("p_wall" in r for r in res["reasons"])


def test_encounter_alignment_fails_on_param_mismatch():
    shapes = {"omega_z": (120, 192, 96), "p_wall": (120, 192), "C_L": (120,), "C_D": (120,)}
    attrs = {"case_id": "A", "G": 1.0, "D": 0.5, "Y": 0.1}
    res = check_encounter_alignment(
        shapes,
        attrs,
        expected={"case_id": "A", "G": 2.0, "D": 0.5, "Y": 0.1},
        n_frames_expected=120,
    )
    assert res["ok"] is False
    assert any("G" in r for r in res["reasons"])


def test_pressure_alignment_passes_for_canonical_shape():
    assert check_pressure_alignment((120, 192), n_frames=120, n_surface_expected=192)["ok"] is True


def test_pressure_alignment_fails_on_wrong_surface_count():
    res = check_pressure_alignment((120, 96), n_frames=120, n_surface_expected=192)
    assert res["ok"] is False
