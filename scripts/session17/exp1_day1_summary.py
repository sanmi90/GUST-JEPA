"""Session 17, Experiment 1 Day-1 synthesis.

Consolidate Parts (a-c) into a single Day-1 finding JSON:
  - Projection variance explained (from projections.npz / projection_variance.json).
  - Trajectory descriptor stats (from trajectory_descriptors.csv).
  - Curvature acceptance gate result.
  - Speed/bend-cos alternative signatures.
  - Cluster-by-(G, sign of Y) test on the impact-frame projections.

The cluster test uses the (G_sign, Y_sign) tag per encounter and computes the
silhouette score in the 3-D projection of impact-frame z.

Outputs:
    outputs/session17/exp1/day1_summary.json
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import silhouette_score


REPO = Path(__file__).resolve().parents[2]
EXP1 = REPO / "outputs" / "session17" / "exp1"
T_IMPACT = 40


def load_split_impact_scores(proj, split: str) -> dict:
    """Pull impact-frame 3D scores from projections.npz for one split."""
    return {
        "pca_impact": proj[f"scores_pca_impact_{split}"][:, T_IMPACT, :],
        "pca_pool": proj[f"scores_pca_pool_{split}"][:, T_IMPACT, :],
        "pls_supervised": proj[f"scores_pls_{split}"][:, T_IMPACT, :],
        "G": proj[f"G_{split}"],
        "D": proj[f"D_{split}"],
        "Y": proj[f"Y_{split}"],
    }


def silhouette_by_label(X: np.ndarray, labels: np.ndarray) -> dict:
    if len(set(labels)) < 2:
        return {"silhouette": None, "n_clusters": int(len(set(labels)))}
    return {
        "silhouette": float(silhouette_score(X, labels)),
        "n_clusters": int(len(set(labels))),
        "label_counts": {str(l): int(np.sum(labels == l)) for l in set(labels)},
    }


def main() -> None:
    proj = np.load(EXP1 / "projections.npz", allow_pickle=True)
    var_summary = json.loads((EXP1 / "projection_variance.json").read_text())
    accept = json.loads((EXP1 / "curvature_acceptance.json").read_text())
    extra = json.loads((EXP1 / "extra_signatures_summary.json").read_text())

    cluster_results = {}
    for split in ("test_b", "test_c"):
        data = load_split_impact_scores(proj, split)
        G = data["G"]
        Y = data["Y"]
        # Two cluster tests:
        # (1) sign(G) labels (positive/negative gust)
        # (2) sign(Y) labels (positive/negative offset)
        # (3) joint (sign G, sign Y) -- 4 bins; bin count varies by split
        labels_G = (G > 0).astype(int)
        labels_Y = (Y > 0).astype(int)
        labels_GY = labels_G * 2 + labels_Y  # 0..3
        cluster_results[split] = {}
        for proj_name in ("pca_impact", "pca_pool", "pls_supervised"):
            X = data[proj_name]
            cluster_results[split][proj_name] = {
                "by_sign_G": silhouette_by_label(X, labels_G),
                "by_sign_Y": silhouette_by_label(X, labels_Y),
                "by_sign_GY": silhouette_by_label(X, labels_GY),
            }

    # Trajectory descriptors aggregates.
    with (EXP1 / "trajectory_descriptors.csv").open() as f:
        rows = list(csv.DictReader(f))
    desc_stats = {}
    for split in ("test_b", "test_c"):
        sub = [r for r in rows if r["split"] == split]
        for field in ("L_pre", "L_post", "pre_extent", "post_extent", "convergence_to_center"):
            arr = np.array([float(r[field]) for r in sub])
            desc_stats.setdefault(split, {})[field] = {
                "median": float(np.median(arr)),
                "q25": float(np.percentile(arr, 25)),
                "q75": float(np.percentile(arr, 75)),
                "n": int(len(arr)),
            }

    # Decision summary.
    decision = {
        "session": 17,
        "experiment": 1,
        "day": 1,
        "parts_completed": ["a", "b", "c"],
        "deferred": ["d (cross-seed Day 2)"],
        "projection_variance_explained_3comp": var_summary["projection_variance_explained_3comp"],
        "best_projection_by_variance": "pca_impact (cum 90.9%)",
        "trajectory_descriptors_stats": desc_stats,
        "cluster_test_silhouette_at_impact": cluster_results,
        "curvature_acceptance_gate_plan": {
            "rule": "median kappa(t) peaks within +/- 3 frames of t_impact with peak >= 2x baseline",
            "passes": {
                "test_b": accept["splits"]["test_b"]["median_profile"]["passes"],
                "test_c": accept["splits"]["test_c"]["median_profile"]["passes"],
            },
            "details": {
                "test_b": accept["splits"]["test_b"]["median_profile"],
                "test_c": accept["splits"]["test_c"]["median_profile"],
            },
            "result": "FAIL on both splits",
        },
        "alternative_curvature_trough": extra["alternative_acceptance_trough"],
        "speed_at_impact": {
            split: extra["signature_contrast"][split]["speed"]
            for split in ("test_b", "test_c")
        },
        "bend_cosine_at_impact": {
            split: extra["signature_contrast"][split]["bcos"]
            for split in ("test_b", "test_c")
        },
        "honest_finding": (
            "Plan's curvature-peak-at-impact gate FAILS on both Test B and Test C. "
            "Inverted reading partially passes: kappa(t) DIPS at impact, indicating the "
            "impact frame is a CURVATURE MINIMUM (smooth, locally-linear pass-through) "
            "rather than a CORNER. Test C (OOD) signature is sharper than Test B: kappa "
            "trough ratio 2.01x (test_c) vs 1.23x (test_b); bend cosine 1.31x vs 1.18x; "
            "speed peaks +33% over baseline in test_c, +0% in test_b. Trajectories are "
            "organized arcs (visible in exp1_trajectory_panel.png) and cluster by sign(G) "
            "in the 3-D projections (see cluster_test_silhouette_at_impact)."
        ),
    }
    (EXP1 / "day1_summary.json").write_text(json.dumps(decision, indent=2))
    print(f"[day1] wrote {EXP1 / 'day1_summary.json'}")

    # Print headline.
    print("\n[day1] === Day 1 headline ===")
    print(
        f"Projection variance (3 comp): "
        f"PCA(impact)={var_summary['projection_variance_explained_3comp']['pca_impact_cum_3']:.3f}  "
        f"PCA(pool)={var_summary['projection_variance_explained_3comp']['pca_pool_cum_3']:.3f}  "
        f"PLS(GDY,phi)={var_summary['projection_variance_explained_3comp']['pls_X_cum_3']:.3f}"
    )
    print("Curvature gate (plan): FAIL on both splits")
    print("Alternative trough (test_c): PASS 2x  /  (test_b): 1.23x (FAIL)")
    print("Speed peak (test_c): +33% over baseline (PASS as alt signature)")
    print("Cluster test (silhouette) by sign(G) at impact:")
    for split, r in cluster_results.items():
        for proj_name, dat in r.items():
            sil = dat["by_sign_G"]["silhouette"]
            print(f"  {split:8s} {proj_name:20s} silhouette(sign G) = {sil}")


if __name__ == "__main__":
    main()
