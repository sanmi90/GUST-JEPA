"""Session 17, Experiment 1, Part (b): trajectory visualization and descriptors.

Reads the three projections produced by exp1a_projections.py and:
  1. Selects 10 representative Test B encounters (5 G>0 + 5 G<0).
  2. Plots each encounter's 3D trajectory in each projection, colored by
     impact-relative phase, with the impact frame marked.
  3. Computes per-encounter trajectory descriptors in the *full 64-D z space*:
       L_pre, L_post, pre-impact extent, post-impact extent,
       convergence to the train z-center for the nearest (G, D, Y) bin.

Outputs:
    outputs/session17/exp1/trajectory_descriptors.csv
    outputs/session17/exp1/representative_encounters.json
    outputs/session17/figures/exp1_trajectory_panel.png
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


REPO = Path(__file__).resolve().parents[2]
LATENTS = REPO / "outputs" / "session14" / "latents" / "S12_E_d64"
EXP1 = REPO / "outputs" / "session17" / "exp1"
FIGS = REPO / "outputs" / "session17" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)
T_IMPACT = 40
T_ENC = 120


# Hand-picked 10 representative Test B encounters (5 G>0, 5 G<0).
# Indices into the v1p5 test_b latent block (28 v1 + 28 supplement, in load order).
REP = {
    "G>0": [
        {"idx": 0,  "label": "G+1.00 D1.00 Y+0.10"},
        {"idx": 20, "label": "G+0.50 D1.00 Y+0.20"},
        {"idx": 36, "label": "G+2.00 D1.00 Y+0.20"},
        {"idx": 44, "label": "G+3.00 D1.00 Y-0.20"},
        {"idx": 48, "label": "G+3.00 D1.50 Y-0.10"},
    ],
    "G<0": [
        {"idx": 12, "label": "G-1.50 D0.50 Y-0.20"},
        {"idx": 16, "label": "G-0.50 D1.00 Y+0.00"},
        {"idx": 52, "label": "G-2.00 D1.00 Y+0.00"},
        {"idx": 13, "label": "G-1.50 D0.50 Y-0.20  enc1"},
        {"idx": 53, "label": "G-2.00 D1.00 Y+0.00  enc1"},
    ],
}


def load_z_full(split: str, supplement: str | None = None) -> dict:
    d = np.load(LATENTS / f"{split}.npz", allow_pickle=True)
    out = {
        "z": d["z"].astype(np.float32),
        "z_full": d["z_full"].astype(np.float32),
        "G": d["G"].astype(np.float32),
        "D": d["D"].astype(np.float32),
        "Y": d["Y"].astype(np.float32),
        "case_id": np.asarray(d["case_id"]).astype(object),
        "encounter_index": d["encounter_index"].astype(np.int32),
        "impact_frame": d["impact_frame"].astype(np.int32),
    }
    if supplement is not None:
        ds = np.load(LATENTS / f"{supplement}.npz", allow_pickle=True)
        out["z_full"] = np.concatenate(
            [out["z_full"], ds["z_full"].astype(np.float32)], axis=0
        )
        out["z"] = np.concatenate([out["z"], ds["z"].astype(np.float32)], axis=0)
        for k in ("G", "D", "Y"):
            out[k] = np.concatenate([out[k], ds[k].astype(np.float32)], axis=0)
        out["case_id"] = np.concatenate(
            [out["case_id"], np.asarray(ds["case_id"]).astype(object)], axis=0
        )
        out["encounter_index"] = np.concatenate(
            [out["encounter_index"], ds["encounter_index"].astype(np.int32)], axis=0
        )
        out["impact_frame"] = np.concatenate(
            [out["impact_frame"], ds["impact_frame"].astype(np.int32)], axis=0
        )
    return out


def train_z_center(train: dict, target_GDY: tuple[float, float, float]) -> np.ndarray:
    """Return the mean impact-frame z for the nearest (G,D,Y) bin in train."""
    G, D, Y = train["G"], train["D"], train["Y"]
    g, d, y = target_GDY
    # Distance in (G, D, Y) -- scaled so units are comparable (Y has small range).
    dist2 = (
        ((G - g) / 1.0) ** 2
        + ((D - d) / 0.5) ** 2
        + ((Y - y) / 0.1) ** 2
    )
    # Find the nearest training case (use case-mean to denoise).
    nearest_idx = int(np.argmin(dist2))
    case_id_match = train["case_id"][nearest_idx]
    mask = train["case_id"] == case_id_match
    z_imp_mean = train["z"][mask].mean(axis=0)
    return z_imp_mean, case_id_match, float(np.sqrt(dist2[nearest_idx]))


def compute_descriptors(zt: np.ndarray, t_imp: int) -> dict:
    """zt: (T, d) per-frame latent for one encounter."""
    diff = np.diff(zt, axis=0)
    step = np.linalg.norm(diff, axis=1)  # (T-1,)
    pre = step[:t_imp].sum()
    post = step[t_imp:].sum()
    # extents
    pre_ext = float(np.linalg.norm(zt[:t_imp] - zt[0], axis=1).max() if t_imp > 0 else 0)
    post_ext = float(np.linalg.norm(zt[t_imp:] - zt[t_imp], axis=1).max())
    return {
        "L_pre": float(pre),
        "L_post": float(post),
        "pre_extent": pre_ext,
        "post_extent": post_ext,
    }


def main() -> None:
    proj = np.load(EXP1 / "projections.npz", allow_pickle=True)
    train = load_z_full("train")
    test_b = load_z_full("test_b", supplement="test_b_v1p5_supplement")
    print(f"[exp1b] train z_full {train['z_full'].shape}  test_b {test_b['z_full'].shape}")

    rep_flat = []
    for sign, items in REP.items():
        for r in items:
            rep_flat.append({**r, "sign": sign})

    # Per-encounter descriptors.
    desc_rows = []
    for r in rep_flat:
        idx = r["idx"]
        zt = test_b["z_full"][idx]  # (T, 64)
        t_imp = int(test_b["impact_frame"][idx])
        g, d, y = (
            float(test_b["G"][idx]),
            float(test_b["D"][idx]),
            float(test_b["Y"][idx]),
        )
        descriptors = compute_descriptors(zt, t_imp)
        z_center, case_match, dist = train_z_center(train, (g, d, y))
        conv = float(np.linalg.norm(zt[t_imp] - z_center))
        desc_rows.append(
            {
                "split": "test_b",
                "idx": idx,
                "case_id": str(test_b["case_id"][idx]),
                "encounter_index": int(test_b["encounter_index"][idx]),
                "G": g,
                "D": d,
                "Y": y,
                "sign": r["sign"],
                "label": r["label"],
                "nearest_train_case": str(case_match),
                "nearest_train_dist": dist,
                "L_pre": descriptors["L_pre"],
                "L_post": descriptors["L_post"],
                "pre_extent": descriptors["pre_extent"],
                "post_extent": descriptors["post_extent"],
                "convergence_to_center": conv,
            }
        )

    # Compute descriptors for ALL Test B and Test C encounters (for context).
    for split_name, split in [
        ("test_b", test_b),
        ("test_c", load_z_full("test_c")),
    ]:
        for i in range(split["z_full"].shape[0]):
            zt = split["z_full"][i]
            t_imp = int(split["impact_frame"][i])
            g, d, y = (
                float(split["G"][i]),
                float(split["D"][i]),
                float(split["Y"][i]),
            )
            descriptors = compute_descriptors(zt, t_imp)
            z_center, case_match, dist = train_z_center(train, (g, d, y))
            conv = float(np.linalg.norm(zt[t_imp] - z_center))
            # Only add if not already in rep_flat (avoid duplicates).
            if split_name == "test_b" and any(
                row["idx"] == i and row["split"] == "test_b" for row in desc_rows
            ):
                continue
            desc_rows.append(
                {
                    "split": split_name,
                    "idx": i,
                    "case_id": str(split["case_id"][i]),
                    "encounter_index": int(split["encounter_index"][i]),
                    "G": g,
                    "D": d,
                    "Y": y,
                    "sign": "G>0" if g > 0 else "G<0",
                    "label": "",
                    "nearest_train_case": str(case_match),
                    "nearest_train_dist": dist,
                    "L_pre": descriptors["L_pre"],
                    "L_post": descriptors["L_post"],
                    "pre_extent": descriptors["pre_extent"],
                    "post_extent": descriptors["post_extent"],
                    "convergence_to_center": conv,
                }
            )

    # Save CSV.
    csv_path = EXP1 / "trajectory_descriptors.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(desc_rows[0].keys()))
        writer.writeheader()
        writer.writerows(desc_rows)
    print(f"[exp1b] wrote {csv_path}")

    rep_idx = [r["idx"] for r in rep_flat]
    rep_labels = [r["label"] for r in rep_flat]
    rep_signs = [r["sign"] for r in rep_flat]
    print(f"[exp1b] representative encounters: {rep_idx}")

    # Figure: 1 row x 3 cols (one per projection), each a 3D scatter+line.
    proj_keys = [
        ("scores_pca_impact_test_b", "PCA(impact)"),
        ("scores_pca_pool_test_b", "PCA(pool)"),
        ("scores_pls_test_b", "PLS(GDY,phase)"),
    ]
    fig = plt.figure(figsize=(18, 6))
    cmap = plt.get_cmap("RdBu_r")
    colors_by_idx = {}
    # color per encounter by G sign and magnitude
    for r in rep_flat:
        g = float(test_b["G"][r["idx"]])
        # Map G in [-3, +3] -> [0, 1]
        colors_by_idx[r["idx"]] = cmap((g + 3.0) / 6.0)

    for pi, (key, title) in enumerate(proj_keys):
        ax = fig.add_subplot(1, 3, pi + 1, projection="3d")
        scores = proj[key]  # (n_enc, T, 3)
        for r in rep_flat:
            idx = r["idx"]
            traj = scores[idx]  # (T, 3)
            color = colors_by_idx[idx]
            ls = "-" if r["sign"] == "G>0" else "--"
            ax.plot(
                traj[:, 0], traj[:, 1], traj[:, 2],
                ls, color=color, lw=1.0, alpha=0.7,
            )
            # phase coloring as scatter overlay
            phase = (np.arange(T_ENC) - T_IMPACT) / 40.0
            ax.scatter(
                traj[:, 0], traj[:, 1], traj[:, 2],
                c=phase, cmap="viridis", s=4, alpha=0.6,
            )
            # mark impact
            ax.scatter(
                [traj[T_IMPACT, 0]],
                [traj[T_IMPACT, 1]],
                [traj[T_IMPACT, 2]],
                marker="*",
                s=120,
                edgecolor="k",
                facecolor=color,
                linewidth=0.8,
                zorder=10,
            )
        ax.set_title(f"{title}\n(10 Test B encounters)")
        ax.set_xlabel("comp 1")
        ax.set_ylabel("comp 2")
        ax.set_zlabel("comp 3")

    # Add a legend for line style (G sign) and a colorbar.
    # Custom legend with line style
    handles = [
        plt.Line2D([], [], color="k", lw=1.2, ls="-", label="G > 0"),
        plt.Line2D([], [], color="k", lw=1.2, ls="--", label="G < 0"),
        plt.Line2D(
            [], [], color="k", marker="*", linestyle="None", markersize=10,
            markerfacecolor="lightgray", label="impact frame",
        ),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False)

    fig.suptitle("Latent trajectories of 10 representative Test B encounters", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS / "exp1_trajectory_panel.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[exp1b] wrote {FIGS / 'exp1_trajectory_panel.png'}")

    # Save representative-encounter metadata.
    rep_path = EXP1 / "representative_encounters.json"
    rep_path.write_text(json.dumps(rep_flat, indent=2))
    print(f"[exp1b] wrote {rep_path}")

    # Quick summary stats for the representative encounters.
    print("\n[exp1b] representative-encounter descriptors:")
    print(
        f"{'idx':>4} {'sign':>5} {'G':>5} {'D':>4} {'Y':>5}  "
        f"{'L_pre':>7} {'L_post':>7}  {'pre_ext':>7} {'post_ext':>7}  conv"
    )
    for row in desc_rows:
        if not row["label"]:
            continue
        print(
            f"{row['idx']:>4} {row['sign']:>5} "
            f"{row['G']:>+5.2f} {row['D']:>4.2f} {row['Y']:>+5.2f}  "
            f"{row['L_pre']:>7.2f} {row['L_post']:>7.2f}  "
            f"{row['pre_extent']:>7.2f} {row['post_extent']:>7.2f}  "
            f"{row['convergence_to_center']:>5.2f}"
        )


if __name__ == "__main__":
    main()
