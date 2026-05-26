"""Session 16 Exp 3: improved hero figure showing the ACTUAL omega input for
each hero encounter (loaded from cache) plus the baseline omega and the SHAP
attribution. Three hero encounters with very different (G, D, Y) cases.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.data.omega_pipeline import OmegaPipeline  # noqa: E402

OUT = REPO / "outputs" / "session16" / "exp3"
FIG_DIR = REPO / "outputs" / "session16" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OMEGA_MANIFEST = REPO / "outputs" / "data_pipeline" / "v1" / "manifest.json"
PARTITION = "v1"
CACHE_ROOT = Path(
    os.environ.get(
        "VORTEX_JEPA_CACHE",
        str(Path(os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT")))
            / "data" / "processed" / "vortex-jepa"),
    )
)

X_EXTENT = (-1.5, 4.5)
Y_EXTENT = (-1.5, 1.5)


def pick_hero_indices(shap, split: str) -> list[int]:
    """Pick 3 hero indices: low-|G|, mid-|G|, high-|G| (or representative spread)."""
    Gs = shap[f"{split}_G"]
    sorted_idx = np.argsort(np.abs(Gs))
    picks = [sorted_idx[0], sorted_idx[len(sorted_idx) // 2], sorted_idx[-1]]
    return [int(i) for i in picks]


def load_encounter_omega_raw(case_id: str, k: int) -> np.ndarray:
    path = CACHE_ROOT / PARTITION / case_id / f"encounter_{int(k):02d}.h5"
    with h5py.File(path, "r") as f:
        return np.asarray(f["omega_z"], dtype=np.float32)


def main() -> None:
    shap = np.load(OUT / "shap_attribution.npz", allow_pickle=True)
    pipeline = OmegaPipeline.from_manifest(OMEGA_MANIFEST)
    targets = ["centroid_x", "circulation_pos", "peak_neg_omega"]

    for split in ("test_b", "test_c"):
        hero_idx = pick_hero_indices(shap, split)
        n_heroes = len(hero_idx)
        fig, axes = plt.subplots(
            n_heroes * len(targets), 3, figsize=(11, 2.4 * n_heroes * len(targets)), squeeze=False
        )
        row = 0
        for hero_i in hero_idx:
            impact = int(shap[f"{split}_impact_frame"][hero_i])
            G = float(shap[f"{split}_G"][hero_i])
            D = float(shap[f"{split}_D"][hero_i])
            Y = float(shap[f"{split}_Y"][hero_i])
            case_id = str(shap[f"{split}_case_id"][hero_i])
            k = int(shap[f"{split}_encounter_index"][hero_i])
            omega_raw = load_encounter_omega_raw(case_id, k)
            omega_clean = pipeline.preprocess_raw(omega_raw, case_id, k)
            omega_norm = pipeline.normalize(omega_clean).astype(np.float32)
            omega_at_impact = omega_norm[min(impact, omega_norm.shape[0] - 1)]
            baseline_at_impact = shap["baseline_mean_per_frame"][min(impact, shap["baseline_mean_per_frame"].shape[0] - 1)]
            for target_name in targets:
                attr = shap[f"{split}_{target_name}_attr"][hero_i]
                pred = float(shap[f"{split}_{target_name}_pred"][hero_i])
                pred_b = float(shap[f"{split}_{target_name}_pred_baseline"][hero_i])

                ax_omega = axes[row, 0]
                ax_omega.imshow(
                    omega_at_impact.T, origin="lower", extent=(*X_EXTENT, *Y_EXTENT),
                    cmap="RdBu_r", norm=Normalize(vmin=-2, vmax=2), aspect="equal",
                )
                ax_omega.set_title(
                    f"omega (normalised, frame={impact})\n"
                    f"{case_id}, enc {k}, G={G:+.1f} D={D:.1f} Y={Y:+.2f}",
                    fontsize=8,
                )
                ax_omega.set_ylabel(f"target = {target_name}", fontsize=8)
                ax_omega.set_xticks([])
                ax_omega.set_yticks([])

                ax_base = axes[row, 1]
                ax_base.imshow(
                    baseline_at_impact.T, origin="lower", extent=(*X_EXTENT, *Y_EXTENT),
                    cmap="RdBu_r", norm=Normalize(vmin=-2, vmax=2), aspect="equal",
                )
                ax_base.set_title(
                    f"phase-matched G=0 baseline (frame={impact})\n"
                    f"baseline_pred={pred_b:.3f}", fontsize=8,
                )
                ax_base.set_xticks([])
                ax_base.set_yticks([])

                ax_attr = axes[row, 2]
                attr_lim = np.percentile(np.abs(attr), 99)
                ax_attr.imshow(
                    attr.T, origin="lower", extent=(*X_EXTENT, *Y_EXTENT),
                    cmap="RdBu_r", norm=Normalize(vmin=-attr_lim, vmax=attr_lim), aspect="equal",
                )
                ax_attr.set_title(
                    f"SHAP attr (signed, 99th pct = {attr_lim:.4f})\n"
                    f"actual_pred={pred:.3f}", fontsize=8,
                )
                ax_attr.set_xticks([])
                ax_attr.set_yticks([])
                row += 1
        plt.suptitle(
            f"Exp 3: gradient-SHAP attribution on {split} (3 hero encounters x 3 targets)\n"
            "columns: actual omega @ impact | phase-matched G=0 baseline @ impact | SHAP attribution",
            fontsize=10,
        )
        plt.tight_layout(rect=(0, 0, 1, 0.97))
        fig_path = FIG_DIR / f"exp3_shap_hero_{split}.png"
        fig.savefig(fig_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[exp3-fig-v2] wrote {fig_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
