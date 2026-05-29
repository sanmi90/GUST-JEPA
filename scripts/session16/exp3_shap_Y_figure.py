"""Session 16 Exp 3-Y: figures for Y SHAP attribution.

Two figures:
1. Hero: 3 representative test_c encounters (one Y<0, one Y~0, one Y>0)
   showing omega input, baseline omega, and Y SHAP attribution. The
   attribution should localize differently for different Y values --
   evidence the encoder uses Y-sensitive pixel regions.
2. Mean attribution over the bootstrap-stable subset.
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
from matplotlib.patches import Polygon

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.data.omega_pipeline import OmegaPipeline  # noqa: E402


def load_airfoil_xy() -> np.ndarray:
    """Load NACA 0012 airfoil surface coordinates (x, y) in chord-normalized
    physical units from the raw Baseline file. Returns a closed-polygon (N, 2)
    array. The imshow panels in this figure use ``extent=(*X_EXTENT, *Y_EXTENT)``
    with ``aspect="equal"`` so the polygon can be drawn directly in physical
    (chord-normalized) coordinates without pixel conversion.
    """
    raw = Path(os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT"))) \
        / "data" / "raw" / "periodic" / "Baseline.h5"
    with h5py.File(raw, "r") as f:
        xy = np.asarray(f["airfoil_xy"])
    if not np.allclose(xy[0], xy[-1]):
        xy = np.vstack([xy, xy[0:1]])
    return xy


def _add_airfoil(ax, airfoil_xy: np.ndarray) -> None:
    """Overlay a black filled NACA 0012 polygon on a vorticity panel."""
    ax.add_patch(Polygon(airfoil_xy, closed=True, facecolor="black",
                         edgecolor="black", linewidth=0.7, zorder=10))

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


def pick_hero_by_y_value(shap, split: str) -> list[int]:
    """Pick 3 stable encounters with Y < 0, Y ~ 0, Y > 0."""
    Ys = shap[f"{split}_Y"]
    stable = shap[f"{split}_Y_bootstrap_stable"]
    stable_idx = np.where(stable)[0]
    picks = []
    for criterion in (lambda y: y < -0.1, lambda y: abs(y) < 0.05, lambda y: y > 0.1):
        cand = [i for i in stable_idx if criterion(Ys[i])]
        if cand:
            picks.append(int(cand[len(cand) // 2]))
        else:
            picks.append(int(stable_idx[0]))
    return picks


def load_encounter_omega_raw(case_id: str, k: int) -> np.ndarray:
    path = CACHE_ROOT / PARTITION / case_id / f"encounter_{int(k):02d}.h5"
    with h5py.File(path, "r") as f:
        return np.asarray(f["omega_z"], dtype=np.float32)


def main() -> None:
    shap = np.load(OUT / "shap_Y_attribution.npz", allow_pickle=True)
    pipeline = OmegaPipeline.from_manifest(OMEGA_MANIFEST)
    airfoil_xy = load_airfoil_xy()

    for split in ("test_b", "test_c"):
        hero = pick_hero_by_y_value(shap, split)
        fig, axes = plt.subplots(len(hero), 3, figsize=(12, 3 * len(hero)), squeeze=False)
        for row, hero_i in enumerate(hero):
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
            attr = shap[f"{split}_Y_attr"][hero_i]
            pred = float(shap[f"{split}_Y_pred"][hero_i])

            ax_o = axes[row, 0]
            ax_o.imshow(omega_at_impact.T, origin="lower", extent=(*X_EXTENT, *Y_EXTENT),
                         cmap="RdBu_r", norm=Normalize(vmin=-2, vmax=2), aspect="equal")
            _add_airfoil(ax_o, airfoil_xy)
            ax_o.set_title(f"omega (frame={impact})\n{case_id}, enc {k}\nG={G:+.1f} D={D:.1f} **Y={Y:+.2f}**", fontsize=9)
            ax_o.set_xticks([]); ax_o.set_yticks([])

            ax_base = axes[row, 1]
            base = shap["baseline_mean_per_frame"][min(impact, shap["baseline_mean_per_frame"].shape[0] - 1)]
            ax_base.imshow(base.T, origin="lower", extent=(*X_EXTENT, *Y_EXTENT),
                          cmap="RdBu_r", norm=Normalize(vmin=-2, vmax=2), aspect="equal")
            _add_airfoil(ax_base, airfoil_xy)
            ax_base.set_title(f"phase-matched G=0 baseline (frame={impact})", fontsize=9)
            ax_base.set_xticks([]); ax_base.set_yticks([])

            ax_attr = axes[row, 2]
            attr_lim = float(np.percentile(np.abs(attr), 99))
            ax_attr.imshow(attr.T, origin="lower", extent=(*X_EXTENT, *Y_EXTENT),
                            cmap="RdBu_r", norm=Normalize(vmin=-attr_lim, vmax=attr_lim), aspect="equal")
            _add_airfoil(ax_attr, airfoil_xy)
            ax_attr.set_title(f"Y SHAP attribution\n probe_pred_Y={pred:+.3f} (true {Y:+.2f})", fontsize=9)
            ax_attr.set_xticks([]); ax_attr.set_yticks([])
        plt.suptitle(f"Exp 3 Y SHAP {split}: 3 hero encounters spanning Y<0 / Y~0 / Y>0\n"
                     f"Asymmetric pixel localisation = evidence the encoder's Y-encoding has a structural footprint.",
                     fontsize=10)
        plt.tight_layout(rect=(0, 0, 1, 0.94))
        fig_path = FIG_DIR / f"exp3_shap_Y_hero_{split}.png"
        fig.savefig(fig_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[exp3-Y-fig] wrote {fig_path.relative_to(REPO)}")

    # Mean attribution over stable subset
    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4))
    for col, split in enumerate(("test_b", "test_c")):
        attrs = shap[f"{split}_Y_attr"]
        stable = shap[f"{split}_Y_bootstrap_stable"]
        kept = attrs[stable] if stable.any() else attrs
        mean_attr = np.mean(np.abs(kept), axis=0) if len(kept) > 0 else np.zeros_like(attrs[0])
        attr_lim = float(np.percentile(mean_attr, 99)) if mean_attr.max() > 0 else 1.0
        ax = axes2[col]
        ax.imshow(mean_attr.T, origin="lower", extent=(*X_EXTENT, *Y_EXTENT),
                   cmap="hot", norm=Normalize(vmin=0, vmax=attr_lim), aspect="equal")
        _add_airfoil(ax, airfoil_xy)
        ax.set_title(f"Mean |Y SHAP attribution| | {split}\n(n_stable = {len(kept)}/{len(attrs)})", fontsize=10)
        ax.set_xlabel("x / chord"); ax.set_ylabel("y / chord")
    plt.suptitle("Exp 3 Y SHAP: mean attribution over bootstrap-stable subset", fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    fig_path = FIG_DIR / "exp3_shap_Y_mean.png"
    fig2.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig2)
    print(f"[exp3-Y-fig] wrote {fig_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
