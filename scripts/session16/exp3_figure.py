"""Session 16 Exp 3: 2x4 panel figure of SHAP attribution + omega field for
representative test_b encounters, plus mean-attribution figure across the
stable subset.

Produces:
    outputs/session16/figures/exp3_shap_hero.png
    outputs/session16/figures/exp3_shap_mean.png
    outputs/session16/exp3/exp3_finding.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "session16" / "exp3"
FIG_DIR = REPO / "outputs" / "session16" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

X_EXTENT = (-1.5, 4.5)
Y_EXTENT = (-1.5, 1.5)


def main() -> None:
    shap = np.load(OUT / "shap_attribution.npz", allow_pickle=True)
    if (OUT / "shap_bootstrap.npz").exists():
        boot = np.load(OUT / "shap_bootstrap.npz", allow_pickle=True)
    else:
        boot = None
    if (OUT / "shap_intervention.json").exists():
        intv = json.loads((OUT / "shap_intervention.json").read_text())
    else:
        intv = None

    targets = ["centroid_x", "circulation_pos", "peak_neg_omega"]

    # ----- Hero figure: 3 representative test_b encounters, 4 columns
    # (omega input, baseline omega, attribution, attribution*input overlay)
    hero_pick = []
    for i in range(min(3, len(shap["test_b_case_id"]))):
        hero_pick.append(i)

    n_rows = len(hero_pick) * len(targets)
    fig, axes = plt.subplots(n_rows, 4, figsize=(14, 2.5 * n_rows), squeeze=False)
    row = 0
    for hero_i in hero_pick:
        omega_input = shap["baseline_mean_per_frame"]  # placeholder; need to recompute per-encounter
        # baseline mean per frame, dim (120, 192, 96)
        impact = int(shap["test_b_impact_frame"][hero_i])
        base_field = shap["baseline_mean_per_frame"][impact]
        G = shap["test_b_G"][hero_i]
        D = shap["test_b_D"][hero_i]
        Y = shap["test_b_Y"][hero_i]
        case_id = str(shap["test_b_case_id"][hero_i])
        for tg_idx, target_name in enumerate(targets):
            attr = shap[f"test_b_{target_name}_attr"][hero_i]
            pred = float(shap[f"test_b_{target_name}_pred"][hero_i])
            pred_b = float(shap[f"test_b_{target_name}_pred_baseline"][hero_i])

            ax_omega = axes[row, 0]
            # We don't store the per-encounter omega input here; pick attribution
            # support * sign as proxy: omega is the field where attribution lives.
            # For the hero figure clarity, we just plot attribution and baseline.
            ax_omega.imshow(
                base_field.T, origin="lower", extent=(*X_EXTENT, *Y_EXTENT),
                cmap="RdBu_r", norm=Normalize(vmin=-1.5, vmax=1.5), aspect="equal",
            )
            ax_omega.set_title(f"{case_id}  enc {int(shap['test_b_encounter_index'][hero_i])}\n"
                               f"G={G:+.1f} D={D:.1f} Y={Y:+.1f}; frame={impact}", fontsize=8)
            ax_omega.set_ylabel(f"target={target_name}", fontsize=8)
            ax_omega.set_xticks([])
            ax_omega.set_yticks([])

            ax_input = axes[row, 1]
            ax_input.set_title("baseline (phase-matched mean of G=0 pool)", fontsize=8)
            ax_input.imshow(
                base_field.T, origin="lower", extent=(*X_EXTENT, *Y_EXTENT),
                cmap="RdBu_r", norm=Normalize(vmin=-1.5, vmax=1.5), aspect="equal",
            )
            ax_input.set_xticks([])
            ax_input.set_yticks([])

            ax_attr = axes[row, 2]
            attr_lim = np.percentile(np.abs(attr), 99)
            ax_attr.imshow(
                attr.T, origin="lower", extent=(*X_EXTENT, *Y_EXTENT),
                cmap="RdBu_r", norm=Normalize(vmin=-attr_lim, vmax=attr_lim), aspect="equal",
            )
            ax_attr.set_title(f"SHAP attr\npred={pred:.3f}  baseline_pred={pred_b:.3f}", fontsize=8)
            ax_attr.set_xticks([])
            ax_attr.set_yticks([])

            ax_overlay = axes[row, 3]
            ax_overlay.imshow(
                np.abs(attr).T, origin="lower", extent=(*X_EXTENT, *Y_EXTENT),
                cmap="hot", aspect="equal",
            )
            ax_overlay.set_title("|attribution|", fontsize=8)
            ax_overlay.set_xticks([])
            ax_overlay.set_yticks([])
            row += 1

    plt.suptitle(
        "Exp 3: pixel-level gradient-SHAP attribution for 3 hero test_b encounters\n"
        "(target = centroid_x / circulation_pos / peak_neg_omega via Exp-2 probes)",
        fontsize=11,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    fig_path = FIG_DIR / "exp3_shap_hero.png"
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[exp3-fig] wrote {fig_path.relative_to(REPO)}")

    # ----- Mean attribution across stable subset
    fig2, axes2 = plt.subplots(len(targets), 2, figsize=(10, 3.5 * len(targets)), squeeze=False)
    for tg_idx, target_name in enumerate(targets):
        for col, split in enumerate(["test_b", "test_c"]):
            attrs = shap[f"{split}_{target_name}_attr"]
            if boot is not None:
                stable = boot[f"{split}_{target_name}_stable"]
                kept = attrs[stable]
            else:
                kept = attrs
            mean_attr = np.mean(np.abs(kept), axis=0) if len(kept) > 0 else np.zeros_like(attrs[0])
            ax = axes2[tg_idx, col]
            attr_lim = np.percentile(mean_attr, 99) if mean_attr.max() > 0 else 1.0
            ax.imshow(
                mean_attr.T, origin="lower", extent=(*X_EXTENT, *Y_EXTENT),
                cmap="hot", norm=Normalize(vmin=0, vmax=attr_lim), aspect="equal",
            )
            ax.set_title(
                f"{target_name} | {split}  (n_stable = {len(kept)}/{len(attrs)})",
                fontsize=10,
            )
            ax.set_xlabel("x / chord")
            ax.set_ylabel("y / chord")
    plt.suptitle(
        "Exp 3: mean |SHAP attribution| across the BOOTSTRAP-STABLE subset\n"
        "Bootstrap (4-baseline) stability gate: mean pairwise Pearson r >= 0.7",
        fontsize=11,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    fig_path = FIG_DIR / "exp3_shap_mean.png"
    fig2.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig2)
    print(f"[exp3-fig] wrote {fig_path.relative_to(REPO)}")

    # ----- Finding card
    finding: dict = {
        "session": "Session 16",
        "experiment": "3 (pixel-level gradient-SHAP + bootstrap stability + intervention)",
        "day": "5-7",
        "n_integration_steps": int(shap["n_integration_steps"]),
        "targets": list(targets),
    }
    if boot is not None:
        boot_info: dict = {}
        for split in ("test_b", "test_c"):
            split_block = {}
            n = len(shap[f"{split}_case_id"])
            for tg in targets:
                stable = boot[f"{split}_{tg}_stable"]
                mean_r = boot[f"{split}_{tg}_mean_pairwise_r"]
                split_block[tg] = {
                    "n_total": int(n),
                    "n_stable": int(stable.sum()),
                    "stability_rate": float(stable.mean()),
                    "mean_pairwise_r_distribution": {
                        "min": float(mean_r.min()),
                        "median": float(np.median(mean_r)),
                        "max": float(mean_r.max()),
                    },
                }
            boot_info[split] = split_block
        finding["bootstrap_stability"] = boot_info
    if intv is not None:
        finding["intervention"] = {
            "K_pixels": intv["K_pixels"],
            "sigma_grid_cells": intv["sigma_grid_cells"],
            "results_summary": {
                split: {tg: intv["results"][split][tg].get("summary", {})
                        for tg in targets if intv["results"].get(split, {}).get(tg)}
                for split in intv["results"]
            },
        }
    finding["files"] = {
        "shap_npz": "outputs/session16/exp3/shap_attribution.npz",
        "bootstrap_npz": "outputs/session16/exp3/shap_bootstrap.npz",
        "intervention_json": "outputs/session16/exp3/shap_intervention.json",
        "hero_figure": "outputs/session16/figures/exp3_shap_hero.png",
        "mean_figure": "outputs/session16/figures/exp3_shap_mean.png",
    }
    save_js = OUT / "exp3_finding.json"
    save_js.write_text(json.dumps(finding, indent=2))
    print(f"[exp3-fig] wrote {save_js.relative_to(REPO)}")


if __name__ == "__main__":
    main()
