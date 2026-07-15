"""Part D figure (F20b+F20c composite): own-stack family DA comparison.

Panels
  (a) phase-resolved C_L RMSE (pre / impact / relax) grouped bars for the three
      own-stack families: JEPA-CLW (native vector flagship, jepa_pool_vec),
      JEPA-CLN (jepa_pool_ln_s0) and the kit AE-LW (ae_wake_pool). Each family
      runs its OWN E_obs + latent-REX + decode-floor stack; the bar reports the
      family's best recipe by impact C_L RMSE (min over rex_enkf/linear_lae/eobs,
      the da_dims_grid criterion).
  (b) impact C_L RMSE vs sensor budget K in {2, 4, 8, 16} (no tap noise).
  (c) impact C_L RMSE vs tap noise {0, 5, 10, 20}% at K = 8.

Every number is loaded from its JSON at build time:
  outputs/session34/da_phase_eval.json               (JEPA-CLW, K=8, n=0)
  outputs/session34/da_phase_jepa_pool_ln_s0.json    (JEPA-CLN, K=8, n=0)
  outputs/session34/da_phase_ae_wake_pool.json       (AE-LW,    K=8, n=0)
  outputs/session34/da_grid/{model}_K{2,4,16}_n0.0.json      (K sweep)
  outputs/session34/da_grid/{model}_K8_n{0.05,0.1,0.2}.json  (noise sweep)

Split test_b (42 encounters), single filter seed (seed 0 in every cell).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "session21"))
import figstyle  # noqa: E402

RECIPES = ("rex_enkf", "linear_lae", "eobs")

# family key: (display label, phase-file path, da_grid model prefix, colour, marker)
FAMILIES = {
    "jepa_pool_vec": {
        "label": "predictive (wake)",
        "phase_file": REPO / "outputs/session34/da_phase_eval.json",
        "color": figstyle.FAMILY_COLOR["jepa"],
        "marker": figstyle.FAMILY_MARKER["jepa"],
    },
    "jepa_pool_ln_s0": {
        "label": "predictive (lift)",
        "phase_file": REPO / "outputs/session34/da_phase_jepa_pool_ln_s0.json",
        "color": "#74c476",  # lighter green: same family ramp as figstyle shading
        "marker": "D",
    },
    "ae_wake_pool": {
        "label": "AE (wake)",
        "phase_file": REPO / "outputs/session34/da_phase_ae_wake_pool.json",
        "color": figstyle.FAMILY_COLOR["fukami"],
        "marker": figstyle.FAMILY_MARKER["fukami"],
    },
}

K_SWEEP = (2, 4, 8, 16)
NOISE_SWEEP = ("0.0", "0.05", "0.1", "0.2")  # fraction strings as in filenames


def load_summary(path: Path) -> dict:
    with open(path) as f:
        d = json.load(f)
    return d["summary"]


def best_recipe(summary: dict) -> tuple[str, dict]:
    """Best recipe by impact-phase C_L RMSE (da_dims_grid criterion)."""
    impact = {r: summary[r]["impact"]["cl_rmse"] for r in RECIPES}
    r = min(impact, key=impact.get)
    return r, summary[r]


def grid_path(model: str, k: int, noise: str) -> Path:
    return REPO / f"outputs/session34/da_grid/{model}_K{k}_n{noise}.json"


def cell_impact_rmse(model: str, k: int, noise: str) -> float:
    """Impact C_L RMSE for a (model, K, noise) cell, best recipe.

    The K=8 / n=0 anchor lives in the main phase-resolved file, not da_grid.
    """
    if k == 8 and noise == "0.0":
        summary = load_summary(FAMILIES[model]["phase_file"])
    else:
        summary = load_summary(grid_path(model, k, noise))
    _, best = best_recipe(summary)
    return best["impact"]["cl_rmse"]


def main() -> None:
    figstyle.use_style()

    fig, axes = plt.subplots(
        1, 3, figsize=(figstyle.TEXTWIDTH_IN, figstyle.TEXTWIDTH_IN * 0.36)
    )
    ax_a, ax_b, ax_c = axes

    # ------------------------------------------------------------- panel (a)
    phases = ("pre", "impact", "relax")
    phase_labels = ("pre", "impact", "relax")
    n_fam = len(FAMILIES)
    width = 0.8 / n_fam
    x = np.arange(len(phases))
    for i, (model, spec) in enumerate(FAMILIES.items()):
        summary = load_summary(spec["phase_file"])
        _, best = best_recipe(summary)
        vals = [best[ph]["cl_rmse"] for ph in phases]
        ax_a.bar(
            x + (i - (n_fam - 1) / 2) * width,
            vals,
            width=width * 0.92,
            color=spec["color"],
            edgecolor="black",
            linewidth=0.4,
            label=spec["label"],
        )
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(phase_labels)
    ax_a.set_ylabel(r"$C_L$ RMSE")
    ax_a.set_xlabel("encounter phase")
    ax_a.set_ylim(0, ax_a.get_ylim()[1] * 1.30)
    ax_a.legend(fontsize=6, loc="upper left", handlelength=1.0,
                borderaxespad=0.2, ncol=1)
    ax_a.set_title(r"(a) phase-resolved, $K=8$", fontsize=8)

    # ------------------------------------------------------------- panel (b)
    for model, spec in FAMILIES.items():
        vals = [cell_impact_rmse(model, k, "0.0") for k in K_SWEEP]
        ax_b.plot(
            K_SWEEP,
            vals,
            marker=spec["marker"],
            color=spec["color"],
            markersize=4,
            linewidth=1.0,
        )
    ax_b.set_xscale("log", base=2)
    ax_b.set_xticks(K_SWEEP)
    ax_b.set_xticklabels([str(k) for k in K_SWEEP])
    ax_b.minorticks_off()
    ax_b.set_xlabel(r"sensor budget $K$")
    ax_b.set_ylabel(r"impact $C_L$ RMSE")
    ax_b.set_title("(b) sensor budget (clean)", fontsize=8)

    # ------------------------------------------------------------- panel (c)
    noise_pct = [100.0 * float(n) for n in NOISE_SWEEP]
    for model, spec in FAMILIES.items():
        vals = [cell_impact_rmse(model, 8, n) for n in NOISE_SWEEP]
        ax_c.plot(
            noise_pct,
            vals,
            marker=spec["marker"],
            color=spec["color"],
            markersize=4,
            linewidth=1.0,
        )
    ax_c.set_xticks(noise_pct)
    ax_c.set_xticklabels(["0", "5", "10", "20"])
    ax_c.set_xlabel("tap noise (%)")
    ax_c.set_ylabel(r"impact $C_L$ RMSE")
    ax_c.set_title("(c) tap noise, $K=8$", fontsize=8)

    for ax in (ax_b, ax_c):
        ax.set_ylim(bottom=0.0)

    fig.text(
        0.995,
        0.01,
        "own-stack DA, split test B (42 encounters), single filter seed",
        ha="right",
        va="bottom",
        fontsize=6,
        color="#404040",
    )

    fig.tight_layout(w_pad=1.2, rect=(0, 0.03, 1, 1))

    out_dir = REPO / "paper/sections/figures/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig_ownstack_da_v4.pdf")
    fig.savefig(out_dir / "fig_ownstack_da_v4.png", dpi=200)
    print("wrote", out_dir / "fig_ownstack_da_v4.pdf")
    print("wrote", out_dir / "fig_ownstack_da_v4.png")


if __name__ == "__main__":
    main()
