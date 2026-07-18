"""Part D figure (F20d): DA-vs-dimension grid, POD vs Fukami vs JEPA.

Standalone closing study. Every family runs its OWN stack (own E_obs + own
latent-REX + own decode-floor decoder) at d in {4, 8, 16, 32}; each cell reports
its best recipe (min impact-phase C_L RMSE over rex_enkf/linear_lae/eobs).

Panels (both LOG y; the Fukami excursions reach 5.9 / 337%)
  (a) impact-phase C_L RMSE vs latent dimension d
  (b) peak relative C_L error (%) vs latent dimension d

Main lines: POD (blue), Fukami AE (red), JEPA CLW (green). Extra markers:
JEPA CLN-rexpred points (d = 4, 32) and the kit AE-LW d = 32 anchor. At the
Fukami d = 16 point, all three seed values from the session 35 seed band are
drawn as individual open red markers joined by a vertical line ("3 seeds");
every other point is single-seed.

Every number is loaded from its JSON at build time:
  outputs/session34/da_dims_grid.json      (the 15-row grid)
  outputs/session35/fk16_seed_band.json    (Fukami d=16 3-seed band)

Split test_b, K = 8 taps, every-frame wall pressure, no added noise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "session21"))
import figstyle  # noqa: E402

DIMS = (4, 8, 16, 32)

MAIN_FAMILIES = {
    "POD": {
        "label": "linear (POD, own stack)",
        "color": figstyle.FAMILY_COLOR["pod"],
        "marker": figstyle.FAMILY_MARKER["pod"],
    },
    "Fukami AE": {
        "label": "reconstructive (Fukami, own stack)",
        "color": figstyle.FAMILY_COLOR["fukami"],
        "marker": figstyle.FAMILY_MARKER["fukami"],
    },
    "JEPA CLW": {
        "label": "predictive (wake, own stack)",
        "color": figstyle.FAMILY_COLOR["jepa"],
        "marker": figstyle.FAMILY_MARKER["jepa"],
    },
}

CLN_COLOR = "#74c476"  # lighter green, same family ramp convention
AE_KIT_COLOR = "#404040"  # neutral grey for the single kit AE-LW anchor


def load_grid() -> list[dict]:
    with open(REPO / "outputs/session34/da_dims_grid.json") as f:
        return json.load(f)["grid"]


def load_fk16_band() -> dict:
    with open(REPO / "outputs/session35/fk16_seed_band.json") as f:
        return json.load(f)


def rows_for(grid: list[dict], family: str) -> dict[int, dict]:
    return {r["d"]: r for r in grid if r["family"] == family}


def draw_panel(ax, grid: list[dict], band: dict, metric: str, band_key: str,
               ylabel: str, title: str, show_seed_note: bool) -> None:
    # main family lines over d
    for fam, spec in MAIN_FAMILIES.items():
        rows = rows_for(grid, fam)
        ds = [d for d in DIMS if d in rows]
        vals = [rows[d][metric] for d in ds]
        ax.plot(ds, vals, marker=spec["marker"], color=spec["color"],
                markersize=4, linewidth=1.0, label=spec["label"], zorder=3)

    # JEPA CLN-rexpred extra points (d = 4 and 32 in the grid)
    cln = rows_for(grid, "JEPA CLN-rex")
    cln_ds = sorted(cln)
    ax.plot(cln_ds, [cln[d][metric] for d in cln_ds], linestyle=":",
            linewidth=0.8, marker="D", markersize=4, color=CLN_COLOR,
            markerfacecolor="none", markeredgewidth=1.0,
            label="lift-focused predictive", zorder=3)

    # kit AE-LW d = 32 anchor
    ae = rows_for(grid, "kit AE-LW")
    for d, row in ae.items():
        ax.plot([d], [row[metric]], marker="v", markersize=4.5,
                color=AE_KIT_COLOR, markerfacecolor="none",
                markeredgewidth=1.0, linestyle="none",
                label=f"AE (wake) ($d={d}$)", zorder=3)

    # Fukami d = 16 three-seed band: individual open markers + vertical line
    seed_vals = band[band_key]["values"]
    n_seeds = band[band_key]["n"]
    fk_col = figstyle.FAMILY_COLOR["fukami"]
    ax.plot([16, 16], [min(seed_vals), max(seed_vals)], color=fk_col,
            linewidth=0.8, alpha=0.7, zorder=2)
    ax.plot([16] * len(seed_vals), seed_vals, linestyle="none",
            marker=figstyle.FAMILY_MARKER["fukami"], markersize=4.5,
            markerfacecolor="none", markeredgecolor=fk_col,
            markeredgewidth=1.0, zorder=4)
    ax.annotate(f"{n_seeds} seeds", xy=(16, max(seed_vals)),
                xytext=(4, 2), textcoords="offset points",
                fontsize=6, color=fk_col)

    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.set_xticks(DIMS)
    ax.set_xticklabels([str(d) for d in DIMS])
    ax.minorticks_off()
    ax.set_xlabel(r"latent dimension $d$")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=8)

    if show_seed_note:
        ax.text(0.03, 0.03, "single seed per cell\n(except Fukami $d=16$)",
                transform=ax.transAxes, fontsize=6, color="#404040",
                ha="left", va="bottom")


def main() -> None:
    figstyle.use_style()
    grid = load_grid()
    band = load_fk16_band()

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(figstyle.TEXTWIDTH_IN, figstyle.TEXTWIDTH_IN * 0.50)
    )

    draw_panel(ax_a, grid, band, metric="impact_cl_rmse",
               band_key="impact_cl_rmse_band",
               ylabel=r"impact $C_L$ RMSE",
               title=r"(a) impact $C_L$ RMSE vs $d$",
               show_seed_note=True)
    draw_panel(ax_b, grid, band, metric="peak_rel_error_pct",
               band_key="peak_rel_error_pct_band",
               ylabel=r"peak relative $C_L$ error (%)",
               title=r"(b) peak relative error vs $d$",
               show_seed_note=False)

    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=6, loc="lower center", ncol=3,
               handlelength=1.4, columnspacing=1.0,
               bbox_to_anchor=(0.5, 0.045))

    fig.text(0.995, 0.005,
             "in-distribution test set, $K=8$ taps, every-frame wall pressure",
             ha="right", va="bottom", fontsize=6, color="#404040")

    fig.tight_layout(w_pad=1.5, rect=(0, 0.14, 1, 1))

    out_dir = REPO / "paper/sections/figures/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig_da_dims_grid_v4.pdf")
    fig.savefig(out_dir / "fig_da_dims_grid_v4.png", dpi=200)
    print("wrote", out_dir / "fig_da_dims_grid_v4.pdf")
    print("wrote", out_dir / "fig_da_dims_grid_v4.png")


if __name__ == "__main__":
    main()
