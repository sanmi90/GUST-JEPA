"""SPEC 1 - Fig 4: held-out forward closure, dots with bootstrap-CI whiskers.

Replaces the broken bar-chart centrepiece. One idea: held-out forward closure
separates the families, most clearly on the wake observable. Source is the
verified closure table outputs/session20/closure_r2/closure_r2_heldout.csv
(z_markov mode = Markov rollout from impact; mae + 95% bootstrap CI).

Layout: 2 rows (test_b, test_c) x 3 cols (C_L force, I_y impulse, wake
enstrophy). Dots, not bars; family colour key; DNS-oracle floor (the no-rollout
representational closure, z_dns mode) as a horizontal dashed line per panel.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import figstyle as fs

REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "outputs/session20/closure_r2/closure_r2_heldout.csv"
OUT_PDF = REPO / "paper/sections/figures/results/fig4_closure.pdf"
OUT_PNG = REPO / "outputs/session21/figs/fig4_closure.png"

H = 16
COLS = ["C_L", "I_y", "wake_enstrophy"]
SPLITS = ["test_b", "test_c"]
# x layout: three family groups with gaps, d-variants adjacent within a group.
LAYOUT = [
    ("fukami_d3_noBN", 0.0), ("fukami_d32_noBN", 0.8), ("fukami_d64_noBN", 1.6),
    ("pod_d16_noBN", 3.0), ("pod_d32_noBN", 3.8), ("pod_d64_noBN", 4.6),
    ("jepa_d32_noBN", 6.0), ("jepa_d64_test1_noBN", 6.8),
]
GROUPS = [("recon.", 0.8), ("POD", 3.8), ("JEPA", 6.4)]


def load() -> dict:
    rows = {}
    with open(CSV) as f:
        for r in csv.DictReader(f):
            key = (r["baseline"], r["split"], r["metric"], int(r["horizon"]), r["mode"])
            rows[key] = r
    return rows


def main() -> None:
    fs.use_style()
    data = load()
    fig, axes = plt.subplots(2, 3, figsize=fs.figure_size(1.0, aspect=0.78))

    for i, split in enumerate(SPLITS):
        for j, metric in enumerate(COLS):
            ax = axes[i, j]
            # DNS-oracle floor: best (smallest) no-rollout representational MAE.
            zdns = [float(data[(t, split, metric, H, "z_dns")]["mae"])
                    for (t, _) in LAYOUT
                    if (t, split, metric, H, "z_dns") in data]
            if zdns:
                ax.axhline(min(zdns), ls=(0, (4, 3)), lw=0.9,
                           color=fs.FAMILY_COLOR["oracle"], zorder=1)
            for tag, x in LAYOUT:
                key = (tag, split, metric, H, "z_markov")
                if key not in data:
                    continue
                r = data[key]
                mae = float(r["mae"])
                lo, hi = float(r["mae_ci_lo"]), float(r["mae_ci_hi"])
                fam = fs.BASELINE[tag][0]
                ax.errorbar(x, mae, yerr=[[mae - lo], [hi - mae]],
                            fmt=fs.FAMILY_MARKER[fam], ms=4.5,
                            color=fs.family_color(tag), ecolor=fs.family_color(tag),
                            elinewidth=0.9, capsize=2.0, capthick=0.9, zorder=3)
            ax.set_xticks([gx for _, gx in GROUPS])
            ax.set_xticklabels([g for g, _ in GROUPS] if i == 1 else [])
            ax.set_xlim(-0.6, 7.4)
            ax.tick_params(axis="x", length=0)
            ax.margins(y=0.12)
            if i == 0:
                ax.set_title(fs.METRIC_LABEL[metric])
            if j == 0:
                ax.set_ylabel(f"{fs.SPLIT_LABEL[split]}\nMAE at $H={H}$"
                              if i == 0 else f"{fs.SPLIT_LABEL[split]}\nMAE at $H={H}$")

    # one shared legend (families + oracle), below the panels
    handles = fs.family_legend_handles(include_oracle=True)
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02), columnspacing=1.4, handletextpad=0.3)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=200)
    print(f"wrote {OUT_PDF}\nwrote {OUT_PNG}")


if __name__ == "__main__":
    main()
