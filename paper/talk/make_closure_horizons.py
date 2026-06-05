#!/usr/bin/env python3
"""Forward-closure MAE figures at multiple horizons (H = 4, 8, 16) for the talk.

Same layout as the paper's fig4_closure (2 splits x 3 observables: dots with
bootstrap-CI whiskers and a DNS-oracle floor), regenerated per horizon from the
same verified table, outputs/session20/closure_r2/closure_r2_heldout.csv.
No model re-evaluation: that table already stores every horizon (1, 4, 8, 16,
32, 64). Reuses the project figure style from scripts/session21/figstyle.py.

Output: figs/closure_H{H}.png
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "session21"))
import figstyle as fs  # noqa: E402

CSV = REPO / "outputs/session20/closure_r2/closure_r2_heldout.csv"
OUTDIR = Path(__file__).resolve().parent / "figs"

COLS = ["C_L", "I_y", "wake_enstrophy"]
SPLITS = ["test_b", "test_c"]
LAYOUT = [
    ("fukami_d3_noBN", 0.0), ("fukami_d32_noBN", 0.8), ("fukami_d64_noBN", 1.6),
    ("pod_d16_noBN", 3.0), ("pod_d32_noBN", 3.8), ("pod_d64_noBN", 4.6),
    ("jepa_d32_noBN", 6.0), ("jepa_d64_test1_noBN", 6.8),
]
GROUPS = [("AE", 0.8), ("POD", 3.8), ("JEPA", 6.4)]


def load():
    rows = {}
    with open(CSV) as f:
        for r in csv.DictReader(f):
            rows[(r["baseline"], r["split"], r["metric"], int(r["horizon"]), r["mode"])] = r
    return rows


def make(data, H):
    fs.use_style()
    fig, axes = plt.subplots(2, 3, figsize=fs.figure_size(1.0, aspect=0.78))
    for i, split in enumerate(SPLITS):
        for j, metric in enumerate(COLS):
            ax = axes[i, j]
            zdns = [float(data[(t, split, metric, H, "z_dns")]["mae"])
                    for (t, _) in LAYOUT if (t, split, metric, H, "z_dns") in data]
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
                ax.set_ylabel(f"{fs.SPLIT_LABEL[split]}\nMAE at $H={H}$")
    handles = fs.family_legend_handles(include_oracle=True)
    for h in handles:
        if h.get_label() == "reconstructive":
            h.set_label("reconstructive (AE)")
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02),
               columnspacing=1.4, handletextpad=0.3)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = OUTDIR / f"closure_H{H}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    d = load()
    for H in (4, 8, 16):
        make(d, H)
