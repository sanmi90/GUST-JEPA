"""Session 18 Figure 4: paper-centerpiece Markov-closure comparison.

Submission-quality version of build_comparison_figure.py. Layout follows
the SESSION18 plan's Figure 4 description:

    4-panel grid: C_L, I_y^w, wake_enstrophy, spectral lambda ratio at H=16
    Test B (top row) and Test C (bottom row) — but we use 3 panels (drop
    spectral lambda ratio, not in B1's metric set) with two-row Test B / C.

Reads outputs/session18/exp_b1_test3/physical_closure_noBN_unified.csv
(unified no-output-BN recipe across all 7 baselines).

Aesthetic choices:
- Serif font (matplotlib's mathtext default; closest match for JFM LaTeX)
- Bootstrap 95% CIs as black error bars
- JEPA in green (the headline method), Fukami in red shades, POD in
  blue shades — colorblind-friendly distinct hues per family
- Y-axis log scale so the 3x JEPA advantage on wake_enstrophy reads
  cleanly (linear-scale would have the bar dwarfed)
- Horizontal dashed gray = DNS-oracle baseline (lower bound for the
  generic predictor+probe pipeline)

Output: outputs/session18/figures/figure4_markov_closure_centerpiece.{png,pdf}
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PRIMARY_METRICS = ("C_L", "I_y", "wake_enstrophy")
METRIC_LABELS = {
    "C_L": r"$C_L$ absolute error",
    "I_y": r"$I_y$ absolute error",
    "wake_enstrophy": r"Wake enstrophy absolute error",
}
H = 16
SPLIT_LABELS = {"test_b": "Test B (in-distribution)", "test_c": "Test C (G = +4, OOD)"}

BASELINE_ORDER = (
    "fukami_d3_noBN",
    "fukami_d32_noBN",
    "fukami_d64_noBN",
    "pod_d16_noBN",
    "pod_d32_noBN",
    "pod_d64_noBN",
    "jepa_d64_test1_noBN",
)
BASELINE_LABELS = {
    "fukami_d3_noBN": r"Fukami AE $d=3$",
    "fukami_d32_noBN": r"Fukami AE $d=32$",
    "fukami_d64_noBN": r"Fukami AE $d=64$",
    "pod_d16_noBN": r"POD $d=16$",
    "pod_d32_noBN": r"POD $d=32$",
    "pod_d64_noBN": r"POD $d=64$",
    "jepa_d64_test1_noBN": r"JEPA $d=64$",
}
BASELINE_COLORS = {
    "fukami_d3_noBN": "#fcbba1",
    "fukami_d32_noBN": "#fb6a4a",
    "fukami_d64_noBN": "#cb181d",
    "pod_d16_noBN": "#9ecae1",
    "pod_d32_noBN": "#4292c6",
    "pod_d64_noBN": "#08519c",
    "jepa_d64_test1_noBN": "#238b45",
}


def load_csv(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r["d"] = int(r["d"])
            r["horizon"] = int(r["horizon"])
            r["n_enc"] = int(r["n_enc"])
            for k in ("abs_err_mean", "abs_err_median", "ci_lo", "ci_hi"):
                r[k] = float(r[k]) if r[k] not in ("", "nan") else float("nan")
            rows.append(r)
    return rows


def get(rows, **filters):
    matches = [r for r in rows if all(r[k] == v for k, v in filters.items())]
    return matches[0] if matches else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Figure 4 (centerpiece)")
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("outputs/session18/exp_b1_test3/physical_closure_noBN_unified.csv"),
    )
    p.add_argument(
        "--output-png",
        type=Path,
        default=Path("outputs/session18/figures/figure4_markov_closure_centerpiece.png"),
    )
    p.add_argument(
        "--output-pdf",
        type=Path,
        default=Path("outputs/session18/figures/figure4_markov_closure_centerpiece.pdf"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_csv(args.csv)
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.8,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })

    fig, axes = plt.subplots(
        2, len(PRIMARY_METRICS),
        figsize=(11, 6.8),
        sharex=True,
        constrained_layout=True,
    )

    x = np.arange(len(BASELINE_ORDER))
    bar_w = 0.55

    for row_idx, split in enumerate(("test_b", "test_c")):
        for col_idx, metric in enumerate(PRIMARY_METRICS):
            ax = axes[row_idx, col_idx]

            heights, lows, highs = [], [], []
            dns_height = np.nan
            for bl in BASELINE_ORDER:
                row = get(rows, baseline=bl, metric=metric, horizon=H, split=split, mode="z_markov")
                if row is None:
                    heights.append(np.nan); lows.append(np.nan); highs.append(np.nan)
                    continue
                h = row["abs_err_mean"]
                heights.append(h)
                lows.append(max(0, h - row["ci_lo"]))
                highs.append(max(0, row["ci_hi"] - h))
            # DNS oracle (z_dns) per baseline — take the median across baselines
            dns_rows = [get(rows, baseline=bl, metric=metric, horizon=H, split=split, mode="z_dns")
                        for bl in BASELINE_ORDER]
            dns_heights = [r["abs_err_mean"] for r in dns_rows if r is not None]
            if dns_heights:
                dns_height = float(np.median(dns_heights))

            colors = [BASELINE_COLORS[b] for b in BASELINE_ORDER]
            bars = ax.bar(
                x, heights, width=bar_w,
                color=colors, edgecolor="black", linewidth=0.5,
            )
            # Highlight JEPA bar
            jepa_idx = BASELINE_ORDER.index("jepa_d64_test1_noBN")
            bars[jepa_idx].set_edgecolor("black")
            bars[jepa_idx].set_linewidth(1.2)
            bars[jepa_idx].set_hatch("//")

            ax.errorbar(
                x, np.where(np.isnan(heights), 0, heights),
                yerr=np.vstack([np.where(np.isnan(lows), 0, lows),
                                np.where(np.isnan(highs), 0, highs)]),
                fmt="none", ecolor="black", elinewidth=0.7, capsize=2.5,
            )

            # DNS oracle line
            if not np.isnan(dns_height):
                ax.axhline(dns_height, color="gray", linestyle=":", linewidth=1.0,
                           alpha=0.7,
                           label="DNS-oracle floor" if (row_idx == 0 and col_idx == 0) else None)

            ax.set_yscale("log")
            ax.grid(True, axis="y", which="major", alpha=0.25, linewidth=0.5)
            ax.grid(True, axis="y", which="minor", alpha=0.1, linewidth=0.3)

            # Y-axis label only on the left column
            if col_idx == 0:
                ax.set_ylabel(METRIC_LABELS[metric] if metric != "wake_enstrophy" else
                              r"Wake enstrophy $|\Delta E_w|$",
                              fontsize=10)
            else:
                ax.set_ylabel(METRIC_LABELS[metric].split()[0], fontsize=10)

            # Title: metric on top row, split label on rightmost column
            if row_idx == 0:
                ax.set_title(METRIC_LABELS[metric].split()[0] if metric != "wake_enstrophy"
                             else r"Wake enstrophy ($\Omega_w$)",
                             fontsize=11, fontweight="normal")
            if col_idx == len(PRIMARY_METRICS) - 1:
                ax.text(1.04, 0.5, SPLIT_LABELS[split], transform=ax.transAxes,
                        rotation=270, va="center", ha="left", fontsize=10)

            # X-axis labels only on bottom row
            if row_idx == 1:
                ax.set_xticks(x)
                ax.set_xticklabels([BASELINE_LABELS[b] for b in BASELINE_ORDER],
                                   rotation=35, ha="right", fontsize=8.5)
            else:
                ax.set_xticks(x)
                ax.set_xticklabels([])

    fig.suptitle(
        f"Physical Markov closure at H={H} frames after impact "
        r"(unified $-$no$-$output$-$BN predictor across all baselines)",
        fontsize=12, y=1.02,
    )

    # Single legend at the bottom of figure
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [
        Patch(facecolor="#238b45", edgecolor="black", hatch="//", label=r"JEPA $d=64$ (this work)"),
        Patch(facecolor="#cb181d", edgecolor="black", label="Fukami AE family"),
        Patch(facecolor="#08519c", edgecolor="black", label="POD family"),
        Line2D([0], [0], color="gray", linestyle=":", linewidth=1.4,
               label="DNS-oracle baseline (lower bound)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.05), fontsize=9)

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=300, bbox_inches="tight")
    fig.savefig(args.output_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output_png}")
    print(f"Wrote {args.output_pdf}")


if __name__ == "__main__":
    main()
