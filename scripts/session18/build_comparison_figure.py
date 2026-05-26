"""Session 18 B1 Part (d) step 3: build the headline comparison figure.

Reads ``outputs/session18/exp_b1/physical_closure_comparison.csv`` and
produces ``outputs/session18/figures/exp_b1_markov_closure_baselines.png``.

The figure shows physical-metric absolute error at H = 16 for each
baseline, with bootstrap 95% CIs as error bars, for the three primary
metrics (C_L, I_y, wake_enstrophy) on Test B (top row) and Test C
(bottom row). Modes shown: ``z_dns`` (sanity check), ``z_markov``
(Markov-only rollout), ``z_full`` (Full-context rollout).

The figure caption-ready ordering:
    JEPA d=64 | Fukami d=3 | Fukami d=32 | Fukami d=64 | POD d=16 | POD d=32 | POD d=64

Usage:
    python scripts/session18/build_comparison_figure.py \\
        --csv outputs/session18/exp_b1/physical_closure_comparison.csv \\
        --output outputs/session18/figures/exp_b1_markov_closure_baselines.png
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
PRIMARY_HORIZON = 16
MODES_TO_PLOT = ("z_dns", "z_markov", "z_full")
MODE_LABELS = {"z_dns": "DNS oracle", "z_markov": "Markov-only", "z_full": "Full-context"}
MODE_COLORS = {"z_dns": "#888888", "z_markov": "#1f77b4", "z_full": "#ff7f0e"}

BASELINE_ORDER = (
    "jepa_d64",
    "fukami_d3",
    "fukami_d32",
    "fukami_d64",
    "pod_d16",
    "pod_d32",
    "pod_d64",
)
BASELINE_LABELS = {
    "jepa_d64": "JEPA d=64",
    "fukami_d3": "Fukami d=3",
    "fukami_d32": "Fukami d=32",
    "fukami_d64": "Fukami d=64",
    "pod_d16": "POD d=16",
    "pod_d32": "POD d=32",
    "pod_d64": "POD d=64",
}


def load_csv(path: Path) -> list[dict]:
    with open(path) as f:
        reader = csv.DictReader(f)
        rows: list[dict] = []
        for r in reader:
            r["d"] = int(r["d"])
            r["horizon"] = int(r["horizon"])
            r["n_enc"] = int(r["n_enc"])
            for k in ("abs_err_mean", "abs_err_median", "ci_lo", "ci_hi"):
                r[k] = float(r[k]) if r[k] not in ("", "nan") else float("nan")
            rows.append(r)
    return rows


def filter_rows(
    rows: list[dict], split: str, metric: str, horizon: int, modes: tuple[str, ...]
) -> dict[str, dict]:
    """Return a dict mode -> {baseline -> row} for the given filter."""
    out: dict[str, dict] = {mode: {} for mode in modes}
    for r in rows:
        if r["split"] != split or r["metric"] != metric or r["horizon"] != horizon:
            continue
        if r["mode"] not in modes:
            continue
        out[r["mode"]][r["baseline"]] = r
    return out


def make_figure(rows: list[dict], output_path: Path) -> None:
    fig, axes = plt.subplots(
        2, len(PRIMARY_METRICS), figsize=(13, 7), sharex=True, constrained_layout=True
    )

    n_baselines = len(BASELINE_ORDER)
    x = np.arange(n_baselines)
    bar_w = 0.27

    for row_idx, split in enumerate(("test_b", "test_c")):
        for col_idx, metric in enumerate(PRIMARY_METRICS):
            ax = axes[row_idx, col_idx]
            mode_to_baseline = filter_rows(rows, split, metric, PRIMARY_HORIZON, MODES_TO_PLOT)

            for k, mode in enumerate(MODES_TO_PLOT):
                heights, lows, highs = [], [], []
                for tag in BASELINE_ORDER:
                    row = mode_to_baseline[mode].get(tag)
                    if row is None or np.isnan(row["abs_err_mean"]):
                        heights.append(np.nan)
                        lows.append(np.nan)
                        highs.append(np.nan)
                        continue
                    h = row["abs_err_mean"]
                    heights.append(h)
                    lows.append(max(0.0, h - row["ci_lo"]))
                    highs.append(max(0.0, row["ci_hi"] - h))
                heights = np.asarray(heights)
                err = np.vstack([lows, highs])
                ax.bar(
                    x + (k - 1) * bar_w,
                    np.where(np.isnan(heights), 0.0, heights),
                    width=bar_w,
                    label=MODE_LABELS[mode] if (row_idx == 0 and col_idx == 0) else None,
                    color=MODE_COLORS[mode],
                    edgecolor="black",
                    linewidth=0.4,
                )
                ax.errorbar(
                    x + (k - 1) * bar_w,
                    np.where(np.isnan(heights), 0.0, heights),
                    yerr=np.where(np.isnan(err), 0.0, err),
                    fmt="none",
                    ecolor="black",
                    elinewidth=0.6,
                    capsize=2,
                )

            ax.set_title(
                f"{metric} abs err ({split.upper().replace('_', ' ')})",
                fontsize=10,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(
                [BASELINE_LABELS[t] for t in BASELINE_ORDER],
                rotation=30, ha="right", fontsize=8,
            )
            if col_idx == 0:
                ax.set_ylabel("Absolute error at H = 16")
            ax.grid(axis="y", alpha=0.3)

    # Single legend at top
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(MODES_TO_PLOT), frameon=False,
               bbox_to_anchor=(0.5, 1.02), fontsize=10)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] wrote {output_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build B1 comparison figure")
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("outputs/session18/exp_b1/physical_closure_comparison.csv"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/session18/figures/exp_b1_markov_closure_baselines.png"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.csv.exists():
        raise SystemExit(f"CSV missing: {args.csv}; run physical_metrics_from_rollouts.py first")
    rows = load_csv(args.csv)
    if not rows:
        raise SystemExit(f"CSV empty: {args.csv}")
    make_figure(rows, args.output)


if __name__ == "__main__":
    main()
