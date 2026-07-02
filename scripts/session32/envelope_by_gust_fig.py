"""Diagnostic figure for the Session 32 gust-intensity operating-envelope analysis.

Reads outputs/session32/envelope_by_gust.json and renders a 2x3 diagnostic:
metric vs |G|, one series per D (0.5 / 1.0 / 1.5), with per-encounter scatter
coloured by split on the closure panels. Answers "up to what |G| does each method
do something" for the filter, forecast, and static recovery.

Run: taskset -c 0-15 python -m scripts.session32.envelope_by_gust_fig
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
GNUM = {"0": 0.0, "0.25-0.5": 0.5, "1": 1.0, "1.5": 1.5, "2": 2.0, "3": 3.0, "4": 4.0}
GORDER = ["0", "0.25-0.5", "1", "1.5", "2", "3", "4"]
DCOLORS = {"0.5": "#1b9e77", "1.0": "#d95f02", "1.5": "#7570b3"}
SPLIT_COLORS = {"train": "#4c72b0", "val": "#55a868", "test_b": "#dd8452", "test_c": "#c44e52"}
SPLIT_MARK = {"train": "o", "val": "o", "test_b": "s", "test_c": "D"}


def _bucket(g: float) -> str:
    for name in GORDER:
        lo = {"0": -0.1, "0.25-0.5": 0.24, "1": 0.9, "1.5": 1.4, "2": 1.9, "3": 2.9, "4": 3.9}[name]
        hi = {"0": 0.0, "0.25-0.5": 0.5, "1": 1.0, "1.5": 1.5, "2": 2.0, "3": 3.0, "4": 4.0}[name]
        if lo < g <= hi or (name == "0" and g == 0.0):
            return name
    return "?"


def line_by_D(ax, agg_by_gd, metric, stat, ref=None, ref_label=None, title="", ylabel=""):
    for dname, col in DCOLORS.items():
        xs, ys = [], []
        for g in GORDER:
            cell = agg_by_gd.get(g, {}).get(dname)
            if cell and cell.get("n", 0) > 0:
                v = cell[metric][stat] if stat else cell[metric]
                if v is not None and np.isfinite(v):
                    xs.append(GNUM[g])
                    ys.append(v)
        if xs:
            ax.plot(xs, ys, "-o", color=col, label=f"D={dname}", ms=4, lw=1.5)
    if ref is not None:
        ax.axhline(ref, color="0.4", ls="--", lw=0.8, label=ref_label)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xlabel("|G| (inventory = -G_phys)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)


def scatter_split(ax, records, lens, key, clip=(-2.0, 1.05)):
    jit = {"train": -0.06, "val": -0.02, "test_b": 0.02, "test_c": 0.06}
    for sp in ("train", "val", "test_b", "test_c"):
        xs, ys = [], []
        for r in records:
            if r["split"] != sp:
                continue
            v = r[lens][key]
            if v is None or not np.isfinite(v):
                continue
            xs.append(r["abs_G"] + jit[sp])
            ys.append(min(max(v, clip[0]), clip[1]))
        if xs:
            ax.scatter(
                xs,
                ys,
                s=8,
                c=SPLIT_COLORS[sp],
                marker=SPLIT_MARK[sp],
                alpha=0.5,
                edgecolors="none",
                label=sp,
            )


def main():
    blob = json.loads((REPO_ROOT / "outputs/session32/envelope_by_gust.json").read_text())
    m = blob["models"]["jepa_pool"]
    agg = m["aggregates"]
    gd = agg["by_G_and_D"]
    recs = m["records"]

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0))

    # (1) filter divergence rate
    line_by_D(
        axes[0, 0],
        gd,
        "div_rate",
        None,
        ref=0.5,
        ref_label="50% break",
        title="FILTER: divergence rate",
        ylabel="frac diverged (NIS-tail, D231)",
    )
    axes[0, 0].set_ylim(-0.03, 1.03)

    # (2) filter mean NIS
    line_by_D(
        axes[0, 1],
        gd,
        "filter_mean_nis",
        "mean",
        ref=8.0,
        ref_label="K=8 (calibrated)",
        title="FILTER: mean NIS (calibration)",
        ylabel="mean NIS",
    )

    # (3) filter analysis C_L closure (median R2, impact) + split scatter
    scatter_split(axes[0, 2], recs, "filter", "CL_analysis_r2_impact")
    line_by_D(
        axes[0, 2],
        gd,
        "filter_CL_analysis_r2_impact",
        "median",
        ref=0.0,
        ref_label="R2=0",
        title="FILTER: analysis C_L closure (impact)",
        ylabel="C_L R2 (median line; dots=enc)",
    )
    axes[0, 2].set_ylim(-2.1, 1.1)

    # (4) recovery C_L closure (median R2, impact) + split scatter
    scatter_split(axes[1, 0], recs, "recovery", "CL_r2_impact")
    line_by_D(
        axes[1, 0],
        gd,
        "recovery_CL_r2_impact",
        "median",
        ref=0.0,
        ref_label="R2=0",
        title="STATIC RECOVERY (O1): C_L closure (impact)",
        ylabel="C_L R2 (median line; dots=enc)",
    )
    axes[1, 0].set_ylim(-2.1, 1.1)

    # (5) recovery state R2
    line_by_D(
        axes[1, 1],
        gd,
        "recovery_state_r2",
        "median",
        ref=0.0,
        ref_label="R2=0",
        title="STATIC RECOVERY (O1): pooled-state R2",
        ylabel="state R2 (median)",
    )
    axes[1, 1].set_ylim(-2.1, 1.1)

    # (6) fraction of encounters with positive C_L closure: filter vs recovery vs forecast
    byg = agg["by_G"]
    xs = [GNUM[g] for g in GORDER if byg.get(g, {}).get("n", 0) > 0]

    def fp(metric):
        return [byg[g][metric]["frac_pos"] for g in GORDER if byg.get(g, {}).get("n", 0) > 0]

    axes[1, 2].plot(
        xs,
        fp("filter_CL_analysis_r2_impact"),
        "-o",
        color="#c44e52",
        label="filter (analysis)",
        ms=4,
    )
    axes[1, 2].plot(
        xs, fp("recovery_CL_r2_impact"), "-s", color="#1b9e77", label="static recovery", ms=4
    )
    axes[1, 2].plot(
        xs, fp("forecast_CL_r2_impact"), "-^", color="#7570b3", label="forecast (open-loop)", ms=4
    )
    axes[1, 2].axhline(0.5, color="0.4", ls="--", lw=0.8)
    axes[1, 2].set_title("C_L closure: frac encounters with R2>0", fontsize=9)
    axes[1, 2].set_ylabel("frac positive (pooled over D)", fontsize=8)
    axes[1, 2].set_xlabel("|G| (inventory = -G_phys)", fontsize=8)
    axes[1, 2].set_ylim(-0.03, 1.03)
    axes[1, 2].tick_params(labelsize=7)
    axes[1, 2].grid(alpha=0.25)

    # shade the |G|=4 = test_c extrapolation boundary on all panels
    for ax in axes.ravel():
        ax.axvspan(3.55, 4.2, color="#c44e52", alpha=0.06)
        ax.set_xlim(-0.2, 4.25)

    for ax in (axes[0, 0], axes[0, 1], axes[1, 2]):
        ax.legend(fontsize=6.5, loc="best")
    axes[0, 2].legend(fontsize=6, loc="lower left", ncol=2)

    fig.suptitle(
        "Gust-intensity operating envelope (jepa_pool, frozen filter rho=1.0, K=8 OSP taps) -- "
        "|G|=4 (shaded) is test_c (extrapolation boundary)",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = REPO_ROOT / "outputs/session32/envelope_by_gust.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
