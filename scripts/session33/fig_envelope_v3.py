"""F8: the gust-intensity operating envelope (Table V visual + sign asymmetry).

SESSION_33_MANUSCRIPT_V3.md Section 9 (F8). Data from the frozen
outputs/session32/envelope_by_gust.json (jepa_pool, frozen D220 filter).

(a) median analysis C_L R2 (impact) vs |G| for the filter, the static
    single-frame-window recovery and the truth-init open-loop forecast, with
    the filter divergence rate as a bar strip beneath (shared x);
(b) the sign asymmetry: filter closure per |G| by PHYSICAL gust sign
    (G_phys = -G_inv, HANDOFF D224 convention).

The reduced-budget (filter-vs-K) and effective-dimension overlays live in
fig_t_dimension.pdf (scripts/session33/fig_track_t.py); at LaTeX time the two
figures can merge into one two-row F8 if the layout prefers it.

Run (CPU):
    taskset -c 16-23 python -m scripts.session33.fig_envelope_v3
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "session21"))

import matplotlib.pyplot as plt  # noqa: E402

from figstyle import TEXTWIDTH_IN, use_style  # noqa: E402

S32 = REPO_ROOT / "outputs" / "session32"
OUT = REPO_ROOT / "outputs" / "session33" / "figures"

GORDER = ["0.25-0.5", "1", "1.5", "2", "3", "4"]
GX = {"0.25-0.5": 0.5, "1": 1.0, "1.5": 1.5, "2": 2.0, "3": 3.0, "4": 4.0}
LENSES = {
    "filter": ("filter_CL_analysis_r2_impact", "#1b7837", "-", "D",
               "filter analysis"),
    "recovery": ("recovery_CL_r2_impact", "#2166ac", ":", "^",
                 "static recovery"),
    "forecast": ("forecast_CL_r2_impact", "#8c8c8c", "--", "o",
                 "open-loop forecast (truth-init)"),
}


def series(byg, metric, stat="median"):
    xs, ys = [], []
    for g in GORDER:
        blk = byg.get(g, {})
        if blk.get("n", 0) == 0:
            continue
        v = blk[metric][stat]
        if v is not None and np.isfinite(v):
            xs.append(GX[g])
            ys.append(v)
    return xs, ys


def main():
    use_style()
    # Session 38 figure-provenance audit: the OLD source here was the
    # session32 jepa_pool envelope while tab:envelope quotes the vec
    # envelope; re-pointed to the flagship generation.
    env = json.loads((REPO_ROOT / "outputs" / "session33" / "envelope_vec.json").read_text())
    m = env["models"]["jepa_pool_vec"]
    byg = m["aggregates"]["by_G"]
    bysg = m["aggregates"]["by_sign_and_G"]

    fig = plt.figure(figsize=(TEXTWIDTH_IN, 0.46 * TEXTWIDTH_IN),
                     constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[3.2, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    axd = fig.add_subplot(gs[1, 0], sharex=ax)
    axs = fig.add_subplot(gs[:, 1])

    # (a) closure ladder. The truth-init open-loop medians are catastrophic
    # (-17 to -540): pin them at the clip edge as downward markers + a note.
    for key, (metric, c, ls, mk, label) in LENSES.items():
        xs, ys = series(byg, metric)
        if key == "forecast":
            ax.plot(xs, [-2.35] * len(xs), ls="none", marker="v", ms=3.5,
                    color=c, label=label + " (off scale)", clip_on=False)
            continue
        ax.plot(xs, ys, ls=ls, marker=mk, ms=3, lw=1.2 if key == "filter" else 0.9,
                color=c, label=label)
    ax.axhline(0.0, color="0.85", lw=0.6, zorder=0)
    ax.axhspan(-2.6, 0.0, color="0.93", zorder=-1)
    ax.set_ylim(-2.6, 1.05)
    ax.set_ylabel(r"median $C_L$ $R^2$ (impact)")
    ax.legend(frameon=False, fontsize=6, loc="center left")
    ax.set_title("(a) the operating envelope", fontsize=7)
    ax.tick_params(labelbottom=False)
    ax.text(3.9, -1.6, "worse than the mean", fontsize=5.5, color="0.4",
            ha="right")
    _ol = [blk["forecast_CL_r2_impact"]["median"] for blk in byg.values()
           if blk.get("forecast_CL_r2_impact")
           and blk["forecast_CL_r2_impact"].get("median") is not None]
    _lo, _hi = min(_ol), max(_ol)
    ax.text(0.45, -2.05,
            rf"open-loop medians ${_hi:.0f}$ to ${_lo:.0f}$", fontsize=5.0,
            color="0.45", ha="left")

    # divergence strip
    xs, ys = series(byg, "filter_mean_nis", stat="mean")
    div_x, div_y = [], []
    for g in GORDER:
        blk = byg.get(g, {})
        if blk.get("n", 0):
            div_x.append(GX[g])
            div_y.append(blk["div_rate"])
    axd.bar(div_x, div_y, width=0.22, color="#c0392b", alpha=0.75)
    axd.axhline(0.5, color="0.6", lw=0.6, ls="--")
    axd.set_ylim(0, 1.0)
    axd.set_ylabel("div.\nrate", fontsize=6)
    axd.set_xlabel(r"gust strength $|G|$")

    # (b) sign asymmetry (physical sign; G_phys = -G_inv per D224)
    for inv_key, label, c in (("neg_Ginv", r"$G > 0$", "#1b7837"),
                              ("pos_Ginv", r"$G < 0$", "#7570b3")):
        xs, ys = [], []
        for g in GORDER:
            blk = bysg.get(g, {}).get(inv_key, {})
            if blk.get("n", 0):
                v = blk["filter_CL_analysis_r2_impact"]["median"]
                if v is not None and np.isfinite(v):
                    xs.append(GX[g])
                    ys.append(v)
        axs.plot(xs, ys, marker="o", ms=3, lw=1.0, color=c, label=label)
    axs.axhline(0.0, color="0.85", lw=0.6, zorder=0)
    axs.set_xlabel(r"gust strength $|G|$")
    axs.set_ylabel(r"median analysis $C_L$ $R^2$ (impact)")
    axs.legend(frameon=False, fontsize=6, loc="lower left")
    axs.set_title("(b) sign asymmetry of the envelope", fontsize=7)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_envelope_v3.pdf")
    plt.close(fig)
    print(f"[fig] wrote {OUT / 'fig_envelope_v3.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
