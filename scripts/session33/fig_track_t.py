"""Draft figures for the Track T results (F6 trade panel + F8 dimension overlay).

SESSION_33_MANUSCRIPT_V3.md Section 9 (F6, F8). Draft PDFs to
outputs/session33/figures/ for review; final placement/naming happens in the
Phase 5 figure pass. All values come from the committed session33 JSONs (no
hand-typed numbers). Style via scripts/session21/figstyle.py.

fig_t_trade.pdf   : (a) T1 recovery-vs-window curves at K=8 (test_b);
                    (b) T2 K x W coefficient-state recovery heatmap (test_b),
                        reference and matching cells outlined.
fig_t_dimension.pdf: (a) d_eff (GP, CI) + PR + chi3d vs |G| with the
                        m_needed steps and the envelope divergence boundary;
                    (b) the honest reduced-budget filter panel: median analysis
                        C_L R2 (impact) per |G| for K in {1, 2, 4, 8}.

Run (CPU):
    taskset -c 16-23 python -m scripts.session33.fig_track_t
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "session21"))

import matplotlib.pyplot as plt  # noqa: E402

from figstyle import METRIC_LABEL, TEXTWIDTH_IN, use_style  # noqa: E402

S32 = REPO_ROOT / "outputs" / "session32"
S33 = REPO_ROOT / "outputs" / "session33"
OUT = S33 / "figures"

GORDER = ["0.25-0.5", "1", "1.5", "2", "3", "4"]
GX = {"0.25-0.5": 0.5, "1": 1.0, "1.5": 1.5, "2": 2.0, "3": 3.0, "4": 4.0}


def _load(p):
    return json.loads(Path(p).read_text())


# ------------------------------------------------------------------ figure 1
def fig_trade(grid):
    ws = sorted(grid["params"]["grid_Ws"])
    ks = sorted(grid["params"]["grid_Ks"])
    fig, axes = plt.subplots(
        1, 2, figsize=(TEXTWIDTH_IN, 0.42 * TEXTWIDTH_IN), constrained_layout=True
    )

    # (a) T1 curves at K=8, test_b
    ax = axes[0]
    k8 = max(ks)
    series = {
        "C_L": ("#404040", "o"),
        "wake_enstrophy": ("#1b7837", "s"),
        "circulation_neg": ("#c0392b", "^"),
    }
    for obs, (c, m) in series.items():
        vals = [
            grid["cells"][f"K{k8}_W{w}"]["obs"]["test_b"]["per_observable_recovered_r2"][obs]
            for w in ws
        ]
        ax.plot(ws, vals, marker=m, ms=3, lw=1.0, color=c,
                label=METRIC_LABEL.get(obs, obs))
    state = [grid["cells"][f"K{k8}_W{w}"]["state_r2"]["test_b"]["window"] for w in ws]
    ax.plot(ws, state, marker="D", ms=3, lw=1.0, ls="--", color="#2166ac",
            label="coefficient state")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ws)
    ax.set_xticklabels([str(w) for w in ws])
    ax.set_xlabel(r"window $W$ (frames), $K = 8$ taps")
    ax.set_ylabel(r"recovered $R^2$ (test B)")
    ax.axhline(0.0, color="0.8", lw=0.6, zorder=0)
    ax.legend(frameon=False, fontsize=6, loc="lower right")
    ax.set_title("(a) delays recover the wall-blind coordinate", fontsize=7)

    # (b) T2 heatmap, test_b state recovery
    ax = axes[1]
    mat = np.array([
        [grid["cells"][f"K{k}_W{w}"]["state_r2"]["test_b"]["window"] for w in ws]
        for k in ks
    ])
    im = ax.imshow(mat, origin="lower", aspect="auto", cmap="viridis",
                   vmin=0.0, vmax=0.7)
    for i, k in enumerate(ks):
        for j, w in enumerate(ws):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=5.5,
                    color="white" if mat[i, j] < 0.45 else "black")
    # outline the reference (K=8, W=1) and the matching (K=1, W=30) cells
    ref = (ks.index(8), ws.index(1))
    match = (ks.index(1), ws.index(30))
    for (i, j), ec in ((ref, "#c0392b"), (match, "#e08214")):
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor=ec, lw=1.4))
    ax.set_xticks(range(len(ws)))
    ax.set_xticklabels([str(w) for w in ws])
    ax.set_yticks(range(len(ks)))
    ax.set_yticklabels([str(k) for k in ks])
    ax.set_xlabel(r"window $W$ (frames)")
    ax.set_ylabel(r"taps $K$")
    ax.set_title("(b) the spatial-for-temporal trade", fontsize=7)
    cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cb.set_label(r"state recovery $R^2$", fontsize=6)
    cb.ax.tick_params(labelsize=6)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_t_trade.pdf")
    plt.close(fig)
    print(f"[fig] wrote {OUT / 'fig_t_trade.pdf'}")


# ------------------------------------------------------------------ figure 2
def fig_dimension(t3, t2b, env):
    fig, axes = plt.subplots(
        1, 2, figsize=(TEXTWIDTH_IN, 0.42 * TEXTWIDTH_IN), constrained_layout=True
    )

    # (a) d_eff + PR + chi3d vs |G|, with m_needed and the divergence boundary
    ax = axes[0]
    table = t3["table_T3"]
    xs = [GX[g] for g in GORDER if g in table]
    deff = [table[g]["d_eff"] for g in GORDER if g in table]
    lo = [table[g]["d_eff_ci95"][0] for g in GORDER if g in table]
    hi = [table[g]["d_eff_ci95"][1] for g in GORDER if g in table]
    pr = [table[g]["participation_ratio"] for g in GORDER if g in table]
    ax.fill_between(xs, lo, hi, color="#1b7837", alpha=0.18, lw=0)
    ax.plot(xs, deff, marker="o", ms=3, lw=1.0, color="#1b7837",
            label=r"$d_{\mathrm{eff}}$ (correlation dim.)")
    ax.plot(xs, pr, marker="s", ms=3, lw=0.9, ls=":", color="#2166ac",
            label="participation ratio")
    mneed = [(GX[g], table[g]["m_needed_abs0.5"]) for g in GORDER
             if g in table and table[g].get("m_needed_abs0.5") is not None]
    ax2 = ax.twinx()
    if mneed:
        ax2.step([m[0] for m in mneed], [m[1] for m in mneed], where="mid",
                 color="#e08214", lw=1.1)
        ax2.plot([m[0] for m in mneed], [m[1] for m in mneed], marker="^", ms=3,
                 ls="none", color="#e08214")
    ax2.set_ylabel(r"window $m$ needed at $K=8$ ($R^2 \geq 0.5$)", fontsize=6,
                   color="#e08214")
    ax2.tick_params(axis="y", labelsize=6, colors="#e08214")
    # divergence boundary (D <= 1.0 compact cores) from the envelope thresholds
    thr = env["models"]["jepa_pool"]["thresholds"].get("1.0", {})
    b = thr.get("div_rate_exceeds_50pct_at_absG")
    if b is not None:
        ax.axvline(b, color="#c0392b", lw=0.9, ls="--")
        ax.text(b + 0.05, ax.get_ylim()[1] * 0.55, "filter\ndivergence\nboundary",
                fontsize=5.5, color="#c0392b")
    ax.set_xlabel(r"gust strength $|G|$")
    ax.set_ylabel("effective dimension")
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    ax.set_title("(a) dimension growth and the sensing requirement", fontsize=7)

    # (b) reduced-budget filter: median analysis C_L R2 per |G| per K
    ax = axes[1]
    kcolors = {"1": "#d95f0e", "2": "#7570b3", "4": "#1b9e77"}
    for k, c in kcolors.items():
        blk = t2b["by_K"].get(k)
        if not blk:
            continue
        xs, ys = [], []
        for g in GORDER:
            agg = blk["aggregates"]["by_G"].get(g, {})
            med = agg.get("filter_CL_analysis_r2_impact", {}).get("median")
            if med is not None:
                xs.append(GX[g])
                ys.append(med)
        ax.plot(xs, ys, marker="o", ms=3, lw=1.0, color=c, label=f"$K = {k}$")
    xs, ys = [], []
    for g in GORDER:
        agg = env["models"]["jepa_pool"]["aggregates"]["by_G"].get(g, {})
        med = agg.get("filter_CL_analysis_r2_impact", {}).get("median")
        if med is not None:
            xs.append(GX[g])
            ys.append(med)
    ax.plot(xs, ys, marker="D", ms=3.5, lw=1.4, color="#1b7837",
            label="$K = 8$ (frozen)")
    ax.axhline(0.0, color="0.8", lw=0.6, zorder=0)
    ax.set_xlabel(r"gust strength $|G|$")
    ax.set_ylabel(r"median analysis $C_L$ $R^2$ (impact)")
    ax.set_ylim(-3.0, 1.05)
    ax.legend(frameon=False, fontsize=6, loc="lower left")
    ax.set_title("(b) the frozen filter needs its eight taps", fontsize=7)

    fig.savefig(OUT / "fig_t_dimension.pdf")
    plt.close(fig)
    print(f"[fig] wrote {OUT / 'fig_t_dimension.pdf'}")


def main():
    use_style()
    grid = _load(S33 / "track_t_recovery_grid.json")
    t3 = _load(S33 / "track_t3_effective_dimension.json")
    t2b = _load(S33 / "t2b_reduced_filter.json")
    env = _load(S32 / "envelope_by_gust.json")
    fig_trade(grid)
    fig_dimension(t3, t2b, env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
