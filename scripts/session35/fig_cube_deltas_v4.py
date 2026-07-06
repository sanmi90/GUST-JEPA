"""F6: Peak-lift closure per cube cell with paired deltas vs CL (Session 35 v4).

Panel (a): pooled peak-region R2 of the frozen LINEAR lift probe per Track C
cell (seed mean, min-max whiskers over 3 seeds), cells ordered C0, CN, CW,
CWN, CL, CLN, CLW, CLWN. Cells without the lift head in the fukami red, cells
with it in the jepa green (same coding as F5).

Panel (b): forest plot of the paired per-encounter peak-R2 deltas vs CL for
every <cell>_vs_cl comparison in the pre-registered gates file. Point =
case-mean delta, error bar = case-clustered bootstrap 95% CI
(scripts/session28/stats_lib.case_cluster_bootstrap, case-mean CI). A filled
marker means the CI excludes zero; open means it straddles zero. The x axis
is symlog: the no-L deltas are one to two orders of magnitude larger than the
L-cell deltas.

Every plotted number is read at build time from
``outputs/session34/trackc_lift.json`` (per cell x seed
``results[cell][s{seed}]['linear']['pooled_peak_r2']``) and
``outputs/session34/trackc_gates.json``
(``Q*[...]['peak_r2']['boot']['case_mean(_ci)']``). Nothing is hardcoded.

Run (repo root, CPU):
    OMP_NUM_THREADS=4 .venv/bin/python scripts/session35/fig_cube_deltas_v4.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.session21.figstyle import FAMILY_COLOR, TEXTWIDTH_IN, use_style  # noqa: E402

LIFT_JSON = REPO_ROOT / "outputs/session34/trackc_lift.json"
GATES_JSON = REPO_ROOT / "outputs/session34/trackc_gates.json"
OUT_PDF = REPO_ROOT / "paper/sections/figures/results/fig_cube_deltas_v4.pdf"

# Paper order of the cube cells (no-L half first); same coding as F5.
CELL_ORDER = ("c0", "cn", "cw", "cwn", "cl", "cln", "clw", "clwn")
CELL_LABEL = {"c0": "C0", "cn": "CN", "cw": "CW", "cwn": "CWN",
              "cl": "CL", "cln": "CLN", "clw": "CLW", "clwn": "CLWN"}
NO_L_CELLS = frozenset({"c0", "cn", "cw", "cwn"})
SEEDS = ("s0", "s1", "s2")
PROBE = "linear"  # pre-registered primary readout (trackc_gates.json)

# vs-CL comparisons: (gates question key, comparison key, left cell).
VS_CL = (
    ("Q2_D255", "cw_vs_cl", "cw"),
    ("Q2alt_D256", "cn_vs_cl", "cn"),
    ("Q2_D255", "clw_vs_cl", "clw"),
    ("Q2alt_D256", "cln_vs_cl", "cln"),
)


def cell_color(cell: str) -> str:
    return FAMILY_COLOR["fukami"] if cell in NO_L_CELLS else FAMILY_COLOR["jepa"]


def main() -> None:
    use_style()

    lift = json.loads(LIFT_JSON.read_text())["results"]
    gates = json.loads(GATES_JSON.read_text())

    peak_r2 = {cell: np.array([lift[cell][s][PROBE]["pooled_peak_r2"] for s in SEEDS])
               for cell in CELL_ORDER}

    deltas = []  # (label, left cell, case_mean, ci_low, ci_high, n_enc, n_cases)
    for qkey, ckey, left in VS_CL:
        boot = gates[qkey][ckey]["peak_r2"]["boot"]
        deltas.append((f"{CELL_LABEL[left]} - CL", left, boot["case_mean"],
                       boot["case_mean_ci"][0], boot["case_mean_ci"][1],
                       boot["n_encounters"], boot["n_cases"]))
    n_enc = {d[5] for d in deltas}
    n_cases = {d[6] for d in deltas}
    assert len(n_enc) == 1 and len(n_cases) == 1, "inconsistent pairing counts"
    n_enc, n_cases = n_enc.pop(), n_cases.pop()

    fig_w = TEXTWIDTH_IN
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(fig_w, fig_w * 0.42),
        gridspec_kw={"width_ratios": [1.4, 1.0], "wspace": 0.34})

    # ---- panel (a): pooled peak-region R2 per cell, seed mean + min-max ----
    for i, cell in enumerate(CELL_ORDER):
        vals = peak_r2[cell]
        mean = vals.mean()
        color = cell_color(cell)
        ax_a.errorbar(i, mean,
                      yerr=[[mean - vals.min()], [vals.max() - mean]],
                      fmt="o", ms=3.8, color=color, ecolor=color,
                      elinewidth=0.9, capsize=2.0, capthick=0.9, zorder=3)
    ax_a.axhline(0.0, color="0.55", lw=0.7, zorder=1)
    ax_a.set_xticks(range(len(CELL_ORDER)))
    ax_a.set_xticklabels([CELL_LABEL[c] for c in CELL_ORDER], fontsize=6.5,
                         rotation=30, ha="right", rotation_mode="anchor")
    ax_a.set_xlim(-0.6, len(CELL_ORDER) - 0.4)
    ax_a.set_ylabel(r"pooled peak-region $R^{2}$ ($C_L$)")
    ax_a.set_xlabel("conditioning cell")

    handles = [
        plt.Line2D([], [], color=FAMILY_COLOR["jepa"], marker="o", ls="none",
                   ms=3.6, label="with lift head"),
        plt.Line2D([], [], color=FAMILY_COLOR["fukami"], marker="o", ls="none",
                   ms=3.6, label="without lift head"),
    ]
    ax_a.legend(handles=handles, loc="upper left", fontsize=6.5,
                handletextpad=0.4, borderaxespad=0.3)
    ax_a.text(0.985, 0.02, "linear probe, n = 3 seeds, split test B",
              transform=ax_a.transAxes, ha="right", va="bottom",
              fontsize=6, color="0.35")

    # ---- panel (b): paired case-mean deltas vs CL (forest plot) -----------
    ys = np.arange(len(deltas))[::-1]  # first comparison at the top
    for y, (label, left, mean, lo, hi, _, _) in zip(ys, deltas):
        color = cell_color(left)
        excludes_zero = (lo > 0.0) or (hi < 0.0)
        ax_b.errorbar(mean, y, xerr=[[mean - lo], [hi - mean]],
                      fmt="o", ms=4.2, color=color, ecolor=color,
                      elinewidth=0.9, capsize=2.0, capthick=0.9,
                      mfc=color if excludes_zero else "white",
                      mec=color, mew=0.8, zorder=3)
    ax_b.axvline(0.0, color="0.35", ls=(0, (4, 3)), lw=0.9, zorder=2)
    ax_b.set_yticks(ys)
    ax_b.set_yticklabels([d[0] for d in deltas], fontsize=6.5)
    ax_b.set_ylim(-1.1, len(deltas) - 0.15)
    ax_b.set_xscale("symlog", linthresh=5.0)
    ax_b.set_xticks([-50, -10, 0, 5])
    ax_b.set_xticklabels(["-50", "-10", "0", "5"], fontsize=6.5)
    ax_b.set_xlabel(r"case-mean $\Delta R^{2}_{\mathrm{peak}}$ vs CL (symlog)")

    filled = plt.Line2D([], [], color="0.25", marker="o", ls="none", ms=4.2,
                        label="95% CI excludes 0")
    openm = plt.Line2D([], [], color="0.25", marker="o", ls="none", ms=4.2,
                       mfc="white", mew=0.8, label="CI straddles 0")
    leg = ax_b.legend(handles=[filled, openm], loc="upper right", fontsize=6,
                      handletextpad=0.4, borderaxespad=0.2, frameon=True,
                      framealpha=1.0, edgecolor="none", facecolor="white")
    leg.set_zorder(4)
    ax_b.text(0.02, 0.02,
              f"n = 3 seeds, split test B\n"
              f"case-clustered 95% CI ({n_enc} enc, {n_cases} cases)",
              transform=ax_b.transAxes, ha="left", va="bottom",
              fontsize=6, color="0.35")

    for ax, tag in ((ax_a, "(a)"), (ax_b, "(b)")):
        ax.text(-0.02, 1.04, tag, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8.5)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PDF.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_PDF.with_suffix('.png')}")
    for cell in CELL_ORDER:
        v = peak_r2[cell]
        print(f"  {CELL_LABEL[cell]:5s} peak R2 mean {v.mean():+.3f} "
              f"[{v.min():+.3f}, {v.max():+.3f}]")
    for label, _, mean, lo, hi, _, _ in deltas:
        ex = "excludes 0" if (lo > 0 or hi < 0) else "straddles 0"
        print(f"  {label:10s} {mean:+8.3f} CI [{lo:+8.3f}, {hi:+8.3f}]  {ex}")


if __name__ == "__main__":
    main()
