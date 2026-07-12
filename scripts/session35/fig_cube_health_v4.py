"""F5: Conditioning-cube latent health (Session 35 manuscript v4).

Panel (a): final participation ratio PR(z) per Track C cube cell (3 seeds per
cell, small horizontal jitter), cells ordered C0, CN, CW, CWN, CL, CLN, CLW,
CLWN, with the pre-registered PR floor as a dashed reference line. Cells
without the lift head (C0, CN, CW, CWN) are drawn in the fukami red, cells
with it in the jepa green: all runs are JEPA runs, the red/green codes
collapsed vs healthy.

Panel (b): PR vs iteration for one representative collapsed cell (CW s0) and
one healthy cell (CLN s0), showing that collapse is flat from the start.

Every plotted number is read at build time from the per-run
``outputs/runs/session34/<run>/metrics.jsonl`` diagnostic series (key
``diag/pr`` at key ``step``) and from ``outputs/session34/trackc_gates.json``
(pre-registered PR floor). Nothing is hardcoded.

Run (repo root, CPU):
    OMP_NUM_THREADS=4 .venv/bin/python scripts/session35/fig_cube_health_v4.py
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
from scripts.session34.trackc_cells import CELLS, RUNS_BASE  # noqa: E402

GATES_JSON = REPO_ROOT / "outputs/session34/trackc_gates.json"
OUT_PDF = REPO_ROOT / "paper/sections/figures/results/fig_cube_health_v4.pdf"
# Session 39 (Carlos's assessment, fig 6): the main-text cube figure keeps only the
# final-PR panel; the PR-versus-iteration training history moves to the supplement.
OUT_PDF_HISTORY = REPO_ROOT / "paper/sections/figures/results/fig_cube_history_v4.pdf"

# Paper order of the 2x2x2 conditioning-cube cells (no-L half first).
# Session 39 (D-A): the main-text conditioning figure is trimmed to the L and W
# axes (the N/near-body cells move to the appendix with the CLN increment).
CELL_ORDER = ("c0", "cw", "cl", "clw")
CELL_LABEL = {"c0": "C0", "cn": "CN", "cw": "CW", "cwn": "CWN",
              "cl": "CL", "cln": "CLN", "clw": "CLW", "clwn": "CLWN"}
NO_L_CELLS = frozenset({"c0", "cw"})
SEEDS = (0, 1, 2)

# Representative trajectories for panel (b): one collapsed, one healthy.
TRAJ_CELLS = (("cw", 0), ("clw", 0))


def pr_series(run_name: str) -> tuple[np.ndarray, np.ndarray]:
    """(steps, PR) diagnostic series from a run's metrics.jsonl."""
    path = RUNS_BASE / run_name / "metrics.jsonl"
    steps, prs = [], []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if "diag/pr" in rec:
                steps.append(rec["step"])
                prs.append(rec["diag/pr"])
    if not prs:
        raise KeyError(f"no diag/pr records in {path}")
    return np.asarray(steps), np.asarray(prs)


def main() -> None:
    use_style()

    # Pre-registered PR floor (0.3 * d, d = 32; scripts/session34/trackc_gates.py:85),
    # read back from the frozen gates JSON rather than retyped.
    gates = json.loads(GATES_JSON.read_text())
    pr_floor = gates["pre_registered"]["pr_floor"]

    final_pr = {cell: [pr_series(CELLS[cell][s])[1][-1] for s in SEEDS]
                for cell in CELL_ORDER}
    traj = {(cell, seed): pr_series(CELLS[cell][seed]) for cell, seed in TRAJ_CELLS}

    fig_w = TEXTWIDTH_IN
    # Main-text figure: the final-PR panel alone. History panel is a separate
    # single-panel supplementary figure (built below).
    fig_a, ax_a = plt.subplots(1, 1, figsize=(fig_w * 0.64, fig_w * 0.50))
    fig_b, ax_b = plt.subplots(1, 1, figsize=(fig_w * 0.55, fig_w * 0.44))

    # ---- panel (a): final PR per cell, 3 seeds with jitter -----------------
    jitter = (-0.16, 0.0, 0.16)
    for i, cell in enumerate(CELL_ORDER):
        color = FAMILY_COLOR["fukami"] if cell in NO_L_CELLS else FAMILY_COLOR["jepa"]
        for j, s in enumerate(SEEDS):
            ax_a.plot(i + jitter[j], final_pr[cell][j], marker="o", ms=3.6,
                      mfc=color, mec=color, mew=0.5, ls="none", alpha=0.9,
                      zorder=3)
    ax_a.axhline(pr_floor, color="0.35", ls=(0, (4, 3)), lw=0.9, zorder=2)
    ax_a.text(len(CELL_ORDER) - 0.55, pr_floor + 0.35, "PR floor",
              ha="right", va="bottom", fontsize=6.5, color="0.35")

    ax_a.set_xticks(range(len(CELL_ORDER)))
    ax_a.set_xticklabels([CELL_LABEL[c] for c in CELL_ORDER], fontsize=6.5,
                         rotation=30, ha="right", rotation_mode="anchor")
    ax_a.set_xlim(-0.6, len(CELL_ORDER) - 0.4)
    ax_a.set_ylim(0, None)
    ax_a.set_ylabel(r"participation ratio $\mathrm{PR}(z)$")
    ax_a.set_xlabel("conditioning cell")

    handles = [
        plt.Line2D([], [], color=FAMILY_COLOR["jepa"], marker="o", ls="none",
                   ms=3.6, label="with lift head"),
        plt.Line2D([], [], color=FAMILY_COLOR["fukami"], marker="o", ls="none",
                   ms=3.6, label="without lift head"),
    ]
    ax_a.legend(handles=handles, loc="upper left", fontsize=6.5,
                handletextpad=0.4, borderaxespad=0.3)
    ax_a.text(0.985, 0.02, "3 seeds per cell", transform=ax_a.transAxes,
              ha="right", va="bottom", fontsize=6, color="0.35")

    # ---- panel (b): PR vs iteration, collapsed vs healthy -----------------
    traj_style = {("cw", 0): (FAMILY_COLOR["fukami"], "s", "CW (s0)"),
                  ("clw", 0): (FAMILY_COLOR["jepa"], "o", "CLW (s0)")}
    for key, (steps, prs) in traj.items():
        color, marker, label = traj_style[key]
        ax_b.plot(steps / 1000.0, prs, color=color, marker=marker, ms=3.0,
                  lw=1.0, mew=0.5, label=label, zorder=3)
    ax_b.axhline(pr_floor, color="0.35", ls=(0, (4, 3)), lw=0.9, zorder=2)
    ax_b.set_xlabel(r"iteration ($\times 10^{3}$)")
    ax_b.set_ylabel(r"$\mathrm{PR}(z)$")
    ax_b.set_ylim(0, None)
    ax_b.legend(loc="center right", fontsize=6.5, handletextpad=0.4,
                borderaxespad=0.3)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    for fig, out in ((fig_a, OUT_PDF), (fig_b, OUT_PDF_HISTORY)):
        fig.tight_layout()
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")
    print(f"pr_floor = {pr_floor}")
    for cell in CELL_ORDER:
        print(f"  {CELL_LABEL[cell]:5s} final PR: "
              + ", ".join(f"{v:.2f}" for v in final_pr[cell]))


if __name__ == "__main__":
    main()
