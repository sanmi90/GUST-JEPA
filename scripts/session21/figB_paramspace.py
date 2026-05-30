"""SPEC 3 - NEW FIG B: the parameter-space sampling.

One idea: the split is stratified, the |G|=4 set is the out-of-distribution
boundary, and training mass concentrates near Y=0 (which is why Y is weakly
resolved later). Three 2D projections of the (G, D, Y) cube, points coloured by
split and test-B tier. Source: configs/splits/split_v2.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import figstyle as fs

REPO = Path(__file__).resolve().parents[2]
SPLIT = REPO / "configs/splits/split_v2.json"
OUT_PDF = REPO / "paper/sections/figures/results/figB_paramspace.pdf"
OUT_PNG = REPO / "outputs/session21/figs/figB_paramspace.png"


def tier_key(case: dict) -> str:
    s = case["split"]
    if s == "train":
        return "train"
    if s == "test_c":
        return "test_c"
    t = (case.get("tier") or "").lower()
    return "test_b_boundary" if "bound" in t else "test_b_interior"


def main() -> None:
    fs.use_style()
    cases = json.load(open(SPLIT))["cases"]
    pts = {k: [] for k in fs.SPLIT_COLOR}
    for c in cases.values():
        pts[tier_key(c)].append((c["G"], c["D"], c["Y"]))
    pts = {k: np.array(v) for k, v in pts.items() if v}

    fig, axes = plt.subplots(1, 3, figsize=fs.figure_size(1.0, aspect=0.40))
    proj = [(0, 1, "$G$", "$D$"), (0, 2, "$G$", "$Y/c$"), (1, 2, "$D$", "$Y/c$")]
    # draw order: training underneath, held-out on top
    order = ["train", "test_b_interior", "test_b_boundary", "test_c"]
    for ax, (a, b, xl, yl) in zip(axes, proj):
        for key in order:
            if key not in pts:
                continue
            P = pts[key]
            ax.scatter(P[:, a], P[:, b], s=26 if key != "train" else 18,
                       c=fs.SPLIT_COLOR[key], marker=fs.SPLIT_MARKER[key],
                       edgecolors="white", linewidths=0.4,
                       alpha=0.95 if key != "train" else 0.7,
                       zorder=3 if key != "train" else 2,
                       label=fs.SPLIT_TIER_LABEL[key])
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.axvline(0, color="0.85", lw=0.6, zorder=0)
        if yl == "$Y/c$":
            ax.axhline(0, color="0.85", lw=0.6, zorder=0)
    # mark the |G|=4 OOD boundary on the G projections
    for ax, (a, b, xl, yl) in zip(axes, proj):
        if xl == "$G$":
            ax.axvline(4, color=fs.SPLIT_COLOR["test_c"], lw=0.7,
                       ls=(0, (3, 2)), zorder=1)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.04), columnspacing=1.3, handletextpad=0.25)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=200)
    print(f"wrote {OUT_PDF}\nwrote {OUT_PNG}")


if __name__ == "__main__":
    main()
