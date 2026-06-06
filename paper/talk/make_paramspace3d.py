#!/usr/bin/env python3
"""3D isometric view of the (G, D, Y) training envelope for the talk Data slide.

Clearer than the 2D projection: one point per case in gust-strength G, core
diameter D, wall-normal offset Y, coloured by split (train / test_b interpolation
/ test_c |G|=4 extrapolation). Output: figs/paramspace3d.png
"""
import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "session21"))
import figstyle as fs  # noqa: E402

SPLIT = REPO / "configs/splits/split_v2.json"
OUT = Path(__file__).resolve().parent / "figs" / "paramspace3d.png"
STYLE = {  # split -> (color, marker, size, label)
    "train":  ("#1F3864", "o", 34, "train (70 cases)"),
    "test_b": ("#C0520F", "D", 46, "test B, interpolation (10)"),
    "test_c": ("#B03A2E", "s", 60, "test C, |G| = 4 extrapolation (4)"),
}


def main():
    fs.use_style()
    cases = json.load(open(SPLIT))["cases"]
    pts = {k: [] for k in STYLE}
    for c in cases.values():
        pts.setdefault(c["split"], []).append((c["G"], c["D"], c["Y"]))

    fig = plt.figure(figsize=(7.4, 5.6))
    ax = fig.add_subplot(111, projection="3d")
    for split, (col, mk, sz, lab) in STYLE.items():
        if not pts.get(split):
            continue
        a = np.array(pts[split])
        ax.scatter(a[:, 0], a[:, 1], a[:, 2], c=col, marker=mk, s=sz,
                   edgecolors="white", linewidths=0.5, depthshade=True, label=lab)
    # baseline (no gust) marker
    ax.scatter([0], [0], [0], c="#2E7D4F", marker="*", s=180, edgecolors="white",
               linewidths=0.6, label="baseline (no gust)", zorder=10)

    ax.set_xlabel("gust strength  G", labelpad=8, fontsize=12)
    ax.set_ylabel("core diameter  D", labelpad=8, fontsize=12)
    ax.set_zlabel("offset  Y/c", labelpad=4, fontsize=12)
    ax.set_box_aspect((1.0, 0.7, 0.7))
    ax.view_init(elev=20, azim=-58)
    ax.xaxis.pane.set_alpha(0.04); ax.yaxis.pane.set_alpha(0.04); ax.zaxis.pane.set_alpha(0.04)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=9)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), fontsize=9.5,
              frameon=False, handletextpad=0.2, borderpad=0.2)
    fig.subplots_adjust(left=0.02, right=0.98, top=1.0, bottom=0.02)
    fig.savefig(OUT, dpi=200)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
