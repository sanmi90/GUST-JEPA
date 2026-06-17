"""FIG B (v2.1), 3D variant: the parameter-space sampling as one (G, D, Y) cube.

Single isometric 3D scatter of the split, coloured by split and test-B tier with
the same family key as the rest of the paper (figstyle), the |G|=4 out-of-
distribution plane drawn translucent, and a faint grey projection of every case
onto the floor (Y = min) so the concentration of training mass near Y/c = 0 stays
legible in print. Source: configs/splits/split_v2p1.json. Replaces the three 2D
projections of figB_paramspace_v2p1.py at the same output path.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402

import figstyle as fs  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SPLIT = REPO / "configs/splits/split_v2p1.json"
OUT_PDF = REPO / "paper/sections/figures/results/figB_paramspace_v2p1.pdf"
OUT_PNG = REPO / "outputs/session21/figs/figB_paramspace3d_v2p1.png"

ORDER = ["train", "test_b_interior", "test_b_boundary", "test_c"]


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
        # paper convention G = -s (dataset case G is +s)
        pts[tier_key(c)].append((-c["G"], c["D"], c["Y"]))
    pts = {k: np.array(v) for k, v in pts.items() if v}

    all_pts = np.concatenate([pts[k] for k in ORDER if k in pts])
    y_floor = all_pts[:, 2].min() - 0.06

    fig = plt.figure(figsize=fs.figure_size(0.78, aspect=0.82))
    ax = fig.add_subplot(111, projection="3d")

    # faint floor projection of every case (shows the Y/c ~ 0 concentration)
    ax.scatter(all_pts[:, 0], all_pts[:, 1], np.full(len(all_pts), y_floor),
               c="0.78", marker=".", s=10, alpha=0.6, depthshade=False, zorder=0)

    for key in ORDER:
        if key not in pts:
            continue
        P = pts[key]
        held = key != "train"
        ax.scatter(P[:, 0], P[:, 1], P[:, 2],
                   c=fs.SPLIT_COLOR[key], marker=fs.SPLIT_MARKER[key],
                   s=42 if held else 26, edgecolors="white",
                   linewidths=0.4, alpha=0.96 if held else 0.82,
                   depthshade=True, zorder=3 if held else 2,
                   label=fs.SPLIT_TIER_LABEL[key])

    # baseline (no gust) at the origin
    ax.scatter([0], [0], [0], c="#2E7D4F", marker="*", s=130,
               edgecolors="white", linewidths=0.5, depthshade=False,
               zorder=5, label="baseline (no gust)")

    # |G| = 4 out-of-distribution plane (paper convention G = -s: the held-out
    # |G|=4 set now sits at paper G = -4, so the marker plane follows)
    dlim = (all_pts[:, 1].min() - 0.1, all_pts[:, 1].max() + 0.1)
    ylim = (y_floor, all_pts[:, 2].max() + 0.06)
    dd, yy = np.meshgrid(np.linspace(*dlim, 2), np.linspace(*ylim, 2))
    ax.plot_surface(np.full_like(dd, -4.0), dd, yy, color=fs.SPLIT_COLOR["test_c"],
                    alpha=0.10, shade=False, zorder=1)

    ax.set_xlabel("$G$", labelpad=6)
    ax.set_ylabel("$D$", labelpad=6)
    ax.set_zlabel("$Y/c$", labelpad=2)
    ax.set_box_aspect((1.0, 0.62, 0.6))
    ax.view_init(elev=18, azim=-62)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0.04)
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, 0.99), ncol=1,
              handletextpad=0.2, borderpad=0.2, labelspacing=0.3)
    fig.subplots_adjust(left=0.0, right=0.98, top=1.0, bottom=0.0)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=200)
    print(f"wrote {OUT_PDF}\nwrote {OUT_PNG}")


if __name__ == "__main__":
    main()
