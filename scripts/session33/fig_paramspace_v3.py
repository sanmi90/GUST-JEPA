"""Figure F2 (v3 manuscript): parameter-space sampling of the split v2.2.

Source spec: SESSION_33_MANUSCRIPT_V3.md Section 9, item F2 ("parameter-space
sampling of the v2.2 split (update; symmetric Test C)"). Port of the v2.1
figure `scripts/session21/figB_paramspace_v2p1.py` to split v2.2
(`configs/splits/split_v2p2.json`, 102 cases / 450 encounters).

Design choices (kept from v2.1 unless noted):
  * Three 2D projections of the (G, D, Y) cube, points coloured by split and
    test-B tier using the shared `figstyle` split palette and markers.
  * Test-B tiers are read from the `tier` field that split_v2p2.json carries
    for every test_b case (6 interior + 4 boundary), the same criterion the
    v2.1 script used; nothing is hardcoded.
  * Headline change vs v2.1: Test C is SYMMETRIC, 4 cases at G = +4
    (periodic) and 4 at G = -4 (run4). Both extrapolation boundaries are
    marked with dashed lines at G = +4 AND G = -4 in the G projections
    (v2.1 drew only the one-sided G = +4 line), so the two-sided |G| = 4
    envelope is immediately visible.

Output: outputs/session33/figures/fig_paramspace_v3.{pdf,png}.
CPU-only; no GPU, no cache access. Run: taskset -c 16-23 python
scripts/session33/fig_paramspace_v3.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/session21"))

import figstyle as fs  # noqa: E402

SPLIT = REPO / "configs/splits/split_v2p2.json"
OUT_PDF = REPO / "outputs/session33/figures/fig_paramspace_v3.pdf"
OUT_PNG = REPO / "outputs/session33/figures/fig_paramspace_v3.png"


def tier_key(case: dict) -> str:
    """Map a case record to a split/tier key of the figstyle split palette."""
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
    counts = {k: len(v) for k, v in pts.items() if v}
    pts = {k: np.array(v) for k, v in pts.items() if v}

    fig, axes = plt.subplots(1, 3, figsize=fs.figure_size(1.0, aspect=0.40))
    proj = [(0, 1, "$G$", "$D$"), (0, 2, "$G$", "$Y/c$"), (1, 2, "$D$", "$Y/c$")]
    # draw order: training underneath, held-out on top
    order = ["train", "test_b_interior", "test_b_boundary", "test_c"]
    # Physical legend labels (Carlos's assessment, fig 2): archive tier names
    # (test_b interior/boundary, test_c) are internal; the figure reads physically.
    phys_label = {"train": "training",
                  "test_b_interior": "in-distribution test",
                  "test_b_boundary": "edge-of-training test",
                  "test_c": r"$|G|=4$ extrapolation test"}
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
                       label=phys_label[key])
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.axvline(0, color="0.85", lw=0.6, zorder=0)
        if yl == "$Y/c$":
            ax.axhline(0, color="0.85", lw=0.6, zorder=0)
        # explicit xticks: the |G|=4 extrapolation boundary must be readable
        # directly off the axis (not just the dashed marker line), and the D
        # axis should show exactly the three sampled core diameters
        if xl == "$G$":
            ax.set_xticks(range(-4, 5))
        elif xl == "$D$":
            ax.set_xticks([0.5, 1.0, 1.5])
    # mark the SYMMETRIC |G|=4 extrapolation boundary (both signs, v2.2)
    for ax, (a, b, xl, yl) in zip(axes, proj):
        if xl == "$G$":
            for g_ood in (-4, 4):
                ax.axvline(g_ood, color=fs.SPLIT_COLOR["test_c"], lw=0.7,
                           ls=(0, (3, 2)), zorder=1)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.04), columnspacing=1.3, handletextpad=0.25)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=200)
    total = sum(counts.values())
    print(f"case counts by split/tier: {counts} (total {total})")
    print(f"wrote {OUT_PDF}\nwrote {OUT_PNG}")


if __name__ == "__main__":
    main()
