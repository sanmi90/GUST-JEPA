#!/usr/bin/env python3
"""Talk version of the graphical abstract (slide 2).

Same content and data as scripts/session23/graphical_abstract.py, but with larger
fonts and a horizontal "predict-the-mean floor" note (the paper version puts it as a
tiny rotated label on the zero line, which is unreadable once shrunk onto a slide).
Output: figs/graphical_abstract_talk.png
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "session21"))
import figstyle as fs  # noqa: E402

OUT = Path(__file__).resolve().parent / "figs" / "graphical_abstract_talk.png"
CASE = "G-3.00_D1.50_Y-0.10"
WAKE_R2 = {"jepa": 0.449, "fukami": -0.478, "pod": -0.089}  # forecast H=16, d=64 (sec 4.1)


def main() -> None:
    fs.use_style()
    dec = np.load(REPO / "outputs/session20/decoded/test_b.npz", allow_pickle=True)
    cid = np.array([str(c) for c in dec["case_ids"]])
    offs = list(dec["offsets"])
    gi = int(np.where(cid == CASE)[0][0])
    oi = offs.index(16)

    fig = plt.figure(figsize=(9.4, 4.7))
    gs = GridSpec(2, 3, height_ratios=[2.0, 1.1], hspace=0.5, wspace=0.06,
                  left=0.15, right=0.99, top=0.95, bottom=0.17)

    panels = [("target_norm", "simulation", "oracle"),
              ("jepa_norm", "predictive (JEPA)", "jepa"),
              ("fukami_norm", "reconstructive (AE)", "fukami")]
    for c, (key, title, fam) in enumerate(panels):
        ax = fig.add_subplot(gs[0, c])
        fs.vort_panel(ax, dec[key][gi, oi], vlim=2.0)
        ax.set_title(title, fontsize=15, color=fs.FAMILY_COLOR[fam], pad=5)

    axb = fig.add_subplot(gs[1, :])
    fams = ["jepa", "fukami", "pod"]
    vals = [WAKE_R2[f] for f in fams]
    ypos = np.arange(len(fams))[::-1]
    axb.barh(ypos, vals, color=[fs.FAMILY_COLOR[f] for f in fams], height=0.62)
    axb.axvline(0, color="k", lw=1.1)
    axb.text(0.02, 2.78, "0 = predict-the-mean floor", fontsize=12, va="bottom",
             ha="left", color="0.35")
    for y, v in zip(ypos, vals):
        axb.text(v + (0.03 if v >= 0 else -0.03), y, f"{v:+.2f}",
                 va="center", ha="left" if v >= 0 else "right", fontsize=14)
    axb.set_yticks(ypos)
    axb.set_yticklabels([fs.FAMILY_LABEL[f] for f in fams], fontsize=13)
    axb.set_xlabel(r"forecast wake-enstrophy closure $R^2$ at $H=16$ (held-out)", fontsize=13)
    axb.set_xlim(-0.75, 0.85)
    axb.set_ylim(-0.5, 3.05)
    axb.tick_params(axis="x", labelsize=11)
    for s in ("top", "right", "left"):
        axb.spines[s].set_visible(False)
    axb.tick_params(left=False)

    fig.savefig(OUT, dpi=200)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
