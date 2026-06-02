"""Graphical abstract: the headline in one figure.

Top row: mid-plane vorticity 16 frames after gust impact for the strong negative
gust, simulation against the predictive (JEPA) and reconstructive (AE) decodes at
matched d=64. Bottom: forecast wake-enstrophy closure (held-out R^2 at H=16) for
the three encoder families; only the predictive latent clears the
predict-the-mean floor. Numbers are the validated values quoted in section 4.1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/carlos/GUST-JEPA")
sys.path.insert(0, str(REPO / "scripts" / "session21"))
import figstyle as fs  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

PDF = REPO / "paper/sections/figures/results/graphical_abstract.pdf"
PNG = REPO / "outputs/session23/graphical_abstract.png"
CASE = "G-3.00_D1.50_Y-0.10"
WAKE_R2 = {"jepa": 0.449, "fukami": -0.478, "pod": -0.089}  # forecast H=16, d=64 (sec 4.1)


def main() -> None:
    fs.use_style()
    dec = np.load(REPO / "outputs/session20/decoded/test_b.npz", allow_pickle=True)
    cid = np.array([str(c) for c in dec["case_ids"]])
    offs = list(dec["offsets"])
    gi = int(np.where(cid == CASE)[0][0])
    oi = offs.index(16)

    fig = plt.figure(figsize=fs.figure_size(1.0, 0.66))
    gs = GridSpec(2, 3, height_ratios=[2.1, 1.0], hspace=0.32, wspace=0.06,
                  left=0.04, right=0.985, top=0.88, bottom=0.13)

    panels = [("target_norm", "simulation", "oracle"),
              ("jepa_norm", "predictive (JEPA)", "jepa"),
              ("fukami_norm", "reconstructive (AE)", "fukami")]
    for c, (key, title, fam) in enumerate(panels):
        ax = fig.add_subplot(gs[0, c])
        fs.vort_panel(ax, dec[key][gi, oi])
        ax.set_title(title, fontsize=9, color=fs.FAMILY_COLOR[fam], pad=3)

    axb = fig.add_subplot(gs[1, :])
    fams = ["jepa", "fukami", "pod"]
    vals = [WAKE_R2[f] for f in fams]
    ypos = np.arange(len(fams))[::-1]
    axb.barh(ypos, vals, color=[fs.FAMILY_COLOR[f] for f in fams], height=0.6)
    axb.axvline(0, color="k", lw=0.8)
    axb.text(0.02, len(fams) - 0.5, "predict-the-mean floor", fontsize=6.5,
             rotation=90, va="top", ha="left", color="0.35")
    for y, v in zip(ypos, vals):
        axb.text(v + (0.03 if v >= 0 else -0.03), y, f"{v:+.2f}",
                 va="center", ha="left" if v >= 0 else "right", fontsize=8)
    axb.set_yticks(ypos)
    axb.set_yticklabels([fs.FAMILY_LABEL[f] for f in fams], fontsize=8)
    axb.set_xlabel(r"forecast wake-enstrophy closure $R^2$ at $H=16$ (held-out)", fontsize=8)
    axb.set_xlim(-0.75, 0.75)
    for s in ("top", "right", "left"):
        axb.spines[s].set_visible(False)
    axb.tick_params(left=False)

    fig.suptitle("A predictive latent forecasts the gust wake; the reconstructive "
                 "latent collapses (matched $d=64$, $16$ frames after impact)",
                 fontsize=9.5, y=0.965)
    PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF)
    fig.savefig(PNG, dpi=150)
    print(f"wrote {PDF}\nwrote {PNG}")


if __name__ == "__main__":
    main()
