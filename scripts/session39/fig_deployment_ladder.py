#!/usr/bin/env python3
"""The deployment ladder (Carlos, 2026-07-11): the money exhibit tying Part II
together. Field fidelity is a wash; the predictive state is the only family that
holds all the way down the deployment chain -- read the observable, forecast it,
recover it from the wall, assimilate it. Colour = verdict (holds / partial /
fails); annotation = the representative metric, each macro-bound in its own
table. Sources per rung in the caption."""
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.session21.figstyle import TEXTWIDTH_IN, use_style  # noqa: E402

OUT = REPO / "paper/sections/figures/results/fig_deployment_ladder.pdf"

RUNGS = [
    "decode the field\n(SSIM, impact)",
    "read the wake\n(readability $R^2$)",
    "forecast it\n(through impact, $h{=}8$)",
    "recover it from the wall\n(observable $R^2$)",
    "assimilate it\n(filter $R^2$, $|G|{=}4$)",
]
FAMS = ["predictive\n(JEPA)", "reconstruction\n(Fukami)", "linear\n(POD)"]
# verdict: 2 holds, 1 partial, 0 fails ; label = the macro-bound number
V = np.array([
    [2, 1, 2],   # field decode
    [2, 0, 0],   # read the wake
    [2, 0, 1],   # forecast
    [2, 1, 1],   # wall recovery
    [2, 0, 0],   # assimilate
])
L = [
    ["0.81", "0.72", "0.82"],
    ["0.75", "$-$0.09", "0.19"],
    ["0.63", "0.00", "0.38"],
    ["0.59", "0.47", "0.53"],
    ["0.84", "diverges", "0.25"],
]


def main():
    use_style()
    cmap = ListedColormap(["#c0392b", "#e5a83a", "#3f8f5f"])  # fail / partial / hold
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 0.62 * TEXTWIDTH_IN))
    ax.imshow(V, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            ax.text(j, i, L[i][j], ha="center", va="center", fontsize=9,
                    color="white", fontweight="bold")
    ax.set_xticks(range(len(FAMS)))
    ax.set_xticklabels(FAMS, fontsize=8.5)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(range(len(RUNGS)))
    ax.set_yticklabels(RUNGS, fontsize=8.5)
    ax.set_xticks(np.arange(-.5, len(FAMS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(RUNGS), 1), minor=True)
    ax.grid(which="minor", color="white", lw=2.5)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("field fidelity is a wash; the gap opens and widens toward deployment",
                 fontsize=8.5, style="italic", color="#555", pad=26)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
