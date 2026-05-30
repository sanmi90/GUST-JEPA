"""Consolidated reconstruction figure: replaces the three separate figS_recon_*
panels with one new-style exhibit.

One idea: at the impact frame the reconstructive autoencoder collapses toward a
near-uniform field while the predictive decode localises the impingement; the
linear basis is amplitude-accurate, and all families degrade in the |G|=4 regime.
Rows are the three held-out splits, columns the simulation and the three decodes.
Source: outputs/session20/decoded/{test_a,test_b,test_c}.npz.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import figstyle as fs

REPO = Path(__file__).resolve().parents[2]
DEC = REPO / "outputs/session20/decoded"
OUT_PDF = REPO / "paper/sections/figures/results/figD_reconstructions.pdf"
OUT_PNG = REPO / "outputs/session21/figs/figD_reconstructions.png"

ROWS = [("test_a", "test A\n(trained gust)"),
        ("test_b", "test B\n(interpolation)"),
        ("test_c", r"test C ($|G|{=}4$)")]
COLS = [("target_norm", "simulation"), ("jepa_norm", "predictive"),
        ("fukami_norm", "reconstructive"), ("pod_norm", "POD")]
IMPACT_IDX = 1   # offsets [-8,0,8,16,24,32,40]; index 1 = impact (offset 0)


def main() -> None:
    fs.use_style()
    fig, axes = plt.subplots(3, 4, figsize=fs.figure_size(1.0, aspect=0.52))
    im = None
    for r, (split, rlab) in enumerate(ROWS):
        d = np.load(DEC / f"{split}.npz", allow_pickle=True)
        G = d["G"]
        # representative gust encounter: strongest |G| (excludes the no-gust case)
        rep = int(np.argmax(np.abs(G)))
        for c, (key, clab) in enumerate(COLS):
            ax = axes[r, c]
            im = fs.vort_panel(ax, d[key][rep, IMPACT_IDX])
            if r == 0:
                ax.set_title(clab, fontsize=7.5)
            if c == 0:
                ax.text(-0.06, 0.5, rlab, transform=ax.transAxes, rotation=90,
                        ha="right", va="center", fontsize=7)

    fig.subplots_adjust(left=0.07, right=0.9, top=0.92, bottom=0.03,
                        wspace=0.06, hspace=0.08)
    cax = fig.add_axes([0.915, 0.2, 0.013, 0.55])
    fig.colorbar(im, cax=cax, label=r"$\omega_z$ (norm.)")

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=200)
    print(f"wrote {OUT_PDF.name}")


if __name__ == "__main__":
    main()
