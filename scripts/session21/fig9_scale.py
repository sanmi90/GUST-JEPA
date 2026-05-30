"""SPEC 6 - Fig 9 (de-cluttered): Gaussian scale decomposition.

One idea: the predictive decode retains the large-scale leading-edge vortex and
shear layer that the reconstructive decode smooths away, and its large-scale wake
enstrophy tracks the simulation through the staged encounter. The baked-in
"|G|=4 is 3-D" annotation moves to the caption. Triptych (top) + staged-enstrophy
curve (bottom). Source: scale_decomp.json, decoded/test_b.npz.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

import figstyle as fs

REPO = Path(__file__).resolve().parents[2]
SD = REPO / "outputs/session20/scale_decomp/scale_decomp.json"
DEC = REPO / "outputs/session20/decoded/test_b.npz"
OUT_PDF = REPO / "paper/sections/figures/results/fig_scale_decomp.pdf"
OUT_PNG = REPO / "outputs/session21/figs/fig_scale_decomp.png"

CURVE = [("dns", "oracle", "simulation"), ("jepa_d64", "jepa", "predictive (JEPA)"),
         ("fukami", "fukami", "reconstructive"), ("pod", "pod", "POD")]


def main() -> None:
    fs.use_style()
    sd = json.load(open(SD))
    sigma = sd["config"]["sigma_px"]
    offsets = sd["config"]["offsets"]
    h16 = offsets.index(16)
    dec = np.load(DEC, allow_pickle=True)
    rep = int(np.argmax(np.abs(dec["G"])))   # strong gust, clearest LEV

    fig = plt.figure(figsize=fs.figure_size(1.0, aspect=0.62))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.25], hspace=0.35, wspace=0.08)

    fields = [("target_norm", "simulation"), ("jepa_norm", "predictive decode"),
              ("fukami_norm", "reconstructive decode")]
    for j, (key, title) in enumerate(fields):
        ax = fig.add_subplot(gs[0, j])
        large = gaussian_filter(dec[key][rep, h16].astype(float), sigma=sigma)
        fs.vort_panel(ax, large)
        ax.set_title(title, fontsize=7.5)

    axc = fig.add_subplot(gs[1, :])
    for key, fam, lab in CURVE:
        st = sd["splits"]["test_b"]["staged"][key]
        mean = np.asarray(st["mean"], float)
        std = np.asarray(st["std"], float)
        col = fs.FAMILY_COLOR[fam]
        lw = 1.6 if key == "dns" else 1.0
        axc.plot(offsets, mean, color=col, lw=lw,
                 marker="o" if key != "dns" else None, ms=3, label=lab,
                 zorder=4 if key == "dns" else 3)
        axc.fill_between(offsets, mean - std, mean + std, color=col, alpha=0.12,
                         lw=0)
    axc.axvline(0, color="0.85", lw=0.6, zorder=0)
    axc.set_xlabel("frames relative to impact")
    axc.set_ylabel("large-scale wake enstrophy")
    axc.legend(loc="upper left", ncol=2, columnspacing=1.2, handletextpad=0.4,
               borderpad=0.2)
    axc.margins(x=0.02)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=200)
    print(f"wrote {OUT_PDF}\nwrote {OUT_PNG}")


if __name__ == "__main__":
    main()
