#!/usr/bin/env python3
"""Decode SSIM at the impact instant vs latent dimension d (Carlos: "for more d").
Reads outputs/session39/multi_d_critical_ssim.json. Two panels, full field and
near-body band. Shows that the compact nonlinear states and the energy-optimal
linear basis all render the field well and improve with d, while the
published-recipe lineage stays low; field decode is not the discriminator."""
from pathlib import Path
import json
import sys

import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.session21.figstyle import TEXTWIDTH_IN, use_style  # noqa: E402

OUT = REPO / "paper/sections/figures/results/fig_critical_ssim_dim.pdf"
DIMS = [4, 8, 16, 32]
STYLE = {"predictive": ("#1b7837", "o", "predictive (wake)"),
         "AE (wake)": ("#762a83", "s", "AE (wake)"),
         "Fukami (wake)": ("#e08214", "^", "Fukami (wake)"),
         "POD": ("#2166ac", "D", "POD")}


def main():
    use_style()
    d = json.load(open(REPO / "outputs/session39/multi_d_critical_ssim.json"))["families"]
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 0.44 * TEXTWIDTH_IN),
                             sharey=True)
    for ax, mask, title in [(axes[0], "full", "full field"),
                            (axes[1], "nearbody", "near-body band")]:
        for fam, (c, mk, lab) in STYLE.items():
            xs, ys = [], []
            for dd in DIMS:
                r = d.get(fam, {}).get(str(dd))
                if r and f"impact_{mask}" in r:
                    xs.append(dd); ys.append(r[f"impact_{mask}"])
            if xs:
                ax.plot(xs, ys, marker=mk, color=c, ms=4, lw=1.1,
                        label=lab if mask == "full" else None)
        ax.set_xscale("log", base=2)
        ax.set_xticks(DIMS); ax.set_xticklabels(DIMS)
        ax.set_xlabel("latent dimension $d$")
        ax.set_title(title, fontsize=8)
    axes[0].set_ylabel("decode SSIM at impact")
    axes[0].legend(fontsize=6, loc="lower right", frameon=False)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
