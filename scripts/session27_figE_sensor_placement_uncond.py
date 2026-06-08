"""UNCONDITIONED sensor placement (Session 27 copy of figE_sensor_placement.py).

Where the optimal (TCSI) taps sit for the fully-unconditioned (no-c) JEPA latent,
and that the selector matters most at very low K.

(a) The NACA 0012 surface with the TCSI taps for K=2/4/8 marked.
(b) Unconditioned JEPA d=64 latent recovery (test_b R2) versus sensor count for
    TCSI vs qDEIM.

Inputs : outputs/session27/pressure_noc_tf/{sensor_picks_v2,method_comparison_v2}.json
Output : outputs/session27/figs_uncond/figE_sensor_placement.pdf (+ .png)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "session21"))
import figstyle as fs  # noqa: E402

PV = REPO / "outputs/session27/pressure_noc_tf"
AIRFOIL = np.load(REPO / "outputs/session21/airfoil_xy.npy")
OUT_PDF = REPO / "outputs/session27/figs_uncond/figE_sensor_placement.pdf"
OUT_PNG = REPO / "outputs/session27/figs_uncond/figE_sensor_placement.png"


def main() -> None:
    fs.use_style()
    picks = json.load(open(PV / "sensor_picks_v2.json"))["TCSI"]
    mc = json.load(open(PV / "method_comparison_v2.json"))

    fig, (axa, axb) = plt.subplots(1, 2, figsize=fs.figure_size(1.0, aspect=0.42),
                                   gridspec_kw={"width_ratios": [1.25, 1.0]})

    # (a) airfoil + TCSI taps, nested by K
    axa.plot(AIRFOIL[:, 0], AIRFOIL[:, 1], color="0.25", lw=1.0, zorder=2)
    axa.fill(AIRFOIL[:, 0], AIRFOIL[:, 1], color="0.9", zorder=1)
    ranked = picks["8"]  # greedy order; index 0 = rank 1 = most informative
    n = len(ranked)
    pts = AIRFOIL[ranked]
    base = plt.cm.Blues(np.linspace(0.93, 0.38, n))  # dark -> light
    lcmap = ListedColormap(base)
    norm = BoundaryNorm(np.arange(n + 1) - 0.5, n)
    sc = axa.scatter(pts[:, 0], pts[:, 1], c=np.arange(n), cmap=lcmap, norm=norm,
                     s=42, marker="o", edgecolors="0.25", linewidths=0.5, zorder=3)
    axa.set_aspect("equal")
    axa.set_xlim(-0.08, 1.05); axa.set_ylim(-0.16, 0.16)
    axa.set_xlabel("$x/c$"); axa.set_ylabel("$y/c$")
    axa.set_title("TCSI sensor placement (no-c)", fontsize=8)
    cb = fig.colorbar(sc, ax=axa, location="bottom", fraction=0.07, pad=0.30,
                      aspect=30)
    cb.set_ticks([0, n - 1]); cb.set_ticklabels(["most", "least"])
    cb.set_label("TCSI relevance rank", fontsize=6, labelpad=1.5)
    cb.ax.tick_params(labelsize=6, length=2, pad=1)

    # (b) method comparison vs K (unconditioned JEPA d=64 latent recovery, test_b)
    styles = {"TCSI": ("#1b7837", "o", "-", "TCSI (ours)"),
              "qDEIM": ("#762a83", "s", "--", "qDEIM")}
    for m, (col, mk, ls, lab) in styles.items():
        pts = sorted((x["K"], x["R2_z"]) for x in mc
                     if x["method"] == m and x["split"] == "test_b")
        K = [p[0] for p in pts]; r2 = [p[1] for p in pts]
        axb.plot(K, r2, marker=mk, ms=3.5, lw=1.0, ls=ls, color=col, label=lab)
    axb.set_xscale("log", base=2); axb.set_xticks([2, 4, 8, 16])
    axb.set_xticklabels(["2", "4", "8", "16"])
    axb.set_xlabel("sensors $K$"); axb.set_ylabel("JEPA state recovery $R^2$")
    axb.set_ylim(0.3, 0.95)
    axb.legend(loc="lower right", handletextpad=0.4, borderpad=0.2)

    fig.tight_layout()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF); fig.savefig(OUT_PNG, dpi=200)
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
