"""Sensor placement: where the optimal (TCSI) taps sit, and that the method
matters most at very low K.

(a) The NACA 0012 surface with the TCSI taps for K=2/4/8 marked: they cluster at
    the leading edge on both surfaces, where the gust impinges.
(b) JEPA d=64 latent recovery (test_b R2) versus sensor count for TCSI, qDEIM, and
    uniform placement: TCSI is decisively better at K=2 but the wall pressure is
    information-rich enough that the method barely matters by K>=4.
Source: outputs/session21/pressure_v2/{sensor_picks_v2,method_comparison_v2}.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import figstyle as fs

REPO = Path(__file__).resolve().parents[2]
PV = REPO / "outputs/session21/pressure_v2"
AIRFOIL = np.load(REPO / "outputs/session21/airfoil_xy.npy")
OUT_PDF = REPO / "paper/sections/figures/results/figE_sensor_placement.pdf"
OUT_PNG = REPO / "outputs/session21/figs/figE_sensor_placement.png"

KMARK = [2, 4, 8]
KCOL = {2: "#d6604d", 4: "#f4a582", 8: "#4393c3"}


def main() -> None:
    fs.use_style()
    picks = json.load(open(PV / "sensor_picks_v2.json"))["TCSI"]
    mc = json.load(open(PV / "method_comparison_v2.json"))

    fig, (axa, axb) = plt.subplots(1, 2, figsize=fs.figure_size(1.0, aspect=0.42),
                                   gridspec_kw={"width_ratios": [1.25, 1.0]})

    # (a) airfoil + TCSI taps, nested by K
    axa.plot(AIRFOIL[:, 0], AIRFOIL[:, 1], color="0.25", lw=1.0, zorder=2)
    axa.fill(AIRFOIL[:, 0], AIRFOIL[:, 1], color="0.9", zorder=1)
    for K in reversed(KMARK):
        idx = picks[str(K)]
        pts = AIRFOIL[idx]
        axa.scatter(pts[:, 0], pts[:, 1], s=24 + 8 * (8 - K), color=KCOL[K],
                    edgecolors="white", linewidths=0.4, zorder=3,
                    label=f"$K={K}$")
    axa.set_aspect("equal")
    axa.set_xlim(-0.08, 1.05); axa.set_ylim(-0.16, 0.16)
    axa.set_xlabel("$x/c$"); axa.set_ylabel("$y/c$")
    axa.set_title("TCSI sensor placement", fontsize=8)
    axa.legend(loc="upper right", handletextpad=0.2, borderpad=0.2, ncol=3,
               columnspacing=0.8)
    axa.annotate("leading edge", (0.0, 0.0), textcoords="offset points",
                 xytext=(18, -22), fontsize=6, color="0.4",
                 arrowprops=dict(arrowstyle="->", color="0.6", lw=0.6))

    # (b) method comparison vs K (JEPA d=64 latent recovery, test_b)
    styles = {"TCSI": ("#1b7837", "o", "-", "TCSI (optimal)"),
              "qDEIM": ("#762a83", "s", "--", "qDEIM"),
              "uniform": ("0.5", "^", ":", "uniform")}
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
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF); fig.savefig(OUT_PNG, dpi=200)
    print(f"wrote {OUT_PDF.name}")


if __name__ == "__main__":
    main()
