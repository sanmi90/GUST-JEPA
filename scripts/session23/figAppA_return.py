"""Appendix A figure: diameter-controlled return-to-baseline sweep.

Backs the load-bearing caveat in section 4.4 that the non-return of the gust
trajectory within the encounter window is a window-length limitation set by the
gust-release cadence and, at fixed core diameter, is not strength-dependent.
Holds D=1.0 and sign(G)<0 fixed, Y as constant as the data allows, averages the
distance-to-baseline-orbit over each case's encounters, and varies only |G|.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

REPO = Path("/home/carlos/GUST-JEPA")
sys.path.insert(0, str(REPO / "scripts" / "session21"))
import figstyle as fs  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

LAT = REPO / "outputs/session18/exp_b1/latents_jepa_d64_test1_noBN"
OUT = REPO / "paper/sections/figures/results/appA_orbit_return.pdf"
IMPACT = 40


def main() -> None:
    fs.use_style()
    tr = np.load(LAT / "train.npz", allow_pickle=True)
    G, D, Y, zf = tr["G"], tr["D"], tr["Y"], tr["z_full"]
    base = zf[np.isclose(G, 0.0)].reshape(-1, 64)
    diam = np.linalg.norm(base - base.mean(0), axis=1).max() * 2.0
    dd = cdist(base, base)
    np.fill_diagonal(dd, np.inf)
    band = float(np.percentile(dd.min(1), 95) / diam)

    Dfix, Ytarget = 1.0, 0.0
    levels = [-0.5, -1.0, -1.5, -2.0, -3.0]
    trel = np.arange(120) - IMPACT
    fig, ax = plt.subplots(figsize=fs.figure_size(1.0, 0.52))
    ax.axhspan(0, band, color="0.88", lw=0, label="baseline orbit thickness")
    cmap = plt.cm.viridis(np.linspace(0.12, 0.88, len(levels)))
    for c, g in zip(cmap, levels):
        sub = np.where(np.isclose(G, g) & np.isclose(D, Dfix))[0]
        if len(sub) == 0:
            continue
        yv = Y[sub][np.argmin(np.abs(Y[sub] - Ytarget))]
        m = np.isclose(G, g) & np.isclose(D, Dfix) & np.isclose(Y, yv)
        mean = np.stack([cdist(zf[j], base).min(1) / diam for j in np.where(m)[0]]).mean(0)
        ax.plot(trel, mean, color=c, lw=1.3, label=f"$G={g:+.1f}$")
    ax.axvline(0, color="k", lw=0.6, ls=":")
    ax.text(0, 3.62, "impact", fontsize=7, ha="center")
    ax.axvline(79, color="0.55", lw=0.6, ls="--")
    ax.text(78, 3.62, "next gust", fontsize=7, ha="right", color="0.45")
    ax.set_xlabel("frames relative to impact")
    ax.set_ylabel("distance to baseline orbit ($/$ diameter)")
    ax.set_xlim(-40, 79)
    ax.set_ylim(0, 3.85)
    ax.legend(fontsize=6.5, ncol=3, loc="lower left", frameon=False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"diam={diam:.3f} band={band:.3f} D={Dfix} Y~{Ytarget}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
