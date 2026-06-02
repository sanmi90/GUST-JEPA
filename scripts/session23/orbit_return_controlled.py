"""Diameter-controlled return-to-baseline sweep.

The earlier sweep (orbit_return_vs_strength.py) confounded |G| with core
diameter D, offset Y, and gust sign. Here we hold D and sign(G) fixed and Y as
constant as the data allows, average the distance-to-orbit over each case's
encounters (removing the encounter-0 start artifact), and vary only |G|. This
isolates whether "strong stalls, weak still contracting" is a G effect or was a
D effect.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402

REPO = Path("/home/carlos/GUST-JEPA")
LAT = REPO / "outputs/session18/exp_b1/latents_jepa_d64_test1_noBN"
OUT = REPO / "outputs/session23/orbit_return_controlled.png"
IMPACT = 40


def case_curve(z_full, G, D, Y, gval, dval, yval):
    """Mean distance-to-orbit curve over all encounters of the closest case."""
    m = np.isclose(G, gval) & np.isclose(D, dval) & np.isclose(Y, yval)
    if m.sum() == 0:
        return None
    return m


def main() -> None:
    tr = np.load(LAT / "train.npz", allow_pickle=True)
    G, D, Y = tr["G"], tr["D"], tr["Y"]
    zf = tr["z_full"]
    base = zf[np.isclose(G, 0.0)].reshape(-1, 64)
    mu = base.mean(0)
    diam = np.linalg.norm(base - mu, axis=1).max() * 2.0
    dd = cdist(base, base)
    np.fill_diagonal(dd, np.inf)
    band = float(np.percentile(dd.min(1), 95) / diam)

    Dfix = 1.0
    # choose the Y available for the most |G| levels at D=1.0, neg sign
    levels = [-0.5, -1.0, -1.5, -2.0, -3.0]
    cand_Y = sorted(set(np.round(Y[(np.isclose(D, Dfix)) & (G < 0)], 2)))
    bestY, bestn = None, -1
    for yv in cand_Y:
        n = sum(((np.isclose(G, g)) & np.isclose(D, Dfix) & np.isclose(Y, yv)).any() for g in levels)
        if n > bestn:
            bestY, bestn = yv, n
    print(f"diam={diam:.3f} band={band:.3f}  fixed D={Dfix}, Y={bestY} (covers {bestn}/{len(levels)} |G|)")

    trel = np.arange(120) - IMPACT
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.axhspan(0, band, color="0.88", label="baseline orbit thickness")
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(levels)))
    for c, g in zip(cmap, levels):
        m = np.isclose(G, g) & np.isclose(D, Dfix) & np.isclose(Y, bestY)
        if m.sum() == 0:  # fall back to nearest Y at this (G,D)
            sub = np.where(np.isclose(G, g) & np.isclose(D, Dfix))[0]
            if len(sub) == 0:
                continue
            yv = Y[sub][np.argmin(np.abs(Y[sub] - bestY))]
            m = np.isclose(G, g) & np.isclose(D, Dfix) & np.isclose(Y, yv)
            ytag = f"Y={yv:+.2f}"
        else:
            ytag = ""
        curves = np.stack([cdist(zf[j], base).min(1) / diam for j in np.where(m)[0]])
        mean = curves.mean(0)
        ax.plot(trel, mean, color=c, lw=1.6, label=f"G={g:+.1f} {ytag}")
        print(f"  G={g:+.1f} D={Dfix} {ytag or f'Y={bestY:+.2f}'}: n_enc={m.sum()} "
              f"d@max={mean.max():.2f} d@end={mean[-1]:.2f} slope@end={mean[-1]-mean[-6]:+.2f}")
    ax.axvline(0, color="k", lw=0.6, ls=":")
    ax.text(0, ax.get_ylim()[1] * 0.96, "impact", fontsize=7, ha="center")
    ax.axvline(79, color="0.5", lw=0.6, ls="--")
    ax.text(79, ax.get_ylim()[1] * 0.96, "next gust", fontsize=7, ha="right", color="0.4")
    ax.set_xlabel("frames relative to impact (window ends at +79)")
    ax.set_ylabel("distance to baseline orbit (/ diameter)")
    ax.set_title(f"Return vs |G| at FIXED D={Dfix}, Y={bestY:+.2f} (mean over encounters)")
    ax.legend(fontsize=7, ncol=2)
    ax.set_xlim(-40, 79)
    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
