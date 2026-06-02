"""Does the gust wake return to baseline within the 6 t/c inter-gust window?

Gusts are released every 120 frames (6 t/c at dt=0.05); impact is at frame ~40
(2 t/c), leaving ~4 t/c of observable recovery before the next release. Hypothesis
(C. Sanmiguel Vila): strong gusts never reach baseline because their recovery time
exceeds the inter-gust period, not because the latent fails to return. Test:
distance of the predictive latent to the baseline orbit over the full 120-frame
window, for a spread of |G|, with the baseline-orbit thickness band.
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
OUT = REPO / "outputs/session23/orbit_return_vs_strength.png"
IMPACT = 40


def main() -> None:
    tr = np.load(LAT / "train.npz", allow_pickle=True)
    tb = np.load(LAT / "test_b.npz", allow_pickle=True)
    base = tr["z_full"][np.isclose(tr["G"], 0.0)].reshape(-1, 64)
    mu = base.mean(0)
    diam = np.linalg.norm(base - mu, axis=1).max() * 2.0
    dd = cdist(base, base)
    np.fill_diagonal(dd, np.inf)
    band = float(np.percentile(dd.min(1), 95) / diam)

    G = tb["G"]
    cids = np.array([str(c) for c in tb["case_id"]])
    # one encounter per |G| level, spanning weak..strong
    levels = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.axhspan(0, band, color="0.88", label="baseline orbit thickness")
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(levels)))
    trel = np.arange(120) - IMPACT
    summary = []
    for c, lev in zip(cmap, levels):
        idx = np.where(np.isclose(np.abs(G), lev))[0]
        if len(idx) == 0:
            continue
        gi = int(idx[0])
        d = cdist(tb["z_full"][gi], base).min(1) / diam
        ax.plot(trel, d, color=c, lw=1.4, label=f"|G|={lev}")
        summary.append((lev, str(cids[gi]), float(d[0]), float(d.max()), float(d[-1]),
                        float(d[60:].min())))
    ax.axvline(0, color="k", lw=0.6, ls=":")
    ax.text(0, ax.get_ylim()[1] * 0.96, "impact", fontsize=7, ha="center")
    ax.axvline(79, color="0.5", lw=0.6, ls="--")
    ax.text(79, ax.get_ylim()[1] * 0.96, "next gust", fontsize=7, ha="right", color="0.4")
    ax.set_xlabel("frames relative to impact (window ends at +79 = next release)")
    ax.set_ylabel("distance to baseline orbit (/ diameter)")
    ax.set_title("Return to baseline vs gust strength (predictive latent)")
    ax.legend(fontsize=7, ncol=2)
    ax.set_xlim(-40, 79)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print(f"band(baseline thickness)={band:.3f}")
    print(f"{'|G|':>5} {'case':22} {'d@start':>8} {'d@max':>7} {'d@end':>7} {'d@min(post)':>11}")
    for lev, cid, d0, dmax, dend, dmin in summary:
        print(f"{lev:>5} {cid:22} {d0:>8.2f} {dmax:>7.2f} {dend:>7.2f} {dmin:>11.2f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
