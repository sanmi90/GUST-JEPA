"""Phase-amplitude figure (figC_cycle_v2p1.pdf): the gust encounter in the
predictive latent. v2.1, two panels (panel c dropped per the v2.1 caption).

Matches the v2.1 caption (fig:phase_amplitude, section_4_results.tex):
  (a) Distance of the gust trajectory to the baseline orbit (full latent, in units
      of the orbit diameter; the grey band is the baseline orbit thickness): the
      encounter departs the baseline cycle and only partially relaxes back within
      the inter-gust window.
  (b) The orbit phase relative to the impact frame (Hilbert analytic signal of the
      leading PC of the predictive latent). The numbered stages are those of
      fig:traces.

The baseline (no-gust) orbit is the pooled predictive-latent trajectory of the
Baseline (no-gust) train encounters; the gust trajectory is a representative strong
test_b encounter projected against that orbit in the full 64-D latent. Predictive
(jepa_tf_noc) seed 42 latents (the only no-gust orbit latents produced this
campaign; an s46 latents dir does not exist).

CPU only; numpy/scipy/sklearn. Read-only on the latents; writes only the PDF.

Output: paper/sections/figures/results/figC_cycle_v2p1.pdf
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import hilbert  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

import _common as cm  # noqa: E402

fs = cm.fs
LAT = cm.SESSION28 / "latents" / "jepa_tf_noc_d64_s42"
STAGES = [-8, 0, 16, 32]  # frames relative to impact; glyphs 1..4


def main() -> None:
    fs.use_style()
    tr = np.load(LAT / "train.npz", allow_pickle=True)
    tb = np.load(LAT / "test_b.npz", allow_pickle=True)

    base = tr["z_full"][np.isclose(tr["G"].astype(float), 0.0)]  # (n_base, 120, d)
    base_flat = base.reshape(-1, base.shape[-1])
    pca = PCA(n_components=2).fit(base_flat)

    tb_cid = np.array([str(c) for c in tb["case_ids"]])
    G = tb["G"].astype(float)
    rep_cid = tb_cid[int(np.argmax(np.abs(G)))]  # representative strong gust
    gi = int(np.flatnonzero(tb_cid == rep_cid)[0])
    impact = int(tb["impact_frame"][gi])
    gust64 = tb["z_full"][gi]  # (120, d)

    jepa = fs.FAMILY_COLOR["jepa"]
    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=fs.figure_size(1.0, aspect=0.42), layout="constrained"
    )

    # (a) departure: per-frame min distance to the baseline orbit cloud, in
    # baseline-orbit-diameter units; grey band = orbit thickness (95th-pctl
    # nearest-neighbour spacing within the orbit).
    d_gust = cdist(gust64, base_flat).min(axis=1)
    mu = base_flat.mean(0)
    diam = np.linalg.norm(base_flat - mu, axis=1).max() * 2.0
    dd = cdist(base_flat, base_flat)
    np.fill_diagonal(dd, np.inf)
    band = float(np.percentile(dd.min(1), 95) / diam)
    dn = d_gust / diam
    trel = np.arange(120) - impact
    axa.axhspan(0, band, color="0.88", zorder=0)
    axa.plot(trel, dn, color=jepa, lw=1.4, zorder=2)
    for n, s in enumerate(STAGES, start=1):
        axa.axvline(s, color="0.8", lw=0.5, ls="--", zorder=0)
        axa.text(s, dn.max() * 1.04, str(n), ha="center", fontsize=5.5, color="0.45")
    axa.set_xlim(-12, 79)
    axa.set_ylim(0, dn.max() * 1.14)
    axa.set_xlabel("frames relative to impact")
    axa.set_ylabel("distance to baseline orbit\n(orbit diameters)")
    axa.set_title("(a) departure", fontsize=8, loc="left")

    # (b) orbit phase relative to impact, via Hilbert analytic signal of PC1.
    gust_pc = pca.transform(gust64)
    sig = gust_pc[:, 0] - gust_pc[:, 0].mean()
    theta = np.unwrap(np.angle(hilbert(sig)))
    theta = (theta - theta[impact]) % (2 * np.pi)
    axb.plot(trel, theta, color=jepa, lw=1.1)
    for n, s in enumerate(STAGES, start=1):
        axb.axvline(s, color="0.8", lw=0.6, ls="--", zorder=0)
        axb.text(s, 2 * np.pi * 1.04, str(n), ha="center", fontsize=5.5, color="0.4")
    axb.set_xlim(-12, 40)
    axb.set_ylim(0, 2 * np.pi * 1.12)
    axb.set_xticks([0, 20, 40])
    axb.set_yticks([0, np.pi, 2 * np.pi])
    axb.set_yticklabels(["0", r"$\pi$", r"$2\pi$"])
    axb.set_xlabel("frames relative to impact")
    axb.set_ylabel(r"orbit phase $\theta$")
    axb.set_title("(b) orbit phase", fontsize=8, loc="left")

    out = cm.out_path("figC_cycle")
    fig.savefig(out)
    print(f"wrote {out}")
    print(
        f"  representative encounter {rep_cid} (G={G[gi]:+.2f}); "
        f"orbit band {band:.3f}, peak departure {dn.max():.2f} orbit-diameters"
    )


if __name__ == "__main__":
    main()
