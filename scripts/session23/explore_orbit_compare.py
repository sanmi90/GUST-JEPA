"""Honest baseline-vs-gust orbit comparison, two candidate depictions.

The two latent orbits live in different subspaces at very different scales, so no
single low-D projection shows both cleanly. Two honest attempts:

(A) 3-D frame built FROM the comparison: floor axes = the baseline limit-cycle
    plane (its top 2 PCs); vertical axis = the direction in which the gust most
    departs that plane (top PC of the gust residual orthogonal to the baseline
    plane). Baseline ring grey on the floor; gust time-coloured. Honest IF
    labelled this way.

(B) Projection-free: distance from the gust trajectory to the baseline orbit
    point cloud (full 64-D), per frame, in units of the baseline orbit diameter,
    with the baseline's own RMS thickness as a grey reference band. This shows the
    departure and the (incomplete, per S4.4) return without any projection.

Diagnostic; writes a PNG to eyeball. No .tex touched.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

REPO = Path("/home/carlos/GUST-JEPA")
LAT = REPO / "outputs/session18/exp_b1/latents_jepa_d64_test1_noBN"
OUT = REPO / "outputs/session23/orbit_compare.png"
STAGES = [-8, 0, 16, 32]


def main() -> None:
    tr = np.load(LAT / "train.npz", allow_pickle=True)
    tb = np.load(LAT / "test_b.npz", allow_pickle=True)
    base = tr["z_full"][np.isclose(tr["G"], 0.0)]            # (n_base, 120, 64)
    base_flat = base.reshape(-1, 64)
    gi = int(np.argmax(np.abs(tb["G"])))
    gust = tb["z_full"][gi]                                  # (120, 64)
    impact = 40
    f = np.arange(120)

    # baseline plane (its own top-2 PCs); report how planar the baseline is
    pca_b = PCA(n_components=3).fit(base_flat)
    evr_b = pca_b.explained_variance_ratio_
    mu = base_flat.mean(0)
    W2 = pca_b.components_[:2]                                # (2, 64) baseline plane
    # gust residual orthogonal to the baseline plane -> departure direction
    gust_c = gust - mu
    inplane = gust_c @ W2.T                                  # (120, 2)
    resid = gust_c - inplane @ W2                            # (120, 64)
    dep_dir = PCA(n_components=1).fit(resid).components_[0]   # (64,)
    gust_dep = gust_c @ dep_dir                               # (120,)
    base_c = base_flat - mu
    base_inplane = base_c @ W2.T
    base_dep = base_c @ dep_dir
    one_base = base[0] - mu
    ring_xy = one_base @ W2.T
    ring_z = one_base @ dep_dir

    # distance of each gust frame to the baseline orbit cloud (full 64-D)
    d_gust = cdist(gust, base_flat).min(1)                   # (120,)
    diam = np.linalg.norm(base_flat - mu, axis=1).max() * 2  # rough orbit diameter
    # baseline self-distance (leave-one-out nearest neighbour) for the thickness band
    dd = cdist(base_flat, base_flat)
    np.fill_diagonal(dd, np.inf)
    base_self = dd.min(1)
    band = np.percentile(base_self, 95) / diam

    fig = plt.figure(figsize=(12, 4.6))

    # (A) 3-D comparison frame
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.scatter(base_inplane[:, 0], base_inplane[:, 1], base_dep, s=3,
               color="0.7", alpha=0.5, label="baseline cloud")
    ax.plot(ring_xy[:, 0], ring_xy[:, 1], ring_z, color="0.35", lw=1.2,
            label="baseline cycle")
    ax.scatter(inplane[:, 0], inplane[:, 1], gust_dep, c=f, cmap="viridis", s=12)
    ax.scatter(inplane[0, 0], inplane[0, 1], gust_dep[0], color="k", s=35, marker="o")
    ax.scatter(inplane[-1, 0], inplane[-1, 1], gust_dep[-1], color="r", s=35, marker="s")
    ax.set_xlabel("baseline PC1"); ax.set_ylabel("baseline PC2")
    ax.set_zlabel("gust departure")
    ax.set_title(f"(A) 3-D: baseline plane + gust departure\n"
                 f"baseline planarity (PC1+PC2) = {evr_b[:2].sum():.2f}", fontsize=9)
    ax.view_init(elev=20, azim=-65)
    ax.legend(fontsize=6, loc="upper left")

    # (B) distance-to-baseline-orbit vs frame
    axb = fig.add_subplot(1, 2, 2)
    trel = f - impact
    axb.axhspan(0, band, color="0.85", label="baseline orbit thickness")
    axb.plot(trel, d_gust / diam, color="#1b7837", lw=1.5)
    for n, s in enumerate(STAGES, start=1):
        axb.axvline(s, color="0.8", lw=0.6, ls="--")
        axb.text(s, (d_gust / diam).max() * 1.02, str(n), ha="center", fontsize=7, color="0.4")
    axb.set_xlabel("frames relative to impact")
    axb.set_ylabel("distance to baseline orbit\n(/ orbit diameter)")
    axb.set_title("(B) projection-free: departs, partial return", fontsize=9)
    axb.set_xlim(-40, 79); axb.legend(fontsize=7, loc="upper right")

    fig.suptitle(f"baseline vs gust orbit, encounter {tb['case_id'][gi]} "
                 "(gust colour = frame; black=start, red=end)", fontsize=10)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print(f"wrote {OUT}  baseline EVR={evr_b.round(2)}  band={band:.3f}  "
          f"max d_gust/diam={ (d_gust/diam).max():.3f}  final={ (d_gust/diam)[-1]:.3f}")


if __name__ == "__main__":
    main()
