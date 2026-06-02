"""Exploratory: is the predictive-latent gust orbit clearer in 3 coordinates?

Compares the 2-D projection (used in Fig 8a) against 3-D projections, coloured by
frame (early -> late), on the representative strong-gust test_b encounter. Two
projections: the baseline limit-cycle PCs (consistent with Fig 8a) and the gust
encounter's own PCs (which should show the loop with the least self-intersection).
Purely diagnostic; writes a PNG to eyeball, touches no .tex.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

REPO = Path("/home/carlos/GUST-JEPA")
LAT = REPO / "outputs/session18/exp_b1/latents_jepa_d64_test1_noBN"
OUT = REPO / "outputs/session23/latent_3d_explore.png"


def main() -> None:
    tr = np.load(LAT / "train.npz", allow_pickle=True)
    tb = np.load(LAT / "test_b.npz", allow_pickle=True)
    base = tr["z_full"][np.isclose(tr["G"], 0.0)]            # (n_base, 120, 64)
    gi = int(np.argmax(np.abs(tb["G"])))
    gust64 = tb["z_full"][gi]                                # (120, 64)

    pca_base = PCA(n_components=3).fit(base.reshape(-1, 64))
    base_pc = pca_base.transform(base[0])                    # one baseline episode
    gust_base = pca_base.transform(gust64)                   # gust in baseline PCs
    evr_base = pca_base.explained_variance_ratio_

    pca_gust = PCA(n_components=3).fit(gust64)
    gust_own = pca_gust.transform(gust64)                    # gust in its own PCs
    evr_gust = pca_gust.explained_variance_ratio_

    f = np.arange(gust64.shape[0])
    fig = plt.figure(figsize=(13, 4.4))
    panels = [
        ("3D, baseline PCs", gust_base, base_pc, evr_base),
        ("3D, gust-own PCs", gust_own, None, evr_gust),
    ]
    for k, (title, traj, bc, evr) in enumerate(panels):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        if bc is not None:
            ax.plot(bc[:, 0], bc[:, 1], bc[:, 2], color="0.65", lw=0.8, label="baseline")
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color="0.8", lw=0.5, zorder=1)
        ax.scatter(traj[:, 0], traj[:, 1], traj[:, 2], c=f, cmap="viridis", s=10, zorder=2)
        ax.scatter(*traj[0], color="k", s=30, marker="o")   # start
        ax.scatter(*traj[-1], color="r", s=30, marker="s")  # end
        ax.set_title(f"{title}\nvar {evr[0]:.2f}/{evr[1]:.2f}/{evr[2]:.2f}", fontsize=9)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
        ax.view_init(elev=22, azim=-60)

    # third panel: the gust-own 3D from a second angle
    ax = fig.add_subplot(1, 3, 3, projection="3d")
    ax.plot(gust_own[:, 0], gust_own[:, 1], gust_own[:, 2], color="0.8", lw=0.5)
    ax.scatter(gust_own[:, 0], gust_own[:, 1], gust_own[:, 2], c=f, cmap="viridis", s=10)
    ax.scatter(*gust_own[0], color="k", s=30, marker="o")
    ax.scatter(*gust_own[-1], color="r", s=30, marker="s")
    ax.set_title("3D, gust-own PCs (angle 2)", fontsize=9)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.view_init(elev=35, azim=130)

    fig.suptitle(
        f"Predictive-latent orbit, encounter {tb['case_id'][gi]} "
        "(black=start, red=end, colour=frame)", fontsize=10)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print(f"wrote {OUT}  base var3={evr_base.sum():.2f}  gust var3={evr_gust.sum():.2f}")


if __name__ == "__main__":
    main()
