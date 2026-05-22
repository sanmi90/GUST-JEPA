"""Isomap counterpart of session11_latent_disentanglement.py.

Loads the impact-frame-averaged latents Z_imp (n=282, d=32) saved by
the PCA diagnostic, fits Isomap (geodesic-distance-preserving nonlinear
embedding), and produces three diagnostics:

1. Residual variance vs n_components (Isomap "elbow").
2. Per-dim R^2 of the Isomap embedding against (G, D, Y).
3. 3D scatter of (Iso1, Iso2, Iso3) coloured by G and D.

Wang et al. (arXiv:2604.18059, 2026) use Isomap as a baseline against
their disentangled VAE; the parallel here is direct.

Usage::

    python scripts/session11_isomap_disentanglement.py \\
        --bundle outputs/runs/session11/W0_C_lam100/decoder_pca_k12/disentanglement.npz \\
        --output outputs/runs/session11/W0_C_lam100/decoder_pca_k12/isomap_diagnostic.png \\
        --n-neighbors 10 --n-components 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.manifold import Isomap


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True, type=str)
    p.add_argument("--output", required=True, type=str)
    p.add_argument("--n-neighbors", type=int, default=10)
    p.add_argument("--n-components", type=int, default=10,
                   help="Max embedding dim used in the R^2 heatmap.")
    p.add_argument("--max-components-curve", type=int, default=12,
                   help="Sweep n_components from 1 to this for residual curve.")
    return p.parse_args()


def per_axis_r2(Z: np.ndarray, y: np.ndarray) -> np.ndarray:
    Zc = Z - Z.mean(0, keepdims=True)
    yc = y - y.mean()
    num = (Zc * yc[:, None]).mean(0)
    var_z = (Zc ** 2).mean(0) + 1e-12
    var_y = (yc ** 2).mean() + 1e-12
    r = num / np.sqrt(var_z * var_y)
    return r ** 2


def multi_r2(Z: np.ndarray, y: np.ndarray) -> float:
    Zc = np.concatenate([np.ones((Z.shape[0], 1)), Z], axis=1)
    beta, *_ = np.linalg.lstsq(Zc, y, rcond=None)
    yhat = Zc @ beta
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum() + 1e-12
    return 1.0 - ss_res / ss_tot


def main() -> None:
    args = parse_args()
    b = np.load(args.bundle, allow_pickle=True)
    Z = b["Z_imp"].astype(np.float64)
    G = b["G"]; D = b["D"]; Yp = b["Y"]
    splits = b["splits"]
    n, d = Z.shape
    print(f"[iso] Z shape={Z.shape}; n_neighbors={args.n_neighbors}")

    # Residual variance curve: fit Isomap for k=1..K, get reconstruction_error.
    Kmax = args.max_components_curve
    rerr = np.zeros(Kmax, dtype=np.float64)
    for k_ in range(1, Kmax + 1):
        iso = Isomap(n_neighbors=args.n_neighbors, n_components=k_)
        iso.fit(Z)
        rerr[k_ - 1] = iso.reconstruction_error()
    # Normalize so curve starts at 1 (when k=1) and approaches 0.
    rerr_norm = rerr / (rerr[0] + 1e-12)
    print("[iso] residual error per k:", [f"{e:.4f}" for e in rerr_norm])

    # Full-dim embedding for downstream R^2 analysis.
    K = args.n_components
    iso_full = Isomap(n_neighbors=args.n_neighbors, n_components=K)
    Z_iso = iso_full.fit_transform(Z)  # (n, K)
    print(f"[iso] Z_iso shape={Z_iso.shape}")

    r2_iso = np.stack([per_axis_r2(Z_iso, v) for v in (G, D, Yp)], axis=0)  # (3, K)
    multi = [multi_r2(Z_iso, v) for v in (G, D, Yp)]
    print(f"[iso] full-subspace R^2 (K={K}): "
          f"G={multi[0]:.3f} D={multi[1]:.3f} Y={multi[2]:.3f}")

    # ---------------- figure ----------------
    fig = plt.figure(figsize=(13.5, 9.5))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.6],
                          width_ratios=[1, 1, 1, 0.04],
                          hspace=0.55, wspace=0.25)

    # Panel A: residual variance curve.
    axA = fig.add_subplot(gs[0, 0:3])
    axA.plot(np.arange(1, Kmax + 1), rerr_norm, "o-", color="#1f77b4")
    axA.set_xlabel("Isomap n_components")
    axA.set_ylabel("normalized residual reconstruction error")
    axA.set_xticks(np.arange(1, Kmax + 1))
    axA.set_title(f"Isomap residual variance vs n_components (n_neighbors={args.n_neighbors})")
    axA.grid(True, alpha=0.4)
    # mark drop below 0.1
    for k_ in range(1, Kmax + 1):
        if rerr_norm[k_ - 1] < 0.10:
            axA.axvline(k_, color="k", linestyle="--", alpha=0.5)
            axA.text(k_ + 0.1, 0.5, f"k={k_} reaches\nresidual<10%",
                     fontsize=8, va="center")
            break

    # Panel B: per-dim R^2 heatmap.
    axB = fig.add_subplot(gs[1, 0:3])
    imB = axB.imshow(r2_iso, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    axB.set_yticks(range(3)); axB.set_yticklabels(["G", "D", "Y"])
    axB.set_xticks(np.arange(0, K))
    axB.set_xticklabels([f"Iso{i+1}" for i in range(K)], fontsize=8)
    axB.set_xlabel(f"Isomap dimension (1 .. {K})")
    axB.set_title(
        f"Per-dim R$^2$, Isomap embedding (K={K})  |  full-subspace R$^2$: "
        f"G={multi[0]:.2f}, D={multi[1]:.2f}, Y={multi[2]:.2f}"
    )
    cbB = fig.add_subplot(gs[1, 3])
    fig.colorbar(imB, cax=cbB, label="R$^2$")

    # Panel C: 3D scatter coloured by G, D, Y.
    for ci, (name, v, cmap) in enumerate(
        (("G", G, "coolwarm"), ("D", D, "viridis"), ("Y", Yp, "PuOr"))
    ):
        ax = fig.add_subplot(gs[2, ci], projection="3d")
        sc = ax.scatter(Z_iso[:, 0], Z_iso[:, 1], Z_iso[:, 2], c=v, cmap=cmap,
                        s=22, alpha=0.9, edgecolor="k", linewidth=0.25)
        ax.set_xlabel("Iso1"); ax.set_ylabel("Iso2"); ax.set_zlabel("Iso3")
        ax.view_init(elev=22, azim=-65)
        ax.set_title(f"coloured by {name}", fontsize=10)
        fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.12)

    fig.suptitle(
        f"Isomap disentanglement diagnostic (n={n}, d_in={d}, n_neighbors={args.n_neighbors})\n"
        f"impact frame averaged latents, after Wang/Tirelli/Discetti/Ianiro arXiv:2604.18059 (2026)",
        fontsize=11, y=1.00,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    np.savez(out.with_suffix(".npz"), Z_iso=Z_iso, r2_iso=r2_iso, multi=multi,
             rerr_norm=rerr_norm, n_neighbors=args.n_neighbors)
    print(f"[iso] saved {out}")


if __name__ == "__main__":
    main()
