"""3D scatter of (PC1, PC2, PC3) coloured by G and D.

Loads the cached output of session11_latent_disentanglement.py (the
.npz it writes alongside the disentanglement figure) and draws two
3D scatter panels side by side: left coloured by G, right coloured by
D. Different marker shapes per split (train, test_a, test_b, test_c)
so we can see whether held-out encounters stay on the same manifold.

Usage::

    python scripts/session11_latent_3d_scatter.py \\
        --bundle outputs/runs/session11/W0_C_lam100/decoder_pca_k12/disentanglement.npz \\
        --output outputs/runs/session11/W0_C_lam100/decoder_pca_k12/latent3d_gd.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (3D registration)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True, type=str,
                   help="disentanglement.npz from session11_latent_disentanglement.py")
    p.add_argument("--output", required=True, type=str)
    p.add_argument("--components", type=int, nargs=3, default=[0, 1, 2],
                   help="Which PC indices (0-based) to use as x, y, z.")
    p.add_argument("--elev", type=float, default=22.0)
    p.add_argument("--azim", type=float, default=-65.0)
    return p.parse_args()


_SPLIT_MARKER = {
    "train":  ("o", 26),
    "test_a": ("^", 36),
    "test_b": ("s", 50),
    "test_c": ("D", 60),
}


def _draw(ax, X, Y, Z, vals, splits, label, cmap):
    for split, (marker, size) in _SPLIT_MARKER.items():
        mask = splits == split
        if not mask.any():
            continue
        sc = ax.scatter(X[mask], Y[mask], Z[mask], c=vals[mask], cmap=cmap,
                        marker=marker, s=size, edgecolor="k", linewidth=0.3,
                        alpha=0.85, vmin=float(vals.min()), vmax=float(vals.max()),
                        label=f"{split} (n={int(mask.sum())})")
    return sc


def main() -> None:
    args = parse_args()
    b = np.load(args.bundle, allow_pickle=True)
    Zp = b["Z_pca"]  # (n, k)
    G = b["G"]; D = b["D"]; Y_param = b["Y"]
    splits = b["splits"]
    cx, cy, cz = args.components
    assert Zp.shape[1] >= max(cx, cy, cz) + 1, "components out of range"

    Xx = Zp[:, cx]; Xy = Zp[:, cy]; Xz = Zp[:, cz]
    print(f"[3d] components ({cx},{cy},{cz}); n={len(G)}; splits: "
          + ", ".join(f"{s}={int((splits == s).sum())}"
                       for s in ('train', 'test_a', 'test_b', 'test_c')))

    fig = plt.figure(figsize=(14, 6.5))
    axG = fig.add_subplot(1, 2, 1, projection="3d")
    axD = fig.add_subplot(1, 2, 2, projection="3d")

    scG = _draw(axG, Xx, Xy, Xz, G, splits, "G", "coolwarm")
    scD = _draw(axD, Xx, Xy, Xz, D, splits, "D", "viridis")

    for ax, sc, name in ((axG, scG, "G (gust strength)"),
                         (axD, scD, "D (gust diameter)")):
        ax.set_xlabel(f"PC{cx+1}")
        ax.set_ylabel(f"PC{cy+1}")
        ax.set_zlabel(f"PC{cz+1}")
        ax.view_init(elev=args.elev, azim=args.azim)
        ax.set_title(f"coloured by {name}")
        cb = fig.colorbar(sc, ax=ax, shrink=0.65, pad=0.10)
        cb.set_label(name.split()[0])
        # legend only once per panel, deduplicated
        h, lbls = ax.get_legend_handles_labels()
        if h:
            uniq = dict(zip(lbls, h))
            ax.legend(uniq.values(), uniq.keys(), loc="upper left",
                      bbox_to_anchor=(0.0, 0.95), fontsize=8, framealpha=0.85)

    fig.suptitle(
        f"Latent manifold of W0_C_lam100 in PC{cx+1}-PC{cy+1}-PC{cz+1} "
        f"(impact frame, average ±5)\n"
        f"282 encounters across train (165) / test_a (65) / test_b (28) / test_c (24)",
        fontsize=11, y=1.00,
    )
    plt.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[3d] saved {out}")


if __name__ == "__main__":
    main()
