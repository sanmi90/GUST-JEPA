"""Single 3D Isomap plot: G as colour, D as marker shape.

Loads the saved Isomap embedding (Z_iso) from isomap_diagnostic.npz and
the physical parameters from disentanglement.npz, then draws one 3D
scatter where both factors are visible at the same time:

- Marker COLOUR encodes G (continuous coolwarm colormap).
- Marker SHAPE encodes D in {0.0, 0.5, 1.0, 1.5}.

This is the disentanglement diagnostic in a single figure: a clean
G-gradient along one Iso axis + clustering by marker shape along
another would be visual evidence that the JEPA latent disentangles
strength from diameter.

Usage::

    python scripts/session11_isomap_g_color_d_marker.py \\
        --bundle outputs/runs/session11/W0_C_lam100/decoder_pca_k12/disentanglement.npz \\
        --iso    outputs/runs/session11/W0_C_lam100/decoder_pca_k12/isomap_diagnostic.npz \\
        --output outputs/runs/session11/W0_C_lam100/decoder_pca_k12/isomap_g_color_d_marker.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True, type=str,
                   help="disentanglement.npz (provides G, D, Y, splits)")
    p.add_argument("--iso", required=True, type=str,
                   help="isomap_diagnostic.npz (provides Z_iso)")
    p.add_argument("--output", required=True, type=str)
    p.add_argument("--components", type=int, nargs=3, default=[0, 1, 2])
    p.add_argument("--elev", type=float, default=22.0)
    p.add_argument("--azim", type=float, default=-65.0)
    return p.parse_args()


_D_MARKER = {  # D -> (marker, base size)
    0.0: ("X", 70),   # baseline, no gust
    0.5: ("o", 60),
    1.0: ("s", 60),
    1.5: ("^", 75),
}


def main() -> None:
    args = parse_args()
    b = np.load(args.bundle, allow_pickle=True)
    iso = np.load(args.iso, allow_pickle=True)
    G = b["G"].astype(np.float32)
    D = b["D"].astype(np.float32)
    Z_iso = iso["Z_iso"].astype(np.float32)
    cx, cy, cz = args.components
    print(f"[plot] n={len(G)}, Z_iso.shape={Z_iso.shape}, components=({cx},{cy},{cz})")
    print(f"[plot] D unique: {sorted(set(D.tolist()))}, "
          f"G range: [{G.min():.2f}, {G.max():.2f}]")

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(vmin=float(G.min()), vmax=float(G.max()))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    for d_val, (marker, size) in _D_MARKER.items():
        mask = np.isclose(D, d_val)
        if not mask.any():
            continue
        colors = sm.to_rgba(G[mask])
        ax.scatter(Z_iso[mask, cx], Z_iso[mask, cy], Z_iso[mask, cz],
                   c=colors, marker=marker, s=size,
                   edgecolor="k", linewidth=0.4, alpha=0.92,
                   label=f"D = {d_val:.1f} (n={int(mask.sum())})")

    ax.set_xlabel(f"Iso{cx+1}")
    ax.set_ylabel(f"Iso{cy+1}")
    ax.set_zlabel(f"Iso{cz+1}")
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_title(
        "Isomap embedding of W0_C_lam100 impact latents\n"
        "colour = G (gust strength), marker = D (gust diameter)",
        fontsize=11,
    )
    cb = fig.colorbar(sm, ax=ax, shrink=0.65, pad=0.10)
    cb.set_label("G (gust strength)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.95), fontsize=9,
              framealpha=0.9, title="D (gust diameter)")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {out}")


if __name__ == "__main__":
    main()
