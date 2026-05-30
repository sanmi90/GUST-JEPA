"""Shared figure style for the Session 21B JFM figure set.

One import, one visual language. Every 21B figure uses:
  * the same four family colours (predictive/JEPA, reconstructive/Fukami, POD,
    and a neutral DNS/DNS-oracle colour),
  * one marker per family,
  * serif/mathtext fonts at 7-8 pt to match the jfm body text,
  * line and marker weights tuned for the MEASURED text width of 360 pt = 5.0 in
    (article, 11pt, a4paper). Build at this width, include at scale 1.

Do not design large and let LaTeX shrink it: call ``figure_size(fraction)`` to
get inches at print size.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Measured once from the compiled document (\the\textwidth). Do not guess.
TEXTWIDTH_PT = 360.0
TEXTHEIGHT_PT = 595.8
PT_PER_INCH = 72.27  # TeX points
TEXTWIDTH_IN = TEXTWIDTH_PT / PT_PER_INCH   # ~4.98 in
TEXTHEIGHT_IN = TEXTHEIGHT_PT / PT_PER_INCH

# ---- the four-colour family key (fixed across the whole paper) --------------
# Colour-blind-safe, distinct in greyscale order. Keyed by canonical family.
FAMILY_COLOR = {
    "jepa": "#1b7837",    # green   - predictive / JEPA (this work)
    "fukami": "#c0392b",  # red     - reconstructive autoencoder
    "pod": "#2166ac",     # blue    - POD linear basis
    "oracle": "#404040",  # grey    - DNS / DNS-oracle reference
}
FAMILY_MARKER = {"jepa": "o", "fukami": "s", "pod": "^", "oracle": "D"}
FAMILY_LABEL = {"jepa": "predictive (JEPA)", "fukami": "reconstructive",
                "pod": "POD", "oracle": "DNS oracle"}

# baseline tag -> (family, latent dim d)
BASELINE = {
    "jepa_d64_test1_noBN": ("jepa", 64),
    "jepa_d32_noBN": ("jepa", 32),
    "fukami_d3_noBN": ("fukami", 3),
    "fukami_d32_noBN": ("fukami", 32),
    "fukami_d64_noBN": ("fukami", 64),
    "pod_d16_noBN": ("pod", 16),
    "pod_d32_noBN": ("pod", 32),
    "pod_d64_noBN": ("pod", 64),
}

# physical-observable display names (math in mathtext so it matches the body).
METRIC_LABEL = {
    "C_L": r"$C_L$", "C_D": r"$C_D$", "I_y": r"$I_y$",
    "wake_enstrophy": r"wake enstrophy $\Omega_w$",
    "circulation_pos": r"$\Gamma^{+}$", "circulation_neg": r"$\Gamma^{-}$",
}
SPLIT_LABEL = {"test_b": "in-distribution (test B)",
               "test_c": r"extrapolation $|G|=4$ (test C)"}

# split/tier palette for the data-sampling figure (distinct from the family key).
SPLIT_COLOR = {"train": "#9e9e9e", "test_b_interior": "#2166ac",
               "test_b_boundary": "#e08214", "test_c": "#b2182b"}
SPLIT_MARKER = {"train": "o", "test_b_interior": "s",
                "test_b_boundary": "D", "test_c": "^"}
SPLIT_TIER_LABEL = {"train": "training", "test_b_interior": "test B interior",
                    "test_b_boundary": "test B boundary",
                    "test_c": r"test C ($|G|=4$)"}

# Vorticity-snapshot convention (identical everywhere): red-blue diverging,
# symmetric fixed range, black airfoil, one shared colourbar.
VORT_CMAP = "RdBu_r"
VORT_VLIM = 3.0  # normalised omega in [-3, +3] (matches the pipeline 3-sigma scale)


def use_style() -> None:
    """Apply the shared rcParams. Call once at the top of every figure script."""
    mpl.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.serif": ["DejaVu Serif", "Computer Modern Roman"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.0,
        "lines.markersize": 4.5,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def figure_size(width_fraction: float = 1.0, aspect: float = 0.62) -> tuple[float, float]:
    """Inches at print size: width = fraction * textwidth, height = width * aspect."""
    w = TEXTWIDTH_IN * width_fraction
    return (w, w * aspect)


def family_color(tag: str, shade_by_d: bool = True) -> str:
    """Family colour for a baseline tag, optionally lightened for smaller d."""
    fam, d = BASELINE[tag]
    base = FAMILY_COLOR[fam]
    if not shade_by_d:
        return base
    # lighten toward white for the smaller-d variants so a family reads as a ramp
    ds = sorted({dd for (f, dd) in BASELINE.values() if f == fam})
    if len(ds) <= 1:
        return base
    frac = ds.index(d) / (len(ds) - 1)        # 0 (smallest d) -> 1 (largest d)
    rgb = np.array(mpl.colors.to_rgb(base))
    light = 0.55 - 0.55 * frac                 # smallest d lightened most
    return tuple(rgb + (1.0 - rgb) * light)


def stage_glyph(ax, x: float, y: float, n: int, color: str = "black",
                fontsize: float = 6.5, z: float = 5, s: float = 90) -> None:
    """Numbered stage glyph 1..4: a small filled circle with a white numeral.

    Byte-identical between NEW FIG A and NEW FIG C so the reader maps the
    abstract space to the physical flow with one shared key.
    """
    ax.scatter([x], [y], s=s, c=color, marker="o", zorder=z,
               edgecolors="white", linewidths=0.7)
    ax.text(x, y, str(n), color="white", ha="center", va="center",
            fontsize=fontsize, fontweight="bold", zorder=z + 1)


def _airfoil_xy() -> np.ndarray:
    """NACA 0012 surface polygon from Baseline.h5, cached to a small .npy."""
    import os
    from pathlib import Path
    cache = Path(__file__).resolve().parents[2] / "outputs/session21/airfoil_xy.npy"
    if cache.exists():
        return np.load(cache)
    import h5py
    raw = (Path(os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT")))
           / "data/raw/periodic/Baseline.h5")
    with h5py.File(raw, "r") as f:
        xy = np.asarray(f["airfoil_xy"])
    if not np.allclose(xy[0], xy[-1]):
        xy = np.vstack([xy, xy[0:1]])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, xy)
    return xy


def airfoil_pixels(H: int = 192, W: int = 96, x_ext=(-1.5, 4.5),
                   y_ext=(-1.5, 1.5)) -> np.ndarray:
    """Airfoil polygon in image-pixel coordinates for an omega.T panel."""
    xy = _airfoil_xy()
    px = (xy[:, 0] - x_ext[0]) * (H - 1) / (x_ext[1] - x_ext[0])
    py = (xy[:, 1] - y_ext[0]) * (W - 1) / (y_ext[1] - y_ext[0])
    return np.stack([px, py], axis=-1)


def vort_panel(ax, omega: np.ndarray, vlim: float = VORT_VLIM):
    """Draw one mid-plane vorticity snapshot in the shared convention:
    red-blue diverging, fixed symmetric range, solid black airfoil, no ticks.
    ``omega`` is (192, 96) in normalised units.
    """
    from matplotlib.patches import Polygon
    im = ax.imshow(omega.T, origin="lower", cmap=VORT_CMAP, vmin=-vlim,
                   vmax=vlim, aspect="equal", interpolation="nearest")
    ax.add_patch(Polygon(airfoil_pixels(), closed=True, facecolor="black",
                         edgecolor="black", lw=0.3, zorder=5))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def family_legend_handles(include_oracle: bool = True):
    """Proxy handles for the shared family legend, in paper order.

    The oracle is drawn as a dashed reference line in panels, so its legend
    entry is a dashed line, not a marker.
    """
    handles = []
    for fam in ("jepa", "fukami", "pod"):
        handles.append(plt.Line2D([], [], color=FAMILY_COLOR[fam],
                                  marker=FAMILY_MARKER[fam], linestyle="none",
                                  markersize=4.5, label=FAMILY_LABEL[fam]))
    if include_oracle:
        handles.append(plt.Line2D([], [], color=FAMILY_COLOR["oracle"],
                                  linestyle=(0, (4, 3)), linewidth=0.9,
                                  label="DNS-oracle floor"))
    return handles
