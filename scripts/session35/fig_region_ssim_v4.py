"""F11: Region SSIM per Track C cell (fig_region_ssim_v4).

Grouped bars per conditioning cell (Track C cube order C0, CN, CW, CWN, CL,
CLN, CLW, CLWN, plus the AE anchors AE-L, AE-W, AE-LW): three bars per cell
for the near-body / wake / full decode-SSIM masks. Every number is loaded from
outputs/session34/trackc_region_ssim.json at build time.

The source JSON holds ONE decode run per cell (seed s0 decoders), so no seed
whiskers are drawn; instead thin whiskers show the encounter IQR over the 42
test_b encounters (labelled on the figure). Collapsed cells (the no-L
predictive cells C0, CN, CW, CWN; HANDOFF D253-D258) sit on a grey background
band and carry hatched bars.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "session21"))
import figstyle  # noqa: E402

SRC = REPO / "outputs/session34/trackc_region_ssim.json"
OUT_DIR = REPO / "paper/sections/figures/results"
OUT_PDF = OUT_DIR / "fig_region_ssim_v4.pdf"
OUT_PNG = OUT_DIR / "fig_region_ssim_v4.png"

# Cube order first (no-L collapsed cells leftmost), then AE anchors.
CELL_ORDER = ["c0", "cn", "cw", "cwn", "cl", "cln", "clw", "clwn", "ae_l", "ae_w", "ae_lw"]
CELL_LABEL = {
    "c0": "C0", "cn": "CN", "cw": "CW", "cwn": "CWN",
    "cl": "CL", "cln": "CLN", "clw": "CLW", "clwn": "CLWN",
    "ae_l": "AE-L", "ae_w": "AE-W", "ae_lw": "AE-LW",
}
COLLAPSED = {"c0", "cn", "cw", "cwn"}  # every no-L predictive cell collapses (D253-D258)
MASKS = ["nearbody", "wake", "full"]
MASK_LABEL = {"nearbody": "near-body", "wake": "wake", "full": "full frame"}
# lightness ramp within a family colour: near-body darkest, full lightest
MASK_LIGHTEN = {"nearbody": 0.0, "wake": 0.35, "full": 0.62}


def lighten(color: str, frac: float):
    rgb = np.array(mpl.colors.to_rgb(color))
    return tuple(rgb + (1.0 - rgb) * frac)


def main() -> None:
    with open(SRC) as f:
        data = json.load(f)
    proto = data["protocol"]
    results = data["results"]
    split = proto["split"]

    cells = [c for c in CELL_ORDER if c in results]
    missing = [c for c in CELL_ORDER if c not in results]
    if missing:
        print(f"note: cells absent from JSON, skipped: {missing}")

    figstyle.use_style()
    fig, ax = plt.subplots(figsize=figstyle.figure_size(1.0, aspect=0.46))

    width = 0.26
    xs = np.arange(len(cells))
    n_frames = sorted({results[c]["n_frames"] for c in cells})

    for j, mask in enumerate(MASKS):
        vals, err_lo, err_hi, colors, hatches = [], [], [], [], []
        for c in cells:
            entry = results[c]
            v = entry["ssim"][mask]
            per_enc = np.array([e[mask] for e in entry["per_encounter"]])
            q25, q75 = np.percentile(per_enc, [25, 75])
            vals.append(v)
            err_lo.append(max(v - q25, 0.0))
            err_hi.append(max(q75 - v, 0.0))
            fam = "fukami" if c.startswith("ae_") else "jepa"
            colors.append(lighten(figstyle.FAMILY_COLOR[fam], MASK_LIGHTEN[mask]))
            hatches.append("////" if c in COLLAPSED else None)
        bars = ax.bar(
            xs + (j - 1) * width, vals, width=width, color=colors,
            edgecolor="black", linewidth=0.4,
            yerr=[err_lo, err_hi], error_kw=dict(lw=0.6, capsize=1.4, capthick=0.6,
                                                 ecolor="0.25"),
        )
        for b, h in zip(bars, hatches):
            if h:
                b.set_hatch(h)

    # grey background band over the collapsed cells
    idx_collapsed = [i for i, c in enumerate(cells) if c in COLLAPSED]
    if idx_collapsed:
        lo, hi = min(idx_collapsed) - 0.5, max(idx_collapsed) + 0.5
        ax.axvspan(lo, hi, color="0.88", zorder=0)
        ax.text((lo + hi) / 2.0, 0.965, "collapsed (no L)", ha="center", va="top",
                transform=ax.get_xaxis_transform(), fontsize=7, color="0.35")

    # separator before the AE anchors
    idx_ae = [i for i, c in enumerate(cells) if c.startswith("ae_")]
    if idx_ae:
        ax.axvline(min(idx_ae) - 0.5, color="0.5", lw=0.7, ls=(0, (2, 2)))
        ax.text(min(idx_ae) - 0.35, 0.965, "AE anchors", ha="left", va="top",
                transform=ax.get_xaxis_transform(), fontsize=7, color="0.35")

    ax.set_xticks(xs)
    ax.set_xticklabels([CELL_LABEL[c] for c in cells])
    ax.set_ylabel("decoded SSIM")
    ax.set_ylim(0.0, 0.92)
    ax.set_xlim(-0.6, len(cells) - 0.4)

    # legend: mask shading (JEPA green ramp; AE anchors use the same ramp in red)
    handles = [
        mpl.patches.Patch(facecolor=lighten(figstyle.FAMILY_COLOR["jepa"],
                                            MASK_LIGHTEN[m]),
                          edgecolor="black", linewidth=0.4, label=MASK_LABEL[m])
        for m in MASKS
    ]
    handles.append(mpl.patches.Patch(facecolor="white", edgecolor="black",
                                     linewidth=0.4, hatch="////",
                                     label="collapsed cell"))
    ax.legend(handles=handles, ncol=4, loc="lower left",
              bbox_to_anchor=(0.0, 1.01), frameon=False, handlelength=1.4,
              columnspacing=1.0)

    nf = n_frames[0] if len(n_frames) == 1 else n_frames
    ax.text(0.995, -0.13, f"{split}, {nf} frames pooled; one decoder seed per cell; "
                          "whiskers: encounter IQR (42 enc)",
            ha="right", va="top", transform=ax.transAxes, fontsize=6.2,
            color="0.35")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=200)
    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
