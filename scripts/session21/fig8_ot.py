"""SPEC 6 - Fig 8 (de-cluttered): optimal-transport reconstruction comparison.

Two panels, no "D-i"/"D-ii" sub-titles (the caption describes them):
 (a) OT field distance vs structural similarity at impact, one point per family:
     the linear basis has the highest SSIM but no transport advantage, and the
     collapsed reconstructive field is correctly penalised under transport.
 (b) Shepard alignment: per-pair latent distance vs simulation OT-geodesic
     distance; the predictive latent is the better isometry (higher Spearman).
Source: outputs/session20/ot/ot_results.json, ot/shepard_data.npz.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import figstyle as fs

REPO = Path(__file__).resolve().parents[2]
OTJ = REPO / "outputs/session20/ot/ot_results.json"
SHEP = REPO / "outputs/session20/ot/shepard_data.npz"
OUT_PDF = REPO / "paper/sections/figures/results/fig_ot.pdf"
OUT_PNG = REPO / "outputs/session21/figs/fig_ot.png"

# d_i method tag -> (family, label, label offset in points)
PANEL_A = [("jepa_d64", "jepa", r"JEPA $d{=}64$", (4, -11)),
           ("jepa_d32", "jepa", r"JEPA $d{=}32$", (6, 3)),
           ("fukami", "fukami", "reconstructive", (6, 3)),
           ("pod", "pod", "POD", (5, -3))]


def main() -> None:
    fs.use_style()
    ot = json.load(open(OTJ))
    shep = np.load(SHEP)
    di = ot["d_i"]["test_b"]["methods"]
    dii = ot["d_ii"]["per_method"]

    fig, (axa, axb) = plt.subplots(1, 2, figsize=fs.figure_size(1.0, aspect=0.48))

    # (a) OT vs SSIM at impact, one marker per family
    for tag, fam, lab, off in PANEL_A:
        x = di[tag]["ssim_impact"]
        y = di[tag]["ot_field_impact"]
        col = (fs.family_color("jepa_d64_test1_noBN") if tag == "jepa_d64" else
               fs.family_color("jepa_d32_noBN") if tag == "jepa_d32" else
               fs.FAMILY_COLOR[fam])
        axa.scatter(x, y, s=42, color=col, marker=fs.FAMILY_MARKER[fam],
                    edgecolors="white", linewidths=0.5, zorder=3)
        axa.annotate(lab, (x, y), textcoords="offset points", xytext=off,
                     fontsize=6.5)
    axa.set_xlabel("structural similarity (SSIM)")
    axa.set_ylabel("OT field distance")
    axa.set_xlim(0.44, 0.74)
    axa.annotate("similarity and transport\ndisagree", (0.55, 10.7),
                 fontsize=6, ha="center", color="0.4")

    # (b) Shepard: latent vs OT geodesic, predictive vs reconstructive
    rng = np.random.default_rng(0)
    for key, fam, lab in [("jepa_d64", "jepa", "predictive (JEPA)"),
                          ("fukami", "fukami", "reconstructive")]:
        ot_d = shep[f"{key}__ot"].astype(float)
        lat = shep[f"{key}__lat"].astype(float)
        # min-max normalise each axis so the two families overlay comparably
        ot_n = (ot_d - ot_d.min()) / (ot_d.ptp() + 1e-12)
        lat_n = (lat - lat.min()) / (lat.ptp() + 1e-12)
        idx = rng.choice(ot_n.size, size=min(1500, ot_n.size), replace=False)
        rho = dii[key]["spearman_mean"]
        axb.scatter(ot_n[idx], lat_n[idx], s=2.5, alpha=0.18,
                    color=fs.FAMILY_COLOR[fam], edgecolors="none", zorder=2,
                    label=fr"{lab}: $\rho={rho:.2f}$")
    axb.set_xlabel("simulation OT-geodesic distance (norm.)")
    axb.set_ylabel("latent distance (norm.)")
    leg = axb.legend(loc="upper left", markerscale=4, handletextpad=0.3,
                     borderpad=0.2)
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)

    fig.tight_layout()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=200)
    print(f"wrote {OUT_PDF}\nwrote {OUT_PNG}")


if __name__ == "__main__":
    main()
