"""F16: Training through the direct predictor (fig_rexpred_v4).

Single panel: pooled peak-region C_L R2 on test_b for the CLN baseline cells
(jepa_pool_ln_s0/s1/s2, outputs/session34/trackc_lift.json) vs the CLN-rexpred
retrains (jepa_pool_ln_rexpred s0/s1/s2, outputs/session35/rexpred_d32_band.json).
Individual seed points plus a mean bar per (cell, probe); linear probe filled
markers, MLP probe open markers. Every number is loaded from JSON at build
time. Bands annotated as mean +- sd (n=3, ddof=1, matching the seed_sd stored
in rexpred_d32_band.json). The honest read: the rexpred advantage is small and
the seed bands nearly touch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "session21"))
import figstyle  # noqa: E402

TRACKC = REPO / "outputs/session34/trackc_lift.json"
BAND35 = REPO / "outputs/session35/rexpred_d32_band.json"
OUT_DIR = REPO / "paper/sections/figures/results"
OUT_PDF = OUT_DIR / "fig_rexpred_v4.pdf"
OUT_PNG = OUT_DIR / "fig_rexpred_v4.png"

SEEDS = ["s0", "s1", "s2"]


def main() -> None:
    with open(TRACKC) as f:
        trackc = json.load(f)
    with open(BAND35) as f:
        band35 = json.load(f)

    cln = trackc["results"]["cln"]
    data = {
        "CLN": {
            "linear": [cln[s]["linear"]["pooled_peak_r2"] for s in SEEDS],
            "mlp": [cln[s]["mlp"]["pooled_peak_r2"] for s in SEEDS],
        },
        "CLN-rexpred": {
            "linear": list(band35["band"]["linear_peak_r2_per_seed"]),
            "mlp": list(band35["band"]["mlp_peak_r2_per_seed"]),
        },
    }
    # cross-check: our ddof=1 sd convention reproduces the stored seed_sd
    lin_rex = np.array(data["CLN-rexpred"]["linear"])
    assert abs(np.std(lin_rex, ddof=1) - band35["band"]["seed_sd"]) < 1e-9
    assert abs(np.mean(lin_rex) - band35["band"]["seed_mean"]) < 1e-9

    green = figstyle.FAMILY_COLOR["jepa"]
    marker = figstyle.FAMILY_MARKER["jepa"]

    figstyle.use_style()
    fig, ax = plt.subplots(figsize=figstyle.figure_size(0.62, aspect=0.78))

    group_x = {"CLN": 0.0, "CLN-rexpred": 1.0}
    probe_dx = {"linear": -0.17, "mlp": +0.17}
    jit = np.array([-0.035, 0.0, 0.035])

    for gname, probes in data.items():
        for pname, vals in probes.items():
            vals = np.asarray(vals, dtype=float)
            x0 = group_x[gname] + probe_dx[pname]
            filled = pname == "linear"
            ax.plot(x0 + jit, vals, ls="none", marker=marker, ms=4.5,
                    mfc=green if filled else "white", mec=green, mew=0.9,
                    alpha=0.9, zorder=3)
            mean, sd = float(np.mean(vals)), float(np.std(vals, ddof=1))
            ax.hlines(mean, x0 - 0.10, x0 + 0.10, color=green, lw=1.6, zorder=4)
            ax.annotate(f"{mean:.3f}\n$\\pm${sd:.3f}",
                        xy=(x0, np.min(vals)), xytext=(0, -7),
                        textcoords="offset points", ha="center", va="top",
                        fontsize=6, color="0.25")

    ax.set_xticks(list(group_x.values()))
    ax.set_xticklabels(list(group_x.keys()))
    ax.set_xlim(-0.55, 1.55)
    all_vals = np.concatenate([np.asarray(v) for p in data.values() for v in p.values()])
    ax.set_ylim(all_vals.min() - 0.022, all_vals.max() + 0.008)
    ax.set_ylabel(r"pooled peak-region $C_L$ $R^2$")

    handles = [
        plt.Line2D([], [], ls="none", marker=marker, ms=4.5, mfc=green,
                   mec=green, label="linear probe"),
        plt.Line2D([], [], ls="none", marker=marker, ms=4.5, mfc="white",
                   mec=green, mew=0.9, label="MLP probe"),
        plt.Line2D([], [], color=green, lw=1.6, label="seed mean"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.01),
              ncol=3, frameon=False, fontsize=6, handlelength=1.2,
              columnspacing=0.9)
    ax.text(0.985, 0.025, "test_b, $n=3$ seeds", ha="right", va="bottom",
            transform=ax.transAxes, fontsize=6.2, color="0.35")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=200)
    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
