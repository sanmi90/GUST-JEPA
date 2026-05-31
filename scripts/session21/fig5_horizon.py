"""SPEC 6 - Fig 5 (de-cluttered): horizon dependence of held-out forecast closure.

One idea: the predictive latent degrades gracefully with horizon while the
reconstructive and linear latents fail abruptly on the wake. R^2 of the Markov
rollout vs horizon H, for three observables, two splits. No in-image title; the
R^2 = 0.5 reference line is kept; all explanation goes to the caption.
Source: outputs/session20/horizon_sweep/horizon_summary.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import figstyle as fs

REPO = Path(__file__).resolve().parents[2]
J = REPO / "outputs/session20/horizon_sweep/horizon_summary.json"
OUT_PDF = REPO / "paper/sections/figures/results/fig_horizon_sweep.pdf"
OUT_PNG = REPO / "outputs/session21/figs/fig_horizon_sweep.png"

COLS = [("wake_enstrophy", r"wake enstrophy $\Omega_w$"),
        ("C_L", r"$C_L$"), ("circulation_neg", r"$\Gamma^{-}$")]
SPLITS = [("test_b", "test B"),
          ("test_c", r"test C, $|G|{=}4$")]
FAMILIES = [("jepa_d64_test1_noBN", "predictive (JEPA) $d{=}64$"),
            ("jepa_d32_noBN", "predictive (JEPA) $d{=}32$"),
            ("fukami_d64_noBN", "reconstructive $d{=}64$"),
            ("pod_d64_noBN", "POD $d{=}64$")]


def main() -> None:
    fs.use_style()
    data = json.load(open(J))
    fig, axes = plt.subplots(2, 3, figsize=fs.figure_size(1.0, aspect=0.72),
                             sharex=True, layout="constrained")
    for i, (split, slab) in enumerate(SPLITS):
        for j, (metric, mlab) in enumerate(COLS):
            ax = axes[i, j]
            ax.axhline(0.5, ls=":", lw=0.8, color="0.5", zorder=1)
            ax.axhline(0.0, ls="-", lw=0.6, color="0.85", zorder=0)
            blob = data.get(split, {}).get(metric, {})
            for tag, _ in FAMILIES:
                if tag not in blob:
                    continue
                H = np.asarray(blob[tag]["horizons"], float)
                r2 = np.asarray(blob[tag]["r2"], float)
                ax.plot(H, r2, marker=fs.FAMILY_MARKER[fs.BASELINE[tag][0]],
                        ms=3.5, lw=1.0, color=fs.family_color(tag), zorder=3)
            ax.set_xscale("log", base=2)
            ax.set_xticks([1, 4, 16, 64])
            ax.set_xticklabels(["1", "4", "16", "64"])
            ax.set_ylim(-1.4, 1.05)
            if i == 0:
                ax.set_title(mlab)
            if i == 1:
                ax.set_xlabel("horizon $H$ (frames)")
            if j == 0:
                ax.set_ylabel(f"{slab}\nforecast $R^2$")

    handles = [plt.Line2D([], [], color=fs.family_color(t),
                          marker=fs.FAMILY_MARKER[fs.BASELINE[t][0]], ms=3.5,
                          lw=1.0, label=l) for t, l in FAMILIES]
    fig.legend(handles=handles, loc="outside lower center", ncol=4,
               columnspacing=1.2, handletextpad=0.3)
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=200)
    print(f"wrote {OUT_PDF}\nwrote {OUT_PNG}")


if __name__ == "__main__":
    main()
