#!/usr/bin/env python3
"""Clean 3-line lead-time C_L-inference figure for the talk (slide 16).

Numbers are the JEPA d=64, test_b row of HEADLINE_NUMBERS.md section 4
(oracle / direct pressure sensing / predictor-in-loop), so the figure matches
the slide bullets exactly. Output: figs/cl_inference_simple.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "figs" / "cl_inference_simple.png"

tau = [30, 20, 10, 5, 2]                       # frames before impact
oracle = [0.683, 0.683, 0.683, 0.683, 0.683]   # probe on the DNS-encoded latent
direct = [-0.084, -0.122, 0.134, 0.542, 0.690]  # pressure -> C_L directly
ploop = [0.139, 0.078, 0.350, 0.600, 0.637]    # roll the latent then probe

NAVY, ACCENT, STEEL, GRAY = "#1F3864", "#C05210", "#4C78A8", "#8A8F98"
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})

fig, ax = plt.subplots(figsize=(5.4, 3.2))
ax.axhline(0, color="0.85", lw=0.8, zorder=1)
ax.plot(tau, oracle, "--", color=GRAY, lw=1.8, label="oracle (DNS latent)", zorder=2)
ax.plot(tau, direct, "-o", color=STEEL, lw=2.0, ms=5, label="direct pressure sensing", zorder=3)
ax.plot(tau, ploop, "-o", color=ACCENT, lw=2.6, ms=6.5, label="predictor-in-loop (JEPA)", zorder=4)

ax.set_xlim(31, 1)            # reversed: impact approaches to the right
ax.set_xticks(tau)
ax.set_xlabel("lead time before impact (frames)")
ax.set_ylabel(r"$C_L$(impact) forecast $R^2$")
ax.set_ylim(-0.25, 0.80)
ax.annotate("0.35 vs 0.13\nat 10 frames", xy=(10, 0.35), xytext=(15, 0.56),
            fontsize=8.5, color=ACCENT,
            arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.1))
ax.legend(frameon=False, fontsize=9, loc="lower left")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="0.93")
fig.tight_layout()
fig.savefig(OUT, dpi=200)
print("wrote", OUT)
