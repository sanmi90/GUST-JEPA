#!/usr/bin/env python3
"""Manuscript figure: the collective wake-forecast code, three families. Per
family, best single coordinate vs full-latent (combination) held-out forecast
skill toward the future wake enstrophy. JEPA has a large gap (collective code);
Fukami and POD have ~zero gap (no collective structure) and lower skill.
Writes the manuscript figure paper/sections/figures/results/fig_wake_code.pdf."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "session21"))
J = json.loads(Path("outputs_causal/jepa_modes/cross_encoder3.json").read_text())
FAM = ["JEPA", "Fukami", "POD"]
LAB = {"JEPA": "JEPA\n(predictive)", "Fukami": "Fukami\n(reconstructive)", "POD": "POD\n(linear)"}

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import figstyle
    figstyle.use_style()
    COL = {"JEPA": figstyle.FAMILY_COLOR["jepa"], "Fukami": figstyle.FAMILY_COLOR["fukami"],
           "POD": figstyle.FAMILY_COLOR["pod"]}
    W = figstyle.TEXTWIDTH_IN
except Exception:
    COL = {"JEPA": "#1b7837", "Fukami": "#c0392b", "POD": "#2166ac"}; W = 4.98

fig, ax = plt.subplots(figsize=(0.66 * W, 0.44 * W))
x = np.arange(3); w = 0.36
for i, fam in enumerate(FAM):
    bs, cb = J[fam]["best_single"], J[fam]["combo"]
    ax.bar(i - w/2, bs, w, color=COL[fam], alpha=0.40)
    ax.bar(i + w/2, cb, w, color=COL[fam], alpha=1.0)
ax.set_xticks(x); ax.set_xticklabels([LAB[f] for f in FAM])
ax.set_ylim(0, 1.0)
ax.set_ylabel(r"held-out future-wake forecast skill $|\rho_s|$")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor="0.5", alpha=0.40, label="best single coordinate"),
                   Patch(facecolor="0.5", alpha=1.0, label="full latent (combination)")],
          loc="upper right", frameon=False, fontsize=6.5)
fig.tight_layout(pad=0.3)
out = Path("paper/sections/figures/results/fig_wake_code.pdf")
fig.savefig(out, bbox_inches="tight")
print(f"[fig] JEPA gap {J['JEPA']['combo']-J['JEPA']['best_single']:.2f}, "
      f"Fukami {J['Fukami']['combo']-J['Fukami']['best_single']:.2f}, "
      f"POD {J['POD']['combo']-J['POD']['best_single']:.2f}")
print(f"[fig] wrote {out}")
