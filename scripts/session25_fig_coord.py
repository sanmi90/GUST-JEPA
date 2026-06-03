#!/usr/bin/env python3
"""Figure: per-observable best-single-coordinate forecast skill vs full-latent
(combination) skill. The gap = how distributed the forecast code is. Forces are
redundant (small gap, many skillful coordinates); the wake enstrophy is collective
(large gap, no single skillful coordinate). Review artifact."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "session21"))
npz = np.load("outputs_causal/jepa_modes/coord_by_coord.npz", allow_pickle=True)
M = npz["skill_matrix"]; obs_order = list(npz["observables"])
J = json.loads(Path("outputs_causal/jepa_modes/obs_nopca.json").read_text())

OBS = ["C_L", "C_D", "wake_enstrophy", "circulation_pos", "circulation_neg"]
LAB = {"C_L": r"$C_L$", "C_D": r"$C_D$", "wake_enstrophy": r"$\Omega_w$",
       "circulation_pos": r"$\Gamma^{+}$", "circulation_neg": r"$\Gamma^{-}$"}
KIND = {"C_L": "force", "C_D": "force", "wake_enstrophy": "wake",
        "circulation_pos": "wake", "circulation_neg": "wake"}

best_single = {o: float(M[:, obs_order.index(o)].max()) for o in OBS}
combo = {o: J[o]["skill_te"] for o in OBS}
n_strong = {o: int((M[:, obs_order.index(o)] > 0.5).sum()) for o in OBS}

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import figstyle
    figstyle.use_style()
    green, grey, W = figstyle.FAMILY_COLOR["jepa"], figstyle.FAMILY_COLOR["oracle"], figstyle.TEXTWIDTH_IN
except Exception:
    green, grey, W = "#1b7837", "#404040", 4.98

fig, ax = plt.subplots(figsize=(0.66 * W, 0.42 * W))
x = np.arange(len(OBS)); w = 0.38
for i, o in enumerate(OBS):
    c = green if KIND[o] == "wake" else grey
    ax.bar(i - w/2, best_single[o], w, color=c, alpha=0.4)
    ax.bar(i + w/2, combo[o], w, color=c, alpha=1.0)
    # gap arrow
    ax.annotate("", xy=(i + w/2, combo[o]), xytext=(i + w/2, best_single[o]),
                arrowprops=dict(arrowstyle="-|>", color="0.25", lw=0.7))
    ax.text(i, max(combo[o], best_single[o]) + 0.02, f"{n_strong[o]} coord$>$0.5",
            ha="center", va="bottom", fontsize=5.5, color="0.35")
ax.set_xticks(x); ax.set_xticklabels([LAB[o] for o in OBS])
ax.set_ylim(0, 1.08); ax.set_ylabel(r"held-out forecast skill $|\rho_s|$")
ax.set_title("forecast of each observable: best single coordinate vs the combination", loc="left", fontsize=7.5)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor="0.5", alpha=0.4, label="best single coordinate"),
                   Patch(facecolor="0.5", alpha=1.0, label="full latent (combination)")],
          loc="lower center", frameon=False, fontsize=6, ncol=2)
fig.tight_layout(pad=0.4)
out = Path("outputs_causal/jepa_modes/fig_coord.pdf")
fig.savefig(out, bbox_inches="tight")
print(f"[fig] best_single={ {o: round(best_single[o],2) for o in OBS} }")
print(f"[fig] combination={ {o: round(combo[o],2) for o in OBS} }")
print(f"[fig] wrote {out}")
