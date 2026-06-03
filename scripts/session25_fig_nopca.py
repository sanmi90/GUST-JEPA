#!/usr/bin/env python3
"""Render the basis-free (no-PCA) two-panel figure from obs_nopca.json.
(a) wake forecast-direction asymmetry; (b) per-observable held-out forecast skill
from the JEPA latent, with each direction's variance share annotated. Review
artifact: writes to outputs_causal/jepa_modes/fig_nopca.pdf, NOT the manuscript."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "session21"))
J = json.loads(Path("outputs_causal/jepa_modes/obs_nopca.json").read_text())

OBS = ["C_L", "C_D", "I_y", "wake_enstrophy", "circulation_pos", "circulation_neg"]
LAB = {"C_L": r"$C_L$", "C_D": r"$C_D$", "I_y": r"$I_y$", "wake_enstrophy": r"$\Omega_w$",
       "circulation_pos": r"$\Gamma^{+}$", "circulation_neg": r"$\Gamma^{-}$"}

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import figstyle
    figstyle.use_style()
    green, grey, W = figstyle.FAMILY_COLOR["jepa"], figstyle.FAMILY_COLOR["oracle"], figstyle.TEXTWIDTH_IN
except Exception:
    green, grey, W = "#1b7837", "#404040", 4.98

fig, (axA, axB) = plt.subplots(1, 2, figsize=(W, 0.42 * W),
                               gridspec_kw={"width_ratios": [1.0, 1.25]})

# (a) wake forecast-direction asymmetry
a_tr, a_te = J["_asym"]["train"], J["_asym"]["test_b"]
cats = ["raw", "| forces", "| forces\n+ wake", "$C_L$ | $s$"]
keys = ["s_raw", "s_par_F", "s_par_FW", "cl_par_s"]
tr = [a_tr[k] for k in keys]; te = [a_te[k] for k in keys]
cols = [green, green, green, grey]; x = np.arange(4); w = 0.38
axA.bar(x - w/2, tr, w, color=cols, alpha=0.45)
axA.bar(x + w/2, te, w, color=cols, alpha=1.0)
axA.set_xticks(x); axA.set_xticklabels(cats); axA.set_ylim(0, 0.95)
axA.set_ylabel(r"$|\rho_s|$ to future wake $\Omega_w(t{+}H)$")
axA.axhline(0.25, ls=":", lw=0.8, color="0.5")
from matplotlib.patches import Patch
axA.legend(handles=[Patch(facecolor="0.5", alpha=0.45, label="train"),
                    Patch(facecolor="0.5", alpha=1.0, label="test B")],
           loc="upper right", frameon=False)
axA.set_title(r"(a) latent wake-forecast signal $s$", loc="left", fontsize=8)

# (b) per-observable held-out forecast skill, variance share annotated
skill = [J[o]["skill_te"] for o in OBS]
var = [J[o]["var_share"] * 100 for o in OBS]
kind = [J[o]["kind"] for o in OBS]
cb = [green if k == "wake" else grey for k in kind]
xb = np.arange(len(OBS))
axB.bar(xb, skill, 0.7, color=cb)
for i, (s, v) in enumerate(zip(skill, var)):
    axB.text(i, s + 0.01, f"{v:.2f}%", ha="center", va="bottom", fontsize=5.5, color="0.3")
axB.set_xticks(xb); axB.set_xticklabels([LAB[o] for o in OBS])
axB.set_ylim(0, 1.05); axB.set_ylabel(r"held-out forecast skill $|\rho_s|$")
axB.set_title("(b) forecast skill per observable", loc="left", fontsize=8)
axB.legend(handles=[Patch(facecolor=grey, label="force"), Patch(facecolor=green, label="wake")],
           loc="upper left", frameon=False, ncol=2, fontsize=6)

fig.tight_layout(pad=0.4)
out = Path("outputs_causal/jepa_modes/fig_nopca.pdf")
fig.savefig(out, bbox_inches="tight")
print(f"[fig] wrote {out}")
