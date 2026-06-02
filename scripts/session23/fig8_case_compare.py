"""Compare candidate gust cases for the Figure 8 staged-flow panel.

Figure 8 uses one representative encounter (the strongest gust). This renders the
impact-frame vorticity (simulation / predictive decode / reconstructive decode)
for several test_b candidates so we can pick the case that most clearly shows the
LEV in the simulation, the predictive decode tracking it, and the reconstructive
decode collapsing. Diagnostic PNG; no .tex touched.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import sys  # noqa: E402

REPO = Path("/home/carlos/GUST-JEPA")
sys.path.insert(0, str(REPO / "scripts" / "session21"))
import figstyle as fs  # noqa: E402

dec = np.load(REPO / "outputs/session20/decoded/test_b.npz", allow_pickle=True)
G = dec["G"]; D = dec["D"]; Y = dec["Y"]
cids = np.array([str(c) for c in dec["case_ids"]])
offs = list(dec["offsets"])
impact_i = offs.index(0)                       # impact frame index

# candidate selection: a spread of strong/moderate, +/- G, and core sizes
order = np.argsort(-np.abs(G))                 # strongest first
picks = list(order[:6])                        # 6 strongest
# add the strongest POSITIVE gust and a moderate one if not already in
pos = [i for i in order if G[i] > 0][:2]
mod = [i for i in np.argsort(np.abs(np.abs(G) - 1.5))[:2]]
for i in pos + mod:
    if i not in picks:
        picks.append(i)
picks = picks[:9]

n = len(picks)
fig, axes = plt.subplots(n, 3, figsize=(6.5, 1.5 * n))
for r, gi in enumerate(picks):
    for c, (key, title) in enumerate([("target_norm", "simulation"),
                                       ("jepa_norm", "predictive"),
                                       ("fukami_norm", "reconstructive")]):
        ax = axes[r, c]
        fs.vort_panel(ax, dec[key][gi, impact_i])
        if r == 0:
            ax.set_title(title, fontsize=9)
        if c == 0:
            ax.set_ylabel(f"G={G[gi]:+.1f}\nD={D[gi]:.1f} Y={Y[gi]:+.2f}",
                          fontsize=7, rotation=0, ha="right", va="center", labelpad=22)
fig.suptitle("Figure 8 candidate cases (impact frame): which shows the clearest "
             "sim-LEV / predictive-track / reconstructive-collapse?", fontsize=9)
fig.tight_layout()
out = REPO / "outputs/session23/fig8_case_compare.png"
fig.savefig(out, dpi=130)
print("wrote", out, "| candidates:", [(f"{G[i]:+.1f}", f"{D[i]:.1f}", f"{Y[i]:+.2f}") for i in picks])
