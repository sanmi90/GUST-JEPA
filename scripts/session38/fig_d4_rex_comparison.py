"""d = 4 family comparison under ONE filter: the REX-EnKF (Session 38).

Same recipe for every family (rex_enkf arm of the D261 per-family
end-to-end phase evals; own taps / own E_obs / own REX operator / own
decode floor per family, single stack per cell): C_L estimation (impact
analysis R2 and RMSE) and the decoded-field state readout (impact SSIM,
full frame and near-body) at matched d = 4. Sources:
outputs/session34/da_phase_dim_{pod_d4,fukami_wake_d4,jepa_pool_vec_d4,
cln_rexpred_d4_s0}.json.

Run (CPU): taskset -c 0-15 python -m scripts.session38.fig_d4_rex_comparison
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "session21"))

import matplotlib.pyplot as plt  # noqa: E402
from figstyle import TEXTWIDTH_IN, use_style  # noqa: E402

MODELS = [
    ("pod_d4", "POD", "#4d4d4d"),
    ("fukami_wake_d4", "AE + wake head", "#b2182b"),
    ("jepa_pool_vec_d4", "JEPA (wake)", "#1b7837"),
    ("cln_rexpred_d4_s0", "JEPA (lift-focused)", "#5aae61"),
]
# C_L panels: the session38 band-1.77 re-run (protocol-clean production band;
# the frozen dims-grid arm was band 4.0, test-peeked). SSIM panel: the frozen
# band-4.0 arm (the only arm with decoded-field scores), labelled as such.
AGG = REPO_ROOT / "outputs/session38/d4_band177_aggregates.json"


def main() -> int:
    use_style()
    agg = json.loads(AGG.read_text())
    rows = []
    for m, label, c in MODELS:
        d = json.loads((REPO_ROOT / f"outputs/session34/da_phase_dim_{m}.json").read_text())
        s = d["summary"]["rex_enkf"]
        rows.append((label, c, agg[m]["median_impact_r2"],
                     agg[m]["median_impact_rmse"],
                     s["impact"]["ssim_full"], s["impact"]["ssim_nearbody"]))
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 0.28 * TEXTWIDTH_IN),
                             constrained_layout=True)
    panels = [
        ("(a) impact $C_L$ analysis $R^2$ (median, band 1.77)", 2, (-0.6, 1.0)),
        ("(b) impact $C_L$ RMSE (median, band 1.77)", 3, None),
        ("(c) decoded-field SSIM (band-4.0 arm)", None, (0, 1.0)),
    ]
    for ax, (title, idx, ylim) in zip(axes, panels):
        if idx is not None:
            vals = [r[idx] for r in rows]
            clipped = [max(v, ylim[0]) if ylim else v for v in vals]
            ax.bar(x, clipped, color=[r[1] for r in rows], width=0.62)
            for xi, v in zip(x, vals):
                if ylim and v < ylim[0]:
                    ax.annotate(f"{v:.1f}", xy=(xi, ylim[0] + 0.05), ha="center",
                                fontsize=5.5, color="w")
            if ylim:
                ax.set_ylim(*ylim)
            ax.axhline(0, color="0.4", lw=0.6)
        else:
            w = 0.32
            ax.bar(x - w / 2, [r[4] for r in rows], w, color=[r[1] for r in rows],
                   label="full frame")
            ax.bar(x + w / 2, [r[5] for r in rows], w, color=[r[1] for r in rows],
                   alpha=0.45, label="near-body")
            ax.set_ylim(*ylim)
            ax.legend(fontsize=5.5, frameon=False, loc="upper right")
        ax.set_title(title, fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([r[0] for r in rows], rotation=20, ha="right", fontsize=5.8)
    fig.suptitle("$d = 4$, one filter for all families (REX-EnKF, per-family "
                 "end-to-end stacks, in-distribution test)", fontsize=7.5, y=1.06)
    out = REPO_ROOT / "outputs/session38"
    for ext in ("pdf", "png"):
        path = out / f"fig_d4_rex_comparison.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
