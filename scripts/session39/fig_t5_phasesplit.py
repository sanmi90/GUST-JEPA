#!/usr/bin/env python3
"""T5 figure: shared-operator forecast merit split by phase (pre / through /
post impact), per family at horizons 8 and 16. Reads
outputs/session39/t5_phase_split.json. CPU."""
from pathlib import Path
import json
import sys

import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.session21.figstyle import TEXTWIDTH_IN, use_style  # noqa: E402

OUT = REPO / "paper/sections/figures/results/fig_t5_phasesplit_v4.pdf"
FAM_ORDER = ["JepaWake", "AeWake", "SupOnly", "Fukami", "Pod"]
FAM_LABEL = {"JepaWake": "pred.\n(wake)", "AeWake": "AE\n(wake)",
             "SupOnly": "sup.\nonly", "Fukami": "publ.\nrecipe", "Pod": "POD"}
PHASES = [("pre", "pre-impact", "#2a9d8f"),
          ("through", "through impact", "#e76f51"),
          ("post", "post-impact", "#6a4c93")]


def main():
    use_style()
    d = json.load(open(REPO / "outputs/session39/t5_phase_split.json"))["families"]
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 0.42 * TEXTWIDTH_IN),
                             sharey=True)
    x = np.arange(len(FAM_ORDER))
    w = 0.26
    for ax, h in zip(axes, (8, 16)):
        for j, (pk, plabel, pc) in enumerate(PHASES):
            vals, los, his = [], [], []
            for fam in FAM_ORDER:
                rec = d[fam].get(f"h{h}_{pk}")
                if rec is None:
                    vals.append(np.nan); los.append(0); his.append(0)
                else:
                    vals.append(rec["merit_mean"])
                    los.append(rec["merit_mean"] - rec["ci_lo"])
                    his.append(rec["ci_hi"] - rec["merit_mean"])
            ax.bar(x + (j - 1) * w, vals, w, color=pc, label=plabel,
                   yerr=[los, his], error_kw=dict(lw=0.6, capsize=1.5))
        ax.axhline(0, color="0.4", lw=0.6)
        ax.set_title(f"horizon {h}", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([FAM_LABEL[f] for f in FAM_ORDER], fontsize=6)
    axes[0].set_ylabel("forecast merit ($R^2$)")
    axes[1].legend(fontsize=6, loc="lower left", frameon=False)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
