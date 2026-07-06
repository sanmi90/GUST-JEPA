"""F17 trim (v4 paper): the multi-step-stability panel, standalone.

Single-panel rebuild of panel (b) of the v3 mechanism figure
(scripts/session33/fig_mechanism_hroll_v3.py): mean observable R^2 versus
forecast horizon for H_roll = 1 vs H_roll = 8. Multi-step training is what
keeps the long rollout usable (H_roll = 1 collapses toward zero merit at
h = 16 while H_roll = 8 holds), and the mechanism (slower open-loop latent
drift) is annotated from the same JSON.

Data loading is IMPORTED VERBATIM from the v3 script (load_hroll and the
shared HROLL_STYLE), so this figure reads the identical
outputs/session3{3,2}/hroll_ablation.json resolution chain; nothing is
hand-typed. Built at half text width for side-by-side composition.

Outputs: paper/sections/figures/results/fig_hroll_trim_v4.{pdf,png}
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "session21"))
import figstyle  # noqa: E402

# Provenance: data loading reused verbatim from the v3 mechanism figure
# (scripts/session33/fig_mechanism_hroll_v3.py, load_hroll + HROLL_STYLE +
# the HROLL_JSON session33-else-session32 fallback).
from scripts.session33.fig_mechanism_hroll_v3 import (  # noqa: E402
    HROLL_JSON,
    HROLL_STYLE,
    load_hroll,
)

OUT_DIR = REPO / "paper/sections/figures/results"


def main() -> int:
    figstyle.use_style()
    curves = load_hroll()

    fig, ax = plt.subplots(figsize=figstyle.figure_size(0.5, aspect=0.80))
    for tag in ("H_roll_1", "H_roll_8"):
        st = HROLL_STYLE[tag]
        ax.plot(curves[tag]["h"], curves[tag]["merit"], color=st["color"],
                ls=st["ls"], marker=st["marker"], ms=3.5, lw=1.1,
                label=st["label"])
    ax.axhline(0.0, color="0.85", lw=0.6, zorder=0)
    ax.set_xscale("log", base=2)
    hs = curves["H_roll_8"]["h"]
    ax.set_xticks(hs)
    ax.set_xticklabels([str(h) for h in hs])
    ax.minorticks_off()
    ax.set_xlabel(r"forecast horizon $h$ (frames)")
    ax.set_ylabel(r"mean observable $R^2$")
    ax.legend(loc="lower left", fontsize=6.2, handlelength=1.8)
    ax.set_ylim(-0.03, 0.80)

    # the mechanism, from the same JSON: multi-step training slows the
    # open-loop latent drift that destroys the H_roll = 1 rollout.
    d1 = curves["H_roll_1"]["drift"][-1]
    d8 = curves["H_roll_8"]["drift"][-1]
    h_last = curves["H_roll_8"]["h"][-1]
    ax.text(0.02, 0.975,
            f"open-loop latent drift at $h = {h_last}$:\n"
            rf"{d1:.2f} ($H_{{\rm roll}} = 1$) vs {d8:.2f} ($H_{{\rm roll}} = 8$)",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=6.0, color="0.35", linespacing=1.4)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig_hroll_trim_v4.pdf")
    fig.savefig(OUT_DIR / "fig_hroll_trim_v4.png", dpi=200)
    plt.close(fig)
    print("wrote", OUT_DIR / "fig_hroll_trim_v4.pdf")
    print("wrote", OUT_DIR / "fig_hroll_trim_v4.png")
    print(f"  data: {HROLL_JSON}")
    for tag in ("H_roll_1", "H_roll_8"):
        c = curves[tag]
        print(f"  {tag}: merit h1={c['merit'][0]:.3f} h{c['h'][-1]}="
              f"{c['merit'][-1]:.3f} drift h{c['h'][-1]}={c['drift'][-1]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
