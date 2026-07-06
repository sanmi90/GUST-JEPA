"""Standalone phi_L panel for the Chang-head Methods subsection.

Reuses draw_phi_panel and draw_lift_dir_inset from fig_architecture_v4
(same data: outputs/data_pipeline/v2p2/phi_L.npz, adjacent mask, band).
The architecture schematic itself is the TikZ figure
(sections/figures/tikz/fig1_jepa_architecture.pdf); this file emits only
the physics panel: fig_phi_panel_v4.{pdf,png}.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "session21"))
import figstyle  # noqa: E402

from scripts.session35.fig_architecture_v4 import (  # noqa: E402
    draw_lift_dir_inset,
    draw_phi_panel,
)

OUT = REPO / "paper/sections/figures/results"


def main() -> None:
    figstyle.use_style()
    fig = plt.figure(figsize=(figstyle.TEXTWIDTH_IN, 1.85))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.45, 1.0], wspace=0.12)
    draw_phi_panel(fig.add_subplot(gs[0]))
    draw_lift_dir_inset(fig.add_subplot(gs[1]))
    fig.tight_layout()
    fig.savefig(OUT / "fig_phi_panel_v4.pdf")
    fig.savefig(OUT / "fig_phi_panel_v4.png", dpi=200)
    print("wrote", OUT / "fig_phi_panel_v4.pdf")


if __name__ == "__main__":
    main()
