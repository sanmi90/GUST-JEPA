"""F4 (Session 35 P2): evaluation-protocol map with split hygiene.

A referee-facing schematic: which split feeds which stage. Fits on train
only; tuning on test_a only; test_b one-shot; test_c reporting-only; probes
never enter the filter innovation (leakage guard, src/estimation/enkf.py:269).
Split sizes from configs/splits/split_v2p2.json at build time.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.session21 import figstyle  # noqa: E402

GREEN = figstyle.FAMILY_COLOR["jepa"]
RED = figstyle.FAMILY_COLOR["fukami"]
GREY = figstyle.FAMILY_COLOR["oracle"]


def box(ax, x, y, w, h, text, fc="#f2f2f2", ec="#404040", fs=6.4, lw=0.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                facecolor=fc, edgecolor=ec, linewidth=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, xy0, xy1, color="#404040", ls="-", lw=0.9):
    ax.add_patch(FancyArrowPatch(xy0, xy1, arrowstyle="-|>", mutation_scale=7,
                                 color=color, linestyle=ls, lw=lw,
                                 shrinkA=1, shrinkB=1))


def main() -> None:
    split = json.loads((REPO / "configs/splits/split_v2p2.json").read_text())["cases"]
    n_train_cases = sum(1 for c in split.values() if c["split"] == "train")
    n_tb = sum(1 for c in split.values() if c["split"] == "test_b")
    n_tc = sum(1 for c in split.values() if c["split"] == "test_c")
    n_val_enc = sum(len(c.get("val_encounter_indices", []))
                    for c in split.values() if c["split"] == "train")

    figstyle.use_style()
    fig, ax = plt.subplots(figsize=(figstyle.TEXTWIDTH_IN, 2.9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    # column 1: splits
    box(ax, 0.1, 4.7, 2.0, 0.9,
        f"train\n{n_train_cases} cases", fc="#e8f0e8", ec=GREEN)
    box(ax, 0.1, 3.4, 2.0, 0.9,
        f"test A (val)\n{n_val_enc} enc, held-out\nenc of train cases",
        fc="#eef2ee", ec=GREEN, fs=5.8)
    box(ax, 0.1, 2.1, 2.0, 0.9, f"test B\n{n_tb} cases, one-shot", ec=GREY)
    box(ax, 0.1, 0.8, 2.0, 0.9,
        f"test C, |G| = 4\n{n_tc} cases, report-only", fc="#f7ecec", ec=RED)

    # column 2: frozen states
    box(ax, 2.85, 3.9, 2.5, 1.7,
        "frozen encoders\nJEPA cells, AE anchors,\nFukami AE, POD\n"
        r"(fits: train only)", fc="#e8f0e8", ec=GREEN, fs=6.0)

    # column 3: frozen readouts
    box(ax, 5.9, 4.9, 3.9, 0.75, "probes (linear / MLP): readability", fs=5.6)
    box(ax, 5.9, 4.0, 3.9, 0.75, "decode-floor decoder: fields", fs=5.6)
    box(ax, 5.9, 3.1, 3.9, 0.75, "latent-REX shared forecast operator", fs=5.6)
    box(ax, 5.9, 1.5, 3.9, 1.3,
        "own-stack DA:\nOSP taps K = 8, E_obs delay 10,\n"
        "ladder: static / linear LAE /\nREX-EnKF / two-stage", fs=5.6)

    arrow(ax, (2.1, 5.15), (3.0, 4.9), color=GREEN)
    for y in (5.27, 4.37, 3.47, 2.15):
        arrow(ax, (5.2, 4.7), (6.0, y))

    # tuning + reporting flows
    arrow(ax, (2.1, 3.85), (6.0, 2.4), color=GREEN, ls=(0, (3, 2)))
    ax.text(2.35, 3.02, "tuning: test A only", fontsize=5.6, color=GREEN,
            ha="left", rotation=-14)
    arrow(ax, (2.1, 2.55), (6.0, 1.9), ls=(0, (1.5, 1.5)))
    ax.text(3.55, 2.42, "one frozen run", fontsize=5.6, ha="left",
            rotation=-8)
    arrow(ax, (2.1, 1.25), (6.0, 1.6), color=RED, ls=(0, (1.5, 1.5)))
    ax.text(2.5, 1.02, "reporting only, never selection", fontsize=5.6,
            color=RED, ha="left", rotation=4)

    ax.text(6.0, 1.05, "leakage guard: probes never enter\nthe filter "
                       "innovation (taps only)",
            fontsize=5.6, style="italic", color="#404040", va="top")
    ax.set_title("evaluation protocol: fits on train, tuning on test A, "
                 "test B one-shot, test C report-only", fontsize=7.5)

    out = REPO / "paper/sections/figures/results"
    fig.savefig(out / "fig_protocol_v4.pdf")
    fig.savefig(out / "fig_protocol_v4.png", dpi=200)
    print("wrote", out / "fig_protocol_v4.pdf")


if __name__ == "__main__":
    main()
