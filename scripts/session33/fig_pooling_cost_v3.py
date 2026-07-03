"""F10 (v3, appendix): the decodability-vs-estimability pooling trade.

Dumbbell (slope) chart of the P4 pooling-cost gate: for every family with a
spatial (per-token) counterpart, the spatial -> pooled shift in (a) decode
SSIM, (b) matched-predictor merit at h = 8, and (c) field VRMSE at h = 8.
Open markers are the spatial latent, filled markers the pooled latent.

Every plotted value is read from outputs/session32/track_p_gates.json
(P4_pooling_cost); entries carrying a "note" (no spatial counterpart) are
skipped. Nothing is hand-typed.

Usage:
    taskset -c 16-23 python scripts/session33/fig_pooling_cost_v3.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "session21"))

from figstyle import FAMILY_COLOR, TEXTWIDTH_IN, use_style  # noqa: E402

GATES = REPO / "outputs/session32/track_p_gates.json"
OUT_PDF = REPO / "outputs/session33/figures/fig_pooling_cost_v3.pdf"

# json key -> (display label, colour); order = top-to-bottom row order
FAMILIES = {
    "jepa_wake_pool": ("JEPA + wake", FAMILY_COLOR["jepa"]),
    "supervised_only_pool": ("supervised only", "#e08214"),
    "ae_wake_pool": ("AE + wake", FAMILY_COLOR["fukami"]),
    "jepa_nowake_pool": ("JEPA", FAMILY_COLOR["jepa"]),
    "ae_nowake_pool": ("AE", FAMILY_COLOR["fukami"]),
    "regAE_pool": ("reg. AE", FAMILY_COLOR["fukami"]),
}

# (json key, xlabel, direction of improvement)
METRICS = [
    ("decode_ssim", "decode SSIM", "higher"),
    ("merit_h8", r"merit ($h = 8$)", "higher"),
    ("field_vrmse_h8", r"field VRMSE ($h = 8$)", "lower"),
]


def main() -> None:
    use_style()
    p4 = json.loads(GATES.read_text())["P4_pooling_cost"]

    rows = [(key, lab, col) for key, (lab, col) in FAMILIES.items()
            if key in p4 and "note" not in p4[key]]
    skipped = [k for k in p4 if "note" in p4[k]]
    ys = list(range(len(rows) - 1, -1, -1))

    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 2.15), sharey=True,
                             gridspec_kw={"wspace": 0.14})
    fig.subplots_adjust(left=0.155, right=0.99, top=0.82, bottom=0.34)

    for ax, (metric, xlabel, better), letter in zip(axes, METRICS, "abc"):
        vals = []
        for (key, lab, col), y in zip(rows, ys):
            sp = float(p4[key][metric]["spatial"])
            po = float(p4[key][metric]["pooled"])
            vals += [sp, po]
            ax.plot([sp, po], [y, y], color=col, lw=1.1, zorder=2,
                    solid_capstyle="round")
            ax.plot([sp], [y], marker="o", ms=4.0, mfc="white", mec=col,
                    mew=1.0, ls="none", zorder=3)
            ax.plot([po], [y], marker="o", ms=4.0, mfc=col, mec=col,
                    ls="none", zorder=3)
        lo, hi = min(vals), max(vals)
        pad = 0.10 * (hi - lo)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_xlabel(xlabel)
        ax.tick_params(axis="y", length=0)
        ax.text(0.0, 1.06, f"({letter})", transform=ax.transAxes, fontsize=8,
                style="italic", va="bottom")
        # improvement-direction cue
        x_tail, x_tip = (0.78, 0.99) if better == "higher" else (0.99, 0.78)
        ax.annotate("", xy=(x_tip, 1.10), xytext=(x_tail, 1.10),
                    xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", lw=0.7, color="0.4"))
        ax.text(min(x_tail, x_tip) - 0.03, 1.10, "better", ha="right",
                va="center", fontsize=6, color="0.4",
                transform=ax.transAxes)

    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([lab for _, lab, _ in rows])

    handles = [
        plt.Line2D([], [], marker="o", ms=4.0, mfc="white", mec="0.15",
                   mew=1.0, ls="none", label="spatial (per-token) latent"),
        plt.Line2D([], [], marker="o", ms=4.0, mfc="0.15", mec="0.15",
                   ls="none", label="pooled latent"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=6.5,
               bbox_to_anchor=(0.55, 0.0), handletextpad=0.4,
               columnspacing=1.6)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    plt.close(fig)

    print(f"wrote {OUT_PDF}")
    if skipped:
        print(f"  skipped (no spatial counterpart): {', '.join(skipped)}")
    for key, lab, _ in rows:
        parts = []
        for metric, _, _ in METRICS:
            m = p4[key][metric]
            parts.append(f"{metric}: {m['spatial']:.3f}->{m['pooled']:.3f} "
                         f"(d={m['delta']:+.3f})")
        print(f"  {lab:<16} " + " | ".join(parts))


if __name__ == "__main__":
    main()
