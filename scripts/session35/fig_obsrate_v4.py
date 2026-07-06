"""F22 inset (v4 paper): the assimilation-rate sweep, standalone small panel.

Median impact-phase C_L R^2 versus assimilation interval m in {1, 2, 4, 8}
(sensors recorded at full rate dt = 0.05, EnKF update applied only every m-th
frame; --obs-every in scripts/session34/{lae_enkf_pilot,lae_hybrid}.py) for

  - the linear-A LAE filter (jepa_pool_vec latents), which degrades
    gracefully:  outputs/session34/lae_pilot_obs{1,2,4,8}.json
  - the transformer-forecast hybrid (frozen AutoregressivePredictor forecast
    + E_obs latent-encoded taps), which collapses already at m = 2:
    m = 1 anchor  outputs/session34/lae_hybrid___gamma_mode_phase___q_scale_1_0.json
    m = 2, 4, 8   outputs/session34/lae_hybrid_obs{2,4,8}.json
    (all four share obs_mode = eobs [script default], gamma_mode = phase,
    q_scale = 1.0; only obs_every differs)

Every plotted value is aggregates.median_CL_r2_impact read from its JSON at
build time. Split test_b (42 encounters), single filter seed (seed 0).
Hybrid values below the axis floor are drawn clipped with their true values
printed. Designed for LaTeX composition next to the reused fig_t_trade.

Outputs: paper/sections/figures/results/fig_obsrate_v4.{pdf,png}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "session21"))
import figstyle  # noqa: E402

OUT_DIR = REPO / "paper/sections/figures/results"
MS = (1, 2, 4, 8)

LINEAR_FILES = {m: REPO / f"outputs/session34/lae_pilot_obs{m}.json" for m in MS}
HYBRID_FILES = {
    1: REPO / "outputs/session34/lae_hybrid___gamma_mode_phase___q_scale_1_0.json",
    2: REPO / "outputs/session34/lae_hybrid_obs2.json",
    4: REPO / "outputs/session34/lae_hybrid_obs4.json",
    8: REPO / "outputs/session34/lae_hybrid_obs8.json",
}

GREEN = figstyle.FAMILY_COLOR["jepa"]
ORANGE = "#e08214"  # accent for the transformer hybrid (not an encoder family)
CLIP_AT = -1.05     # axis floor for drawing; true values printed at the marker


def median_impact(path: Path) -> float:
    return float(json.loads(path.read_text())["aggregates"]["median_CL_r2_impact"])


def main() -> int:
    figstyle.use_style()
    lin = [median_impact(LINEAR_FILES[m]) for m in MS]
    hyb = [median_impact(HYBRID_FILES[m]) for m in MS]

    fig, ax = plt.subplots(figsize=figstyle.figure_size(0.42, aspect=0.86))

    ax.axhline(0.0, color="0.85", lw=0.6, zorder=0)
    ax.plot(MS, lin, color=GREEN, marker="o", ms=3.8, lw=1.1,
            label=r"linear-$A$ LAE filter", zorder=4)

    hyb_clip = [max(v, CLIP_AT) for v in hyb]
    ax.plot(MS, hyb_clip, color=ORANGE, marker="D", ms=3.4, lw=1.1,
            ls="-", label="transformer hybrid", zorder=3)
    for m, v in zip(MS, hyb):
        if v < CLIP_AT:  # clipped: open down-triangle + true value printed
            ax.plot([m], [CLIP_AT], marker="v", ms=4.2, mfc="white",
                    mec=ORANGE, mew=0.9, ls="none", zorder=5)
            ax.annotate(f"{v:.1f}", (m, CLIP_AT), xytext=(0, 5),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=6.0, color=ORANGE)

    ax.set_xscale("log", base=2)
    ax.set_xticks(MS)
    ax.set_xticklabels([str(m) for m in MS])
    ax.minorticks_off()
    ax.set_xlabel(r"assimilation interval $m$ (frames)")
    ax.set_ylabel(r"median impact $C_L$ $R^2$")
    ax.set_ylim(CLIP_AT - 0.22, 1.05)
    ax.legend(loc="upper right", fontsize=6.2, handlelength=1.6,
              borderaxespad=0.25)
    ax.text(0.55, 0.30, "sensors at full rate,\nupdates every $m$-th frame",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=6.0, color="0.35", linespacing=1.35)
    ax.set_title("assimilation-rate sweep (test B)", fontsize=8)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig_obsrate_v4.pdf")
    fig.savefig(OUT_DIR / "fig_obsrate_v4.png", dpi=200)
    plt.close(fig)
    print("wrote", OUT_DIR / "fig_obsrate_v4.pdf")
    print("wrote", OUT_DIR / "fig_obsrate_v4.png")
    print("  linear-A LAE :", ", ".join(f"m={m}: {v:+.3f}" for m, v in zip(MS, lin)))
    print("  transformer hybrid:", ", ".join(f"m={m}: {v:+.3f}" for m, v in zip(MS, hyb)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
