"""Part D figure (F21): deployment realism, streaming protocol + noise bands.

Panels
  (a) streaming impact median C_L R^2 vs tap noise (0/5/10/20%), protocol-clean
      band_scale = 1.77 arm ONLY, 3-seed mean with min/max band. Streaming
      multi-encounter protocol, pressure-only initialisation.
  (b) the d = 4 filter band as a dot plot: 5 member-noise seeds, mean line.
      K = 8 taps, d = 4 over-determined observation.

Every number is loaded from its JSON at build time
(aggregates.median_CL_r2_impact in each file):
  outputs/session35/rex_stream_b177_noise{0.0,0.05,0.1,0.2}_s{0,1,2}.json
  outputs/session34/rex_filter_d4.json               (seed 0)
  outputs/session35/rex_filter_d4_s{1,2,3,4}.json    (seeds 1-4)

Split test_b (42 encounters).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "session21"))
import figstyle  # noqa: E402

NOISES = ("0.0", "0.05", "0.1", "0.2")
STREAM_SEEDS = (0, 1, 2)
JEPA = figstyle.FAMILY_COLOR["jepa"]


def load_impact_r2(path: Path) -> float:
    with open(path) as f:
        return json.load(f)["aggregates"]["median_CL_r2_impact"]


def main() -> None:
    figstyle.use_style()

    # --- data: streaming band (protocol-clean 1.77 arm only) ----------------
    stream = {}  # noise fraction -> list of 3 seed values
    for n in NOISES:
        stream[n] = [
            load_impact_r2(REPO / f"outputs/session35/rex_stream_b177_noise{n}_s{s}.json")
            for s in STREAM_SEEDS
        ]

    # --- data: d = 4 filter band (5 member-noise seeds) ---------------------
    d4_files = [REPO / "outputs/session34/rex_filter_d4.json"] + [
        REPO / f"outputs/session35/rex_filter_d4_s{s}.json" for s in (1, 2, 3, 4)
    ]
    d4_vals = [load_impact_r2(f) for f in d4_files]

    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(figstyle.TEXTWIDTH_IN, figstyle.TEXTWIDTH_IN * 0.42),
        gridspec_kw={"width_ratios": [1.5, 1.0]},
    )

    # ------------------------------------------------------------- panel (a)
    noise_pct = [100.0 * float(n) for n in NOISES]
    means = np.array([np.mean(stream[n]) for n in NOISES])
    lo = np.array([np.min(stream[n]) for n in NOISES])
    hi = np.array([np.max(stream[n]) for n in NOISES])

    ax_a.fill_between(noise_pct, lo, hi, color=JEPA, alpha=0.18, linewidth=0)
    ax_a.plot(noise_pct, means, marker=figstyle.FAMILY_MARKER["jepa"],
              color=JEPA, markersize=4, linewidth=1.0,
              label=r"3-seed mean $\pm$ min/max")
    for i, n in enumerate(NOISES):
        ax_a.plot([noise_pct[i]] * len(stream[n]), stream[n], linestyle="none",
                  marker=".", markersize=3, color=JEPA, alpha=0.6)

    ax_a.set_xticks(noise_pct)
    ax_a.set_xticklabels(["0", "5", "10", "20"])
    ax_a.set_xlabel("tap noise (%)")
    ax_a.set_ylabel(r"impact median $C_L$ $R^2$")
    ax_a.set_title("(a) streaming filter vs tap noise", fontsize=8)
    lo_a, hi_a = ax_a.get_ylim()
    ax_a.set_ylim(lo_a - 0.06 * (hi_a - lo_a), hi_a)
    ax_a.legend(fontsize=6, loc="upper right", handlelength=1.4)
    ax_a.text(0.03, 0.03,
              "streaming multi-encounter protocol,\n"
              "pressure-only init, $n=3$ seeds",
              transform=ax_a.transAxes, fontsize=6, color="#404040",
              ha="left", va="bottom")

    # ------------------------------------------------------------- panel (b)
    rng = np.random.default_rng(0)  # jitter layout only; data values from JSON
    xj = rng.uniform(-0.12, 0.12, size=len(d4_vals))
    ax_b.plot(xj, d4_vals, linestyle="none", marker="o", markersize=4.5,
              markerfacecolor="none", markeredgecolor=JEPA, markeredgewidth=1.1)
    mean_d4 = float(np.mean(d4_vals))
    ax_b.axhline(mean_d4, color=JEPA, linewidth=1.0)
    ax_b.annotate(f"mean {mean_d4:.3f}", xy=(0.52, mean_d4),
                  xytext=(0, 3), textcoords="offset points",
                  fontsize=6, color=JEPA, ha="right")

    ax_b.set_xlim(-0.55, 0.55)
    ax_b.set_xticks([])
    ax_b.set_ylim(min(d4_vals) - 0.010, max(d4_vals) + 0.004)
    ax_b.set_ylabel(r"impact median $C_L$ $R^2$")
    ax_b.set_title(r"(b) $d=4$ filter band", fontsize=8)
    ax_b.text(0.05, 0.04,
              "$K=8$ taps, $d=4$ over-determined\n"
              "observation, $n=5$ member-noise seeds",
              transform=ax_b.transAxes, fontsize=6, color="#404040",
              ha="left", va="bottom")

    fig.text(0.995, 0.01, "in-distribution test set", ha="right", va="bottom",
             fontsize=6, color="#404040")

    fig.tight_layout(w_pad=1.5, rect=(0, 0.03, 1, 1))

    out_dir = REPO / "paper/sections/figures/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig_deployment_v4.pdf")
    fig.savefig(out_dir / "fig_deployment_v4.png", dpi=200)
    print("wrote", out_dir / "fig_deployment_v4.pdf")
    print("wrote", out_dir / "fig_deployment_v4.png")


if __name__ == "__main__":
    main()
