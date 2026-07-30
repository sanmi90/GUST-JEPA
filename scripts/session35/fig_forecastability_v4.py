"""Part C figure (F15): family forecastability under the shared REX operator
plus the conditioning null (registers R10 and R12).

Panels
  (a) decoded C_L R^2 per latent family under the SHARED default latent-REX
      operator (identical architecture, training recipe and protocol per
      family; the operator confound fix). Families: JEPA-CLW (native vector
      flagship), JEPA-CLN (Chang lift-element head cell), kit AE-LW. Bar =
      mean over operator seeds, points = individual seeds, n from the
      files present on disk. The seeds are OPERATOR seeds: latent_rex.py
      loads a frozen encoder run by --run and uses --seed only for operator
      training (the _s{n} suffix), so the three files per family share one
      encoder. An earlier version of this docstring and of panel (a)'s title
      said "encoder seeds"; that was wrong and contradicted panel (b) below,
      which uses the same mechanism. See editorial/REVIEW_LOG.md F22.1.
  (b) the conditioning null on the tuned REX (LSTM h512, CLW latents): arms
      none / +phase / +phase+(G,D,Y) oracle, 3 operator seeds each. Oracle
      gust parameters DEGRADE the forecast; the deployable phase covariate is
      a wash with unconditioned.

Every number is loaded from its JSON at build time:
  outputs/session34/latent_rex_jepa_pool_vec{,_s1,_s2}.json   (CLW seeds)
  outputs/session34/latent_rex_jepa_pool_ln_s{0,1,2}.json     (CLN seeds)
  outputs/session34/latent_rex_ae_wake_pool{,_s1,_s2}.json    (AE-LW seeds)
  outputs/session34/rex2_cov.json                             (null, seed 0)
  outputs/session35/rex2_cov_s{1,2}.json                      (null, seeds 1-2)

Protocol: direct 40-step forecast, context 25, split test_b (42 encounters),
decoded C_L via the frozen affine probe (scripts/session34/latent_rex.py,
scripts/session34/rex2_cov.py).
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

S34 = REPO / "outputs/session34"
S35 = REPO / "outputs/session35"

# explicit candidate lists (NOT a glob: latent_rex_*_d{4,8,16}.json and the
# rexpred cells share the prefix and must not leak into the seed count).
FAMILIES = {
    "clw": {
        "label": "predictive\n(wake)",
        "files": [S34 / "latent_rex_jepa_pool_vec.json",
                  S34 / "latent_rex_jepa_pool_vec_s1.json",
                  S34 / "latent_rex_jepa_pool_vec_s2.json"],
        "color": figstyle.FAMILY_COLOR["jepa"],
        "marker": figstyle.FAMILY_MARKER["jepa"],
    },
    "cln": {
        "label": "predictive\n(lift)",
        "files": [S34 / "latent_rex_jepa_pool_ln_s0.json",
                  S34 / "latent_rex_jepa_pool_ln_s1.json",
                  S34 / "latent_rex_jepa_pool_ln_s2.json"],
        "color": "#74c476",  # lighter green, same CLN key as fig_ownstack_da_v4
        "marker": "D",
    },
    "ae_lw": {
        "label": "AE (wake)",
        "files": [S34 / "latent_rex_ae_wake_pool.json",
                  S34 / "latent_rex_ae_wake_pool_s1.json",
                  S34 / "latent_rex_ae_wake_pool_s2.json"],
        "color": figstyle.FAMILY_COLOR["fukami"],
        "marker": figstyle.FAMILY_MARKER["fukami"],
    },
}

NULL_SEED_FILES = [S34 / "rex2_cov.json",
                   S35 / "rex2_cov_s1.json",
                   S35 / "rex2_cov_s2.json"]
NULL_ARMS = [("none", "none", figstyle.FAMILY_COLOR["jepa"]),
             ("phase", "+phase", "#74c476"),
             ("phase_gdy", "+phase\n+(G,D,Y)", "#8c8c8c")]


def load_vals(files: list[Path]) -> list[float]:
    vals = []
    for f in files:
        if f.exists():
            with open(f) as fh:
                vals.append(json.load(fh)["decoded_cl_r2"])
    return vals


def seed_scatter(ax, x, vals, color, marker):
    off = np.linspace(-0.10, 0.10, len(vals)) if len(vals) > 1 else [0.0]
    ax.scatter(x + np.asarray(off), vals, s=22, marker=marker,
               facecolors="white", edgecolors=color, linewidths=1.0, zorder=4)


def main() -> None:
    figstyle.use_style()

    # Session 39 (Carlos's assessment, fig 11): the main-text forecast figure keeps
    # only the shared-operator family merit; the conditioning/oracle null moves to
    # the appendix as a separate figure.
    fig_a, ax_a = plt.subplots(
        1, 1, figsize=(figstyle.TEXTWIDTH_IN * 0.60, figstyle.TEXTWIDTH_IN * 0.40))
    fig_b, ax_b = plt.subplots(
        1, 1, figsize=(figstyle.TEXTWIDTH_IN * 0.50, figstyle.TEXTWIDTH_IN * 0.40))

    # ------------------------------------------------------------- panel (a)
    for i, (key, spec) in enumerate(FAMILIES.items()):
        vals = load_vals(spec["files"])
        if not vals:
            raise FileNotFoundError(f"no latent-REX JSON found for {key}")
        mean = float(np.mean(vals))
        ax_a.bar(i, mean, width=0.62, color=spec["color"],
                 edgecolor="black", linewidth=0.4, zorder=2)
        seed_scatter(ax_a, i, vals, "black", spec["marker"])
        ax_a.text(i, 0.02, f"$n={len(vals)}$", ha="center", va="bottom",
                  fontsize=6, color="white", zorder=5)
        print(f"  (a) {spec['label']:9s} n={len(vals)}  "
              f"mean={mean:+.4f}  seeds={[round(v, 4) for v in vals]}")
    ax_a.set_xticks(range(len(FAMILIES)))
    ax_a.set_xticklabels([s["label"] for s in FAMILIES.values()], fontsize=6.5)
    ax_a.set_ylabel(r"decoded $C_L$ $R^2$")
    ax_a.set_ylim(0.0, 0.85)
    ax_a.set_title("shared direct forecaster, operator seeds", fontsize=8)

    # ------------------------------------------------------------- panel (b)
    null = []
    for f in NULL_SEED_FILES:
        with open(f) as fh:
            null.append(json.load(fh))
    for i, (arm, label, color) in enumerate(NULL_ARMS):
        vals = [seed[arm]["decoded_cl_r2"] for seed in null]
        mean = float(np.mean(vals))
        ax_b.bar(i, mean, width=0.62, color=color,
                 edgecolor="black", linewidth=0.4, zorder=2)
        seed_scatter(ax_b, i, vals, "black", "o")
        print(f"  (b) {arm:9s} n={len(vals)}  "
              f"mean={mean:+.4f}  seeds={[round(v, 4) for v in vals]}")
    ax_b.set_xticks(range(len(NULL_ARMS)))
    ax_b.set_xticklabels([a[1] for a in NULL_ARMS], fontsize=6.5)
    ax_b.set_ylabel(r"decoded $C_L$ $R^2$")
    ax_b.set_ylim(0.0, 0.85)
    ax_b.text(2, 0.045, "oracle", ha="center", va="bottom",
              fontsize=6, color="white", zorder=5)
    ax_b.set_title(f"conditioning null (predictive wake),\n"
                   f"$n={len(null)}$ operator seeds", fontsize=8)

    out_dir = REPO / "paper/sections/figures/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    for fig, out in ((fig_a, out_dir / "fig_forecastability_v4.pdf"),
                     (fig_b, out_dir / "fig_forecast_null_v4.pdf")):
        fig.tight_layout()
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
