"""Part C figure (F14): direct vs autoregressive 40-step forecast (register R11).

Panels
  (a) representative test B encounter: DNS truth C_L vs the as-built transformer
      predictor rolled autoregressively for 40 steps from a 25-frame context
      (decoded through the frozen affine probe). The rollout collapses through
      the gust impact; shown for the CLW flagship cell and the CLN cell.
  (b) pooled decoded-C_L R^2 over the identical 40-step window (context 25,
      split test_b): the DIRECT multi-horizon latent-REX forecaster (tuned
      h512 and the default operator across encoder seeds) vs the
      autoregressive transformer rollouts. One shot, no compounding, is the
      difference between about +0.70 and about -0.62.

Every number is loaded from its JSON at build time:
  outputs/session34/trackc_forecast.json           (AR rollouts: rolled_cl_r2,
                                                    rep_truth, rep_roll_cl,
                                                    latent vs-persistence skill)
  outputs/session34/latent_rex_tuned_testb.json    (direct REX, tuned winner)
  outputs/session34/latent_rex_jepa_pool_vec.json  (direct REX default, seed 0)
  outputs/session34/latent_rex_jepa_pool_vec_s1.json (direct REX default, seed 1)

Protocol (documented in scripts/session34/{latent_rex,tirex_forecast}.py):
context frames [0, 25), horizon 40 through the gust impact, split test_b
(42 encounters), decoded C_L via the frozen affine probe.
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

# operator colours (both operators run on JEPA-family latents, so the family
# green is reserved for the direct operator and a paired PRGn purple marks the
# autoregressive rollout; DNS truth uses the oracle grey).
C_DIRECT = figstyle.FAMILY_COLOR["jepa"]   # #1b7837
C_AR = "#762a83"                            # PRGn purple, AR rollout (CLW)
C_AR_LIGHT = "#9970ab"                      # lighter purple, AR rollout (CLN)
C_TRUTH = figstyle.FAMILY_COLOR["oracle"]  # #404040

CTX, H = 25, 40  # context frames [0, 25), horizon 40 (protocol constants)


def load(path: str) -> dict:
    with open(REPO / path) as f:
        return json.load(f)


def main() -> None:
    figstyle.use_style()

    fc = load("outputs/session34/trackc_forecast.json")
    rex_tuned = load("outputs/session34/latent_rex_tuned_testb.json")
    rex_s0 = load("outputs/session34/latent_rex_jepa_pool_vec.json")
    rex_s1 = load("outputs/session34/latent_rex_jepa_pool_vec_s1.json")

    truth = np.asarray(fc["clw"]["rep_truth"])
    roll_clw = np.asarray(fc["clw"]["rep_roll_cl"])
    roll_cln = np.asarray(fc["cln"]["rep_roll_cl"])
    assert len(roll_clw) == H and len(roll_cln) == H

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2,
        figsize=(figstyle.TEXTWIDTH_IN, figstyle.TEXTWIDTH_IN * 0.40),
        gridspec_kw={"width_ratios": [1.45, 1.0]},
    )

    # ------------------------------------------------------------- panel (a)
    frames = np.arange(len(truth))
    ax_a.axvspan(0, CTX - 1, color="0.92", zorder=0)
    ax_a.axvline(40, color="0.55", linestyle=":", linewidth=0.8, zorder=1)
    ax_a.plot(frames, truth, color=C_TRUTH, linewidth=1.0, label="DNS truth")
    f_roll = np.arange(CTX, CTX + H)
    ax_a.plot(f_roll, roll_clw, color=C_AR, linewidth=1.0,
              label="AR rollout (wake)")
    ax_a.plot(f_roll, roll_cln, color=C_AR_LIGHT, linewidth=1.0,
              linestyle="--", label="AR rollout (lift)")
    ymin = min(truth.min(), roll_clw.min(), roll_cln.min())
    ymax = max(truth.max(), roll_clw.max(), roll_cln.max())
    ax_a.text(CTX / 2, ymax, "context", ha="center", va="top",
              fontsize=6.5, color="0.35")
    ax_a.text(41, ymin, "impact", ha="left", va="bottom",
              fontsize=6.5, color="0.35")
    ax_a.set_xlim(0, len(truth) - 1)
    ax_a.set_xlabel(r"frame ($\Delta t = 0.05\,t/c$)")
    ax_a.set_ylabel(r"$C_L$")
    ax_a.legend(fontsize=6, loc="upper right", handlelength=1.4,
                borderaxespad=0.2)
    ax_a.set_title("representative encounter", fontsize=8)

    # ------------------------------------------------------------- panel (b)
    rows = [
        {
            "label": "direct forecaster (tuned)",
            "vals": [rex_tuned["decoded_cl_r2"]],
            "color": C_DIRECT, "marker": "o", "filled": True,
        },
        {
            "label": "direct forecaster (default,\n2 encoder seeds)",
            "vals": [rex_s0["decoded_cl_r2"], rex_s1["decoded_cl_r2"]],
            "color": C_DIRECT, "marker": "o", "filled": False,
        },
        {
            "label": "AR transformer (wake)",
            "vals": [fc["clw"]["rolled_cl_r2"]],
            "color": C_AR, "marker": "s", "filled": True,
        },
        {
            "label": "AR transformer (lift)",
            "vals": [fc["cln"]["rolled_cl_r2"]],
            "color": C_AR_LIGHT, "marker": "s", "filled": True,
        },
    ]
    ax_b.axvline(0.0, color="0.6", linewidth=0.7, zorder=0)
    y_pos = np.arange(len(rows))[::-1]
    for y, row in zip(y_pos, rows):
        vals = row["vals"]
        face = row["color"] if row["filled"] else "none"
        ax_b.scatter(vals, [y] * len(vals), s=26, marker=row["marker"],
                     facecolors=face, edgecolors=row["color"],
                     linewidths=1.0, zorder=3)
        anchor = np.mean(vals)
        ax_b.text(anchor, y + 0.22, f"{anchor:+.2f}" if len(vals) == 1
                  else f"{min(vals):+.2f} / {max(vals):+.2f}",
                  ha="center", va="bottom", fontsize=6, color="0.25")
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels([r["label"] for r in rows], fontsize=6.5)
    ax_b.set_ylim(-0.6, len(rows) - 0.2)
    ax_b.set_xlim(-0.85, 0.95)
    ax_b.set_xlabel(r"decoded $C_L$ $R^2$")
    ax_b.set_title("40-step forecast, context 25", fontsize=8)
    ax_b.spines["left"].set_visible(False)
    ax_b.tick_params(axis="y", length=0)

    # latent-space skill vs persistence, from the same JSONs (footer note).
    skill_rex = rex_s0["latent_r2_vs_persistence"]
    skill_ar40 = fc["clw"]["latent"]["40"]["vs_persist"]
    fig.text(
        0.995, 0.01,
        "latent-space skill vs persistence: direct forecaster "
        f"{skill_rex:+.2f} (pooled H=40), AR rollout {skill_ar40:+.2f} at H=40; "
        "test set (42 encounters)",
        ha="right", va="bottom", fontsize=6, color="#404040",
    )

    fig.tight_layout(w_pad=1.4, rect=(0, 0.04, 1, 1))

    out_dir = REPO / "paper/sections/figures/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig_forecast40_v4.pdf")
    fig.savefig(out_dir / "fig_forecast40_v4.png", dpi=200)
    print("wrote", out_dir / "fig_forecast40_v4.pdf")
    print("wrote", out_dir / "fig_forecast40_v4.png")

    # build-time echo of the headline numbers (traceability)
    print(f"  direct REX tuned  : {rex_tuned['decoded_cl_r2']:+.4f}")
    print(f"  direct REX default: {rex_s0['decoded_cl_r2']:+.4f} / "
          f"{rex_s1['decoded_cl_r2']:+.4f}")
    print(f"  AR rollout CLW    : {fc['clw']['rolled_cl_r2']:+.4f}")
    print(f"  AR rollout CLN    : {fc['cln']['rolled_cl_r2']:+.4f}")


if __name__ == "__main__":
    main()
