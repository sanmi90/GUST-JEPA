"""F5 (v3 paper): the mechanism figure.

Supervision-induced covariance anisotropy builds a protected subspace (panel a,
near-null departure-energy fractions + condition numbers from
outputs/session32/track_p3_mechanism.json), and the multi-step rollout training
objective is what keeps the long forecast horizon usable (panels b, c; merit and
drift versus horizon for H_roll = 1 vs 8 from outputs/session32/hroll_ablation.json).

All plotted values are read from the two JSONs; nothing is hand-typed.
Output: outputs/session33/figures/fig_mechanism_hroll_v3.pdf
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "session21"))
from figstyle import FAMILY_COLOR, TEXTWIDTH_IN, use_style  # noqa: E402

MECH_JSON = REPO / "outputs/session32/track_p3_mechanism.json"
HROLL_JSON = REPO / "outputs/session32/hroll_ablation.json"
OUT_PDF = REPO / "outputs/session33/figures/fig_mechanism_hroll_v3.pdf"

# Model display order and colours for panel (a): the reconstruction-regularised
# candidate in the reconstructive family red, supervised-only as the intermediate
# (orange, matches the figstyle boundary accent), the full objective in JEPA green.
BAR_MODELS = [
    ("regAE_pool", "regAE", FAMILY_COLOR["fukami"]),
    ("supervised_only_pool", "supervised\nonly", "#e08214"),
    ("jepa_wake_pool", "JEPA\n+wake", FAMILY_COLOR["jepa"]),
]
HROLL_STYLE = {
    "H_roll_1": {"color": "#7f7f7f", "ls": "--", "marker": "s",
                 "label": r"$H_{\rm roll} = 1$ (single step)"},
    "H_roll_8": {"color": FAMILY_COLOR["jepa"], "ls": "-", "marker": "o",
                 "label": r"$H_{\rm roll} = 8$ (multi-step)"},
}


def load_mechanism() -> dict:
    """Near-null fractions, condition numbers, and the isotropic null."""
    d = json.loads(MECH_JSON.read_text())
    comps = d["comparisons"]["empirical"]
    frac = {
        "regAE_pool": comps["regAE_pool_vs_jepa_wake_pool"]["cand_frac"],
        "jepa_wake_pool": comps["regAE_pool_vs_jepa_wake_pool"]["comp_frac"],
        "supervised_only_pool": comps["regAE_pool_vs_supervised_only_pool"]["comp_frac"],
    }
    cond = {}
    for m in frac:
        ev = d["models"][m]["estimators"]["empirical"]["eigenvalues_ascending"]
        cond[m] = max(ev) / min(ev)
    null_frac = float(d["isotropic_null_fraction"])
    dim = int(d["params"]["d"])
    k = round(null_frac * dim)
    return {"frac": frac, "cond": cond, "null_frac": null_frac, "k": k, "d": dim}


def load_hroll() -> dict:
    """Merit and drift curves versus horizon for H_roll in {1, 8}."""
    d = json.loads(HROLL_JSON.read_text())["axis_a_forecast"]
    out = {}
    for tag in ("H_roll_1", "H_roll_8"):
        merit = d[tag]["merit_curve"]
        drift = d[tag]["drift_curve"]
        hs = sorted(int(h) for h in merit)
        out[tag] = {
            "h": hs,
            "merit": [merit[str(h)] for h in hs],
            "drift": [drift[str(h)] for h in hs],
        }
    return out


def panel_a(ax, mech: dict) -> None:
    xs = range(len(BAR_MODELS))
    for x, (key, label, color) in zip(xs, BAR_MODELS):
        f = mech["frac"][key]
        ax.bar(x, f, width=0.62, color=color, alpha=0.9, zorder=3)
        ax.text(x, f * 1.12, f"{f:.3f}" + "\n" + rf"$\kappa = {mech['cond'][key]:.0f}$",
                ha="center", va="bottom", fontsize=6.2, linespacing=1.15)
    ax.axhline(mech["null_frac"], color="0.35", ls="--", lw=0.9, zorder=2)
    ax.text(len(BAR_MODELS) - 0.52, mech["null_frac"] * 1.12,
            rf"isotropic null $k/d = {mech['k']}/{mech['d']}$",
            ha="right", va="bottom", fontsize=6.2, color="0.35")
    ax.set_yscale("log")
    ax.set_ylim(6e-3, 0.55)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([lbl for _, lbl, _ in BAR_MODELS], fontsize=6.2)
    ax.set_ylabel("near-null departure\nenergy fraction")
    ax.set_title("(a) supervision builds the\nprotected subspace", fontsize=7.5)


def panel_curves(ax, curves: dict, which: str, ylabel: str, title: str) -> None:
    for tag in ("H_roll_1", "H_roll_8"):
        st = HROLL_STYLE[tag]
        ax.plot(curves[tag]["h"], curves[tag][which], color=st["color"],
                ls=st["ls"], marker=st["marker"], ms=3.5, lw=1.1,
                label=st["label"])
    ax.set_xscale("log", base=2)
    hs = curves["H_roll_8"]["h"]
    ax.set_xticks(hs)
    ax.set_xticklabels([str(h) for h in hs])
    ax.minorticks_off()
    ax.set_xlabel(r"forecast horizon $h$ (frames)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=7.5)


def main() -> int:
    use_style()
    mech = load_mechanism()
    curves = load_hroll()

    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 2.05))
    panel_a(axes[0], mech)
    panel_curves(axes[1], curves, "merit",
                 r"mean observable $R^2$",
                 "(b) multi-step objective holds\nmerit at long horizon")
    axes[1].axhline(0.0, color="0.85", lw=0.6, zorder=0)
    axes[1].legend(loc="lower left", fontsize=6.2, handlelength=1.8)
    panel_curves(axes[2], curves, "drift",
                 r"latent drift (relative $L_2$)",
                 "(c) and slows the open-loop\nlatent drift")
    fig.subplots_adjust(left=0.085, right=0.995, top=0.85, bottom=0.19, wspace=0.42)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    plt.close(fig)
    print(f"wrote {OUT_PDF}")
    for key, lbl, _ in BAR_MODELS:
        print(f"  {key}: frac={mech['frac'][key]:.4f} kappa={mech['cond'][key]:.0f}")
    for tag in ("H_roll_1", "H_roll_8"):
        print(f"  {tag}: merit h1={curves[tag]['merit'][0]:.3f} "
              f"h16={curves[tag]['merit'][-1]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
