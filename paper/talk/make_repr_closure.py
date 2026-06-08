#!/usr/bin/env python3
"""Slide-12 figure: REPRESENTATIONAL closure of the UNCONDITIONED tf-no-c JEPA.

Probe the ENCODED (z_dns) latent at impact+16 for each of the six observables on
held-out test_b. Two panels:

  (left)  per-observable held-out R^2 for tf-no-c (the unconditioned predictive
          latent) from z_dns at impact+16. Numbers are read from
          outputs/session27/closure6_noc.json (mode label "dns"), so the figure
          stays in sync with the closure script. I_y is excluded: the latent
          does not encode it and it is not comparable to the other observables.
  (right) the wake-enstrophy R^2 across families: tf-no-c (unconditioned z_dns)
          vs the reconstructive AE (Fukami) and POD. Baseline wake R^2 are the
          panel-(a) "wake R^2" column of paper/sections/tables/results_tables.tex
          (their least-bad d is used, to be maximally fair to the baselines).

Output: paper/talk/figs/repr_closure.png  (nothing else touched).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CLOSURE_JSON = REPO / "outputs" / "session27" / "closure6_noc.json"
OUT = HERE / "figs" / "repr_closure.png"

# family colours (match scripts/session21/figstyle.py)
C_JEPA = "#1b7837"   # green  - predictive / JEPA (this work)
C_FUK = "#c0392b"    # red    - reconstructive autoencoder
C_POD = "#2166ac"    # blue   - POD linear basis
INK = "#242933"
MUTE = "#6b7280"

METRIC_LABEL = {
    "C_L": r"$C_L$",
    "C_D": r"$C_D$",
    "wake_enstrophy": "wake\nenstrophy",
    "circulation_pos": r"$\Gamma_+$",
    "circulation_neg": r"$\Gamma_-$",
}
# left-panel order: wake first (the point), then the two circulations, then forces
ORDER = ["wake_enstrophy", "circulation_pos", "circulation_neg", "C_L", "C_D"]

# Panel-(a) wake-enstrophy R^2 (results_tables.tex). Use each baseline's best
# (least-bad) value across d; tf-no-c is the UNCONDITIONED z_dns wake number.
WAKE_R2 = {
    "tf-no-c": None,          # filled from closure6_noc.json (z_dns, impact+16, test_b)
    "Fukami": 0.06,           # best of {0.06, 0.06, -0.41, -0.21}
    "POD": -0.17,             # best of {-0.32, -0.17, -0.31}
}


def main() -> None:
    blob = json.loads(CLOSURE_JSON.read_text())
    per = blob["headline"]["tf-no-c"]["test_b"]["16"]["dns"]["per_metric"]
    WAKE_R2["tf-no-c"] = float(per["wake_enstrophy"])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.linewidth": 0.8, "savefig.dpi": 220})
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.4, 3.9),
                                   gridspec_kw={"width_ratios": [1.7, 1.0]})

    # ---- left: per-observable R^2 for the unconditioned latent ----
    vals = [per[m] for m in ORDER]
    xs = range(len(ORDER))
    bars = axL.bar(xs, vals, color=C_JEPA, width=0.66, edgecolor="white", linewidth=0.6)
    # de-emphasise the negative (weak) observable
    for b, m in zip(bars, ORDER):
        if per[m] < 0:
            b.set_color("#9bbf9e")
    axL.axhline(0.0, color="#000000", lw=0.8)
    for x, m in zip(xs, ORDER):
        v = per[m]
        axL.text(x, v + (0.03 if v >= 0 else -0.03), f"{v:+.2f}",
                 ha="center", va="bottom" if v >= 0 else "top",
                 fontsize=10, color=INK, fontweight="bold")
    axL.set_xticks(list(xs))
    axL.set_xticklabels([METRIC_LABEL[m] for m in ORDER], fontsize=10.5)
    axL.set_ylabel("held-out $R^2$  (impact + 16)", fontsize=11)
    axL.set_ylim(-0.06, 1.08)
    axL.set_title("Unconditioned latent encodes each observable",
                  fontsize=12, color=INK, pad=8)
    axL.spines[["top", "right"]].set_visible(False)

    # ---- right: wake-enstrophy R^2 across families ----
    fams = ["tf-no-c", "Fukami", "POD"]
    fam_lbl = ["predictive\n(unconditioned)", "reconstructive\n(Fukami)", "POD"]
    fam_col = [C_JEPA, C_FUK, C_POD]
    wv = [WAKE_R2[f] for f in fams]
    bx = range(len(fams))
    axR.bar(bx, wv, color=fam_col, width=0.62, edgecolor="white", linewidth=0.6)
    axR.axhline(0.0, color="#000000", lw=0.8)
    for x, v in zip(bx, wv):
        axR.text(x, v + (0.03 if v >= 0 else -0.03), f"{v:+.2f}",
                 ha="center", va="bottom" if v >= 0 else "top",
                 fontsize=11, color=INK, fontweight="bold")
    axR.set_xticks(list(bx))
    axR.set_xticklabels(fam_lbl, fontsize=10)
    axR.set_ylabel("wake-enstrophy $R^2$", fontsize=11)
    axR.set_ylim(-0.55, 0.95)
    axR.set_title("The wake is the discriminator", fontsize=12, color=INK, pad=8)
    axR.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(pad=0.8, w_pad=2.0)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")
    print(f"  tf-no-c per-observable z_dns R^2 (impact+16, test_b): "
          + ", ".join(f"{m}={per[m]:+.3f}" for m in ORDER))
    print(f"  wake R^2: tf-no-c {WAKE_R2['tf-no-c']:+.2f} vs Fukami {WAKE_R2['Fukami']:+.2f} "
          f"vs POD {WAKE_R2['POD']:+.2f}")


if __name__ == "__main__":
    main()
