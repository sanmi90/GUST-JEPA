#!/usr/bin/env python3
"""Headline figure (fig4_closure.pdf): REPRESENTATIONAL closure of the
UNCONDITIONED tf-no-c JEPA.

Probe the ENCODED (z_dns) latent at impact+16 for each of the six observables on
held-out test_b, for the fully unconditioned predictive model (no gust parameters
in the encoder or the predictor). Two panels:

  (left)  per-observable held-out R^2 for tf-no-c from z_dns at impact+16. Numbers
          are read from outputs/session27/closure6_noc.json (mode label "dns"), so
          the figure stays in sync with the closure script. The dashed line marks
          the mean-over-6.
  (right) the wake-enstrophy R^2 across families: tf-no-c (unconditioned z_dns)
          against the reconstructive AE (Fukami) and POD. Baseline wake R^2 are
          the panel-(a) "wake R^2" column of results_tables.tex (each baseline's
          least-bad d, to be maximally fair).

This is the manuscript counterpart of paper/talk/make_repr_closure.py, emitted as
a JFM-width PDF straight into the results figure directory. Read-only on
outputs/session27; writes only the manuscript figure.

Output: paper/sections/figures/results/fig4_closure.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
# .../<repo>/paper/sections/figures -> repo root is three levels up.
REPO = HERE.parent.parent.parent
CLOSURE_JSON = REPO / "outputs" / "session27" / "closure6_noc.json"
OUT = HERE / "results" / "fig4_closure.pdf"

# JFM measured text width (figstyle.py), include at scale 1.
PT_PER_INCH = 72.27
TEXTWIDTH_IN = 360.0 / PT_PER_INCH  # ~4.98 in

# family colours (match scripts/session21/figstyle.py)
C_JEPA = "#1b7837"   # green  - predictive / JEPA (this work)
C_FUK = "#c0392b"    # red    - reconstructive autoencoder
C_POD = "#2166ac"    # blue   - POD linear basis
INK = "#242933"
MUTE = "#6b7280"

METRIC_LABEL = {
    "C_L": r"$C_L$",
    "C_D": r"$C_D$",
    "wake_enstrophy": r"wake $\Omega_w$",
    "circulation_pos": r"$\Gamma^{+}$",
    "circulation_neg": r"$\Gamma^{-}$",
    "I_y": r"$I_y$",
}
# left-panel order: wake first (the point), then circulations, forces, I_y last
ORDER = ["wake_enstrophy", "circulation_pos", "circulation_neg", "C_L", "C_D", "I_y"]

# Panel-(a) wake-enstrophy R^2 (results_tables.tex). Use each baseline's best
# (least-bad) value across d; tf-no-c is the UNCONDITIONED z_dns wake number.
WAKE_R2 = {
    "tf-no-c": None,  # filled from closure6_noc.json (z_dns, impact+16, test_b)
    "Fukami": 0.06,   # best of {0.06, 0.06, -0.41, -0.21}
    "POD": -0.17,     # best of {-0.32, -0.17, -0.31}
}
# Conditioned representational reference (results_tables.tex panel (a), JEPA d=64).
WAKE_R2_COND = 0.75


def main() -> None:
    blob = json.loads(CLOSURE_JSON.read_text())
    node = blob["headline"]["tf-no-c"]["test_b"]["16"]["dns"]
    per = node["per_metric"]
    mean6 = node["mean_over_6"]
    WAKE_R2["tf-no-c"] = float(per["wake_enstrophy"])

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.serif": ["DejaVu Serif", "Computer Modern Roman"],
        "font.size": 8,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
    })
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.42),
        gridspec_kw={"width_ratios": [1.7, 1.0]})

    # ---- left: per-observable R^2 for the unconditioned latent ----
    vals = [per[m] for m in ORDER]
    xs = range(len(ORDER))
    bars = axL.bar(xs, vals, color=C_JEPA, width=0.66, edgecolor="white", linewidth=0.5)
    for b, m in zip(bars, ORDER):
        if per[m] < 0:
            b.set_color("#9bbf9e")  # de-emphasise the one weak (negative) observable
    axL.axhline(0.0, color="#000000", lw=0.8)
    axL.axhline(mean6, color=MUTE, lw=1.0, ls="--")
    axL.text(0.0, mean6 + 0.03, f"mean over six $= {mean6:.2f}$",
             ha="left", va="bottom", fontsize=7, color=MUTE, style="italic")
    for x, m in zip(xs, ORDER):
        v = per[m]
        axL.text(x, v + (0.025 if v >= 0 else -0.025), f"{v:+.2f}",
                 ha="center", va="bottom" if v >= 0 else "top",
                 fontsize=7, color=INK, fontweight="bold")
    axL.set_xticks(list(xs))
    axL.set_xticklabels([METRIC_LABEL[m] for m in ORDER], fontsize=7.5)
    axL.set_ylabel(r"held-out $R^2$ (impact $+\,16$)", fontsize=8)
    axL.set_ylim(-1.05, 1.05)
    axL.set_title("(a) Unconditioned latent encodes each observable",
                  fontsize=8, color=INK, pad=5)
    axL.spines[["top", "right"]].set_visible(False)

    # ---- right: wake-enstrophy R^2 across families ----
    fams = ["tf-no-c", "Fukami", "POD"]
    fam_lbl = ["predict.\n(uncond.)", "reconstr.\n(Fukami)", "POD"]
    fam_col = [C_JEPA, C_FUK, C_POD]
    wv = [WAKE_R2[f] for f in fams]
    bx = range(len(fams))
    axR.bar(bx, wv, color=fam_col, width=0.62, edgecolor="white", linewidth=0.5)
    axR.axhline(0.0, color="#000000", lw=0.8)
    axR.axhline(WAKE_R2_COND, color=MUTE, lw=0.9, ls=":")
    axR.text(len(fams) - 0.5, WAKE_R2_COND + 0.02,
             f"conditioned $= {WAKE_R2_COND:.2f}$",
             ha="right", va="bottom", fontsize=6.5, color=MUTE, style="italic")
    for x, v in zip(bx, wv):
        axR.text(x, v + (0.025 if v >= 0 else -0.025), f"{v:+.2f}",
                 ha="center", va="bottom" if v >= 0 else "top",
                 fontsize=7.5, color=INK, fontweight="bold")
    axR.set_xticks(list(bx))
    axR.set_xticklabels(fam_lbl, fontsize=7)
    axR.set_ylabel(r"wake-enstrophy $R^2$", fontsize=8)
    axR.set_ylim(-0.55, 0.95)
    axR.set_title("(b) The wake is the discriminator", fontsize=8, color=INK, pad=5)
    axR.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(pad=0.5, w_pad=1.6)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")
    print("  tf-no-c per-observable z_dns R^2 (impact+16, test_b): "
          + ", ".join(f"{m}={per[m]:+.3f}" for m in ORDER))
    print(f"  mean over six = {mean6:.3f}")
    print(f"  wake R^2: tf-no-c {WAKE_R2['tf-no-c']:+.2f} vs Fukami "
          f"{WAKE_R2['Fukami']:+.2f} vs POD {WAKE_R2['POD']:+.2f} "
          f"(conditioned {WAKE_R2_COND:+.2f})")


if __name__ == "__main__":
    main()
