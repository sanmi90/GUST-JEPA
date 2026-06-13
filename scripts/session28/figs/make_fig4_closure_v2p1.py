"""HEADLINE figure (fig4_closure_v2p1.pdf): representational closure of the
UNCONDITIONED predictive (jepa_tf_noc) latent at H = 16 after gust impact, v2.1.

Two panels, matching the v2.1 caption (sec:fig4_closure, section_4_results.tex):

  (a) per-observable held-out R^2 of a FIXED LINEAR (ridge) probe applied to the
      simulation-encoded latent (no rollout) for the six observables on the
      in-distribution test set (test_b, n = 42). The dashed line is the mean over
      the six; the one weak observable, the mid-plane impulse I_y, is shaded light.
  (b) the wake-enstrophy R^2 across families: the predictive latent against the
      reconstructive autoencoder (Fukami) and the linear basis (POD), each d = 64.

All numbers are read from outputs/session28/numbers.json by macro name, so the
figure is byte-faithful to the manuscript text. Read-only; writes only the PDF.

Output: paper/sections/figures/results/fig4_closure_v2p1.pdf
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import _common as cm  # noqa: E402

fs = cm.fs
INK = "#242933"
MUTE = "#6b7280"

# left-panel order: wake first (the point), then circulations, forces, I_y last.
ORDER = ["wake_enstrophy", "circulation_pos", "circulation_neg", "C_L", "C_D", "I_y"]
METRIC_LABEL = {
    "wake_enstrophy": r"wake $\Omega_w$",
    "circulation_pos": r"$\Gamma^{+}$",
    "circulation_neg": r"$\Gamma^{-}$",
    "C_L": r"$C_L$",
    "C_D": r"$C_D$",
    "I_y": r"$I_y$",
}
# macro per observable for the predictive (jepa_tf_noc) representational R^2.
REPR_MACRO = {
    "wake_enstrophy": "NumReprWakeJepaTf",
    "circulation_pos": "NumCloReprCircPosJepaTf",
    "circulation_neg": "NumCloReprCircNegJepaTf",
    "C_L": "NumCloReprCLJepaTf",
    "C_D": "NumCloReprCDJepaTf",
    "I_y": "NumCloReprIyJepaTf",
}
WAKE_FAM_MACRO = [
    ("jepa", "NumReprWakeJepaTf", "predict.\n(uncond.)"),
    ("fukami", "NumReprWakeFukami", "reconstr.\n(Fukami)"),
    ("pod", "NumReprWakePod", "POD"),
]


def main() -> None:
    fs.use_style()
    numbers = cm.load_numbers()
    per = {m: cm.macro_value(numbers, REPR_MACRO[m]) for m in ORDER}
    mean6 = cm.macro_value(numbers, "NumCloReprMeanSixJepaTf")

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=fs.figure_size(1.0, aspect=0.42), gridspec_kw={"width_ratios": [1.7, 1.0]}
    )

    # ---- (a) per-observable representational R^2 of the predictive latent ----
    vals = [per[m] for m in ORDER]
    xs = range(len(ORDER))
    bars = axL.bar(
        xs, vals, color=fs.FAMILY_COLOR["jepa"], width=0.66, edgecolor="white", linewidth=0.5
    )
    for b, m in zip(bars, ORDER):
        if per[m] < 0:  # de-emphasise the one weak (negative) observable, I_y
            b.set_color("#9bbf9e")
    axL.axhline(0.0, color="#000000", lw=0.8)
    axL.axhline(mean6, color=MUTE, lw=1.0, ls="--")
    axL.text(
        0.0,
        mean6 + 0.03,
        f"mean over six $= {mean6:.2f}$",
        ha="left",
        va="bottom",
        fontsize=7,
        color=MUTE,
        style="italic",
    )
    for x, m in zip(xs, ORDER):
        v = per[m]
        axL.text(
            x,
            v + (0.025 if v >= 0 else -0.025),
            f"{v:+.2f}",
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=7,
            color=INK,
            fontweight="bold",
        )
    axL.set_xticks(list(xs))
    axL.set_xticklabels([METRIC_LABEL[m] for m in ORDER], fontsize=7.5)
    axL.set_ylabel(r"held-out $R^2$ (impact $+\,16$)", fontsize=8)
    axL.set_ylim(-1.05, 1.05)
    axL.set_title("(a) Unconditioned latent encodes each observable", fontsize=8, color=INK, pad=5)
    axL.spines[["top", "right"]].set_visible(False)

    # ---- (b) wake-enstrophy R^2 across the three families ----
    wv = [cm.macro_value(numbers, mac) for _, mac, _ in WAKE_FAM_MACRO]
    cols = [fs.FAMILY_COLOR[f] for f, _, _ in WAKE_FAM_MACRO]
    lbls = [lab for _, _, lab in WAKE_FAM_MACRO]
    bx = range(len(WAKE_FAM_MACRO))
    axR.bar(bx, wv, color=cols, width=0.62, edgecolor="white", linewidth=0.5)
    axR.axhline(0.0, color="#000000", lw=0.8)
    for x, v in zip(bx, wv):
        axR.text(
            x,
            v + (0.025 if v >= 0 else -0.025),
            f"{v:+.2f}",
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=7.5,
            color=INK,
            fontweight="bold",
        )
    axR.set_xticks(list(bx))
    axR.set_xticklabels(lbls, fontsize=7)
    axR.set_ylabel(r"wake-enstrophy $R^2$", fontsize=8)
    axR.set_ylim(-0.55, 0.95)
    axR.set_title("(b) The wake is the discriminator", fontsize=8, color=INK, pad=5)
    axR.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(pad=0.5, w_pad=1.6)
    out = cm.out_path("fig4_closure")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")
    print(
        "  (a) per-observable repr R^2: "
        + ", ".join(f"{m}={per[m]:+.2f}" for m in ORDER)
        + f"; mean6={mean6:.2f}"
    )
    print(f"  (b) wake R^2: jepa {wv[0]:+.2f}, fukami {wv[1]:+.2f}, pod {wv[2]:+.2f}")


if __name__ == "__main__":
    main()
