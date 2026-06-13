"""Observability figure (figF_observability_v2p1.pdf): cross-family wall-pressure
observability under the target-blind qDEIM sensor placement. v2.1, two panels.

Matches the v2.1 caption (fig:observability, section_4_results.tex):
  (a) latent-state recovery R^2 versus sensor count K: the predictive (JEPA) state
      is the most recoverable, the linear basis (POD) the least.
  (b) recovered-field large-scale similarity by family (pressure -> latent ->
      decoder) at the headline K = 8, with the decode-ceiling shown alongside: the
      predictive latent beats the reconstructive autoencoder and ties the linear
      basis, whose field parity is its decoder reconstruction ceiling rather than
      better state recovery.

This composes the two panels the 4.7 caption describes from the existing v2.1
sensing artefacts:
  panel (a) from outputs/session28/sensing/cf_results.npz (<fam>_state_r2 vs K_list),
  panel (b) from outputs/session28/sensing/field_recovery/results.npz
            (<fam>_test_b_K8_pr_ssim and _ceil_ssim, mean over encounters).
The values reproduce the manuscript macros (SenseState*, FieldPrSsim*). beta-VAE
appears in (a) only; it has no visualisation decoder, so it is absent from the field
panel (b) by construction (field_recovery families = jepa/fukami/pod).

CPU only; numpy. Read-only on the sensing NPZs; writes only the PDF.

Output: paper/sections/figures/results/figF_observability_v2p1.pdf
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import _common as cm  # noqa: E402

fs = cm.fs

CF = cm.SESSION28 / "sensing" / "cf_results.npz"
FR = cm.SESSION28 / "sensing" / "field_recovery" / "results.npz"

# (data family key, figstyle colour key, display label)
STATE_FAMS = [
    ("jepa_tf_noc", "jepa", "predictive (JEPA)"),
    ("fukami", "fukami", "reconstructive"),
    ("bvae", "oracle", r"$\beta$-VAE"),
    ("pod", "pod", "POD"),
]
FIELD_FAMS = [
    ("jepa_tf_noc", "jepa", "predictive"),
    ("fukami", "fukami", "reconstr."),
    ("pod", "pod", "POD"),
]
BVAE_MARKER = "v"
BVAE_COLOR = "#8c6bb1"
HEADLINE_K = 8


def main() -> None:
    fs.use_style()
    cf = np.load(CF, allow_pickle=True)
    fr = np.load(FR, allow_pickle=True)
    K = cf["K_list"].astype(int)

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=fs.figure_size(1.0, aspect=0.46), gridspec_kw={"width_ratios": [1.25, 1.0]}
    )

    # ---- (a) state recovery R^2 vs K ----
    for famkey, colkey, lab in STATE_FAMS:
        r2 = cf[f"{famkey}_state_r2"].astype(float)
        if famkey == "bvae":
            col, mk = BVAE_COLOR, BVAE_MARKER
        else:
            col, mk = fs.FAMILY_COLOR[colkey], fs.FAMILY_MARKER[colkey]
        axA.plot(K, r2, color=col, marker=mk, ms=4.0, lw=1.2, label=lab)
    axA.set_xscale("log", base=2)
    axA.set_xticks(K)
    axA.set_xticklabels([str(k) for k in K])
    axA.set_xlabel("sensor count $K$")
    axA.set_ylabel(r"latent-state recovery $R^2$ (test\_b)")
    axA.set_ylim(0, 0.9)
    axA.set_title("(a) state recovery", fontsize=8, loc="left")
    axA.legend(
        loc="lower right",
        fontsize=6,
        handlelength=1.3,
        handletextpad=0.4,
        borderpad=0.3,
        labelspacing=0.3,
    )
    axA.grid(alpha=0.2)

    # ---- (b) recovered-field large-scale SSIM at K=8, with the decode ceiling ----
    xs = np.arange(len(FIELD_FAMS))
    w = 0.36
    for j, (famkey, colkey, _) in enumerate(FIELD_FAMS):
        pr = float(np.mean(fr[f"{famkey}_test_b_K{HEADLINE_K}_pr_ssim"]))
        ce = float(np.mean(fr[f"{famkey}_test_b_K{HEADLINE_K}_ceil_ssim"]))
        col = fs.FAMILY_COLOR[colkey]
        axB.bar(
            xs[j] - w / 2,
            pr,
            width=w,
            color=col,
            edgecolor="white",
            linewidth=0.5,
            label="pressure-recovered" if j == 0 else None,
        )
        axB.bar(
            xs[j] + w / 2,
            ce,
            width=w,
            color=col,
            alpha=0.45,
            edgecolor="white",
            linewidth=0.5,
            hatch="///",
            label="decode ceiling" if j == 0 else None,
        )
        axB.text(xs[j] - w / 2, pr + 0.01, f"{pr:.2f}", ha="center", va="bottom", fontsize=6)
    axB.set_xticks(xs)
    axB.set_xticklabels([lab for _, _, lab in FIELD_FAMS], fontsize=7)
    axB.set_ylabel(r"recovered-field SSIM ($K=8$)")
    axB.set_ylim(0, 0.85)
    axB.set_title("(b) field recovery", fontsize=8, loc="left")
    axB.legend(loc="upper right", fontsize=6, handlelength=1.0, handletextpad=0.4, borderpad=0.3)

    fig.tight_layout(pad=0.5, w_pad=1.6)
    out = cm.out_path("figF_observability")
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    i8 = list(K).index(HEADLINE_K)
    print(
        "  (a) state R^2 at K=8: "
        + ", ".join(f"{f}={cf[f'{f}_state_r2'][i8]:.2f}" for f, _, _ in STATE_FAMS)
    )
    print(
        "  (b) field K=8 pr_ssim: "
        + ", ".join(f"{f}={np.mean(fr[f'{f}_test_b_K8_pr_ssim']):.2f}" for f, _, _ in FIELD_FAMS)
    )


if __name__ == "__main__":
    main()
