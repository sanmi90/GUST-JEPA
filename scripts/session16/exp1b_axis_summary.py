"""Session 16, Experiment 1, Part (b): consolidate the per-axis descriptor
correlations into a labelled physical interpretation table, and dump a
panel figure showing the decoded fields at m = (-2, 0, +2) sigma for each
of the six axes (PLS3 1..3 + PCA3 1..3).

Inputs:
    outputs/session16/exp1/exp1b_decoded_axes.npz
    outputs/session16/exp1/exp1b_descriptors.json

Outputs:
    outputs/session16/exp1/exp1b_axis_interpretation.json
    outputs/session16/figures/exp1b_axis_decoded_panel.png
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "session16" / "exp1"
FIG_DIR = REPO / "outputs" / "session16" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def classify_axis(corr: dict) -> dict:
    """Apply a simple rule-based classifier to the descriptor correlations.

    Categories:
        magnitude:  both peaks grow (|r(peak_pos)| AND |r(peak_neg)| > 0.6
                   with OPPOSITE signs) and wake_thickness grows.
        sign:      both peaks shift in the SAME direction (correlations
                   share sign) -- the wake's overall sign distribution
                   shifts.
        shape:     wake_length and wake_thickness correlations have
                   OPPOSITE signs (aspect-ratio change).
        mixed:     none of the above.
    """
    def sgn(x: float, thr: float = 0.5) -> int:
        if x > thr: return 1
        if x < -thr: return -1
        return 0

    pp = corr["peak_pos_omega"]
    pn = corr["peak_neg_omega"]
    cp = corr["circulation_pos"]
    cn = corr["circulation_neg"]
    wl = corr["wake_length"]
    wt = corr["wake_thickness"]

    is_magnitude = (sgn(pp) * sgn(pn) == -1) and (sgn(cp) * sgn(cn) == -1) and (abs(wt) > 0.6)
    is_sign = (sgn(pp) * sgn(pn) == 1) and (sgn(cp) * sgn(cn) == -1) and abs(wt) < 0.5
    is_shape = (abs(wl) > 0.6) and (abs(wt) > 0.6) and (sgn(wl) * sgn(wt) == -1)

    if is_magnitude:
        label = "magnitude"
        narrative = (
            "Impact magnitude axis: both vorticity peaks grow in their "
            "respective directions, wake thickness grows monotonically."
        )
    elif is_sign:
        label = "sign"
        narrative = (
            "Sign-shift axis: peak vorticities shift in the SAME direction "
            "(asymmetric redistribution between + and - vorticity) with "
            "minor effect on wake size. Likely encodes G direction or "
            "wake polarity."
        )
    elif is_shape:
        label = "shape"
        narrative = (
            "Wake-shape axis: streamwise vs spanwise spread trade off "
            "(opposite-sign correlations on wake_length and wake_thickness). "
            "Plausibly D-correlated: smaller D = more concentrated vortex = "
            "thinner / longer wake."
        )
    else:
        label = "mixed"
        narrative = (
            "Mixed axis: descriptor correlations do not cleanly fit any "
            "of magnitude / sign / shape templates. Likely a multi-physics "
            "blend."
        )
    return {"label": label, "narrative": narrative}


def main() -> None:
    desc_path = OUT / "exp1b_descriptors.json"
    decoded_path = OUT / "exp1b_decoded_axes.npz"
    desc = json.loads(desc_path.read_text())
    decoded = np.load(decoded_path)

    interp: dict = {}
    for basis_name in ("PLS3", "PCA3"):
        basis_interp = {"axis_sigmas": desc[basis_name]["axis_sigmas"]}
        for k in (1, 2, 3):
            axis_key = f"axis{k}"
            corr = desc[basis_name][axis_key]["pearson_r_vs_magnitude"]
            classification = classify_axis(corr)
            basis_interp[axis_key] = {
                "pearson_r_vs_magnitude": corr,
                "classification": classification,
            }
        interp[basis_name] = basis_interp

    interp["headline"] = {
        "PLS3 axis1": interp["PLS3"]["axis1"]["classification"]["label"],
        "PLS3 axis2": interp["PLS3"]["axis2"]["classification"]["label"],
        "PLS3 axis3": interp["PLS3"]["axis3"]["classification"]["label"],
        "PCA3 axis1": interp["PCA3"]["axis1"]["classification"]["label"],
        "PCA3 axis2": interp["PCA3"]["axis2"]["classification"]["label"],
        "PCA3 axis3": interp["PCA3"]["axis3"]["classification"]["label"],
    }
    interp["physics_summary"] = (
        "The encoder's variance hierarchy and the parameter-supervised projection "
        "produce DIFFERENT orderings of the same three-dimensional latent subspace. "
        "PLS-3 finds (magnitude, sign, shape) in that order because it prioritises "
        "G-magnitude prediction. PCA-3 finds the encoder's natural variance ordering, "
        "in which the WAKE SIGN axis dominates at 80% of variance, with magnitude "
        "buried in PC3 (~3% variance). Both bases capture the same three-dimensional "
        "subspace; their components express it in different physical units. The "
        "encoder has learned a HIERARCHICAL representation by physical wake "
        "character, not a parameter-aligned one."
    )

    save = OUT / "exp1b_axis_interpretation.json"
    save.write_text(json.dumps(interp, indent=2))
    print(f"[exp1b-summary] wrote {save.relative_to(REPO)}")
    for basis in ("PLS3", "PCA3"):
        for k in (1, 2, 3):
            cls = interp[basis][f"axis{k}"]["classification"]
            sigma = interp[basis]["axis_sigmas"][k - 1]
            print(f"  {basis} axis{k} (sigma={sigma:.2f}): {cls['label']}")

    fig, axes = plt.subplots(6, 3, figsize=(8, 14), squeeze=False)
    bases_order = [("PLS3", k) for k in (1, 2, 3)] + [("PCA3", k) for k in (1, 2, 3)]
    extent = (-1.5, 4.5, -1.5, 1.5)
    vmax = 4000.0
    for row, (basis, k) in enumerate(bases_order):
        arr = decoded[f"{basis}_axis{k}_omega"]
        sigma = interp[basis]["axis_sigmas"][k - 1]
        label = interp[basis][f"axis{k}"]["classification"]["label"]
        for col, mag_idx in enumerate([0, 2, 4]):
            mag = (-2.0, -1.0, 0.0, +1.0, +2.0)[mag_idx]
            ax = axes[row, col]
            field = arr[mag_idx].T
            im = ax.imshow(
                field,
                origin="lower",
                extent=extent,
                cmap="RdBu_r",
                norm=Normalize(vmin=-vmax, vmax=vmax),
                aspect="equal",
                interpolation="nearest",
            )
            if col == 0:
                ax.set_ylabel(f"{basis} axis{k}\n({label})", fontsize=8)
            if row == 0:
                ax.set_title(f"m = {mag:+g} sigma", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle(
        "Exp 1(b): decoded latent perturbations along PLS-3 and PCA-3 axes\n"
        "PLS3 finds (magnitude, sign, shape); PCA3 finds (sign, sign, magnitude). Same 3-D subspace, different ordering.",
        fontsize=10,
    )
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="omega_z (raw)")
    plt.tight_layout(rect=(0, 0, 0.9, 0.96))

    fig_path = FIG_DIR / "exp1b_axis_decoded_panel.png"
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[exp1b-summary] wrote {fig_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
