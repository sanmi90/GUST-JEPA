"""Drift figure (fig_horizon_sweep_v2p1.pdf): relative drift of the predictive
rollout on test_b versus forecast horizon H, v2.1.

Matches the v2.1 caption (fig:drift_uncond, section_4_results.tex): the deviation
of the rolled latent from the simulation-encoded latent normalised by the encoded
latent scale,

    || z_rollout - z_DNS || / || z_DNS ||,

versus the forecast horizon H, for the unconditioned transformer predictor
(tf-no-c). The relative deviation is bounded and grows gradually; it understates
the family separation, which is the Mahalanobis manifold-departure of
table~\\ref{tab:latent_drift}.

DATA NOTE (reported, not hidden): outputs/session28/drift/ce1_results.npz carries
the per-encounter rel-l2-vs-H array for tf-no-c (key ``jepa_d64__rel_l2``, the
full-context rollout) but NOT for the recurrent lstm-no-c variant: the LSTM has no
matched-predictor rollout in this campaign (drift/README.md: "matched-predictor
rollout absent in Phase B"; table8.json rel_l2_H24 = null), so only its encoded
Mahalanobis is available. This figure therefore plots the tf-no-c relative drift
that exists; the caption's lstm-no-c curve cannot be drawn from the available data
and is flagged in the regeneration report.

CPU only; numpy. Read-only on the drift NPZ; writes only the PDF.

Output: paper/sections/figures/results/fig_horizon_sweep_v2p1.pdf
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import _common as cm  # noqa: E402

fs = cm.fs
DRIFT_NPZ = cm.SESSION28 / "drift" / "ce1_results.npz"


def main() -> None:
    fs.use_style()
    blob = np.load(DRIFT_NPZ, allow_pickle=True)
    horizons = blob["horizons"].astype(float)
    rel = blob["jepa_d64__rel_l2"].astype(float)  # (n_enc, n_H), full-context rollout
    mean = np.nanmean(rel, axis=0)
    q25 = np.nanpercentile(rel, 25, axis=0)
    q75 = np.nanpercentile(rel, 75, axis=0)

    fig, ax = plt.subplots(figsize=fs.figure_size(0.7, aspect=0.72))
    col = fs.FAMILY_COLOR["jepa"]
    ax.fill_between(
        horizons, q25, q75, color=col, alpha=0.18, lw=0, label="tf-no-c interquartile range"
    )
    ax.plot(
        horizons,
        mean,
        color=col,
        marker=fs.FAMILY_MARKER["jepa"],
        ms=4.5,
        lw=1.4,
        label="tf-no-c (transformer)",
    )
    ax.set_xlabel("forecast horizon $H$ (frames after impact)")
    ax.set_ylabel(
        r"relative drift "
        r"$\|\widehat{z}_{\mathrm{roll}}-z_{\mathrm{DNS}}\|/"
        r"\|z_{\mathrm{DNS}}\|$"
    )
    ax.set_ylim(0, max(q75) * 1.12)
    ax.set_xticks(horizons)
    ax.legend(loc="upper left", handlelength=1.4, handletextpad=0.5, borderpad=0.3)
    ax.grid(alpha=0.22)

    out = cm.out_path("fig_horizon_sweep")
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    print(
        "  tf-no-c mean rel-l2 per H "
        + ", ".join(f"H{int(h)}:{m:.2f}" for h, m in zip(horizons, mean))
    )
    print(
        "  WARNING: lstm-no-c rel-l2 is ABSENT in ce1_results.npz "
        "(no matched-predictor rollout in Phase B); only tf-no-c is plotted."
    )


if __name__ == "__main__":
    main()
