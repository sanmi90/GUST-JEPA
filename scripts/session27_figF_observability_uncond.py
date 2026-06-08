"""UNCONDITIONED pressure observability (Session 27 copy of figF_observability.py).

Latent-recoverability vs the physical estimate, for the fully-unconditioned
(no-c) JEPA d=64 latent only (the production cross-family panel compared JEPA /
recon / POD; here we have a single unconditioned representation, so the figure is
the unconditioned JEPA series with the family-independent direct pressure->C_L
baseline for context).

(a) State (latent) recovery R2 versus sensor count K under TCSI placement.
(b) Impact C_L mean absolute error routed through the recovered latent, with the
    direct pressure->C_L baseline.

Input  : outputs/session27/pressure_noc_tf/pressure_obs_v2.csv
Output : outputs/session27/figs_uncond/figF_observability.pdf (+ .png)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "session21"))
import figstyle as fs  # noqa: E402

CSV = REPO / "outputs/session27/pressure_noc_tf/pressure_obs_v2.csv"
OUT_PDF = REPO / "outputs/session27/figs_uncond/figF_observability.pdf"
OUT_PNG = REPO / "outputs/session27/figs_uncond/figF_observability.png"

KS = [2, 4, 8, 16]
TAG = "jepa_d64"


def main() -> None:
    fs.use_style()
    rows = [r for r in csv.DictReader(open(CSV)) if r["split"] == "test_b"]

    def series(col):
        d = {int(r["K"]): float(r[col]) for r in rows if r["tag"] == TAG}
        return [d[k] for k in KS]

    fig, (axa, axb) = plt.subplots(1, 2, figsize=fs.figure_size(1.0, aspect=0.46))

    c = fs.FAMILY_COLOR["jepa"]
    mk = fs.FAMILY_MARKER["jepa"]
    axa.plot(KS, series("R2_z"), marker=mk, ms=3.6, lw=1.1, color=c,
             label=r"JEPA $d64$ (no-c)")
    axb.plot(KS, series("cl_mae_via"), marker=mk, ms=3.6, lw=1.1, color=c,
             label=r"via latent")
    # direct pressure -> C_L baseline (representation-independent)
    axb.plot(KS, series("cl_mae_direct"), color="0.15", lw=1.4, ls=(0, (4, 2)),
             zorder=6, label="direct (no latent)")

    for ax in (axa, axb):
        ax.set_xscale("log", base=2); ax.set_xticks(KS)
        ax.set_xticklabels([str(k) for k in KS])
        ax.set_xlabel("sensors $K$")
    axa.set_ylabel("state recovery $R^2$")
    axa.set_title("(a) latent recoverability", fontsize=8)
    axa.axhline(0, color="0.85", lw=0.6, zorder=0)
    axa.legend(loc="lower right", fontsize=6.5, handletextpad=0.4, borderpad=0.2)
    axb.set_ylabel(r"impact $C_L$ error (MAE)")
    axb.set_title("(b) physical estimate", fontsize=8)
    axb.legend(loc="best", fontsize=6.5, handletextpad=0.4, borderpad=0.2)

    fig.tight_layout()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF); fig.savefig(OUT_PNG, dpi=200)
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
