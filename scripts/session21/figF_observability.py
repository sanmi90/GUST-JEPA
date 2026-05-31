"""Cross-family pressure observability on v2, and why R2 is not enough.

(a) Latent (state) recovery R2 versus sensor count for every family under the
    TCSI optimal placement: the JEPA state is the most recoverable at K>=4.
(b) The physical estimate: impact C_L mean absolute error (in C_L units) routed
    through each family's recovered latent, with the direct pressure->C_L baseline.
    The contrast with (a) is the point: the reconstructive d=3 latent is the
    easiest to recover (highest R2 at low K) yet gives the worst C_L estimate, and
    C_L is read most accurately straight off the pressure. Recoverability and
    quantitative quality are different axes.
Source: outputs/session21/pressure_v2/pressure_obs_v2.csv.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import figstyle as fs

REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "outputs/session21/pressure_v2/pressure_obs_v2.csv"
OUT_PDF = REPO / "paper/sections/figures/results/figF_observability.pdf"
OUT_PNG = REPO / "outputs/session21/figs/figF_observability.png"

# (tag, kind, d, label) in legend order
FAMS = [("jepa_d64", "jepa", 64, r"JEPA $d64$"), ("jepa_d32", "jepa", 32, r"JEPA $d32$"),
        ("fukami_d3", "fukami", 3, r"recon.\ $d3$"), ("fukami_d32", "fukami", 32, r"recon.\ $d32$"),
        ("fukami_d64", "fukami", 64, r"recon.\ $d64$"), ("pod_d16", "pod", 16, r"POD $d16$"),
        ("pod_d32", "pod", 32, r"POD $d32$"), ("pod_d64", "pod", 64, r"POD $d64$")]
KS = [2, 4, 8, 16]


def shade(kind, d):
    ds = sorted({dd for _, k, dd, _ in FAMS if k == kind})
    base = np.array(plt.matplotlib.colors.to_rgb(fs.FAMILY_COLOR[kind]))
    frac = ds.index(d) / max(1, len(ds) - 1)
    light = 0.5 - 0.5 * frac
    return tuple(base + (1 - base) * light)


def main() -> None:
    fs.use_style()
    rows = [r for r in csv.DictReader(open(CSV)) if r["split"] == "test_b"]
    def series(tag, col):
        d = {int(r["K"]): float(r[col]) for r in rows if r["tag"] == tag}
        return [d[k] for k in KS]

    fig, (axa, axb) = plt.subplots(1, 2, figsize=fs.figure_size(1.0, aspect=0.46))

    for tag, kind, d, lab in FAMS:
        c = shade(kind, d)
        mk = fs.FAMILY_MARKER[kind]
        axa.plot(KS, series(tag, "R2_z"), marker=mk, ms=3.2, lw=1.0, color=c, label=lab)
        axb.plot(KS, series(tag, "cl_mae_via"), marker=mk, ms=3.2, lw=1.0, color=c)
    # direct pressure -> C_L baseline (family-independent)
    direct = series("jepa_d64", "cl_mae_direct")
    axb.plot(KS, direct, color="0.15", lw=1.4, ls=(0, (4, 2)), zorder=6,
             label="direct (no latent)")

    for ax in (axa, axb):
        ax.set_xscale("log", base=2); ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS])
        ax.set_xlabel("sensors $K$")
    axa.set_ylabel("state recovery $R^2$"); axa.set_title("(a) latent recoverability", fontsize=8)
    axa.axhline(0, color="0.85", lw=0.6, zorder=0)
    axb.set_ylabel(r"impact $C_L$ error (MAE)"); axb.set_title("(b) physical estimate", fontsize=8)
    axa.legend(loc="lower center", ncol=2, fontsize=5.6, handletextpad=0.3,
               columnspacing=0.8, borderpad=0.2)
    axb.legend(loc="upper left", fontsize=6, handletextpad=0.4, borderpad=0.2)

    fig.tight_layout()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF); fig.savefig(OUT_PNG, dpi=200)
    print(f"wrote {OUT_PDF.name}")


if __name__ == "__main__":
    main()
