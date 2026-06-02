"""Figures for the EDRFCM2026 two-page abstract (two-column, double-col width
169.7 mm = 6.68 in; single-col 3.15 in). Reuses the paper vorticity convention.

(1) triptych.pdf  : double-column, mid-plane vorticity 16 frames after impact,
    simulation vs predictive (JEPA) and reconstructive (AE) decode, strong gust.
(2) closure_bars.pdf : single-column, forecast wake-enstrophy closure R^2 (H=16)
    for the three encoder families; only the predictive latent clears the floor.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/carlos/GUST-JEPA")
sys.path.insert(0, str(REPO / "scripts" / "session21"))
import figstyle as fs  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

OUT = REPO / "paper/edrfcm2026/Figures"
CASE = "G-3.00_D1.50_Y-0.10"
WAKE_R2 = {"jepa": 0.449, "fukami": -0.478, "pod": -0.089}


def main() -> None:
    fs.use_style()
    OUT.mkdir(parents=True, exist_ok=True)
    dec = np.load(REPO / "outputs/session20/decoded/test_b.npz", allow_pickle=True)
    cid = np.array([str(c) for c in dec["case_ids"]])
    offs = list(dec["offsets"])
    gi = int(np.where(cid == CASE)[0][0])
    oi = offs.index(16)

    # (1) double-column triptych, wide and short
    fig, axes = plt.subplots(1, 3, figsize=(6.68, 1.40))
    for ax, (key, title, fam) in zip(axes, [
            ("target_norm", "simulation", "oracle"),
            ("jepa_norm", "predictive (JEPA)", "jepa"),
            ("fukami_norm", "reconstructive (AE)", "fukami")]):
        fs.vort_panel(ax, dec[key][gi, oi])
        ax.set_title(title, fontsize=8.5, color=fs.FAMILY_COLOR[fam], pad=2)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.86, bottom=0.02, wspace=0.04)
    fig.savefig(OUT / "triptych.pdf")
    plt.close(fig)

    # (2) single-column wall-pressure recovery K-sweep (matched d=64, test_b)
    import csv
    rows = list(csv.DictReader(open(REPO / "outputs/session21/pressure_v2/pressure_obs_v2.csv")))
    Ks = [2, 4, 8, 16]
    fig, ax = plt.subplots(figsize=(3.15, 2.15))
    for fam, tag in [("jepa", "jepa_d64"), ("fukami", "fukami_d64"), ("pod", "pod_d64")]:
        v = {int(r["K"]): float(r["R2_z"]) for r in rows
             if r["tag"] == tag and r["split"] == "test_b"}
        ax.plot(Ks, [v[k] for k in Ks], marker=fs.FAMILY_MARKER[fam],
                color=fs.FAMILY_COLOR[fam], lw=1.4, ms=4.5, label=fs.FAMILY_LABEL[fam])
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ks)
    ax.set_xticklabels(Ks)
    ax.set_xlabel("number of wall-pressure sensors $K$", fontsize=8)
    ax.set_ylabel(r"state recovery $R^2$ (held out)", fontsize=8)
    ax.set_ylim(0.0, 1.0)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.legend(fontsize=6.8, loc="lower left", frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "pressure_recovery.pdf")
    plt.close(fig)
    print("wrote", OUT / "triptych.pdf", "and", OUT / "pressure_recovery.pdf")


if __name__ == "__main__":
    main()
