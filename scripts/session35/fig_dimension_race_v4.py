"""F12: Dimension race with the probe-dilution control (fig_dimension_race_v4).

Three panels, every number loaded from JSON at build time:
  (a) peak-region pooled C_L R2 (frozen linear probe) vs latent dimension d for
      the three lineages cln_rexpred / flagship_clw / fukami_wake
      (outputs/session34/lift_dimension_ladder.json; d=4 full-precision seed
      bands from outputs/session34/lowd_d4_seedband.json; d=32 cln_rexpred
      3-seed band from outputs/session35/rexpred_d32_band.json, replacing the
      old single seed). Shaded min-max bands where 3 seeds exist; open markers
      denote single-seed points.
  (b) probe-dilution control on the CLW lineage
      (outputs/session34/probe_dilution_test.json): MLP probe R2 is d-invariant
      (~0.88-0.90) while the best-4-coordinate linear probe drops to 0.55-0.66
      at d >= 8: lift information does not dilute, linear accessibility does.
  (c) decoded SSIM vs d for cln_rexpred (full frame and near-body,
      outputs/session34/cln_rexpred_ssim_ladder.json).

Split: test_b throughout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "session21"))
import figstyle  # noqa: E402

LADDER = REPO / "outputs/session34/lift_dimension_ladder.json"
LOWD = REPO / "outputs/session34/lowd_d4_seedband.json"
DILUTION = REPO / "outputs/session34/probe_dilution_test.json"
BAND35 = REPO / "outputs/session35/rexpred_d32_band.json"
SSIM_LADDER = REPO / "outputs/session34/cln_rexpred_ssim_ladder.json"

OUT_DIR = REPO / "paper/sections/figures/results"
OUT_PDF = OUT_DIR / "fig_dimension_race_v4.pdf"
OUT_PNG = OUT_DIR / "fig_dimension_race_v4.png"

DS = [4, 8, 16, 32]


def lighten(color: str, frac: float):
    rgb = np.array(mpl.colors.to_rgb(color))
    return tuple(rgb + (1.0 - rgb) * frac)


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def seeds_for(ladder: dict, lowd: dict, band35: dict, lineage: str, d: int) -> list[float]:
    """Per-seed linear peak R2 list for one (lineage, d), with the documented
    substitutions: full-precision d=4 bands from lowd_d4_seedband.json and the
    session35 3-seed d=32 band for cln_rexpred (replacing the old single seed)."""
    entry = ladder[lineage].get(f"d{d}")
    if entry is None:
        return []
    lin = list(entry["lin"])
    if d == 4:
        lowd_key = {"flagship_clw": "jepa_clw_d4", "fukami_wake": "fukami_wake_d4"}.get(lineage)
        if lowd_key is not None:
            full = sorted(lowd[lowd_key])
            # consistency check: the ladder holds the rounded copies of these seeds
            assert [round(v, 3) for v in full] == sorted(round(v, 3) for v in lin), (
                lineage, d, full, lin)
            lin = full
    if lineage == "cln_rexpred" and d == 32:
        band = list(band35["band"]["linear_peak_r2_per_seed"])
        # s0 of the band reproduces the ladder's old single seed (0.903)
        assert any(abs(round(v, 3) - lin[0]) < 5e-4 for v in band), (band, lin)
        lin = band
    return lin


def plot_lineage(ax, ds, seed_lists, color, ls, marker, label, zorder=3):
    ds = [d for d, s in zip(ds, seed_lists) if s]
    seed_lists = [s for s in seed_lists if s]
    means = np.array([np.mean(s) for s in seed_lists])
    mins = np.array([np.min(s) for s in seed_lists])
    maxs = np.array([np.max(s) for s in seed_lists])
    ax.fill_between(ds, mins, maxs, color=color, alpha=0.18, lw=0, zorder=zorder - 1)
    ax.plot(ds, means, color=color, ls=ls, lw=1.0, zorder=zorder, label=label)
    for d, s, m in zip(ds, seed_lists, means):
        banded = len(s) >= 3
        ax.plot([d], [m], marker=marker, ms=4.5, ls="none",
                mfc=color if banded else "white", mec=color, mew=0.9, zorder=zorder + 1)


def main() -> None:
    ladder = load_json(LADDER)
    lowd = load_json(LOWD)
    dilution = load_json(DILUTION)["results"]
    band35 = load_json(BAND35)
    ssim = load_json(SSIM_LADDER)

    green = figstyle.FAMILY_COLOR["jepa"]
    green_lt = lighten(green, 0.45)
    red = figstyle.FAMILY_COLOR["fukami"]

    figstyle.use_style()
    fig, axes = plt.subplots(1, 3, figsize=figstyle.figure_size(1.0, aspect=0.40))
    ax_a, ax_b, ax_c = axes

    # ---------------- (a) linear-probe peak R2 vs d, three lineages ----------
    for lineage, color, ls, marker, label in [
        ("cln_rexpred", green, "-", figstyle.FAMILY_MARKER["jepa"], "lift-focused"),
        ("flagship_clw", green_lt, "--", figstyle.FAMILY_MARKER["jepa"], "wake-supervised"),
        ("fukami_wake", red, "-", figstyle.FAMILY_MARKER["fukami"], "reconstructive (Fukami)"),
    ]:
        seed_lists = [seeds_for(ladder, lowd, band35, lineage, d) for d in DS]
        plot_lineage(ax_a, DS, seed_lists, color, ls, marker, label)

    ax_a.set_ylabel(r"peak $C_L$ $R^2$ (linear probe)")
    ax_a.set_ylim(0.70, 0.965)
    ax_a.legend(loc="lower right", fontsize=5, handlelength=1.2,
                borderaxespad=0.2, labelspacing=0.3)
    ax_a.text(0.03, 0.975, "open markers: single seed\nbands: seed min-max",
              transform=ax_a.transAxes, fontsize=6, color="0.35", va="top")

    # ---------------- (b) probe-dilution control (CLW lineage) ---------------
    dds = sorted(int(k) for k in dilution)
    mlp = np.array([dilution[str(d)]["mlp_test"] for d in dds])
    lin_full = np.array([dilution[str(d)]["lin_test"] for d in dds])
    best4 = np.array([dilution[str(d)]["best4_lin"] for d in dds])

    ax_b.fill_between(dds, best4, mlp, color="0.85", lw=0, zorder=1)
    ax_b.plot(dds, mlp, color=green, ls="-", marker="o", ms=4.0, lw=1.0,
              zorder=3, label="MLP, all $d$ coords")
    ax_b.plot(dds, lin_full, color=green, ls="--", marker="o", ms=4.0, lw=1.0,
              mfc="white", mew=0.9, zorder=3, label="linear, all $d$ coords")
    ax_b.plot(dds, best4, color="0.35", ls=":", marker="v", ms=4.0, lw=1.0,
              mfc="white", mec="0.35", mew=0.9, zorder=3,
              label="linear, best 4 coords")
    mid = len(dds) // 2
    ax_b.annotate("linear\naccessibility\ngap",
                  xy=(dds[mid], 0.5 * (mlp[mid] + best4[mid])),
                  ha="center", va="center", fontsize=6, color="0.3")
    ax_b.set_ylabel(r"full-trace $C_L$ $R^2$")
    ax_b.set_ylim(0.5, 0.965)
    ax_b.legend(loc="lower left", fontsize=6, handlelength=1.5,
                borderaxespad=0.2, labelspacing=0.3)
    ax_b.text(0.97, 0.975, "wake-supervised lineage", transform=ax_b.transAxes,
              fontsize=6, color="0.35", ha="right", va="top")

    # ---------------- (c) decoded SSIM vs d, cln_rexpred ---------------------
    sds = sorted(int(k.lstrip("d")) for k in ssim)
    for mask, ls, mfc_shift, label in [("full", "-", 0.0, "full frame"),
                                       ("nearbody", "--", 0.45, "near-body")]:
        lists = [ssim[f"d{d}"][mask] for d in sds]
        color = lighten(green, mfc_shift)
        plot_lineage(ax_c, sds, lists, color, ls, "o", label)
    ax_c.set_ylabel("decoded SSIM")
    ax_c.set_ylim(0.50, 0.82)
    ax_c.legend(loc="lower right", fontsize=6, handlelength=1.7,
                borderaxespad=0.2)
    ax_c.text(0.03, 0.97, "lift-focused lineage", transform=ax_c.transAxes,
              fontsize=6, color="0.35", va="top")

    for ax, lab in zip(axes, "abc"):
        ax.set_xscale("log", base=2)
        ax.set_xticks(DS)
        ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.minorticks_off()
        ax.set_xlabel(r"latent dimension $d$")
        ax.text(-0.14, 1.04, f"({lab})", transform=ax.transAxes,
                fontsize=8.5, fontweight="bold", va="bottom")

    fig.text(0.995, 0.01, "in-distribution test set", ha="right", va="bottom",
             fontsize=6.2, color="0.35")
    fig.tight_layout(w_pad=1.2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=200)
    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
