"""Error-maps figure (fig_error_maps_v2p1.png): per-encounter paired improvement
in held-out wake-enstrophy error against the gust axes. v2.1.

Matches the v2.1 caption (fig:error_maps, section_4_results.tex): per-encounter
paired improvement delta_e = e_AE - e_JEPA in held-out wake-enstrophy error against
the gust strength G, core diameter D, wall-normal offset Y, and the baseline
shedding phase at impact phi (test_b, n = 42); positive delta_e means the predictive
latent carries the smaller error. Each panel shows the per-encounter values, a
locally weighted (LOWESS) trend, and a bootstrap band over encounters.

The paired errors are the REPRESENTATIONAL (encoded-latent) wake-enstrophy probe
errors at H = 16 on test_b, read from the closure_matrix per-encounter dump
(per-encounter y_pred/y_true for the predictive jepa_tf_noc and the reconstructive
fukami families, headline seed 42), so they are the same numbers as the closure
matrix. Error metric is the absolute residual |y_pred - y_true| per encounter (a
per-encounter location, the units the LOWESS trend is read in). The gust parameters
and the impact shedding phase phi come from per_encounter_physics.npz, joined on
(case_id, encounter_index).

The Spearman trend of the advantage against each axis is printed (these are the
error-map trend numbers the macro audit wants); they are measured on v2.1 and may
differ from the v2 literals still in the text, which is reported.

CPU only; numpy/scipy/statsmodels. Read-only; writes only the PNG (the caption
\\includegraphics is fig_error_maps.png).

Output: paper/sections/figures/results/fig_error_maps_v2p1.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from statsmodels.nonparametric.smoothers_lowess import lowess  # noqa: E402

import _common as cm  # noqa: E402

fs = cm.fs

DUMP = cm.SESSION28 / "closure_matrix" / "per_encounter" / "dump.npz"
PHYS = cm.SESSION28 / "physics" / "per_encounter_physics.npz"

JEPA_KEY = "jepa_tf_noc_d64_s42|wake_enstrophy|representational|ridge|test_b|pooled|H16"
FUKAMI_KEY = "fukami_d64_s42|wake_enstrophy|representational|ridge|test_b|pooled|H16"
AXES = [
    ("G", r"gust strength $G$"),
    ("D", r"core diameter $D$"),
    ("Y", r"wall-normal offset $Y$"),
    ("phi", r"impact shedding phase $\phi$"),
]
N_BOOT = 2000


def paired_advantage() -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Per-encounter advantage delta_e = |e_fukami| - |e_jepa| + the (cid, enc) keys."""
    d = np.load(DUMP, allow_pickle=True)
    jp, jt = d[f"{JEPA_KEY}__y_pred"], d[f"{JEPA_KEY}__y_true"]
    jc = np.array([str(c) for c in d[f"{JEPA_KEY}__case_id"]])
    je = d[f"{JEPA_KEY}__encounter_index"].astype(int)
    fp, ft = d[f"{FUKAMI_KEY}__y_pred"], d[f"{FUKAMI_KEY}__y_true"]
    fc = np.array([str(c) for c in d[f"{FUKAMI_KEY}__case_id"]])
    fe = d[f"{FUKAMI_KEY}__encounter_index"].astype(int)
    fkey = {(c, int(e)): i for i, (c, e) in enumerate(zip(fc, fe))}
    order = np.array([fkey[(c, int(e))] for c, e in zip(jc, je)])
    e_j = np.abs(jp - jt)
    e_f = np.abs(fp[order] - ft[order])
    return {"delta": e_f - e_j, "case_id": jc, "enc": je}, e_f - e_j


def gust_axes(cid: np.ndarray, enc: np.ndarray) -> dict[str, np.ndarray]:
    """G, D, Y, phi per encounter from per_encounter_physics.npz, joined on (cid, enc)."""
    p = np.load(PHYS, allow_pickle=True)
    pc = np.array([str(c) for c in p["case_id"]])
    pe = p["encounter_index"].astype(int)
    pkey = {(c, int(e)): i for i, (c, e) in enumerate(zip(pc, pe))}
    idx = np.array([pkey[(c, int(e))] for c, e in zip(cid, enc)])
    return {
        "G": p["G"][idx].astype(float),
        "D": p["D"][idx].astype(float),
        "Y": p["Y"][idx].astype(float),
        "phi": p["phi_imp"][idx].astype(float),
    }


def boot_band(x: np.ndarray, y: np.ndarray, grid: np.ndarray, frac: float = 0.7):
    """LOWESS on the data + a bootstrap-over-encounters band evaluated on ``grid``."""
    rng = np.random.default_rng(0)
    base = lowess(y, x, frac=frac, return_sorted=True)
    fit = np.interp(grid, base[:, 0], base[:, 1])
    boots = np.empty((N_BOOT, grid.size))
    n = x.size
    for b in range(N_BOOT):
        s = rng.integers(0, n, n)
        lo = lowess(y[s], x[s], frac=frac, return_sorted=True)
        boots[b] = np.interp(grid, lo[:, 0], lo[:, 1])
    return fit, np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)


def main() -> None:
    fs.use_style()
    adv, delta = paired_advantage()
    ax_vals = gust_axes(adv["case_id"], adv["enc"])

    fig, axes = plt.subplots(1, 4, figsize=fs.figure_size(1.0, aspect=0.30), layout="constrained")
    col = fs.FAMILY_COLOR["jepa"]
    trends = {}
    for k, (key, lab) in enumerate(AXES):
        ax = axes[k]
        x = ax_vals[key]
        ax.axhline(0, color="0.6", lw=0.7, zorder=0)
        ax.scatter(x, delta, s=10, color=col, alpha=0.55, zorder=2, edgecolors="none")
        grid = np.linspace(x.min(), x.max(), 60)
        fit, lo, hi = boot_band(x, delta, grid)
        ax.fill_between(grid, lo, hi, color=col, alpha=0.15, lw=0, zorder=1)
        ax.plot(grid, fit, color=col, lw=1.3, zorder=3)
        rho, p = spearmanr(x, delta)
        trends[key] = (rho, p)
        ax.set_xlabel(lab)
        if k == 0:
            ax.set_ylabel(r"advantage $\Delta e = e_{\mathrm{AE}} - e_{\mathrm{JEPA}}$", fontsize=7)
        ax.text(
            0.04,
            0.96,
            rf"$\rho = {rho:+.2f}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.5,
            color="0.25",
        )

    out = cm.out_path("fig_error_maps").with_suffix(".png")
    fig.savefig(out, dpi=300)
    print(f"wrote {out}")
    print(
        "  v2.1 representational (encoded) wake-enstrophy advantage Spearman trends "
        "(n=42, test_b, fukami seed 42):"
    )
    for key, (rho, p) in trends.items():
        print(f"    advantage vs {key}: rho = {rho:+.2f} (p = {p:.3f})")
    print(
        "  NOTE: the text literals (D rho=0.42, Y rho=0.49) are v2-era; the v2.1 "
        "values above should replace them."
    )


if __name__ == "__main__":
    main()
