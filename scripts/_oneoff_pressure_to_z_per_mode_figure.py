"""Replace Figure 5 (pressure-based state estimation diagnostic) with a clean,
high-readability version that actually matches the caption in Section 4.5.

The previous figure at outputs/session18/exp_b1_test3/baseline_pressure_observability_figure.png
showed the z->c (gust parameter recovery from latent) result, NOT the
per-POD-mode pressure -> z_impact recoverability that Section 4.5 claims.

This script produces the right figure: per-POD-mode R^2 of the KRR-RBF
pressure -> z_impact regressor on test_b, with the rank correlation against
mode-by-mode Q-overlap.

Source: outputs/session18/exp_b1_test3/pod_q_overlap_pressure.json
Output: paper/sections/figures/results/figS_pressure_to_z.png and .pdf
"""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

REPO = Path("/home/carlos/GUST-JEPA")
SRC = REPO / "outputs/session18/exp_b1_test3/pod_q_overlap_pressure.json"
OUT_PNG = REPO / "paper/sections/figures/results/figS_pressure_to_z.png"
OUT_PDF = REPO / "paper/sections/figures/results/figS_pressure_to_z.pdf"


def main():
    d = json.load(open(SRC))
    r2 = np.asarray(d["r2_per_mode"], dtype=float)
    q_hard = np.asarray(d["q_overlap_hard"], dtype=float)
    n_modes = len(r2)
    pearson_r = d["pearson_hard"]["r"]
    pearson_p = d["pearson_hard"]["p"]
    spear_r = d["spearman_hard"]["r"]
    spear_p = d["spearman_hard"]["p"]
    K = int(d["K_sensors"])
    n_enc = int(d["n_test_b_encounters"])

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "axes.linewidth": 1.1,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "legend.fontsize": 12,
        "figure.titlesize": 16,
    })

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.0), constrained_layout=True)

    # LEFT: per-mode R^2 bar chart
    ax0 = axes[0]
    x = np.arange(n_modes)
    colors = ["#238b45" if v > 0.3 else ("#fdae61" if v > 0 else "#cb181d") for v in r2]
    ax0.bar(x, r2, color=colors, edgecolor="black", linewidth=0.6)
    ax0.axhline(0, color="black", linewidth=0.8)
    ax0.set_xlabel("POD mode index")
    ax0.set_ylabel(r"Pressure $\to$ $\mathbf{z}_{\mathrm{impact}}$ recovery $R^{2}$ on test\_b")
    ax0.set_title(
        rf"Per-mode recoverability ($K = {K}$ sensors, $n = {n_enc}$ enc.)",
        fontweight="bold")
    ax0.set_xticks(x)
    ax0.set_xticklabels([str(i) for i in x], fontsize=11)
    ax0.grid(True, axis="y", alpha=0.3)
    ax0.set_ylim(min(r2.min() - 0.1, -0.1), max(r2.max() + 0.1, 1.0))

    # RIGHT: scatter Q-overlap vs R^2 with regression line + Spearman/Pearson stats
    ax1 = axes[1]
    ax1.scatter(q_hard, r2, s=90, c="#08519c", edgecolor="black",
                linewidth=0.8, zorder=3)
    # Linear fit for visual guide
    slope, intercept = np.polyfit(q_hard, r2, 1)
    xx = np.linspace(q_hard.min(), q_hard.max(), 50)
    ax1.plot(xx, slope * xx + intercept, "--", color="#cb181d",
             linewidth=1.5, zorder=2, label="linear fit")
    # Annotate each point with its mode index
    for i, (qx, qy) in enumerate(zip(q_hard, r2)):
        ax1.annotate(str(i), (qx, qy), textcoords="offset points",
                     xytext=(5, 4), fontsize=9, color="black", alpha=0.75)

    ax1.set_xlabel("POD mode Q-criterion overlap (hard indicator)")
    ax1.set_ylabel(r"Pressure $\to$ $\mathbf{z}_{\mathrm{impact}}$ recovery $R^{2}$")
    ax1.set_title("Q-overlap mechanism", fontweight="bold")
    ax1.grid(True, alpha=0.3)
    txt = (rf"Spearman $\rho = {spear_r:+.3f}$, $p = {spear_p:.3f}$" + "\n"
           rf"Pearson $r = {pearson_r:+.3f}$, $p = {pearson_p:.3f}$")
    ax1.text(0.05, 0.95, txt, transform=ax1.transAxes,
             fontsize=12, verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                       edgecolor="black", alpha=0.9))
    ax1.legend(loc="lower right", fontsize=12, framealpha=0.95)

    fig.suptitle(
        r"Pressure-based state estimation: per-POD-mode recoverability and Q-criterion mechanism",
        fontweight="bold")
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
