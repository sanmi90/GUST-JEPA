"""SPEC 5 - compact persistent-homology figure (de-clutter of the old Fig 6).

One idea: the predictive encoding is a single clean loop; the reconstructive
encoding fragments (median 1 vs ~4 significant H1 generators, p=4.4e-8). Two
representative H1 persistence diagrams + the generator-count histogram. The
analysis paragraph that was baked into the old figure moves to the caption; the
confounded H1-lifetime-vs-horizon panel is dropped (one sentence in the text).

Diagrams and counts are recomputed from the simulation-encoded latents with the
verified Session 20 Vietoris-Rips machinery (ripser, same noise floor).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import figstyle as fs

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "session20"))
from exp_persistent_homology import (  # noqa: E402
    FUKAMI_TAG, JEPA_TAG, NOISE_FRAC, ROLLOUTS_ROOT, loop_summary, rips_h1,
)

OUT_PDF = REPO / "paper/sections/figures/results/fig_persistence.pdf"
OUT_PNG = REPO / "outputs/session21/figs/fig_persistence.png"
PVAL = 4.4e-8   # Mann-Whitney one-sided, from persistent_homology.json


def counts(tag: str):
    z = np.load(ROLLOUTS_ROOT / f"rollouts_{tag}" / "test_b.npz",
                allow_pickle=True)["z_dns"].astype(np.float64)
    summ = [loop_summary(z[i]) for i in range(len(z))]
    nsig = np.array([s["n_significant"] for s in summ])
    prom = np.array([s["max_lifetime_rel"] for s in summ])
    return z, nsig, prom


def diagram(ax, z, i, color, title):
    """Plot the H1 diagram of encounter `i`."""
    _, h1, scale = rips_h1(z[i])
    floor = NOISE_FRAC * scale
    b, d = h1[:, 0], h1[:, 1]
    sig = (d - b) > floor
    lim = max(d.max(), scale) * 1.05
    ax.plot([0, lim], [0, lim], ls="--", lw=0.7, color="0.7", zorder=1)
    ax.plot([0, lim - floor], [floor, lim], ls=":", lw=0.7, color="0.5", zorder=1)
    ax.scatter(b[~sig], d[~sig], s=8, color="0.7", zorder=2)
    ax.scatter(b[sig], d[sig], s=20, color=color, zorder=3,
               edgecolors="white", linewidths=0.4)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("birth"); ax.set_ylabel("death")
    ax.set_title(title, fontsize=7.5)
    ax.text(0.04, 0.95, f"{int(sig.sum())} significant\n$H_1$ generators",
            transform=ax.transAxes, va="top", fontsize=6, color=color)


def main() -> None:
    fs.use_style()
    zj, nj, pj = counts(JEPA_TAG)
    zf, nf, pf = counts(FUKAMI_TAG)

    # representative encounters: the most prominent loop among the typical count
    ij = int(np.where(nj == 1, pj, -1).argmax())              # clean single loop
    fuk_mask = nf >= 3
    ifk = int(np.where(fuk_mask, pf, -1).argmax())            # fragmented, prominent

    fig, axes = plt.subplots(1, 3, figsize=fs.figure_size(1.0, aspect=0.40),
                             layout="constrained")
    diagram(axes[0], zj, ij, fs.FAMILY_COLOR["jepa"], "predictive (JEPA) encoding")
    diagram(axes[1], zf, ifk, fs.FAMILY_COLOR["fukami"], "reconstructive encoding")

    ax = axes[2]
    mx = int(max(nj.max(), nf.max()))
    bins = np.arange(-0.5, mx + 1.5)
    nj_h, _, _ = ax.hist(nj, bins=bins, color=fs.FAMILY_COLOR["jepa"], alpha=0.7,
                         rwidth=0.9, label=f"predictive (median {int(np.median(nj))})")
    nf_h, _, _ = ax.hist(nf, bins=bins, color=fs.FAMILY_COLOR["fukami"], alpha=0.55,
                         rwidth=0.9, label=f"reconstructive (median {int(np.median(nf))})")
    ax.set_xlabel("$H_1$ generators")
    ax.set_ylabel("encounters")
    ax.set_xticks(range(0, mx + 1, 2))
    ax.set_xlim(-0.6, mx + 0.6)
    ax.set_ylim(0, max(nj_h.max(), nf_h.max()) * 1.45)
    # legend above the panel so it never sits on the bars
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1,
              handlelength=1.0, handletextpad=0.4, borderpad=0.2,
              labelspacing=0.3, fontsize=6)
    ax.text(0.96, 0.88, f"$p = {PVAL:.0e}$", transform=ax.transAxes,
            ha="right", va="top", fontsize=6, color="0.3")
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=200)
    print(f"JEPA median {np.median(nj)}, Fukami median {np.median(nf)}; wrote {OUT_PDF.name}")


if __name__ == "__main__":
    main()
