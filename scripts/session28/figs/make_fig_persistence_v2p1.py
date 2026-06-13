"""Persistence figure (fig_persistence_v2p1.pdf): persistent homology of the
encoded NO-GUST limit cycle on test_b, under the per-family WHITENED (stdz) metric
that removes anisotropic coordinate scale. v2.1.

Three panels, matching the v2.1 caption (fig:persistence, section_4_results.tex):
  Left, centre: representative Vietoris-Rips H_1 persistence diagrams of the
      encoded predictive and reconstructive latents on a single shedding period of
      the undisturbed Baseline orbit; coloured points are significant generators
      (lifetime above the dotted noise floor), grey points are noise.
  Right: the significant-H_1 generator-count distribution over the undisturbed
      shedding periods. The predictive encoding is a single clean loop (median
      TopoBaseWhtMedJepa = 1.0) while the reconstructive encoding fragments (median
      TopoBaseWhtMedFukami = 2.5).

The per-family single-period generator counts and the representative diagrams are
recomputed with the SAME ripser machinery and the SAME per-family stdz whitening
as scripts/session28/topology_ce2.py (imported verbatim), on the Baseline (no-gust)
encounters in the train split, so the medians reproduce the macro-wired numbers.
The whitened metric (stdz) is the headline metric of topology_ce2 (its README).

CPU only; ripser 0.6.15. Read-only on the latents; writes only the PDF.

Output: paper/sections/figures/results/fig_persistence_v2p1.pdf
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.distance import pdist  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402

import _common as cm  # noqa: E402
import topology_ce2 as topo  # noqa: E402  (verbatim ripser + whitening machinery)

fs = cm.fs

METRIC = "stdz"  # the headline whitened metric of topology_ce2 (its README)
FLOOR = topo.CANONICAL_FLOOR  # 0.05 of the cloud diameter
JEPA_FAMILY = "jepa_tf_noc_d64_s42"
FUKAMI_FAMILY = "fukami_d64_s42"


def single_period_segments(family: str) -> tuple[list[np.ndarray], np.ndarray]:
    """Per-family whitened (stdz) single-period Baseline segments + their H1 counts.

    Reproduces the topology_ce2 "single-period no-gust control": the Baseline
    encounters live in the train split; each 120-frame trajectory is whitened by
    the family's own train stdz transform and segmented into single shedding
    periods (period estimated as in topology_ce2). For each segment we return the
    whitened point cloud and its significant-H1 generator count at the 5% floor.
    """
    whit = topo.fit_whiteners(family)
    train = topo.load_split(family, "train")
    base_idx = np.nonzero(train["case_ids"] == topo.BASELINE_CASE)[0]
    period = topo.estimate_period_frames(train["z_full"][base_idx[0]])
    segs: list[np.ndarray] = []
    counts: list[int] = []
    for i in base_idx:
        whitened = topo.apply_metric(train["z_full"][i], METRIC, whit)
        for seg in topo.segment_periods(whitened, period):
            segs.append(seg)
            counts.append(topo.persistence_counts(seg, (FLOOR,))[FLOOR]["H1"])
    return segs, np.asarray(counts, dtype=float)


def diagram(ax, cloud: np.ndarray, color: str, title: str) -> None:
    """Plot the Vietoris-Rips H_1 persistence diagram of one whitened point cloud."""
    res = topo.ripser_fn(cloud, maxdim=1)
    dgm1 = res["dgms"][1]
    diam = float(pdist(cloud).max())
    floor = FLOOR * diam
    if dgm1.size:
        fin = dgm1[np.isfinite(dgm1[:, 1])]
    else:
        fin = np.zeros((0, 2))
    b, d = (fin[:, 0], fin[:, 1]) if fin.size else (np.zeros(0), np.zeros(0))
    sig = (d - b) >= floor
    lim = max(d.max() if d.size else diam, diam) * 1.05
    ax.plot([0, lim], [0, lim], ls="--", lw=0.7, color="0.7", zorder=1)
    ax.plot([0, lim - floor], [floor, lim], ls=":", lw=0.7, color="0.5", zorder=1)
    if (~sig).any():
        ax.scatter(b[~sig], d[~sig], s=8, color="0.7", zorder=2)
    if sig.any():
        ax.scatter(b[sig], d[sig], s=22, color=color, zorder=3, edgecolors="white", linewidths=0.4)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("birth")
    ax.set_ylabel("death")
    ax.set_title(title, fontsize=7.5)
    ax.text(
        0.04,
        0.95,
        f"{int(sig.sum())} significant\n$H_1$ generators",
        transform=ax.transAxes,
        va="top",
        fontsize=6,
        color=color,
    )


def main() -> None:
    fs.use_style()
    numbers = cm.load_numbers()
    med_jepa = cm.macro_value(numbers, "TopoBaseWhtMedJepa")
    med_fukami = cm.macro_value(numbers, "TopoBaseWhtMedFukami")

    segs_j, cj = single_period_segments(JEPA_FAMILY)
    segs_f, cf = single_period_segments(FUKAMI_FAMILY)
    # Mann-Whitney one-sided: reconstructive fragments more (greater H1) than predictive.
    p = float(mannwhitneyu(cf, cj, alternative="greater").pvalue)

    # representative single periods: predictive = a clean single loop; reconstructive
    # = a typical fragmented one (>= 2 generators). Pick the segment whose count is
    # the family median and, among those, the most prominent loop / fragmentation.
    ij = int(np.flatnonzero(cj == 1)[0]) if (cj == 1).any() else int(np.argmin(cj))
    frag = np.flatnonzero(cf >= 2)
    ifk = int(frag[0]) if frag.size else int(np.argmax(cf))

    fig, axes = plt.subplots(1, 3, figsize=fs.figure_size(1.0, aspect=0.40), layout="constrained")
    diagram(axes[0], segs_j[ij], fs.FAMILY_COLOR["jepa"], "predictive (JEPA) encoding")
    diagram(axes[1], segs_f[ifk], fs.FAMILY_COLOR["fukami"], "reconstructive encoding")

    ax = axes[2]
    mx = int(max(cj.max(), cf.max()))
    bins = np.arange(-0.5, mx + 1.5)
    nj_h, _, _ = ax.hist(
        cj,
        bins=bins,
        color=fs.FAMILY_COLOR["jepa"],
        alpha=0.7,
        rwidth=0.9,
        label=f"predictive (median {med_jepa:.1f})",
    )
    nf_h, _, _ = ax.hist(
        cf,
        bins=bins,
        color=fs.FAMILY_COLOR["fukami"],
        alpha=0.55,
        rwidth=0.9,
        label=f"reconstructive (median {med_fukami:.1f})",
    )
    ax.set_xlabel("$H_1$ generators")
    ax.set_ylabel("shedding periods")
    ax.set_xticks(range(0, mx + 1, max(1, mx // 5)))
    ax.set_xlim(-0.6, mx + 0.6)
    ax.set_ylim(0, max(nj_h.max(), nf_h.max()) * 1.45)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=1,
        handlelength=1.0,
        handletextpad=0.4,
        borderpad=0.2,
        labelspacing=0.3,
        fontsize=6,
    )
    ax.text(
        0.96,
        0.88,
        f"$p = {p:.0e}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6,
        color="0.3",
    )

    out = cm.out_path("fig_persistence")
    fig.savefig(out)
    print(f"wrote {out}")
    print(
        f"  predictive single-period H1: n={cj.size}, median {np.median(cj):.1f} "
        f"(macro {med_jepa:.1f})"
    )
    print(
        f"  reconstructive single-period H1: n={cf.size}, median {np.median(cf):.1f} "
        f"(macro {med_fukami:.1f}); Mann-Whitney one-sided p={p:.2e}"
    )


if __name__ == "__main__":
    main()
