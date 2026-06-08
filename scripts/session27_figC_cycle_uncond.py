"""SPEC 4 - NEW FIG C: the encounter as a cycle (centrepiece, Smith Fig 5 grammar).

One idea: the gust encounter is a single closed cycle in the predictive latent;
it departs the baseline limit cycle, executes LEV growth and shedding, and
returns, and the predictive decode tracks the staged flow while the
reconstructive decode collapses.

Panels:
 (a) latent loop in (PC1, PC2): baseline limit cycle (light ring) + gust
     trajectory, four numbered stage glyphs, a direction arrow.
 (b) phase theta(t) along the orbit vs frames relative to impact, stages marked.
 (c,d,e) four stage rows x {simulation, JEPA decode, Fukami decode} vorticity
     snapshots, each decode annotated with its OT field distance.
Sources: latents z_full (PCA), decoded/test_b.npz, ot_results.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

import sys

# SESSION27 UNCONDITIONED VARIANT of scripts/session21/figC_cycle.py.
# Same cycle/phase/staged-flow construction, but the predictive (JEPA) latents and
# decoded fields are the fully-unconditioned (no-c) model:
#   LAT -> outputs/session27/latents_own_jepa_noc_tf (plural keys),
#   DEC -> outputs/session27/decoded_noc_tf/test_b.npz.
# The OT helpers (build_cost_matrix, d_field) are reused unchanged from production.
REPO = Path(__file__).resolve().parents[1]  # session27 copy lives directly under scripts/
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "session21"))
sys.path.insert(0, str(REPO / "scripts" / "session20"))
import figstyle as fs  # noqa: E402
from exp_ot_field_and_alignment import build_cost_matrix, d_field  # noqa: E402

LAT = REPO / "outputs/session27/latents_own_jepa_noc_tf"
DEC = REPO / "outputs/session27/decoded_noc_tf/test_b.npz"
OUT_PDF = REPO / "outputs/session27/figs_uncond/figC_cycle.pdf"
OUT_PNG = REPO / "outputs/session27/figs_uncond/figC_cycle.png"

STAGES = [-8, 0, 16, 32]          # frames relative to impact -> glyphs 1..4
STAGE_NAME = ["baseline", "impact", "peak load", "recovery"]


def main() -> None:
    fs.use_style()
    tr = np.load(LAT / "train.npz", allow_pickle=True)
    tb = np.load(LAT / "test_b.npz", allow_pickle=True)
    dec = np.load(DEC, allow_pickle=True)
    cost = build_cost_matrix()   # for per-snapshot Sinkhorn OT field distance

    # baseline (no-gust) defines the orbit plane; project everything into it
    trG = tr["G"]
    base = tr["z_full"][np.isclose(trG, 0.0)]            # (n_base, 120, 64)
    pca = PCA(n_components=2).fit(base.reshape(-1, 64))

    # representative strong-gust encounter, matched across latents and decodes
    # (unconditioned latents use the plural key 'case_ids'; production used 'case_id')
    tb_cid_key = "case_ids" if "case_ids" in tb.files else "case_id"
    tb_cid = np.array([str(c) for c in tb[tb_cid_key]])
    rep_cid = tb_cid[int(np.argmax(np.abs(tb["G"])))]
    gi = int(np.where(tb_cid == rep_cid)[0][0])
    gust_pc = pca.transform(tb["z_full"][gi])            # (120, 2)
    dec_cid = np.array([str(c) for c in dec["case_ids"]])
    dgi = int(np.where(dec_cid == rep_cid)[0][0])
    impact = int(dec["impact_frame"][dgi])
    dec_off = list(dec["offsets"])
    stage_idx = [dec_off.index(s) for s in STAGES]

    jepa = fs.FAMILY_COLOR["jepa"]
    fig = plt.figure(figsize=fs.figure_size(1.0, aspect=1.06))
    # decoupled spacings: generous between the two left panels, tight among the
    # snapshot tiles, and a clear horizontal gap between the two blocks. The right
    # block carries most of the width so the staged-flow fields are large.
    outer = fig.add_gridspec(1, 2, width_ratios=[0.62, 2.35], wspace=0.16,
                             left=0.085, right=0.88, top=0.91, bottom=0.09)
    gsL = outer[0, 0].subgridspec(2, 1, hspace=0.55)
    gsR = outer[0, 1].subgridspec(4, 3, hspace=0.12, wspace=0.10)

    # (a) departure from the baseline orbit -----------------------------------
    # A 2-D projection of the encounter is a self-intersecting shadow of a >2-D
    # orbit, so we instead show the distance of the gust trajectory to the
    # baseline orbit point cloud (full 64-D), per frame, in units of the baseline
    # orbit diameter. The encounter departs the baseline cycle and only partially
    # relaxes back: it does not close within the window (S4.4). The single-cycle
    # topology is established coordinate-free by the persistent-homology figure.
    axa = fig.add_subplot(gsL[0])
    gust64 = tb["z_full"][gi]                            # (120, 64)
    base_flat = base.reshape(-1, 64)
    d_gust = cdist(gust64, base_flat).min(axis=1)        # (120,)
    mu = base_flat.mean(0)
    diam = np.linalg.norm(base_flat - mu, axis=1).max() * 2.0
    dd = cdist(base_flat, base_flat)
    np.fill_diagonal(dd, np.inf)
    band = float(np.percentile(dd.min(1), 95) / diam)
    dn = d_gust / diam
    trel = np.arange(120) - impact
    axa.axhspan(0, band, color="0.88", zorder=0)
    axa.plot(trel, dn, color=jepa, lw=1.4, zorder=2)
    for n, s in enumerate(STAGES, start=1):
        axa.axvline(s, color="0.8", lw=0.5, ls="--", zorder=0)
        axa.text(s, dn.max() * 1.03, str(n), ha="center", fontsize=5.5, color="0.45")
    axa.set_xlim(-12, 79)
    axa.set_ylim(0, dn.max() * 1.13)
    axa.set_xlabel("frames rel. impact", labelpad=2)
    axa.set_ylabel("distance to\nbaseline orbit", labelpad=2)
    axa.set_title("(a) departure", fontsize=7.5, loc="left")

    # (b) phase along the orbit ---------------------------------------------
    axb = fig.add_subplot(gsL[1])
    sig = gust_pc[:, 0] - gust_pc[:, 0].mean()
    theta = np.unwrap(np.angle(hilbert(sig)))
    theta = (theta - theta[impact]) % (2 * np.pi)
    trel = np.arange(120) - impact
    axb.plot(trel, theta, color=jepa, lw=1.0)
    for n, s in enumerate(STAGES, start=1):
        axb.axvline(s, color="0.8", lw=0.6, ls="--", zorder=0)
        axb.text(s, 2 * np.pi * 1.04, str(n), ha="center", fontsize=5.5,
                 color="0.4")
    axb.set_xlim(-12, 40); axb.set_ylim(0, 2 * np.pi * 1.12)
    axb.set_xticks([0, 20, 40])
    axb.set_yticks([0, np.pi, 2 * np.pi]); axb.set_yticklabels(["0", r"$\pi$", r"$2\pi$"])
    axb.set_xlabel("frames relative to impact", labelpad=2)
    axb.set_ylabel(r"phase $\theta$", labelpad=2)
    axb.set_title("(b) orbit phase", fontsize=7.5, loc="left")

    # (c) snapshot grid: rows = stages, cols = sim / JEPA / Fukami -----------
    cols = [("target_norm", "simulation", False),
            ("jepa_norm", "predictive", True),
            ("fukami_norm", "reconstructive", True)]
    im = None
    ax00 = None
    for r, (s, si) in enumerate(zip(STAGES, stage_idx)):
        for c, (key, title, show_ot) in enumerate(cols):
            ax = fig.add_subplot(gsR[r, c])
            if r == 0 and c == 0:
                ax00 = ax
            im = fs.vort_panel(ax, dec[key][dgi, si])
            if r == 0:
                ax.set_title(title, fontsize=7, pad=3)
            if c == 0:
                fs.stage_glyph(ax, 14, 80, r + 1, color="0.15", s=64, fontsize=5.5)
            if show_ot:
                otv = d_field(dec[key][dgi, si], dec["target_norm"][dgi, si], cost)
                ax.text(0.97, 0.06, f"OT {otv:.1f}", transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=5,
                        bbox=dict(boxstyle="round,pad=0.1", fc="white",
                                  ec="none", alpha=0.7))
    ax00.text(0.0, 1.42, "(c) staged flow", transform=ax00.transAxes,
              ha="left", va="bottom", fontweight="bold", fontsize=7.5)
    cax = fig.add_axes([0.905, 0.14, 0.012, 0.30])
    fig.colorbar(im, cax=cax, label=r"$\omega_z$ (norm.)")

    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=200)
    print(f"rep encounter: {rep_cid}; wrote {OUT_PDF.name}")


if __name__ == "__main__":
    main()
