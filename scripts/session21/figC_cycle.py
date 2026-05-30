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
from sklearn.decomposition import PCA

import figstyle as fs

REPO = Path(__file__).resolve().parents[2]
LAT = REPO / "outputs/session18/exp_b1/latents_jepa_d64_test1_noBN"
DEC = REPO / "outputs/session20/decoded/test_b.npz"
OUT_PDF = REPO / "paper/sections/figures/results/figC_cycle.pdf"
OUT_PNG = REPO / "outputs/session21/figs/figC_cycle.png"

import sys  # noqa: E402
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "session20"))
from exp_ot_field_and_alignment import build_cost_matrix, d_field  # noqa: E402

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
    base_pc = pca.transform(base.reshape(-1, 64)).reshape(base.shape[0], -1, 2)

    # representative strong-gust encounter, matched across latents and decodes
    tb_cid = np.array([str(c) for c in tb["case_id"]])
    rep_cid = tb_cid[int(np.argmax(np.abs(tb["G"])))]
    gi = int(np.where(tb_cid == rep_cid)[0][0])
    gust_pc = pca.transform(tb["z_full"][gi])            # (120, 2)
    dec_cid = np.array([str(c) for c in dec["case_ids"]])
    dgi = int(np.where(dec_cid == rep_cid)[0][0])
    impact = int(dec["impact_frame"][dgi])
    dec_off = list(dec["offsets"])
    stage_idx = [dec_off.index(s) for s in STAGES]

    fig = plt.figure(figsize=fs.figure_size(1.0, aspect=0.92))
    gs = fig.add_gridspec(4, 5, hspace=0.18, wspace=0.12,
                          width_ratios=[1, 1, 0.9, 0.9, 0.9])

    # (a) latent loop
    axa = fig.add_subplot(gs[0:2, 0:2])
    axa.plot(base_pc[0, :, 0], base_pc[0, :, 1], color="0.62", lw=0.9, zorder=1)
    axa.text(base_pc[0, :, 0].mean(), base_pc[0, :, 1].min(), "baseline\ncycle",
             ha="center", va="top", fontsize=5.5, color="0.5")
    axa.plot(gust_pc[:, 0], gust_pc[:, 1], color=fs.FAMILY_COLOR["jepa"],
             lw=1.2, zorder=2)
    for n, s in enumerate(STAGES, start=1):
        f = impact + s
        fs.stage_glyph(axa, gust_pc[f, 0], gust_pc[f, 1], n,
                       color=fs.FAMILY_COLOR["jepa"], s=70, fontsize=6)
    # direction arrow near impact
    f0, f1 = impact + 2, impact + 6
    axa.annotate("", xy=gust_pc[f1], xytext=gust_pc[f0],
                 arrowprops=dict(arrowstyle="-|>", color=fs.FAMILY_COLOR["jepa"],
                                 lw=1.0))
    axa.set_xlabel("PC1"); axa.set_ylabel("PC2")
    axa.set_title("predictive latent: encounter as a cycle", fontsize=7.5)
    axa.tick_params(labelleft=False, labelbottom=False)

    # (b) phase along the orbit
    axb = fig.add_subplot(gs[2:4, 0:2])
    sig = gust_pc[:, 0] - gust_pc[:, 0].mean()
    theta = np.unwrap(np.angle(hilbert(sig)))
    theta = (theta - theta[impact]) % (2 * np.pi)
    trel = np.arange(120) - impact
    axb.plot(trel, theta, color=fs.FAMILY_COLOR["jepa"], lw=1.0)
    for n, s in enumerate(STAGES, start=1):
        axb.axvline(s, color="0.8", lw=0.6, ls="--", zorder=0)
        axb.text(s, 2 * np.pi * 1.02, str(n), ha="center", fontsize=5.5,
                 color="0.4")
    axb.set_xlim(-12, 40); axb.set_ylim(0, 2 * np.pi * 1.08)
    axb.set_yticks([0, np.pi, 2 * np.pi]); axb.set_yticklabels(["0", r"$\pi$", r"$2\pi$"])
    axb.set_xlabel("frames relative to impact"); axb.set_ylabel(r"phase $\theta$")

    # (c,d,e) snapshot grid: rows = stages, cols = sim / JEPA / Fukami
    cols = [("target_norm", "simulation", False),
            ("jepa_norm", "predictive", True),
            ("fukami_norm", "reconstructive", True)]
    im = None
    for r, (s, si) in enumerate(zip(STAGES, stage_idx)):
        for c, (key, title, show_ot) in enumerate(cols):
            ax = fig.add_subplot(gs[r, 2 + c])
            im = fs.vort_panel(ax, dec[key][dgi, si])
            if r == 0:
                ax.set_title(title, fontsize=7)
            if c == 0:
                fs.stage_glyph(ax, 14, 80, r + 1, color="0.15", s=70, fontsize=6)
            if show_ot:
                otv = d_field(dec[key][dgi, si], dec["target_norm"][dgi, si], cost)
                ax.text(0.97, 0.04, f"OT {otv:.1f}", transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=5,
                        bbox=dict(boxstyle="round,pad=0.1", fc="white",
                                  ec="none", alpha=0.7))
    cax = fig.add_axes([0.92, 0.12, 0.012, 0.30])
    fig.colorbar(im, cax=cax, label=r"$\omega_z$ (norm.)")

    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"rep encounter: {rep_cid}; wrote {OUT_PDF.name}")


if __name__ == "__main__":
    main()
