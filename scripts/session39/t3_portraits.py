#!/usr/bin/env python3
"""T3 (paper_redesign.md section 6): latent trajectory portraits.

PC1-PC2 latent trajectories per family (d = 32) for the undisturbed case and a
representative gusted encounter, coloured by cycle fraction. Extends the atlas
to a three/four-family comparison panel: the predictive and linear latents trace
coherent limit cycles perturbed by the gust, the published-recipe reconstruction
is irregular. CPU-only (cached latents).

Run:
    OMP_NUM_THREADS=4 taskset -c 0-15 .venv/bin/python \\
        scripts/session39/t3_portraits.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "outputs/session34/trackc_latents"
OUT_FIG = REPO / "paper/sections/figures/results/fig_t3_portraits_v4.pdf"

FAMILIES = {
    "(a) linear (POD)": "pod",
    "(b) predictive (wake)": "jepa_pool_vec",
    "(c) AE (wake)": "ae_wake_pool",
    "(d) reconstructive (Fukami)": "fukami",
}
REP_CASE = "G-0.50_D1.00_Y-0.40"     # representative gusted encounter (decode fig)


def parse_g(case: str) -> float:
    if "aseline" in case or case.startswith("G+0.00"):
        return 0.0
    try:
        tok = case.split("_")[0].replace("G", "")
        return float(tok)
    except ValueError:
        return np.nan


def load_family(run: str):
    z = np.load(CACHE / f"latents_{run}_train.npz", allow_pickle=True)
    Z = z["z_gap"].astype(np.float64)
    cid = np.asarray([str(c) for c in z["case_id"]])
    enc = np.asarray(z["encounter_index"])
    fr = np.asarray(z["frame"])
    return Z, cid, enc, fr


def traj(Z, cid, enc, fr, case, e):
    m = (cid == case) & (enc == e)
    if not m.any():
        return None, None
    idx = np.where(m)[0]
    idx = idx[np.argsort(fr[idx])]
    return Z[idx], fr[idx]


def main() -> None:
    import matplotlib.pyplot as plt
    import sys
    sys.path.insert(0, str(REPO))
    from scripts.session21.figstyle import TEXTWIDTH_IN, use_style
    from sklearn.decomposition import PCA
    use_style()

    fig, axes = plt.subplots(2, 2, figsize=(TEXTWIDTH_IN, 0.82 * TEXTWIDTH_IN))
    for ax, (label, run) in zip(axes.ravel(), FAMILIES.items()):
        Z, cid, enc, fr = load_family(run)
        pca = PCA(n_components=2).fit(Z)
        # baseline limit cycle
        gvals = np.array([parse_g(c) for c in cid])
        base_cases = np.unique(cid[np.abs(gvals) < 1e-6])
        # representative gusted encounter (fallback to first |G| in [1,2])
        rep = REP_CASE if REP_CASE in set(cid) else None
        if rep is None:
            cand = np.unique(cid[(np.abs(gvals) >= 1.0) & (np.abs(gvals) <= 2.0)])
            rep = cand[0] if len(cand) else np.unique(cid)[0]
        # plot baseline (grey limit cycle)
        for bc in base_cases[:1]:
            zb, _ = traj(Z, cid, enc, fr, bc, int(enc[cid == bc][0]))
            if zb is not None:
                yb = pca.transform(zb)
                ax.plot(yb[:, 0], yb[:, 1], color="0.6", lw=0.8, zorder=1,
                        label="undisturbed")
        # plot representative gusted encounter, coloured by cycle fraction
        e0 = int(enc[cid == rep][0])
        zr, frr = traj(Z, cid, enc, fr, rep, e0)
        if zr is not None:
            yr = pca.transform(zr)
            cf = (frr - frr.min()) / max(1, (frr.max() - frr.min()))
            sc = ax.scatter(yr[:, 0], yr[:, 1], c=cf, cmap="viridis", s=6,
                            zorder=3)
            ax.plot(yr[:, 0], yr[:, 1], color="k", lw=0.4, alpha=0.4, zorder=2)
        ax.set_title(label, fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].legend(fontsize=5, loc="upper right", frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel("PC1", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("PC2", fontsize=7)
    cb = fig.colorbar(sc, ax=axes, shrink=0.8)
    cb.set_label("cycle fraction", fontsize=7)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, bbox_inches="tight")
    print(f"wrote {OUT_FIG} (representative gusted case {REP_CASE})")


if __name__ == "__main__":
    main()
