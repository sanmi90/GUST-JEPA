#!/usr/bin/env python3
"""
session25_pc_physical_id.py
===========================

EXPLORATORY interpretability (follow-up to session25_jepa_mode_surd.py). Identify
what the leading PCA modes of the JEPA d=64 latent physically encode, by
correlating the (pooled, post-impact, per-frame) PC scores against the physical
flow descriptors stored alongside the latent. The motivating finding: PC3 (a
low-variance mode) carries the most UNIQUE information about the future wake
enstrophy, more than the dominant gust-strength mode PC1.

Reports |Spearman rho| (robust to the heavy-tailed descriptors) for PC1, PC2, PC3
against each current-frame descriptor, plus a partial-correlation pass that removes
gust strength G, to isolate what PC3 tracks BEYOND the dominant gust-strength axis.
PC signs are arbitrary, so magnitudes are reported. Same pooled regime as the SURD
script so the PCs are identical.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

H = 16
SEED = 0
PER_FRAME = "outputs/session16/exp2/per_frame_targets/train.npz"
DESCRIPTORS = ["G", "D", "Y", "C_L", "C_D", "peak_pos_omega", "peak_neg_omega",
               "centroid_x", "centroid_y", "circulation_pos", "circulation_neg",
               "wake_length", "wake_thickness", "wake_enstrophy"]


def partial_spearman(a, b, c):
    """Partial Spearman corr of a,b controlling for c (rank-residual method)."""
    ra, rb, rc = rankdata(a), rankdata(b), rankdata(c)
    rc1 = np.column_stack([np.ones_like(rc), rc])
    res_a = ra - rc1 @ np.linalg.lstsq(rc1, ra, rcond=None)[0]
    res_b = rb - rc1 @ np.linalg.lstsq(rc1, rb, rcond=None)[0]
    return float(np.corrcoef(res_a, res_b)[0, 1])


def main():
    from sklearn.decomposition import PCA
    d = np.load(PER_FRAME, allow_pickle=True)
    z = np.asarray(d["z_full"], float)
    imp = np.asarray(d["impact_frame"], int)
    n, T, dlat = z.shape
    desc = {k: np.asarray(d[k], float) for k in DESCRIPTORS}

    # pooled post-impact per-frame samples (same regime as the SURD script)
    rows = [(i, t) for i in range(n) for t in range(int(imp[i]), T - H)]
    idx_i = np.array([r[0] for r in rows]); idx_t = np.array([r[1] for r in rows])
    Z = z[idx_i, idx_t, :]
    D = {k: desc[k][idx_i, idx_t] for k in DESCRIPTORS}

    pca = PCA(n_components=3, random_state=SEED).fit(Z)
    P = pca.transform(Z)
    print(f"[data] pooled n={len(Z)} frames; PC var fracs "
          f"{np.round(pca.explained_variance_ratio_, 3).tolist()}\n")

    print(f"{'descriptor':<16}{'|rho|PC1':>9}{'|rho|PC2':>9}{'|rho|PC3':>9}")
    print("-" * 43)
    rho = {}
    for k in DESCRIPTORS:
        rho[k] = [abs(spearmanr(P[:, j], D[k]).statistic) for j in range(3)]
        print(f"{k:<16}{rho[k][0]:>9.3f}{rho[k][1]:>9.3f}{rho[k][2]:>9.3f}")

    print("\nTop current-frame correlate per mode:")
    for j, name in enumerate(["PC1", "PC2", "PC3"]):
        ranked = sorted(DESCRIPTORS, key=lambda k: -rho[k][j])[:3]
        print(f"  {name}: " + ", ".join(f"{k} ({rho[k][j]:.3f})" for k in ranked))

    print("\nPC3 BEYOND gust strength: partial |Spearman(PC3, desc | G)|")
    pc3_partial = {k: abs(partial_spearman(P[:, 2], D[k], D["G"]))
                   for k in DESCRIPTORS if k != "G"}
    for k in sorted(pc3_partial, key=lambda k: -pc3_partial[k])[:6]:
        print(f"  {k:<16}{pc3_partial[k]:.3f}")

    # also: PC1 vs G sanity, and PC3 vs CURRENT wake state vs FUTURE wake
    print(f"\nsanity: |rho|(PC1, G) = {rho['G'][0]:.3f}   "
          f"|rho|(PC3, G) = {rho['G'][2]:.3f}")
    fut_we = desc["wake_enstrophy"][idx_i, np.clip(idx_t + H, 0, T - 1)]
    print(f"PC3 vs current wake_enstrophy |rho| = {abs(spearmanr(P[:,2], D['wake_enstrophy']).statistic):.3f}; "
          f"PC3 vs future wake_enstrophy(t+H) |rho| = {abs(spearmanr(P[:,2], fut_we).statistic):.3f}")


if __name__ == "__main__":
    main()
