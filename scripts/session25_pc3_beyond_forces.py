#!/usr/bin/env python3
"""
session25_pc3_beyond_forces.py
==============================

EXPLORATORY follow-up. PC3 of the JEPA latent correlates with gust strength G AND
the aerodynamic loads C_L, C_D (the "forcing/load" mode) and carries the most
unique information about the future wake. The decisive test for the paper's thesis
(the wake changes while the forces do not, so a representation can hold the force
signature without the wake state): does PC3 retain future-wake and wake-geometry
content AFTER the force signature {G, C_L, C_D} is partialled out?

Reports rank-partial |Spearman| of PC3 against the future wake and against the
current wake-geometry descriptors, controlling for {G, C_L, C_D}. If PC3's
future-wake correlation survives the control, PC3 carries forecast-relevant wake
information beyond the force signature; if it collapses, PC3 is essentially the
load mode. Same pooled post-impact per-frame regime as the other Session 25 scripts.
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
GEOM = ["centroid_x", "centroid_y", "wake_thickness", "circulation_pos",
        "circulation_neg", "wake_enstrophy", "Y", "D", "peak_pos_omega"]


def partial_spearman(a, b, ctrl):
    """Rank-partial Spearman of a,b controlling for columns of ctrl (n,k)."""
    ra, rb = rankdata(a), rankdata(b)
    RC = np.column_stack([np.ones(len(a))] + [rankdata(ctrl[:, j]) for j in range(ctrl.shape[1])])
    ba = np.linalg.lstsq(RC, ra, rcond=None)[0]
    bb = np.linalg.lstsq(RC, rb, rcond=None)[0]
    return float(np.corrcoef(ra - RC @ ba, rb - RC @ bb)[0, 1])


def main():
    from sklearn.decomposition import PCA
    d = np.load(PER_FRAME, allow_pickle=True)
    z = np.asarray(d["z_full"], float)
    imp = np.asarray(d["impact_frame"], int)
    n, T, _ = z.shape
    get = lambda k: np.asarray(d[k], float)

    rows = [(i, t) for i in range(n) for t in range(int(imp[i]), T - H)]
    ii = np.array([r[0] for r in rows]); tt = np.array([r[1] for r in rows])
    Z = z[ii, tt, :]
    cur = {k: get(k)[ii, tt] for k in ["G", "C_L", "C_D", "wake_enstrophy"] + GEOM}
    fut_we = get("wake_enstrophy")[ii, np.clip(tt + H, 0, T - 1)]

    P = PCA(n_components=3, random_state=SEED).fit_transform(Z)
    pc3 = P[:, 2]
    forces = np.column_stack([cur["G"], cur["C_L"], cur["C_D"]])

    def r(a, b):
        return abs(spearmanr(a, b).statistic)

    print(f"[data] pooled n={len(Z)} post-impact frames, H={H}\n")
    print("=== PC3 vs FUTURE wake enstrophy ===")
    print(f"  raw            |rho|(PC3, wake_fut)            = {r(pc3, fut_we):.3f}")
    print(f"  control G      |rho|(PC3, wake_fut | G)        = {abs(partial_spearman(pc3, fut_we, forces[:, :1])):.3f}")
    print(f"  control G,CL,CD|rho|(PC3, wake_fut | G,CL,CD)  = {abs(partial_spearman(pc3, fut_we, forces)):.3f}")
    print("  reference force-only predictors of the future wake:")
    print(f"     |rho|(C_L, wake_fut) = {r(cur['C_L'], fut_we):.3f}   "
          f"|rho|(G, wake_fut) = {r(cur['G'], fut_we):.3f}")
    # does the force signature alone predict the future wake, controlling for PC3?
    print(f"     |rho|(C_L, wake_fut | PC3) = "
          f"{abs(partial_spearman(cur['C_L'], fut_we, pc3[:, None])):.3f}")

    print("\n=== what PC3 encodes BEYOND the force signature {G, C_L, C_D} ===")
    print(f"  {'descriptor':<16}{'raw|rho|':>9}{'|rho| | forces':>16}")
    for k in GEOM + ["wake_enstrophy"]:
        print(f"  {k:<16}{r(pc3, cur[k]):>9.3f}{abs(partial_spearman(pc3, cur[k], forces)):>16.3f}")


if __name__ == "__main__":
    main()
