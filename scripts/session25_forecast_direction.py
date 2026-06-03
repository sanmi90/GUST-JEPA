#!/usr/bin/env python3
"""
session25_forecast_direction.py
===============================

Option 2 (basis-free): instead of asking whether a PRINCIPAL COMPONENT of the
latent forecasts the wake, directly find THE latent direction most predictive of
the future wake enstrophy, with no PCA. This sidesteps the "PCA of an already
reduced model" concern: we are not rotating or re-reducing the JEPA latent, we are
asking which single direction of it best linearly predicts the future wake.

Method. On the pooled post-impact per-frame TRAINING latent, fit a ridge regression
of rank(future wake enstrophy) on the 64 latent signals (rank target = robust to
the heavy-tailed wake). The unit coefficient vector u is the "forecast direction";
s(t) = u . z(t) is the scalar forecast signal. Then, applying the train-fit u
unchanged to held-out test_b:

  * does s predict the future wake beyond the force signature {G, C_L, C_D}?
  * does the lift coefficient add anything once s is known?
  * what physical quantity does s track (|Spearman| vs descriptors)?
  * how much of the latent variance does the forecast direction occupy
    (u^T Cov(z) u / trace Cov(z))?  random unit dir ~ 1/64 = 1.6%.

The last number is the honest test of the "a reconstruction objective would discard
this direction" claim, with no PCA-component selection.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

H = 16
SEED = 0
PF = "outputs/session16/exp2/per_frame_targets/{}.npz"
DESC = ["G", "C_L", "C_D", "centroid_x", "centroid_y", "wake_thickness",
        "circulation_pos", "circulation_neg", "wake_enstrophy", "Y", "D"]


def partial_spearman(a, b, ctrl):
    ra, rb = rankdata(a), rankdata(b)
    RC = np.column_stack([np.ones(len(a))] + [rankdata(ctrl[:, j]) for j in range(ctrl.shape[1])])
    res = lambda r: r - RC @ np.linalg.lstsq(RC, r, rcond=None)[0]
    return abs(float(np.corrcoef(res(ra), res(rb))[0, 1]))


def pooled(split):
    d = np.load(PF.format(split), allow_pickle=True)
    z = np.asarray(d["z_full"], float); imp = np.asarray(d["impact_frame"], int)
    n, T, _ = z.shape
    g = lambda k: np.asarray(d[k], float)
    rows = [(i, t) for i in range(n) for t in range(int(imp[i]), T - H)]
    ii = np.array([r[0] for r in rows]); tt = np.array([r[1] for r in rows])
    X = z[ii, tt, :]
    cur = {k: g(k)[ii, tt] for k in DESC if k in d.files}
    fut_w = g("wake_enstrophy")[ii, np.clip(tt + H, 0, T - 1)]
    return X, cur, fut_w


def asym(X, cur, fut, u):
    s = X @ u
    forces = np.column_stack([cur["G"], cur["C_L"], cur["C_D"]])
    return {
        "s_raw": abs(spearmanr(s, fut).statistic),
        "s_par_forces": partial_spearman(s, fut, forces),
        "cl_raw": abs(spearmanr(cur["C_L"], fut).statistic),
        "cl_par_s": partial_spearman(cur["C_L"], fut, s[:, None]),
    }


def main():
    Xtr, ctr, ftr = pooled("train")
    Xte, cte, fte = pooled("test_b")

    # forecast direction: ridge of rank(future wake) on the latent (train)
    ytr = rankdata(ftr); ytr = (ytr - ytr.mean()) / ytr.std()
    w = Ridge(alpha=1.0).fit(Xtr, ytr).coef_
    u = w / np.linalg.norm(w)

    # variance share of the forecast direction (raw latent coords)
    C = np.cov(Xtr, rowvar=False)
    share = float(u @ C @ u / np.trace(C))
    pca = PCA().fit(Xtr)
    ev = pca.explained_variance_ratio_
    # overlap of the forecast direction with the leading PCs
    cos = [abs(float(u @ pca.components_[k])) for k in range(5)]

    print(f"[data] train n={len(Xtr)}, test_b n={len(Xte)}\n")
    print("=== forecast direction (ridge, basis-free; no PCA selection) ===")
    print(f"variance share of the forecast direction: {share*100:.1f}%  "
          f"(random unit dir = {100/64:.1f}%, PC1 = {ev[0]*100:.1f}%, PC3 = {ev[2]*100:.1f}%)")
    print(f"|cos| with PC1..PC5: {[round(c,2) for c in cos]}")
    print(f"effective # of PCs the direction spans (1/sum cos^4-ish proxy): "
          f"{1/np.sum(np.array(cos)**4):.1f}")

    print("\n=== forecast asymmetry (train | test_b) ===")
    a_tr = asym(Xtr, ctr, ftr, u); a_te = asym(Xte, cte, fte, u)
    for lab, k in [("s -> future wake (raw)", "s_raw"),
                   ("s -> future wake | G,C_L,C_D", "s_par_forces"),
                   ("C_L -> future wake (raw)", "cl_raw"),
                   ("C_L -> future wake | s", "cl_par_s")]:
        print(f"  {lab:<32}{a_tr[k]:>8.3f}{a_te[k]:>9.3f}")

    print("\n=== what the forecast signal s tracks (|Spearman|, current frame, train) ===")
    s = Xtr @ u
    for k in DESC:
        if k in ctr:
            print(f"  {k:<16}{abs(spearmanr(s, ctr[k]).statistic):.3f}")

    # honest verdict on the low-variance claim
    print("\n[verdict] forecast-direction variance share =", f"{share*100:.1f}%;",
          "LOW (reconstruction-discards story holds basis-free)" if share < 0.15
          else "NOT low (the low-variance framing was partly a PCA-selection artifact)")


if __name__ == "__main__":
    main()
