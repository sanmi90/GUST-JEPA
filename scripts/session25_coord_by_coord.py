#!/usr/bin/env python3
"""
session25_coord_by_coord.py
===========================

Coordinate-by-coordinate analysis of the JEPA latent: no PCA, no ridge
combination, just each of the 64 native (BatchNorm-unit) latent signals on its own.
For every coordinate z_k we compute, on the pooled post-impact per-frame data:

  * forecast skill |Spearman(z_k(t), obs(t+H))| toward each of five future
    observables (the matrix), train and held-out test_b;
  * forecast-beyond-forces partial |Spearman(z_k, future wake | G,C_L,C_D)| per
    coordinate (is the forecast-beyond-forces signal in any single coordinate, or
    spread?);
  * what each leading coordinate physically tracks (|Spearman| vs descriptors).

Answers: is the structure carried by individual coordinates (specialised neurons),
or distributed across many (so that only a combination, PC or ridge, reveals it)?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

H = 16
PF = "outputs/session16/exp2/per_frame_targets/{}.npz"
OBS = ["C_L", "C_D", "wake_enstrophy", "circulation_pos", "circulation_neg"]
DESC = ["G", "C_L", "C_D", "wake_enstrophy", "circulation_pos", "circulation_neg",
        "centroid_x", "centroid_y", "wake_thickness"]


def partial(a, b, ctrl):
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
    fut = {k: g(k)[ii, np.clip(tt + H, 0, T - 1)] for k in OBS}
    cur = {k: g(k)[ii, tt] for k in DESC if k in d.files}
    return X, fut, cur


def main():
    Xtr, ftr, ctr = pooled("train")
    Xte, fte, cte = pooled("test_b")
    D = Xtr.shape[1]

    # 64 x 5 forecast-skill matrix (held-out test_b)
    M = np.zeros((D, len(OBS)))
    for k in range(D):
        for j, ob in enumerate(OBS):
            M[k, j] = abs(spearmanr(Xte[:, k], fte[ob]).statistic)

    print(f"[data] D={D} coordinates; train n={len(Xtr)}, test_b n={len(Xte)}\n")
    print("=== per-coordinate forecast skill toward each future observable (test_b) ===")
    print(f"{'observable':<16}{'max_k':>7}{'argmax':>8}{'#|>0.3':>8}{'#|>0.5':>8}{'mean':>7}")
    for j, ob in enumerate(OBS):
        col = M[:, j]
        print(f"{ob:<16}{col.max():>7.3f}{int(col.argmax()):>8}{int((col>0.3).sum()):>8}"
              f"{int((col>0.5).sum()):>8}{col.mean():>7.3f}")

    # forecast-beyond-forces per coordinate (test_b), future wake
    F = np.column_stack([cte["G"], cte["C_L"], cte["C_D"]])
    pj = np.array([partial(Xte[:, k], fte["wake_enstrophy"], F) for k in range(D)])
    print("\n=== future-wake forecast BEYOND forces, per coordinate (test_b) ===")
    print(f"  max over coords = {pj.max():.3f} (coord {int(pj.argmax())}); "
          f"#coords>0.3 = {int((pj>0.3).sum())}; #>0.4 = {int((pj>0.4).sum())}; mean = {pj.mean():.3f}")
    print(f"  (for reference, the ridge COMBINATION reached 0.83 beyond forces; "
          f"the best single coordinate reaches {pj.max():.2f})")

    # physical identity of the top-5 coordinates for the future wake
    order = np.argsort(M[:, OBS.index("wake_enstrophy")])[::-1][:5]
    print("\n=== top-5 coordinates for the future wake: what they track (|Spearman|, train) ===")
    hdr = "coord   skill " + "".join(f"{d[:7]:>8}" for d in DESC)
    print(hdr)
    for k in order:
        s = M[k, OBS.index("wake_enstrophy")]
        ids = "".join(f"{abs(spearmanr(Xtr[:,k], ctr[d]).statistic):>8.2f}" for d in DESC)
        print(f"  {k:<5}{s:>6.2f}{ids}")

    np.savez("outputs_causal/jepa_modes/coord_by_coord.npz",
             skill_matrix=M, observables=np.array(OBS), partial_beyond_forces=pj)
    print("\n[done] wrote outputs_causal/jepa_modes/coord_by_coord.npz")


if __name__ == "__main__":
    main()
