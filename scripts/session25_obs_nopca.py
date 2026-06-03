#!/usr/bin/env python3
"""
session25_obs_nopca.py
======================

Observability analysis of the JEPA latent WITHOUT PCA. For each of the six future
observables we find, by ridge regression on the raw JEPA latent (no rotation, no
component selection), the single latent direction that best forecasts it, and
report:

  * held-out forecast skill (|Spearman| of the projection vs the future observable
    on test_b);
  * the variance share that direction occupies in the latent (u^T Cov(z) u /
    trace Cov(z)); random unit dir = 1/64 = 1.6%;
  * |cos| with the leading latent principal component (high => the forecast
    direction is a high-variance one a reconstruction objective keeps; low => a
    low-variance one it discards). PCA is used only to MEASURE this, never to find
    the direction.

Plus the held-out observability O = I(full latent; future obs)/H of the full
64-dim latent for each observable (basis-free, surrogate-null debiased), to confirm
every observable is observable from the latent.

Panel-(a) data for the figure (wake forecast direction, partialling controls) is
also computed. Same pooled post-impact per-frame regime as the other scripts.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import h5py
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from infotheory.estimators import surrogate_null_mi, mutual_information_knn  # noqa: E402
from infotheory.observability import target_entropy  # noqa: E402

H = 16
SEED = 0
PF = "outputs/session16/exp2/per_frame_targets/{}.npz"
OBS = ["C_L", "C_D", "I_y", "wake_enstrophy", "circulation_pos", "circulation_neg"]
KIND = {"C_L": "force", "C_D": "force", "I_y": "force",
        "wake_enstrophy": "wake", "circulation_pos": "wake", "circulation_neg": "wake"}


def cache_root():
    c = os.environ.get("VORTEX_JEPA_CACHE")
    return Path(c) if c else Path(os.environ["PREVENT_ROOT"]) / "data" / "processed" / "vortex-jepa"


def Iy_future(cid, enc, frames, root):
    f = root / "v1" / str(cid) / f"encounter_{int(enc):02d}.h5"
    if not f.exists():
        return None
    with h5py.File(f, "r") as g:
        om = np.asarray(g["omega_z"][frames[0]:frames[1]], float)
    x = np.linspace(-1.5, 4.5, om.shape[1])
    return (x[None, :] * om.sum(axis=2)).sum(axis=1)  # over [frames]


def pooled(split, root):
    d = np.load(PF.format(split), allow_pickle=True)
    z = np.asarray(d["z_full"], float); imp = np.asarray(d["impact_frame"], int)
    n, T, _ = z.shape
    cid = np.asarray(d["case_id"]); enc = np.asarray(d["encounter_index"], int)
    g = lambda k: np.asarray(d[k], float)
    have = {k: g(k) for k in OBS if k != "I_y" and k in d.files}
    Iy = np.full((n, T), np.nan)
    for i in range(n):
        v = Iy_future(cid[i], enc[i], (0, T), root)
        if v is not None:
            Iy[i, :len(v)] = v
    have["I_y"] = Iy
    rows = [(i, t) for i in range(n) for t in range(int(imp[i]), T - H)]
    ii = np.array([r[0] for r in rows]); tt = np.array([r[1] for r in rows])
    X = z[ii, tt, :]
    fut = {k: have[k][ii, np.clip(tt + H, 0, T - 1)] for k in OBS}
    cur = {k: g(k)[ii, tt] for k in ["G", "C_L", "C_D", "wake_enstrophy",
                                     "circulation_pos", "circulation_neg"]}
    return X, fut, cur


def partial(a, b, ctrl):
    ra, rb = rankdata(a), rankdata(b)
    RC = np.column_stack([np.ones(len(a))] + [rankdata(ctrl[:, j]) for j in range(ctrl.shape[1])])
    res = lambda r: r - RC @ np.linalg.lstsq(RC, r, rcond=None)[0]
    return abs(float(np.corrcoef(res(ra), res(rb))[0, 1]))


def main():
    root = cache_root()
    Xtr, ftr, ctr = pooled("train", root)
    Xte, fte, cte = pooled("test_b", root)
    C = np.cov(Xtr, rowvar=False); trC = np.trace(C)
    from sklearn.decomposition import PCA
    pc1 = PCA(n_components=1, random_state=SEED).fit(Xtr).components_[0]

    res = {}
    print(f"[data] train n={len(Xtr)}, test_b n={len(Xte)}\n")
    print(f"{'observable':<16}{'kind':<7}{'skill_te':>9}{'var%':>8}{'|cosPC1|':>9}{'O_full_te':>10}")
    print("-" * 59)
    for ob in OBS:
        ytr = rankdata(ftr[ob]); ytr = (ytr - ytr.mean()) / ytr.std()
        u = Ridge(alpha=1.0).fit(Xtr, ytr).coef_; u = u / np.linalg.norm(u)
        var = float(u @ C @ u / trC)
        skill = abs(spearmanr(Xte @ u, fte[ob]).statistic)
        cos1 = abs(float(u @ pc1))
        # held-out observability of the FULL latent toward this observable
        null = surrogate_null_mi(fte[ob], Xte, n_surrogate=30, k=4, random_state=SEED)
        O = float(np.clip(null["mi_debiased"] / target_entropy(fte[ob]), 0, 1))
        res[ob] = {"kind": KIND[ob], "skill_te": skill, "var_share": var,
                   "cos_pc1": cos1, "O_full_te": O}
        print(f"{ob:<16}{KIND[ob]:<7}{skill:>9.3f}{var*100:>8.2f}{cos1:>9.3f}{O:>10.3f}")

    # panel (a): wake forecast direction asymmetry
    yw = rankdata(ftr["wake_enstrophy"]); yw = (yw - yw.mean()) / yw.std()
    uw = Ridge(alpha=1.0).fit(Xtr, yw).coef_; uw = uw / np.linalg.norm(uw)
    def asym(X, fut, cur):
        s = X @ uw
        F = np.column_stack([cur["G"], cur["C_L"], cur["C_D"]])
        W = np.column_stack([cur["wake_enstrophy"], cur["circulation_pos"], cur["circulation_neg"]])
        return {"s_raw": abs(spearmanr(s, fut["wake_enstrophy"]).statistic),
                "s_par_F": partial(s, fut["wake_enstrophy"], F),
                "s_par_FW": partial(s, fut["wake_enstrophy"], np.column_stack([F, W])),
                "cl_par_s": partial(cur["C_L"], fut["wake_enstrophy"], s[:, None])}
    a_tr, a_te = asym(Xtr, ftr, ctr), asym(Xte, fte, cte)
    print("\n=== wake forecast direction asymmetry (train | test_b) ===")
    for lab, k in [("s -> future wake (raw)", "s_raw"), ("s | forces", "s_par_F"),
                   ("s | forces + current wake", "s_par_FW"), ("C_L | s", "cl_par_s")]:
        print(f"  {lab:<28}{a_tr[k]:>8.3f}{a_te[k]:>9.3f}")
    res["_asym"] = {"train": a_tr, "test_b": a_te}

    out = Path("outputs_causal/jepa_modes"); out.mkdir(parents=True, exist_ok=True)
    (out / "obs_nopca.json").write_text(json.dumps(res, indent=2))
    print(f"\n[done] wrote {out/'obs_nopca.json'}")


if __name__ == "__main__":
    main()
