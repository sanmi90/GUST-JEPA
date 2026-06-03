#!/usr/bin/env python3
"""
session25_cross_encoder.py
==========================

Cross-encoder test of the coordinate-by-coordinate finding. The JEPA (predictive)
latent encodes the future wake collectively (no single coordinate forecasts it; the
combination does, ~0.84 held-out). Does the RECONSTRUCTIVE (Fukami) latent, at
matched d=64, also carry that distributed wake-forecast code, or lack it?

For each encoder family, on the pooled post-impact per-frame data, toward the
future wake enstrophy at H=16 (targets aligned from per_frame_targets by
(case_id, encounter)):

  best single coordinate skill, full-latent (ridge) combination skill, the gap
  (= distributedness), forecast-beyond-forces (partial | G,C_L,C_D), and #coords
  with single skill > 0.5. Held-out on test_b; Fukami averaged over its 3 seeds.

If the predictive latent's combination skill is high and the reconstructive one's
is low, the reconstructive objective fails to retain the future-wake code, which is
the direct mechanism behind the manuscript's wake-specific advantage.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

H = 16
PF = "outputs/session16/exp2/per_frame_targets/{}.npz"
FUK = "outputs/session24/seed_robustness_wake/fukami_latents_seed{}/{}.npz"


def partial(a, b, ctrl):
    ra, rb = rankdata(a), rankdata(b)
    RC = np.column_stack([np.ones(len(a))] + [rankdata(ctrl[:, j]) for j in range(ctrl.shape[1])])
    res = lambda r: r - RC @ np.linalg.lstsq(RC, r, rcond=None)[0]
    return abs(float(np.corrcoef(res(ra), res(rb))[0, 1]))


def targets_map(split):
    """(case_id, enc) -> dict of per-frame observable series, from per_frame_targets."""
    d = np.load(PF.format(split), allow_pickle=True)
    cid = np.asarray(d["case_id"]); enc = np.asarray(d["encounter_index"], int)
    g = lambda k: np.asarray(d[k], float)
    out = {}
    for i in range(len(cid)):
        out[(str(cid[i]), int(enc[i]))] = {
            "wake": g("wake_enstrophy")[i], "C_L": g("C_L")[i], "C_D": g("C_D")[i],
            "G": g("G")[i], "impact": int(np.asarray(d["impact_frame"])[i])}
    return out


def build(zfull, cids, encs, imps, tmap):
    """Pooled post-impact (X, future_wake, forces) aligning targets by (case,enc)."""
    X, fw, F = [], [], []
    miss = 0
    for i in range(len(cids)):
        key = (str(cids[i]), int(encs[i]))
        if key not in tmap:
            miss += 1; continue
        t = tmap[key]; imp = int(imps[i]); Tn = zfull.shape[1]
        for tt in range(imp, Tn - H):
            X.append(zfull[i, tt, :]); fw.append(t["wake"][tt + H])
            F.append([t["G"][tt], t["C_L"][tt], t["C_D"][tt]])
    return np.asarray(X), np.asarray(fw), np.asarray(F), miss


def analyse(Xtr, fwtr, Xte, fwte, Fte):
    ytr = rankdata(fwtr); ytr = (ytr - ytr.mean()) / ytr.std()
    u = Ridge(alpha=1.0).fit(Xtr, ytr).coef_; u = u / np.linalg.norm(u)
    s = Xte @ u
    single = np.array([abs(spearmanr(Xte[:, k], fwte).statistic) for k in range(Xte.shape[1])])
    return {"combo": abs(spearmanr(s, fwte).statistic),
            "beyond_forces": partial(s, fwte, Fte),
            "best_single": float(single.max()),
            "n_strong": int((single > 0.5).sum())}


def jepa_arrays(split):
    d = np.load(PF.format(split), allow_pickle=True)
    return (np.asarray(d["z_full"], float), np.asarray(d["case_id"]),
            np.asarray(d["encounter_index"], int), np.asarray(d["impact_frame"], int))


def fukami_arrays(split, seed):
    d = np.load(FUK.format(seed, split), allow_pickle=True)
    return (np.asarray(d["z_full"], float), np.asarray(d["case_ids"]),
            np.asarray(d["encounter_indices"], int), np.asarray(d["impact_frame"], int))


def main():
    tmap_tr, tmap_te = targets_map("train"), targets_map("test_b")

    print(f"{'encoder':<18}{'combo':>7}{'best1':>7}{'gap':>7}{'>0.5':>6}{'|forces':>9}")
    print("-" * 54)

    # JEPA (predictive)
    z, c, e, im = jepa_arrays("train"); Xtr, fwtr, _, _ = build(z, c, e, im, tmap_tr)
    z, c, e, im = jepa_arrays("test_b"); Xte, fwte, Fte, _ = build(z, c, e, im, tmap_te)
    r = analyse(Xtr, fwtr, Xte, fwte, Fte)
    print(f"{'JEPA (predictive)':<18}{r['combo']:>7.3f}{r['best_single']:>7.3f}"
          f"{r['combo']-r['best_single']:>7.3f}{r['n_strong']:>6}{r['beyond_forces']:>9.3f}")
    jepa = r

    # Fukami (reconstructive), 3 seeds
    fuk = {k: [] for k in ["combo", "best_single", "n_strong", "beyond_forces"]}
    for seed in (0, 1, 2):
        z, c, e, im = fukami_arrays("train", seed); Xtr, fwtr, _, m1 = build(z, c, e, im, tmap_tr)
        z, c, e, im = fukami_arrays("test_b", seed); Xte, fwte, Fte, m2 = build(z, c, e, im, tmap_te)
        r = analyse(Xtr, fwtr, Xte, fwte, Fte)
        for k in fuk:
            fuk[k].append(r[k])
        print(f"{'Fukami seed'+str(seed)+' (recon)':<18}{r['combo']:>7.3f}{r['best_single']:>7.3f}"
              f"{r['combo']-r['best_single']:>7.3f}{r['n_strong']:>6}{r['beyond_forces']:>9.3f}"
              + (f"   (missing {m1+m2})" if (m1 + m2) else ""))

    print("-" * 54)
    print(f"{'Fukami mean':<18}{np.mean(fuk['combo']):>7.3f}{np.mean(fuk['best_single']):>7.3f}"
          f"{np.mean(fuk['combo'])-np.mean(fuk['best_single']):>7.3f}"
          f"{np.mean(fuk['n_strong']):>6.1f}{np.mean(fuk['beyond_forces']):>9.3f}")
    print(f"\n[verdict] future-wake combination skill: JEPA {jepa['combo']:.2f} vs "
          f"Fukami {np.mean(fuk['combo']):.2f}; beyond-forces: JEPA {jepa['beyond_forces']:.2f} vs "
          f"Fukami {np.mean(fuk['beyond_forces']):.2f}")

    import json
    Path("outputs_causal/jepa_modes/cross_encoder.json").write_text(json.dumps(
        {"jepa": jepa, "fukami_seeds": fuk}, indent=2))


if __name__ == "__main__":
    main()
