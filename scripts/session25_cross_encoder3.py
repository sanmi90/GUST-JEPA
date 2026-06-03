#!/usr/bin/env python3
"""
session25_cross_encoder3.py
===========================

Three-family cross-encoder test (predictive JEPA, reconstructive Fukami, linear
POD), at matched d=64, from the consistent B1 latent export
(outputs/session18/exp_b1/latents_{jepa_d64,fukami_d64_noBN,pod_d64_noBN}). For
each family, toward the future wake enstrophy at H=16 on held-out test_b (targets
aligned from per_frame_targets by (case_id, encounter)):

  best single coordinate skill, full-latent (ridge) combination skill, the gap
  (= how distributed/collective the forecast code is), forecast-beyond-forces, and
  #coordinates with single skill > 0.5.

X is standardised per coordinate (fit on train) before the ridge so the
regularisation is comparable across families (POD coefficients are energy-scaled,
JEPA/Fukami are BatchNorm-unit). Spearman skills are scale-free. A Fukami 3-seed
robustness check (session24) is also reported.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

H = 16
PF = "outputs/session16/exp2/per_frame_targets/{}.npz"
B1 = "outputs/session18/exp_b1/latents_{}/{}.npz"
FUKSEED = "outputs/session24/seed_robustness_wake/fukami_latents_seed{}/{}.npz"
FAMDIR = {"JEPA": "jepa_d64", "Fukami": "fukami_d64_noBN", "POD": "pod_d64_noBN"}


def partial(a, b, ctrl):
    ra, rb = rankdata(a), rankdata(b)
    RC = np.column_stack([np.ones(len(a))] + [rankdata(ctrl[:, j]) for j in range(ctrl.shape[1])])
    res = lambda r: r - RC @ np.linalg.lstsq(RC, r, rcond=None)[0]
    return abs(float(np.corrcoef(res(ra), res(rb))[0, 1]))


def targets_map(split):
    d = np.load(PF.format(split), allow_pickle=True)
    cid = np.asarray(d["case_id"]); enc = np.asarray(d["encounter_index"], int)
    g = lambda k: np.asarray(d[k], float)
    return {(str(cid[i]), int(enc[i])): {
        "wake": g("wake_enstrophy")[i], "C_L": g("C_L")[i], "C_D": g("C_D")[i],
        "G": g("G")[i]} for i in range(len(cid))}


def load_family(path):
    d = np.load(path, allow_pickle=True)
    cids = d["case_ids"] if "case_ids" in d.files else d["case_id"]
    encs = d["encounter_indices"] if "encounter_indices" in d.files else d["encounter_index"]
    return (np.asarray(d["z_full"], float), np.asarray(cids),
            np.asarray(encs, int), np.asarray(d["impact_frame"], int))


def build(z, cids, encs, imps, tmap):
    X, fw, F = [], [], []
    for i in range(len(cids)):
        key = (str(cids[i]), int(encs[i]))
        if key not in tmap:
            continue
        t = tmap[key]; imp = int(imps[i]); Tn = z.shape[1]
        for tt in range(imp, Tn - H):
            X.append(z[i, tt, :]); fw.append(t["wake"][tt + H])
            F.append([t["G"][tt], t["C_L"][tt], t["C_D"][tt]])
    return np.asarray(X), np.asarray(fw), np.asarray(F)


def analyse(trp, tep, tmap_tr, tmap_te):
    Xtr, fwtr, _ = build(*trp, tmap_tr)
    Xte, fwte, Fte = build(*tep, tmap_te)
    sc = StandardScaler().fit(Xtr)
    ytr = rankdata(fwtr); ytr = (ytr - ytr.mean()) / ytr.std()
    u = Ridge(alpha=1.0).fit(sc.transform(Xtr), ytr).coef_
    s = sc.transform(Xte) @ (u / np.linalg.norm(u))
    single = np.array([abs(spearmanr(Xte[:, k], fwte).statistic) for k in range(Xte.shape[1])])
    return {"combo": abs(spearmanr(s, fwte).statistic), "best_single": float(single.max()),
            "beyond_forces": partial(s, fwte, Fte), "n_strong": int((single > 0.5).sum())}


def main():
    tmap_tr, tmap_te = targets_map("train"), targets_map("test_b")
    out = {}
    print(f"{'family':<10}{'combo':>7}{'best1':>7}{'gap':>7}{'>0.5':>6}{'|forces':>9}")
    print("-" * 46)
    for fam, sub in FAMDIR.items():
        r = analyse(load_family(B1.format(sub, "train")), load_family(B1.format(sub, "test_b")),
                    tmap_tr, tmap_te)
        out[fam] = r
        print(f"{fam:<10}{r['combo']:>7.3f}{r['best_single']:>7.3f}"
              f"{r['combo']-r['best_single']:>7.3f}{r['n_strong']:>6}{r['beyond_forces']:>9.3f}")

    # Fukami 3-seed robustness (session24)
    seeds = [analyse(load_family(FUKSEED.format(s, "train")),
                     load_family(FUKSEED.format(s, "test_b")), tmap_tr, tmap_te) for s in (0, 1, 2)]
    out["Fukami_seeds"] = seeds
    cg = [s["combo"] for s in seeds]; gg = [s["combo"] - s["best_single"] for s in seeds]
    print(f"\n[robustness] Fukami 3 seeds: combo {np.mean(cg):.2f}+-{np.std(cg):.2f}, "
          f"gap {np.mean(gg):+.2f}+-{np.std(gg):.2f}")
    Path("outputs_causal/jepa_modes/cross_encoder3.json").write_text(json.dumps(out, indent=2))
    print("[done] wrote cross_encoder3.json")


if __name__ == "__main__":
    main()
