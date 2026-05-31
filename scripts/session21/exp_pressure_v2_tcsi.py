"""v2 pressure-observability with TCSI optimal sensor placement.

Redo of the deployment study on the production (v2, noBN) latents that the main
results use, fixing two problems in the old appendix: (1) it used evenly-spaced
sensors instead of the optimal target-conditioned (TCSI) placement we computed,
and (2) it reported only R2. Here we:

  * re-derive the TCSI greedy sensor selection on the v2 JEPA d=64 latent
    (target = its first principal component), plus qDEIM and uniform baselines;
  * apply the one optimal array to recover every family's latent from K=2/4/8/16
    wall-pressure taps (JEPA 64/32, Fukami 3/32/64, POD 16/32/64);
  * report latent R2 AND the physical estimate quality: impact C_L recovered
    through each family's latent, as MAE in C_L units and R2, plus a direct
    pressure->C_L baseline.

CPU only (KernelRidge / ridge). Reuses the verified Session 14 TCSI selection and
the pressure-window loader.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "session20"))
from _oneoff_baseline_pressure_obs import gather_pressure_and_z  # noqa: E402
from session14_tcsi_pilot import (  # noqa: E402
    greedy_forward_selection, selector_qdeim, selector_uniform,
)
from exp_closure_r2 import match_index  # noqa: E402

LAT = REPO / "outputs/session18/exp_b1"
DNS = np.load(REPO / "outputs/session17/exp2/dns_physical_metrics.npz", allow_pickle=True)
OUT = REPO / "outputs/session21/pressure_v2"
K_LIST = [2, 4, 8, 16]
SPLITS = ["train", "test_b", "test_c"]

FAMILIES = [
    ("jepa_d64", "latents_jepa_d64_test1_noBN", 64, "jepa"),
    ("jepa_d32", "latents_jepa_d32_noBN", 32, "jepa"),
    ("fukami_d3", "latents_fukami_d3_noBN", 3, "fukami"),
    ("fukami_d32", "latents_fukami_d32_noBN", 32, "fukami"),
    ("fukami_d64", "latents_fukami_d64_noBN", 64, "fukami"),
    ("pod_d16", "latents_pod_d16_noBN", 16, "pod"),
    ("pod_d32", "latents_pod_d32_noBN", 32, "pod"),
    ("pod_d64", "latents_pod_d64_noBN", 64, "pod"),
]


def r2_score(yhat, y):
    yhat, y = np.asarray(yhat), np.asarray(y)
    ss_res = ((yhat - y) ** 2).sum(0)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0)
    return float(np.mean(1.0 - ss_res / np.maximum(ss_tot, 1e-12)))


def load_family(subdir):
    """Per-split (X_pressure n×W×192, z n×d, npz)."""
    d = {}
    for sp in SPLITS:
        npz = np.load(LAT / subdir / f"{sp}.npz", allow_pickle=True)
        X, z, _ = gather_pressure_and_z(npz, sp)
        d[sp] = (X.astype(np.float64), z.astype(np.float64), npz)
    return d


def impact_cl(npz, split):
    """True impact-frame C_L aligned to the family npz encounter order."""
    cid = (npz["case_id"] if "case_id" in npz.files else npz["case_ids"]).astype(str)
    ei = (npz["encounter_index"] if "encounter_index" in npz.files
          else npz["encounter_indices"]).astype(int)
    imp = npz["impact_frame"].astype(int) if "impact_frame" in npz.files else None
    di = match_index(cid, ei, DNS[f"{split}_case_id"], DNS[f"{split}_encounter_index"])
    cl = np.full(len(cid), np.nan)
    for i in range(len(cid)):
        if di[i] >= 0:
            fr = int(imp[i]) if imp is not None else int(DNS[f"{split}_impact_frame"][di[i]])
            cl[i] = DNS[f"{split}_C_L"][di[i], fr]
    return cl


def krr_fit(Xtr, ytr):
    sx = StandardScaler().fit(Xtr)
    sy = StandardScaler().fit(ytr.reshape(len(ytr), -1))
    m = KernelRidge(alpha=0.1, kernel="rbf", gamma=0.01)
    m.fit(sx.transform(Xtr), sy.transform(ytr.reshape(len(ytr), -1)))
    return (m, sx, sy)


def krr_pred(model, X):
    m, sx, sy = model
    return sy.inverse_transform(m.predict(sx.transform(X)))


def feats(X, sensors):
    return X[:, :, sensors].reshape(len(X), -1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("loading families (pressure windows + latents)...", flush=True)
    fam = {tag: load_family(sub) for tag, sub, d, k in FAMILIES}

    # --- 1. derive TCSI on v2 JEPA d=64 (target = first PC of its latent) ---
    Xtr_j, ztr_j, _ = fam["jepa_d64"]["train"]
    target = PCA(n_components=1).fit_transform(ztr_j).ravel()
    Xw = Xtr_j.transpose(0, 2, 1)                      # (n, 192, W)
    tcsi16 = greedy_forward_selection(Xw, target, K=16)
    snap = Xtr_j.mean(axis=1)                          # (n, 192) window-mean snapshot
    qdeim16 = selector_qdeim(snap, 16)
    picks = {
        "TCSI": {K: tcsi16[:K] for K in K_LIST},
        "qDEIM": {K: qdeim16[:K] for K in K_LIST},
        "uniform": {K: selector_uniform(K) for K in K_LIST},
    }
    print("TCSI picks:", picks["TCSI"])
    json.dump({m: {str(K): v for K, v in d.items()} for m, d in picks.items()},
              open(OUT / "sensor_picks_v2.json", "w"), indent=0)

    # --- 2. method comparison: JEPA d=64 latent recovery vs selector ---
    method_rows = []
    for method, perK in picks.items():
        for K in K_LIST:
            s = perK[K]
            mdl = krr_fit(feats(Xtr_j, s), ztr_j)
            for sp in ("test_b", "test_c"):
                Xte, zte, _ = fam["jepa_d64"][sp]
                method_rows.append(dict(method=method, K=K, split=sp,
                                        R2_z=r2_score(krr_pred(mdl, feats(Xte, s)), zte)))
    json.dump(method_rows, open(OUT / "method_comparison_v2.json", "w"), indent=0)

    # --- 3. cross-family recovery + physical C_L, with TCSI placement ---
    rows = []
    for tag, sub, d, kind in FAMILIES:
        Xtr, ztr, ntr = fam[tag]["train"]
        cl_tr = impact_cl(ntr, "train")
        zcl = krr_fit(ztr, cl_tr)                       # latent -> C_L probe (train)
        for K in K_LIST:
            s = picks["TCSI"][K]
            pz = krr_fit(feats(Xtr, s), ztr)            # pressure -> z
            pcl = krr_fit(feats(Xtr, s), cl_tr)         # pressure -> C_L (direct)
            for sp in ("test_b", "test_c"):
                Xte, zte, nte = fam[tag][sp]
                cl_te = impact_cl(nte, sp)
                zhat = krr_pred(pz, feats(Xte, s))
                clhat_via = krr_pred(zcl, zhat).ravel()        # pressure->z->C_L
                clhat_dir = krr_pred(pcl, feats(Xte, s)).ravel()  # pressure->C_L direct
                rows.append(dict(
                    tag=tag, kind=kind, d=d, K=K, split=sp,
                    R2_z=r2_score(zhat, zte),
                    cl_mae_via=float(np.nanmean(np.abs(clhat_via - cl_te))),
                    cl_r2_via=r2_score(clhat_via[~np.isnan(cl_te)], cl_te[~np.isnan(cl_te)]),
                    cl_mae_direct=float(np.nanmean(np.abs(clhat_dir - cl_te))),
                    cl_std=float(np.nanstd(cl_te)),
                ))
            print(f"  {tag} K={K}: test_b R2_z={rows[-2]['R2_z']:+.3f} "
                  f"C_L MAE(via)={rows[-2]['cl_mae_via']:.3f}", flush=True)
    import csv
    with open(OUT / "pressure_obs_v2.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {OUT/'pressure_obs_v2.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
