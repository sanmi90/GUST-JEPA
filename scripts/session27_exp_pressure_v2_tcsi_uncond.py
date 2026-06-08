"""UNCONDITIONED (tf-no-c) pressure-observability with TCSI optimal placement.

Session 27 copy of scripts/session21/exp_pressure_v2_tcsi.py, repointed to the
fully-unconditioned (no-c) end-to-end JEPA d=64 latent. Methodology is identical
to the production study (TCSI / qDEIM / uniform sensor selection, KernelRidge(RBF)
pressure->latent map on the flattened K-sensor x pre-impact-window vector,
impact-C_L routed through the latent), but:

  * latents come from outputs/session27/latents_own_jepa_noc_tf/{train,test_b,test_c}.npz
    (keys z_full (n,120,64), impact_frame, case_ids, encounter_indices, G, D, Y);
  * the impact-frame z is sliced from z_full per encounter;
  * wall-pressure windows are read from the v2 cache (the no-c latents are v2);
  * impact-frame C_L comes from the DNS metrics npz, aligned by (case_id, enc);
  * only the single unconditioned JEPA family is studied (no Fukami/POD columns);
  * every artifact is written under outputs/session27/pressure_noc_tf/ so nothing
    in outputs/session21 is touched and the production pressure_v2 fits are NOT
    reused (the pressure->latent KRR is refit on the unconditioned z here).

CPU only (KernelRidge / ridge). Reuses the verified Session 14 TCSI selection.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "session20"))
from session14_tcsi_pilot import (  # noqa: E402
    greedy_forward_selection, selector_qdeim, selector_uniform,
)
from exp_closure_r2 import match_index  # noqa: E402

# --- UNCONDITIONED repointing ---
LAT = REPO / "outputs/session27/latents_own_jepa_noc_tf"
DNS = np.load(REPO / "outputs/session17/exp2/dns_physical_metrics.npz", allow_pickle=True)
OUT = REPO / "outputs/session27/pressure_noc_tf"
PREVENT = Path(os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT")))
CACHE = Path(os.environ.get("VORTEX_JEPA_CACHE", str(PREVENT / "data/processed/vortex-jepa"))) / "v2"

K_LIST = [2, 4, 8, 16]
SPLITS = ["train", "test_b", "test_c"]
WINDOW = 30                     # pre-impact pressure window (matches production)
N_PRESSURE_FULL = 192

# Single unconditioned family. Tag mirrors the production "jepa_d64" so the
# downstream figure readers can pick it up unchanged.
FAMILIES = [("jepa_d64", "uncond", 64)]


def r2_score(yhat, y):
    yhat, y = np.asarray(yhat), np.asarray(y)
    ss_res = ((yhat - y) ** 2).sum(0)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0)
    return float(np.mean(1.0 - ss_res / np.maximum(ss_tot, 1e-12)))


def load_pressure_window(case_id: str, k: int, impact_frame: int) -> np.ndarray:
    """Pre-impact (WINDOW, 192) wall-pressure window from the v2 cache."""
    p = CACHE / case_id / f"encounter_{int(k):02d}.h5"
    with h5py.File(p, "r") as f:
        p_wall = np.asarray(f["p_wall"], dtype=np.float32)   # (120, 192)
    t_start = max(0, impact_frame - WINDOW)
    window = p_wall[t_start:impact_frame]
    if window.shape[0] < WINDOW:
        pad = np.zeros((WINDOW - window.shape[0], N_PRESSURE_FULL), dtype=np.float32)
        window = np.concatenate([pad, window], axis=0)
    return window                                            # (WINDOW, 192)


def gather_uncond(npz, split):
    """(X_pressure n x WINDOW x 192, z_impact n x d, npz). Unconditioned schema."""
    cids = npz["case_ids"].astype(str)
    eis = npz["encounter_indices"].astype(int)
    imp = npz["impact_frame"].astype(int)
    z_full = npz["z_full"]
    z = np.stack([z_full[i, int(imp[i])] for i in range(len(cids))], axis=0)
    X = np.zeros((len(cids), WINDOW, N_PRESSURE_FULL), dtype=np.float32)
    for i in range(len(cids)):
        X[i] = load_pressure_window(cids[i], int(eis[i]), int(imp[i]))
    return X.astype(np.float64), z.astype(np.float64), npz


def load_family(subdir):
    d = {}
    for sp in SPLITS:
        npz = np.load(LAT / f"{sp}.npz", allow_pickle=True)
        d[sp] = gather_uncond(npz, sp)
    return d


def impact_cl(npz, split):
    """True impact-frame C_L aligned to the npz encounter order (DNS metrics)."""
    cid = npz["case_ids"].astype(str)
    ei = npz["encounter_indices"].astype(int)
    imp = npz["impact_frame"].astype(int)
    di = match_index(cid, ei, DNS[f"{split}_case_id"], DNS[f"{split}_encounter_index"])
    cl = np.full(len(cid), np.nan)
    for i in range(len(cid)):
        if di[i] >= 0:
            cl[i] = DNS[f"{split}_C_L"][di[i], int(imp[i])]
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
    print("loading unconditioned family (pressure windows + latents)...", flush=True)
    fam = {tag: load_family(sub) for tag, sub, d in FAMILIES}

    # --- 1. derive TCSI on the unconditioned JEPA d=64 latent (target = first PC) ---
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
    json.dump({m: {str(K): v for K, v in dd.items()} for m, dd in picks.items()},
              open(OUT / "sensor_picks_v2.json", "w"), indent=0)

    # --- 2. method comparison: unconditioned JEPA latent recovery vs selector ---
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

    # --- 3. unconditioned recovery + physical C_L, with TCSI placement ---
    rows = []
    for tag, sub, d in FAMILIES:
        kind = "jepa"
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
                clhat_via = krr_pred(zcl, zhat).ravel()           # pressure->z->C_L
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
    with open(OUT / "pressure_obs_v2.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {OUT/'pressure_obs_v2.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
