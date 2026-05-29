"""Baseline pressure observability: pressure (K sensors × W window) → baseline_z.

Mirrors scripts/session17/exp5_nonlinear.py but applied to ALL B1 baselines
(Fukami AE × 3 + POD × 3 + JEPA d=32 + JEPA d=64) — answers the question
"is JEPA's pressure-observability a representation-specific advantage or just
physical observability of the gust state?"

For each baseline, for each K in {2, 4, 8, 16}, fit pressure_window → z_impact
using KernelRidge(RBF) (fast + competitive with TCN/MLP on this task). Reports
test_b and test_c R² of z, plus probe-driven R² of (G, D, Y).

Output: outputs/session18/exp_b1_test3/baseline_pressure_observability.csv
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
import warnings; warnings.filterwarnings("ignore")

REPO = Path("/home/carlos/GUST-JEPA")
sys.path.insert(0, str(REPO))
CACHE = Path("/home/carlos/PREVENT/data/processed/vortex-jepa/v1")

# Pre-impact pressure window (matches exp5_nonlinear.py convention)
WINDOW = 30
N_PRESSURE_FULL = 192

# K values to sweep (number of sensors)
K_LIST = [2, 4, 8, 16]

# Baselines to analyze (tag → latents-dir, d, kind)
BASELINES = [
    ("jepa_d64",   "latents_jepa_d64",     64, "jepa"),
    ("jepa_d32",   "latents_jepa_d32",     32, "jepa"),
    ("fukami_d3",  "latents_fukami_d3",     3, "fukami"),
    ("fukami_d32", "latents_fukami_d32",   32, "fukami"),
    ("fukami_d64", "latents_fukami_d64",   64, "fukami"),
    ("pod_d16",    "latents_pod_d16",      16, "pod"),
    ("pod_d32",    "latents_pod_d32",      32, "pod"),
    ("pod_d64",    "latents_pod_d64",      64, "pod"),
]


def r2(yhat, y):
    ss_res = ((yhat - y) ** 2).sum(0)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-9)


def load_pressure_window(case_id: str, k: int, impact_frame: int) -> np.ndarray:
    """Load pre-impact (WINDOW, 192) pressure window for one encounter."""
    p = CACHE / case_id / f"encounter_{int(k):02d}.h5"
    with h5py.File(p, "r") as f:
        p_wall = np.asarray(f["p_wall"], dtype=np.float32)  # (120, 192)
    t_start = max(0, impact_frame - WINDOW)
    window = p_wall[t_start:impact_frame]
    if window.shape[0] < WINDOW:
        pad = np.zeros((WINDOW - window.shape[0], N_PRESSURE_FULL), dtype=np.float32)
        window = np.concatenate([pad, window], axis=0)
    return window  # (WINDOW, 192)


def gather_pressure_and_z(latents_npz: dict, split_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X_pressure: n × WINDOW × 192, z_impact: n × d, GDY: n × 3).
    Handles both JEPA schema (case_id, z) and baseline schema (case_ids, no z)."""
    cid_key = "case_id" if "case_id" in latents_npz.files else "case_ids"
    ei_key  = "encounter_index" if "encounter_index" in latents_npz.files else "encounter_indices"
    cids = latents_npz[cid_key].astype(str)
    eis = latents_npz[ei_key].astype(int)
    impact = latents_npz["impact_frame"].astype(int) if "impact_frame" in latents_npz.files else np.full(len(cids), 40)
    if "z" in latents_npz.files:
        z = latents_npz["z"]  # impact-frame z (n, d)
    else:
        # Extract impact-frame z from z_full
        z_full = latents_npz["z_full"]
        z = np.stack([z_full[i, int(impact[i])] for i in range(len(cids))], axis=0)
    GDY = np.stack([latents_npz["G"], latents_npz["D"], latents_npz["Y"]], axis=1)

    X_pressure = np.zeros((len(cids), WINDOW, N_PRESSURE_FULL), dtype=np.float32)
    for i, (cid, ei, imp) in enumerate(zip(cids, eis, impact)):
        X_pressure[i] = load_pressure_window(cid, int(ei), int(imp))
    return X_pressure, z, GDY


def select_sensors_evenly(K: int) -> list[int]:
    """K evenly-spaced sensor indices around the airfoil (0..192)."""
    return list((np.linspace(0, N_PRESSURE_FULL - 1, K, dtype=int)).tolist())


def fit_and_eval(X_tr_flat, z_tr, X_te_flat, z_te):
    """KernelRidge(RBF) on pressure_flat → z. Returns (yhat_test, R² per dim)."""
    sx = StandardScaler().fit(X_tr_flat); sy = StandardScaler().fit(z_tr)
    Xtrs = sx.transform(X_tr_flat); ztrs = sy.transform(z_tr)
    m = KernelRidge(alpha=0.1, kernel="rbf", gamma=0.01)
    m.fit(Xtrs, ztrs)
    yhat = sy.inverse_transform(m.predict(sx.transform(X_te_flat)))
    return yhat, r2(yhat, z_te)


def probe_gdy(z_train, gdy_train, z_test, gdy_test):
    """Linear probe z → (G, D, Y) on train, eval on test."""
    sx = StandardScaler().fit(z_train); sy = StandardScaler().fit(gdy_train)
    Xtrs = sx.transform(z_train); Ytrs = sy.transform(gdy_train)
    m = KernelRidge(alpha=0.1, kernel="rbf", gamma=0.05)
    m.fit(Xtrs, Ytrs)
    yhat = sy.inverse_transform(m.predict(sx.transform(z_test)))
    return r2(yhat, gdy_test)


def main():
    print("Loading pressure windows and z_impact for each baseline...", flush=True)
    rows = []
    for tag, lat_subdir, d, kind in BASELINES:
        lat_dir = REPO / "outputs/session18/exp_b1" / lat_subdir
        if not (lat_dir / "train.npz").exists():
            print(f"  SKIP {tag}: {lat_dir / 'train.npz'} missing")
            continue
        try:
            train_npz = np.load(lat_dir / "train.npz", allow_pickle=True)
            test_b_npz = np.load(lat_dir / "test_b.npz", allow_pickle=True)
            test_c_npz = np.load(lat_dir / "test_c.npz", allow_pickle=True)
        except Exception as e:
            print(f"  SKIP {tag}: load error {e}")
            continue

        print(f"\n=== {tag} (d={d}, kind={kind}) ===", flush=True)
        try:
            X_train_full, z_train, gdy_train = gather_pressure_and_z(train_npz, "train")
            X_test_b_full, z_test_b, gdy_test_b = gather_pressure_and_z(test_b_npz, "test_b")
            X_test_c_full, z_test_c, gdy_test_c = gather_pressure_and_z(test_c_npz, "test_c")
        except Exception as e:
            print(f"  FAIL gather {tag}: {e}")
            continue
        print(f"  shapes: train pressure {X_train_full.shape}, z {z_train.shape}")

        for K in K_LIST:
            sensors = select_sensors_evenly(K)
            X_tr_K = X_train_full[:, :, sensors]   # (n_tr, WINDOW, K)
            X_tb_K = X_test_b_full[:, :, sensors]
            X_tc_K = X_test_c_full[:, :, sensors]
            X_tr_flat = X_tr_K.reshape(len(X_tr_K), -1)  # (n_tr, WINDOW*K)
            X_tb_flat = X_tb_K.reshape(len(X_tb_K), -1)
            X_tc_flat = X_tc_K.reshape(len(X_tc_K), -1)

            # Fit pressure → z
            yhat_b, r2_z_test_b = fit_and_eval(X_tr_flat, z_train, X_tb_flat, z_test_b)
            yhat_c, r2_z_test_c = fit_and_eval(X_tr_flat, z_train, X_tc_flat, z_test_c)

            # Probe estimated z → (G, D, Y) on test
            # Use train_z (true) → gdy_train as probe, apply on estimated test z
            # (i.e., probe trained on true encoded latents, applied to pressure-estimated)
            gdy_b_r2 = probe_gdy(z_train, gdy_train, yhat_b, gdy_test_b)
            gdy_c_r2 = probe_gdy(z_train, gdy_train, yhat_c, gdy_test_c)

            row = {
                "tag": tag, "kind": kind, "d": d, "K": K,
                "test_b_R2_z_mean": float(np.mean(r2_z_test_b)),
                "test_c_R2_z_mean": float(np.mean(r2_z_test_c)),
                "test_b_G_R2": float(gdy_b_r2[0]),
                "test_b_D_R2": float(gdy_b_r2[1]),
                "test_b_Y_R2": float(gdy_b_r2[2]),
                "test_c_G_R2": float(gdy_c_r2[0]),
                "test_c_D_R2": float(gdy_c_r2[1]),
                "test_c_Y_R2": float(gdy_c_r2[2]),
            }
            rows.append(row)
            print(f"  K={K:2d}: test_b R²(z)={row['test_b_R2_z_mean']:+.3f}  "
                  f"G={row['test_b_G_R2']:+.3f}  D={row['test_b_D_R2']:+.3f}  Y={row['test_b_Y_R2']:+.3f}")

    out_csv = REPO / "outputs/session18/exp_b1_test3/baseline_pressure_observability.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(out_csv, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
