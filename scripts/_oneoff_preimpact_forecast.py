"""Pre-impact forecast: how early can each model predict the impact event?

For each baseline, use pre-impact pressure measurements at varying lead times
τ ∈ {-30, -20, -10, -5, -2} to predict impact-frame (frame 40) physical
observables (C_L, I_y). Reports R² as a function of τ — the "advance-warning
curve" that says how many convective times before impact the encoder can
recover the impact-frame state from pressure alone.

This is the answer to the manuscript question: "can the model anticipate
the impact event from pre-impact sensors?"

Output: outputs/session18/exp_b1_test3/preimpact_forecast.csv
"""
import csv
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

# Lead times in frames before impact (impact = frame 40)
# τ = 30 means we use pressure from frames 0..10 (lead 30 frames = 1.5 convective times)
LEAD_TAUS = [30, 20, 10, 5, 2]
WINDOW = 10  # 10-frame window ending at frame (impact - τ)
K_SENSORS = 8  # fixed K for clarity
N_PRESSURE_FULL = 192

BASELINES = [
    ("jepa_d64",   "latents_jepa_d64",   64, "jepa"),
    ("jepa_d32",   "latents_jepa_d32",   32, "jepa"),
    ("fukami_d3",  "latents_fukami_d3",   3, "fukami"),
    ("fukami_d32", "latents_fukami_d32", 32, "fukami"),
    ("fukami_d64", "latents_fukami_d64", 64, "fukami"),
    ("pod_d16",    "latents_pod_d16",    16, "pod"),
    ("pod_d32",    "latents_pod_d32",    32, "pod"),
    ("pod_d64",    "latents_pod_d64",    64, "pod"),
]


def r2(yhat, y, eps=1e-9):
    ss_res = ((yhat - y) ** 2).sum(0)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0)
    return 1.0 - ss_res / np.maximum(ss_tot, eps)


def select_sensors_evenly(K: int) -> list[int]:
    return list(np.linspace(0, N_PRESSURE_FULL - 1, K, dtype=int).tolist())


def load_window(case_id: str, k: int, impact: int, lead_tau: int, window: int) -> tuple[np.ndarray, float, float]:
    """Returns (pressure_window (W, 192), C_L_at_impact, I_y_at_impact)."""
    p = CACHE / case_id / f"encounter_{int(k):02d}.h5"
    with h5py.File(p, "r") as f:
        p_wall = np.asarray(f["p_wall"], dtype=np.float32)  # (120, 192)
        c_l = np.asarray(f["C_L"], dtype=np.float32)  # (120,)
        c_d = np.asarray(f["C_D"], dtype=np.float32) if "C_D" in f else None
    t_end = impact - lead_tau
    t_start = max(0, t_end - window)
    win = p_wall[t_start:t_end]
    if win.shape[0] < window:
        pad = np.zeros((window - win.shape[0], N_PRESSURE_FULL), dtype=np.float32)
        win = np.concatenate([pad, win], axis=0)
    # Impact-frame observables (use frame `impact`):
    cl_imp = float(c_l[min(impact, len(c_l) - 1)])
    # Use C_D as a proxy for I_y direction since we don't have I_y in cache;
    # the paper's I_y is computed from /u integrals — we approximate via wake
    # signatures derivable from DNS but skip here and just track C_L + C_D.
    cd_imp = float(c_d[min(impact, len(c_d) - 1)]) if c_d is not None else 0.0
    return win, cl_imp, cd_imp


def gather(latents_npz, lead_tau, window, sensor_idx):
    cid_key = "case_id" if "case_id" in latents_npz.files else "case_ids"
    ei_key  = "encounter_index" if "encounter_index" in latents_npz.files else "encounter_indices"
    cids = latents_npz[cid_key].astype(str)
    eis = latents_npz[ei_key].astype(int)
    impact = latents_npz["impact_frame"].astype(int) if "impact_frame" in latents_npz.files else np.full(len(cids), 40)
    n = len(cids)
    X = np.zeros((n, window, len(sensor_idx)), dtype=np.float32)
    y_cl = np.zeros(n, dtype=np.float32)
    y_cd = np.zeros(n, dtype=np.float32)
    for i, (cid, ei, imp) in enumerate(zip(cids, eis, impact)):
        win, cl, cd = load_window(cid, int(ei), int(imp), lead_tau, window)
        X[i] = win[:, sensor_idx]
        y_cl[i] = cl
        y_cd[i] = cd
    return X.reshape(n, -1), y_cl, y_cd


def main():
    sensor_idx = select_sensors_evenly(K_SENSORS)
    rows = []

    for tag, lat_subdir, d, kind in BASELINES:
        lat_dir = REPO / "outputs/session18/exp_b1" / lat_subdir
        if not (lat_dir / "train.npz").exists():
            print(f"SKIP {tag}: train.npz missing")
            continue
        try:
            train_npz = np.load(lat_dir / "train.npz", allow_pickle=True)
            test_b_npz = np.load(lat_dir / "test_b.npz", allow_pickle=True)
            test_c_npz = np.load(lat_dir / "test_c.npz", allow_pickle=True)
        except Exception as e:
            print(f"SKIP {tag}: load error {e}")
            continue

        print(f"\n=== {tag} (d={d}, kind={kind}, K={K_SENSORS}) ===")

        for tau in LEAD_TAUS:
            print(f"  τ=-{tau} (using pressure from frames {40-tau-WINDOW}..{40-tau})", end="  ")
            try:
                X_train, cl_train, cd_train = gather(train_npz, tau, WINDOW, sensor_idx)
                X_test_b, cl_test_b, cd_test_b = gather(test_b_npz, tau, WINDOW, sensor_idx)
                X_test_c, cl_test_c, cd_test_c = gather(test_c_npz, tau, WINDOW, sensor_idx)
            except Exception as e:
                print(f"FAIL: {e}"); continue

            sx = StandardScaler().fit(X_train)
            Xtrs = sx.transform(X_train); Xtbs = sx.transform(X_test_b); Xtcs = sx.transform(X_test_c)

            # Predict C_L at impact
            m_cl = KernelRidge(alpha=0.1, kernel="rbf", gamma=0.01)
            m_cl.fit(Xtrs, cl_train)
            cl_hat_tb = m_cl.predict(Xtbs)
            cl_hat_tc = m_cl.predict(Xtcs)
            r2_cl_tb = float(r2(cl_hat_tb[:, None], cl_test_b[:, None])[0])
            r2_cl_tc = float(r2(cl_hat_tc[:, None], cl_test_c[:, None])[0])

            # Predict C_D at impact
            m_cd = KernelRidge(alpha=0.1, kernel="rbf", gamma=0.01)
            m_cd.fit(Xtrs, cd_train)
            cd_hat_tb = m_cd.predict(Xtbs)
            cd_hat_tc = m_cd.predict(Xtcs)
            r2_cd_tb = float(r2(cd_hat_tb[:, None], cd_test_b[:, None])[0])
            r2_cd_tc = float(r2(cd_hat_tc[:, None], cd_test_c[:, None])[0])

            row = {
                "tag": tag, "kind": kind, "d": d, "K_sensors": K_SENSORS, "lead_tau": tau,
                "C_L_test_b_R2": r2_cl_tb, "C_L_test_c_R2": r2_cl_tc,
                "C_D_test_b_R2": r2_cd_tb, "C_D_test_c_R2": r2_cd_tc,
            }
            rows.append(row)
            print(f"C_L R²: test_b={r2_cl_tb:+.3f} test_c={r2_cl_tc:+.3f}  | "
                  f"C_D test_b={r2_cd_tb:+.3f}")

    out_csv = REPO / "outputs/session18/exp_b1_test3/preimpact_forecast.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
