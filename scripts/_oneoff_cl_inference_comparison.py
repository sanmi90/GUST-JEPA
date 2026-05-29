"""Compare two pipelines for inferring C_L at impact from pre-impact pressure:

  (A) Direct:        pressure(τ window) → C_L(impact)    [no latent, KRR]
  (B) Via baseline:  pressure(τ window) → ẑ(impact) → C_L(impact)
                      [pressure → z via KRR, then z → C_L via train-fit probe]

The question: does routing through each baseline's representation HELP or HURT
the inference of C_L at impact, compared to a model-free pressure regression?

Hypothesis:
- JEPA latents capture C_L-relevant structure → (B) ≈ (A)
- Fukami AE preserves enough for (B) ≈ (A) but with some loss
- POD coefficients are not C_L-aligned → (B) < (A)

If (B) beats (A) for any baseline at any τ, that means the baseline's
representation is a *useful inductive prior* for the inverse problem.

Output: outputs/session18/exp_b1_test3/cl_inference_comparison.csv
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

LEAD_TAUS = [30, 20, 10, 5, 2]
WINDOW = 10
K_SENSORS = 8
N_PRESSURE_FULL = 192

BASELINES = [
    ("jepa_d64",   "latents_jepa_d64",   64),
    ("jepa_d32",   "latents_jepa_d32",   32),
    ("fukami_d3",  "latents_fukami_d3",   3),
    ("fukami_d32", "latents_fukami_d32", 32),
    ("fukami_d64", "latents_fukami_d64", 64),
    ("pod_d16",    "latents_pod_d16",    16),
    ("pod_d32",    "latents_pod_d32",    32),
    ("pod_d64",    "latents_pod_d64",    64),
]


def r2(yhat, y, eps=1e-9):
    ss_res = ((yhat - y) ** 2).sum(0)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0)
    return 1.0 - ss_res / np.maximum(ss_tot, eps)


def select_sensors_evenly(K: int) -> list[int]:
    return list(np.linspace(0, N_PRESSURE_FULL - 1, K, dtype=int).tolist())


def load_pressure_and_cl(case_id, k, impact, lead_tau, window):
    """Returns (pressure_window (W, 192), C_L_at_impact)."""
    p = CACHE / case_id / f"encounter_{int(k):02d}.h5"
    with h5py.File(p, "r") as f:
        p_wall = np.asarray(f["p_wall"], dtype=np.float32)
        c_l = np.asarray(f["C_L"], dtype=np.float32)
    t_end = impact - lead_tau
    t_start = max(0, t_end - window)
    win = p_wall[t_start:t_end]
    if win.shape[0] < window:
        pad = np.zeros((window - win.shape[0], N_PRESSURE_FULL), dtype=np.float32)
        win = np.concatenate([pad, win], axis=0)
    return win, float(c_l[min(impact, len(c_l) - 1)])


def get_cid_ei(npz):
    cid_key = "case_id" if "case_id" in npz.files else "case_ids"
    ei_key  = "encounter_index" if "encounter_index" in npz.files else "encounter_indices"
    return npz[cid_key].astype(str), npz[ei_key].astype(int)


def get_impact(npz):
    if "impact_frame" in npz.files:
        return npz["impact_frame"].astype(int)
    cids, _ = get_cid_ei(npz)
    return np.full(len(cids), 40)


def get_z_impact(npz):
    """Returns impact-frame z (n, d)."""
    if "z" in npz.files:
        return npz["z"]
    z_full = npz["z_full"]
    impact = get_impact(npz)
    return np.stack([z_full[i, int(impact[i])] for i in range(len(impact))], axis=0)


def gather_for_baseline(latents_npz, lead_tau, window, sensor_idx):
    cids, eis = get_cid_ei(latents_npz)
    impact = get_impact(latents_npz)
    n = len(cids)
    X_pressure = np.zeros((n, window, len(sensor_idx)), dtype=np.float32)
    cl_target = np.zeros(n, dtype=np.float32)
    for i, (cid, ei, imp) in enumerate(zip(cids, eis, impact)):
        win, cl = load_pressure_and_cl(cid, int(ei), int(imp), lead_tau, window)
        X_pressure[i] = win[:, sensor_idx]
        cl_target[i] = cl
    return X_pressure.reshape(n, -1), cl_target


def fit_predict(X_tr, y_tr, X_te):
    sx = StandardScaler().fit(X_tr)
    m = KernelRidge(alpha=0.1, kernel="rbf", gamma=0.01)
    m.fit(sx.transform(X_tr), y_tr)
    return m.predict(sx.transform(X_te))


def main():
    sensor_idx = select_sensors_evenly(K_SENSORS)
    rows = []

    for tag, lat_subdir, d in BASELINES:
        lat_dir = REPO / "outputs/session18/exp_b1" / lat_subdir
        if not (lat_dir / "train.npz").exists():
            print(f"SKIP {tag}: train.npz missing"); continue
        try:
            train_npz = np.load(lat_dir / "train.npz", allow_pickle=True)
            test_b_npz = np.load(lat_dir / "test_b.npz", allow_pickle=True)
            test_c_npz = np.load(lat_dir / "test_c.npz", allow_pickle=True)
        except Exception as e:
            print(f"SKIP {tag}: {e}"); continue

        # Impact-frame z and C_L target for ALL splits (used by Pipeline B)
        z_tr_imp = get_z_impact(train_npz)
        z_tb_imp = get_z_impact(test_b_npz)
        z_tc_imp = get_z_impact(test_c_npz)

        # Get DNS C_L at impact for train (for fitting the z → C_L probe).
        # We can use the load_pressure_and_cl helper just to get the cl scalar,
        # with lead_tau=0 (i.e. the C_L value at the impact frame regardless of pressure).
        cids_tr, eis_tr = get_cid_ei(train_npz); imp_tr = get_impact(train_npz)
        cids_tb, eis_tb = get_cid_ei(test_b_npz); imp_tb = get_impact(test_b_npz)
        cids_tc, eis_tc = get_cid_ei(test_c_npz); imp_tc = get_impact(test_c_npz)
        cl_tr = np.array([load_pressure_and_cl(c, e, im, 0, 1)[1]
                          for c, e, im in zip(cids_tr, eis_tr, imp_tr)])
        cl_tb = np.array([load_pressure_and_cl(c, e, im, 0, 1)[1]
                          for c, e, im in zip(cids_tb, eis_tb, imp_tb)])
        cl_tc = np.array([load_pressure_and_cl(c, e, im, 0, 1)[1]
                          for c, e, im in zip(cids_tc, eis_tc, imp_tc)])

        # Fit the train-only z → C_L probe (KRR-RBF; same recipe as our other probes)
        sx_z = StandardScaler().fit(z_tr_imp)
        probe = KernelRidge(alpha=0.1, kernel="rbf", gamma=0.05)
        probe.fit(sx_z.transform(z_tr_imp), cl_tr)
        # As sanity, probe R² on test_b using ENCODED z (the "perfect" upper bound)
        cl_hat_tb_oracle = probe.predict(sx_z.transform(z_tb_imp))
        r2_tb_oracle = float(r2(cl_hat_tb_oracle[:, None], cl_tb[:, None])[0])
        cl_hat_tc_oracle = probe.predict(sx_z.transform(z_tc_imp))
        r2_tc_oracle = float(r2(cl_hat_tc_oracle[:, None], cl_tc[:, None])[0])

        print(f"\n=== {tag} (d={d}) ===")
        print(f"  z(true)→C_L probe (upper bound): test_b R²={r2_tb_oracle:+.3f}  test_c R²={r2_tc_oracle:+.3f}")

        for tau in LEAD_TAUS:
            # Pressure inputs at lead τ
            try:
                X_tr_p, _ = gather_for_baseline(train_npz, tau, WINDOW, sensor_idx)
                X_tb_p, _ = gather_for_baseline(test_b_npz, tau, WINDOW, sensor_idx)
                X_tc_p, _ = gather_for_baseline(test_c_npz, tau, WINDOW, sensor_idx)
            except Exception as e:
                print(f"  τ=-{tau}: FAIL gather: {e}"); continue

            # Pipeline A: direct pressure → C_L
            cl_hat_tb_A = fit_predict(X_tr_p, cl_tr, X_tb_p)
            cl_hat_tc_A = fit_predict(X_tr_p, cl_tr, X_tc_p)
            r2_tb_A = float(r2(cl_hat_tb_A[:, None], cl_tb[:, None])[0])
            r2_tc_A = float(r2(cl_hat_tc_A[:, None], cl_tc[:, None])[0])

            # Pipeline B: pressure → ẑ(impact) → C_L probe
            # Stage B-1: pressure → ẑ_impact (per-dim KRR)
            sy_z = StandardScaler().fit(z_tr_imp)
            sx_p = StandardScaler().fit(X_tr_p)
            m_pz = KernelRidge(alpha=0.1, kernel="rbf", gamma=0.01)
            m_pz.fit(sx_p.transform(X_tr_p), sy_z.transform(z_tr_imp))
            z_hat_tb = sy_z.inverse_transform(m_pz.predict(sx_p.transform(X_tb_p)))
            z_hat_tc = sy_z.inverse_transform(m_pz.predict(sx_p.transform(X_tc_p)))
            # Stage B-2: apply train-fit z→C_L probe on the estimated ẑ
            cl_hat_tb_B = probe.predict(sx_z.transform(z_hat_tb))
            cl_hat_tc_B = probe.predict(sx_z.transform(z_hat_tc))
            r2_tb_B = float(r2(cl_hat_tb_B[:, None], cl_tb[:, None])[0])
            r2_tc_B = float(r2(cl_hat_tc_B[:, None], cl_tc[:, None])[0])

            row = {
                "tag": tag, "d": d, "lead_tau": tau,
                "test_b_R2_oracle":       r2_tb_oracle,
                "test_b_R2_direct":       r2_tb_A,
                "test_b_R2_via_baseline": r2_tb_B,
                "test_c_R2_oracle":       r2_tc_oracle,
                "test_c_R2_direct":       r2_tc_A,
                "test_c_R2_via_baseline": r2_tc_B,
            }
            rows.append(row)
            winner_tb = "direct" if r2_tb_A > r2_tb_B else "via_baseline"
            print(f"  τ=-{tau:2d}  test_b: direct={r2_tb_A:+.3f}  via_baseline={r2_tb_B:+.3f}  "
                  f"({winner_tb} wins)   |  test_c: direct={r2_tc_A:+.3f}  via_baseline={r2_tc_B:+.3f}")

    out_csv = REPO / "outputs/session18/exp_b1_test3/cl_inference_comparison.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
