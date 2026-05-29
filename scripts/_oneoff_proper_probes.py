"""Proper-rigor probe analysis: CV-tuned KRR + bootstrap CI + cross-seed.

Compares JEPA d=64 latents (4 seeds) vs POD d=64 coefficients on parameter
recovery (G, D, Y) at the impact frame. Uses train-only CV for hyperparameter
tuning, then bootstrap CIs on test_b and test_c.
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import warnings; warnings.filterwarnings("ignore")

REPO = Path("/home/carlos/GUST-JEPA")
sys.path.insert(0, str(REPO))
from src.data.omega_pipeline import OmegaPipeline  # noqa: E402

PIPE_PATH = REPO / "outputs/data_pipeline/v1/manifest.json"
SPLIT     = REPO / "configs/splits/split_v2.json"
CACHE     = Path("/home/carlos/PREVENT/data/processed/vortex-jepa/v1")
JEPA_LATENT_ROOT = REPO / "outputs/session17/seed_latents"
POD_BASIS = REPO / "outputs/session18/exp_b1/pod_d64/pod_basis.npz"
IMPACT = 40


def r2(yhat, y, eps=1e-9):
    """Per-column R²; returns array of shape (n_outputs,)."""
    ss_res = ((yhat - y) ** 2).sum(0)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0)
    return 1.0 - ss_res / np.maximum(ss_tot, eps)


def gather_pod_coeffs_one_split(manifest, split_name, pod_basis, pod_mean, pipe):
    """Project impact-frame omega for one split onto the POD basis. Returns
    (N, d) coefficients plus (G, D, Y) labels and case_id."""
    coeffs, G, D, Y, cids, eis = [], [], [], [], [], []
    for cid, case in manifest["cases"].items():
        if split_name == "train" and case["split"] == "train":
            ks = list(case["train_encounter_indices"])
        elif split_name == "test_a" and case["split"] == "train":
            ks = list(case.get("val_encounter_indices") or case["test_a_encounter_indices"])
        elif split_name == "test_b" and case["split"] == "test_b":
            ks = list(range(case["n_encounters_full"]))
        elif split_name == "test_c" and case["split"] == "test_c":
            ks = list(range(case["n_encounters_full"]))
        else:
            continue
        for k in ks:
            p = CACHE / cid / f"encounter_{int(k):02d}.h5"
            if not p.exists():
                continue
            with h5py.File(p, "r") as f:
                omega_raw = np.asarray(f["omega_z"][IMPACT], dtype=np.float32)  # (192, 96)
            # Pipeline: preprocess raw (mask + per-encounter clip) -> normalize
            omega_clean = pipe.preprocess_raw(omega_raw[None, ...], cid, int(k))[0]
            omega_norm = pipe.normalize(torch.from_numpy(omega_clean)).numpy()
            x = omega_norm.reshape(-1) - pod_mean
            c = pod_basis.T @ x  # (d,)
            coeffs.append(c.astype(np.float32))
            G.append(float(case["G"]))
            D.append(float(case["D"]))
            Y.append(float(case["Y"]))
            cids.append(cid); eis.append(int(k))
    return (np.array(coeffs), np.array(G), np.array(D), np.array(Y),
            np.array(cids), np.array(eis))


def get_jepa_split(seed_dir, split_name):
    """Load JEPA latents from a seed directory. Returns (z_impact, G, D, Y, cids)."""
    npz = np.load(seed_dir / f"{split_name}.npz", allow_pickle=True)
    if "z_full" in npz.files and npz["z_full"].ndim == 3:
        z = npz["z_full"][:, IMPACT, :]
    else:
        z = npz["z"]
    G = npz["G"].astype(np.float32)
    D = npz["D"].astype(np.float32)
    Y = npz["Y"].astype(np.float32)
    cids = npz["case_id"].astype(str)
    return z, G, D, Y, cids


def cv_tune_krr(X, Y, alphas=(0.01, 0.1, 1.0, 10.0), gammas=(0.005, 0.05, 0.5, 5.0),
                n_folds=5):
    """5-fold k-fold CV on X,Y (already standardized). Returns best (alpha, gamma)
    by mean validation R² (averaged across the 3 output dims and 5 folds).
    """
    n = len(X)
    fold = np.repeat(np.arange(n_folds), int(np.ceil(n / n_folds)))[:n]
    rng = np.random.default_rng(0)
    rng.shuffle(fold)
    best = (-np.inf, None)
    for a in alphas:
        for g in gammas:
            r2s = []
            for f in range(n_folds):
                m = KernelRidge(alpha=a, kernel='rbf', gamma=g)
                tr = fold != f; va = fold == f
                m.fit(X[tr], Y[tr])
                yhat = m.predict(X[va])
                r2_arr = r2(yhat, Y[va])
                r2s.append(r2_arr.mean())
            mean_r2 = np.mean(r2s)
            if mean_r2 > best[0]:
                best = (mean_r2, (a, g))
    return best  # (best_cv_R², (a, g))


def cv_tune_ridge(X, Y, alphas=(0.01, 0.1, 1.0, 10.0, 100.0), n_folds=5):
    n = len(X)
    fold = np.repeat(np.arange(n_folds), int(np.ceil(n / n_folds)))[:n]
    rng = np.random.default_rng(0)
    rng.shuffle(fold)
    best = (-np.inf, None)
    for a in alphas:
        r2s = []
        for f in range(n_folds):
            m = Ridge(alpha=a)
            tr = fold != f; va = fold == f
            m.fit(X[tr], Y[tr])
            yhat = m.predict(X[va])
            r2s.append(r2(yhat, Y[va]).mean())
        mean_r2 = np.mean(r2s)
        if mean_r2 > best[0]:
            best = (mean_r2, a)
    return best


def bootstrap_r2(yhat_test, y_test, n_boot=2000, seed=0):
    """Bootstrap the per-output R² with replacement on test encounters. Returns
    (mean, 2.5pct, 97.5pct) for each output column."""
    rng = np.random.default_rng(seed)
    n = len(y_test)
    r2_samples = np.zeros((n_boot, y_test.shape[1]), dtype=np.float32)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ys = y_test[idx]; yhs = yhat_test[idx]
        r2_samples[i] = r2(yhs, ys)
    mean = r2_samples.mean(0)
    lo = np.percentile(r2_samples, 2.5, axis=0)
    hi = np.percentile(r2_samples, 97.5, axis=0)
    return mean, lo, hi


def evaluate_encoder(X_tr, Y_tr, splits_tagged, label, tune_krr=True):
    """For one encoder (= JEPA seed or POD basis), CV-tune Ridge & KRR on train,
    then bootstrap-CI on each test split. Returns nested dict of results."""
    sx = StandardScaler().fit(X_tr); sy = StandardScaler().fit(Y_tr)
    Xtrs = sx.transform(X_tr); Ytrs = sy.transform(Y_tr)

    # CV-tune
    ridge_cv, ridge_alpha = cv_tune_ridge(Xtrs, Ytrs)
    if tune_krr:
        krr_cv, (krr_alpha, krr_gamma) = cv_tune_krr(Xtrs, Ytrs)
    else:
        krr_alpha, krr_gamma = 0.1, 0.05; krr_cv = np.nan
    ridge = Ridge(alpha=ridge_alpha).fit(Xtrs, Ytrs)
    krr = KernelRidge(alpha=krr_alpha, kernel='rbf', gamma=krr_gamma).fit(Xtrs, Ytrs)

    result = {
        "label": label,
        "ridge_alpha": ridge_alpha, "ridge_train_cv_R²": float(ridge_cv),
        "krr_alpha": krr_alpha, "krr_gamma": krr_gamma, "krr_train_cv_R²": float(krr_cv),
    }
    for sp_name, (X_te, Y_te) in splits_tagged.items():
        Xtes = sx.transform(X_te)
        yhat_r = sy.inverse_transform(ridge.predict(Xtes))
        yhat_k = sy.inverse_transform(krr.predict(Xtes))
        r_mean, r_lo, r_hi = bootstrap_r2(yhat_r, Y_te)
        k_mean, k_lo, k_hi = bootstrap_r2(yhat_k, Y_te)
        result[sp_name] = {
            "n": len(Y_te),
            "ridge_R²_pointmean":  [float(r_mean[i]) for i in range(3)],
            "ridge_R²_lo":         [float(r_lo[i])   for i in range(3)],
            "ridge_R²_hi":         [float(r_hi[i])   for i in range(3)],
            "krr_R²_pointmean":    [float(k_mean[i]) for i in range(3)],
            "krr_R²_lo":           [float(k_lo[i])   for i in range(3)],
            "krr_R²_hi":           [float(k_hi[i])   for i in range(3)],
        }
    return result


def main():
    print("Loading manifests...", flush=True)
    with open(SPLIT) as f:
        manifest = json.load(f)
    pipe = OmegaPipeline.from_manifest(PIPE_PATH)

    # ---- Build JEPA dataset (4 seeds × 4 splits) ----
    print("Loading JEPA latents (4 seeds)...", flush=True)
    jepa_seeds = {}
    for seed in ["production", "seed0", "seed1", "seed2"]:
        seed_dir = JEPA_LATENT_ROOT / seed
        d = {}
        for sp in ["train", "test_b", "test_c"]:
            z, G, D, Y, cids = get_jepa_split(seed_dir, sp)
            d[sp] = {"X": z, "G": G, "D": D, "Y": Y, "cid": cids}
        jepa_seeds[seed] = d
        print(f"  {seed}: train n={d['train']['X'].shape[0]}, test_b={d['test_b']['X'].shape[0]}, test_c={d['test_c']['X'].shape[0]}")

    # ---- Build POD dataset (1 basis, projected impact-frame coeffs) ----
    print("Loading POD basis + projecting impact-frame coeffs...", flush=True)
    pod_npz = np.load(POD_BASIS, allow_pickle=True)
    pod_basis = pod_npz["Phi"]; pod_mean = pod_npz["mean"]
    pod_data = {}
    for sp in ["train", "test_b", "test_c"]:
        c, G, D, Y, cid, ei = gather_pod_coeffs_one_split(manifest, sp, pod_basis, pod_mean, pipe)
        pod_data[sp] = {"X": c, "G": G, "D": D, "Y": Y, "cid": cid}
        print(f"  POD {sp}: n={c.shape[0]}, coeffs dim={c.shape[1]}")

    # ---- Run probes ----
    YCOLS = ["G", "D", "Y"]
    results = {"jepa_seeds": {}, "pod": None}

    print("\n=== JEPA d=64 ===", flush=True)
    for seed in ["production", "seed0", "seed1", "seed2"]:
        d = jepa_seeds[seed]
        X_tr = d["train"]["X"]
        Y_tr = np.stack([d["train"]["G"], d["train"]["D"], d["train"]["Y"]], axis=1)
        splits_tagged = {
            "test_b": (d["test_b"]["X"],
                       np.stack([d["test_b"]["G"], d["test_b"]["D"], d["test_b"]["Y"]], axis=1)),
            "test_c": (d["test_c"]["X"],
                       np.stack([d["test_c"]["G"], d["test_c"]["D"], d["test_c"]["Y"]], axis=1)),
        }
        res = evaluate_encoder(X_tr, Y_tr, splits_tagged, f"JEPA-{seed}")
        results["jepa_seeds"][seed] = res
        # Compact print
        kY = res["test_b"]["krr_R²_pointmean"][2]
        kY_lo = res["test_b"]["krr_R²_lo"][2]; kY_hi = res["test_b"]["krr_R²_hi"][2]
        print(f"  {seed:<11}  Ridge α*={res['ridge_alpha']:>6.2f}  KRR α*={res['krr_alpha']:.2f} γ*={res['krr_gamma']:.3f}  "
              f"KRR test_b Y R²={kY:+.3f} [{kY_lo:+.3f}, {kY_hi:+.3f}]")

    print("\n=== POD d=64 (single basis, no seeds) ===", flush=True)
    X_tr = pod_data["train"]["X"]
    Y_tr = np.stack([pod_data["train"]["G"], pod_data["train"]["D"], pod_data["train"]["Y"]], axis=1)
    splits_tagged = {
        "test_b": (pod_data["test_b"]["X"],
                   np.stack([pod_data["test_b"]["G"], pod_data["test_b"]["D"], pod_data["test_b"]["Y"]], axis=1)),
        "test_c": (pod_data["test_c"]["X"],
                   np.stack([pod_data["test_c"]["G"], pod_data["test_c"]["D"], pod_data["test_c"]["Y"]], axis=1)),
    }
    pod_res = evaluate_encoder(X_tr, Y_tr, splits_tagged, "POD")
    results["pod"] = pod_res
    pkY = pod_res["test_b"]["krr_R²_pointmean"][2]
    pkY_lo = pod_res["test_b"]["krr_R²_lo"][2]; pkY_hi = pod_res["test_b"]["krr_R²_hi"][2]
    print(f"  POD       Ridge α*={pod_res['ridge_alpha']:>6.2f}  KRR α*={pod_res['krr_alpha']:.2f} γ*={pod_res['krr_gamma']:.3f}  "
          f"KRR test_b Y R²={pkY:+.3f} [{pkY_lo:+.3f}, {pkY_hi:+.3f}]")

    # ---- Final summary tables ----
    print("\n" + "="*80)
    print("FINAL: CV-tuned probes with bootstrap CI (n=2000) and cross-seed std")
    print("="*80)

    for probe_kind in ["ridge", "krr"]:
        for split in ["test_b", "test_c"]:
            print(f"\n— {probe_kind.upper()} on {split} —")
            print(f"  {'encoder':<14}  G               D               Y")
            # JEPA: 4-seed mean ± std + bootstrap CI on mean R²
            for col_idx, col in enumerate(YCOLS):
                pass  # handled below
            jepa_pointmeans = {col: [] for col in YCOLS}
            for seed in ["production", "seed0", "seed1", "seed2"]:
                rs = results["jepa_seeds"][seed][split][f"{probe_kind}_R²_pointmean"]
                for j, col in enumerate(YCOLS):
                    jepa_pointmeans[col].append(rs[j])

            line = f"  {'JEPA (4-seed)':<14}  "
            for col_idx, col in enumerate(YCOLS):
                vals = np.array(jepa_pointmeans[col])
                if vals.std() > 100 or np.any(np.abs(vals) > 100):
                    line += f"{'degenerate':>15}  "
                else:
                    line += f"{vals.mean():+.3f} ± {vals.std():.3f}  "
            print(line)

            line = f"  {'POD':<14}  "
            for col_idx, col in enumerate(YCOLS):
                pm = pod_res[split][f"{probe_kind}_R²_pointmean"][col_idx]
                lo = pod_res[split][f"{probe_kind}_R²_lo"][col_idx]
                hi = pod_res[split][f"{probe_kind}_R²_hi"][col_idx]
                if abs(pm) > 100:
                    line += f"{'degenerate':>15}  "
                else:
                    line += f"{pm:+.3f} [{lo:+.2f},{hi:+.2f}]  "
            print(line)

    # Write json for traceability
    out_json = REPO / "outputs/session18/exp_b1/proper_probes_v2.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {out_json}")


if __name__ == "__main__":
    main()
