"""Session 16, Experiment 1, Part (a) DIAGNOSTICS.

Run after exp1a_pls_base.py if the acceptance gate fails. This script answers
"why did PLS-3 fail" without retuning the recipe -- it characterises the
encoder geometry that PLS-3 has to live with.

Diagnostics produced:

    1. PCA spectrum of train impact-frame z (cumulative variance for k=1..16).
    2. Per-parameter ridge regression baselines (G, D, Y separately) with
       multiple alpha values on train/test_a/test_b/test_c, to show what a
       more flexible linear map recovers from the same 64-D z.
    3. PLS n_components sweep (k=1..8) on the same data, to show the R^2
       curve and locate the elbow.
    4. Symmetry diagnostic for Y: regress |Y| from z and compare to Y.
       If R^2(|Y|) >> R^2(Y), the encoder is symmetry-confused on the
       lateral offset axis.
    5. Per-axis breakdown of the dominant z PCs vs (G, D, Y) correlations.

These are *characterisations* of the existing encoder, not hyperparameter
tuning to rescue the result. The PLS-3 result remains the recipe-locked
artefact in pls_base.json.

Output: outputs/session16/exp1/pls_base_diagnostics.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


REPO = Path(__file__).resolve().parents[2]
LATENTS = REPO / "outputs" / "session14" / "latents" / "S12_E_d64"
OUT = REPO / "outputs" / "session16" / "exp1"
OUT.mkdir(parents=True, exist_ok=True)


def load_split(name: str) -> dict:
    d = np.load(LATENTS / f"{name}.npz", allow_pickle=True)
    return {
        "z": d["z"].astype(np.float64),
        "G": d["G"].astype(np.float64),
        "D": d["D"].astype(np.float64),
        "Y": d["Y"].astype(np.float64),
    }


def per_param_r2(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    names = ("G", "D", "Y")
    return {
        n: float(r2_score(y_true[:, i], y_pred[:, i]))
        for i, n in enumerate(names)
    }


def main() -> None:
    splits = {n: load_split(n) for n in ("train", "test_a", "test_b", "test_c")}
    out: dict = {"recipe_locked_gate": "PLS-3 was the locked recipe; this is diagnostic only."}

    X_train = splits["train"]["z"]
    Y_train = np.stack(
        [splits["train"]["G"], splits["train"]["D"], splits["train"]["Y"]],
        axis=1,
    )
    print(f"[diag] train z shape: {X_train.shape}")
    print(f"[diag] train target distribution:")
    for axis in ("G", "D", "Y"):
        vals, counts = np.unique(splits["train"][axis], return_counts=True)
        print(f"  {axis}: unique={vals.tolist()} counts={counts.tolist()}")

    # --- 1. PCA spectrum ---
    pca = PCA(n_components=16, svd_solver="full")
    pca.fit(X_train)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    out["pca_spectrum"] = {
        "k": list(range(1, 17)),
        "cumulative_variance": [float(c) for c in cumvar],
        "k_for_90pct": int(np.searchsorted(cumvar, 0.90) + 1),
        "k_for_95pct": int(np.searchsorted(cumvar, 0.95) + 1),
        "k_for_99pct": int(np.searchsorted(cumvar, 0.99) + 1),
    }
    print(
        f"[diag] PCA cumvar: k=1 -> {cumvar[0]:.3f}; "
        f"k=3 -> {cumvar[2]:.3f}; k=8 -> {cumvar[7]:.3f}; k=16 -> {cumvar[15]:.3f}"
    )

    # --- 2. Per-parameter Ridge baselines ---
    # Standardize X (Ridge is sensitive to scale). Use only train stats to
    # honor cross-pool generalization.
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    splits_s = {
        n: scaler.transform(s["z"]) for n, s in splits.items()
    }

    alphas = [0.1, 1.0, 10.0, 100.0]
    ridge_results: dict = {}
    for alpha in alphas:
        per_split = {}
        for split_name in ("train", "test_a", "test_b", "test_c"):
            r2_dict = {}
            rmse_dict = {}
            for axis_idx, axis in enumerate(("G", "D", "Y")):
                model = Ridge(alpha=alpha)
                model.fit(X_train_s, Y_train[:, axis_idx])
                y_true = splits[split_name][axis]
                y_pred = model.predict(splits_s[split_name])
                r2_dict[axis] = float(r2_score(y_true, y_pred))
                rmse_dict[axis] = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            r2_dict["mean"] = float(np.mean(list(r2_dict.values())))
            per_split[split_name] = {"r2": r2_dict, "rmse": rmse_dict}
        ridge_results[f"alpha_{alpha}"] = per_split

    out["ridge_per_parameter"] = ridge_results
    print("[diag] Ridge alpha=1.0:")
    for split_name in ("train", "test_a", "test_b", "test_c"):
        rd = ridge_results["alpha_1.0"][split_name]["r2"]
        print(
            f"  {split_name:8s} G={rd['G']:+.3f} D={rd['D']:+.3f} Y={rd['Y']:+.3f}"
            f"  mean={rd['mean']:+.3f}"
        )

    # --- 3. PLS n_components sweep ---
    sweep: dict = {}
    for k in (1, 2, 3, 4, 5, 6, 8):
        if k > X_train.shape[1]:
            continue
        pls = PLSRegression(n_components=k, scale=True)
        pls.fit(X_train, Y_train)
        sweep_k: dict = {}
        for split_name in ("train", "test_a", "test_b", "test_c"):
            split = splits[split_name]
            y_true = np.stack([split["G"], split["D"], split["Y"]], axis=1)
            y_pred = pls.predict(split["z"])
            r2_dict = per_param_r2(y_true, y_pred)
            r2_dict["mean"] = float(np.mean(list(r2_dict.values())))
            sweep_k[split_name] = r2_dict
        sweep[f"k={k}"] = sweep_k
    out["pls_n_components_sweep"] = sweep
    print("[diag] PLS n_components sweep (mean R^2 across G,D,Y):")
    print(f"  {'k':>3s}  {'train':>8s}  {'test_a':>8s}  {'test_b':>8s}  {'test_c':>8s}")
    for k_key, k_results in sweep.items():
        line = f"  {k_key:>3s}"
        for sn in ("train", "test_a", "test_b", "test_c"):
            line += f"  {k_results[sn]['mean']:+8.3f}"
        print(line)

    # --- 4. Symmetry diagnostic for Y (|Y| vs Y) ---
    sym: dict = {}
    for axis_label, y_transform in [("Y", lambda y: y), ("absY", np.abs)]:
        y_train = y_transform(splits["train"]["Y"])
        model = Ridge(alpha=1.0)
        model.fit(X_train_s, y_train)
        per_split = {}
        for split_name in ("train", "test_a", "test_b", "test_c"):
            y_true = y_transform(splits[split_name]["Y"])
            y_pred = model.predict(splits_s[split_name])
            per_split[split_name] = {
                "r2": float(r2_score(y_true, y_pred)),
                "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
            }
        sym[axis_label] = per_split
    out["y_symmetry_diagnostic"] = sym
    print("[diag] Y symmetry (Ridge alpha=1.0):")
    for split_name in ("train", "test_a", "test_b", "test_c"):
        r2_y = sym["Y"][split_name]["r2"]
        r2_ay = sym["absY"][split_name]["r2"]
        print(
            f"  {split_name:8s} R^2(Y) = {r2_y:+.3f}   R^2(|Y|) = {r2_ay:+.3f}   "
            f"gap = {r2_ay - r2_y:+.3f}"
        )

    # --- 5. PCA-axis correlations with (G, D, Y) ---
    # Project train z onto top-3 PCs and report correlation with (G, D, Y).
    pca_corr = PCA(n_components=8, svd_solver="full")
    Z_pc_train = pca_corr.fit_transform(X_train)
    corr_table = {}
    for pc_idx in range(8):
        pc = Z_pc_train[:, pc_idx]
        row = {}
        for axis in ("G", "D", "Y"):
            target = splits["train"][axis]
            corr = float(np.corrcoef(pc, target)[0, 1])
            row[axis] = corr
        row["explained_variance_ratio"] = float(pca_corr.explained_variance_ratio_[pc_idx])
        corr_table[f"PC{pc_idx + 1}"] = row
    out["pca_axes_vs_parameters"] = corr_table
    print("[diag] PCA-axes correlations with (G, D, Y):")
    print(f"  {'PC':>4s}  {'var_ratio':>10s}  {'r(PC,G)':>9s}  {'r(PC,D)':>9s}  {'r(PC,Y)':>9s}")
    for pc_key, row in corr_table.items():
        print(
            f"  {pc_key:>4s}  {row['explained_variance_ratio']:>10.3f}  "
            f"{row['G']:>+9.3f}  {row['D']:>+9.3f}  {row['Y']:>+9.3f}"
        )

    save = OUT / "pls_base_diagnostics.json"
    save.write_text(json.dumps(out, indent=2))
    print(f"[diag] wrote {save.relative_to(REPO)}")


if __name__ == "__main__":
    main()
