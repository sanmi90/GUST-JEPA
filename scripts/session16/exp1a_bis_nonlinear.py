"""Session 16, Experiment 1, Part (a-bis): nonlinear recovery of (G, D, Y).

Tests whether the encoder's information about (G, D, Y) is recoverable via
NONLINEAR methods that the recipe-locked PLS-3 + linear Ridge baselines
cannot reach. Five methods compared per parameter, with Test B / Test C R^2
as the headline metric:

    PLS-3                          : recipe-locked baseline (D118)
    Ridge (alpha=1.0)              : full-64-D linear baseline (D118 diag)
    Isomap(d=3, k=K) + Ridge       : geodesic manifold embed + linear regression
    KernelPCA(RBF, d=3) + Ridge    : RBF-kernel nonlinear embed + linear regression
    KNeighborsRegressor (k)        : local-average nonlinear regression on full z
    KernelRidge(RBF)               : full-64-D RBF kernel regression

For embed-then-regress methods (Isomap, KernelPCA), each parameter is fit
separately with Ridge on the 3 embedded coordinates.

For direct regressors (KNN, KernelRidge), each parameter is fit separately
on full 64-D z.

Hyperparameter sweep (kept small per session priority 4 -- no hyperparameter
tuning -- but reporting the full sweep honestly so the reader can see the
sensitivity):

    Isomap n_neighbors  in (5, 10, 15)
    KernelPCA gamma     in (0.01, 0.05, 0.1)
    KNN n_neighbors     in (3, 5, 10, 20)
    KernelRidge gamma   in (0.01, 0.05, 0.1, 0.3) x alpha (0.1, 1.0, 10.0)

Output:
    outputs/session16/exp1/exp1a_bis_nonlinear.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import KernelPCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.manifold import Isomap
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
LATENTS = REPO / "outputs" / "session14" / "latents" / "S12_E_d64"
OUT = REPO / "outputs" / "session16" / "exp1"


def load_split(name: str) -> dict:
    d = np.load(LATENTS / f"{name}.npz", allow_pickle=True)
    return {
        "z": d["z"].astype(np.float64),
        "G": d["G"].astype(np.float64),
        "D": d["D"].astype(np.float64),
        "Y": d["Y"].astype(np.float64),
    }


def per_param_r2(model, X_test_dict: dict, name: str) -> dict:
    return {
        sp: {
            n: float(r2_score(s[n], model[n].predict(X_test_dict[sp])))
            for n in ("G", "D", "Y")
        }
        for sp, s in X_test_dict.items() if False  # placeholder; we build differently below
    }


def fit_per_param_ridge(X: np.ndarray, splits_targets: dict, alpha: float = 1.0) -> dict:
    """Train one Ridge per (G, D, Y) on X -> target. Returns {param: model}."""
    models = {}
    for name in ("G", "D", "Y"):
        m = Ridge(alpha=alpha)
        m.fit(X, splits_targets[name])
        models[name] = m
    return models


def eval_per_param(
    models: dict, X_test_by_split: dict, targets_by_split: dict
) -> dict:
    out = {}
    for sp, X_test in X_test_by_split.items():
        per_param = {}
        for name in ("G", "D", "Y"):
            y_pred = models[name].predict(X_test)
            per_param[name] = float(r2_score(targets_by_split[sp][name], y_pred))
        per_param["mean"] = float(np.mean(list(per_param.values())))
        out[sp] = per_param
    return out


def main() -> None:
    splits = {n: load_split(n) for n in ("train", "test_a", "test_b", "test_c")}
    train = splits["train"]
    X_train = train["z"]
    print(f"[a-bis] X_train shape: {X_train.shape}")

    targets_train = {n: train[n] for n in ("G", "D", "Y")}
    targets_by_split = {sp: {n: splits[sp][n] for n in ("G", "D", "Y")} for sp in splits}

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s_by_split = {sp: scaler.transform(splits[sp]["z"]) for sp in splits}

    results: dict = {}

    # ----- Baseline 1: PLS-3 (recipe locked)
    print("\n[a-bis] PLS-3 baseline (recipe locked)")
    pls = PLSRegression(n_components=3, scale=True)
    pls.fit(X_train, np.stack([train["G"], train["D"], train["Y"]], axis=1))
    results["pls_3"] = {}
    for sp in splits:
        y_pred = pls.predict(splits[sp]["z"])
        per_param = {
            "G": float(r2_score(splits[sp]["G"], y_pred[:, 0])),
            "D": float(r2_score(splits[sp]["D"], y_pred[:, 1])),
            "Y": float(r2_score(splits[sp]["Y"], y_pred[:, 2])),
        }
        per_param["mean"] = float(np.mean(list(per_param.values())))
        results["pls_3"][sp] = per_param

    # ----- Baseline 2: Ridge on full 64-D z (per-parameter)
    print("[a-bis] Ridge (alpha=1.0) baseline on full 64-D z")
    ridge_models = fit_per_param_ridge(X_train_s, targets_train, alpha=1.0)
    results["ridge_alpha_1.0"] = eval_per_param(ridge_models, X_test_s_by_split, targets_by_split)

    # ----- Method A: Isomap(d=3, k) + Ridge
    print("\n[a-bis] Isomap(d=3, k_neighbors=...) + Ridge")
    isomap_block: dict = {}
    for k in (5, 10, 15):
        iso = Isomap(n_components=3, n_neighbors=k)
        Xtr_iso = iso.fit_transform(X_train_s)
        models = fit_per_param_ridge(Xtr_iso, targets_train, alpha=1.0)
        X_emb_by_split = {sp: iso.transform(X_test_s_by_split[sp]) for sp in splits}
        eval_res = eval_per_param(models, X_emb_by_split, targets_by_split)
        isomap_block[f"k_neighbors={k}"] = eval_res
    results["isomap_3_then_ridge"] = isomap_block

    # ----- Method B: KernelPCA(RBF, d=3) + Ridge
    print("[a-bis] KernelPCA(RBF, d=3, gamma=...) + Ridge")
    kpca_block: dict = {}
    for gamma in (0.01, 0.05, 0.1):
        kpca = KernelPCA(n_components=3, kernel="rbf", gamma=gamma, fit_inverse_transform=False)
        Xtr_kpca = kpca.fit_transform(X_train_s)
        models = fit_per_param_ridge(Xtr_kpca, targets_train, alpha=1.0)
        X_emb_by_split = {sp: kpca.transform(X_test_s_by_split[sp]) for sp in splits}
        eval_res = eval_per_param(models, X_emb_by_split, targets_by_split)
        kpca_block[f"gamma={gamma}"] = eval_res
    results["kernel_pca_rbf_3_then_ridge"] = kpca_block

    # ----- Method C: KNeighborsRegressor direct on full 64-D z
    print("[a-bis] KNeighborsRegressor on full 64-D z (k_neighbors=...)")
    knn_block: dict = {}
    for k in (3, 5, 10, 20):
        models = {}
        for name in ("G", "D", "Y"):
            m = KNeighborsRegressor(n_neighbors=k, weights="distance")
            m.fit(X_train_s, train[name])
            models[name] = m
        eval_res = eval_per_param(models, X_test_s_by_split, targets_by_split)
        knn_block[f"k_neighbors={k}"] = eval_res
    results["knn_regressor_full_z"] = knn_block

    # ----- Method D: KernelRidge(RBF) on full 64-D z
    print("[a-bis] KernelRidge(RBF) on full 64-D z (gamma=..., alpha=...)")
    krr_block: dict = {}
    for gamma in (0.01, 0.05, 0.1, 0.3):
        for alpha in (0.1, 1.0, 10.0):
            models = {}
            for name in ("G", "D", "Y"):
                m = KernelRidge(kernel="rbf", gamma=gamma, alpha=alpha)
                m.fit(X_train_s, train[name])
                models[name] = m
            eval_res = eval_per_param(models, X_test_s_by_split, targets_by_split)
            krr_block[f"gamma={gamma},alpha={alpha}"] = eval_res
    results["kernel_ridge_rbf_full_z"] = krr_block

    # ----- Print compact summary
    print("\n[a-bis] Summary: Test B per-parameter R^2 across all methods")
    print(f"  {'method/variant':<45s}  {'G':>7s}  {'D':>7s}  {'Y':>7s}  {'mean':>7s}")

    def _print_row(label: str, rr: dict, sp: str = "test_b") -> None:
        r = rr[sp]
        print(f"  {label:<45s}  {r['G']:>+7.3f}  {r['D']:>+7.3f}  {r['Y']:>+7.3f}  {r['mean']:>+7.3f}")

    _print_row("PLS-3 (recipe-locked baseline)", results["pls_3"])
    _print_row("Ridge alpha=1.0 (64-D linear)", results["ridge_alpha_1.0"])
    for k_label, rr in results["isomap_3_then_ridge"].items():
        _print_row(f"Isomap(d=3) + Ridge | {k_label}", rr)
    for g_label, rr in results["kernel_pca_rbf_3_then_ridge"].items():
        _print_row(f"KernelPCA(RBF, d=3) + Ridge | {g_label}", rr)
    for k_label, rr in results["knn_regressor_full_z"].items():
        _print_row(f"KNN(weights=distance) | {k_label}", rr)
    for ga_label, rr in results["kernel_ridge_rbf_full_z"].items():
        _print_row(f"KernelRidge(RBF) | {ga_label}", rr)

    print("\n[a-bis] Summary: Test C per-parameter R^2 (G=+4 OOD)")
    print(f"  {'method/variant':<45s}  {'G':>7s}  {'D':>7s}  {'Y':>7s}  {'mean':>7s}")
    _print_row("PLS-3", results["pls_3"], "test_c")
    _print_row("Ridge alpha=1.0", results["ridge_alpha_1.0"], "test_c")
    for k_label, rr in results["isomap_3_then_ridge"].items():
        _print_row(f"Isomap(d=3) + Ridge | {k_label}", rr, "test_c")
    for g_label, rr in results["kernel_pca_rbf_3_then_ridge"].items():
        _print_row(f"KernelPCA(RBF, d=3) + Ridge | {g_label}", rr, "test_c")
    for k_label, rr in results["knn_regressor_full_z"].items():
        _print_row(f"KNN(weights=distance) | {k_label}", rr, "test_c")
    for ga_label, rr in results["kernel_ridge_rbf_full_z"].items():
        _print_row(f"KernelRidge(RBF) | {ga_label}", rr, "test_c")

    save = OUT / "exp1a_bis_nonlinear.json"
    save.write_text(json.dumps(results, indent=2))
    print(f"\n[a-bis] wrote {save.relative_to(REPO)}")


if __name__ == "__main__":
    main()
