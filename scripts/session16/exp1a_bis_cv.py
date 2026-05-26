"""Session 16, Experiment 1, Part (a-bis) addendum: CV-honest hyperparameter
selection for KernelRidge(RBF) and KNeighborsRegressor.

The sweep version (exp1a_bis_nonlinear.py) reported every (gamma, alpha) on
Test B / Test C, which is honest about sensitivity but selects-on-test if you
read off "the best variant". This addendum picks hyperparameters via 5-fold
CV on train only, then reports a single number per method on Test B / Test C.

Method: GridSearchCV with cv=5, scoring='r2' (variance-weighted multi-output
when training one model per parameter).

For each of G, D, Y separately:
    KernelRidge: gamma in (0.01, 0.05, 0.1, 0.3), alpha in (0.1, 1.0, 10.0)
    KNeighborsRegressor: n_neighbors in (3, 5, 10, 20), weights in ('uniform', 'distance')

Output:
    outputs/session16/exp1/exp1a_bis_cv.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold
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


def fit_per_param(estimator_factory, param_grid: dict, X_train, targets_train, cv=5):
    """Returns {param: (best_estimator, best_params, best_cv_score)}."""
    out = {}
    for name in ("G", "D", "Y"):
        gs = GridSearchCV(
            estimator_factory(),
            param_grid=param_grid,
            cv=KFold(n_splits=cv, shuffle=True, random_state=0),
            scoring="r2",
            n_jobs=-1,
        )
        gs.fit(X_train, targets_train[name])
        out[name] = {
            "best_estimator": gs.best_estimator_,
            "best_params": gs.best_params_,
            "best_cv_r2": float(gs.best_score_),
        }
    return out


def eval_per_param(fitted: dict, X_test_by_split: dict, targets_by_split: dict) -> dict:
    out: dict = {}
    for sp, X_test in X_test_by_split.items():
        per_param = {}
        for name in ("G", "D", "Y"):
            y_pred = fitted[name]["best_estimator"].predict(X_test)
            per_param[name] = float(r2_score(targets_by_split[sp][name], y_pred))
        per_param["mean"] = float(np.mean([per_param["G"], per_param["D"], per_param["Y"]]))
        out[sp] = per_param
    return out


def main() -> None:
    splits = {n: load_split(n) for n in ("train", "test_a", "test_b", "test_c")}
    train = splits["train"]
    X_train = train["z"]
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s_by_split = {sp: scaler.transform(splits[sp]["z"]) for sp in splits}
    targets_train = {n: train[n] for n in ("G", "D", "Y")}
    targets_by_split = {sp: {n: splits[sp][n] for n in ("G", "D", "Y")} for sp in splits}

    print("[a-bis-cv] Cross-validated hyperparameter selection on train (5-fold KFold, scoring=r2)")

    results: dict = {}

    # Ridge baseline (single hyperparameter)
    ridge_fitted = fit_per_param(
        lambda: Ridge(),
        {"alpha": [0.1, 1.0, 10.0, 100.0]},
        X_train_s, targets_train,
    )
    results["ridge_cv"] = {
        "best_params": {n: ridge_fitted[n]["best_params"] for n in ("G", "D", "Y")},
        "best_cv_r2": {n: ridge_fitted[n]["best_cv_r2"] for n in ("G", "D", "Y")},
        "splits": eval_per_param(ridge_fitted, X_test_s_by_split, targets_by_split),
    }
    print(f"[a-bis-cv] Ridge best alpha per param: {results['ridge_cv']['best_params']}")
    print(f"  Train CV r2: G={results['ridge_cv']['best_cv_r2']['G']:.3f} "
          f"D={results['ridge_cv']['best_cv_r2']['D']:.3f} "
          f"Y={results['ridge_cv']['best_cv_r2']['Y']:.3f}")

    # KernelRidge RBF
    krr_fitted = fit_per_param(
        lambda: KernelRidge(kernel="rbf"),
        {"alpha": [0.1, 1.0, 10.0], "gamma": [0.01, 0.05, 0.1, 0.3]},
        X_train_s, targets_train,
    )
    results["kernel_ridge_rbf_cv"] = {
        "best_params": {n: krr_fitted[n]["best_params"] for n in ("G", "D", "Y")},
        "best_cv_r2": {n: krr_fitted[n]["best_cv_r2"] for n in ("G", "D", "Y")},
        "splits": eval_per_param(krr_fitted, X_test_s_by_split, targets_by_split),
    }
    print(f"\n[a-bis-cv] KernelRidge(RBF) best params per param: {results['kernel_ridge_rbf_cv']['best_params']}")
    print(f"  Train CV r2: G={results['kernel_ridge_rbf_cv']['best_cv_r2']['G']:.3f} "
          f"D={results['kernel_ridge_rbf_cv']['best_cv_r2']['D']:.3f} "
          f"Y={results['kernel_ridge_rbf_cv']['best_cv_r2']['Y']:.3f}")

    # KNeighborsRegressor
    knn_fitted = fit_per_param(
        lambda: KNeighborsRegressor(),
        {"n_neighbors": [3, 5, 10, 20], "weights": ["uniform", "distance"]},
        X_train_s, targets_train,
    )
    results["knn_cv"] = {
        "best_params": {n: knn_fitted[n]["best_params"] for n in ("G", "D", "Y")},
        "best_cv_r2": {n: knn_fitted[n]["best_cv_r2"] for n in ("G", "D", "Y")},
        "splits": eval_per_param(knn_fitted, X_test_s_by_split, targets_by_split),
    }
    print(f"\n[a-bis-cv] KNN best params per param: {results['knn_cv']['best_params']}")
    print(f"  Train CV r2: G={results['knn_cv']['best_cv_r2']['G']:.3f} "
          f"D={results['knn_cv']['best_cv_r2']['D']:.3f} "
          f"Y={results['knn_cv']['best_cv_r2']['Y']:.3f}")

    print("\n[a-bis-cv] CV-honest Test B / Test C R^2 per method:")
    print(f"  {'method':<25s} {'split':<8s} {'G':>7s} {'D':>7s} {'Y':>7s} {'mean':>7s}")
    for method_key, label in (
        ("ridge_cv", "Ridge"),
        ("kernel_ridge_rbf_cv", "KernelRidge(RBF)"),
        ("knn_cv", "KNN"),
    ):
        for sp in ("test_b", "test_c"):
            r = results[method_key]["splits"][sp]
            print(f"  {label:<25s} {sp:<8s} {r['G']:>+7.3f} {r['D']:>+7.3f} "
                  f"{r['Y']:>+7.3f} {r['mean']:>+7.3f}")

    save = OUT / "exp1a_bis_cv.json"
    # Strip un-serializable estimators
    for key in ("ridge_cv", "kernel_ridge_rbf_cv", "knn_cv"):
        for n in ("G", "D", "Y"):
            pass
    save.write_text(json.dumps(results, indent=2, default=lambda o: str(o)))
    print(f"\n[a-bis-cv] wrote {save.relative_to(REPO)}")


if __name__ == "__main__":
    main()
