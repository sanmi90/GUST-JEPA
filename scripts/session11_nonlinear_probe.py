"""kNN and RBF kernel-ridge probes for (G, D, Y) on three latent views.

Companion to session11_latent_disentanglement.py and
session11_isomap_disentanglement.py. Same data, different probe family:

- Linear probe (already reported): assumes a linear mapping z -> param.
- kNN regression (k = 5): averages over nearest neighbours; nonparametric.
- RBF kernel ridge: Gaussian-kernel smoothed regression; nonparametric.

Each R^2 is reported as 5-fold cross-validated to avoid in-sample
overfit. Runs on three latent representations:

1. Raw z in R^32 (impact-frame averaged)
2. PCA k = 12 (top 12 PCs)
3. Isomap K = 10 (geodesic embedding)

Output: a markdown-style table that we can paste into the report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True, type=str,
                   help="disentanglement.npz (Z_imp, G, D, Y)")
    p.add_argument("--iso", required=True, type=str,
                   help="isomap_diagnostic.npz (Z_iso)")
    p.add_argument("--pca-basis", required=True, type=str,
                   help="pca_basis.npz (mean, P)")
    p.add_argument("--output-json", required=True, type=str)
    p.add_argument("--k", type=int, default=5, help="kNN neighbours.")
    p.add_argument("--cv", type=int, default=5, help="CV folds.")
    p.add_argument("--rbf-gamma", type=float, default=None,
                   help="RBF gamma. If None, use 1/(d * Var(X)) heuristic.")
    return p.parse_args()


def linear_r2_full(Z: np.ndarray, y: np.ndarray, cv: KFold) -> float:
    """In-sample full multivariate linear R^2 was already reported; this
    function returns the CV linear R^2 (i.e. unregularised OLS via
    KernelRidge with linear kernel, alpha = 0)."""
    # KernelRidge with kernel='linear', alpha=0 is just OLS.
    est = KernelRidge(alpha=1e-6, kernel="linear")
    scores = cross_val_score(est, Z, y, scoring="r2", cv=cv, n_jobs=-1)
    return float(scores.mean())


def knn_r2(Z: np.ndarray, y: np.ndarray, k: int, cv: KFold) -> float:
    est = KNeighborsRegressor(n_neighbors=k, weights="distance")
    scores = cross_val_score(est, Z, y, scoring="r2", cv=cv, n_jobs=-1)
    return float(scores.mean())


def rbf_r2(Z: np.ndarray, y: np.ndarray, cv: KFold, gamma: float | None) -> float:
    # RBF kernel ridge with light regularisation; gamma from heuristic.
    if gamma is None:
        gamma = 1.0 / (Z.shape[1] * Z.var() + 1e-9)
    est = KernelRidge(alpha=1e-2, kernel="rbf", gamma=gamma)
    scores = cross_val_score(est, Z, y, scoring="r2", cv=cv, n_jobs=-1)
    return float(scores.mean())


def main() -> None:
    args = parse_args()
    b = np.load(args.bundle, allow_pickle=True)
    iso = np.load(args.iso, allow_pickle=True)
    pca = np.load(args.pca_basis)

    Z_raw = b["Z_imp"].astype(np.float64)  # (n, 32)
    G = b["G"].astype(np.float64)
    D = b["D"].astype(np.float64)
    Y = b["Y"].astype(np.float64)

    mean = pca["mean"].astype(np.float64)
    P = pca["P"].astype(np.float64)
    Z_pca = (Z_raw - mean[None]) @ P  # (n, 12)
    Z_iso = iso["Z_iso"].astype(np.float64)  # (n, 10)

    print(f"[probe] Z_raw={Z_raw.shape}, Z_pca={Z_pca.shape}, Z_iso={Z_iso.shape}")
    print(f"[probe] k={args.k} (kNN), cv={args.cv}-fold")

    representations = {
        "raw d=32":   Z_raw,
        "PCA k=12":   Z_pca,
        "Isomap K=10": Z_iso,
    }
    factors = {"G": G, "D": D, "Y": Y}

    cv = KFold(n_splits=args.cv, shuffle=True, random_state=0)
    results = {}
    for name, Z in representations.items():
        Zs = StandardScaler().fit_transform(Z)
        row = {}
        for fname, y in factors.items():
            r2_lin = linear_r2_full(Zs, y, cv)
            r2_knn = knn_r2(Zs, y, args.k, cv)
            r2_rbf = rbf_r2(Zs, y, cv, args.rbf_gamma)
            row[fname] = {"linear": r2_lin, "knn": r2_knn, "rbf": r2_rbf}
        results[name] = row

    # Pretty print as a markdown table.
    print()
    print(f"| representation | probe   |  R^2(G) | R^2(D) | R^2(Y) |")
    print(f"|----------------|---------|---------|--------|--------|")
    probe_order = ["linear", "knn", "rbf"]
    pretty_probe = {"linear": "linear ", "knn": f"kNN k={args.k}", "rbf": "RBF KR "}
    for rep_name, row in results.items():
        for probe in probe_order:
            r = row
            r2g = r["G"][probe]; r2d = r["D"][probe]; r2y = r["Y"][probe]
            print(f"| {rep_name:<14} | {pretty_probe[probe]:<7} "
                  f"|  {r2g:+.3f} | {r2d:+.3f} | {r2y:+.3f} |")
        print(f"|----------------|---------|---------|--------|--------|")

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"k": args.k, "cv": args.cv, "results": results}, f, indent=2)
    print(f"\n[probe] saved {out}")


if __name__ == "__main__":
    main()
