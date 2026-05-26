"""Session 16, Experiment 1, Part (c) -- pairwise subspace overlap matrix.

Reads outputs/session16/exp1/exp1c_seed_variance.json (which already has
per-seed PLS-3 and PCA-3 bases), then computes the FULL pairwise overlap
matrix across all (seed_i, seed_j) pairs, both for PLS-3 and PCA-3.

The random-subspace overlap baseline for two K-dim subspaces in d-dim
ambient space is K/d. With K=3, d=64: 3/64 = 0.047.

Output:
    outputs/session16/exp1/exp1c_pairwise.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "session16" / "exp1"


def orthonormalize(M: np.ndarray) -> np.ndarray:
    Q, _ = np.linalg.qr(M.T)
    return Q.T


def mean_cos2(U1: np.ndarray, U2: np.ndarray) -> float:
    U1o = orthonormalize(U1)
    U2o = orthonormalize(U2)
    s = np.linalg.svd(U1o @ U2o.T, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return float(np.mean(s ** 2))


def main() -> None:
    blob = json.loads((OUT / "exp1c_seed_variance.json").read_text())
    seeds = list(blob["per_seed_pls3"].keys())
    n = len(seeds)
    pls_bases = {s: np.array(blob["per_seed_pls3"][s]["pls_basis"]) for s in seeds}
    pca_bases = {s: np.array(blob["per_seed_pca3"][s]["pca3_basis"]) for s in seeds}

    pls_mat = np.zeros((n, n))
    pca_mat = np.zeros((n, n))
    for i, si in enumerate(seeds):
        for j, sj in enumerate(seeds):
            pls_mat[i, j] = mean_cos2(pls_bases[si], pls_bases[sj])
            pca_mat[i, j] = mean_cos2(pca_bases[si], pca_bases[sj])

    K = 3
    d = next(iter(pls_bases.values())).shape[1]
    random_baseline = K / d

    out = {
        "seeds": seeds,
        "random_subspace_baseline_mean_cos2": random_baseline,
        "K_dim": K,
        "d_ambient": d,
        "pls3_pairwise_mean_cos2": pls_mat.tolist(),
        "pca3_pairwise_mean_cos2": pca_mat.tolist(),
        "pls3_offdiag_mean": float(
            (pls_mat.sum() - np.trace(pls_mat)) / (n * (n - 1))
        ),
        "pca3_offdiag_mean": float(
            (pca_mat.sum() - np.trace(pca_mat)) / (n * (n - 1))
        ),
    }

    print(f"[pairwise] random baseline (K/d = {K}/{d}) = {random_baseline:.4f}")
    print(f"[pairwise] PLS-3 mean off-diagonal cos^2 = {out['pls3_offdiag_mean']:.4f}")
    print(f"[pairwise] PCA-3 mean off-diagonal cos^2 = {out['pca3_offdiag_mean']:.4f}")

    print("\n[pairwise] PLS-3 pairwise matrix (mean cos^2):")
    header = "             " + "  ".join(f"{s:>10s}" for s in seeds)
    print(header)
    for i, si in enumerate(seeds):
        row = f"  {si:<10s} " + "  ".join(f"{pls_mat[i, j]:10.3f}" for j in range(n))
        print(row)

    print("\n[pairwise] PCA-3 pairwise matrix (mean cos^2):")
    print(header)
    for i, si in enumerate(seeds):
        row = f"  {si:<10s} " + "  ".join(f"{pca_mat[i, j]:10.3f}" for j in range(n))
        print(row)

    save = OUT / "exp1c_pairwise.json"
    save.write_text(json.dumps(out, indent=2))
    print(f"\n[pairwise] wrote {save.relative_to(REPO)}")


if __name__ == "__main__":
    main()
