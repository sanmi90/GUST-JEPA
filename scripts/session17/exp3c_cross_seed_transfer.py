"""Session 17, Experiment 3, Part (c): cross-seed function transfer for Y.

The test: fit KernelRidge(RBF) on z_{seed_i}(t_impact) -> Y, then *apply this
regressor to z_{seed_j}(t_impact)* and measure R^2 on Test B.

Each seed's latents must be in comparable normalization. Standardize each
seed's train latents to zero mean unit variance per dimension before
fitting; apply the same per-seed standardization to its own Test B latents.

Acceptance gate (SESSION17_PLAN.md):
  R^2 > 0.5 for Y between at least 4 of 6 seed pairs.

This isolates whether the same input data (case_id, encounter_index) produces
COMPARABLE Y-extraction functions across seeds (== the nonlinear function is
canonical) or whether each seed produces a distinct latent encoding (== the
function is seed-specific).

Outputs:
    outputs/session17/exp3/cross_seed_function_transfer.json
    outputs/session17/figures/exp3_function_transfer_heatmap.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold


REPO = Path(__file__).resolve().parents[2]
SEED_LATENTS = REPO / "outputs" / "session17" / "seed_latents"
EXP3 = REPO / "outputs" / "session17" / "exp3"
FIGS = REPO / "outputs" / "session17" / "figures"
SEEDS = ("production", "seed0", "seed1", "seed2")


def load_seed_impact(seed: str) -> dict:
    out = {}
    for split in ("train", "test_b", "test_c"):
        d = np.load(SEED_LATENTS / seed / f"{split}.npz", allow_pickle=True)
        out[split] = {
            "z": d["z"].astype(np.float64),
            "G": d["G"].astype(np.float64),
            "D": d["D"].astype(np.float64),
            "Y": d["Y"].astype(np.float64),
            "case_id": np.asarray(d["case_id"]).astype(object),
            "encounter_index": d["encounter_index"].astype(np.int32),
        }
    return out


def cv_alpha_gamma(X: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    alphas = [0.01, 0.1, 1.0, 10.0]
    gammas = [0.005, 0.01, 0.05, 0.1]
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    best = (None, None, -np.inf)
    for a in alphas:
        for g in gammas:
            scores = []
            for tr, va in kf.split(X):
                m = KernelRidge(alpha=a, gamma=g, kernel="rbf")
                m.fit(X[tr], y[tr])
                pred = m.predict(X[va])
                scores.append(r2_score(y[va], pred))
            mean_r2 = float(np.mean(scores))
            if mean_r2 > best[2]:
                best = (a, g, mean_r2)
    return best


def main() -> None:
    seed_data = {s: load_seed_impact(s) for s in SEEDS}
    print(f"[exp3c] loaded {len(SEEDS)} seeds")

    # Standardize each seed's train z; apply same to test_b/test_c per seed.
    standardized = {}
    for s in SEEDS:
        train_z = seed_data[s]["train"]["z"]
        mu = train_z.mean(axis=0, keepdims=True)
        sigma = train_z.std(axis=0, keepdims=True).clip(min=1e-9)
        standardized[s] = {
            "mu": mu,
            "sigma": sigma,
            "train": (train_z - mu) / sigma,
            "test_b": (seed_data[s]["test_b"]["z"] - mu) / sigma,
            "test_c": (seed_data[s]["test_c"]["z"] - mu) / sigma,
        }

    # Sanity-check that all seeds have the same (case_id, encounter_index) order.
    base = seed_data["production"]["train"]
    for s in SEEDS:
        if s == "production":
            continue
        for split in ("train", "test_b", "test_c"):
            d = seed_data[s][split]
            base_split = seed_data["production"][split]
            assert np.array_equal(d["case_id"], base_split["case_id"]), s
            assert np.array_equal(d["encounter_index"], base_split["encounter_index"]), s
            assert np.allclose(d["Y"], base_split["Y"])

    # Fit KRR on each seed's train z -> Y; record CV-selected hyperparameters.
    models = {}
    cv_choices = {}
    for s in SEEDS:
        X_tr = standardized[s]["train"]
        y_tr = seed_data[s]["train"]["Y"]
        a, g, cv = cv_alpha_gamma(X_tr, y_tr)
        m = KernelRidge(alpha=a, gamma=g, kernel="rbf")
        m.fit(X_tr, y_tr)
        models[s] = m
        cv_choices[s] = {"alpha": a, "gamma": g, "cv_r2": cv}
        print(f"[exp3c] {s}: KRR(Y) alpha={a} gamma={g}  CV R^2={cv:.3f}")

    # Cross-seed transfer matrix: rows = trained-on seed, cols = applied-to seed.
    transfer_b = np.zeros((len(SEEDS), len(SEEDS)))
    transfer_c = np.zeros((len(SEEDS), len(SEEDS)))
    for i, si in enumerate(SEEDS):
        for j, sj in enumerate(SEEDS):
            # Apply model trained on si to z_sj (test_b, test_c)
            Xb = standardized[sj]["test_b"]
            Xc = standardized[sj]["test_c"]
            yb = seed_data[sj]["test_b"]["Y"]
            yc = seed_data[sj]["test_c"]["Y"]
            transfer_b[i, j] = float(r2_score(yb, models[si].predict(Xb)))
            transfer_c[i, j] = float(r2_score(yc, models[si].predict(Xc)))

    # Pairwise transfer R^2 (off-diagonal mean across i != j) for the 6 pairs.
    pair_records = []
    for i in range(len(SEEDS)):
        for j in range(i + 1, len(SEEDS)):
            # Bidirectional: si -> sj and sj -> si; take both numbers.
            r2_ij_b = transfer_b[i, j]
            r2_ji_b = transfer_b[j, i]
            r2_ij_c = transfer_c[i, j]
            r2_ji_c = transfer_c[j, i]
            pair_records.append(
                {
                    "seed_a": SEEDS[i],
                    "seed_b": SEEDS[j],
                    "test_b": {
                        "r2_a_to_b": r2_ij_b,
                        "r2_b_to_a": r2_ji_b,
                        "mean": float((r2_ij_b + r2_ji_b) / 2),
                    },
                    "test_c": {
                        "r2_a_to_b": r2_ij_c,
                        "r2_b_to_a": r2_ji_c,
                        "mean": float((r2_ij_c + r2_ji_c) / 2),
                    },
                }
            )

    # Acceptance gate.
    means_b = np.array([p["test_b"]["mean"] for p in pair_records])
    n_pass_b = int(np.sum(means_b > 0.5))
    gate_b = n_pass_b >= 4

    summary = {
        "seeds": list(SEEDS),
        "cv_choices_per_seed": cv_choices,
        "transfer_matrix_test_b": transfer_b.tolist(),
        "transfer_matrix_test_c": transfer_c.tolist(),
        "diagonal_test_b": [transfer_b[i, i] for i in range(len(SEEDS))],
        "diagonal_test_c": [transfer_c[i, i] for i in range(len(SEEDS))],
        "pair_records": pair_records,
        "acceptance_gate": {
            "rule": "Y test_b mean transfer R^2 > 0.5 on at least 4 of 6 seed pairs",
            "n_pass_above_0p5": n_pass_b,
            "n_total_pairs": int(len(pair_records)),
            "passes": gate_b,
            "pair_means_test_b": means_b.tolist(),
        },
    }
    (EXP3 / "cross_seed_function_transfer.json").write_text(json.dumps(summary, indent=2))
    print(f"[exp3c] wrote {EXP3 / 'cross_seed_function_transfer.json'}")

    print("\n[exp3c] Y transfer R^2 on Test B (row=train seed, col=apply seed):")
    print(f"  {'':10s}" + "".join(f"{s:>10s}" for s in SEEDS))
    for i, si in enumerate(SEEDS):
        line = f"  {si:10s}"
        for j in range(len(SEEDS)):
            line += f"{transfer_b[i, j]:>10.3f}"
        print(line)

    print("\n[exp3c] Pair-level mean transfer (Test B):")
    print(f"  {'seed_a':<12s} {'seed_b':<12s} {'mean_R^2':>10s}  {'PASS?':>6s}")
    for p in pair_records:
        flag = "PASS" if p["test_b"]["mean"] > 0.5 else "FAIL"
        print(
            f"  {p['seed_a']:<12s} {p['seed_b']:<12s} "
            f"{p['test_b']['mean']:>10.3f}  {flag:>6s}"
        )
    print(f"\n[exp3c] gate (>=4/6 pairs > 0.5): {n_pass_b}/6  -> "
          f"{'PASS' if gate_b else 'FAIL'}")

    # Figure: 1x2 transfer heatmaps for test_b and test_c.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mat, title in (
        (axes[0], transfer_b, "Test B  Y transfer R^2"),
        (axes[1], transfer_c, "Test C  Y transfer R^2"),
    ):
        im = ax.imshow(mat, cmap="coolwarm", vmin=-0.5, vmax=1.0)
        ax.set_xticks(range(len(SEEDS)))
        ax.set_yticks(range(len(SEEDS)))
        ax.set_xticklabels(SEEDS, rotation=45, ha="right")
        ax.set_yticklabels(SEEDS)
        ax.set_xlabel("applied to seed")
        ax.set_ylabel("trained on seed")
        for i in range(len(SEEDS)):
            for j in range(len(SEEDS)):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        color="white" if abs(mat[i, j]) > 0.4 else "black", fontsize=10)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"Cross-seed KRR(Y) function transfer\n"
        f"gate (test_b >=4/6 pairs > 0.5): "
        f"{n_pass_b}/6 -> {'PASS' if gate_b else 'FAIL'}"
    )
    fig.tight_layout()
    fig.savefig(FIGS / "exp3_function_transfer_heatmap.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[exp3c] wrote {FIGS / 'exp3_function_transfer_heatmap.png'}")


if __name__ == "__main__":
    main()
