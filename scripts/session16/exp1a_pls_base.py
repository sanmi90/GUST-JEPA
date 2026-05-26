"""Session 16, Experiment 1, Part (a): PLS projection of the impact-frame latent
onto (G, D, Y).

Spec (SESSION16_PLAN.md, inlined in session prompt):
    Use scikit-learn PLSRegression with n_components=3. Train on the train split's
    impact-frame latents from production E d=64 (not the seed retrains; use seed
    retrains only for variance estimates on Test B/C). Save P_base for use in
    Experiments 2 and 3.

Gate (acceptance):
    PLS R^2 > 0.85 on all three (G, D, Y) on Test B.
    If the gate fails, this script reports the failure honestly; the fallback
    documented in SESSION16_PLAN.md should then be followed instead of softening
    the result here.

Outputs:
    outputs/session16/exp1/pls_base.json   summary + per-parameter R^2 / RMSE
    outputs/session16/exp1/pls_base.npz    fitted PLS attributes (x_mean, y_mean,
                                            x_scale, y_scale, x_weights, x_loadings,
                                            y_loadings, x_rotations, y_rotations,
                                            coef, intercept) + predictions per split

Provenance source: outputs/session14/latents/S12_E_d64/{train,test_a,test_b,test_c}.npz
which carry impact-frame z (180/70/28/24, 64) plus (G, D, Y) labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score


REPO = Path(__file__).resolve().parents[2]
LATENTS = REPO / "outputs" / "session14" / "latents" / "S12_E_d64"
OUT = REPO / "outputs" / "session16" / "exp1"
OUT.mkdir(parents=True, exist_ok=True)

ENCODER_CKPT = (
    REPO / "outputs" / "runs" / "session12" / "S12_E_d64" / "encoder"
    / "checkpoint_iter020000.pt"
)


def load_split(name: str) -> dict:
    f = LATENTS / f"{name}.npz"
    d = np.load(f, allow_pickle=True)
    return {
        "z": d["z"].astype(np.float64),
        "G": d["G"].astype(np.float64),
        "D": d["D"].astype(np.float64),
        "Y": d["Y"].astype(np.float64),
        "case_id": d["case_id"],
        "encounter_index": d["encounter_index"].astype(np.int32),
        "impact_frame": d["impact_frame"].astype(np.int32),
    }


def stack_targets(split: dict) -> np.ndarray:
    return np.stack([split["G"], split["D"], split["Y"]], axis=1)


def per_param_r2(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    names = ("G", "D", "Y")
    return {
        n: float(r2_score(y_true[:, i], y_pred[:, i]))
        for i, n in enumerate(names)
    }


def per_param_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    names = ("G", "D", "Y")
    return {
        n: float(np.sqrt(np.mean((y_true[:, i] - y_pred[:, i]) ** 2)))
        for i, n in enumerate(names)
    }


def main() -> None:
    splits = {n: load_split(n) for n in ("train", "test_a", "test_b", "test_c")}

    print(
        f"[exp1a] loaded splits: "
        + ", ".join(
            f"{n}={s['z'].shape}" for n, s in splits.items()
        )
    )

    X_train = splits["train"]["z"]
    Y_train = stack_targets(splits["train"])

    pls = PLSRegression(n_components=3, scale=True)
    pls.fit(X_train, Y_train)

    results: dict = {
        "encoder_ckpt": str(ENCODER_CKPT.relative_to(REPO)),
        "n_components": 3,
        "scaler": "PLSRegression default (per-feature standardization on X and Y)",
        "splits": {},
    }

    for name, split in splits.items():
        z = split["z"]
        y_true = stack_targets(split)
        y_pred = pls.predict(z)
        r2 = per_param_r2(y_true, y_pred)
        rmse = per_param_rmse(y_true, y_pred)
        r2_mean = float(np.mean(list(r2.values())))
        results["splits"][name] = {
            "n": int(z.shape[0]),
            "r2_per_param": r2,
            "r2_mean": r2_mean,
            "rmse_per_param": rmse,
        }
        print(
            f"[exp1a] {name:8s} (n={z.shape[0]:3d}) "
            f"r2: G={r2['G']:+.3f} D={r2['D']:+.3f} Y={r2['Y']:+.3f}  "
            f"mean={r2_mean:+.3f}"
        )

    gate = {
        "rule": "Test B per-parameter R^2 > 0.85 on all of (G, D, Y).",
        "test_b": results["splits"]["test_b"]["r2_per_param"],
        "pass": all(
            v > 0.85 for v in results["splits"]["test_b"]["r2_per_param"].values()
        ),
    }
    results["acceptance_gate"] = gate
    print(
        f"[exp1a] ACCEPTANCE GATE on Test B (R^2 > 0.85 all params): "
        f"{'PASS' if gate['pass'] else 'FAIL'}"
    )

    save_path_json = OUT / "pls_base.json"
    save_path_json.write_text(json.dumps(results, indent=2, sort_keys=False))
    print(f"[exp1a] wrote {save_path_json.relative_to(REPO)}")

    save_path_npz = OUT / "pls_base.npz"
    pred_blocks = {}
    for name, split in splits.items():
        z = split["z"]
        y_true = stack_targets(split)
        y_pred = pls.predict(z)
        pred_blocks[f"y_true_{name}"] = y_true
        pred_blocks[f"y_pred_{name}"] = y_pred
        pred_blocks[f"z_{name}"] = z
        pred_blocks[f"case_id_{name}"] = split["case_id"]
        pred_blocks[f"encounter_index_{name}"] = split["encounter_index"]

    np.savez(
        save_path_npz,
        n_components=np.array(3),
        x_mean=pls._x_mean,
        x_std=pls._x_std,
        y_mean=pls._y_mean,
        y_std=pls._y_std,
        x_weights=pls.x_weights_,
        x_loadings=pls.x_loadings_,
        y_loadings=pls.y_loadings_,
        x_rotations=pls.x_rotations_,
        y_rotations=pls.y_rotations_,
        x_scores_train=pls.x_scores_,
        y_scores_train=pls.y_scores_,
        coef=pls.coef_,
        intercept=pls.intercept_,
        **pred_blocks,
    )
    print(f"[exp1a] wrote {save_path_npz.relative_to(REPO)}")

    print("[exp1a] PLS x_rotations shape:", pls.x_rotations_.shape)
    print("[exp1a] PLS y_loadings shape:", pls.y_loadings_.shape)


if __name__ == "__main__":
    main()
