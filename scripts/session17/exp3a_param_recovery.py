"""Session 17, Experiment 3, Part (a): per-frame parameter recovery R^2(tau).

For each frame offset tau in {-20, -10, -5, -2, 0, +2, +5, +10, +20, +40}
relative to t_impact = 40, fit a KernelRidge regressor with CV-tuned
hyperparameters on the train split's z(t_impact + tau) -> (G, D, Y) and
evaluate R^2 on Test B (v1p5, 56 encounters) and Test C (24 encounters).

Reports R^2 per (parameter, tau, split). Companion to Session 16 D118-bis
which already showed Y reaches R^2 = 0.73 with KernelRidge at tau = 0.

Outputs:
    outputs/session17/exp3/per_frame_recovery.csv
    outputs/session17/exp3/per_frame_recovery_summary.json
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold


REPO = Path(__file__).resolve().parents[2]
SEED_LATENTS = REPO / "outputs" / "session17" / "seed_latents"
OUT = REPO / "outputs" / "session17" / "exp3"
OUT.mkdir(parents=True, exist_ok=True)

T_IMPACT = 40
TAUS = [-20, -10, -5, -2, 0, +2, +5, +10, +20, +40]
PARAMS = ("G", "D", "Y")
SEED = "production"


def load_for_seed(seed: str) -> dict:
    """Return dict with z_full for train/test_b and impact-frame z for test_c."""
    splits = {}
    for sp, full in [("train", False), ("test_b", True), ("test_c", False)]:
        d = np.load(SEED_LATENTS / seed / f"{sp}.npz", allow_pickle=True)
        splits[sp] = {
            "z": d["z"].astype(np.float64),  # impact-frame
            "G": d["G"].astype(np.float64),
            "D": d["D"].astype(np.float64),
            "Y": d["Y"].astype(np.float64),
            "case_id": np.asarray(d["case_id"]).astype(object),
            "encounter_index": d["encounter_index"].astype(np.int32),
        }
        if full and "z_full" in d.files:
            splits[sp]["z_full"] = d["z_full"].astype(np.float64)
    return splits


def encode_train_z_full() -> np.ndarray:
    """Train z_full lives in the session14 pre-extracted cache."""
    f = REPO / "outputs" / "session14" / "latents" / "S12_E_d64" / "train.npz"
    d = np.load(f, allow_pickle=True)
    return d["z_full"].astype(np.float64), d["G"].astype(np.float64), d[
        "D"
    ].astype(np.float64), d["Y"].astype(np.float64)


def encode_test_c_z_full() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """test_c z_full from session14 cache (24 encounters)."""
    f = REPO / "outputs" / "session14" / "latents" / "S12_E_d64" / "test_c.npz"
    d = np.load(f, allow_pickle=True)
    return (
        d["z_full"].astype(np.float64),
        d["G"].astype(np.float64),
        d["D"].astype(np.float64),
        d["Y"].astype(np.float64),
    )


def cv_alpha_gamma(X: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """5-fold CV over (alpha, gamma); return best alpha, gamma, CV-R^2."""
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
    seed_data = load_for_seed(SEED)
    train_z_full, train_G, train_D, train_Y = encode_train_z_full()
    test_c_z_full, test_c_G, test_c_D, test_c_Y = encode_test_c_z_full()
    test_b_z_full = seed_data["test_b"]["z_full"]
    test_b_G = seed_data["test_b"]["G"]
    test_b_D = seed_data["test_b"]["D"]
    test_b_Y = seed_data["test_b"]["Y"]

    print(
        f"[exp3a] train z_full {train_z_full.shape} "
        f"test_b {test_b_z_full.shape} test_c {test_c_z_full.shape}"
    )

    rows = []
    for tau in TAUS:
        t = T_IMPACT + tau
        if t < 0 or t >= train_z_full.shape[1]:
            print(f"[exp3a] tau={tau:+d} (t={t}) out of bounds, skip")
            continue
        X_tr = train_z_full[:, t, :]
        Xb = test_b_z_full[:, t, :]
        Xc = test_c_z_full[:, t, :]
        for param, y_tr, y_b, y_c in [
            ("G", train_G, test_b_G, test_c_G),
            ("D", train_D, test_b_D, test_c_D),
            ("Y", train_Y, test_b_Y, test_c_Y),
        ]:
            a, g, cv_r2 = cv_alpha_gamma(X_tr, y_tr)
            m = KernelRidge(alpha=a, gamma=g, kernel="rbf")
            m.fit(X_tr, y_tr)
            r2_tr = float(r2_score(y_tr, m.predict(X_tr)))
            r2_b = float(r2_score(y_b, m.predict(Xb)))
            r2_c = float(r2_score(y_c, m.predict(Xc)))
            rows.append(
                {
                    "tau": tau,
                    "frame": t,
                    "param": param,
                    "alpha": a,
                    "gamma": g,
                    "cv_r2": cv_r2,
                    "train_r2": r2_tr,
                    "test_b_r2": r2_b,
                    "test_c_r2": r2_c,
                }
            )
            print(
                f"[exp3a] tau={tau:+3d} (t={t:>3d}) param={param}  "
                f"alpha={a:5.2f} gamma={g:5.3f}  "
                f"cv={cv_r2:+.3f}  train={r2_tr:+.3f}  test_b={r2_b:+.3f}  test_c={r2_c:+.3f}"
            )

    # Save CSV.
    csv_path = OUT / "per_frame_recovery.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[exp3a] wrote {csv_path}")

    # Save JSON summary keyed by param.
    summary: dict = {"taus": TAUS, "seed": SEED, "per_param": {}}
    for param in PARAMS:
        sub = [r for r in rows if r["param"] == param]
        summary["per_param"][param] = {
            "test_b_r2": [r["test_b_r2"] for r in sub],
            "test_c_r2": [r["test_c_r2"] for r in sub],
            "train_r2": [r["train_r2"] for r in sub],
            "cv_r2": [r["cv_r2"] for r in sub],
            "tau": [r["tau"] for r in sub],
            "alpha": [r["alpha"] for r in sub],
            "gamma": [r["gamma"] for r in sub],
        }
    json_path = OUT / "per_frame_recovery_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"[exp3a] wrote {json_path}")


if __name__ == "__main__":
    main()
