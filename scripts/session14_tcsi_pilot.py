"""Session 14 Thrust 7: TCSI sparse-pressure sensor selection pilot.

This script implements the target-conditioned structural-information (TCSI)
sensor selection pilot from ``SESSION14_PLAN_UPDATE_SENSOR_PILOT.md``. TCSI is
INSPIRED BY the epiplexity framework of Finzi et al., "From Entropy to
Epiplexity" (arXiv:2601.03220v2, 2026), but the proxies used here are MSE-
based and are NOT log-likelihood-calibrated. The naming distinction matters
for the paper; see ``src/evaluation/conditional_structural_information.py``
for the longer rationale.

Pipeline overview:

  1. Load per-encounter latents (E d=64 encoder) from
     ``outputs/session14/latents/S12_E_d64/{split}.npz`` and stack to
     ``Y_z`` (n, 64), ``Y_CL`` (n, 120), ``Y_phase`` (n,).
  2. Load matching wall-pressure ``p_wall`` (120, 192) from the partition-v1
     cache and build the W=17 impact-centred window per sensor:
     ``X[i, j, :] = p_wall[impact_frame-8 : impact_frame+9, j]``.
  3. Screen every (sensor, target) pair with a ridge proxy and the four
     TCSI proxies; rank with ``objective_J`` (balanced weights).
  4. Greedy forward selection at K in {8, 16, 32} using the joint feature
     vector ``concat([X_j_window for j in S])``; target is z first-PC.
  5. Baselines: ``uniform_K``, ``random_K`` (50 seeds), ``qDEIM_pressure_K``.
  6. Evaluation on test_b (and test_c for OOD diagnostic):
     ``z_R2`` (CV-honest multi-output ridge), ``C_L_R2``, ``phase_RMSE``.
  7. Decision-gate figure plus ``results.json``.

Run from the repo root after activating the venv and exporting
``PREVENT_ROOT``::

    python scripts/session14_tcsi_pilot.py

CPU-only ridge fits. Wall time 5-30 min depending on machine.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Limit BLAS thread oversubscription: each ridge solve is tiny, so a small
# thread pool is faster than letting MKL/OpenBLAS spawn 64 threads per call.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "4")
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.conditional_structural_information import (  # noqa: E402
    RidgeProxyLearner,
    compute_proxies_for_sensor,
    objective_J,
)


LATENT_DIR = REPO_ROOT / "outputs" / "session14" / "latents" / "S12_E_d64"
OUT_DIR = REPO_ROOT / "outputs" / "session14" / "tcsi_pilot"
SPLITS = ("train", "test_a", "test_b", "test_c")
PARTITION = "v1"
WINDOW = 17  # symmetric: impact_frame-8 .. impact_frame+8 inclusive
HALF_WINDOW = WINDOW // 2
N_SENSORS = 192
N_FRAMES = 120
K_VALUES: Tuple[int, ...] = (2, 3, 4, 8, 16, 32)
RANDOM_SEEDS = 50


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------


def resolve_cache_root() -> Path:
    """Return the partition-v1 cache directory.

    Mirrors the loader convention used by ``scripts/session14_encode_latents``.
    """
    env_cache = os.environ.get("VORTEX_JEPA_CACHE")
    if env_cache:
        return Path(env_cache) / PARTITION
    prevent = Path(os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT")))
    return prevent / "data" / "processed" / "vortex-jepa" / PARTITION


def load_split_records(split: str) -> Dict[str, np.ndarray]:
    """Load latent NPZ for a single split."""
    path = LATENT_DIR / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing latent NPZ: {path}")
    with np.load(path, allow_pickle=True) as data:
        out = {k: data[k] for k in data.files}
    return out


def force_impact_phase(
    cl_trace: np.ndarray,
    cl_baseline: np.ndarray,
    window: Tuple[int, int] = (25, 55),
) -> int:
    """Force-domain impact frame: argmax(|C_L - C_L_baseline|) over window.

    Baseline encounters typically have a small but non-zero peak from
    natural shedding, so the metric is defined for every row.
    """
    lo, hi = window
    diff = np.abs(cl_trace[lo:hi] - cl_baseline[lo:hi])
    return int(lo + int(np.argmax(diff)))


def build_data_arrays() -> Dict[str, Dict[str, np.ndarray]]:
    """Stack p_wall windows, C_L traces, latents, and phase per split.

    Returns a dict keyed by split name with arrays:

      * ``X_window`` (n, 192, W) - impact-centred pressure window per sensor
      * ``Y_z``      (n, 64)     - impact-frame latent (from NPZ)
      * ``Y_CL_imp`` (n,)        - C_L at the impact frame
      * ``Y_phase``  (n,)        - force-domain impact-phase frame (int)
      * ``case_id``  (n,)        - case identifier
      * ``encounter_index`` (n,) - encounter index
    """
    cache_root = resolve_cache_root()

    # Build a baseline C_L lookup so the phase metric is well-defined for
    # every encounter (including Baseline itself, which gets its own trace).
    # Baseline is in the train split with encounters 0-3 and in test_a with
    # encounters 4-5. The encounter-index is the canonical key.
    baseline_cl: Dict[int, np.ndarray] = {}
    for split in SPLITS:
        recs = load_split_records(split)
        for i, case_id in enumerate(recs["case_id"]):
            if case_id != "Baseline":
                continue
            enc = int(recs["encounter_index"][i])
            path = cache_root / "Baseline" / f"encounter_{enc:02d}.h5"
            with h5py.File(path, "r") as f:
                baseline_cl[enc] = f["C_L"][...].astype(np.float64)
    # If Baseline is missing for some encounter indices, fall back to
    # encounter 0 (close enough as a reference; phase is robust to this).
    fallback_cl = baseline_cl.get(0)
    if fallback_cl is None:
        raise RuntimeError("no Baseline encounter found across splits")

    out: Dict[str, Dict[str, np.ndarray]] = {}
    skipped: List[Tuple[str, str, int]] = []
    for split in SPLITS:
        recs = load_split_records(split)
        n = len(recs["case_id"])
        X_window = np.zeros((n, N_SENSORS, WINDOW), dtype=np.float64)
        Y_CL_imp = np.zeros(n, dtype=np.float64)
        Y_phase = np.zeros(n, dtype=np.int64)
        valid = np.ones(n, dtype=bool)
        for i in range(n):
            case_id = str(recs["case_id"][i])
            enc = int(recs["encounter_index"][i])
            impact = int(recs["impact_frame"][i])
            path = cache_root / case_id / f"encounter_{enc:02d}.h5"
            with h5py.File(path, "r") as f:
                p_wall = f["p_wall"][...].astype(np.float64)
                cl = f["C_L"][...].astype(np.float64)
            ref_cl = baseline_cl.get(enc % 6, fallback_cl)
            lo = impact - HALF_WINDOW
            hi = impact + HALF_WINDOW + 1
            if lo < 0 or hi > N_FRAMES:
                raise ValueError(
                    f"impact window [{lo}, {hi}) out of bounds for "
                    f"{case_id} encounter {enc:02d}"
                )
            window = p_wall[lo:hi].T  # (192, W)
            # Skip encounters whose impact-window pressure or C_L contains
            # NaN (some run3 trailing encounters in v1 cache are short).
            if np.isnan(window).any() or np.isnan(cl[lo:hi]).any():
                valid[i] = False
                skipped.append((split, case_id, enc))
                continue
            X_window[i] = window
            Y_CL_imp[i] = float(cl[impact])
            Y_phase[i] = force_impact_phase(cl, ref_cl)

        out[split] = {
            "X_window": X_window[valid],
            "Y_z": recs["z"].astype(np.float64)[valid],
            "Y_CL_imp": Y_CL_imp[valid],
            "Y_phase": Y_phase[valid],
            "case_id": recs["case_id"][valid],
            "encounter_index": recs["encounter_index"].astype(np.int64)[valid],
        }
    if skipped:
        print(
            f"[pilot] skipped {len(skipped)} encounter(s) with NaN p_wall in "
            f"impact window: {skipped}",
            flush=True,
        )
    return out


# ---------------------------------------------------------------------------
# Individual sensor screening (Thrust 7b)
# ---------------------------------------------------------------------------


def screen_individual_sensors(
    X_train: np.ndarray,
    Y_targets: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Fit a ridge proxy per sensor per target; return per-sensor proxies.

    Args:
        X_train: Pressure windows of shape ``(n_train, 192, W)``.
        Y_targets: Map from target name to target array of shape
            ``(n_train,)`` (1D regression).

    Returns:
        Dict keyed by target name with arrays of length 192 for each of
        ``G``, ``S_preq``, ``H_res``, ``Eff``, ``L_star``, ``L_null``,
        and the derived scalar ``J``.
    """
    n, n_sensors, _ = X_train.shape
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for tname, target in Y_targets.items():
        G_arr = np.zeros(n_sensors)
        S_arr = np.zeros(n_sensors)
        H_arr = np.zeros(n_sensors)
        Eff_arr = np.zeros(n_sensors)
        Lstar = np.zeros(n_sensors)
        Lnull = np.zeros(n_sensors)
        J_arr = np.zeros(n_sensors)
        learner = RidgeProxyLearner(alpha=1.0)
        for j in range(n_sensors):
            proxies = compute_proxies_for_sensor(
                sensor_data=X_train[:, j, :],
                target=target,
                learner=learner,
            )
            G_arr[j] = proxies.G
            S_arr[j] = proxies.S_preq
            H_arr[j] = proxies.H_res
            Eff_arr[j] = proxies.Eff
            Lstar[j] = proxies.L_star
            Lnull[j] = proxies.L_null
            J_arr[j] = objective_J(proxies)
        out[tname] = {
            "G": G_arr,
            "S_preq": S_arr,
            "H_res": H_arr,
            "Eff": Eff_arr,
            "L_star": Lstar,
            "L_null": Lnull,
            "J": J_arr,
        }
    return out


# ---------------------------------------------------------------------------
# Greedy forward selection (Thrust 7c)
# ---------------------------------------------------------------------------


def _subset_features(X_window: np.ndarray, sensors: Sequence[int]) -> np.ndarray:
    """Concatenate the W-length windows for the given sensors.

    Args:
        X_window: ``(n, 192, W)``.
        sensors: iterable of sensor indices.

    Returns:
        ``(n, len(sensors) * W)`` joint feature matrix.
    """
    if len(sensors) == 0:
        return np.zeros((X_window.shape[0], 0))
    return X_window[:, list(sensors), :].reshape(X_window.shape[0], -1)


def greedy_forward_selection(
    X_window: np.ndarray,
    target: np.ndarray,
    K: int,
    candidate_pool: Sequence[int] | None = None,
    alpha: float = 1.0,
    initial: Sequence[int] | None = None,
) -> List[int]:
    """Greedy forward selection on the joint feature vector.

    At each step, add the sensor whose inclusion maximises ``objective_J``
    of the TCSI proxies for the joint subset. Returns the selected sensor
    indices in addition order.

    Args:
        X_window: ``(n, 192, W)``.
        target: ``(n,)`` 1D target.
        K: desired subset size (final length).
        candidate_pool: candidate sensor indices (default: all 192).
        alpha: ridge alpha for the proxy learner.
        initial: optional warm-start prefix; the algorithm extends this
            list up to length ``K``. Useful for staircase ``K`` evaluation
            since greedy at smaller ``K`` is a prefix of greedy at larger
            ``K`` for identical objectives.
    """
    n, n_sensors, W = X_window.shape
    if candidate_pool is None:
        candidate_pool = list(range(n_sensors))
    pool = list(candidate_pool)

    # Pre-centre target so the learner's centring is a no-op and the
    # candidate inner loop is dominated by the small matrix solve.
    y = np.asarray(target, dtype=np.float64)
    y_centered = y - y.mean()
    L_null = float(np.mean(y_centered ** 2))

    selected: List[int] = list(initial) if initial is not None else []
    # Pre-centre the per-sensor window once.
    Xc = X_window - X_window.mean(axis=0, keepdims=True)  # (n, 192, W)

    while len(selected) < K:
        # Build (n, |S|*W) for currently-selected sensors once per step.
        if selected:
            S_feats = Xc[:, list(selected), :].reshape(n, -1)
        else:
            S_feats = np.zeros((n, 0), dtype=np.float64)

        best_j = -1
        best_score = -np.inf
        for j in pool:
            if j in selected:
                continue
            # Stack the new sensor's W-vector as additional columns.
            new_cols = Xc[:, j, :]
            X_try = np.concatenate([S_feats, new_cols], axis=1)
            d = X_try.shape[1]
            gram = X_try.T @ X_try + alpha * np.eye(d)
            rhs = X_try.T @ y_centered
            try:
                w = np.linalg.solve(gram, rhs)
            except np.linalg.LinAlgError:
                continue
            pred = X_try @ w
            L_final = float(np.mean((y_centered - pred) ** 2))
            G = float(n) * max(0.0, L_null - L_final)
            S_preq = max(0.0, L_null - L_final)  # 2-point: only positive area
            H_res = float(n) * L_final
            Eff = G / (S_preq + 1e-6)
            score = 1.0 * G + (-0.5) * S_preq + (-0.5) * H_res + 0.5 * Eff
            if score > best_score:
                best_score = score
                best_j = j
        if best_j < 0:
            break
        selected.append(best_j)
    return selected


# ---------------------------------------------------------------------------
# Baseline selectors (Thrust 7d)
# ---------------------------------------------------------------------------


def selector_uniform(K: int, n_sensors: int = N_SENSORS) -> List[int]:
    """Uniformly-spaced sensors."""
    K = min(K, n_sensors)
    return [int(round(i * n_sensors / K)) % n_sensors for i in range(K)]


def selector_random(K: int, seed: int, n_sensors: int = N_SENSORS) -> List[int]:
    """Random K of n_sensors without replacement, seeded."""
    rng = np.random.default_rng(seed)
    return sorted(int(x) for x in rng.choice(n_sensors, size=K, replace=False))


def selector_qdeim(pressure_matrix: np.ndarray, K: int) -> List[int]:
    """qDEIM selector via QR with column pivoting (Manohar et al. 2018).

    Args:
        pressure_matrix: ``(n_encounters, n_sensors)`` snapshot matrix at the
            impact frame.
        K: number of sensors to pick.

    Returns:
        Sorted sensor indices of length min(K, n_sensors).

    qDEIM works on the right singular vectors of the snapshot matrix.
    Take the top-r modes (r >= K), apply QR with column pivoting, and
    the first K pivots are the optimal interpolation points (DEIM).
    """
    A = np.asarray(pressure_matrix, dtype=np.float64)
    n_enc, n_sens = A.shape
    K_eff = min(K, n_sens)
    r = min(K_eff, n_enc, n_sens)
    # Right singular vectors V_r in (n_sens, r)
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    Vr = Vt[:r, :].T  # (n_sens, r)
    # QR with column pivoting on V_r^T (Manohar et al. 2018 Algorithm 1):
    _, _, piv = _qr_with_pivoting(Vr.T)
    return sorted(int(p) for p in piv[:K_eff])


def _qr_with_pivoting(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """QR with column pivoting via SciPy if available, else a manual loop.

    Returns ``(Q, R, perm)`` where ``perm`` is the column permutation such
    that ``A[:, perm] = Q @ R``. NumPy's qr does not pivot, so we use
    SciPy when present, with a small NumPy fallback for portability.
    """
    try:
        from scipy.linalg import qr  # type: ignore
        Q, R, perm = qr(A, pivoting=True, mode="economic")
        return Q, R, np.asarray(perm, dtype=np.int64)
    except ImportError:
        # Householder QR with greedy column pivoting (max-norm rule).
        m, n = A.shape
        perm = np.arange(n)
        R = A.astype(np.float64).copy()
        Q = np.eye(m)
        for k in range(min(m, n)):
            norms = np.linalg.norm(R[k:, k:], axis=0)
            j = int(np.argmax(norms)) + k
            if j != k:
                R[:, [k, j]] = R[:, [j, k]]
                perm[[k, j]] = perm[[j, k]]
            x = R[k:, k]
            sign = -1.0 if x[0] < 0 else 1.0
            alpha = sign * np.linalg.norm(x)
            v = x.copy()
            v[0] += alpha
            beta = 2.0 / np.dot(v, v) if np.dot(v, v) > 0 else 0.0
            R[k:, k:] -= beta * np.outer(v, v @ R[k:, k:])
            Q[:, k:] -= beta * np.outer(Q[:, k:] @ v, v)
        return Q, R, perm


# ---------------------------------------------------------------------------
# Evaluation (Thrust 7e)
# ---------------------------------------------------------------------------


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination, multi-output-mean if 2D."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.ndim == 1:
        ss_res = float(np.sum((yt - yp) ** 2))
        ss_tot = float(np.sum((yt - yt.mean()) ** 2))
        if ss_tot == 0.0:
            return float("nan")
        return 1.0 - ss_res / ss_tot
    # Multi-output: variance-weighted R^2
    ss_res = np.sum((yt - yp) ** 2, axis=0)
    ss_tot = np.sum((yt - yt.mean(axis=0, keepdims=True)) ** 2, axis=0)
    out = np.zeros_like(ss_tot)
    mask = ss_tot > 0
    out[mask] = 1.0 - ss_res[mask] / ss_tot[mask]
    out[~mask] = np.nan
    return float(np.nanmean(out))


def cv_r2(
    X_features: np.ndarray,
    Y: np.ndarray,
    n_splits: int = 5,
    alpha: float = 1.0,
    seed: int = 0,
) -> float:
    """K-fold CV R^2 with ridge regression."""
    n = X_features.shape[0]
    if n < n_splits:
        # Leave-one-out fallback if the split is tiny (e.g. 4 phases for tb_phase)
        n_splits = max(2, n)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []
    for tr_idx, te_idx in kf.split(X_features):
        model = Ridge(alpha=alpha)
        model.fit(X_features[tr_idx], Y[tr_idx])
        y_pred = model.predict(X_features[te_idx])
        y_true_all.append(Y[te_idx])
        y_pred_all.append(y_pred)
    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)
    return _r2_score(y_true, y_pred)


def cv_rmse(
    X_features: np.ndarray,
    Y: np.ndarray,
    n_splits: int = 5,
    alpha: float = 1.0,
    seed: int = 0,
) -> float:
    """K-fold CV RMSE with ridge regression (1D target)."""
    n = X_features.shape[0]
    if n < n_splits:
        n_splits = max(2, n)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []
    for tr_idx, te_idx in kf.split(X_features):
        model = Ridge(alpha=alpha)
        model.fit(X_features[tr_idx], Y[tr_idx])
        y_pred = model.predict(X_features[te_idx])
        y_true_all.append(Y[te_idx])
        y_pred_all.append(y_pred)
    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def evaluate_selector_on_split(
    sensors: Sequence[int],
    data: Dict[str, np.ndarray],
    alpha: float = 1.0,
) -> Dict[str, float]:
    """Compute z_R2, C_L_R2, phase_RMSE for a sensor subset on one split."""
    if len(sensors) == 0:
        return {"z_R2": float("nan"), "CL_R2": float("nan"), "phase_RMSE": float("nan")}
    feats = _subset_features(data["X_window"], sensors)
    z_r2 = cv_r2(feats, data["Y_z"], n_splits=5, alpha=alpha)
    cl_r2 = cv_r2(feats, data["Y_CL_imp"], n_splits=5, alpha=alpha)
    phase_rmse = cv_rmse(
        feats, data["Y_phase"].astype(np.float64), n_splits=5, alpha=alpha
    )
    return {"z_R2": float(z_r2), "CL_R2": float(cl_r2), "phase_RMSE": float(phase_rmse)}


# ---------------------------------------------------------------------------
# Decision figure
# ---------------------------------------------------------------------------


def load_airfoil_sensor_positions() -> Tuple[np.ndarray, np.ndarray]:
    """Return airfoil polygon and per-sensor (x, y) chordwise positions.

    Sensor j corresponds to ``sensors/xyz[j * 8, :2]`` (the first spanwise
    station for chordwise index j).
    """
    prevent = Path(os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT")))
    path = prevent / "data" / "raw" / "periodic" / "Baseline.h5"
    with h5py.File(path, "r") as f:
        airfoil_xy = f["airfoil_xy"][...].astype(np.float64)
        sensor_xyz = f["sensors"]["xyz"][...].astype(np.float64)
    # 1536 = 192 * 8. The chord-stride is 8.
    sensor_xy = sensor_xyz[::8, :2]
    if sensor_xy.shape[0] < N_SENSORS:
        raise RuntimeError(
            f"expected at least {N_SENSORS} chordwise sensors, got "
            f"{sensor_xy.shape[0]}"
        )
    return airfoil_xy, sensor_xy[:N_SENSORS]


def render_decision_figure(
    selections: Dict[str, Dict[int, List[int]]],
    eval_table: Dict[int, Dict[str, Dict[str, float]]],
    random_table: Dict[int, Dict[str, Dict[str, float]]],
    output_path: Path,
) -> None:
    """Two-panel decision figure.

    Left: airfoil chord with K=16 TCSI sensors highlighted.
    Right: grouped bar chart of z_R2 across selectors and K values.
    """
    airfoil, sensor_xy = load_airfoil_sensor_positions()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left: airfoil + K=16 selected sensors ---
    ax = axes[0]
    poly = Polygon(airfoil, closed=True, facecolor="0.85", edgecolor="black",
                   linewidth=0.7, zorder=1)
    ax.add_patch(poly)
    ax.scatter(sensor_xy[:, 0], sensor_xy[:, 1], s=4, color="0.4",
               alpha=0.4, zorder=2, label="all 192 sensors")
    k16 = selections["conditional_SI"][16]
    ax.scatter(sensor_xy[k16, 0], sensor_xy[k16, 1], s=60,
               color="tab:red", edgecolor="black", linewidth=0.5, zorder=3,
               label=f"TCSI K=16 (n={len(k16)})")
    ax.set_aspect("equal")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.10, 0.10)
    ax.set_xlabel("x / c")
    ax.set_ylabel("y / c")
    ax.set_title("TCSI-selected K=16 pressure sensors")
    ax.legend(loc="upper right", fontsize=8)

    # --- Right: grouped bars for z_R2 ---
    ax = axes[1]
    selector_order = ["uniform_K", "random_K", "qDEIM", "conditional_SI", "all_192"]
    colors = {
        "uniform_K": "tab:blue",
        "random_K": "tab:gray",
        "qDEIM": "tab:orange",
        "conditional_SI": "tab:red",
        "all_192": "0.2",
    }
    n_k = len(K_VALUES)
    n_sel = len(selector_order)
    bar_w = 0.8 / n_sel
    x_pos = np.arange(n_k)
    for s_idx, sel in enumerate(selector_order):
        y_vals = []
        y_errs = []
        for K in K_VALUES:
            entry = eval_table[K].get(sel, {})
            r2 = entry.get("test_b", {}).get("z_R2", float("nan"))
            y_vals.append(r2)
            if sel == "random_K":
                r2s = random_table[K].get("z_R2_list", [])
                if len(r2s) > 0:
                    y_errs.append(float(np.std(r2s)))
                else:
                    y_errs.append(0.0)
            else:
                y_errs.append(0.0)
        ax.bar(
            x_pos + (s_idx - (n_sel - 1) / 2) * bar_w,
            y_vals,
            width=bar_w,
            yerr=y_errs,
            label=sel,
            color=colors[sel],
            capsize=3,
            edgecolor="black",
            linewidth=0.3,
        )
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"K={K}" for K in K_VALUES])
    ax.set_ylabel(r"$z\;R^2$ on Test B (5-fold CV)")
    ax.set_ylim(-0.1, 1.05)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Sensor-selector comparison: latent recovery")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Decision gate
# ---------------------------------------------------------------------------


def evaluate_decision_gate(
    eval_table: Dict[int, Dict[str, Dict[str, float]]],
    random_table: Dict[int, Dict[str, Dict[str, float]]],
) -> Dict[str, object]:
    """Apply the decision criteria from the plan."""
    reasons: List[str] = []
    tcsi_16 = eval_table[16]["conditional_SI"]["test_b"]
    uni_16 = eval_table[16]["uniform_K"]["test_b"]
    rnd_16_med = eval_table[16]["random_K"]["test_b"]
    rnd_16_std_z = float(np.std(random_table[16].get("z_R2_list", [])) or 0.0)
    rnd_16_std_cl = float(np.std(random_table[16].get("CL_R2_list", [])) or 0.0)
    rnd_16_std_phase = float(np.std(random_table[16].get("phase_RMSE_list", [])) or 0.0)
    all_192_z = eval_table[16]["all_192"]["test_b"]["z_R2"]
    qdeim_16 = eval_table[16]["qDEIM"]["test_b"]

    # Pass criterion 1: K=16 TCSI beats uniform AND random_median by > 1 std on
    # at least 2 of 3 target metrics. Higher-is-better for z_R2 and CL_R2;
    # lower-is-better for phase_RMSE.
    wins = []
    if (tcsi_16["z_R2"] > uni_16["z_R2"]) and (
        tcsi_16["z_R2"] - rnd_16_med["z_R2"] > rnd_16_std_z
    ):
        wins.append("z_R2")
    if (tcsi_16["CL_R2"] > uni_16["CL_R2"]) and (
        tcsi_16["CL_R2"] - rnd_16_med["CL_R2"] > rnd_16_std_cl
    ):
        wins.append("CL_R2")
    if (tcsi_16["phase_RMSE"] < uni_16["phase_RMSE"]) and (
        rnd_16_med["phase_RMSE"] - tcsi_16["phase_RMSE"] > rnd_16_std_phase
    ):
        wins.append("phase_RMSE")

    crit1_pass = len(wins) >= 2
    reasons.append(
        f"PASS-1 (TCSI K=16 beats uniform_K AND random_K_median by >1std on "
        f">=2 of 3 metrics): {'PASS' if crit1_pass else 'FAIL'} "
        f"(wins on {wins})"
    )

    crit2_pass = (
        tcsi_16["z_R2"] > 0.85
        and tcsi_16["CL_R2"] > 0.95
        and tcsi_16["phase_RMSE"] < 3.0
    )
    reasons.append(
        f"PASS-2 (z_R2>0.85 AND CL_R2>0.95 AND phase_RMSE<3 at K=16): "
        f"{'PASS' if crit2_pass else 'FAIL'} (z_R2={tcsi_16['z_R2']:.3f}, "
        f"CL_R2={tcsi_16['CL_R2']:.3f}, phase_RMSE={tcsi_16['phase_RMSE']:.2f})"
    )

    tcsi_32_z = eval_table[32]["conditional_SI"]["test_b"]["z_R2"]
    if all_192_z > 0:
        gap_16 = (all_192_z - tcsi_16["z_R2"]) / abs(all_192_z)
        gap_32 = (all_192_z - tcsi_32_z) / abs(all_192_z)
    else:
        gap_16 = float("inf")
        gap_32 = float("inf")
    crit3_pass = min(gap_16, gap_32) < 0.05
    reasons.append(
        f"PASS-3 (K=16 OR K=32 within 5% of all_192 z_R2={all_192_z:.3f}): "
        f"{'PASS' if crit3_pass else 'FAIL'} (gap_16={gap_16:.3f}, "
        f"gap_32={gap_32:.3f})"
    )

    # Fail conditions (any one => fail).
    fail_a = tcsi_16["z_R2"] < rnd_16_med["z_R2"]
    fail_b = tcsi_16["z_R2"] < 0.7
    fail_c = abs(qdeim_16["z_R2"] - tcsi_16["z_R2"]) < 0.02 and (
        qdeim_16["z_R2"] > 0.5
    )
    if fail_a:
        reasons.append("FAIL-A (TCSI K=16 z_R2 < random_K_median z_R2): TRUE")
    if fail_b:
        reasons.append("FAIL-B (TCSI K=16 z_R2 < 0.7): TRUE")
    if fail_c:
        reasons.append("FAIL-C (qDEIM comparable to TCSI within 0.02): TRUE")

    overall = crit1_pass and crit2_pass and crit3_pass and not (
        fail_a or fail_b or fail_c
    )
    return {"pass": bool(overall), "reasons": reasons}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _tolist(o: object) -> object:
    """JSON-friendly recursive cast."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, dict):
        return {k: _tolist(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_tolist(x) for x in o]
    return o


def main(argv: Sequence[str] | None = None) -> int:
    """Run the TCSI pilot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-step progress prints",
    )
    args = parser.parse_args(argv)

    OUT = Path(args.output_dir)
    OUT.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[pilot] loading splits from {LATENT_DIR}", flush=True)
    data = build_data_arrays()
    for split, recs in data.items():
        print(
            f"[pilot]  split={split:7s} n={recs['Y_z'].shape[0]:3d} "
            f"X_window={recs['X_window'].shape} "
            f"Y_z={recs['Y_z'].shape} Y_phase_unique={len(np.unique(recs['Y_phase']))}",
            flush=True,
        )

    # Selection pool: train + test_a (CLAUDE.md leakage rule).
    X_sel = np.concatenate([data["train"]["X_window"], data["test_a"]["X_window"]], axis=0)
    Y_z_sel = np.concatenate([data["train"]["Y_z"], data["test_a"]["Y_z"]], axis=0)
    Y_CL_sel = np.concatenate(
        [data["train"]["Y_CL_imp"], data["test_a"]["Y_CL_imp"]], axis=0
    )
    Y_phase_sel = np.concatenate(
        [data["train"]["Y_phase"], data["test_a"]["Y_phase"]], axis=0
    ).astype(np.float64)
    print(
        f"[pilot] selection pool n={X_sel.shape[0]} "
        f"(train+test_a; phase_unique={len(np.unique(Y_phase_sel))})",
        flush=True,
    )

    # ----- Thrust 7b: individual sensor screening -----
    # Target_z is scored on the FIRST PC of z (1D) per the plan's choice (a).
    pca_z = PCA(n_components=1, random_state=0)
    Y_z_pc1 = pca_z.fit_transform(Y_z_sel).reshape(-1)
    print(
        f"[pilot] PCA(z) first-PC explained variance ratio: "
        f"{float(pca_z.explained_variance_ratio_[0]):.3f}",
        flush=True,
    )

    targets_for_screening = {
        "target_z": Y_z_pc1,
        "target_CL": Y_CL_sel,
        "target_phase": Y_phase_sel,
    }
    print("[pilot] running per-sensor TCSI screening (192 * 3 = 576 ridge fits)",
          flush=True)
    t_screen = time.time()
    individual = screen_individual_sensors(X_sel, targets_for_screening)
    print(f"[pilot]  screening done in {time.time() - t_screen:.1f}s", flush=True)

    individual_top32 = {
        tname: {
            "top_32_sensors": [int(i) for i in np.argsort(stats["J"])[-32:][::-1]],
            "G_per_sensor": stats["G"].tolist(),
        }
        for tname, stats in individual.items()
    }

    # ----- Thrust 7c: greedy forward selection -----
    # Warm-start: greedy at K=8 is a prefix of greedy at K=16, which is a
    # prefix of K=32 (objective monotone, same target). Compute once at the
    # largest K and slice.
    print("[pilot] running greedy forward selection (target = z first-PC)",
          flush=True)
    greedy_selections: Dict[int, List[int]] = {}
    sorted_K = sorted(K_VALUES)
    chain: List[int] = []
    t_g_total = time.time()
    for K in sorted_K:
        t_g = time.time()
        chain = greedy_forward_selection(
            X_sel, Y_z_pc1, K=K, initial=chain,
        )
        greedy_selections[K] = list(chain)
        print(
            f"[pilot]  K={K:2d} sensors={chain} "
            f"({time.time() - t_g:.1f}s; cumulative {time.time() - t_g_total:.1f}s)",
            flush=True,
        )

    # ----- Thrust 7d: baselines + Thrust 7e: evaluation -----
    # Precompute impact-frame pressure matrix for qDEIM (train+test_a pool).
    P_imp_sel = X_sel[:, :, HALF_WINDOW]  # (n, 192) at impact frame

    selections: Dict[str, Dict[int, List[int]]] = {
        "uniform_K": {},
        "qDEIM": {},
        "conditional_SI": {},
    }
    for K in K_VALUES:
        selections["uniform_K"][K] = selector_uniform(K)
        selections["qDEIM"][K] = selector_qdeim(P_imp_sel, K)
        selections["conditional_SI"][K] = greedy_selections[K]

    eval_table: Dict[int, Dict[str, Dict[str, Dict[str, float]]]] = {K: {} for K in K_VALUES}
    random_table: Dict[int, Dict[str, list]] = {K: {} for K in K_VALUES}

    for K in K_VALUES:
        for sel_name in ("uniform_K", "qDEIM", "conditional_SI"):
            sensors = selections[sel_name][K]
            eval_table[K][sel_name] = {
                "sensors": [int(s) for s in sensors],
                "test_b": evaluate_selector_on_split(sensors, data["test_b"]),
                "test_c": evaluate_selector_on_split(sensors, data["test_c"]),
            }

        # Random baseline with 50 seeds: median + per-seed lists for std.
        z_R2s, CL_R2s, phase_RMSEs = [], [], []
        z_R2s_c, CL_R2s_c, phase_RMSEs_c = [], [], []
        all_sel: List[List[int]] = []
        for s in range(RANDOM_SEEDS):
            sensors = selector_random(K, seed=s)
            all_sel.append(sensors)
            tb = evaluate_selector_on_split(sensors, data["test_b"])
            tc = evaluate_selector_on_split(sensors, data["test_c"])
            z_R2s.append(tb["z_R2"])
            CL_R2s.append(tb["CL_R2"])
            phase_RMSEs.append(tb["phase_RMSE"])
            z_R2s_c.append(tc["z_R2"])
            CL_R2s_c.append(tc["CL_R2"])
            phase_RMSEs_c.append(tc["phase_RMSE"])
        eval_table[K]["random_K"] = {
            "sensors_example_seed0": [int(s) for s in all_sel[0]],
            "test_b": {
                "z_R2": float(np.median(z_R2s)),
                "z_R2_p2_5": float(np.percentile(z_R2s, 2.5)),
                "z_R2_p97_5": float(np.percentile(z_R2s, 97.5)),
                "CL_R2": float(np.median(CL_R2s)),
                "CL_R2_p2_5": float(np.percentile(CL_R2s, 2.5)),
                "CL_R2_p97_5": float(np.percentile(CL_R2s, 97.5)),
                "phase_RMSE": float(np.median(phase_RMSEs)),
                "phase_RMSE_p2_5": float(np.percentile(phase_RMSEs, 2.5)),
                "phase_RMSE_p97_5": float(np.percentile(phase_RMSEs, 97.5)),
            },
            "test_c": {
                "z_R2": float(np.median(z_R2s_c)),
                "CL_R2": float(np.median(CL_R2s_c)),
                "phase_RMSE": float(np.median(phase_RMSEs_c)),
            },
        }
        random_table[K] = {
            "z_R2_list": z_R2s,
            "CL_R2_list": CL_R2s,
            "phase_RMSE_list": phase_RMSEs,
        }

        # all_192 reference.
        all_sensors = list(range(N_SENSORS))
        eval_table[K]["all_192"] = {
            "sensors": all_sensors,
            "test_b": evaluate_selector_on_split(all_sensors, data["test_b"]),
            "test_c": evaluate_selector_on_split(all_sensors, data["test_c"]),
        }
        print(f"[pilot] K={K:2d} eval done", flush=True)
        for name in ("uniform_K", "random_K", "qDEIM", "conditional_SI", "all_192"):
            tb = eval_table[K][name]["test_b"]
            print(
                f"[pilot]    {name:16s} z_R2={tb['z_R2']:.3f}  "
                f"CL_R2={tb['CL_R2']:.3f}  phase_RMSE={tb['phase_RMSE']:.2f}",
                flush=True,
            )

    # ----- Decision gate -----
    eval_for_gate = {K: {sel: v for sel, v in eval_table[K].items()} for K in K_VALUES}
    decision = evaluate_decision_gate(eval_for_gate, random_table)
    print("[pilot] DECISION GATE:", "PASS" if decision["pass"] else "FAIL", flush=True)
    for r in decision["reasons"]:
        print(f"[pilot]   {r}", flush=True)

    # ----- Figure -----
    figure_path = OUT / "decision_figure.png"
    render_decision_figure(selections, eval_for_gate, random_table, figure_path)
    print(f"[pilot] decision figure -> {figure_path}", flush=True)

    # ----- JSON results -----
    results = {
        "K_values": list(K_VALUES),
        "selectors": ["uniform_K", "random_K", "qDEIM", "conditional_SI"],
        "window_size": WINDOW,
        "random_seeds": RANDOM_SEEDS,
        "selection_pool": {
            "splits": ["train", "test_a"],
            "n_train": int(data["train"]["Y_z"].shape[0]),
            "n_test_a": int(data["test_a"]["Y_z"].shape[0]),
            "n_total": int(X_sel.shape[0]),
        },
        "phase_target_definition": (
            "argmax(|C_L - C_L_baseline|) over frames [25, 55]; "
            "baseline = Baseline-case C_L for the matching encounter index"
        ),
        "z_target_definition": "first principal component of E d=64 latent z_impact",
        "pca_z_first_pc_explained_variance_ratio": float(
            pca_z.explained_variance_ratio_[0]
        ),
        "individual_screening": individual_top32,
        "greedy_selections": {
            f"K_{K}": [int(s) for s in greedy_selections[K]] for K in K_VALUES
        },
        "evaluation": {f"K_{K}": eval_for_gate[K] for K in K_VALUES},
        "random_baseline_std": {
            f"K_{K}": {
                "z_R2_std": float(np.std(random_table[K]["z_R2_list"])),
                "CL_R2_std": float(np.std(random_table[K]["CL_R2_list"])),
                "phase_RMSE_std": float(np.std(random_table[K]["phase_RMSE_list"])),
            }
            for K in K_VALUES
        },
        "decision_gate": decision,
        "wall_time_seconds": time.time() - t0,
    }
    results_path = OUT / "results.json"
    with open(results_path, "w") as f:
        json.dump(_tolist(results), f, indent=2)
    print(f"[pilot] results -> {results_path}", flush=True)
    print(f"[pilot] wall_time = {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
