"""Session 23: calibrated log-likelihood / MI sensor selection vs MSE-TCSI.

Question (to VALIDATE, not assume).
===================================
The paper places K wall-pressure taps to recover the predictive (JEPA) latent.
The current method is TCSI (``scripts/session14_tcsi_pilot.py`` +
``src/evaluation/conditional_structural_information.py``): greedy forward
selection scored by MSE-based "structural-information" proxies with a cheap
ridge learner. It is NOT log-likelihood calibrated. The project lead asked
whether a CALIBRATED Gaussian log-likelihood / mutual-information selection
differs from, beats, or matches MSE-TCSI, and whether the calibrated approach
is really "fragile at small n" or "too expensive". These are claims to test.

What this script does.
======================
Reuses the Session-14 machinery (do NOT re-derive):
  * ``build_data_arrays``  -> per-split ``X_window`` (n, 192, W=17) impact-centred
    pressure windows + ``Y_z`` (n, 64) impact-frame JEPA latent (E d=64).
  * ``greedy_forward_selection`` (MSE-TCSI greedy, used for a sanity re-derive).
  * ``selector_qdeim`` / ``selector_uniform`` (reference baselines).
  * the CV-honest multi-output ridge recovery R^2 (here upgraded to GroupKFold
    by case_id, the protocol the paper asks for; plain KFold is also reported).

Reference picks (MSE-TCSI, qDEIM) are loaded from the frozen
``outputs/session21/pressure_v2/sensor_picks_v2.json``. Note (reported
honestly): those v2 picks were derived on the production noBN latent with a
W=30 pre-impact window; the recovery here scores against the S12_E_d64 latent
with the W=17 impact-centred window that the Session-14 machinery builds. The
SELECTION pool, target, window, and recovery metric are held identical across
every method compared in THIS script, so the head-to-head is apples-to-apples;
the only cross-provenance item is which exact taps the reference picks name.

Selection methods compared at K in {2, 4, 8}:
  1. MSE-TCSI (reference picks, loaded).
  2. qDEIM     (reference picks, loaded).
  3. Calibrated Gaussian log-likelihood greedy:
       (a) target = latent PC1 (scalar), held-out predictive Gaussian
           log-likelihood with variance from held-out residuals;
       (b) target = full d=64 latent, diagonal held-out residual noise
           covariance, multi-output predictive Gaussian log-likelihood.
  4. Gaussian MI greedy:
       (a) scalar PC1 target, MI = -0.5 log(1 - rho^2) from held-out
           predicted-vs-true correlation;
       (b) full d=64 target, summed canonical Gaussian MI from held-out
           predicted-vs-true per-dim correlations (block-diagonal proxy).
  5. OPTIONAL kNN (Kraskov) MI greedy, attempted under a wall-clock budget.

For each method's K picks: downstream test_b latent-recovery R^2 (z_R2) with the
SAME GroupKFold(case_id) multi-output ridge -> the apples-to-apples metric.

Fragility + cost:
  * pick stability via case-level bootstrap (Jaccard of K-set vs the full-data
    pick) and 5-95 pct spread of downstream z_R2;
  * overfitting check: TRAIN objective vs HELD-OUT recovery for the calibrated
    objectives (does the objective inflate while recovery stays flat/unstable?);
  * per-method wall-clock seconds.

Outputs: ``outputs/session23/calibrated_sensor_select/{results.json,
results.csv, results.png}``.

CPU only. No .tex is touched.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# Keep BLAS pools small: many tiny ridge solves.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.model_selection import GroupKFold, KFold  # noqa: E402

from session14_tcsi_pilot import (  # noqa: E402
    build_data_arrays,
    greedy_forward_selection,
    selector_qdeim,
    selector_uniform,
)

PICKS_V2 = REPO / "outputs" / "session21" / "pressure_v2" / "sensor_picks_v2.json"
OUT_DIR = REPO / "outputs" / "session23" / "calibrated_sensor_select"
K_VALUES: Tuple[int, ...] = (2, 4, 8)
N_SENSORS = 192
RIDGE_ALPHA = 1.0
CV_FOLDS_SELECT = 5      # GroupKFold folds for the calibrated SELECTION objective
CV_FOLDS_RECOVER = 5     # GroupKFold folds for the downstream recovery z_R2
N_BOOTSTRAP = 50
KNN_TIME_BUDGET_S = 180.0   # skip Kraskov MI greedy if it would exceed this


# ---------------------------------------------------------------------------
# Feature assembly helpers (mirror session14 _subset_features)
# ---------------------------------------------------------------------------


def subset_features(X_window: np.ndarray, sensors: Sequence[int]) -> np.ndarray:
    """Concatenate the W-length windows for the given sensors -> (n, |S|*W)."""
    if len(sensors) == 0:
        return np.zeros((X_window.shape[0], 0))
    return X_window[:, list(sensors), :].reshape(X_window.shape[0], -1)


def _r2_multi(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Variance-weighted (mean over dims) R^2; matches session14 _r2_score."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.ndim == 1:
        ss_res = float(np.sum((yt - yp) ** 2))
        ss_tot = float(np.sum((yt - yt.mean()) ** 2))
        return float("nan") if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    ss_res = np.sum((yt - yp) ** 2, axis=0)
    ss_tot = np.sum((yt - yt.mean(axis=0, keepdims=True)) ** 2, axis=0)
    out = np.full_like(ss_tot, np.nan)
    mask = ss_tot > 0
    out[mask] = 1.0 - ss_res[mask] / ss_tot[mask]
    return float(np.nanmean(out))


def _group_folds(groups: np.ndarray, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """GroupKFold index pairs, clamped to n_unique_groups folds."""
    n_groups = len(np.unique(groups))
    n_splits = int(min(n_splits, n_groups))
    n_splits = max(2, n_splits)
    gkf = GroupKFold(n_splits=n_splits)
    dummy = np.zeros(len(groups))
    return list(gkf.split(dummy, dummy, groups))


# ---------------------------------------------------------------------------
# Recovery metric: GroupKFold(case_id) multi-output ridge z_R2 on a split
# ---------------------------------------------------------------------------


def cv_group_r2(
    X_features: np.ndarray,
    Y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = CV_FOLDS_RECOVER,
    alpha: float = RIDGE_ALPHA,
) -> float:
    """GroupKFold (group = case_id) multi-output ridge R^2 (pooled predictions)."""
    folds = _group_folds(groups, n_splits)
    yt_all, yp_all = [], []
    for tr, te in folds:
        model = Ridge(alpha=alpha)
        model.fit(X_features[tr], Y[tr])
        yp_all.append(model.predict(X_features[te]))
        yt_all.append(Y[te])
    return _r2_multi(np.concatenate(yt_all, 0), np.concatenate(yp_all, 0))


def cv_plain_r2(
    X_features: np.ndarray,
    Y: np.ndarray,
    n_splits: int = CV_FOLDS_RECOVER,
    alpha: float = RIDGE_ALPHA,
    seed: int = 0,
) -> float:
    """Plain shuffled KFold ridge R^2 (matches session14 cv_r2, for transparency)."""
    n = X_features.shape[0]
    n_splits = min(n_splits, n) if n >= 2 else 2
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    yt_all, yp_all = [], []
    for tr, te in kf.split(X_features):
        model = Ridge(alpha=alpha)
        model.fit(X_features[tr], Y[tr])
        yp_all.append(model.predict(X_features[te]))
        yt_all.append(Y[te])
    return _r2_multi(np.concatenate(yt_all, 0), np.concatenate(yp_all, 0))


# ---------------------------------------------------------------------------
# Held-out predictions for the calibrated objectives (group = case_id)
# ---------------------------------------------------------------------------


def heldout_predictions(
    X_feats: np.ndarray,
    Y: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    alpha: float = RIDGE_ALPHA,
    standardize: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Out-of-fold (held-out) predictions and the matching truths, in fold order.

    Returns (Y_true_oof, Y_pred_oof) both reordered as concatenation of test
    folds. Used to estimate held-out predictive Gaussian variance / correlation.

    If ``standardize`` is True, a StandardScaler is fit on each TRAIN fold and
    applied to the held-out fold before the ridge. Standardising the raw
    pressure features is the textbook practice for a calibrated estimator and
    fixes the out-of-group magnitude miscalibration that an unstandardised
    single-tap ridge suffers; we report both raw and standardised log-lik so
    the comparison to MSE-TCSI (which uses raw features) is honest.
    """
    from sklearn.preprocessing import StandardScaler
    yt_all, yp_all = [], []
    for tr, te in folds:
        Xtr, Xte = X_feats[tr], X_feats[te]
        if standardize:
            sx = StandardScaler().fit(Xtr)
            Xtr, Xte = sx.transform(Xtr), sx.transform(Xte)
        model = Ridge(alpha=alpha)
        model.fit(Xtr, Y[tr])
        yp_all.append(model.predict(Xte))
        yt_all.append(Y[te])
    return np.concatenate(yt_all, 0), np.concatenate(yp_all, 0)


def heldout_gaussian_loglik_scalar(
    X_feats: np.ndarray, y: np.ndarray, folds, alpha=RIDGE_ALPHA, eps=1e-9,
    standardize=False,
) -> float:
    """Mean held-out predictive Gaussian log-likelihood for a SCALAR target.

    The predictive variance is estimated from the held-out residuals (a single
    pooled sigma^2), so the log-likelihood is properly calibrated on data the
    selection never fit. Higher is better.
    """
    yt, yp = heldout_predictions(X_feats, y.reshape(-1, 1), folds, alpha, standardize)
    resid = (yt - yp).ravel()
    var = float(np.mean(resid ** 2)) + eps
    n = resid.size
    ll = -0.5 * n * np.log(2.0 * np.pi * var) - 0.5 * np.sum(resid ** 2) / var
    return float(ll / n)  # per-sample, comparable across subset sizes


def heldout_gaussian_loglik_multi(
    X_feats: np.ndarray, Y: np.ndarray, folds, alpha=RIDGE_ALPHA, eps=1e-9,
    standardize=False,
) -> float:
    """Mean held-out predictive Gaussian log-likelihood for a MULTI-OUTPUT target.

    Uses a DIAGONAL held-out residual noise covariance (per-dim sigma_d^2 from
    held-out residuals). Per-sample log-likelihood summed over output dims.
    Higher is better.
    """
    yt, yp = heldout_predictions(X_feats, Y, folds, alpha, standardize)
    resid = yt - yp                      # (n, d)
    var = np.mean(resid ** 2, axis=0) + eps   # (d,)
    n, d = resid.shape
    ll = -0.5 * np.sum(np.log(2.0 * np.pi * var))
    ll += -0.5 * np.mean(np.sum(resid ** 2 / var[None, :], axis=1))
    return float(ll)  # per-sample (mean over n), summed over d


def heldout_gaussian_mi_scalar(
    X_feats: np.ndarray, y: np.ndarray, folds, alpha=RIDGE_ALPHA
) -> float:
    """Gaussian MI proxy for a scalar target: -0.5 log(1 - rho^2).

    rho is the correlation between held-out predictions and truth. This is the
    classic Gaussian MI between target and the (predictable part of the) joint
    sensor window. Higher is better; clipped for numerical safety.
    """
    yt, yp = heldout_predictions(X_feats, y.reshape(-1, 1), folds, alpha)
    yt, yp = yt.ravel(), yp.ravel()
    if np.std(yp) < 1e-12 or np.std(yt) < 1e-12:
        return 0.0
    rho = float(np.corrcoef(yt, yp)[0, 1])
    rho2 = min(rho ** 2, 1.0 - 1e-12)
    return float(-0.5 * np.log(1.0 - rho2))


def heldout_gaussian_mi_multi(
    X_feats: np.ndarray, Y: np.ndarray, folds, alpha=RIDGE_ALPHA
) -> float:
    """Multi-output Gaussian MI proxy: sum over dims of -0.5 log(1 - rho_d^2).

    Per-dim correlation between held-out predictions and truth, summed (a
    block-diagonal canonical proxy). Higher is better.
    """
    yt, yp = heldout_predictions(X_feats, Y, folds, alpha)
    total = 0.0
    for k in range(Y.shape[1]):
        a, b = yt[:, k], yp[:, k]
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            continue
        rho = float(np.corrcoef(a, b)[0, 1])
        rho2 = min(rho ** 2, 1.0 - 1e-12)
        total += -0.5 * np.log(1.0 - rho2)
    return float(total)


# ---------------------------------------------------------------------------
# Generic greedy forward selection over an arbitrary scoring callable
# ---------------------------------------------------------------------------


def greedy_generic(
    X_window: np.ndarray,
    score_fn,
    K: int,
    candidate_pool: Sequence[int] | None = None,
    initial: Sequence[int] | None = None,
    verbose: bool = False,
    tag: str = "",
) -> List[int]:
    """Greedy forward selection maximising ``score_fn(feature_matrix)``.

    ``score_fn`` takes the joint flattened (n, |S|*W) feature matrix of the
    candidate subset and returns a scalar to MAXIMISE. The window subsetting is
    identical to the MSE-TCSI greedy so the only difference between methods is
    the objective.
    """
    n_sensors = X_window.shape[1]
    pool = list(range(n_sensors)) if candidate_pool is None else list(candidate_pool)
    selected: List[int] = list(initial) if initial is not None else []
    while len(selected) < K:
        best_j, best_score = -1, -np.inf
        for j in pool:
            if j in selected:
                continue
            cand = selected + [j]
            feats = subset_features(X_window, cand)
            s = score_fn(feats)
            if s > best_score:
                best_score, best_j = s, j
        if best_j < 0:
            break
        selected.append(best_j)
        if verbose:
            print(f"    [{tag}] K={len(selected)} -> {best_j} (score={best_score:.4f})",
                  flush=True)
    return selected


# ---------------------------------------------------------------------------
# OPTIONAL: Kraskov kNN mutual information (model-free), scalar PC1 target
# ---------------------------------------------------------------------------


def knn_mi_scalar(X_feats: np.ndarray, y: np.ndarray, n_neighbors: int = 3) -> float:
    """Kraskov MI estimate between the joint feature vector and scalar y.

    Uses sklearn's mutual_info_regression on a low-dim PCA compression of the
    joint window (kNN MI degrades badly in high dimension; we compress the
    |S|*W feature block to min(8, d) PCA components first, which keeps the
    estimator usable and is model-free w.r.t. the target). Returned in nats.
    """
    from sklearn.feature_selection import mutual_info_regression
    Xc = X_feats - X_feats.mean(0, keepdims=True)
    n_comp = int(min(8, Xc.shape[1], max(1, Xc.shape[0] - 1)))
    if n_comp < Xc.shape[1]:
        Xc = PCA(n_components=n_comp, random_state=0).fit_transform(Xc)
    mi = mutual_info_regression(
        Xc, y.ravel(), n_neighbors=n_neighbors, random_state=0
    )
    # mutual_info_regression returns per-feature MI; sum as a joint upper proxy.
    return float(np.sum(mi))


# ---------------------------------------------------------------------------
# Downstream recovery evaluation for a given pick
# ---------------------------------------------------------------------------
#
# Three recovery protocols, reported side by side because they answer
# different questions and disagree sharply (this disagreement is itself a
# finding worth recording):
#
#   z_R2_train_tb_ridge : PRIMARY. Fit a multi-output linear ridge on the FULL
#       selection pool (train+test_a) and evaluate on the held-out test_b
#       split. Leakage-free by construction (the pool and test_b are DISJOINT
#       case sets in split_v1), so this is the honest, group-clean recovery and
#       the one whose magnitude matches the paper's positive numbers.
#   z_R2_train_tb_krr   : same fit-on-pool / eval-on-test_b protocol but with
#       the paper's KernelRidge(RBF) estimator (StandardScaler x/y, alpha=0.1,
#       gamma=0.01) -> reproduces the pressure_obs_v2.csv recovery figure.
#   z_R2_group          : STRESS diagnostic. GroupKFold(case_id) using ONLY the
#       42 test_b rows; a linear ridge fit on ~34 rows cannot recover a 64-d
#       latent for held-out cases, so this is uniformly negative for every
#       method. Reported to show the metric is underpowered, NOT as the verdict.


def _krr_fit_predict(Xtr, Ytr, Xte):
    """Paper KernelRidge(RBF) recovery: StandardScaler x/y, alpha=0.1, gamma=0.01."""
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.preprocessing import StandardScaler
    sx = StandardScaler().fit(Xtr)
    sy = StandardScaler().fit(Ytr)
    m = KernelRidge(alpha=0.1, kernel="rbf", gamma=0.01)
    m.fit(sx.transform(Xtr), sy.transform(Ytr))
    return sy.inverse_transform(m.predict(sx.transform(Xte)))


def recovery_z_r2(
    sensors: Sequence[int],
    X_pool: np.ndarray,
    Y_pool: np.ndarray,
    X_tb: np.ndarray,
    Y_tb: np.ndarray,
    groups_tb: np.ndarray,
    include_krr: bool = True,
) -> Dict[str, float]:
    """Downstream test_b latent recovery R^2 under the three protocols above."""
    if len(sensors) == 0:
        return {"z_R2_train_tb_ridge": float("nan"),
                "z_R2_train_tb_krr": float("nan"),
                "z_R2_group": float("nan")}
    feats_pool = subset_features(X_pool, sensors)
    feats_tb = subset_features(X_tb, sensors)
    # PRIMARY: linear ridge fit on pool, eval test_b.
    model = Ridge(alpha=RIDGE_ALPHA).fit(feats_pool, Y_pool)
    r2_ridge = _r2_multi(Y_tb, model.predict(feats_tb))
    out = {"z_R2_train_tb_ridge": r2_ridge,
           "z_R2_group": cv_group_r2(feats_tb, Y_tb, groups_tb)}
    if include_krr:
        out["z_R2_train_tb_krr"] = _r2_multi(
            Y_tb, _krr_fit_predict(feats_pool, Y_pool, feats_tb))
    else:
        out["z_R2_train_tb_krr"] = float("nan")
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _tolist(o):
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


def jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ---- 1. Assemble data via Session-14 machinery ----
    print("[s23] building data arrays (W=17 impact-centred windows)...", flush=True)
    data = build_data_arrays()
    for sp, recs in data.items():
        print(f"[s23]   {sp:7s} n={recs['Y_z'].shape[0]:3d} X={recs['X_window'].shape} "
              f"cases={len(np.unique(recs['case_id']))}", flush=True)

    # Selection pool = train + test_a (CLAUDE.md leakage rule, same as TCSI pilot).
    X_sel = np.concatenate([data["train"]["X_window"], data["test_a"]["X_window"]], 0)
    Y_z_sel = np.concatenate([data["train"]["Y_z"], data["test_a"]["Y_z"]], 0)
    groups_sel = np.concatenate(
        [data["train"]["case_id"].astype(str), data["test_a"]["case_id"].astype(str)], 0
    )
    n_train = int(data["train"]["Y_z"].shape[0])
    n_test_a = int(data["test_a"]["Y_z"].shape[0])
    n_test_b = int(data["test_b"]["Y_z"].shape[0])
    n_pool = int(X_sel.shape[0])
    n_pool_cases = int(len(np.unique(groups_sel)))
    print(f"[s23] selection pool n={n_pool} cases={n_pool_cases}; "
          f"n_train={n_train} n_test_a={n_test_a} n_test_b={n_test_b}", flush=True)

    # PC1 scalar target (same convention as TCSI: first PC of the latent).
    pca = PCA(n_components=1, random_state=0)
    y_pc1 = pca.fit_transform(Y_z_sel).ravel()
    print(f"[s23] PC1 explained-variance ratio = "
          f"{float(pca.explained_variance_ratio_[0]):.3f}", flush=True)

    # GroupKFold folds over the selection pool, reused by every calibrated objective.
    sel_folds = _group_folds(groups_sel, CV_FOLDS_SELECT)
    print(f"[s23] selection CV = GroupKFold(case_id), {len(sel_folds)} folds", flush=True)

    # test_b arrays for downstream recovery.
    X_tb = data["test_b"]["X_window"]
    Y_tb = data["test_b"]["Y_z"]
    groups_tb = data["test_b"]["case_id"].astype(str)

    # ---- 2. Reference picks (loaded) + sanity re-derive of MSE-TCSI ----
    with open(PICKS_V2) as f:
        picks_v2 = json.load(f)
    ref_tcsi = {K: list(map(int, picks_v2["TCSI"][str(K)])) for K in K_VALUES}
    ref_qdeim = {K: list(map(int, picks_v2["qDEIM"][str(K)])) for K in K_VALUES}
    print(f"[s23] loaded reference TCSI picks: {ref_tcsi}", flush=True)
    print(f"[s23] loaded reference qDEIM picks: {ref_qdeim}", flush=True)

    runtimes: Dict[str, float] = {}

    # MSE-TCSI re-derived locally on THIS latent/window so we can report whether
    # the frozen v2 picks reproduce here (the v2 picks came from a different
    # latent + W=30 window).
    t0 = time.time()
    tcsi_local_chain: List[int] = []
    tcsi_local: Dict[int, List[int]] = {}
    for K in sorted(K_VALUES):
        tcsi_local_chain = greedy_forward_selection(
            X_sel, y_pc1, K=K, initial=tcsi_local_chain
        )
        tcsi_local[K] = list(tcsi_local_chain)
    runtimes["MSE_TCSI_local_rederive"] = time.time() - t0
    print(f"[s23] MSE-TCSI re-derived locally: {tcsi_local} "
          f"({runtimes['MSE_TCSI_local_rederive']:.1f}s)", flush=True)

    # ---- 3. Calibrated greedy selections ----
    selections: Dict[str, Dict[int, List[int]]] = {}
    objective_curves: Dict[str, Dict[int, Dict[str, float]]] = {}

    # 3a. Calibrated Gaussian log-lik, scalar PC1 (RAW features, apples-to-apples
    # with MSE-TCSI which also uses raw features).
    t0 = time.time()
    chain: List[int] = []
    sel_ll_scalar: Dict[int, List[int]] = {}
    for K in sorted(K_VALUES):
        chain = greedy_generic(
            X_sel, lambda F: heldout_gaussian_loglik_scalar(F, y_pc1, sel_folds),
            K=K, initial=chain, verbose=True, tag="LL-scalar",
        )
        sel_ll_scalar[K] = list(chain)
    runtimes["calib_loglik_scalar"] = time.time() - t0
    selections["calib_loglik_scalar"] = sel_ll_scalar

    # 3a-bis. Calibrated Gaussian log-lik, scalar PC1, STANDARDISED features
    # (textbook calibrated estimator; fixes out-of-group magnitude
    # miscalibration of an unstandardised single-tap ridge).
    t0 = time.time()
    chain = []
    sel_ll_scalar_std: Dict[int, List[int]] = {}
    for K in sorted(K_VALUES):
        chain = greedy_generic(
            X_sel,
            lambda F: heldout_gaussian_loglik_scalar(F, y_pc1, sel_folds,
                                                     standardize=True),
            K=K, initial=chain, verbose=True, tag="LL-scalar-std",
        )
        sel_ll_scalar_std[K] = list(chain)
    runtimes["calib_loglik_scalar_std"] = time.time() - t0
    selections["calib_loglik_scalar_std"] = sel_ll_scalar_std

    # 3b. Calibrated Gaussian log-lik, full d=64 latent (diagonal noise cov).
    t0 = time.time()
    chain = []
    sel_ll_multi: Dict[int, List[int]] = {}
    for K in sorted(K_VALUES):
        chain = greedy_generic(
            X_sel, lambda F: heldout_gaussian_loglik_multi(F, Y_z_sel, sel_folds),
            K=K, initial=chain, verbose=True, tag="LL-multi",
        )
        sel_ll_multi[K] = list(chain)
    runtimes["calib_loglik_multi"] = time.time() - t0
    selections["calib_loglik_multi"] = sel_ll_multi

    # 3c. Gaussian MI greedy, scalar PC1.
    t0 = time.time()
    chain = []
    sel_mi_scalar: Dict[int, List[int]] = {}
    for K in sorted(K_VALUES):
        chain = greedy_generic(
            X_sel, lambda F: heldout_gaussian_mi_scalar(F, y_pc1, sel_folds),
            K=K, initial=chain, verbose=True, tag="MI-scalar",
        )
        sel_mi_scalar[K] = list(chain)
    runtimes["gauss_mi_scalar"] = time.time() - t0
    selections["gauss_mi_scalar"] = sel_mi_scalar

    # 3d. Gaussian MI greedy, full d=64 latent.
    t0 = time.time()
    chain = []
    sel_mi_multi: Dict[int, List[int]] = {}
    for K in sorted(K_VALUES):
        chain = greedy_generic(
            X_sel, lambda F: heldout_gaussian_mi_multi(F, Y_z_sel, sel_folds),
            K=K, initial=chain, verbose=True, tag="MI-multi",
        )
        sel_mi_multi[K] = list(chain)
    runtimes["gauss_mi_multi"] = time.time() - t0
    selections["gauss_mi_multi"] = sel_mi_multi

    # 3e. OPTIONAL Kraskov kNN MI greedy (scalar), under a wall-clock budget.
    knn_status = "skipped"
    knn_sel: Dict[int, List[int]] = {}
    # Cheap timing probe: cost of one full sweep at K=1.
    t_probe = time.time()
    _ = greedy_generic(
        X_sel, lambda F: knn_mi_scalar(F, y_pc1), K=1, initial=[],
    )
    probe_dt = time.time() - t_probe
    # Greedy cost ~ sum over k of (n_sensors-k) sweeps; estimate total for max K.
    est_total = probe_dt * sum(N_SENSORS - k for k in range(max(K_VALUES)))
    print(f"[s23] kNN-MI probe: one K=1 sweep took {probe_dt:.1f}s; "
          f"estimated full greedy ~{est_total:.0f}s (budget {KNN_TIME_BUDGET_S:.0f}s)",
          flush=True)
    if est_total <= KNN_TIME_BUDGET_S:
        t0 = time.time()
        chain = []
        for K in sorted(K_VALUES):
            chain = greedy_generic(
                X_sel, lambda F: knn_mi_scalar(F, y_pc1),
                K=K, initial=chain, verbose=True, tag="kNN-MI",
            )
            knn_sel[K] = list(chain)
        runtimes["knn_mi_scalar"] = time.time() - t0
        selections["knn_mi_scalar"] = knn_sel
        knn_status = "ran"
    else:
        print(f"[s23] kNN-MI greedy SKIPPED (estimated {est_total:.0f}s exceeds "
              f"{KNN_TIME_BUDGET_S:.0f}s budget).", flush=True)

    # ---- 4. Record TRAIN vs HELD-OUT objective for the overfitting check ----
    # For each calibrated method's full-data pick, compare the in-sample
    # (train, no CV) objective against the held-out CV objective and the
    # downstream held-out recovery. Inflation of train >> held-out with flat
    # recovery == overfitting at this n.
    def train_loglik_scalar(F, y):
        # Single in-sample fit, residual variance from the same data.
        model = Ridge(alpha=RIDGE_ALPHA).fit(F, y)
        resid = y - model.predict(F)
        var = float(np.mean(resid ** 2)) + 1e-9
        n = resid.size
        return float((-0.5 * n * np.log(2 * np.pi * var)
                      - 0.5 * np.sum(resid ** 2) / var) / n)

    def train_mi_scalar(F, y):
        model = Ridge(alpha=RIDGE_ALPHA).fit(F, y)
        yp = model.predict(F)
        if np.std(yp) < 1e-12:
            return 0.0
        rho2 = min(float(np.corrcoef(y, yp)[0, 1]) ** 2, 1 - 1e-12)
        return float(-0.5 * np.log(1 - rho2))

    overfit_rows: List[Dict[str, object]] = []
    calib_specs = [
        ("calib_loglik_scalar", "loglik", "scalar"),
        ("calib_loglik_multi", "loglik", "multi"),
        ("gauss_mi_scalar", "mi", "scalar"),
        ("gauss_mi_multi", "mi", "multi"),
    ]
    for name, kind, mode in calib_specs:
        for K in K_VALUES:
            S = selections[name][K]
            F = subset_features(X_sel, S)
            if kind == "loglik" and mode == "scalar":
                train_obj = train_loglik_scalar(F, y_pc1)
                held_obj = heldout_gaussian_loglik_scalar(F, y_pc1, sel_folds)
            elif kind == "loglik" and mode == "multi":
                # train multi log-lik
                model = Ridge(alpha=RIDGE_ALPHA).fit(F, Y_z_sel)
                resid = Y_z_sel - model.predict(F)
                var = np.mean(resid ** 2, axis=0) + 1e-9
                train_obj = float(-0.5 * np.sum(np.log(2 * np.pi * var))
                                  - 0.5 * np.mean(np.sum(resid ** 2 / var[None, :], 1)))
                held_obj = heldout_gaussian_loglik_multi(F, Y_z_sel, sel_folds)
            elif kind == "mi" and mode == "scalar":
                train_obj = train_mi_scalar(F, y_pc1)
                held_obj = heldout_gaussian_mi_scalar(F, y_pc1, sel_folds)
            else:  # mi multi
                model = Ridge(alpha=RIDGE_ALPHA).fit(F, Y_z_sel)
                yp = model.predict(F)
                tobj = 0.0
                for k in range(Y_z_sel.shape[1]):
                    if np.std(yp[:, k]) < 1e-12:
                        continue
                    r2_ = min(float(np.corrcoef(Y_z_sel[:, k], yp[:, k])[0, 1]) ** 2,
                              1 - 1e-12)
                    tobj += -0.5 * np.log(1 - r2_)
                train_obj = float(tobj)
                held_obj = heldout_gaussian_mi_multi(F, Y_z_sel, sel_folds)
            rec = recovery_z_r2(S, X_sel, Y_z_sel, X_tb, Y_tb, groups_tb,
                                include_krr=False)
            overfit_rows.append({
                "method": name, "K": K,
                "train_objective": train_obj,
                "heldout_objective": held_obj,
                "test_b_z_R2_train_tb_ridge": rec["z_R2_train_tb_ridge"],
            })

    # ---- 5. Downstream recovery for every method (incl. references) ----
    all_methods: Dict[str, Dict[int, List[int]]] = {
        "MSE_TCSI_ref": ref_tcsi,
        "qDEIM_ref": ref_qdeim,
        "MSE_TCSI_local": tcsi_local,
        "uniform": {K: selector_uniform(K) for K in K_VALUES},
    }
    all_methods.update(selections)

    recovery_table: Dict[str, Dict[int, Dict[str, float]]] = {}
    for name, perK in all_methods.items():
        recovery_table[name] = {}
        for K in K_VALUES:
            recovery_table[name][K] = recovery_z_r2(
                perK[K], X_sel, Y_z_sel, X_tb, Y_tb, groups_tb, include_krr=True)
        print(f"[s23] recovery {name:24s} "
              + "  ".join(
                  f"K{K}: ridge={recovery_table[name][K]['z_R2_train_tb_ridge']:+.3f}"
                  f"/krr={recovery_table[name][K]['z_R2_train_tb_krr']:+.3f}"
                  for K in K_VALUES), flush=True)

    # ---- 6. Fragility: case-level bootstrap pick stability + recovery spread ----
    print(f"[s23] bootstrap (B={N_BOOTSTRAP}, resample unit = case) ...", flush=True)
    rng = np.random.default_rng(0)
    pool_cases = np.unique(groups_sel)
    tb_cases = np.unique(groups_tb)

    # Methods to bootstrap: all greedy selectors (same per-step cost).
    boot_methods = ["MSE_TCSI_local", "calib_loglik_scalar",
                    "calib_loglik_scalar_std", "gauss_mi_scalar",
                    "calib_loglik_multi", "gauss_mi_multi"]

    def chain_for_method(name, Xs, ys, Ys, folds, Kmax):
        """Full greedy chain of length Kmax (prefix-sliceable for smaller K)."""
        if name == "MSE_TCSI_local":
            return greedy_forward_selection(Xs, ys, K=Kmax)
        if name == "calib_loglik_scalar":
            return greedy_generic(Xs, lambda F: heldout_gaussian_loglik_scalar(F, ys, folds), Kmax)
        if name == "calib_loglik_scalar_std":
            return greedy_generic(
                Xs, lambda F: heldout_gaussian_loglik_scalar(F, ys, folds, standardize=True), Kmax)
        if name == "gauss_mi_scalar":
            return greedy_generic(Xs, lambda F: heldout_gaussian_mi_scalar(F, ys, folds), Kmax)
        if name == "calib_loglik_multi":
            return greedy_generic(Xs, lambda F: heldout_gaussian_loglik_multi(F, Ys, folds), Kmax)
        if name == "gauss_mi_multi":
            return greedy_generic(Xs, lambda F: heldout_gaussian_mi_multi(F, Ys, folds), Kmax)
        raise ValueError(name)

    Kmax = max(K_VALUES)
    boot: Dict[str, Dict[int, Dict[str, object]]] = {m: {K: {} for K in K_VALUES}
                                                     for m in boot_methods}
    t_boot = time.time()
    for m in boot_methods:
        # Accumulate per-K stats across resamples; compute the chain ONCE per
        # resample at Kmax and slice (greedy is a prefix chain).
        per_k_jacc: Dict[int, List[float]] = {K: [] for K in K_VALUES}
        per_k_exact: Dict[int, int] = {K: 0 for K in K_VALUES}
        per_k_r2: Dict[int, List[float]] = {K: [] for K in K_VALUES}
        for b in range(N_BOOTSTRAP):
            # resample CASES with replacement on the selection pool
            draw = rng.choice(pool_cases, size=len(pool_cases), replace=True)
            idx = np.concatenate([np.where(groups_sel == c)[0] for c in draw])
            Xb = X_sel[idx]
            Yb = Y_z_sel[idx]
            gb = np.concatenate([np.full(np.sum(groups_sel == c), f"b{i}")
                                 for i, c in enumerate(draw)])
            yb = PCA(n_components=1, random_state=0).fit_transform(Yb).ravel()
            fb = _group_folds(gb, CV_FOLDS_SELECT)
            chain_b = chain_for_method(m, Xb, yb, Yb, fb, Kmax)
            for K in K_VALUES:
                full_pick = (tcsi_local[K] if m == "MSE_TCSI_local"
                             else selections[m][K])
                pk = chain_b[:K]
                per_k_jacc[K].append(jaccard(pk, full_pick))
                if set(pk) == set(full_pick):
                    per_k_exact[K] += 1
                # recovery propagates resampling uncertainty: fit the linear
                # ridge on the BOOTSTRAP pool, evaluate on the FIXED full test_b.
                per_k_r2[K].append(
                    recovery_z_r2(pk, Xb, Yb, X_tb, Y_tb, groups_tb,
                                  include_krr=False)["z_R2_train_tb_ridge"])
        for K in K_VALUES:
            jaccs = per_k_jacc[K]
            exact = per_k_exact[K]
            r2_arr = np.asarray(per_k_r2[K], dtype=np.float64)
            boot[m][K] = {
                "mean_jaccard_vs_fulldata": float(np.mean(jaccs)),
                "frac_exact_reproduce": exact / N_BOOTSTRAP,
                "z_R2_p5": float(np.nanpercentile(r2_arr, 5)),
                "z_R2_p50": float(np.nanpercentile(r2_arr, 50)),
                "z_R2_p95": float(np.nanpercentile(r2_arr, 95)),
            }
        print(f"[s23]   bootstrapped {m} "
              + "  ".join(f"K{K}:J={boot[m][K]['mean_jaccard_vs_fulldata']:.2f}"
                          for K in K_VALUES), flush=True)
    runtimes["bootstrap_total"] = time.time() - t_boot

    # ---- 7. Assemble results, write JSON + CSV + figure ----
    results = {
        "meta": {
            "selection_pool": {"splits": ["train", "test_a"], "n": n_pool,
                               "cases": n_pool_cases},
            "n_train": n_train, "n_test_a": n_test_a, "n_test_b": n_test_b,
            "n_test_b_cases": int(len(tb_cases)),
            "window": 17, "window_kind": "impact-centred (impact-8..impact+8)",
            "selection_CV": f"GroupKFold(case_id), {len(sel_folds)} folds",
            "recovery_protocols": {
                "z_R2_train_tb_ridge": (
                    "PRIMARY: linear ridge (alpha=1) fit on train+test_a pool, "
                    "eval on held-out test_b (disjoint case sets -> leakage-free)"),
                "z_R2_train_tb_krr": (
                    "paper KernelRidge(RBF, StandardScaler, alpha=0.1, gamma=0.01) "
                    "fit on pool, eval on test_b -> reproduces pressure_obs_v2.csv"),
                "z_R2_group": (
                    "STRESS diagnostic: GroupKFold(case_id) on the 42 test_b rows "
                    "only; underpowered (uniformly negative), NOT the verdict"),
            },
            "latent_target": "S12_E_d64 impact-frame z (n,64); PC1 = first PCA comp",
            "pc1_explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
            "reference_picks_provenance": (
                "sensor_picks_v2.json: derived on production noBN latent + W=30 "
                "pre-impact window; scored here against S12_E_d64 + W=17 window"
            ),
            "K_values": list(K_VALUES),
            "n_bootstrap": N_BOOTSTRAP,
            "knn_mi_status": knn_status,
        },
        "selections": {
            "MSE_TCSI_ref": ref_tcsi, "qDEIM_ref": ref_qdeim,
            "MSE_TCSI_local": tcsi_local,
            "uniform": {K: selector_uniform(K) for K in K_VALUES},
            **selections,
        },
        "recovery_z_R2": recovery_table,
        "overfitting_check": overfit_rows,
        "fragility_bootstrap": boot,
        "runtimes_seconds": runtimes,
        "wall_time_seconds": time.time() - t_start,
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(_tolist(results), f, indent=2)
    print(f"[s23] wrote {OUT_DIR / 'results.json'}", flush=True)

    # CSV: one row per (method, K) with picks + recovery + bootstrap summary.
    csv_path = OUT_DIR / "results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "K", "selected_taps",
                    "test_b_z_R2_ridge", "test_b_z_R2_krr", "test_b_z_R2_group",
                    "boot_mean_jaccard", "boot_frac_exact",
                    "boot_z_R2_ridge_p5", "boot_z_R2_ridge_p50",
                    "boot_z_R2_ridge_p95", "train_objective", "heldout_objective"])
        ov_lookup = {(r["method"], r["K"]): r for r in overfit_rows}
        for name, perK in results["selections"].items():
            for K in K_VALUES:
                rec = recovery_table[name][K]
                bm = boot.get(name, {}).get(K, {})
                ov = ov_lookup.get((name, K), {})
                w.writerow([
                    name, K, " ".join(map(str, perK[K])),
                    f"{rec['z_R2_train_tb_ridge']:.4f}",
                    f"{rec['z_R2_train_tb_krr']:.4f}",
                    f"{rec['z_R2_group']:.4f}",
                    f"{bm.get('mean_jaccard_vs_fulldata', float('nan')):.4f}"
                    if bm else "",
                    f"{bm.get('frac_exact_reproduce', float('nan')):.4f}"
                    if bm else "",
                    f"{bm.get('z_R2_p5', float('nan')):.4f}" if bm else "",
                    f"{bm.get('z_R2_p50', float('nan')):.4f}" if bm else "",
                    f"{bm.get('z_R2_p95', float('nan')):.4f}" if bm else "",
                    f"{ov.get('train_objective', float('nan')):.4f}" if ov else "",
                    f"{ov.get('heldout_objective', float('nan')):.4f}" if ov else "",
                ])
    print(f"[s23] wrote {csv_path}", flush=True)

    _render_figure(results, OUT_DIR / "results.png")
    print(f"[s23] wrote {OUT_DIR / 'results.png'}", flush=True)
    print(f"[s23] TOTAL wall time {time.time() - t_start:.1f}s", flush=True)
    return 0


def _render_figure(results: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    rec = results["recovery_z_R2"]
    boot = results["fragility_bootstrap"]
    Ks = results["meta"]["K_values"]

    plot_methods = [
        ("MSE_TCSI_ref", "tab:red", "o", "-"),
        ("qDEIM_ref", "tab:orange", "s", "-"),
        ("MSE_TCSI_local", "firebrick", "o", "--"),
        ("calib_loglik_scalar", "tab:blue", "^", "-"),
        ("calib_loglik_scalar_std", "navy", "^", "--"),
        ("calib_loglik_multi", "tab:cyan", "v", "-"),
        ("gauss_mi_scalar", "tab:green", "D", "-"),
        ("gauss_mi_multi", "tab:olive", "P", "-"),
        ("uniform", "0.5", "x", ":"),
    ]
    if "knn_mi_scalar" in rec:
        plot_methods.append(("knn_mi_scalar", "tab:purple", "*", "-"))

    def _get(d, K):
        return d[str(K)] if str(K) in d else d[K]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: PRIMARY downstream recovery z_R2 (fit-on-pool, eval-test_b) vs K.
    # Solid = linear ridge; faint dotted grey overlay = paper KernelRidge(RBF).
    ax = axes[0]
    for name, color, mk, ls in plot_methods:
        if name not in rec:
            continue
        y = [_get(rec[name], K)["z_R2_train_tb_ridge"] for K in Ks]
        ax.plot(Ks, y, marker=mk, color=color, ls=ls, label=name, lw=1.5, ms=7)
    # KRR overlay for the three headline methods.
    for name, color in (("MSE_TCSI_ref", "tab:red"),
                        ("gauss_mi_multi", "tab:olive"),
                        ("calib_loglik_multi", "tab:cyan")):
        if name in rec:
            yk = [_get(rec[name], K)["z_R2_train_tb_krr"] for K in Ks]
            ax.plot(Ks, yk, marker=".", color=color, ls=":", lw=1.0, alpha=0.6)
    ax.set_xticks(Ks)
    ax.set_xlabel("K (number of pressure taps)")
    ax.set_ylabel(r"test_b $z\,R^2$  (fit pool, eval test_b)")
    ax.set_title("Downstream JEPA-latent recovery vs selection method\n"
                 "(solid = linear ridge; dotted = paper KernelRidge-RBF)")
    ax.grid(alpha=0.3)
    ax.axhline(0, color="black", lw=0.5)
    ax.legend(fontsize=7, loc="lower left", ncol=2)

    # Panel 2: pick stability (mean Jaccard) with R^2 bootstrap spread bars.
    ax = axes[1]
    bm_methods = list(boot.keys())
    x = np.arange(len(Ks))
    w = 0.8 / max(1, len(bm_methods))
    cmap = {"MSE_TCSI_local": "firebrick", "calib_loglik_scalar": "tab:blue",
            "calib_loglik_scalar_std": "navy", "gauss_mi_scalar": "tab:green",
            "calib_loglik_multi": "tab:cyan", "gauss_mi_multi": "tab:olive"}
    for i, m in enumerate(bm_methods):
        jacc = [boot[m][str(K)]["mean_jaccard_vs_fulldata"] if str(K) in boot[m]
                else boot[m][K]["mean_jaccard_vs_fulldata"] for K in Ks]
        ax.bar(x + (i - (len(bm_methods) - 1) / 2) * w, jacc, width=w,
               label=m, color=cmap.get(m, None), edgecolor="black", lw=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={K}" for K in Ks])
    ax.set_ylabel("mean Jaccard of bootstrap pick vs full-data pick")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Pick stability (case-level bootstrap, B={results['meta']['n_bootstrap']})")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
