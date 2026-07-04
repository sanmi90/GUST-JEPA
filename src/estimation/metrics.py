"""Filter diagnostics and closure metrics for the Session 32 predict-correct filter.

All functions are pure numpy and operate on already-computed series (readouts,
innovations, ensembles), so this module is free of any coupling between the
filter innovation and the observable probes. The observable closure metrics score
E_w and C_L read from prior/analysis STATES through the frozen Q1 probes; those
readouts are computed by the caller and passed in as arrays. The probes are never
part of the filter.

Metrics (session-32 plan, Track B):
- closure per observable per phase (analysis / prior / open-loop): R^2 and RMSE;
- innovation whiteness: lag-1 autocorrelation of the innovation series;
- NIS (normalized innovation squared) chi-square coverage;
- analysis Mahalanobis ratio (needs a truth state; diagnostic);
- divergence flag: NIS above the 0.99 chi-square quantile for >= 5 consecutive
  frames OR analysis Mahalanobis ratio > 3;
- CRPS and spread-skill on C_L and E_w from the ensemble readouts.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

PHASE_NAMES = ("pre_impact", "impact", "post_impact")


# =========================================================================== basics
def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination (aggregated), NaN if the target is constant."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.size == 0:
        return float("nan")
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.size == 0:
        return float("nan")
    return float(np.sqrt(((y_true - y_pred) ** 2).mean()))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error in the observable's own physical units."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.size == 0:
        return float("nan")
    return float(np.abs(y_true - y_pred).mean())


# =========================================================================== whiteness
def lag1_autocorr(series: np.ndarray) -> dict:
    """Lag-1 autocorrelation of an innovation series (whiteness diagnostic).

    ``series`` is ``(T,)`` or ``(T, K)``. For a well-tuned filter the innovations
    are approximately white, so lag-1 autocorrelation is near zero. Returns the
    per-component values and their mean-abs summary.
    """
    x = np.asarray(series, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    T, K = x.shape
    ac = np.full(K, np.nan)
    for j in range(K):
        col = x[:, j]
        col = col - col.mean()
        denom = float((col * col).sum())
        if denom > 0 and T > 1:
            ac[j] = float((col[:-1] * col[1:]).sum() / denom)
    return {
        "per_component": ac.tolist(),
        "mean_abs_lag1": float(np.nanmean(np.abs(ac))),
        "max_abs_lag1": float(np.nanmax(np.abs(ac))) if np.isfinite(ac).any() else float("nan"),
    }


# =========================================================================== NIS
def nis_coverage(nis: np.ndarray, dof: int, alpha: float = 0.99) -> dict:
    """Chi-square coverage of the NIS series.

    Under a consistent filter, ``nis_t ~ chi^2_K`` with mean ``K``. Reports the
    mean NIS, the fraction inside the two-sided 95% chi-square band, and the
    fraction above the ``alpha`` upper quantile (the divergence tail).
    """
    nis = np.asarray(nis, dtype=np.float64)
    lo = float(chi2.ppf(0.025, dof))
    hi = float(chi2.ppf(0.975, dof))
    q_upper = float(chi2.ppf(alpha, dof))
    inside = float(np.mean((nis >= lo) & (nis <= hi))) if nis.size else float("nan")
    frac_above = float(np.mean(nis > q_upper)) if nis.size else float("nan")
    return {
        "mean_nis": float(nis.mean()) if nis.size else float("nan"),
        "expected_nis": float(dof),
        "band_95": [lo, hi],
        "frac_in_band_95": inside,
        "chi2_q_upper": q_upper,
        "frac_above_q_upper": frac_above,
    }


# =========================================================================== Mahalanobis
def analysis_mahalanobis_ratio(
    analysis_ens: np.ndarray, truth_state: np.ndarray, jitter: float = 1e-6
) -> np.ndarray:
    """Per-frame analysis Mahalanobis ratio of the truth state under the ensemble.

    ``analysis_ens`` is ``(T, N, d)`` and ``truth_state`` is ``(T, d)``. For each
    frame the ratio is ``sqrt( (mu - x)^T Pa^{-1} (mu - x) / d )`` with ``mu`` the
    analysis mean and ``Pa`` the analysis covariance. A value near 1 means the
    truth sits within the ensemble spread; large values flag over-confidence /
    divergence. Diagnostic (needs a truth latent), not a headline number.
    """
    analysis_ens = np.asarray(analysis_ens, dtype=np.float64)
    truth_state = np.asarray(truth_state, dtype=np.float64)
    T, N, d = analysis_ens.shape
    out = np.full(T, np.nan)
    for t in range(T):
        ens = analysis_ens[t]
        mu = ens.mean(axis=0)
        dev = ens - mu[None, :]
        cov = dev.T @ dev / (N - 1) + jitter * np.eye(d)
        diff = mu - truth_state[t]
        try:
            m2 = float(diff @ np.linalg.solve(cov, diff))
        except np.linalg.LinAlgError:
            continue
        out[t] = np.sqrt(max(m2, 0.0) / d)
    return out


# =========================================================================== divergence
def divergence_flag(
    nis: np.ndarray,
    dof: int,
    maha_ratio: np.ndarray | None = None,
    *,
    alpha: float = 0.99,
    consecutive: int = 5,
    maha_thresh: float = 3.0,
) -> dict:
    """Divergence flag: sustained NIS tail OR large analysis Mahalanobis ratio.

    Flags divergence if NIS exceeds the ``alpha`` chi-square quantile for
    ``consecutive`` or more consecutive frames, OR (when a Mahalanobis ratio series
    is given) if any analysis ratio exceeds ``maha_thresh``.
    """
    nis = np.asarray(nis, dtype=np.float64)
    q_upper = float(chi2.ppf(alpha, dof))
    above = nis > q_upper
    # longest run of consecutive True
    longest = 0
    cur = 0
    for a in above:
        cur = cur + 1 if a else 0
        longest = max(longest, cur)
    nis_flag = longest >= consecutive
    maha_flag = False
    max_ratio = float("nan")
    if maha_ratio is not None:
        mr = np.asarray(maha_ratio, dtype=np.float64)
        if np.isfinite(mr).any():
            max_ratio = float(np.nanmax(mr))
            maha_flag = bool(max_ratio > maha_thresh)
    return {
        "diverged": bool(nis_flag or maha_flag),
        "nis_tail_run": int(longest),
        "nis_run_threshold": int(consecutive),
        "nis_flag": bool(nis_flag),
        "maha_flag": bool(maha_flag),
        "max_maha_ratio": max_ratio,
        "maha_threshold": float(maha_thresh),
    }


# =========================================================================== CRPS / spread
def crps_ensemble(ensemble: np.ndarray, truth: np.ndarray) -> float:
    """Mean ensemble CRPS over frames (empirical / energy form).

    ``ensemble`` is ``(T, N)`` readouts, ``truth`` is ``(T,)``. Per frame,
    ``CRPS = E|X - y| - 0.5 E|X - X'|`` estimated from the ``N`` members; the mean
    over frames is returned.
    """
    ens = np.asarray(ensemble, dtype=np.float64)
    y = np.asarray(truth, dtype=np.float64).reshape(-1)
    T, N = ens.shape
    vals = np.empty(T)
    for t in range(T):
        x = ens[t]
        term1 = np.abs(x - y[t]).mean()
        term2 = np.abs(x[:, None] - x[None, :]).mean()
        vals[t] = term1 - 0.5 * term2
    return float(vals.mean())


def spread_skill(ensemble: np.ndarray, truth: np.ndarray) -> dict:
    """Spread-skill on an ensemble readout series.

    ``spread`` is the RMS ensemble std over frames; ``skill`` is the RMSE of the
    ensemble mean against truth; ``ratio = spread / skill`` (near 1 is
    well-calibrated). ``ensemble`` is ``(T, N)``, ``truth`` is ``(T,)``.
    """
    ens = np.asarray(ensemble, dtype=np.float64)
    y = np.asarray(truth, dtype=np.float64).reshape(-1)
    mean = ens.mean(axis=1)
    std = ens.std(axis=1, ddof=1)
    spread = float(np.sqrt((std**2).mean()))
    skill = float(np.sqrt(((mean - y) ** 2).mean()))
    ratio = spread / skill if skill > 0 else float("nan")
    return {"spread": spread, "skill_rmse": skill, "spread_skill_ratio": ratio}


# =========================================================================== closure by phase
def closure_by_phase(
    pred_obs: np.ndarray,
    truth_obs: np.ndarray,
    phase_ids: np.ndarray,
    phase_id_map: dict | None = None,
) -> dict:
    """R^2 and RMSE of a scalar observable readout, split by phase.

    Args:
        pred_obs: ``(T,)`` observable read from prior/analysis/open-loop STATE
            through a frozen probe (computed by the caller; a truth quantity is
            never fed to the filter).
        truth_obs: ``(T,)`` ground-truth observable.
        phase_ids: ``(T,)`` phase id per frame (``0/1/2`` for
            ``pre_impact/impact/post_impact``, ``-1`` = no phase).
        phase_id_map: optional ``{phase_name: id}``; defaults to
            ``{pre_impact:0, impact:1, post_impact:2}``.

    Returns:
        ``{phase_name: {r2, rmse, n}}`` plus an ``all`` entry over labelled frames.
    """
    pred_obs = np.asarray(pred_obs, dtype=np.float64).reshape(-1)
    truth_obs = np.asarray(truth_obs, dtype=np.float64).reshape(-1)
    phase_ids = np.asarray(phase_ids)
    id_map = phase_id_map or {"pre_impact": 0, "impact": 1, "post_impact": 2}
    out: dict = {}
    for name, pid in id_map.items():
        m = phase_ids == pid
        out[name] = {
            "r2": r2_score(truth_obs[m], pred_obs[m]),
            "rmse": rmse(truth_obs[m], pred_obs[m]),
            "mae": mae(truth_obs[m], pred_obs[m]),
            "n": int(m.sum()),
        }
    labelled = phase_ids >= 0
    out["all"] = {
        "r2": r2_score(truth_obs[labelled], pred_obs[labelled]),
        "rmse": rmse(truth_obs[labelled], pred_obs[labelled]),
        "mae": mae(truth_obs[labelled], pred_obs[labelled]),
        "n": int(labelled.sum()),
    }
    return out


def closure_gain(analysis_closure: dict, baseline_closure: dict) -> dict:
    """Per-phase RMSE reduction (baseline - analysis) and R^2 gain vs a baseline."""
    out: dict = {}
    for name in list(analysis_closure.keys()):
        a = analysis_closure[name]
        b = baseline_closure.get(name, {})
        out[name] = {
            "rmse_reduction": (
                float(b.get("rmse", np.nan) - a.get("rmse", np.nan))
                if np.isfinite(b.get("rmse", np.nan)) and np.isfinite(a.get("rmse", np.nan))
                else float("nan")
            ),
            "r2_gain": (
                float(a.get("r2", np.nan) - b.get("r2", np.nan))
                if np.isfinite(b.get("r2", np.nan)) and np.isfinite(a.get("r2", np.nan))
                else float("nan")
            ),
        }
    return out


__all__ = [
    "PHASE_NAMES",
    "r2_score",
    "rmse",
    "mae",
    "lag1_autocorr",
    "nis_coverage",
    "analysis_mahalanobis_ratio",
    "divergence_flag",
    "crps_ensemble",
    "spread_skill",
    "closure_by_phase",
    "closure_gain",
]
