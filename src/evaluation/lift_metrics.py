"""Lift-tracking metrics for the Track C conditioning ablation (Session 34).

Three metrics, all per-encounter and aligned on the canonical
``(case_id, encounter_index)`` key:

- :func:`peak_window_mask` / :func:`peak_region_r2` -- decoded-C_L fidelity in
  the gust-peak region (``argmax |C_L|`` +/- ``half_width`` frames, default
  +/- 8 frames = +/- 0.4 convective times at dt = 0.05). The R^2 is pooled
  (aggregate SSE / SST across encounters BEFORE dividing), matching the
  ``aggregated_vrmse`` convention.
- :func:`lift_phase_lag` -- per-encounter cross-correlation phase lag of the
  decoded C_L relative to truth, in convective time. Positive lag means the
  decoded signal TRAILS truth (the lagged-wake-conditioning failure mode, Track
  C Q2). Parabolic sub-frame refinement around the integer-lag peak.

The pre-registered tolerance for Track C is tau_thresh = 0.1 convective time
(see scripts/session34/trackc_gates.py).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = ["peak_window_mask", "peak_region_r2", "lift_phase_lag"]


def peak_window_mask(cl_true: np.ndarray, half_width: int = 8) -> np.ndarray:
    """Boolean mask selecting ``argmax |C_L|`` +/- ``half_width`` frames.

    Args:
        cl_true: ``(T,)`` true lift trace of one encounter.
        half_width: Window half-width in frames (default 8 = 0.4 t/c).

    Returns:
        ``(T,)`` bool mask; the window is clipped at the trace bounds.
    """
    cl_true = np.asarray(cl_true)
    if cl_true.ndim != 1:
        raise ValueError(f"cl_true must be 1-D; got shape {cl_true.shape}")
    peak = int(np.argmax(np.abs(cl_true)))
    lo = max(0, peak - int(half_width))
    hi = min(cl_true.shape[0], peak + int(half_width) + 1)
    mask = np.zeros(cl_true.shape[0], dtype=bool)
    mask[lo:hi] = True
    return mask


def peak_region_r2(
    cl_true: Sequence[np.ndarray],
    cl_pred: Sequence[np.ndarray],
    half_width: int = 8,
) -> float:
    """Pooled peak-region R^2 across encounters.

    SSE and SST are aggregated over every encounter's peak window before the
    final division, so long or high-amplitude encounters weigh in proportion to
    their variance (the ``aggregated_vrmse`` convention). The SST baseline is
    the pooled mean of the true peak-window samples.

    Args:
        cl_true: Per-encounter true ``(T,)`` traces.
        cl_pred: Per-encounter predicted traces, same shapes.
        half_width: Peak-window half-width in frames.

    Returns:
        Pooled R^2 (1 - SSE/SST); can be negative.
    """
    if len(cl_true) != len(cl_pred) or len(cl_true) == 0:
        raise ValueError("cl_true / cl_pred must be equal-length, non-empty lists")
    t_all, p_all = [], []
    for t, p in zip(cl_true, cl_pred):
        t = np.asarray(t, dtype=np.float64)
        p = np.asarray(p, dtype=np.float64)
        if t.shape != p.shape:
            raise ValueError(f"shape mismatch: {t.shape} vs {p.shape}")
        m = peak_window_mask(t, half_width=half_width)
        t_all.append(t[m])
        p_all.append(p[m])
    t_cat = np.concatenate(t_all)
    p_cat = np.concatenate(p_all)
    sse = float(((t_cat - p_cat) ** 2).sum())
    sst = float(((t_cat - t_cat.mean()) ** 2).sum())
    return 1.0 - sse / max(sst, 1e-12)


def lift_phase_lag(
    pred: np.ndarray,
    true: np.ndarray,
    dt: float = 0.05,
    max_lag: int = 20,
) -> tuple[float, float]:
    """Cross-correlation phase lag of ``pred`` relative to ``true``.

    Positive lag means ``pred`` trails ``true`` by that much convective time
    (``pred[n + k] ~ true[n]``). The integer-lag argmax of the normalized
    cross-correlation is refined by a parabolic fit through the peak and its
    two neighbors, so the returned lag has sub-frame resolution.

    Args:
        pred: ``(T,)`` decoded / estimated lift trace.
        true: ``(T,)`` true lift trace, same length.
        dt: Frame spacing in convective time (cache dt_tc = 0.05).
        max_lag: Search half-range in frames.

    Returns:
        ``(lag_tc, peak_corr)`` -- the refined lag in convective time and the
        normalized correlation at the integer peak.
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if pred.shape != true.shape or pred.ndim != 1:
        raise ValueError(f"pred/true must be equal-length 1-D; got {pred.shape}, {true.shape}")
    a = pred - pred.mean()
    b = true - true.mean()
    den = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    # np.correlate(a, b, 'full')[k] = sum_n a[n + lag] * b[n] with
    # lag = k - (T - 1); a positive-argmax lag means pred trails truth.
    xc = np.correlate(a, b, mode="full") / den
    lags = np.arange(-len(a) + 1, len(a))
    keep = np.abs(lags) <= int(max_lag)
    xc_k = xc[keep]
    lags_k = lags[keep]
    i = int(np.argmax(xc_k))
    peak_corr = float(xc_k[i])
    lag = float(lags_k[i])
    # Parabolic sub-frame refinement (guarded at the search-window edges).
    if 0 < i < len(xc_k) - 1:
        y0, y1, y2 = xc_k[i - 1], xc_k[i], xc_k[i + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-12:
            delta = 0.5 * (y0 - y2) / denom
            if abs(delta) <= 1.0:
                lag += float(delta)
    return lag * float(dt), peak_corr
