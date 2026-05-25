"""Prequential epiplexity for JEPA encoder training logs.

This module implements a loss-curve area structural-information proxy in the
spirit of Finzi, Qiu, Jiang, Izmailov, Kolter, and Wilson, "From Entropy to
Epiplexity: Rethinking Information for Computationally Bounded Intelligence"
(arXiv:2601.03220v2, March 2026), Section 4.1 / Equation 8.

Definition
----------
The prequential epiplexity of a learning process is the cumulative excess loss
over a "knowing-the-answer" predictor:

    |P_preq| ~= Sum_i (L_i - L_M)     over training steps i

where ``L_i`` is the loss at step ``i`` and ``L_M`` is the final (asymptotic)
loss of the trained model. With explicit step indices ``N_i`` this is the
Riemann sum approximation of the area under the curve ``(L - L_M)`` against
training tokens. The clamp ``max(L_i - L_M, 0)`` discards transient dips below
the floor so the result is non-negative.

Companion quantities
--------------------
- ``H_T`` (time-bounded entropy): ``L_M * N``, the residual irreducible noise
  the bounded model cannot eliminate (``N`` is the number of training samples
  or steps available for integration).
- ``S_T`` (epiplexity): alias of ``|P_preq|``; the structural part the learner
  actually absorbed.
- ``eff`` (efficiency): ``S_T / max(H_T, eps)``; small means most of the
  per-step loss budget is irreducible noise (good convergence), large means
  the trajectory still encodes a lot of structural information.

Calibration
-----------
Finzi et al. defines epiplexity in BITS via a careful prequential coding
scheme that requires the loss to be a calibrated negative log likelihood.
The JEPA ``loss_total`` exported by ``train_jepa.py`` is a weighted sum of
MSE-style prediction losses, SIGReg / VICReg anti-collapse, observable-head
SmoothL1, wake-head SmoothL1, and total-correlation. It is NOT a calibrated
NLL. We compute the same area-under-loss-curve mechanic, but the units are
"loss-units * iters" rather than bits. Use this number for ranking and
ablation studies within a single ``loss_key``; do not interpret the
absolute scale as Shannon information without calibration to a
log-likelihood predictor.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

_DEFAULT_FLOOR_FRACTION = 0.10  # fraction of trailing iterations used to estimate L_M
_LOSS_COMPONENTS = (
    "loss_total",
    "loss_pred",
    "loss_roll",
    "loss_anticollapse",
    "loss_obs",
    "loss_wake",
    "loss_tc",
)


def load_loss_curve(
    jsonl_path: Path,
    loss_key: str = "loss_total",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load ``(iters, losses, config)`` from a ``metrics.jsonl``.

    Args:
        jsonl_path: Path to a ``metrics.jsonl`` produced by ``train_jepa``.
            Line 1 must be ``{"event": "config", ...}``. Subsequent lines
            with ``"event": "log"`` are scanned for ``loss_key`` and the
            companion ``iter`` field.
        loss_key: Loss component to extract (default ``"loss_total"``).
            Diagnostic lines that do not carry this key are skipped silently,
            so the same loader handles both training-step and diagnostic log
            events without explicit branching.

    Returns:
        A tuple ``(iters, losses, config)`` where ``iters`` is an
        ``(N,)`` ``np.int64`` array, ``losses`` is an ``(N,)`` ``np.float64``
        array, and ``config`` is the first-line config dict (empty if no
        config line is present).

    Raises:
        FileNotFoundError: If ``jsonl_path`` does not exist.
        ValueError: If the file is empty or no lines contain ``loss_key``.
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"metrics file not found: {path}")

    config: dict[str, Any] = {}
    iters_list: list[int] = []
    losses_list: list[float] = []

    with path.open("r") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue
            event = record.get("event")
            if event == "config":
                config = {k: v for k, v in record.items() if k != "event"}
                continue
            if event != "log":
                continue
            if loss_key not in record:
                continue
            value = record[loss_key]
            if value is None:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(value):
                continue
            it = record.get("iter", record.get("step", len(losses_list)))
            try:
                it = int(it)
            except (TypeError, ValueError):
                it = len(losses_list)
            iters_list.append(it)
            losses_list.append(value)

    if not losses_list:
        raise ValueError(f"no log entries with loss_key={loss_key!r} found in {path}")

    iters = np.asarray(iters_list, dtype=np.int64)
    losses = np.asarray(losses_list, dtype=np.float64)
    return iters, losses, config


def _estimate_floor(
    losses: np.ndarray,
    fraction: float = _DEFAULT_FLOOR_FRACTION,
) -> float:
    """Median of the trailing ``fraction`` of ``losses`` (robust to spikes)."""
    n = losses.size
    if n == 0:
        raise ValueError("cannot estimate floor from empty losses array")
    tail = max(1, int(np.ceil(n * fraction)))
    return float(np.median(losses[-tail:]))


def prequential_coding(
    losses: np.ndarray,
    loss_floor: float | None = None,
    iters: np.ndarray | None = None,
) -> float:
    """Compute the prequential epiplexity from a loss curve.

    Integrates ``max(L_i - L_floor, 0)`` over training-step spacing. The
    clamp at zero ensures the integral is monotone in the curve's distance
    above the floor and never goes negative when an intermediate ``L_i``
    drops below the trailing-median estimate.

    Args:
        losses: ``(N,)`` array of per-step losses (assumed already sorted
            by training step).
        loss_floor: If given, the asymptotic loss ``L_M`` used as the floor.
            If ``None``, uses ``median(losses[-ceil(0.1 * N):])`` which is
            more robust to a single noisy final step than ``min(losses)``.
        iters: Optional ``(N,)`` array of training step indices. If provided,
            integrates via a Riemann sum with widths ``diff(iters)`` (with
            the first width set to ``iters[1] - iters[0]`` so the leading
            rectangle has the same spacing as its neighbour). If ``None``,
            treats each entry as one unit (rectangle width 1).

    Returns:
        The scalar ``|P_preq|`` in loss-units (if ``iters`` is ``None``) or
        loss-units * iters (if ``iters`` is given). Always non-negative.
    """
    losses = np.asarray(losses, dtype=np.float64)
    if losses.ndim != 1:
        raise ValueError(f"losses must be 1-D, got shape {losses.shape}")
    if losses.size == 0:
        return 0.0

    if loss_floor is None:
        loss_floor = _estimate_floor(losses)

    excess = np.clip(losses - float(loss_floor), a_min=0.0, a_max=None)

    if iters is None:
        return float(np.sum(excess))

    iters = np.asarray(iters, dtype=np.float64)
    if iters.shape != losses.shape:
        raise ValueError(f"iters shape {iters.shape} does not match losses shape {losses.shape}")
    if iters.size == 1:
        return float(excess[0])

    diffs = np.diff(iters)
    # Repeat the first step-spacing so the leading rectangle has a width
    # consistent with the rest of the curve. This is the trapezoid-style
    # left-endpoint Riemann sum on a non-uniform grid.
    widths = np.concatenate([[diffs[0]], diffs])
    widths = np.where(widths > 0, widths, 1.0)
    return float(np.sum(excess * widths))


def epiplexity_summary(
    metrics_path: Path,
    loss_key: str = "loss_total",
    loss_floor: float | None = None,
) -> dict[str, Any]:
    """Compute a one-shot epiplexity report for one loss component.

    Args:
        metrics_path: Path to ``metrics.jsonl``.
        loss_key: Which loss column to integrate. Defaults to
            ``"loss_total"``.
        loss_floor: If ``None``, uses ``median(losses[-10%:])`` as the
            asymptotic-loss estimate. Pass an explicit value (e.g. ``0.0``)
            to override.

    Returns:
        Dict with keys
        ``P_preq, L_M, H_T, S_T, N, config, loss_key, eff``. All scalar
        fields are Python ``float`` or ``int``. ``eff`` is
        ``S_T / max(H_T, 1e-12)``.
    """
    iters, losses, config = load_loss_curve(metrics_path, loss_key=loss_key)
    if loss_floor is None:
        l_m = _estimate_floor(losses)
    else:
        l_m = float(loss_floor)
    p_preq = prequential_coding(losses, loss_floor=l_m, iters=iters)
    n = int(losses.size)
    h_t = l_m * n
    eff = p_preq / max(abs(h_t), 1e-12)
    return {
        "P_preq": float(p_preq),
        "L_M": float(l_m),
        "H_T": float(h_t),
        "S_T": float(p_preq),
        "N": n,
        "config": config,
        "loss_key": loss_key,
        "eff": float(eff),
    }


def epiplexity_decomposition(metrics_path: Path) -> dict[str, dict[str, Any]]:
    """Compute ``epiplexity_summary`` for every available loss component.

    Iterates over the standard JEPA loss columns
    (``loss_total, loss_pred, loss_roll, loss_anticollapse, loss_obs,
    loss_wake, loss_tc``) and silently skips any that are absent from the
    file. This makes the function robust to runs that do not enable the
    observable or wake heads.

    Args:
        metrics_path: Path to ``metrics.jsonl``.

    Returns:
        Mapping from ``loss_key`` to its ``epiplexity_summary`` dict.
    """
    out: dict[str, dict[str, Any]] = {}
    for key in _LOSS_COMPONENTS:
        try:
            out[key] = epiplexity_summary(metrics_path, loss_key=key)
        except ValueError:
            # loss component not present in this run; skip silently.
            continue
    return out
