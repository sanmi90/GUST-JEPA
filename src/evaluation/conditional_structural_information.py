"""Target-conditioned structural-information (TCSI) proxy estimators.

This module computes four scalar proxies that summarise the prequential
training curve of a cheap learner ``f(p_S) -> Y`` for a sensor subset ``S``
and a target ``Y``. The four proxies are used to rank candidate sensor
subsets in the Session-14 sparse-sensor estimator pilot.

Naming distinction (load-bearing for the paper).
================================================
This module is **TCSI** (target-conditioned structural information). It is
INSPIRED BY the epiplexity framework of Finzi, Qiu, Jiang, Izmailov, Kolter,
and Wilson, "From Entropy to Epiplexity" (arXiv:2601.03220v2, March 2026),
specifically Definition 11. We re-use the *shape* of the prequential
loss-curve decomposition (irreducible floor, learning area, gain over null,
efficiency) but the proxies here are MSE-based and are NOT
log-likelihood-calibrated. Epiplexity proper requires a calibrated
log-likelihood model, which we do not have for the cheap closed-form ridge
learner used for sensor screening.

The companion module ``src/evaluation/epiplexity.py`` (when present) measures
epiplexity directly on JEPA training logs, where the loss term is a proper
prediction MSE on a fixed-dimensional latent, and the caveat is documented
there. The two modules are deliberately named differently so that a peer
reviewer cannot conflate them: TCSI is an ordering / screening proxy,
epiplexity is the calibrated quantity.

Proxy definitions (for a single (sensor-subset, target) pair).
==============================================================
Given a learner ``f`` trained on ``N`` samples to predict ``Y`` from ``p_S``,
with per-step training losses ``L_1, ..., L_T``:

* ``L_star``: asymptotic floor of ``f`` on ``Y`` (irreducible residual at
  this model class). Defaults to the median of the last 10 percent of the
  training-loss curve.
* ``L_null``: loss of a target-only null baseline (e.g. mean predictor of
  ``Y``) on ``Y``. Supplied by the caller.
* ``S_preq``: prequential structural-info area,
  ``S_preq = sum_t max(0, L_t - L_star)``. "How hard was the learning."
* ``H_res``: residual-entropy proxy, ``H_res = N * L_star``. "Irreducible
  noise at this model class."
* ``G``: structural gain over the null, ``G = N * max(0, L_null - L_star)``.
  "Useful captured" relative to the mean predictor.
* ``Eff``: efficiency, ``Eff = G / (S_preq + eps)``. "Useful info per unit
  of learning effort."

For the closed-form ridge learner we synthesise a 2-point loss curve
``[L_null, L_train_final]``: there is no iterative training, but the
two-point area still ranks subsets sensibly under the same formulas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TCSIProxies:
    """Four scalar TCSI proxies for a (sensor-subset, target) pair.

    Attributes:
        S_preq: Prequential structural-info area
            ``sum_t max(0, L_t - L_star)``.
        H_res: Residual-entropy proxy ``N * L_star``.
        G: Structural gain over the null baseline,
            ``N * max(0, L_null - L_star)``.
        Eff: Efficiency, ``G / (S_preq + eps)``.
        L_star: Asymptotic training loss (model-class floor).
        L_null: Loss of the null baseline on the same target.
    """

    S_preq: float
    H_res: float
    G: float
    Eff: float
    L_star: float
    L_null: float


# ---------------------------------------------------------------------------
# Null predictor and proxy computation
# ---------------------------------------------------------------------------


def null_predictor_loss(targets: np.ndarray) -> float:
    """MSE loss of the mean predictor on ``targets``.

    The null baseline is ``y_hat = mean(y, axis=0)``, broadcast over the
    sample axis. For multi-output targets the returned scalar is the mean
    across all output dimensions, matching the convention used by
    :func:`compute_proxies_for_sensor`.

    Args:
        targets: Array of shape ``(N,)`` or ``(N, ...)``.

    Returns:
        Mean squared error of the constant-mean predictor on ``targets``.
    """
    y = np.asarray(targets, dtype=np.float64)
    if y.ndim == 0:
        raise ValueError("targets must have at least one sample axis")
    mean = y.mean(axis=0, keepdims=True)
    residual = y - mean
    return float(np.mean(residual ** 2))


def compute_proxies(
    pred_losses: np.ndarray,
    L_null: float,
    N: Optional[int] = None,
    L_star: Optional[float] = None,
    eps: float = 1e-6,
) -> TCSIProxies:
    """Compute the four TCSI proxies from a learner's training-loss curve.

    Args:
        pred_losses: One scalar loss per training step / epoch, shape
            ``(T,)``. Must be non-empty.
        L_null: Loss of the null baseline (e.g. mean predictor) on the same
            target.
        N: Number of training samples. Defaults to ``len(pred_losses)``,
            which is appropriate when one step corresponds to one sample.
            For batched training, pass the actual sample count.
        L_star: Explicit floor override. If ``None``, defaults to the median
            of the last 10 percent of ``pred_losses`` (at least one step).
        eps: Small constant added to the denominator of ``Eff`` for
            numerical stability.

    Returns:
        :class:`TCSIProxies` with all four scalar proxies plus ``L_star``
        and ``L_null`` echoed back for downstream logging.
    """
    losses = np.asarray(pred_losses, dtype=np.float64)
    if losses.ndim != 1:
        raise ValueError(f"pred_losses must be 1D, got shape {losses.shape}")
    if losses.size == 0:
        raise ValueError("pred_losses must be non-empty")

    if N is None:
        N = int(losses.size)
    else:
        N = int(N)

    if L_star is None:
        tail = max(1, int(np.ceil(0.1 * losses.size)))
        L_star_value = float(np.median(losses[-tail:]))
    else:
        L_star_value = float(L_star)

    L_null_value = float(L_null)

    S_preq = float(np.sum(np.maximum(0.0, losses - L_star_value)))
    H_res = float(N) * L_star_value
    G = float(N) * max(0.0, L_null_value - L_star_value)
    Eff = G / (S_preq + float(eps))

    return TCSIProxies(
        S_preq=S_preq,
        H_res=H_res,
        G=G,
        Eff=Eff,
        L_star=L_star_value,
        L_null=L_null_value,
    )


def objective_J(
    proxies: TCSIProxies,
    weights: Optional[dict] = None,
) -> float:
    """Scalar objective for ranking sensor subsets.

    The default weights reflect a balanced preference: maximise structural
    gain over null, minimise prequential area (learning effort) and
    residual entropy (irreducible noise), and reward efficiency.

    Default weights: ``w_G = 1.0``, ``w_S = -0.5``, ``w_H = -0.5``,
    ``w_Eff = 0.5``.

    Args:
        proxies: Output of :func:`compute_proxies` or
            :func:`compute_proxies_for_sensor`.
        weights: Optional override dict. Recognised keys: ``w_G``, ``w_S``,
            ``w_H``, ``w_Eff``. Missing keys fall back to the defaults.

    Returns:
        ``w_G * G + w_S * S_preq + w_H * H_res + w_Eff * Eff``.
    """
    default_weights = {"w_G": 1.0, "w_S": -0.5, "w_H": -0.5, "w_Eff": 0.5}
    if weights is not None:
        default_weights.update(weights)
    w = default_weights
    return float(
        w["w_G"] * proxies.G
        + w["w_S"] * proxies.S_preq
        + w["w_H"] * proxies.H_res
        + w["w_Eff"] * proxies.Eff
    )


# ---------------------------------------------------------------------------
# Cheap proxy learner
# ---------------------------------------------------------------------------


class RidgeProxyLearner:
    """Closed-form ridge-regression learner for fast sensor screening.

    Solves ``(X^T X + alpha I) W = X^T y`` in closed form, which is cheap
    enough to evaluate all 192 individual pressure sensors as candidate
    subsets. Because there is no iterative training, the loss curve is
    synthesised as the two-point sequence ``[L_null, L_train_final]``: the
    null prediction at step 0 and the fitted residual MSE at step 1. This
    is a "1-epoch" stand-in for the prequential area and ranks subsets
    sensibly even though it does not resolve learning dynamics.

    Args:
        alpha: Ridge regularisation strength. Must be non-negative.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        if alpha < 0.0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        self.alpha: float = float(alpha)
        self._W: Optional[np.ndarray] = None
        self._b: Optional[np.ndarray] = None
        self._n_features: Optional[int] = None
        self._y_ndim: int = 1

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Solve the closed-form ridge problem.

        Args:
            X: Feature matrix of shape ``(N, D)``.
            y: Target of shape ``(N,)`` or ``(N, K)``.
        """
        X64 = np.asarray(X, dtype=np.float64)
        y64 = np.asarray(y, dtype=np.float64)
        if X64.ndim != 2:
            raise ValueError(f"X must be 2D (N, D), got shape {X64.shape}")
        if y64.ndim not in (1, 2):
            raise ValueError(f"y must be 1D or 2D, got shape {y64.shape}")
        if X64.shape[0] != y64.shape[0]:
            raise ValueError(
                f"X and y must share the sample axis: {X64.shape[0]} vs {y64.shape[0]}"
            )

        self._y_ndim = y64.ndim
        n, d = X64.shape

        x_mean = X64.mean(axis=0)
        y_mean = y64.mean(axis=0)
        Xc = X64 - x_mean
        yc = y64 - y_mean

        gram = Xc.T @ Xc + self.alpha * np.eye(d)
        rhs = Xc.T @ yc
        W = np.linalg.solve(gram, rhs)

        self._W = W
        self._b = y_mean - x_mean @ W
        self._n_features = d

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict targets for ``X``.

        Args:
            X: Feature matrix of shape ``(N, D)`` matching the fit features.

        Returns:
            Predictions of shape ``(N,)`` or ``(N, K)`` matching the
            training target's rank.
        """
        if self._W is None or self._b is None:
            raise RuntimeError("RidgeProxyLearner.fit must be called before predict")
        X64 = np.asarray(X, dtype=np.float64)
        if X64.ndim != 2 or X64.shape[1] != self._n_features:
            raise ValueError(
                f"X must be (N, {self._n_features}), got shape {X64.shape}"
            )
        y_hat = X64 @ self._W + self._b
        if self._y_ndim == 1 and y_hat.ndim == 2 and y_hat.shape[-1] == 1:
            y_hat = y_hat.reshape(-1)
        return y_hat

    def fit_with_loss_curve(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, float]:
        """Fit on ``(X, y)``, return a 2-point loss curve and the eval null.

        The loss curve is ``[L_null_eval, L_final_eval]`` where the
        evaluation target is ``y_val`` if provided, otherwise ``y`` itself
        (in-sample). ``L_null_eval`` is the MSE of the mean predictor on
        the evaluation target; ``L_final_eval`` is the MSE of the fitted
        ridge model on the same data.

        Args:
            X: Training feature matrix of shape ``(N, D)``.
            y: Training target of shape ``(N,)`` or ``(N, K)``.
            X_val: Optional held-out features for evaluation.
            y_val: Optional held-out targets matching ``X_val``.

        Returns:
            Tuple ``(loss_curve, L_null_for_eval)`` where ``loss_curve`` is
            a length-2 ``np.ndarray`` and ``L_null_for_eval`` is the scalar
            null baseline used to populate the first entry.
        """
        self.fit(X, y)

        if (X_val is None) ^ (y_val is None):
            raise ValueError("X_val and y_val must be supplied together")

        if X_val is not None and y_val is not None:
            X_eval = np.asarray(X_val, dtype=np.float64)
            y_eval = np.asarray(y_val, dtype=np.float64)
        else:
            X_eval = np.asarray(X, dtype=np.float64)
            y_eval = np.asarray(y, dtype=np.float64)

        L_null = null_predictor_loss(y_eval)
        y_pred = self.predict(X_eval)
        residual = y_eval - y_pred
        L_final = float(np.mean(residual ** 2))

        loss_curve = np.array([L_null, L_final], dtype=np.float64)
        return loss_curve, float(L_null)


def compute_proxies_for_sensor(
    sensor_data: np.ndarray,
    target: np.ndarray,
    learner: Optional[RidgeProxyLearner] = None,
) -> TCSIProxies:
    """Fit a proxy learner on ``sensor -> target`` and return TCSI proxies.

    Args:
        sensor_data: Sensor window of shape ``(N, W)`` (e.g. ``W`` time
            lags or stacked features for one sensor).
        target: Target of shape ``(N,)`` or ``(N, K)``.
        learner: Optional pre-configured :class:`RidgeProxyLearner`. A
            default ``RidgeProxyLearner(alpha=1.0)`` is constructed when
            ``None``.

    Returns:
        :class:`TCSIProxies` summarising the learner's two-point loss curve
        for this sensor.
    """
    if learner is None:
        learner = RidgeProxyLearner(alpha=1.0)

    X = np.asarray(sensor_data, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"sensor_data must be (N, W), got shape {X.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"sensor_data and target must share sample axis: "
            f"{X.shape[0]} vs {y.shape[0]}"
        )

    loss_curve, L_null = learner.fit_with_loss_curve(X, y)
    return compute_proxies(loss_curve, L_null=L_null, N=X.shape[0])


__all__ = [
    "TCSIProxies",
    "RidgeProxyLearner",
    "compute_proxies",
    "compute_proxies_for_sensor",
    "null_predictor_loss",
    "objective_J",
]
