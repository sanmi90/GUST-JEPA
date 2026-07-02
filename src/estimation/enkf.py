"""Ensemble Kalman filter over the frozen pooled latent (Session 32, Track B).

The predict-correct filter that is the scientific centerpiece: the estimator is
held fixed while the representation varies. The STATE is the frozen pooled
coefficient latent ``z_pool in R^d`` (``d = 32``), advanced by the frozen matched
autoregressive predictor; the MEASUREMENT is the ``K``-tap wall pressure through
the frozen observation operator :class:`~src.estimation.obs_operator.ObservationOperator`.

Design commitments (session-32 plan, Track B):

- **Per-member rolling context.** The AR transformer conditions on a HISTORY of
  past frames, so each ensemble member carries its OWN buffer of past ANALYSIS
  states. One filter cycle at frame ``t+1``:
    1. forecast: advance every member one step from its analysis buffer,
       ``prior_i = model.step(buffer_i)`` (the transformer reads the whole buffer,
       not just the last state);
    2. add additive process noise (D222) and apply multiplicative inflation;
    3. analysis: rewrite ONLY the current state via the EnKF update;
    4. append the analysis state to the buffer (capped at ``max_context``), so the
       corrected state enters the NEXT step's context.
  A naive single-state one-step roll that ignores the buffer silently degrades the
  predictor and is explicitly avoided: the forecast model is fed ``buffer`` with
  shape ``(N, T_ctx, d)`` every step.

- **Init is field-free.** The ensemble is seeded at ``t_imp - 24`` from the O1
  windowed pressure -> state estimator mean and its residual covariance (see
  :class:`FieldFreeInit`). A diagnostic truth-init (encoded truth + noise) exists
  behind a debug flag to separate init error from filter error; it is never
  headlined.

- **Process noise (D222).** Additive Gaussian from the one-step predictor
  residuals on held-out trajectories (:func:`process_noise_from_residuals`).
  Multiplicative inflation ``rho in {1.00, 1.02, 1.05}`` is a parameter and is NOT
  auto-selected here.

- **Cadence.** Assimilate every frame from ``t_imp - 24`` to ``t_imp + 48``. Prior
  and analysis ensembles are logged every step. A hook
  (:meth:`EnsembleKalmanFilter.branch_free_run`) forecasts free (no assimilation)
  from an analysis ensemble at a phase boundary, for later Track D reuse.

The filter machinery is limited to the stochastic EnKF (Evensen/Burgers
perturbed-observation form) and a deterministic square-root variant (ETKF,
Hunt et al. 2007) behind one flag, plus inflation. No localization, no 4D-Var
(out of scope for S32).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

from src.estimation.obs_operator import ObservationOperator


# =========================================================================== models
class ForecastModel(Protocol):
    """One-step-ahead forecast from a per-member analysis buffer.

    ``step`` maps a buffer ``(N, T_ctx, d)`` of past analysis states to the next
    prior state ``(N, d)`` for every member. ``max_context`` caps the buffer
    length the model reads (the transformer's ``max_seq_len``).
    """

    max_context: int

    def step(self, buffer: np.ndarray) -> np.ndarray:  # (N, T_ctx, d) -> (N, d)
        ...


@dataclass
class LinearForecast:
    """Exact linear one-step model ``x_{t+1} = F x_t`` (tests / linear baselines).

    Ignores all but the most recent buffer frame, so it is a valid
    :class:`ForecastModel` whose dynamics match a linear-Gaussian system for the
    EnKF-vs-Kalman equivalence test.
    """

    F: np.ndarray
    max_context: int = 1

    def step(self, buffer: np.ndarray) -> np.ndarray:
        x = np.asarray(buffer)[:, -1, :]  # (N, d)
        return x @ self.F.T


class TransformerForecast:
    """One-step forecast through a frozen AR-transformer, reading the full buffer.

    Wraps a trained :class:`~src.models.predictor.AutoregressivePredictor` (or any
    module with the same ``forward(z, cond)`` contract). Every ``step`` feeds the
    whole ``(N, T_ctx, d)`` analysis buffer through the transformer and returns the
    last-position prediction, which is the next-frame prior for each member. The
    buffer therefore threads history into the predictor exactly as the plan
    requires.
    """

    def __init__(
        self, predictor, device, cond: np.ndarray | None = None, max_context: int | None = None
    ) -> None:
        import torch  # local import: keeps this module CPU/import friendly

        self._torch = torch
        self.predictor = predictor
        self.predictor.eval()
        self.device = device
        mc = int(getattr(predictor, "max_seq_len", 32))
        self.max_context = int(max_context) if max_context is not None else mc
        self.cond = None if cond is None else np.asarray(cond, dtype=np.float32)

    def step(self, buffer: np.ndarray) -> np.ndarray:
        torch = self._torch
        buf = np.asarray(buffer, dtype=np.float32)
        n, t_ctx, d = buf.shape
        if t_ctx > self.max_context:
            buf = buf[:, -self.max_context :, :]
        zb = torch.from_numpy(buf).to(self.device)
        cond_t = None
        if self.cond is not None:
            cond_t = torch.from_numpy(np.broadcast_to(self.cond, (n, self.cond.shape[-1])).copy())
            cond_t = cond_t.to(self.device)
        with torch.no_grad():
            if getattr(self.device, "type", "cpu") == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    z_hat = self.predictor(zb, cond_t)
            else:
                z_hat = self.predictor(zb, cond_t)
        return z_hat[:, -1, :].float().cpu().numpy()


# =========================================================================== process noise
def process_noise_from_residuals(
    trajs: list[np.ndarray],
    model: ForecastModel,
    *,
    context_length: int = 2,
    diagonal: bool = False,
    jitter: float = 1e-6,
) -> np.ndarray:
    """Estimate additive process-noise covariance Q from one-step residuals (D222).

    For every held-out trajectory ``z (L, d)`` and every frame ``t`` with a full
    context, the one-step residual is ``r_t = z_{t+1} - model.step(z[:t+1])``
    (using up to ``max_context`` history frames). Q is the sample covariance of the
    stacked residuals (diagonal if requested), with a small jitter added for PD.
    """
    resids: list[np.ndarray] = []
    max_ctx = int(getattr(model, "max_context", 32))
    for z in trajs:
        z = np.asarray(z, dtype=np.float32)
        L = z.shape[0]
        for t in range(context_length - 1, L - 1):
            lo = max(0, t - max_ctx + 1)
            buf = z[lo : t + 1][None, :, :]  # (1, T_ctx, d)
            pred = model.step(buf)[0]  # (d,)
            resids.append(z[t + 1] - pred)
    R = np.asarray(resids, dtype=np.float64)
    d = R.shape[1]
    if diagonal:
        Q = np.diag(R.var(axis=0))
    else:
        Q = np.cov(R, rowvar=False)
        if Q.ndim == 0:
            Q = Q.reshape(1, 1)
    Q = Q + jitter * np.eye(d)
    return Q


# =========================================================================== init
@dataclass
class FieldFreeInit:
    """Field-free ensemble initialiser from the O1 windowed pressure estimator.

    ``mean_fn`` maps a windowed pressure feature ``(F,)`` to a state estimate
    ``(d,)`` (reuse :func:`src.evaluation.pressure_infer.fit_pressure_estimator`);
    ``cov`` is the residual covariance of that estimator on TRAIN. The initial
    ensemble at ``t_imp - 24`` is ``mean_fn(window) + chol(cov) @ N(0, I)``.
    """

    mean_fn: Callable[[np.ndarray], np.ndarray]
    cov: np.ndarray

    def sample(self, window_feat: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
        mean = np.asarray(self.mean_fn(window_feat[None, :]), dtype=np.float64).reshape(-1)
        L = _safe_cholesky(self.cov)
        eps = rng.standard_normal((n, mean.shape[0]))
        return mean[None, :] + eps @ L.T


def truth_init_ensemble(
    z_truth0: np.ndarray, n: int, noise_std: float, rng: np.random.Generator
) -> np.ndarray:
    """Diagnostic-only truth init: encoded truth state + isotropic Gaussian noise.

    Behind the caller's debug flag. Separates init error from filter error; never
    used for a headline number.
    """
    z0 = np.asarray(z_truth0, dtype=np.float64).reshape(-1)
    return z0[None, :] + noise_std * rng.standard_normal((n, z0.shape[0]))


# =========================================================================== linalg
def _safe_cholesky(cov: np.ndarray) -> np.ndarray:
    """Cholesky factor with adaptive jitter (falls back to eigen clip)."""
    cov = np.asarray(cov, dtype=np.float64)
    cov = 0.5 * (cov + cov.T)
    d = cov.shape[0]
    for scale in (0.0, 1e-10, 1e-8, 1e-6, 1e-4):
        try:
            return np.linalg.cholesky(cov + scale * np.eye(d))
        except np.linalg.LinAlgError:
            continue
    w, V = np.linalg.eigh(cov)
    w = np.clip(w, 1e-10, None)
    return V @ np.diag(np.sqrt(w))


def _symmetric_sqrt(mat: np.ndarray) -> np.ndarray:
    """Symmetric positive-(semi)definite square root via eigendecomposition."""
    mat = 0.5 * (np.asarray(mat, dtype=np.float64) + mat.T)
    w, V = np.linalg.eigh(mat)
    w = np.clip(w, 0.0, None)
    return V @ np.diag(np.sqrt(w)) @ V.T


def ensemble_mean_cov(ens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean ``(d,)`` and sample covariance ``(d, d)`` of an ``(N, d)`` ensemble."""
    ens = np.asarray(ens, dtype=np.float64)
    mean = ens.mean(axis=0)
    dev = ens - mean[None, :]
    cov = dev.T @ dev / (ens.shape[0] - 1)
    return mean, cov


# =========================================================================== filter
@dataclass
class FilterStep:
    """Logged quantities for one assimilation frame."""

    frame: int
    prior_ens: np.ndarray  # (N, d)
    analysis_ens: np.ndarray  # (N, d)
    innovation: np.ndarray  # (K,) = y - H(prior_mean)
    innovation_cov: np.ndarray  # (K, K) = Pyy = HPfH^T + R
    nis: float  # innovation^T Pyy^{-1} innovation


@dataclass
class FilterResult:
    """Full record of a filter run over the assimilation window."""

    frames: np.ndarray  # (T,)
    prior_mean: np.ndarray  # (T, d)
    analysis_mean: np.ndarray  # (T, d)
    prior_ens: np.ndarray  # (T, N, d)
    analysis_ens: np.ndarray  # (T, N, d)
    innovation: np.ndarray  # (T, K)
    innovation_cov: np.ndarray  # (T, K, K)
    nis: np.ndarray  # (T,)
    meta: dict = field(default_factory=dict)


class EnsembleKalmanFilter:
    """Stochastic EnKF (default) or deterministic square-root ETKF (flag).

    The state is the pooled latent; the measurement is ``H(z)`` (pressure taps).
    Only ``p_K`` ever enters the innovation, so no observable probe can leak in.
    """

    def __init__(
        self,
        forecast_model: ForecastModel,
        H: ObservationOperator,
        Q: np.ndarray | None,
        *,
        n_members: int = 64,
        inflation: float = 1.0,
        mode: str = "stochastic",
        seed: int = 0,
    ) -> None:
        if mode not in ("stochastic", "sqrt"):
            raise ValueError(f"mode must be 'stochastic' or 'sqrt'; got {mode!r}")
        self.model = forecast_model
        self.H = H
        self.Q = None if Q is None else np.asarray(Q, dtype=np.float64)
        self.n_members = int(n_members)
        self.inflation = float(inflation)
        self.mode = mode
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.max_context = int(getattr(forecast_model, "max_context", 32))

    # ---- forecast ----
    def forecast(self, buffer: np.ndarray) -> np.ndarray:
        """Advance every member one step, add process noise, apply inflation.

        ``buffer`` is ``(N, T_ctx, d)`` of past ANALYSIS states. The order is:
        model step -> additive Q -> multiplicative inflation of the deviations
        (so Q is not itself inflated).
        """
        prior = np.asarray(self.model.step(buffer), dtype=np.float64)  # (N, d)
        n, d = prior.shape
        if self.Q is not None:
            L = _safe_cholesky(self.Q)
            prior = prior + self.rng.standard_normal((n, d)) @ L.T
        if self.inflation != 1.0:
            m = prior.mean(axis=0, keepdims=True)
            prior = m + self.inflation * (prior - m)
        return prior

    # ---- analysis ----
    def analysis(self, prior: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict]:
        """EnKF measurement update. Innovation uses ``H(z)`` (pressure) only."""
        prior = np.asarray(prior, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        R = self.H.R
        Hx = np.asarray(self.H.apply(prior), dtype=np.float64)  # (N, K)
        n = prior.shape[0]
        xbar = prior.mean(axis=0)
        hbar = Hx.mean(axis=0)
        Xf = prior - xbar[None, :]  # (N, d)
        Yf = Hx - hbar[None, :]  # (N, K)
        Pyy = Yf.T @ Yf / (n - 1) + R  # (K, K)
        innov_mean = y - hbar
        Pyy_inv = np.linalg.inv(Pyy)
        nis = float(innov_mean @ Pyy_inv @ innov_mean)

        if self.mode == "stochastic":
            Pxy = Xf.T @ Yf / (n - 1)  # (d, K)
            gain = Pxy @ Pyy_inv  # (d, K)
            Lr = _safe_cholesky(R)
            pert = self.rng.standard_normal((n, R.shape[0])) @ Lr.T
            d_perturbed = y[None, :] + pert  # (N, K)
            innov = d_perturbed - Hx  # (N, K)
            analysis = prior + innov @ gain.T  # (N, d)
        else:  # deterministic ETKF square-root (Hunt et al. 2007), no localization
            analysis = self._etkf_update(prior, Xf, Yf, xbar, y, hbar, R)

        diag = {"innovation": innov_mean, "innovation_cov": Pyy, "nis": nis}
        return analysis, diag

    def _etkf_update(self, prior, Xf, Yf, xbar, y, hbar, R):
        n = prior.shape[0]
        Rinv = np.linalg.inv(R)
        Yb = Yf.T  # (K, N)
        C = Yb.T @ Rinv  # (N, K)
        rho = self.inflation if self.inflation > 0 else 1.0
        A = ((n - 1) / rho) * np.eye(n) + C @ Yb  # (N, N)
        Pa_tilde = np.linalg.inv(A)  # (N, N)
        wbar = Pa_tilde @ C @ (y - hbar)  # (N,)
        Wa = _symmetric_sqrt((n - 1) * Pa_tilde)  # (N, N)
        W = Wa + wbar[:, None]  # (N, N)
        # Xa columns = xbar + Xf^T W ; Xf is (N, d) so Xf.T is (d, N).
        Xa = xbar[None, :] + (W.T @ Xf)  # (N, d)
        return Xa

    # ---- run ----
    def run(
        self,
        y_series: np.ndarray,
        init_ens: np.ndarray,
        frames: np.ndarray,
        *,
        meta: dict | None = None,
    ) -> FilterResult:
        """Run the filter over the assimilation window.

        Args:
            y_series: ``(T, K)`` tap-pressure measurements, one per assimilation
                frame (already restricted to ``H.taps``).
            init_ens: ``(N, d)`` field-free initial ensemble at the first frame.
            frames: ``(T,)`` absolute frame indices (for logging / phase mapping).

        Frame 0 is the seeded prior (no forecast); it is assimilated directly. Each
        subsequent frame forecasts from the rolling ANALYSIS buffer, then updates.
        """
        y_series = np.asarray(y_series, dtype=np.float64)
        frames = np.asarray(frames)
        T = y_series.shape[0]
        n, d = init_ens.shape

        prior_ens_log = np.empty((T, n, d))
        analysis_ens_log = np.empty((T, n, d))
        innov_log = np.empty((T, self.H.output_dim))
        innov_cov_log = np.empty((T, self.H.output_dim, self.H.output_dim))
        nis_log = np.empty(T)

        # The rolling buffer holds ANALYSIS states only. It starts empty; frame 0 is
        # the seeded prior (the field-free init), whose analysis becomes the first
        # buffer entry. Frame i>0 forecasts from the buffer of past analyses, so the
        # buffer at forecast time has length i (analyses of frames 0..i-1).
        buffer: np.ndarray | None = None
        for i in range(T):
            if i == 0:
                prior = np.asarray(init_ens, dtype=np.float64)
            else:
                prior = self.forecast(buffer)
            analysis, diag = self.analysis(prior, y_series[i])
            prior_ens_log[i] = prior
            analysis_ens_log[i] = analysis
            innov_log[i] = diag["innovation"]
            innov_cov_log[i] = diag["innovation_cov"]
            nis_log[i] = diag["nis"]
            # append the corrected (analysis) state; this is what enters the next
            # step's forecast context.
            a = analysis[:, None, :]
            buffer = a if buffer is None else np.concatenate([buffer, a], axis=1)
            if buffer.shape[1] > self.max_context:
                buffer = buffer[:, -self.max_context :, :]

        return FilterResult(
            frames=frames,
            prior_mean=prior_ens_log.mean(axis=1),
            analysis_mean=analysis_ens_log.mean(axis=1),
            prior_ens=prior_ens_log,
            analysis_ens=analysis_ens_log,
            innovation=innov_log,
            innovation_cov=innov_cov_log,
            nis=nis_log,
            meta=dict(meta or {}),
        )

    # ---- free-run branch hook (Track D reuse) ----
    def branch_free_run(
        self, analysis_ens: np.ndarray, buffer: np.ndarray, steps: int
    ) -> np.ndarray:
        """Free-run forecast (no assimilation) from an analysis ensemble.

        Given the analysis ensemble at a phase boundary and its rolling buffer,
        roll ``steps`` frames forward with the forecast model only (no update),
        returning the ``(steps, N, d)`` free-run ensemble trajectory. Provided as a
        hook for the later Track D phase-boundary forecasts.
        """
        buf = np.asarray(buffer, dtype=np.float64)
        out = np.empty((steps, analysis_ens.shape[0], analysis_ens.shape[1]))
        for s in range(steps):
            prior = self.forecast(buf)
            out[s] = prior
            buf = np.concatenate([buf, prior[:, None, :]], axis=1)
            if buf.shape[1] > self.max_context:
                buf = buf[:, -self.max_context :, :]
        return out


__all__ = [
    "ForecastModel",
    "LinearForecast",
    "TransformerForecast",
    "FieldFreeInit",
    "truth_init_ensemble",
    "process_noise_from_residuals",
    "ensemble_mean_cov",
    "EnsembleKalmanFilter",
    "FilterResult",
    "FilterStep",
]
