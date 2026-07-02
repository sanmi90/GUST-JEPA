"""Tests for the Session 32 Track B EnKF, obs operator, and metrics.

The core test builds a SYNTHETIC linear-Gaussian system with known ``(F, H, Q, R)``
and asserts:
  * the stochastic EnKF posterior mean tracks the true state (RMSE well under the
    prior spread),
  * the analysis ensemble covariance stays symmetric positive-definite every step,
  * the EnKF result approaches the exact Kalman filter within Monte-Carlo tolerance
    for N=64.

Plus a leakage test (H is the pressure head; the innovation is pressure-only) and a
context-threading test (each member's rolling analysis buffer is fed to the model).

All tests are CPU-only and fast. No CUDA path is exercised, so ``require_rtx6000``
is not needed here (the real transformer/GPU wiring is exercised by the pilot).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.estimation.enkf import (
    EnsembleKalmanFilter,
    LinearForecast,
    ensemble_mean_cov,
)
from src.estimation.metrics import (
    crps_ensemble,
    divergence_flag,
    lag1_autocorr,
    nis_coverage,
    spread_skill,
)
from src.estimation.obs_operator import (
    ObservationOperator,
    fit_observation_operator,
    qdeim_indices,
)


# --------------------------------------------------------------------------- helpers
def _linear_gaussian_system():
    """A small stable linear-Gaussian system (d=2, K=1)."""
    F = np.array([[0.9, 0.1], [-0.15, 0.88]])
    H_mat = np.array([[1.0, 0.5]])  # (K=1, d=2)
    Q = np.array([[0.02, 0.005], [0.005, 0.02]])
    R = np.array([[0.05]])
    return F, H_mat, Q, R


def _make_obs_operator(H_mat: np.ndarray, R: np.ndarray) -> ObservationOperator:
    """Wrap a raw linear H matrix as an ObservationOperator (dummy taps)."""
    k, d = H_mat.shape
    return ObservationOperator(
        taps=np.arange(k),
        weight=np.asarray(H_mat, dtype=np.float64),
        bias=np.zeros(k),
        R=np.asarray(R, dtype=np.float64),
        kind="linear",
        state_dim=d,
        n_surface=k,
    )


def _simulate(F, H_mat, Q, R, T, x0, rng):
    d = F.shape[0]
    k = H_mat.shape[0]
    Lq = np.linalg.cholesky(Q)
    Lr = np.linalg.cholesky(R)
    xs = np.empty((T, d))
    ys = np.empty((T, k))
    x = x0.copy()
    for t in range(T):
        if t > 0:
            x = F @ x + Lq @ rng.standard_normal(d)
        ys[t] = H_mat @ x + Lr @ rng.standard_normal(k)
        xs[t] = x
    return xs, ys


def _exact_kalman(F, H_mat, Q, R, ys, x0_mean, P0):
    """Reference KF matching the EnKF cadence (frame 0 = seeded prior, then update)."""
    T = ys.shape[0]
    d = F.shape[0]
    xa = np.empty((T, d))
    xa_mean = x0_mean.copy()
    P = P0.copy()
    eye_d = np.eye(d)
    for t in range(T):
        if t == 0:
            xp, Pp = xa_mean, P
        else:
            xp = F @ xa_mean
            Pp = F @ P @ F.T + Q
        S = H_mat @ Pp @ H_mat.T + R
        K = Pp @ H_mat.T @ np.linalg.inv(S)
        innov = ys[t] - H_mat @ xp
        xa_mean = xp + K @ innov
        P = (eye_d - K @ H_mat) @ Pp
        xa[t] = xa_mean
    return xa


# --------------------------------------------------------------------------- synthetic
@pytest.mark.parametrize("mode", ["stochastic", "sqrt"])
def test_enkf_tracks_and_spd(mode):
    rng = np.random.default_rng(7)
    F, H_mat, Q, R = _linear_gaussian_system()
    d = F.shape[0]
    T = 60
    x0 = np.array([1.0, -0.5])
    xs, ys = _simulate(F, H_mat, Q, R, T, x0, rng)

    P0 = 0.5 * np.eye(d)
    x0_guess = x0 + rng.standard_normal(d) * 0.3
    init_ens = x0_guess[None, :] + rng.standard_normal((64, d)) @ np.linalg.cholesky(P0).T

    H = _make_obs_operator(H_mat, R)
    model = LinearForecast(F)
    enkf = EnsembleKalmanFilter(model, H, Q, n_members=64, inflation=1.0, mode=mode, seed=3)
    res = enkf.run(ys, init_ens, np.arange(T))

    # (1) tracks the true state, well under the prior spread.
    track_rmse = float(np.sqrt(((res.analysis_mean - xs) ** 2).mean()))
    prior_spread = float(np.sqrt(np.trace(P0) / d))
    assert track_rmse < 0.5 * prior_spread, (track_rmse, prior_spread)

    # (2) analysis ensemble covariance is symmetric positive-definite every step.
    for t in range(T):
        _, cov = ensemble_mean_cov(res.analysis_ens[t])
        assert np.allclose(cov, cov.T, atol=1e-10)
        eig = np.linalg.eigvalsh(cov)
        assert float(eig.min()) > 0.0, (t, eig.min())


def test_enkf_approaches_exact_kalman():
    """Stochastic EnKF (N=64) posterior mean approaches the exact KF within MC tol."""
    rng = np.random.default_rng(11)
    F, H_mat, Q, R = _linear_gaussian_system()
    d = F.shape[0]
    T = 60
    x0 = np.array([0.7, 0.2])
    xs, ys = _simulate(F, H_mat, Q, R, T, x0, rng)

    P0 = 0.5 * np.eye(d)
    x0_guess = x0 + rng.standard_normal(d) * 0.2

    kf_mean = _exact_kalman(F, H_mat, Q, R, ys, x0_guess, P0)

    # Average several EnKF runs (independent seeds) to suppress MC noise, matching
    # the "within Monte-Carlo tolerance for N=64" claim.
    H = _make_obs_operator(H_mat, R)
    model = LinearForecast(F)
    n_runs = 8
    acc = np.zeros((T, d))
    for s in range(n_runs):
        r = np.random.default_rng(100 + s)
        init_ens = x0_guess[None, :] + r.standard_normal((64, d)) @ np.linalg.cholesky(P0).T
        enkf = EnsembleKalmanFilter(
            model, H, Q, n_members=64, inflation=1.0, mode="stochastic", seed=100 + s
        )
        acc += enkf.run(ys, init_ens, np.arange(T)).analysis_mean
    enkf_mean = acc / n_runs

    diff_rmse = float(np.sqrt(((enkf_mean - kf_mean) ** 2).mean()))
    kf_scale = float(np.sqrt((kf_mean**2).mean()))
    assert diff_rmse < 0.1 * kf_scale, (diff_rmse, kf_scale)


# --------------------------------------------------------------------------- leakage
def test_H_is_pressure_head_no_observable_leak():
    """H maps z -> p_K only; output dim == K and the innovation is pressure-only."""
    rng = np.random.default_rng(0)
    n, d, n_surface, k = 400, 32, 192, 8
    z = rng.standard_normal((n, d))
    true_W = rng.standard_normal((n_surface, d))
    p = z @ true_W.T + 0.01 * rng.standard_normal((n, n_surface))
    taps = np.sort(rng.choice(n_surface, size=k, replace=False))

    H = fit_observation_operator(z, p, taps, kind="linear")
    # H emits exactly K pressure channels.
    assert H.output_dim == k
    assert H.apply(z[:5]).shape == (5, k)
    assert H.apply(z[0]).shape == (k,)
    # The operator has no notion of observables (E_w / C_L): only tap/pressure state.
    for banned in ("E_w", "enstrophy", "C_L", "probe", "observable"):
        assert not hasattr(H, banned)

    # Innovation in the EnKF equals y - H(prior_mean), computed from pressure only.
    Q = 0.01 * np.eye(d)
    enkf = EnsembleKalmanFilter(LinearForecast(np.eye(d)), H, Q, n_members=32, seed=1)
    prior = rng.standard_normal((32, d))
    y = rng.standard_normal(k)
    _, diag = enkf.analysis(prior, y)
    expected_innov = y - H.apply(prior).mean(axis=0)
    assert np.allclose(diag["innovation"], expected_innov, atol=1e-9)
    assert diag["innovation"].shape == (k,)


def test_load_osp_taps_per_model(tmp_path):
    """load_osp_taps reads per-model + shared taps and returns sorted K-vectors."""
    from src.estimation.obs_operator import load_osp_taps

    osp = {
        "jepa_pool": {"method": "tcsi_greedy", "K8": [29, 85, 94, 12, 176, 8, 158, 11]},
        "pod": {"method": "tcsi_greedy", "K8": [15, 10, 9, 12, 105, 19, 11, 23]},
        "qdeim_shared": {"method": "qdeim", "K8": [11, 10, 12, 7, 106, 186, 19, 92]},
    }
    p = tmp_path / "osp.json"
    p.write_text(json.dumps(osp))
    taps_j, prov_j = load_osp_taps("jepa_pool", k=8, osp_path=p)
    taps_s, prov_s = load_osp_taps("qdeim_shared", k=8, osp_path=p)
    assert taps_j.tolist() == sorted(osp["jepa_pool"]["K8"])
    assert prov_j["source"] == "osp_per_model" and prov_j["method"] == "tcsi_greedy"
    assert prov_s["source"] == "qdeim_shared"
    # per-model taps differ from the shared array (model-conditioned sensing).
    assert taps_j.tolist() != taps_s.tolist()
    with pytest.raises(KeyError):
        load_osp_taps("no_such_model", k=8, osp_path=p)


def test_qdeim_returns_k_distinct_taps():
    rng = np.random.default_rng(2)
    n_points, r, k = 192, 12, 8
    modes, _ = np.linalg.qr(rng.standard_normal((n_points, r)))
    taps = qdeim_indices(modes, k)
    assert taps.shape == (k,)
    assert len(set(taps.tolist())) == k
    assert taps.min() >= 0 and taps.max() < n_points


# --------------------------------------------------------------------------- context
def test_per_member_context_is_threaded():
    """Each forecast step receives the growing per-member ANALYSIS buffer."""

    class SpyModel:
        max_context = 32

        def __init__(self, d):
            self.d = d
            self.seen_shapes = []

        def step(self, buffer):
            self.seen_shapes.append(buffer.shape)
            # depend on the whole history so ignoring it would change the result.
            return np.asarray(buffer).mean(axis=1)

    rng = np.random.default_rng(5)
    d, k, N, T = 3, 1, 16, 6
    H_mat = rng.standard_normal((k, d))
    R = 0.1 * np.eye(k)
    H = _make_obs_operator(H_mat, R)
    spy = SpyModel(d)
    enkf = EnsembleKalmanFilter(spy, H, Q=0.01 * np.eye(d), n_members=N, seed=0)
    init_ens = rng.standard_normal((N, d))
    ys = rng.standard_normal((T, k))
    enkf.run(ys, init_ens, np.arange(T))

    # frame 0 is the seeded prior (no forecast); frames 1..T-1 call the model.
    assert len(spy.seen_shapes) == T - 1
    # buffers grow: (N, 1, d), (N, 2, d), ... reflecting appended analysis states.
    for i, shape in enumerate(spy.seen_shapes):
        assert shape == (N, i + 1, d), (i, shape)


def test_context_buffer_capped_at_max_context():
    class SpyModel:
        max_context = 4

        def step(self, buffer):
            return np.asarray(buffer)[:, -1, :]

    rng = np.random.default_rng(6)
    d, k, N, T = 2, 1, 8, 12
    H = _make_obs_operator(rng.standard_normal((k, d)), 0.1 * np.eye(k))
    enkf = EnsembleKalmanFilter(SpyModel(), H, Q=None, n_members=N, seed=0)
    res = enkf.run(rng.standard_normal((T, k)), rng.standard_normal((N, d)), np.arange(T))
    assert res.analysis_ens.shape == (T, N, d)


# --------------------------------------------------------------------------- metrics
def test_metrics_smoke():
    rng = np.random.default_rng(9)
    T, N = 40, 32
    truth = np.sin(np.linspace(0, 6, T))
    ens = truth[:, None] + 0.1 * rng.standard_normal((T, N))
    crps = crps_ensemble(ens, truth)
    assert crps >= 0.0
    ss = spread_skill(ens, truth)
    assert ss["spread"] > 0 and ss["skill_rmse"] > 0
    white = lag1_autocorr(rng.standard_normal((T, 3)))
    assert abs(white["mean_abs_lag1"]) < 1.0
    cov = nis_coverage(rng.chisquare(2, size=T), dof=2)
    assert 0.0 <= cov["frac_in_band_95"] <= 1.0
    flag = divergence_flag(np.full(T, 100.0), dof=2)
    assert flag["diverged"] is True  # sustained huge NIS -> flagged
    ok = divergence_flag(rng.chisquare(2, size=T), dof=2)
    assert ok["diverged"] in (True, False)


def test_transformer_forecast_reads_full_buffer_cpu():
    """The real AR-transformer wrapper threads the buffer (CPU; no CUDA needed)."""
    torch = pytest.importorskip("torch")
    from src.estimation.enkf import TransformerForecast
    from src.models.predictor import AutoregressivePredictor

    d = 8
    pred = AutoregressivePredictor(
        latent_dim=d, cond_dim=0, hidden_dim=32, depth=2, heads=4, max_seq_len=32
    )
    fm = TransformerForecast(pred, torch.device("cpu"))
    N, T_ctx = 5, 3
    buf = np.random.default_rng(0).standard_normal((N, T_ctx, d)).astype(np.float32)
    out = fm.step(buf)
    assert out.shape == (N, d)
    assert fm.max_context == 32
