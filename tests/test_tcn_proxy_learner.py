"""Unit tests for src/evaluation/tcn_proxy_learner.py.

The TCN proxy learner is the stronger drop-in cousin of
:class:`RidgeProxyLearner`. These tests pin its shape contract and a
sanity check on linear signal convergence; full convergence behaviour
on the vortex-jepa pool is exercised by the Session 14 follow-ups
driver script, not in unit tests.
"""

from __future__ import annotations

import numpy as np

from src.evaluation.tcn_proxy_learner import (
    TCNConfig,
    TCNModule,
    TCNProxyLearner,
)


def test_tcn_module_param_count_is_small() -> None:
    """TCN with default config is roughly 20k params at K = 2 and K = 4."""
    m2 = TCNModule(in_channels=2, out_dim=64, config=TCNConfig())
    n2 = sum(p.numel() for p in m2.parameters())
    m4 = TCNModule(in_channels=4, out_dim=64, config=TCNConfig())
    n4 = sum(p.numel() for p in m4.parameters())
    # ~20-25k: TCN trunk is shared and the only K-dependent layer is the 1x1
    # input projection (K * base_channels = K * 32 weights).
    assert 10_000 < n2 < 60_000, f"K=2 model has {n2} params (expected ~20k)"
    assert 10_000 < n4 < 60_000, f"K=4 model has {n4} params (expected ~20k)"


def test_tcn_predict_shapes_1d_and_2d() -> None:
    """1D target returns (N,); 2D target returns (N, out_dim)."""
    rng = np.random.default_rng(0)
    N, K, W = 80, 2, 17
    X = rng.standard_normal((N, K, W)).astype(np.float32)

    y2 = rng.standard_normal((N, 4)).astype(np.float32)
    learner2 = TCNProxyLearner(
        out_dim=4, config=TCNConfig(epochs=2, device="cpu", seed=0)
    )
    learner2.fit(X, y2)
    yhat2 = learner2.predict(X)
    assert yhat2.shape == (N, 4)

    y1 = rng.standard_normal(N).astype(np.float32)
    learner1 = TCNProxyLearner(
        out_dim=1, config=TCNConfig(epochs=2, device="cpu", seed=0)
    )
    learner1.fit(X, y1)
    yhat1 = learner1.predict(X)
    assert yhat1.shape == (N,)


def test_tcn_learns_linear_combination_of_inputs() -> None:
    """TCN should beat the constant null on a clearly linear target."""
    rng = np.random.default_rng(0)
    N, K, W = 200, 2, 17
    X = rng.standard_normal((N, K, W)).astype(np.float32)
    y = X.sum(axis=(1, 2)).astype(np.float32)
    learner = TCNProxyLearner(
        out_dim=1, config=TCNConfig(epochs=30, device="cpu", seed=0)
    )
    learner.fit(X, y)
    y_hat = learner.predict(X)
    ss_res = float(np.mean((y - y_hat) ** 2))
    ss_tot = float(np.var(y))
    train_r2 = 1.0 - ss_res / ss_tot
    assert train_r2 > 0.5, f"TCN failed to fit a linear target: train R^2 = {train_r2}"


def test_tcn_predict_rejects_inconsistent_input_shape() -> None:
    """Predict with a different K than fit should raise."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 2, 17)).astype(np.float32)
    y = rng.standard_normal((20,)).astype(np.float32)
    learner = TCNProxyLearner(
        out_dim=1, config=TCNConfig(epochs=2, device="cpu", seed=0)
    )
    learner.fit(X, y)
    bad_X = rng.standard_normal((20, 3, 17)).astype(np.float32)
    try:
        learner.predict(bad_X)
    except ValueError:
        return
    raise AssertionError("expected ValueError on mismatched K")
