"""Unit tests for src/evaluation/conditional_structural_information.py.

The module under test is TCSI (target-conditioned structural information),
the deliberately-renamed cousin of epiplexity. One of the tests below is a
naming guard that asserts the string "epiplexity" does not appear in any
public function or class name; this guard is load-bearing for the paper
and must be kept.
"""

from __future__ import annotations

import numpy as np

import src.evaluation.conditional_structural_information as tcsi
from src.evaluation.conditional_structural_information import (
    RidgeProxyLearner,
    TCSIProxies,
    compute_proxies,
    compute_proxies_for_sensor,
    null_predictor_loss,
    objective_J,
)


def test_null_predictor_loss_on_constant_targets_is_zero() -> None:
    """A constant target has zero variance, hence zero null-baseline MSE."""
    targets = np.full(64, 3.14, dtype=np.float64)
    assert null_predictor_loss(targets) == 0.0


def test_null_predictor_loss_on_standard_normal_approaches_unit_variance() -> None:
    """For large N the null-baseline MSE on N(0, 1) approaches 1.0."""
    rng = np.random.default_rng(0)
    targets = rng.standard_normal(200_000)
    loss = null_predictor_loss(targets)
    assert abs(loss - 1.0) < 0.02, f"expected ~1.0, got {loss}"


def test_compute_proxies_on_synthetic_curve_matches_hand_computation() -> None:
    """Sanity-check the four proxies on a hand-traced loss curve.

    Curve = [10, 8, 6, 4, 2, 2, 2, 2, 2, 2], L_null = 10.
    With the default "median of last 10 percent" rule and N = 10, the tail
    has length 1 so L_star = 2.0.

    Hand computation:
        S_preq = (10-2)+(8-2)+(6-2)+(4-2)+0+0+0+0+0+0 = 20
        H_res  = 10 * 2 = 20
        G      = 10 * (10 - 2) = 80
        Eff    = 80 / (20 + eps) ~= 4.0
    """
    losses = np.array([10, 8, 6, 4, 2, 2, 2, 2, 2, 2], dtype=np.float64)
    proxies = compute_proxies(losses, L_null=10.0)
    assert proxies.L_star == 2.0
    assert proxies.S_preq == 20.0
    assert proxies.H_res == 20.0
    assert proxies.G == 80.0
    assert abs(proxies.Eff - 4.0) < 1e-4


def test_compute_proxies_with_explicit_L_star_override() -> None:
    """When ``L_star`` is supplied explicitly the median rule is ignored."""
    losses = np.array([5.0, 5.0, 5.0, 5.0, 5.0], dtype=np.float64)
    proxies = compute_proxies(losses, L_null=8.0, L_star=1.0)
    assert proxies.L_star == 1.0
    expected_S_preq = float(np.sum(np.maximum(0.0, losses - 1.0)))
    assert proxies.S_preq == expected_S_preq
    assert proxies.H_res == 5.0 * 1.0
    assert proxies.G == 5.0 * (8.0 - 1.0)


def test_ridge_recovers_known_linear_slope() -> None:
    """Closed-form ridge with weak regularisation recovers y = 2 * X."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((400, 1))
    y = 2.0 * X[:, 0] + 0.01 * rng.standard_normal(400)
    learner = RidgeProxyLearner(alpha=1e-3)
    learner.fit(X, y)
    y_hat = learner.predict(X)
    solution, *_ = np.linalg.lstsq(X - X.mean(0), y_hat - y_hat.mean(), rcond=None)
    slope = float(solution.reshape(-1)[0])
    assert abs(slope - 2.0) < 0.05, f"expected slope ~2.0, got {slope}"


def test_ridge_fit_with_loss_curve_returns_length_two_array() -> None:
    """``fit_with_loss_curve`` returns a 2-point loss curve and the eval null."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal((100, 3))
    y = X @ np.array([1.0, -0.5, 0.25]) + 0.1 * rng.standard_normal(100)
    learner = RidgeProxyLearner(alpha=1.0)
    loss_curve, L_null = learner.fit_with_loss_curve(X, y)
    assert isinstance(loss_curve, np.ndarray)
    assert loss_curve.shape == (2,)
    assert loss_curve[0] == L_null  # null sits at the head of the synthesised curve
    assert loss_curve[1] < loss_curve[0]  # fit beats null on a linear signal


def test_compute_proxies_for_sensor_strong_signal_yields_positive_gain() -> None:
    """A sensor that perfectly explains the target has G > 0 and Eff > 0."""
    rng = np.random.default_rng(3)
    N, W = 300, 4
    sensor = rng.standard_normal((N, W))
    target = sensor.sum(axis=1) + 0.01 * rng.standard_normal(N)
    proxies = compute_proxies_for_sensor(sensor, target)
    assert proxies.G > 0.0
    assert proxies.Eff > 0.0


def test_compute_proxies_for_sensor_noise_signal_has_near_zero_gain() -> None:
    """Independent noise yields a gain comparable to a single-sample slack."""
    rng = np.random.default_rng(4)
    N, W = 300, 4
    sensor = rng.standard_normal((N, W))
    target = rng.standard_normal(N)
    L_null = null_predictor_loss(target)
    proxies = compute_proxies_for_sensor(sensor, target)
    # The cheap learner cannot beat the null on independent noise by more
    # than a small in-sample overfit; require G to be far below the
    # full-information ceiling of N * L_null.
    assert proxies.G < 0.1 * N * L_null


def test_objective_J_default_weights_sum_correctly() -> None:
    """Default weights compose ``w_G G + w_S S_preq + w_H H_res + w_Eff Eff``."""
    proxies = TCSIProxies(
        S_preq=4.0, H_res=6.0, G=10.0, Eff=2.5, L_star=0.5, L_null=1.5
    )
    expected = 1.0 * 10.0 + (-0.5) * 4.0 + (-0.5) * 6.0 + 0.5 * 2.5
    assert objective_J(proxies) == expected


def test_no_public_name_contains_the_word_epiplexity() -> None:
    """The naming distinction is load-bearing for the paper; guard it."""
    forbidden = "epiplexity"
    public_names = [name for name in dir(tcsi) if not name.startswith("_")]
    assert public_names, "module exposes no public names"
    offenders = [name for name in public_names if forbidden in name.lower()]
    assert not offenders, (
        f"public name(s) {offenders} contain '{forbidden}'; TCSI is the "
        "intentionally-renamed cousin and must not use that word"
    )
