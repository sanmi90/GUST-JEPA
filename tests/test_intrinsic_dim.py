"""Unit tests for src/evaluation/intrinsic_dim.py.

The four estimators (PCA, Levina-Bickel MLE, Two-NN, Isomap residual) are
tested on synthetic manifolds with known intrinsic dimension. All checks
use small (<= 1500 sample) clouds so the suite stays CPU-bound and runs in
under ~30 seconds.

Tolerances are deliberately generous: both Levina-Bickel and Two-NN are
known to underestimate intrinsic dim under high curvature, and Isomap's
elbow heuristic is noisy by construction. The point of these tests is to
catch implementation regressions, not to validate the estimators
themselves (those are validated in the original papers).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.intrinsic_dim import (
    agreement_summary,
    isomap_residual,
    levina_bickel_mle,
    pca_dim_at_threshold,
    pca_spectrum,
    two_nn,
)


def _sample_ball(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform sample from the unit d-ball in R^d.

    Standard recipe: Gaussian direction times Uniform^(1/d) radius.
    """
    g = rng.standard_normal(size=(n, d))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    r = rng.uniform(size=(n, 1)) ** (1.0 / d)
    return g * r


def _embed(points: np.ndarray, D: int, rng: np.random.Generator) -> np.ndarray:
    """Linearly embed ``(n, d)`` points into ``R^D`` via a random isometry."""
    n, d = points.shape
    assert D >= d, "ambient dim must be >= intrinsic dim"
    # Random orthonormal columns: QR of a Gaussian matrix.
    A = rng.standard_normal(size=(D, d))
    Q, _ = np.linalg.qr(A)
    return points @ Q.T  # (n, D)


def test_pca_rank3_no_noise() -> None:
    """A rank-3 matrix in 10-D ambient has PCA dim = 3 at threshold 0.99."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal(size=(100, 3))
    B = rng.standard_normal(size=(3, 10))
    X = A @ B
    assert pca_dim_at_threshold(X, threshold=0.99) == 3
    # Spectrum should have exactly 3 non-trivial singular directions.
    spectrum = pca_spectrum(X)
    assert spectrum[:3].sum() > 0.999


def test_pca_unit_sphere_in_10d_ambient() -> None:
    """Unit-norm random 5D vectors projected into 10D have PCA dim = 5."""
    rng = np.random.default_rng(1)
    v = rng.standard_normal(size=(400, 5))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    X = _embed(v, D=10, rng=rng)
    # The 5D ambient subspace is captured fully; PCA cannot see the
    # spherical constraint.
    assert pca_dim_at_threshold(X, threshold=0.95) == 5


def test_levina_bickel_2d_disk_in_10d() -> None:
    """Levina-Bickel on a 2D disk embedded in 10D returns ~2."""
    rng = np.random.default_rng(2)
    disk = _sample_ball(800, d=2, rng=rng)
    X = _embed(disk, D=10, rng=rng)
    est = levina_bickel_mle(X, k=10)
    assert isinstance(est, float)
    assert abs(est - 2.0) < 0.5, f"expected ~2, got {est}"


def test_levina_bickel_5d_ball() -> None:
    """Levina-Bickel on a 5D ball returns ~5."""
    rng = np.random.default_rng(3)
    X = _sample_ball(1200, d=5, rng=rng)
    est = levina_bickel_mle(X, k=10)
    assert isinstance(est, float)
    assert abs(est - 5.0) < 1.0, f"expected ~5, got {est}"


def test_two_nn_3d_gaussian_cloud() -> None:
    """Two-NN on a 3D Gaussian cloud returns ~3."""
    rng = np.random.default_rng(4)
    X = rng.standard_normal(size=(1500, 3))
    est = two_nn(X)
    assert abs(est - 3.0) < 0.5, f"expected ~3, got {est}"


def test_two_nn_helix_1d() -> None:
    """Two-NN on a helix (intrinsically 1D) in 3D returns ~1.

    Two-NN assumes iid uniform sampling on the manifold; uniform grid
    spacing in the parameter ``t`` produces a deterministic mu = 2 and
    blows up the slope. Drawing ``t`` from a uniform distribution gives
    the Poisson-process spacing the estimator expects.
    """
    rng = np.random.default_rng(5)
    t = rng.uniform(0.0, 4.0 * np.pi, size=1500)
    X = np.stack([np.cos(t), np.sin(t), 0.5 * t], axis=1) + 1e-4 * rng.standard_normal(
        size=(1500, 3)
    )
    est = two_nn(X)
    assert abs(est - 1.0) < 0.5, f"expected ~1, got {est}"


def test_isomap_swiss_roll_returns_2() -> None:
    """Isomap residual on a Swiss roll returns elbow_dim = 2 (intrinsic 2D)."""
    rng = np.random.default_rng(6)
    # Standard Swiss-roll construction.
    n = 800
    t = 1.5 * np.pi * (1.0 + 2.0 * rng.uniform(size=n))
    h = 21.0 * rng.uniform(size=n)
    X = np.stack([t * np.cos(t), h, t * np.sin(t)], axis=1)
    # Embed into 6D ambient via random isometry to make the test less
    # trivial than raw 3D coordinates.
    X = _embed(X, D=6, rng=rng)
    result = isomap_residual(X, k_max=6, n_neighbors=10)
    assert result["elbow_dim"] == 2, (
        f"expected elbow at dim 2; got {result['elbow_dim']}; "
        f"curve = {result['residual_variance']}"
    )


def test_agreement_summary_4d_ball_consensus() -> None:
    """agreement_summary on a 4D ball in 20D ambient: consensus within 2 of 4."""
    rng = np.random.default_rng(7)
    ball = _sample_ball(800, d=4, rng=rng)
    X = _embed(ball, D=20, rng=rng)
    summary = agreement_summary(X)
    # Spread is allowed to be wider than the consensus tolerance because
    # PCA gives a hard upper bound (= 4) while the curvature-sensitive
    # estimators may underestimate.
    assert abs(summary["consensus"] - 4.0) <= 2.0, summary
    assert summary["pca_95"] == 4
    assert "mean" in summary["levina_bickel"]


def test_levina_bickel_with_list_of_k() -> None:
    """levina_bickel_mle returns a dict with a 'mean' key when k is a list."""
    rng = np.random.default_rng(8)
    X = _sample_ball(500, d=3, rng=rng)
    est = levina_bickel_mle(X, k=[5, 10, 15])
    assert isinstance(est, dict)
    assert "mean" in est
    assert set(est.keys()) == {5, 10, 15, "mean"}
    # All individual estimates should be positive and roughly near 3.
    for k_val in (5, 10, 15):
        assert est[k_val] > 0.0
        assert abs(est[k_val] - 3.0) < 1.0


def test_errors_on_bad_input() -> None:
    """All public functions raise sensible errors on empty or tiny input."""
    empty = np.empty((0, 5))
    too_small = np.array([[1.0, 2.0]])  # 1 sample

    with pytest.raises(ValueError):
        pca_spectrum(empty)
    with pytest.raises(ValueError):
        pca_spectrum(too_small)
    with pytest.raises(ValueError):
        pca_dim_at_threshold(empty, threshold=0.95)
    with pytest.raises(ValueError):
        pca_dim_at_threshold(np.random.default_rng(0).standard_normal((10, 3)), threshold=1.5)

    # Levina-Bickel needs more samples than k.
    small = np.random.default_rng(0).standard_normal((4, 3))
    with pytest.raises(ValueError):
        levina_bickel_mle(small, k=10)
    with pytest.raises(ValueError):
        levina_bickel_mle(small, k=1)  # k < 2 is invalid

    with pytest.raises(ValueError):
        two_nn(too_small)
    with pytest.raises(ValueError):
        two_nn(empty)
    # discard_fraction out of range.
    X = np.random.default_rng(0).standard_normal((50, 3))
    with pytest.raises(ValueError):
        two_nn(X, discard_fraction=1.5)
    with pytest.raises(ValueError):
        two_nn(X, discard_fraction=-0.1)

    # isomap_residual requires at least n_neighbors + 2 samples.
    tiny = np.random.default_rng(0).standard_normal((3, 3))
    with pytest.raises(ValueError):
        isomap_residual(tiny, k_max=4, n_neighbors=10)
    with pytest.raises(ValueError):
        isomap_residual(empty, k_max=4, n_neighbors=10)
