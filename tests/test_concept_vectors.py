"""Unit tests for :mod:`src.evaluation.concept_vectors`."""

from __future__ import annotations

import itertools
import logging

import numpy as np

from src.evaluation.concept_vectors import (
    consistency_check,
    construct_concept_vectors_averaging,
    construct_concept_vectors_jacobian,
    extrapolation_error,
    linear_extrapolate,
)


_LATENT_DIM = 64


def _make_linear_dataset(
    d: int = _LATENT_DIM,
    g_values: tuple[float, ...] = (-1.0, 0.0, 1.0, 2.0),
    d_values: tuple[float, ...] = (0.5, 1.0, 1.5),
    y_values: tuple[float, ...] = (-0.2, 0.0, 0.2, 0.4),
) -> tuple[np.ndarray, np.ndarray]:
    """Return latents / labels for ``z = 2G e_0 + 3D e_1 + 5Y e_2``.

    The remaining ``d - 3`` latent dimensions are exactly zero so the
    ground-truth concept vectors are
    ``v_G = 2 e_0, v_D = 3 e_1, v_Y = 5 e_2``.
    """
    rows = []
    labels = []
    for g, dd, y in itertools.product(g_values, d_values, y_values):
        z = np.zeros(d, dtype=np.float64)
        z[0] = 2.0 * g
        z[1] = 3.0 * dd
        z[2] = 5.0 * y
        rows.append(z)
        labels.append([g, dd, y])
    return np.asarray(rows), np.asarray(labels)


def _expected_vectors(d: int = _LATENT_DIM) -> dict[str, np.ndarray]:
    v_g = np.zeros(d, dtype=np.float64)
    v_g[0] = 2.0
    v_d = np.zeros(d, dtype=np.float64)
    v_d[1] = 3.0
    v_y = np.zeros(d, dtype=np.float64)
    v_y[2] = 5.0
    return {"G": v_g, "D": v_d, "Y": v_y}


# ---------------------------------------------------------------------------
# 1. Averaging on a clean linear dataset recovers the analytical vectors.
# ---------------------------------------------------------------------------
def test_averaging_recovers_linear_ground_truth() -> None:
    latents, labels = _make_linear_dataset()
    vecs = construct_concept_vectors_averaging(latents, labels)
    expected = _expected_vectors()
    for axis in ("G", "D", "Y"):
        v = vecs[axis]
        assert v is not None
        assert np.allclose(v, expected[axis], atol=1e-3), (
            f"axis {axis}: got {v[:6]}..., expected {expected[axis][:6]}..."
        )


# ---------------------------------------------------------------------------
# 2. Jacobian on the same dataset recovers the same vectors.
# ---------------------------------------------------------------------------
def test_jacobian_recovers_linear_ground_truth() -> None:
    latents, labels = _make_linear_dataset()
    vecs = construct_concept_vectors_jacobian(latents, labels, ridge_alpha=1e-8)
    expected = _expected_vectors()
    for axis in ("G", "D", "Y"):
        v = vecs[axis]
        assert v.shape == (_LATENT_DIM,)
        assert np.allclose(v, expected[axis], atol=1e-3), (
            f"axis {axis}: got {v[:6]}..., expected {expected[axis][:6]}..."
        )


# ---------------------------------------------------------------------------
# 3. Consistency check on (1) and (2) yields cosine_similarity > 0.99.
# ---------------------------------------------------------------------------
def test_consistency_check_agreement_on_linear() -> None:
    latents, labels = _make_linear_dataset()
    vecs_avg = construct_concept_vectors_averaging(latents, labels)
    vecs_jac = construct_concept_vectors_jacobian(latents, labels, ridge_alpha=1e-8)
    report = consistency_check(vecs_avg, vecs_jac)
    for axis in ("G", "D", "Y"):
        entry = report[axis]
        assert entry["axis_name"] == axis
        cos = entry["cosine_similarity"]
        assert cos > 0.99, f"axis {axis}: cosine_similarity {cos} <= 0.99"
        ratio = entry["norm_ratio"]
        assert 0.9 < ratio < 1.1, f"axis {axis}: norm_ratio {ratio} far from 1"


# ---------------------------------------------------------------------------
# 4. linear_extrapolate with a known basis gives the analytical prediction.
# ---------------------------------------------------------------------------
def test_linear_extrapolate_matches_closed_form() -> None:
    d = _LATENT_DIM
    z0 = np.zeros(d, dtype=np.float64)
    c0 = np.array([0.0, 0.0, 0.0])
    c_target = np.array([1.0, 2.0, 3.0])
    vecs = _expected_vectors(d)
    z_pred = linear_extrapolate(z0, c0, c_target, vecs)
    expected = np.zeros(d, dtype=np.float64)
    expected[0] = 2.0
    expected[1] = 6.0
    expected[2] = 15.0
    assert z_pred.shape == (d,)
    assert np.allclose(z_pred, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# 5. extrapolation_error on identical inputs is zero / unit cosine.
# ---------------------------------------------------------------------------
def test_extrapolation_error_self_consistency() -> None:
    rng = np.random.default_rng(0)
    z = rng.normal(size=(_LATENT_DIM,)).astype(np.float64) * 3.0
    metrics = extrapolation_error(z, z)
    assert metrics["rel_l2"] == 0.0
    assert metrics["abs_l2"] == 0.0
    assert abs(metrics["cosine_sim"] - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# 6. Averaging without any matched pairs for an axis returns None + warns.
# ---------------------------------------------------------------------------
def test_averaging_returns_none_when_no_matched_pairs(caplog) -> None:
    # All three labels co-vary -> no two encoders share two axes exactly.
    rng = np.random.default_rng(7)
    n = 12
    g = rng.normal(size=n)
    d_ax = rng.normal(size=n)
    y = rng.normal(size=n)
    labels = np.stack([g, d_ax, y], axis=1)
    latents = rng.normal(size=(n, _LATENT_DIM))
    with caplog.at_level(logging.WARNING, logger="src.evaluation.concept_vectors"):
        vecs = construct_concept_vectors_averaging(latents, labels, min_pairs=2)
    assert vecs["G"] is None
    assert vecs["D"] is None
    assert vecs["Y"] is None
    assert any("matched pair" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# 7. Jacobian is robust to ill-conditioning via the ridge term.
# ---------------------------------------------------------------------------
def test_jacobian_handles_redundant_latents() -> None:
    """Build a degenerate design where z is rank-deficient and verify the
    ridge term keeps the system solvable and the recovered W @ v ~= e_a."""
    d = _LATENT_DIM
    rng = np.random.default_rng(123)
    n_per = 6
    base_grid = np.array([
        [g, dd, y]
        for g, dd, y in itertools.product([-1.0, 0.0, 1.0], [0.5, 1.5], [-0.2, 0.0, 0.2])
    ])
    # Replicate each grid point n_per times with tiny noise on a *single* free
    # latent dim; the other dims are exact duplicates of two free axes so the
    # Gram matrix is severely rank-deficient.
    latents = np.zeros((base_grid.shape[0] * n_per, d), dtype=np.float64)
    labels = np.zeros((base_grid.shape[0] * n_per, 3), dtype=np.float64)
    for i, (g, dd, y) in enumerate(base_grid):
        for k in range(n_per):
            row = np.zeros(d)
            row[0] = 2.0 * g
            row[1] = 3.0 * dd
            row[2] = 5.0 * y
            # Make latent dims 3..d-1 exact copies of dims 0..2 cycled.
            for j in range(3, d):
                row[j] = row[j % 3]
            row += rng.normal(scale=1e-6, size=d)
            latents[i * n_per + k] = row
            labels[i * n_per + k] = [g, dd, y]

    vecs = construct_concept_vectors_jacobian(latents, labels, ridge_alpha=1e-3)
    for axis in ("G", "D", "Y"):
        assert vecs[axis].shape == (d,)
        assert np.all(np.isfinite(vecs[axis])), f"axis {axis}: non-finite output"

    # Verify the recovered (ridge-regularised) W maps each concept vector
    # to an approximately one-hot output: the diagonal entry should clearly
    # dominate the off-diagonal entries even though the ridge shrinks it
    # below 1.0. This confirms axis orthogonality survives rank deficiency.
    z_mu = latents.mean(axis=0, keepdims=True)
    c_mu = labels.mean(axis=0, keepdims=True)
    z_c = latents - z_mu
    c_c = labels - c_mu
    gram_z = z_c.T @ z_c + 1e-3 * np.eye(d)
    w_t = np.linalg.solve(gram_z, z_c.T @ c_c)
    w_mat = w_t.T
    for axis_idx, axis in enumerate(("G", "D", "Y")):
        out = w_mat @ vecs[axis]
        diag = abs(out[axis_idx])
        off = max(abs(out[j]) for j in range(3) if j != axis_idx)
        assert diag > 0.5, f"axis {axis}: diagonal {diag} too small (W @ v = {out})"
        assert off < 1e-2, f"axis {axis}: off-diagonal {off} too large (W @ v = {out})"


# ---------------------------------------------------------------------------
# 8. Averaging respects ``match_tol`` for floating-point labels.
# ---------------------------------------------------------------------------
def test_averaging_match_tol_handles_float_imprecision() -> None:
    # Two pairs that should match within tol on the off-axes:
    # pair 1: differ in G, off-axes (D, Y) at 0.10 vs 0.10001
    # pair 2: differ in D, off-axes (G, Y) at 0.2 vs 0.20001
    latents = np.array([
        [2.0 * (-1.0), 3.0 * 0.10, 5.0 * 0.10000, 0, 0, 0, 0, 0],
        [2.0 * (+1.0), 3.0 * 0.10001, 5.0 * 0.10001, 0, 0, 0, 0, 0],
        [2.0 * 0.2, 3.0 * 0.5, 5.0 * 0.3, 0, 0, 0, 0, 0],
        [2.0 * 0.20001, 3.0 * 1.5, 5.0 * 0.30001, 0, 0, 0, 0, 0],
    ], dtype=np.float64)
    labels = np.array([
        [-1.0, 0.10, 0.10000],
        [+1.0, 0.10001, 0.10001],
        [0.2, 0.5, 0.3],
        [0.20001, 1.5, 0.30001],
    ])

    # With strict tolerance, neither pair matches.
    vecs_strict = construct_concept_vectors_averaging(
        latents, labels, match_tol=(1e-8, 1e-8, 1e-8), min_pairs=1,
    )
    assert vecs_strict["G"] is None
    assert vecs_strict["D"] is None

    # With loose tolerance, the G and D pairs are recovered.
    vecs_loose = construct_concept_vectors_averaging(
        latents, labels, match_tol=(1e-3, 1e-3, 1e-3), min_pairs=1,
    )
    assert vecs_loose["G"] is not None
    assert np.isclose(vecs_loose["G"][0], 2.0, atol=1e-3)
    assert vecs_loose["D"] is not None
    assert np.isclose(vecs_loose["D"][1], 3.0, atol=1e-3)


# ---------------------------------------------------------------------------
# 9. Returned vectors are 1-D with shape ``(d,)`` — not ``(1, d)`` or ``(d, 1)``.
# ---------------------------------------------------------------------------
def test_returned_vectors_have_shape_d() -> None:
    latents, labels = _make_linear_dataset(d=16)
    vecs_avg = construct_concept_vectors_averaging(latents, labels)
    vecs_jac = construct_concept_vectors_jacobian(latents, labels)
    for name in ("G", "D", "Y"):
        assert vecs_avg[name].shape == (16,)
        assert vecs_avg[name].ndim == 1
        assert vecs_jac[name].shape == (16,)
        assert vecs_jac[name].ndim == 1


# ---------------------------------------------------------------------------
# 10. consistency_check returns exactly one entry per axis.
# ---------------------------------------------------------------------------
def test_consistency_check_one_entry_per_axis() -> None:
    latents, labels = _make_linear_dataset()
    vecs_avg = construct_concept_vectors_averaging(latents, labels)
    vecs_jac = construct_concept_vectors_jacobian(latents, labels)
    report = consistency_check(vecs_avg, vecs_jac)
    assert set(report.keys()) == {"G", "D", "Y"}
    for axis, entry in report.items():
        assert "cosine_similarity" in entry
        assert "norm_ratio" in entry
        assert entry["axis_name"] == axis


# ---------------------------------------------------------------------------
# Bonus: linear_extrapolate skips ``None`` axes gracefully.
# ---------------------------------------------------------------------------
def test_linear_extrapolate_skips_none_axes() -> None:
    d = 8
    z0 = np.zeros(d)
    c0 = np.zeros(3)
    c_target = np.array([1.0, 2.0, 3.0])
    vecs = {
        "G": np.array([2.0] + [0.0] * (d - 1)),
        "D": None,
        "Y": np.array([0.0, 0.0, 5.0] + [0.0] * (d - 3)),
    }
    z_pred = linear_extrapolate(z0, c0, c_target, vecs)
    expected = np.zeros(d)
    expected[0] = 2.0      # G: 1 * 2
    expected[2] = 15.0     # Y: 3 * 5
    assert np.allclose(z_pred, expected)
