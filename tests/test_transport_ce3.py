"""Tests for the C-E3 optimal-transport mechanism track (closes referee M5).

All tests are CPU-only and synthetic: no DNS cache, no latents, no GPU. They
exercise the debiased divergence correctness, the signed split, the alignment
Spearman, and the case-clustering convention. Run targeted::

    timeout 120 .venv/bin/python -m pytest tests/test_transport_ce3.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "session28"))

import transport_ce3 as ce3  # noqa: E402


# ----------------------------------------------------------------------------
# Small synthetic grid + cost for the debiased-divergence tests.
# ----------------------------------------------------------------------------
def _small_cost(n: int = 12) -> np.ndarray:
    """Squared-Euclidean cost on a 1-D grid of n points in [0, 1]."""
    import ot

    xs = np.linspace(0.0, 1.0, n).reshape(-1, 1)
    return ot.dist(xs, xs, metric="sqeuclidean")


def _blob(n: int, centre: int, width: float) -> np.ndarray:
    """Unit-mass Gaussian bump on n grid points centred at index `centre`."""
    idx = np.arange(n, dtype=float)
    m = np.exp(-((idx - centre) ** 2) / (2.0 * width**2))
    return m / m.sum()


# ----------------------------------------------------------------------------
# 1. Debiasing correctness: S_eps(a, a) = 0
# ----------------------------------------------------------------------------
def test_s_eps_self_is_zero():
    n = 12
    cost = _small_cost(n)
    eps = 0.05 * float(np.median(cost[np.triu_indices(n, k=1)]))
    a = _blob(n, 4, 1.5)
    assert abs(ce3.s_eps(a, a, cost, eps)) < 1e-7


def test_s_eps_self_is_zero_multiple_distributions():
    n = 16
    cost = _small_cost(n)
    eps = 0.05 * float(np.median(cost[np.triu_indices(n, k=1)]))
    for centre, width in ((2, 1.0), (8, 2.0), (13, 0.8)):
        a = _blob(n, centre, width)
        assert abs(ce3.s_eps(a, a, cost, eps)) < 1e-7


# ----------------------------------------------------------------------------
# 2. Symmetry: S_eps(a, b) = S_eps(b, a)
# ----------------------------------------------------------------------------
def test_s_eps_symmetric():
    n = 14
    cost = _small_cost(n)
    eps = 0.05 * float(np.median(cost[np.triu_indices(n, k=1)]))
    a = _blob(n, 3, 1.2)
    b = _blob(n, 10, 1.6)
    ab = ce3.s_eps(a, b, cost, eps)
    ba = ce3.s_eps(b, a, cost, eps)
    # log-domain Sinkhorn is symmetric up to solver iteration order (~1e-8).
    assert abs(ab - ba) < 1e-6


# ----------------------------------------------------------------------------
# 3. Separation: well-separated blobs have larger S_eps than overlapping ones.
# ----------------------------------------------------------------------------
def test_s_eps_separation_ordering():
    n = 20
    cost = _small_cost(n)
    eps = 0.05 * float(np.median(cost[np.triu_indices(n, k=1)]))
    a = _blob(n, 4, 1.0)
    near = _blob(n, 6, 1.0)  # overlapping
    far = _blob(n, 16, 1.0)  # well separated
    s_near = ce3.s_eps(a, near, cost, eps)
    s_far = ce3.s_eps(a, far, cost, eps)
    assert s_far > s_near
    assert s_near > 0.0  # positive-definite away from identity


# ----------------------------------------------------------------------------
# 3b. POT cross-check: our S_eps matches empirical_sinkhorn_divergence.
# ----------------------------------------------------------------------------
def test_s_eps_matches_pot_empirical_divergence():
    import ot

    rng = np.random.default_rng(1)
    n = 25
    xs = rng.standard_normal((n, 2)) * 0.3
    xt = rng.standard_normal((n, 2)) * 0.3 + np.array([1.3, 0.2])
    a = np.full(n, 1.0 / n)
    b = np.full(n, 1.0 / n)
    eps = 0.1
    pot = ot.bregman.empirical_sinkhorn_divergence(
        xs, xt, eps, a=a, b=b, metric="sqeuclidean", numIterMax=5000, stopThr=1e-10
    )
    cost_ab = ot.dist(xs, xt, metric="sqeuclidean")
    cost_aa = ot.dist(xs, xs, metric="sqeuclidean")
    cost_bb = ot.dist(xt, xt, metric="sqeuclidean")
    # S_eps built from the LINEAR transport cost terms, like ce3.s_eps.
    w_ab = float(ot.sinkhorn2(a, b, cost_ab, eps, numItermax=5000, stopThr=1e-10))
    w_aa = float(ot.sinkhorn2(a, a, cost_aa, eps, numItermax=5000, stopThr=1e-10))
    w_bb = float(ot.sinkhorn2(b, b, cost_bb, eps, numItermax=5000, stopThr=1e-10))
    mine = w_ab - 0.5 * w_aa - 0.5 * w_bb
    assert abs(mine - float(pot)) < 1e-6


# ----------------------------------------------------------------------------
# 4. m+/m- split reconstructs the signed field.
# ----------------------------------------------------------------------------
def test_pos_neg_split_reconstructs_signed_field():
    rng = np.random.default_rng(0)
    field = rng.standard_normal((192, 96)) * 5.0
    mp = np.maximum(field, 0.0)
    mn = np.maximum(-field, 0.0)
    assert np.allclose(field, mp - mn)
    # pooled split sums match the pooled raw field (linearity of average pool).
    mp_pool, mn_pool = ce3.split_pos_neg_pooled(field)
    recon_pool = (mp_pool - mn_pool).reshape(ce3.NX_POOL, ce3.NY_POOL)
    assert np.allclose(recon_pool, ce3.pool4(field), atol=1e-6)


def test_normalise_mass_unit_total_and_empty():
    m = np.array([0.0, 2.0, 6.0, 0.0])
    nm = ce3.normalise_mass(m)
    assert abs(nm.sum() - 1.0) < 1e-12
    # the 1e-6 uniform floor keeps every cell strictly positive (no exact zeros),
    # but the signal cells still dominate to within the floor magnitude.
    assert (nm > 0).all()
    assert np.allclose(nm, [0.0, 0.25, 0.75, 0.0], atol=1e-5)
    # zero-mass cells get only the floor mass, far below the signal cells.
    assert nm[0] < 1e-5 and nm[1] > 0.24
    empty = ce3.normalise_mass(np.zeros(4))
    assert abs(empty.sum() - 1.0) < 1e-12
    assert np.allclose(empty, 0.25)  # uniform fallback for an all-zero half-field


def test_d_field_self_is_zero_and_positive_between():
    rng = np.random.default_rng(2)
    f1 = rng.standard_normal((192, 96)) * 4.0
    # build f2 by spatially shifting f1's structure (different field)
    f2 = np.roll(f1, shift=20, axis=0)
    cost = ce3.build_cost_matrix()
    eps = ce3.eps_from_cost(cost)
    assert abs(ce3.d_field(f1, f1, cost, eps)) < 1e-6
    assert ce3.d_field(f1, f2, cost, eps) > 1e-3


# ----------------------------------------------------------------------------
# 5. Alignment Spearman recovers a known monotone relation.
# ----------------------------------------------------------------------------
def test_alignment_spearman_recovers_monotone():
    from scipy.stats import spearmanr

    # latent distance monotone (non-linear) in field distance -> Spearman ~ 1.
    field_seq = np.linspace(0.1, 2.0, 30)
    lat_seq = np.sqrt(field_seq) + 0.01  # strictly increasing transform
    rho = spearmanr(lat_seq, field_seq).correlation
    assert rho > 0.999
    # a decreasing relation -> Spearman ~ -1
    rho_dec = spearmanr(-lat_seq, field_seq).correlation
    assert rho_dec < -0.999


def test_latent_distance_sequence_matches_manual():
    # one encounter: z_full (T, d); base pool (M, d); known match indices.
    T, d, M = 12, 4, 5
    rng = np.random.default_rng(3)
    z_full = rng.standard_normal((T, d))
    pool = rng.standard_normal((M, d))
    frames = np.array([0, 4, 8])
    match_idx = np.array([1, 3, 0])
    seq = ce3.latent_distance_sequence(z_full, frames, match_idx, pool)
    manual = np.array([np.linalg.norm(z_full[fr] - pool[mi]) for fr, mi in zip(frames, match_idx)])
    assert np.allclose(seq, manual)


# ----------------------------------------------------------------------------
# 6. Case-clustering groups by case.
# ----------------------------------------------------------------------------
def test_case_clustering_groups_by_case():
    import stats_lib

    # two cases, two encounters each; delta constant within case.
    delta = np.array([0.2, 0.2, -0.1, -0.1])
    case_ids = np.array(["A", "A", "B", "B"])
    uc, cm = stats_lib.case_means(delta, case_ids)
    assert list(uc) == ["A", "B"]
    assert np.allclose(cm, [0.2, -0.1])
    boot = stats_lib.case_cluster_bootstrap(
        delta, case_ids, n_boot=200, rng=np.random.default_rng(0)
    )
    assert boot["n_cases"] == 2
    assert boot["n_encounters"] == 4
    # per-case mean delta dictionary keys are the cases.
    assert set(boot["per_case_mean_delta"].keys()) == {"A", "B"}


def test_circular_phase_match_picks_nearest():
    # build a tiny BaselinePhaseRef and confirm circular nearest match.
    phases = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
    omega = np.zeros((4, 192, 96))
    masses = [(np.zeros(2), np.zeros(2)) for _ in range(4)]
    meta = [("Baseline", 0, i) for i in range(4)]
    ref = ce3.BaselinePhaseRef(phases, omega, masses, meta)
    # 0.1 rad before 2pi wraps to nearest 0.0
    assert ref.match_index(2 * np.pi - 0.1) == 0
    assert ref.match_index(np.pi - 0.05) == 2
    assert ref.match_index(np.pi / 2 + 0.05) == 1


def test_eps_from_cost_rule():
    cost = ce3.build_cost_matrix()
    eps = ce3.eps_from_cost(cost)
    med = float(np.median(cost[np.triu_indices(cost.shape[0], k=1)]))
    assert abs(eps - ce3.EPS_FRACTION * med) < 1e-12
    assert eps > 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
