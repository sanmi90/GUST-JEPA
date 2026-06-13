"""CPU synthetic tests for the C-E1 drift-reconciliation primitives.

These tests pin the M6 MECHANISM itself as a property test (a low-variance-axis
perturbation gives small l2 but large Mahalanobis; a high-variance-axis
perturbation gives the reverse), check that the regularised covariance inverse
is stable on a rank-deficient sample, that the departure-spectrum projection
conserves total departure energy, and that case-clustering groups by case.

Run: timeout 120 .venv/bin/python -m pytest tests/test_drift_ce1.py -q
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "drift_ce1", REPO / "scripts" / "session28" / "drift_ce1.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ce1 = _load_module()


# ---------------------------------------------------------------------------
# the M6 mechanism, as a property test
# ---------------------------------------------------------------------------


def test_m6_mechanism_low_vs_high_variance_axis():
    """On an anisotropic Gaussian cloud, a perturbation along a LOW-variance axis
    has small l2 but large Mahalanobis; along a HIGH-variance axis the reverse.

    This is exactly the M6 reconciliation: the reconstructive rollout exits along
    low-encoded-variance directions (small l2, huge Mahalanobis); the predictive
    rollout wanders within high-variance directions (larger l2, modest
    Mahalanobis).
    """
    rng = np.random.default_rng(0)
    d = 8
    stds = np.array([10.0, 8.0, 6.0, 4.0, 2.0, 1.0, 0.3, 0.1])  # high -> low variance
    n = 5000
    x = rng.standard_normal((n, d)) * stds[None, :]
    mean, sigma, _ = ce1.shrunk_covariance(x, eps=1e-8)
    sigma_inv = np.linalg.inv(sigma)

    base = mean.copy()
    step = 1.0  # SAME Euclidean step length along each axis

    # high-variance axis (axis 0, std 10): l2 = step, Mahalanobis ~ step / 10
    z_high = (base + step * np.eye(d)[0])[None, :]
    l2_high = np.linalg.norm(z_high - mean)
    dm_high = ce1.mahalanobis(z_high, mean, sigma_inv)[0]

    # low-variance axis (axis -1, std 0.1): l2 = step, Mahalanobis ~ step / 0.1
    z_low = (base + step * np.eye(d)[-1])[None, :]
    l2_low = np.linalg.norm(z_low - mean)
    dm_low = ce1.mahalanobis(z_low, mean, sigma_inv)[0]

    # equal l2 step by construction
    assert np.isclose(l2_high, l2_low, atol=1e-6)
    # low-variance axis blows up Mahalanobis; high-variance axis keeps it small
    assert dm_low > dm_high
    assert dm_low > 20.0 * dm_high  # ~ (10 / 0.1) order of magnitude separation


def test_departure_along_low_variance_small_l2_huge_maha():
    """Direct AE-vs-JEPA analogue: an AE-like departure (low-variance directions)
    yields small rel-l2 but a Mahalanobis RATIO > 1, while a JEPA-like departure
    (high-variance directions) yields larger rel-l2 but ratio of order 1."""
    rng = np.random.default_rng(1)
    d = 6
    stds = np.array([6.0, 5.0, 4.0, 3.0, 0.5, 0.2])
    train = rng.standard_normal((4000, d)) * stds[None, :]
    mean, sigma, _ = ce1.shrunk_covariance(train, eps=1e-8)
    sinv = np.linalg.inv(sigma)

    # encoded held-out point near the cloud
    z_enc = mean + np.array([3.0, 2.0, 1.0, 1.0, 0.1, 0.05])
    dm_enc = ce1.mahalanobis(z_enc[None, :], mean, sinv)[0]

    # AE-like: depart along the two lowest-variance axes (4, 5)
    delta_ae = np.zeros(d)
    delta_ae[4] = 0.3
    delta_ae[5] = 0.3
    z_ae = z_enc + delta_ae
    rel_ae = np.linalg.norm(delta_ae) / np.linalg.norm(z_enc)
    dm_ae = ce1.mahalanobis(z_ae[None, :], mean, sinv)[0]

    # JEPA-like: depart along the two highest-variance axes (0, 1), bigger step
    delta_jepa = np.zeros(d)
    delta_jepa[0] = 2.0
    delta_jepa[1] = 2.0
    z_jepa = z_enc + delta_jepa
    rel_jepa = np.linalg.norm(delta_jepa) / np.linalg.norm(z_enc)
    dm_jepa = ce1.mahalanobis(z_jepa[None, :], mean, sinv)[0]

    # rel-l2 ordering: AE smaller (the Figure-5 paradox)
    assert rel_ae < rel_jepa
    # Mahalanobis ordering: AE ratio > 1 and larger than JEPA ratio (the Table-8 flip)
    assert (dm_ae / dm_enc) > 1.0
    assert (dm_ae / dm_enc) > (dm_jepa / dm_enc)


# ---------------------------------------------------------------------------
# covariance-inverse regularisation stability
# ---------------------------------------------------------------------------


def test_shrinkage_inverse_stable_on_rank_deficient_sample():
    """n < d sample (rank-deficient) yields a finite, SPD, invertible covariance.

    Without shrinkage the sample covariance is singular (rank <= n - 1); the
    Ledoit-Wolf shrinkage + diagonal floor must produce a well-conditioned SPD
    matrix whose inverse is finite and whose Mahalanobis distances are finite.
    """
    rng = np.random.default_rng(2)
    d = 32
    n = 8  # n << d: raw sample covariance is rank-deficient
    x = rng.standard_normal((n, d)) * 3.0
    mean, sigma, lam = ce1.shrunk_covariance(x)
    assert 0.0 <= lam <= 1.0
    # SPD: all eigenvalues strictly positive
    evals = np.linalg.eigvalsh(sigma)
    assert evals.min() > 0.0
    sinv = np.linalg.inv(sigma)
    assert np.all(np.isfinite(sinv))
    # round-trip
    assert np.allclose(sigma @ sinv, np.eye(d), atol=1e-6)
    # Mahalanobis of the sample points is finite and non-negative
    dm = ce1.mahalanobis(x, mean, sinv)
    assert np.all(np.isfinite(dm))
    assert np.all(dm >= 0.0)


def test_shrinkage_lambda_low_when_well_sampled():
    """When n >> d, shrinkage intensity should be small (we trust the sample)."""
    rng = np.random.default_rng(3)
    d = 4
    x = rng.standard_normal((20000, d)) * np.array([5.0, 3.0, 2.0, 1.0])[None, :]
    _, _, lam = ce1.shrunk_covariance(x)
    assert lam < 0.2


# ---------------------------------------------------------------------------
# departure-spectrum energy conservation
# ---------------------------------------------------------------------------


def test_departure_spectrum_conserves_total_energy():
    """The eigenbasis projection partitions ||delta||^2 exactly (orthonormal)."""
    rng = np.random.default_rng(4)
    d = 10
    n = 30
    # build an orthonormal eigenbasis from a random SPD covariance
    a = rng.standard_normal((d, d))
    cov = a @ a.T + np.eye(d)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evecs = evecs[:, order]
    delta = rng.standard_normal((n, d)) * 2.0

    spec = ce1.departure_spectrum(delta, evecs, k=3)
    # per-direction fractions sum to 1
    assert np.isclose(spec["per_dir_frac"].sum(), 1.0, atol=1e-10)
    # total energy equals sum of squared norms of delta
    assert np.isclose(spec["total_energy"], np.sum(delta**2), atol=1e-8)
    # per-encounter energy equals row-wise squared norm (basis orthonormal)
    assert np.allclose(spec["per_enc_energy"], np.sum(delta**2, axis=1), atol=1e-8)
    # top_k + bottom_k <= 1 and both in [0, 1]
    assert 0.0 <= spec["top_k_frac"] <= 1.0
    assert 0.0 <= spec["bottom_k_frac"] <= 1.0
    assert spec["top_k_frac"] + spec["bottom_k_frac"] <= 1.0 + 1e-9


def test_whitened_spectrum_blows_up_in_near_null_direction():
    """The WHITENED spectrum (the M6 mechanism): a small departure that lands in a
    near-null eigen-direction carries almost no RAW energy there but almost ALL of
    the squared-Mahalanobis distance there.

    This is the exact reconciliation: raw bottom-k energy fraction stays tiny
    while the whitened bottom-k fraction approaches one.
    """
    d = 6
    evecs = np.eye(d)
    # eigenvalues high -> low; the last is near-null (the AE collapse)
    evals = np.array([10.0, 8.0, 6.0, 4.0, 2.0, 1e-6])
    n = 8
    # departure: big component in the TOP direction, tiny in the near-null one
    delta = np.zeros((n, d))
    delta[:, 0] = 3.0  # large raw energy in top dir
    delta[:, -1] = 0.01  # tiny raw energy in near-null dir
    spec = ce1.departure_spectrum(delta, evecs, k=2, eigvals=evals)
    # RAW energy: dominated by the top direction; bottom-2 fraction is tiny
    assert spec["bottom_k_frac"] < 0.01
    assert spec["top_k_frac"] > 0.99
    # WHITENED energy: the near-null direction (0.01^2 / 1e-6 = 100) dwarfs the
    # top direction (3^2 / 10 = 0.9), so the whitened bottom-k fraction is ~1
    assert spec["whitened_bottom_k_frac"] > 0.98
    assert spec["whitened_top_k_frac"] < 0.02


def test_whitened_spectrum_fractions_sum_to_one():
    """Whitened per-direction fractions partition the squared Mahalanobis distance."""
    rng = np.random.default_rng(7)
    d = 8
    evecs = np.eye(d)
    evals = np.linspace(5.0, 0.1, d)
    delta = rng.standard_normal((15, d))
    spec = ce1.departure_spectrum(delta, evecs, k=3, eigvals=evals)
    assert np.isclose(spec["whitened_per_dir_frac"].sum(), 1.0, atol=1e-10)


def test_departure_spectrum_concentrates_on_injected_axis():
    """A departure injected purely along eigen-direction j puts all its energy in
    that direction's fraction (and none elsewhere)."""
    d = 6
    evecs = np.eye(d)  # canonical orthonormal basis ordered high->low
    n = 12
    delta = np.zeros((n, d))
    delta[:, 1] = 1.5  # all energy on direction index 1
    spec = ce1.departure_spectrum(delta, evecs, k=2)
    assert np.isclose(spec["per_dir_frac"][1], 1.0, atol=1e-12)
    # direction 1 is within the top-2 -> top_k_frac == 1, bottom_k_frac == 0
    assert np.isclose(spec["top_k_frac"], 1.0, atol=1e-12)
    assert np.isclose(spec["bottom_k_frac"], 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# case-clustering groups by case
# ---------------------------------------------------------------------------


def test_paired_family_diff_clusters_by_case():
    """case_means inside paired_family_diff must group encounters by their case,
    not by encounter index (the case-clustered bootstrap depends on it)."""
    rng = np.random.default_rng(5)
    # 3 cases, uneven encounter counts
    case_ids = np.array(["caseA", "caseA", "caseA", "caseB", "caseB", "caseC"])
    a = np.array([0.1, 0.2, 0.3, 1.0, 1.2, 5.0])
    b = np.zeros_like(a)
    out = ce1.paired_family_diff(a, b, case_ids, rng)
    # per-case means come straight from stats_lib.case_means on delta = a - b
    uc, cmd = ce1.stats_lib.case_means(a - b, case_ids)
    assert list(uc) == ["caseA", "caseB", "caseC"]
    assert np.isclose(cmd[0], np.mean([0.1, 0.2, 0.3]))
    assert np.isclose(cmd[1], np.mean([1.0, 1.2]))
    assert np.isclose(cmd[2], 5.0)
    # case-level paired stats see exactly 3 cases (clustered), not 6 encounters
    assert out["case_paired"]["n_cases"] == 3
    assert out["n_enc"] == 6


def test_paired_family_diff_sign_test_direction():
    """sign_p_one_sided tests a < b; when a is uniformly below b, p should be tiny."""
    rng = np.random.default_rng(6)
    case_ids = np.array(["c1"] * 5 + ["c2"] * 5)
    a = np.full(10, 0.2)
    b = np.full(10, 0.9)  # a strictly < b in every encounter
    out = ce1.paired_family_diff(a, b, case_ids, rng)
    assert out["sign_k_a_less_b"] == 10
    assert out["sign_n_eff"] == 10
    assert out["sign_p_one_sided_a_less_b"] < 0.01
