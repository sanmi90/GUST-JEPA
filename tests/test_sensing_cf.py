"""Unit tests for scripts/session28/sensing_cf.py (master plan C-F).

CPU only. Synthetic tests for the metrics, the qDEIM placement, the case-level
CV fold construction, and the KernelRidge map wiring; the real-data smoke is
skipped when the v2p1 cache / latents are absent.

Run targeted:
    timeout 180 .venv/bin/python -m pytest tests/test_sensing_cf.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "session28"))
sys.path.insert(0, str(REPO / "scripts" / "session20"))

import sensing_cf as cf  # noqa: E402

# ---------------------------------------------------------------- metric: vw R^2


def test_variance_weighted_r2_matches_sklearn():
    """Pooled-SSE/SST R^2 == sklearn r2_score(multioutput='variance_weighted')."""
    from sklearn.metrics import r2_score

    rng = np.random.default_rng(1)
    y = rng.normal(size=(60, 5)) * np.array([1.0, 4.0, 0.2, 2.0, 0.5])
    yhat = y + rng.normal(scale=0.3, size=y.shape)
    got = cf.variance_weighted_r2(yhat, y)
    ref = r2_score(y, yhat, multioutput="variance_weighted")
    assert abs(got - ref) < 1e-10


def test_variance_weighted_r2_perfect_and_mean():
    """Perfect prediction -> 1.0; constant-at-mean prediction -> 0.0."""
    rng = np.random.default_rng(2)
    y = rng.normal(size=(40, 3))
    assert abs(cf.variance_weighted_r2(y, y) - 1.0) < 1e-12
    mean_pred = np.broadcast_to(y.mean(0), y.shape)
    assert abs(cf.variance_weighted_r2(mean_pred, y) - 0.0) < 1e-12


# ------------------------------------------------- metric: canonical correlation


def test_canonical_correlation_identity_is_one():
    """rho == 1 for an exact (rotated) copy; mean over directions is ~1."""
    rng = np.random.default_rng(3)
    T = rng.normal(size=(200, 4))
    Q, _ = np.linalg.qr(rng.normal(size=(4, 4)))  # arbitrary rotation
    P = T @ Q
    mean_rho, rhos = cf.mean_canonical_correlation(P, T, n_dir=4)
    assert mean_rho > 0.999
    assert all(r > 0.999 for r in rhos)
    assert len(rhos) == 4


def test_canonical_correlation_independent_is_low():
    """Independent signals -> small canonical correlations (finite-sample noise)."""
    rng = np.random.default_rng(4)
    P = rng.normal(size=(500, 3))
    T = rng.normal(size=(500, 3))
    mean_rho, rhos = cf.mean_canonical_correlation(P, T, n_dir=3)
    assert mean_rho < 0.25


def test_canonical_correlation_vs_sklearn_cca():
    """Leading canonical correlation matches sklearn CCA on a correlated pair."""
    from sklearn.cross_decomposition import CCA

    rng = np.random.default_rng(5)
    Z = rng.normal(size=(300, 2))
    P = np.column_stack([Z[:, 0] + 0.1 * rng.normal(size=300), rng.normal(size=300)])
    T = np.column_stack([Z[:, 0] + 0.1 * rng.normal(size=300), rng.normal(size=300)])
    _, rhos = cf.mean_canonical_correlation(P, T, n_dir=2)
    cca = CCA(n_components=1).fit(P, T)
    pc, tc = cca.transform(P, T)
    ref = abs(np.corrcoef(pc[:, 0], tc[:, 0])[0, 1])
    assert abs(rhos[0] - ref) < 0.05


def test_canonical_correlation_caps_n_dir():
    """n_dir is capped at the available rank."""
    rng = np.random.default_rng(6)
    P = rng.normal(size=(50, 3))
    T = rng.normal(size=(50, 3))
    _, rhos = cf.mean_canonical_correlation(P, T, n_dir=99)
    assert len(rhos) == 3


# ------------------------------------------------------------------ qDEIM placement


def test_qdeim_distinct_valid_and_deterministic():
    """qDEIM returns K distinct valid tap indices and is deterministic."""
    rng = np.random.default_rng(7)
    snap = rng.normal(size=(80, cf.N_PRESSURE))
    for K in (2, 4, 8, 16):
        picks = cf.selector_qdeim(snap, K)
        assert len(picks) == K
        assert len(set(picks)) == K
        assert all(0 <= p < cf.N_PRESSURE for p in picks)
        assert picks == sorted(picks)
        # determinism: identical on a second call
        assert cf.selector_qdeim(snap, K) == picks


def test_qdeim_recovers_planted_columns():
    """A rank-2 snapshot supported on two columns -> qDEIM picks those columns."""
    n = 60
    a = np.random.default_rng(8).normal(size=n)
    b = np.random.default_rng(9).normal(size=n)
    snap = np.zeros((n, cf.N_PRESSURE))
    snap[:, 17] = a
    snap[:, 140] = b
    snap += 1e-9 * np.random.default_rng(10).normal(size=snap.shape)
    picks = cf.selector_qdeim(snap, 2)
    assert set(picks) == {17, 140}


def test_uniform_selector_distinct():
    for K in (2, 4, 8, 16):
        picks = cf.selector_uniform(K)
        assert len(set(picks)) == K
        assert all(0 <= p < cf.N_PRESSURE for p in picks)


def test_tcsi_target_aware_selection():
    """TCSI prefers the tap whose window linearly explains the target."""
    rng = np.random.default_rng(11)
    n, W = 120, cf.WINDOW
    X = rng.normal(size=(n, cf.N_PRESSURE, W))
    # tap 5's window mean drives the target
    target = X[:, 5, :].mean(1) + 0.01 * rng.normal(size=n)
    ranked = cf.selector_tcsi(X, target, K=3)
    assert ranked[0] == 5  # most-informative tap selected first
    assert len(set(ranked)) == 3


# ------------------------------------------------------------ case-level CV folds


def test_case_folds_no_straddle_and_deterministic():
    """Every case lands in exactly one fold; assignment is seed-deterministic."""
    case_ids = np.array([f"case_{i // 4}" for i in range(40)])  # 10 cases x 4 enc
    folds = cf.make_case_folds(case_ids, n_folds=5, seed=0)
    # no case straddles folds
    for c in set(case_ids.tolist()):
        assert len(set(folds[case_ids == c].tolist())) == 1
    # deterministic
    assert np.array_equal(folds, cf.make_case_folds(case_ids, n_folds=5, seed=0))
    # all five folds populated for 10 cases
    assert set(folds.tolist()) == set(range(5))


# ------------------------------------------------------------ KRR map shape wiring


def test_krr_map_shapes_multioutput():
    """fit_krr/krr_pred preserve (n, K*W) -> (n, d) wiring for d>1 latents."""
    rng = np.random.default_rng(12)
    n, kw, d = 50, 8 * cf.WINDOW, 64
    X = rng.normal(size=(n, kw))
    z = rng.normal(size=(n, d))
    model = cf.fit_krr(X, z)
    pred = cf.krr_pred(model, X[:7])
    assert pred.shape == (7, d)


def test_krr_map_shapes_singleoutput():
    """A 1-D target round-trips as (n, 1)."""
    rng = np.random.default_rng(13)
    X = rng.normal(size=(40, 4 * cf.WINDOW))
    y = rng.normal(size=40)
    model = cf.fit_krr(X, y)
    pred = cf.krr_pred(model, X[:5])
    assert pred.shape == (5, 1)


def test_feats_flattening_order():
    """feats selects the right taps and flattens (n, W, K) -> (n, W*K) (W outer)."""
    X = np.arange(2 * cf.WINDOW * cf.N_PRESSURE, dtype=float).reshape(2, cf.WINDOW, cf.N_PRESSURE)
    taps = [3, 100]
    f = cf.feats(X, taps)
    assert f.shape == (2, len(taps) * cf.WINDOW)
    # reshaping back to (W, K) recovers exactly the selected taps' windows
    assert np.allclose(f[0].reshape(cf.WINDOW, len(taps)), X[0][:, taps])


# ----------------------------------------------------------- cv_r2 readout wiring


def test_cv_r2_recovers_linear_map():
    """A clean linear pressure->z map is recovered with high CV vw-R^2."""
    rng = np.random.default_rng(14)
    n_cases, per = 15, 6
    case_ids = np.array([f"c{i}" for i in range(n_cases) for _ in range(per)])
    n = n_cases * per
    X = rng.normal(size=(n, 2 * cf.WINDOW))
    A = rng.normal(size=(2 * cf.WINDOW, 8))
    z = X @ A + 0.05 * rng.normal(size=(n, 8))
    mean_r2, oof, folds = cf.cv_r2(X, z, case_ids, cf.variance_weighted_r2)
    assert mean_r2 > 0.5
    assert oof.shape == (n, 8)
    assert not np.isnan(oof).any()


# ------------------------------------------------------- key-convention tolerance


def test_family_seed_tags_resolve_to_dirs():
    """Lead-seed tags name real latents directories (when present)."""
    root = cf.LAT_ROOT
    if not root.exists():
        pytest.skip("latents root absent")
    for fam in cf.FAMILIES:
        lead = root / fam.seed_tags[0]
        if lead.exists():
            # tolerate singular/plural and val/test_a alias as closure does
            assert (lead / "train.npz").exists()


# --------------------------------------------------------------- real-data smoke

_LAT = cf.LAT_ROOT / "jepa_tf_noc_d64_s42" / "train.npz"
_DNS = cf.DNS_PATH
_CACHE = cf.CACHE
_real_present = _LAT.exists() and _DNS.exists() and _CACHE.exists()


@pytest.mark.skipif(not _real_present, reason="v2p1 latents / DNS / cache absent")
def test_real_load_family_split_shapes():
    """A real load yields aligned pressure windows, impact latents and DNS wake."""
    dns = np.load(_DNS, allow_pickle=True)
    b = cf.load_family_split("jepa_tf_noc_d64_s42", "test_b", dns)
    n, d = b.z.shape
    assert b.X.shape == (n, cf.WINDOW, cf.N_PRESSURE)
    assert b.wake.shape == (n,)
    assert b.cl.shape == (n,)
    assert b.Y.shape == (n,)
    assert b.z_perframe.shape[0] == n and b.z_perframe.shape[2] == d
    assert b.wake_perframe.shape == (n, b.z_perframe.shape[1])
    assert np.isfinite(b.X).all()
