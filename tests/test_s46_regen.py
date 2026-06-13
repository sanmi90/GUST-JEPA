"""Tests for scripts/session28/s46_regen.py (the v2.1 Section 4.6 regeneration).

CPU-only, fast. Synthetic tests pin the numerical kernels (orbit Mahalanobis return
distance on a known cloud, KRR-RBF probe R^2 against sklearn, scale-band correlation
on a known two-scale signal, case-clustering by case_id, NPZ key-convention tolerance).
A handful of skipif-real tests exercise the end-to-end paths only when the v2.1 latents,
decode artifacts, and cache are present.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "session28"))

import closure_matrix as cm  # noqa: E402
import physics_prep as pp  # noqa: E402
import s46_regen as s46  # noqa: E402

REAL_LATENTS = (s46.LATENTS_ROOT / s46.PRED_FAMILY / "test_b.npz").exists()
REAL_DECODE = (s46.DECODE_ROOT / "tf_noc" / "recon_test_b.npz").exists()
REAL_CACHE = (
    Path(s46.os.environ.get("VORTEX_JEPA_CACHE", "")).joinpath("v2p1").exists()
    if s46.os.environ.get("VORTEX_JEPA_CACHE")
    else (Path.home() / "PREVENT" / "data" / "processed" / "vortex-jepa" / "v2p1").exists()
)


# --------------------------------------------------------------------------- Q2 metric
def test_euclid_return_distance_on_known_cloud():
    """Euclidean return distance: an orbit point returns ~0; a point offset by k diameters
    along an axis returns ~k in orbit-diameter units."""
    rng = np.random.default_rng(0)
    orbit = rng.standard_normal((400, 3))
    geom = s46._orbit_geometry(orbit)
    # A point inside the cloud returns a small distance (< 0.5 diameters).
    d_in = s46._min_dist_to_cloud(orbit[:5], orbit) / geom.euclid_diameter
    assert np.all(d_in < 0.5)
    # A point far along axis 0: min Euclidean distance ~ offset - (cloud half-width).
    far = geom.mean + np.array([50.0, 0.0, 0.0])
    d_far = s46._min_dist_to_cloud(far[None, :], orbit)[0] / geom.euclid_diameter
    assert d_far == pytest.approx(50.0 / geom.euclid_diameter, rel=0.3)


def test_euclid_diameter_scales_with_cloud():
    """The raw Euclidean orbit diameter scales linearly with the cloud scale, so a fixed
    offset measured in orbit-diameter units shrinks as the cloud grows."""
    rng = np.random.default_rng(1)
    base = rng.standard_normal((300, 2))
    g1 = s46._orbit_geometry(base)
    g2 = s46._orbit_geometry(base * 7.0)
    assert g2.euclid_diameter == pytest.approx(7.0 * g1.euclid_diameter, rel=0.05)


def test_orbit_effective_dim_reports_rank():
    """A rank-1 (line) cloud has effective dim ~1; an isotropic cloud has effective dim
    ~ambient. This is the diagnostic that flags the Mahalanobis ill-conditioning."""
    rng = np.random.default_rng(2)
    direction = np.array([1.0, 2.0, -1.0])
    line = rng.standard_normal((200, 1)) * direction[None, :]
    iso = rng.standard_normal((200, 3))
    assert s46._orbit_geometry(line).effective_dim < 1.5
    assert s46._orbit_geometry(iso).effective_dim > 2.5


def test_anisotropic_orbit_mahalanobis_direction():
    """A cloud stretched along x: a point offset along the THIN (y) axis returns a larger
    Mahalanobis distance than the same raw offset along the WIDE (x) axis."""
    rng = np.random.default_rng(3)
    cloud = rng.standard_normal((500, 2)) * np.array([10.0, 1.0])
    geom = s46._orbit_geometry(cloud)
    orbit_w = (cloud - geom.mean) @ geom.whiten
    off = 5.0
    d_x = s46._min_maha_to_cloud((geom.mean + [off, 0])[None, :], geom, orbit_w)[0]
    d_y = s46._min_maha_to_cloud((geom.mean + [0, off])[None, :], geom, orbit_w)[0]
    assert d_y > d_x


# ----------------------------------------------------------------- Q1 probe vs sklearn
def test_krr_probe_matches_sklearn_oof():
    """The CV-ensemble test prediction equals the mean of per-fold sklearn KRR
    predictions (closure_matrix.fit_krr_rbf is the same estimator under the hood)."""
    from sklearn.kernel_ridge import KernelRidge

    rng = np.random.default_rng(3)
    n_cases, per = 10, 6
    case_ids = np.array([f"c{c}" for c in range(n_cases) for _ in range(per)])
    z_tr = rng.standard_normal((n_cases * per, 5))
    # smooth target so a probe can recover signal
    y_tr = np.sin(z_tr[:, 0]) + 0.5 * z_tr[:, 1]
    z_te = rng.standard_normal((20, 5))

    got = s46.cv_oof_test_predictions(z_tr, y_tr, case_ids, z_te, n_folds=5, cv_seed=0)

    fold_id = cm.make_case_folds(case_ids, n_folds=5, seed=0)
    preds = []
    for k in sorted(set(fold_id.tolist())):
        in_fit = fold_id != k
        mu = z_tr[in_fit].mean(0)
        sg = z_tr[in_fit].std(0).clip(min=1e-9)
        model = KernelRidge(alpha=1.0, kernel="rbf", gamma=0.01)
        model.fit((z_tr[in_fit] - mu) / sg, y_tr[in_fit])
        preds.append(model.predict((z_te - mu) / sg))
    expected = np.mean(preds, axis=0)
    np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-9)


def test_probe_r2_recovers_linear_signal():
    """On a held-out split with a clean linear-in-latent target, the KRR probe R^2 is
    high (sanity that the probe + R^2 plumbing is wired correctly)."""
    rng = np.random.default_rng(4)
    n_cases, per = 12, 6
    case_ids = np.array([f"c{c}" for c in range(n_cases) for _ in range(per)])
    z_tr = rng.standard_normal((n_cases * per, 4))
    w = np.array([1.0, -0.5, 0.3, 0.0])
    y_tr = z_tr @ w
    z_te = rng.standard_normal((30, 4))
    y_te = z_te @ w
    pred = s46.cv_oof_test_predictions(z_tr, y_tr, case_ids, z_te, n_folds=5, cv_seed=0)
    r2, lo, hi = s46.case_cluster_ci_on_r2(
        pred, y_te, np.array([f"t{i // 3}" for i in range(30)]), n_boot_case=200, boot_seed=0
    )
    assert r2 > 0.7
    assert lo <= r2 <= hi


# ---------------------------------------------------------------- Q3 scale-band corr
def test_pearson_known_signal():
    """Pearson of identical series is 1, of negated series is -1, of constant is NaN."""
    a = np.arange(10.0)
    assert s46.pearson(a, a) == pytest.approx(1.0)
    assert s46.pearson(a, -a) == pytest.approx(-1.0)
    assert np.isnan(s46.pearson(a, np.ones(10)))


def test_scale_band_separates_two_scales():
    """A two-scale field: large-scale enstrophy of a Gaussian-filtered field tracks the
    LARGE-scale component, so two fields sharing the large component but differing in
    fine texture correlate highly after large-scale filtering."""
    rng = np.random.default_rng(5)
    T, H, W = 8, 192, 96
    xc = np.linspace(0, 1, H)[:, None]
    yc = np.linspace(0, 1, W)[None, :]
    mask = pp.wake_mask()
    sigma_px = 0.05 / pp.DX
    # large-scale modulated blob, shared; fine texture, independent per field
    large = np.array(
        [
            np.sin(2 * np.pi * (xc + 0.05 * t)) * np.cos(2 * np.pi * yc) * (1 + 0.3 * t)
            for t in range(T)
        ]
    )
    fine_a = rng.standard_normal((T, H, W)) * 0.01
    fine_b = rng.standard_normal((T, H, W)) * 0.01
    field_a = (large + fine_a) * 50.0
    field_b = (large + fine_b) * 50.0
    e_a = pp.large_scale_wake_enstrophy(field_a, sigma_px, mask)
    e_b = pp.large_scale_wake_enstrophy(field_b, sigma_px, mask)
    assert s46.pearson(e_a, e_b) > 0.95
    # a field with a DIFFERENT large-scale time modulation tracks worse
    field_c = (large[::-1] + fine_b) * 50.0
    assert s46.pearson(e_a, pp.large_scale_wake_enstrophy(field_c, sigma_px, mask)) < s46.pearson(
        e_a, e_b
    )


# -------------------------------------------------------------- case-clustering / keys
def test_case_cluster_groups_by_case_id():
    """case_cluster_bootstrap groups by case_id: per-case means are the within-case mean
    of the values, matching stats_lib.case_means (the clustering unit is the case)."""
    vals = np.array([1.0, 3.0, 10.0, 12.0])
    cids = np.array(["a", "a", "b", "b"])
    cb = stats_cluster(vals, cids)
    assert cb["per_case_mean_delta"]["a"] == pytest.approx(2.0)
    assert cb["per_case_mean_delta"]["b"] == pytest.approx(11.0)
    assert cb["n_cases"] == 2


def test_npz_key_convention_tolerance(tmp_path):
    """load_family_split tolerates singular (case_id/encounter_index) and plural keys."""
    n, T, d = 3, 120, 4
    common = dict(
        z_full=np.zeros((n, T, d), np.float32),
        G=np.zeros(n, np.float32),
        D=np.zeros(n, np.float32),
        Y=np.zeros(n, np.float32),
        impact_frame=np.full(n, 40, np.int32),
    )
    # singular convention
    (tmp_path / "fam_singular").mkdir(parents=True)
    np.savez(
        tmp_path / "fam_singular" / "test_b.npz",
        case_id=np.array(["x", "x", "y"]),
        encounter_index=np.array([0, 1, 0], np.int32),
        **common,
    )
    old_root = s46.LATENTS_ROOT
    try:
        s46.LATENTS_ROOT = tmp_path
        data = s46.load_family_split("fam_singular", "test_b")
        assert data["case_ids"].tolist() == ["x", "x", "y"]
        assert data["encounter_indices"].tolist() == [0, 1, 0]
    finally:
        s46.LATENTS_ROOT = old_root


def test_impact_frame_latent_selection():
    """impact_frame_latents reads z at each encounter's own impact frame."""
    n, T, d = 2, 120, 3
    z = np.zeros((n, T, d), np.float64)
    z[0, 40] = [1, 2, 3]
    z[1, 55] = [4, 5, 6]
    data = {"z_full": z, "impact_frame": np.array([40, 55])}
    zi = s46.impact_frame_latents(data)
    np.testing.assert_array_equal(zi, [[1, 2, 3], [4, 5, 6]])


def test_late_horizon_caps_inside_cache():
    """late_horizon_latents picks the largest H <= cap with impact + H < T."""
    n, T, d = 2, 120, 2
    z = np.zeros((n, T, d), np.float64)
    data = {"z_full": z, "impact_frame": np.array([40, 100])}
    _, h = s46.late_horizon_latents(data, horizon_cap=24)
    # max impact is 100, so impact+H < 120 -> H <= 19; cap is 24 so H = 19.
    assert h == 19


def stats_cluster(vals, cids):
    """Local shim so the test reads cleanly."""
    import stats_lib

    return stats_lib.case_cluster_bootstrap(vals, cids, n_boot=50, rng=np.random.default_rng(0))


# ----------------------------------------------------------------- skipif-real, e2e
@pytest.mark.skipif(not REAL_LATENTS, reason="v2.1 latents not present")
def test_real_q1_runs():
    cfg = s46.Config(cache_root="unused", n_boot_case=200)
    q1 = s46.quantity1_parameter_probe(cfg)
    for p in s46.PARAMS:
        assert "r2_test_b" in q1["per_param"][p]
    assert q1["weakest_param"] in s46.PARAMS


@pytest.mark.skipif(not REAL_LATENTS, reason="v2.1 latents not present")
def test_real_q2_runs():
    cfg = s46.Config(cache_root="unused", n_boot_case=200)
    q2 = s46.quantity2_limit_cycle_return(cfg)
    assert q2["n_test_b_enc"] == 42
    assert np.isfinite(q2["pred_median_return"])
    assert np.isfinite(q2["recon_median_return"])


@pytest.mark.skipif(
    not (REAL_DECODE and REAL_CACHE), reason="v2.1 decode artifacts or cache not present"
)
def test_real_q3_runs():
    cache = s46.os.environ.get("VORTEX_JEPA_CACHE") or str(
        Path.home() / "PREVENT" / "data" / "processed" / "vortex-jepa"
    )
    cfg = s46.Config(cache_root=cache, n_boot_case=200)
    q3 = s46.quantity3_scale_decode(cfg)
    for fam in s46.DECODE_FAMILIES:
        assert -1.0 <= q3["per_family"][fam]["median_corr"] <= 1.0
