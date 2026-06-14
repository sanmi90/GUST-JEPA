"""Minimal tests for SESSION29 Track H (manifold-departure diagnostics).

Synthetic contract: a point FAR from a Gaussian blob has a larger
kNN-distance-to-manifold AND a larger local-PCA orthogonal reconstruction
residual than a point sitting INSIDE the blob. Both diagnostics must be
metric-independent (no covariance inverse) and normalised by the cloud scale.
Also covers the cloud-scale definition and the provenance schema.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "session29"))
import _s29_common as cm  # noqa: E402
import manifold_diagnostics as md  # noqa: E402


def _blob(seed: int, n: int = 400, d: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d))


def test_knn_distance_far_point_larger():
    cloud = _blob(0)
    scale = md.cloud_scale(cloud)
    inside = np.zeros((1, cloud.shape[1]))  # at the blob centre
    far = np.full((1, cloud.shape[1]), 50.0)  # far outside
    d_in = md.knn_distance(inside, cloud, k=10, scale=scale)[0]
    d_far = md.knn_distance(far, cloud, k=10, scale=scale)[0]
    assert d_far > d_in
    assert d_far > 5.0  # far point is many cloud-scales away


def test_local_pca_residual_far_point_larger():
    cloud = _blob(1)
    scale = md.cloud_scale(cloud)
    inside = np.zeros((1, cloud.shape[1]))
    far = np.full((1, cloud.shape[1]), 50.0)
    r_in = md.local_pca_residual(inside, cloud, k=20, n_components=3, scale=scale)[0]
    r_far = md.local_pca_residual(far, cloud, k=20, n_components=3, scale=scale)[0]
    assert r_far > r_in


def test_local_pca_residual_on_subspace_is_small():
    # A point lying in the 2D plane spanned by the cloud should have tiny
    # orthogonal residual; an off-plane point should have a large one.
    rng = np.random.default_rng(2)
    n, d = 300, 6
    coords = rng.normal(size=(n, 2)) * np.array([5.0, 3.0])
    basis = np.zeros((2, d))
    basis[0, 0] = 1.0
    basis[1, 1] = 1.0
    cloud = coords @ basis  # lives in the (x0, x1) plane
    scale = md.cloud_scale(cloud)
    on_plane = np.array([[2.0, -1.0, 0.0, 0.0, 0.0, 0.0]])
    off_plane = np.array([[2.0, -1.0, 0.0, 0.0, 0.0, 8.0]])
    r_on = md.local_pca_residual(on_plane, cloud, k=30, n_components=2, scale=scale)[0]
    r_off = md.local_pca_residual(off_plane, cloud, k=30, n_components=2, scale=scale)[0]
    assert r_on < 0.5
    assert r_off > r_on
    # cloud-scale ~= sqrt(25 + 9) ~= 5.83, so an 8-unit off-plane shift is
    # ~1.37 cloud-scales of orthogonal residual: clearly off the manifold.
    assert r_off > 1.0


def test_cloud_scale_positive_and_scales():
    cloud = _blob(3)
    s1 = md.cloud_scale(cloud)
    s2 = md.cloud_scale(cloud * 4.0)
    assert s1 > 0
    assert abs(s2 / s1 - 4.0) < 1e-6  # scale is homogeneous of degree 1


def test_diagnostics_are_metric_independent_of_rotation():
    # Rotating the whole problem (cloud + query) must not change either
    # diagnostic: they depend only on Euclidean geometry.
    cloud = _blob(4)
    scale = md.cloud_scale(cloud)
    q = np.full((1, cloud.shape[1]), 3.0)
    rng = np.random.default_rng(5)
    a = rng.normal(size=(cloud.shape[1], cloud.shape[1]))
    qmat, _ = np.linalg.qr(a)  # orthonormal rotation
    cloud_r = cloud @ qmat
    q_r = q @ qmat
    scale_r = md.cloud_scale(cloud_r)
    assert abs(scale_r - scale) < 1e-9
    knn = md.knn_distance(q, cloud, k=10, scale=scale)[0]
    knn_r = md.knn_distance(q_r, cloud_r, k=10, scale=scale_r)[0]
    assert abs(knn - knn_r) < 1e-6
    pca = md.local_pca_residual(q, cloud, k=20, n_components=3, scale=scale)[0]
    pca_r = md.local_pca_residual(q_r, cloud_r, k=20, n_components=3, scale=scale_r)[0]
    assert abs(pca - pca_r) < 1e-6


def test_provenance_schema():
    prov = cm.provenance([], seed=7)
    for key in ("git_sha", "command", "utc", "versions", "seed", "split"):
        assert key in prov
    assert prov["split"] == "v2p1"
