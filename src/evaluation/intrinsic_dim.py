"""Intrinsic-dimensionality estimators for the JEPA latent manifold.

This module bundles four independent estimators of the intrinsic dimension
of a point cloud in R^D, used to support the Session 11 / 12 / 13 finding
that the JEPA latent representation of the parametric vortex-gust impact
manifold at Re=5000 has an effective rank of roughly 12. The headline
claim Session 14 wants to make is that this number agrees across four
methodologically independent estimators to within an integer or two.

The estimators
--------------
1. PCA explained-variance: the smallest number of principal components
   whose cumulative variance exceeds a threshold (e.g. 0.95). This is a
   linear-subspace dimension; it overestimates the intrinsic dim of any
   curved manifold but provides a useful upper bound.

2. Levina-Bickel MLE (Levina and Bickel, "Maximum Likelihood Estimation
   of Intrinsic Dimension", NIPS 2004, DOI 10.1162/0899766054287873).
   For each point x_i, finds the k nearest neighbours and computes a
   local MLE from the distance ratios, then aggregates via a harmonic
   mean across the point cloud.

3. Two-NN (Facco, d'Errico, Rodriguez, Laio, "Estimating the intrinsic
   dimension of datasets by a minimal neighborhood information",
   Sci. Rep. 7, 12140, 2017). Uses only the ratio of second-to-first
   nearest neighbour distances. log(mu) is exponential with rate d
   under uniform sampling on a d-manifold.

4. Isomap residual variance: builds a k-nearest-neighbour geodesic
   distance matrix, embeds into k=1..k_max dimensions, and tracks
   1 - R^2 between geodesic and embedding distances. The "elbow"
   in this curve estimates the intrinsic dim.

Caveats
-------
Both the Levina-Bickel and Two-NN estimators are known to underestimate
the intrinsic dimension when the manifold has high local curvature or
the sampling is non-uniform. They are at their most reliable on
near-flat patches and uniformly sampled balls. Estimates that are not
near integer values still rank manifolds correctly but should not be
read as exact dimensions.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

try:
    from sklearn.manifold import Isomap
    from sklearn.neighbors import NearestNeighbors

    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover - sklearn is in requirements.txt
    _HAS_SKLEARN = False


def _validate_X(X: np.ndarray, name: str = "X", min_samples: int = 2) -> np.ndarray:
    """Validate that ``X`` is a 2D non-empty point cloud with enough samples."""
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2D (n_samples, n_features); got shape {X.shape}")
    n_samples, n_features = X.shape
    if n_samples < min_samples:
        raise ValueError(f"{name} needs at least {min_samples} samples; got {n_samples}")
    if n_features < 1:
        raise ValueError(f"{name} must have at least 1 feature; got {n_features}")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{name} contains non-finite entries")
    return X


def pca_spectrum(X: np.ndarray) -> np.ndarray:
    """Return the variance spectrum (fraction of total) of ``X``.

    Centers ``X`` then computes singular values via SVD; squares them
    to get variances and normalises so the array sums to one. Values
    are sorted in descending order.

    Args:
        X: ``(n_samples, n_features)`` point cloud.

    Returns:
        ``(min(n_samples - 1, n_features),)`` array of normalised
        variances. The first entry is the fraction of variance carried
        by the leading principal component.

    Raises:
        ValueError: If ``X`` is empty, not 2D, or contains non-finite
            entries.
    """
    X = _validate_X(X, min_samples=2)
    Xc = X - X.mean(axis=0, keepdims=True)
    # full_matrices=False is faster and we only need the non-trivial singular
    # values.
    _, s, _ = np.linalg.svd(Xc, full_matrices=False)
    variances = s**2
    total = float(variances.sum())
    if total <= 0.0:
        # Degenerate (all points identical) -> return uniform spectrum.
        return np.full_like(variances, 1.0 / max(variances.size, 1))
    return variances / total


def pca_dim_at_threshold(X: np.ndarray, threshold: float = 0.95) -> int:
    """Smallest ``k`` such that cumulative PCA variance >= ``threshold``.

    Args:
        X: ``(n_samples, n_features)`` point cloud.
        threshold: Cumulative variance fraction; must lie in (0, 1].

    Returns:
        Integer ``k`` in ``[1, min(n_samples - 1, n_features)]``.

    Raises:
        ValueError: If ``X`` is invalid (see :func:`_validate_X`) or
            ``threshold`` is out of range.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1]; got {threshold}")
    spectrum = pca_spectrum(X)
    cum = np.cumsum(spectrum)
    # searchsorted returns the smallest index i such that cum[i] >= threshold.
    k = int(np.searchsorted(cum, threshold) + 1)
    return min(k, spectrum.size)


def _levina_bickel_single_k(distances: np.ndarray, k: int) -> float:
    """Aggregate Levina-Bickel MLE for a single ``k``.

    ``distances`` is ``(n_samples, k)`` where column ``j`` holds the
    distance from each point to its (j+1)-th nearest neighbour (so the
    self-distance is already excluded). The local MLE per LeBi (Eq. 8) is

        d_hat_k(x_i) = [(1/(k-1)) * sum_{j=1..k-1} log(r_k / r_j)]^(-1).

    The global estimate is the harmonic mean over points:

        d_hat = [(1/n) sum_i (1 / d_hat_k(x_i))]^(-1).
    """
    n_samples, k_cols = distances.shape
    if k_cols != k:
        raise ValueError(
            f"_levina_bickel_single_k expected distances with {k} columns; " f"got {k_cols}"
        )
    if k < 2:
        raise ValueError(f"k must be >= 2 for Levina-Bickel MLE; got {k}")

    # Avoid log(0) by clipping tiny distances. This typically only matters
    # for exact duplicates, which the caller should normally have filtered.
    eps = np.finfo(np.float64).tiny
    r_k = np.maximum(distances[:, -1:], eps)  # (n, 1)
    r_j = np.maximum(distances[:, :-1], eps)  # (n, k-1)
    log_ratios = np.log(r_k / r_j)  # (n, k-1)
    inv_local = log_ratios.mean(axis=1)  # (n,), == 1 / d_hat_k(x_i)
    # Points with all-zero ratios (collapsed neighbourhoods) are degenerate.
    valid = np.isfinite(inv_local) & (inv_local > 0.0)
    if not np.any(valid):
        raise ValueError(
            "Levina-Bickel: all local estimates degenerate; check for " "duplicate points"
        )
    mean_inv = float(inv_local[valid].mean())
    return 1.0 / mean_inv


def levina_bickel_mle(
    X: np.ndarray,
    k: int | Sequence[int] = (5, 10, 15, 20),
) -> float | dict:
    """Levina-Bickel 2004 maximum-likelihood intrinsic dimension.

    Implements Eq. 8 of Levina and Bickel (NIPS 2004), with self-distances
    excluded from the neighbour search. When ``k`` is a sequence, returns a
    dict mapping each ``k`` to its individual estimate plus a ``'mean'`` key
    giving the average.

    Args:
        X: ``(n_samples, n_features)`` point cloud.
        k: Number of neighbours per point. Either a single int (returns a
            float) or a sequence (returns a dict with a ``'mean'`` key).

    Returns:
        Float estimate if ``k`` is an int; ``{k_i: estimate_i, ..., 'mean': ...}``
        dict otherwise.

    Raises:
        ValueError: If ``X`` is invalid, any ``k < 2``, or ``n_samples`` does
            not exceed the largest ``k`` (needed because the self-point is
            skipped).
    """
    if not _HAS_SKLEARN:  # pragma: no cover
        raise ImportError("levina_bickel_mle requires scikit-learn")

    if isinstance(k, int):
        ks: list[int] = [k]
        return_dict = False
    else:
        ks = [int(v) for v in k]
        return_dict = True

    for k_val in ks:
        if k_val < 2:
            raise ValueError(f"Levina-Bickel requires k >= 2 (uses k-1 ratios); got {k_val}")

    k_max = max(ks)
    X = _validate_X(X, min_samples=k_max + 1)

    # Single neighbour search for the largest k; reuse columns for smaller k.
    nn = NearestNeighbors(n_neighbors=k_max + 1, algorithm="auto")
    nn.fit(X)
    distances, _ = nn.kneighbors(X)  # (n, k_max + 1)
    # Column 0 is the self-distance (0.0); drop it.
    distances = distances[:, 1:]

    estimates: dict[int, float] = {}
    for k_val in ks:
        estimates[k_val] = _levina_bickel_single_k(distances[:, :k_val], k_val)

    if not return_dict:
        return estimates[ks[0]]

    result: dict = {int(k_val): float(v) for k_val, v in estimates.items()}
    result["mean"] = float(np.mean(list(estimates.values())))
    return result


def two_nn(X: np.ndarray, discard_fraction: float = 0.1) -> float:
    """Two-NN intrinsic dimension (Facco et al. Sci. Rep. 2017).

    Per point, ``mu_i = r2_i / r1_i`` where ``r1, r2`` are the first and
    second nearest-neighbour distances (self excluded). The cumulative
    distribution of ``log(mu)`` is exponential with rate ``d`` under
    uniform sampling on a d-manifold, so a line fit to
    ``-log(1 - F(mu))`` versus ``log(mu)`` has slope ``d``.

    Args:
        X: ``(n_samples, n_features)`` point cloud.
        discard_fraction: Fraction of the largest ``mu`` values to drop
            before the regression. Defaults to ``0.1`` per common
            practice (Facco et al. Fig. 1; trims the heavy tail caused
            by outliers and curvature). Must be in ``[0, 1)``.

    Returns:
        Slope ``d`` of the through-origin linear fit; positive float.

    Raises:
        ValueError: If ``X`` is invalid, has fewer than 3 samples, or
            ``discard_fraction`` is out of range.
    """
    if not _HAS_SKLEARN:  # pragma: no cover
        raise ImportError("two_nn requires scikit-learn")
    if not 0.0 <= discard_fraction < 1.0:
        raise ValueError(f"discard_fraction must lie in [0, 1); got {discard_fraction}")
    X = _validate_X(X, min_samples=3)
    nn = NearestNeighbors(n_neighbors=3, algorithm="auto")
    nn.fit(X)
    distances, _ = nn.kneighbors(X)  # (n, 3)
    # Column 0 is the self-distance; columns 1, 2 are r1, r2.
    r1 = distances[:, 1]
    r2 = distances[:, 2]
    valid = (r1 > 0.0) & (r2 > 0.0) & np.isfinite(r1) & np.isfinite(r2)
    if not np.any(valid):
        raise ValueError(
            "two_nn: no points with positive neighbour distances; check for "
            "duplicate points or zero variance"
        )
    mu = (r2[valid] / r1[valid]).astype(np.float64)
    # Some mu may be exactly 1 (collinear duplicates); the regression needs
    # log(mu) > 0 so drop those.
    mu = mu[mu > 1.0]
    if mu.size < 2:
        raise ValueError("two_nn: fewer than 2 points with r2 > r1; cannot fit slope")
    mu.sort()
    n = mu.size
    # Empirical CDF: F_i = i / (n + 1) for i = 1..n, per Facco et al.
    F = np.arange(1, n + 1, dtype=np.float64) / (n + 1)
    # Trim the heavy tail before the fit.
    keep = int(round(n * (1.0 - discard_fraction)))
    keep = max(2, min(keep, n))
    log_mu = np.log(mu[:keep])
    rhs = -np.log(1.0 - F[:keep])
    # Through-origin least squares: slope = sum(x*y) / sum(x^2).
    denom = float((log_mu**2).sum())
    if denom <= 0.0:
        raise ValueError("two_nn: zero variance in log(mu) after trimming")
    slope = float((log_mu * rhs).sum() / denom)
    return slope


def _residual_variance(geodesic: np.ndarray, embedding: np.ndarray) -> float:
    """Residual variance ``1 - R^2`` between geodesic and embedded distances.

    Args:
        geodesic: ``(n, n)`` symmetric matrix of geodesic distances.
        embedding: ``(n, k)`` Isomap embedding.

    Returns:
        ``1 - corrcoef(geodesic_offdiag, embedding_offdiag)^2``. The
        correlation is taken over the upper-triangular off-diagonal
        entries to avoid double counting and ignoring self-distances.
    """
    n = geodesic.shape[0]
    iu = np.triu_indices(n, k=1)
    geo_flat = geodesic[iu]
    # Pairwise Euclidean in the embedding.
    diff = embedding[:, None, :] - embedding[None, :, :]
    emb_dist = np.sqrt((diff**2).sum(axis=-1))
    emb_flat = emb_dist[iu]
    # Pearson correlation; fall back to 1.0 - 0 = 1.0 if either is constant.
    geo_std = float(geo_flat.std())
    emb_std = float(emb_flat.std())
    if geo_std <= 0.0 or emb_std <= 0.0:
        return 1.0
    corr = float(np.corrcoef(geo_flat, emb_flat)[0, 1])
    return float(1.0 - corr**2)


def _elbow_dim(dims: np.ndarray, curve: np.ndarray) -> int:
    """Estimate the elbow of a monotonically (mostly) decreasing curve.

    Uses the largest finite second difference: the point where the curve
    flattens most sharply. Tolerant of non-monotonic noise by clipping
    the second difference at zero before argmax.
    """
    if curve.size < 3:
        return int(dims[int(np.argmin(curve))])
    # Second differences approximate curvature on a uniform grid.
    second = np.diff(curve, n=2)
    # We want the most concave-up point (largest positive second diff).
    idx = int(np.argmax(second)) + 1  # +1 because diff shrinks index by 1
    return int(dims[idx])


def isomap_residual(X: np.ndarray, k_max: int = 20, n_neighbors: int = 10) -> dict:
    """Isomap residual-variance curve and elbow estimate.

    Fits a single Isomap with ``n_components = k_max`` (so the geodesic
    distance matrix is built once), then evaluates the residual variance
    of the leading ``k`` columns of the embedding for ``k = 1..k_max``.
    The intrinsic dim is estimated as the elbow of that curve.

    Args:
        X: ``(n_samples, n_features)`` point cloud.
        k_max: Maximum embedding dimension to evaluate. Capped at
            ``min(n_samples - 1, n_features)``.
        n_neighbors: Neighbours per point in the Isomap k-NN graph.

    Returns:
        Dict with keys ``'dims'`` (``np.ndarray`` of evaluated
        dimensions starting at 1), ``'residual_variance'`` (``np.ndarray``,
        one per dim), and ``'elbow_dim'`` (int, the estimated intrinsic
        dim).

    Raises:
        ImportError: If scikit-learn is not installed.
        ValueError: If ``X`` is invalid or has too few samples.
    """
    if not _HAS_SKLEARN:
        raise ImportError("isomap_residual requires scikit-learn; install sklearn>=1.0")
    X = _validate_X(X, min_samples=n_neighbors + 2)
    n_samples, n_features = X.shape
    k_eff = min(k_max, n_samples - 1, n_features)
    if k_eff < 1:
        raise ValueError(f"k_max effective dim {k_eff} < 1; need more samples or features")
    iso = Isomap(n_components=k_eff, n_neighbors=n_neighbors)
    embedding = iso.fit_transform(X)  # (n, k_eff)
    geodesic = iso.dist_matrix_  # (n, n) symmetric

    dims = np.arange(1, k_eff + 1, dtype=np.int64)
    rvar = np.empty(k_eff, dtype=np.float64)
    for k_idx in range(k_eff):
        rvar[k_idx] = _residual_variance(geodesic, embedding[:, : k_idx + 1])

    elbow = _elbow_dim(dims, rvar)
    return {
        "dims": dims,
        "residual_variance": rvar,
        "elbow_dim": elbow,
    }


def agreement_summary(X: np.ndarray) -> dict:
    """Run all four estimators and report the agreement.

    Convenience wrapper for the four estimators implemented in this
    module. Returns a single dict suitable for JSON serialisation.

    Args:
        X: ``(n_samples, n_features)`` point cloud.

    Returns:
        Dict with keys:
        - ``'pca_95'``: PCA dim at 95% cumulative variance.
        - ``'pca_99'``: PCA dim at 99% cumulative variance.
        - ``'levina_bickel'``: Dict over multiple k with ``'mean'`` key.
        - ``'two_nn'``: Float Two-NN estimate.
        - ``'isomap_elbow'``: Int Isomap elbow (or ``None`` if sklearn
          missing or the fit failed).
        - ``'consensus'``: Median of the four scalar estimates
          (pca_95, levina_bickel['mean'], two_nn, isomap_elbow). If
          ``isomap_elbow`` is ``None`` it is dropped from the median.
        - ``'spread'``: Max minus min of the same set of scalar
          estimates.

    Raises:
        ValueError: If ``X`` is invalid.
    """
    X = _validate_X(X, min_samples=22)  # need 22 for LB k_max=20 + self
    pca95 = pca_dim_at_threshold(X, threshold=0.95)
    pca99 = pca_dim_at_threshold(X, threshold=0.99)
    lb = levina_bickel_mle(X)
    assert isinstance(lb, dict)
    tnn = two_nn(X)
    iso_elbow: int | None
    try:
        iso = isomap_residual(X)
        iso_elbow = int(iso["elbow_dim"])
    except (ImportError, ValueError):
        iso_elbow = None

    scalars: list[float] = [float(pca95), float(lb["mean"]), float(tnn)]
    if iso_elbow is not None:
        scalars.append(float(iso_elbow))
    consensus = float(np.median(scalars))
    spread = float(max(scalars) - min(scalars))

    return {
        "pca_95": int(pca95),
        "pca_99": int(pca99),
        "levina_bickel": lb,
        "two_nn": float(tnn),
        "isomap_elbow": iso_elbow,
        "consensus": consensus,
        "spread": spread,
    }


__all__ = [
    "pca_spectrum",
    "pca_dim_at_threshold",
    "levina_bickel_mle",
    "two_nn",
    "isomap_residual",
    "agreement_summary",
]
