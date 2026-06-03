"""
infotheory.estimators
======================

Low-level information-theoretic estimators used by the vortex-jepa causal /
information-theoretic analysis. Everything here is method-agnostic: it knows
nothing about gusts, latents, or the manuscript. It takes numpy arrays and
returns nats (natural log) unless told otherwise.

Two estimator families are provided so that every reported number can carry the
same uncertainty triangulation the manuscript already uses elsewhere:

  1. k-nearest-neighbour estimators (Kraskov-Stoegbauer-Grassberger for mutual
     information, Kozachenko-Leonenko for differential entropy). These are the
     workhorse for continuous variables and do not assume Gaussianity, matching
     the spirit of Arranz & Lozano-Duran (JFM 2024) and SURD
     (Martinez-Sanchez et al., Nat. Commun. 2024).

  2. A histogram / plug-in estimator with Miller-Madow bias correction, used
     (a) as an independent cross-check on the kNN numbers and (b) as the input
     to a discrete SURD when variables are pre-binned.

Surrogate / shuffle nulls are provided for every quantity so a measured value
can be compared against the distribution expected under independence at finite
sample size. This is the information-theoretic analogue of the bootstrap CIs in
the manuscript: it tells you whether a small positive MI is real or is just the
finite-sample floor.

Design notes
------------
* Nothing here imports torch or any project module. It is deliberately a leaf
  dependency so it can be unit-tested in isolation (see tests/).
* All public functions accept X of shape (n_samples,) or (n_samples, n_dims).
* `random_state` is threaded everywhere for reproducibility, matching the
  manuscript's fixed-seed discipline.

References
----------
Kraskov, Stoegbauer & Grassberger, "Estimating mutual information",
  Phys. Rev. E 69, 066138 (2004).
Kozachenko & Leonenko, "Sample estimate of the entropy of a random vector",
  Probl. Peredachi Inf. 23, 9-16 (1987).
Cover & Thomas, "Elements of Information Theory", 2nd ed. (2006).
Arranz & Lozano-Duran, "Informative and non-informative decomposition of
  turbulent flow fields", J. Fluid Mech. 1000, A95 (2024).
"""
from __future__ import annotations

import numpy as np
from scipy.special import digamma, gammaln
from scipy.spatial import cKDTree

NATS_TO_BITS = 1.0 / np.log(2.0)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _atleast_2d_col(x: np.ndarray) -> np.ndarray:
    """Return x as (n_samples, n_dims) float64, treating 1-D input as a column."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    elif x.ndim != 2:
        raise ValueError(f"expected 1-D or 2-D array, got shape {x.shape}")
    return x


def _add_jitter(x: np.ndarray, scale: float, rng: np.random.Generator) -> np.ndarray:
    """
    Add a tiny amount of noise to break ties / exact duplicates, which otherwise
    bias kNN estimators. Scale is relative to each column's std. KSG is known to
    be sensitive to repeated coordinates; this is the standard remedy.
    """
    if scale <= 0:
        return x
    std = x.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return x + scale * std * rng.standard_normal(size=x.shape)


def _standardise(x: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-std per column. KSG is invariant to per-coordinate scaling
    only in the Chebyshev-metric formulation; standardising makes the chosen
    metric behave sensibly across heterogeneous units (e.g. enstrophy vs C_L)."""
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (x - mu) / sd


# --------------------------------------------------------------------------- #
# differential entropy (Kozachenko-Leonenko)
# --------------------------------------------------------------------------- #
def entropy_knn(
    x: np.ndarray,
    k: int = 4,
    *,
    standardise: bool = True,
    jitter: float = 1e-10,
    random_state: int | None = 0,
) -> float:
    """
    Kozachenko-Leonenko differential entropy estimate, in nats.

    H(X) ~ -psi(k) + psi(N) + log(c_d) + (d/N) * sum_i log(eps_i)

    with eps_i twice the distance to the k-th neighbour (Euclidean), c_d the
    volume of the unit d-ball. Note differential entropy is scale dependent; if
    `standardise` is True the returned value is the entropy of the standardised
    variable and should be used only for *differences* (e.g. inside an MI), not
    as an absolute bit count.
    """
    rng = np.random.default_rng(random_state)
    x = _atleast_2d_col(x)
    n, d = x.shape
    if standardise:
        x = _standardise(x)
    x = _add_jitter(x, jitter, rng)

    tree = cKDTree(x)
    # k+1 because the first neighbour is the point itself (distance 0)
    dist, _ = tree.query(x, k=k + 1, p=2.0)
    eps = dist[:, k]
    eps = np.where(eps == 0, np.finfo(float).tiny, eps)

    log_cd = (d / 2.0) * np.log(np.pi) - gammaln(d / 2.0 + 1.0)
    h = -digamma(k) + digamma(n) + log_cd + d * np.mean(np.log(eps))
    return float(h)


# --------------------------------------------------------------------------- #
# mutual information (Kraskov-Stoegbauer-Grassberger, estimator 1)
# --------------------------------------------------------------------------- #
def mutual_information_knn(
    x: np.ndarray,
    y: np.ndarray,
    k: int = 4,
    *,
    standardise: bool = True,
    jitter: float = 1e-10,
    random_state: int | None = 0,
) -> float:
    """
    KSG estimator (algorithm 1) of I(X; Y) in nats, for continuous X, Y of
    arbitrary (possibly multivariate) dimension.

    I(X;Y) ~ psi(k) + psi(N) - <psi(n_x + 1) + psi(n_y + 1)>

    where n_x, n_y count neighbours within the k-th-neighbour Chebyshev radius
    in the marginal X- and Y-subspaces. The estimate is clipped at 0 from below
    only by the caller if desired; raw (possibly slightly negative) values are
    returned here so that surrogate nulls are meaningful.
    """
    rng = np.random.default_rng(random_state)
    x = _atleast_2d_col(x)
    y = _atleast_2d_col(y)
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of samples")
    n = x.shape[0]
    if standardise:
        x = _standardise(x)
        y = _standardise(y)
    x = _add_jitter(x, jitter, rng)
    y = _add_jitter(y, jitter, rng)

    xy = np.hstack([x, y])
    # Chebyshev (max) metric is the KSG-1 choice
    tree_xy = cKDTree(xy)
    dist, _ = tree_xy.query(xy, k=k + 1, p=np.inf)
    radius = dist[:, k]  # distance to k-th neighbour in joint space

    tree_x = cKDTree(x)
    tree_y = cKDTree(y)
    # count strictly within the radius (KSG-1 convention: open ball, minus self)
    nx = np.array(tree_x.query_ball_point(x, radius - 1e-15, p=np.inf, return_length=True)) - 1
    ny = np.array(tree_y.query_ball_point(y, radius - 1e-15, p=np.inf, return_length=True)) - 1
    nx = np.clip(nx, 0, None)
    ny = np.clip(ny, 0, None)

    mi = digamma(k) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1))
    return float(mi)


def conditional_mutual_information_knn(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    k: int = 4,
    *,
    standardise: bool = True,
    random_state: int | None = 0,
) -> float:
    """
    Conditional MI I(X; Y | Z) via the chain rule of KSG estimates:
        I(X;Y|Z) = I(X; [Y,Z]) - I(X; Z).
    This "difference of KSG" form is simple and adequate for the low-dimensional
    scalar sets used here. For higher dimensions prefer a dedicated CMI estimator
    (e.g. Frenzel-Pompe / Runge KSG-CMI); flagged in the module docstring.
    """
    y = _atleast_2d_col(y)
    z = _atleast_2d_col(z)
    yz = np.hstack([y, z])
    i_x_yz = mutual_information_knn(
        x, yz, k=k, standardise=standardise, random_state=random_state
    )
    i_x_z = mutual_information_knn(
        x, z, k=k, standardise=standardise, random_state=random_state
    )
    return float(i_x_yz - i_x_z)


# --------------------------------------------------------------------------- #
# histogram / plug-in estimators (independent cross-check; discrete SURD input)
# --------------------------------------------------------------------------- #
def _miller_madow_correction(counts: np.ndarray, n: int) -> float:
    """Miller-Madow bias term (+(K-1)/(2N)) added back to plug-in entropy."""
    k_occupied = np.count_nonzero(counts)
    return (k_occupied - 1) / (2.0 * n)


def entropy_hist(counts: np.ndarray, *, bias_correct: bool = True) -> float:
    """Plug-in Shannon entropy (nats) from an integer count array, with optional
    Miller-Madow correction."""
    counts = np.asarray(counts, dtype=np.float64).ravel()
    n = counts.sum()
    if n <= 0:
        return 0.0
    p = counts[counts > 0] / n
    h = -np.sum(p * np.log(p))
    if bias_correct:
        h += _miller_madow_correction(counts, int(n))
    return float(h)


def quantise(x: np.ndarray, n_bins: int, *, strategy: str = "quantile") -> np.ndarray:
    """
    Map continuous columns to integer bin labels for histogram MI / discrete SURD.

    strategy='quantile' gives equiprobable bins (recommended for heavy-tailed
    quantities like wake enstrophy and circulation, which are exactly the
    observables the manuscript cares about); 'uniform' gives equal-width bins.
    Returns an int array of the same shape.
    """
    x = _atleast_2d_col(x)
    out = np.empty(x.shape, dtype=np.int64)
    for j in range(x.shape[1]):
        col = x[:, j]
        if strategy == "quantile":
            edges = np.quantile(col, np.linspace(0, 1, n_bins + 1))
            edges[0], edges[-1] = -np.inf, np.inf
            edges = np.unique(edges)
        elif strategy == "uniform":
            lo, hi = col.min(), col.max()
            if hi == lo:
                hi = lo + 1.0
            edges = np.linspace(lo, hi, n_bins + 1)
            edges[0], edges[-1] = -np.inf, np.inf
        else:
            raise ValueError(f"unknown strategy {strategy!r}")
        out[:, j] = np.clip(np.digitize(col, edges[1:-1]), 0, n_bins - 1)
    return out if out.shape[1] > 1 else out[:, 0]


def joint_pmf(labels: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """
    Build a dense joint pmf array of the given `shape` from integer label columns.
    `labels` is (n_samples, n_vars); `shape` is the per-variable cardinality.
    Returns a normalised array summing to 1. This is the input format the SURD
    reference implementation expects.
    """
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim == 1:
        labels = labels[:, None]
    flat = np.ravel_multi_index([labels[:, j] for j in range(labels.shape[1])], shape)
    counts = np.bincount(flat, minlength=int(np.prod(shape))).astype(np.float64)
    p = counts / counts.sum()
    return p.reshape(shape)


def mutual_information_hist(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 8,
    *,
    strategy: str = "quantile",
    bias_correct: bool = True,
    max_joint_cells: int = 2_000_000,
) -> float:
    """Histogram MI (nats): I = H(X) + H(Y) - H(X,Y) with Miller-Madow on each.

    Only well-defined for low-dimensional X, Y: the joint table has
    n_bins**(dx+dy) cells, which blows up combinatorially. Raises ValueError if
    the implied table would exceed `max_joint_cells`; callers that may pass
    multivariate sources (e.g. a 32-dim latent) should catch this and fall back to
    the kNN estimator, which has no such limit.
    """
    xb = quantise(x, n_bins, strategy=strategy)
    yb = quantise(y, n_bins, strategy=strategy)
    xb = _atleast_2d_col(xb).astype(int)
    yb = _atleast_2d_col(yb).astype(int)
    dx, dy = xb.shape[1], yb.shape[1]
    if n_bins ** (dx + dy) > max_joint_cells:
        raise ValueError(
            f"histogram MI table {n_bins}**{dx+dy} exceeds {max_joint_cells}; "
            "use the kNN estimator for multivariate sources."
        )
    # collapse multivariate to a single label per sample
    sx = tuple([n_bins] * xb.shape[1])
    sy = tuple([n_bins] * yb.shape[1])
    fx = np.ravel_multi_index([xb[:, j] for j in range(xb.shape[1])], sx)
    fy = np.ravel_multi_index([yb[:, j] for j in range(yb.shape[1])], sy)
    cx = np.bincount(fx)
    cy = np.bincount(fy)
    fxy = fx * int(np.prod(sy)) + fy
    cxy = np.bincount(fxy)
    return float(
        entropy_hist(cx, bias_correct=bias_correct)
        + entropy_hist(cy, bias_correct=bias_correct)
        - entropy_hist(cxy, bias_correct=bias_correct)
    )


# --------------------------------------------------------------------------- #
# surrogate / shuffle nulls
# --------------------------------------------------------------------------- #
def surrogate_null_mi(
    x: np.ndarray,
    y: np.ndarray,
    estimator=mutual_information_knn,
    *,
    n_surrogate: int = 200,
    random_state: int | None = 0,
    **est_kwargs,
) -> dict:
    """
    Estimate the null distribution of I(X;Y) under independence by permuting Y.

    Returns a dict with the measured MI, the surrogate mean/std, a z-score, a
    one-sided permutation p-value (fraction of surrogates >= measured), and the
    bias-corrected estimate (measured minus surrogate mean). The bias-corrected
    value is the honest "excess information above the finite-sample floor" and is
    what should be reported for small MIs.
    """
    rng = np.random.default_rng(random_state)
    y = _atleast_2d_col(y)
    measured = estimator(x, y, random_state=random_state, **est_kwargs)
    null = np.empty(n_surrogate)
    for s in range(n_surrogate):
        perm = rng.permutation(y.shape[0])
        null[s] = estimator(x, y[perm], random_state=int(rng.integers(2**31)), **est_kwargs)
    mu, sd = float(null.mean()), float(null.std() + 1e-12)
    return {
        "mi": float(measured),
        "mi_debiased": float(measured - mu),
        "null_mean": mu,
        "null_std": sd,
        "z": float((measured - mu) / sd),
        "p_value": float((np.sum(null >= measured) + 1) / (n_surrogate + 1)),
        "n_surrogate": n_surrogate,
    }


def estimator_robustness_sweep(
    x: np.ndarray,
    y: np.ndarray,
    *,
    ks=(3, 4, 6, 8, 12),
    bins=(6, 8, 12, 16),
    random_state: int | None = 0,
) -> dict:
    """
    Report I(X;Y) across a sweep of kNN k and histogram bin counts. A result is
    trustworthy when the two families agree and neither drifts strongly with the
    hyperparameter. This is the MI analogue of the manuscript's seed-variance and
    cross-validation checks, and should accompany every headline MI.
    """
    knn = {int(k): mutual_information_knn(x, y, k=int(k), random_state=random_state) for k in ks}
    hist = {}
    for b in bins:
        try:
            hist[int(b)] = mutual_information_hist(x, y, n_bins=int(b))
        except ValueError:
            # multivariate source: histogram table would blow up; skip this leg.
            pass
    vals = list(knn.values()) + list(hist.values())
    return {
        "knn_by_k": knn,
        "hist_by_bins": hist,
        "spread": float(np.max(vals) - np.min(vals)),
        "median": float(np.median(vals)),
        "hist_skipped": len(hist) == 0,
    }
