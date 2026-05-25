"""AeroJEPA-style concept-vector arithmetic for the (G, D, Y) parameter space.

This module implements the concept-vector analysis introduced for JEPA-style
latents in Giral, Vishwasrao, Arroyo Ramo, Golestanian, Tonti, Lozano-Duran,
Brunton, Hoyas, Gomez, Le Clainche, Vinuesa, "AeroJEPA: Learning Semantic
Latent Representations for Scalable 3D Aerodynamic Field Modeling,"
arXiv:2605.05586, May 2026. It exposes two estimators of per-axis latent
sensitivities and a small extrapolation API to use them.

Definitions
-----------
Let ``z_i in R^d`` be encoded latents and ``c_i = (G_i, D_i, Y_i) in R^3``
the associated (gust amplitude, vortex diameter, lateral offset) labels.

Method 1, AeroJEPA Eq. 9, matched-pair averaging:
    For axis ``a in {G, D, Y}`` collect every pair ``(z_i, z_j)`` whose
    labels differ only in axis ``a`` (the other two axes match within
    ``match_tol``), then estimate

        v_a = E[ (z_i - z_j) / (c_i[a] - c_j[a]) ]

    over all such pairs. ``v_a`` is the average per-unit-increment latent
    response along axis ``a``.

Method 2, AeroJEPA Eq. 11, linear-probe Jacobian:
    Fit a multi-output ridge regression ``c = W @ z + b`` with
    ``W in R^{3 x d}``, ``b in R^3``. The per-axis concept vector is the
    ``a``-th row of ``(W @ W^T + alpha I)^{-1} @ W``, equivalently the
    ``a``-th column of the regularised right-pseudoinverse
    ``W^+ = W^T (W W^T + alpha I)^{-1}``. By construction
    ``W @ v_a = e_a`` so ``v_a`` is the infinitesimal change in ``z`` that
    moves ``c[a]`` by one unit while leaving the other two ``c`` components
    unchanged at the linear-probe solution.

Method 3, linear extrapolation:
    Given a base latent ``z_0`` at labels ``c_0`` and target labels
    ``c_target``, the linearised forecast is

        z_target_pred = z_0 + sum_a v_a * (c_target[a] - c_0[a]).

    Compared to an encoded ``z_target`` this yields a relative L2 / cosine
    error reported by :func:`extrapolation_error`.

Notes
-----
- All inputs / outputs are ``numpy`` arrays. No torch, no sklearn.
- Returned concept vectors always have shape ``(d,)``; callers that need
  them stacked can use ``np.stack(list(d.values()))``.
- ``construct_concept_vectors_averaging`` returns ``None`` for axes with
  fewer than ``min_pairs`` matched pairs and emits a ``logging`` warning;
  callers should branch on ``None`` rather than rely on a sentinel array.
- The ridge term ``alpha`` in the Jacobian estimator is on ``W W^T`` (the
  3 x 3 output Gram), which prevents inversion blow-up when two of the
  three axes are nearly degenerate (e.g. tiny grid).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

_LOG = logging.getLogger(__name__)

_DEFAULT_AXES: tuple[str, str, str] = ("G", "D", "Y")


def _validate_latents_labels(
    latents: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and cast (latents, labels) to float64. Returns the cast pair."""
    z = np.asarray(latents, dtype=np.float64)
    c = np.asarray(labels, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError(f"latents must be 2-D (N, d); got shape {z.shape}")
    if c.ndim != 2 or c.shape[1] != 3:
        raise ValueError(f"labels must be (N, 3); got shape {c.shape}")
    if z.shape[0] != c.shape[0]:
        raise ValueError(
            f"latents and labels must have the same N; got "
            f"{z.shape[0]} vs {c.shape[0]}"
        )
    return z, c


def construct_concept_vectors_averaging(
    latents: np.ndarray,
    labels: np.ndarray,
    axis_names: tuple[str, str, str] = _DEFAULT_AXES,
    match_tol: tuple[float, float, float] = (1e-6, 1e-6, 1e-6),
    min_pairs: int = 2,
) -> dict[str, Optional[np.ndarray]]:
    """AeroJEPA Eq. 9: concept vectors from matched-pair latent differences.

    For each axis ``a``, sweep all ordered pairs ``(i, j)`` with ``i < j``.
    A pair is "matched on axis ``a``" when the two off-axis label values
    agree within ``match_tol`` and the on-axis labels differ. The per-pair
    sensitivity is the finite difference

        delta_z / delta_c[a] = (z_i - z_j) / (c_i[a] - c_j[a]),

    and ``v_a`` is the unweighted mean of these vectors over all such pairs.

    Args:
        latents: ``(N, d)`` array of encoded latents.
        labels: ``(N, 3)`` array of ``(G, D, Y)`` labels for each latent.
        axis_names: Display names for the three columns of ``labels``.
        match_tol: Per-axis tolerance for considering two off-axis labels
            equal. ``(1e-6, 1e-6, 1e-6)`` is appropriate for the
            vortex-jepa grid where label values are exact rationals; use
            a larger tolerance (e.g. ``1e-3``) when labels were rounded.
        min_pairs: Axes with fewer than this many matched pairs are
            returned as ``None`` and a warning is logged.

    Returns:
        Mapping ``{axis_name: vector or None}``. Each returned vector has
        shape ``(d,)``; ``None`` indicates insufficient matched pairs.
    """
    z, c = _validate_latents_labels(latents, labels)
    n, d = z.shape
    tol = np.asarray(match_tol, dtype=np.float64)
    if tol.shape != (3,):
        raise ValueError(f"match_tol must be length 3; got {tol.shape}")

    out: dict[str, Optional[np.ndarray]] = {}

    for axis_idx, name in enumerate(axis_names):
        other_axes = [a for a in range(3) if a != axis_idx]
        # Vectorised pairwise differences keep N small (~hundreds).
        # diff[i, j, a] = c[i, a] - c[j, a]
        diff = c[:, None, :] - c[None, :, :]  # (N, N, 3)
        abs_diff = np.abs(diff)
        # Mask: other axes within tolerance AND axis_idx differs (nonzero diff).
        match_others = np.all(abs_diff[..., other_axes] <= tol[other_axes], axis=-1)
        on_axis_diff = diff[..., axis_idx]
        on_axis_nonzero = np.abs(on_axis_diff) > tol[axis_idx]
        # Take upper triangular (i < j) to avoid double-counting.
        triu = np.triu(np.ones((n, n), dtype=bool), k=1)
        keep = match_others & on_axis_nonzero & triu

        ii, jj = np.where(keep)
        n_pairs = ii.size
        if n_pairs < min_pairs:
            _LOG.warning(
                "Concept vector for axis %s: only %d matched pair(s) found "
                "(need >= %d); returning None.",
                name,
                n_pairs,
                min_pairs,
            )
            out[name] = None
            continue

        delta_z = z[ii] - z[jj]                       # (n_pairs, d)
        delta_c = on_axis_diff[ii, jj][:, None]       # (n_pairs, 1)
        per_pair = delta_z / delta_c                  # (n_pairs, d)
        v_a = per_pair.mean(axis=0).reshape(d)        # (d,)
        out[name] = v_a

    return out


def construct_concept_vectors_jacobian(
    latents: np.ndarray,
    labels: np.ndarray,
    axis_names: tuple[str, str, str] = _DEFAULT_AXES,
    ridge_alpha: float = 1e-6,
) -> dict[str, np.ndarray]:
    """AeroJEPA Eq. 11: concept vectors from a linear-probe Jacobian.

    Fits the multi-output ridge regression

        c = W z + b,    W in R^{3 x d}, b in R^3,

    by minimising ``sum_i || c_i - (W z_i + b) ||^2 + alpha_z ||W||_F^2``
    via the closed form on the mean-centred design (so ``b`` is recovered
    as ``mean(c) - W @ mean(z)``). The per-axis concept vector is then

        v_a = a-th row of (W W^T + ridge_alpha * I_3)^{-1} W,

    equivalently the ``a``-th column of the (ridge-regularised) right
    pseudoinverse ``W^+ = W^T (W W^T + ridge_alpha I)^{-1}``. Because
    ``W @ v_a = e_a`` (up to ``ridge_alpha``), ``v_a`` is the local
    direction in latent space that moves ``c[a]`` by one unit while
    leaving the other two labels fixed at the linear-probe solution.

    Args:
        latents: ``(N, d)`` array of encoded latents.
        labels: ``(N, 3)`` array of ``(G, D, Y)`` labels.
        axis_names: Display names for the three columns of ``labels``.
        ridge_alpha: Tikhonov ridge applied to both the W-fit on the
            ``d x d`` input Gram (``z^T z``) and to the ``3 x 3`` output
            Gram (``W W^T``) before inversion. Set to ``0.0`` for the
            unregularised case; values around ``1e-6`` are safe and
            recover the closed form on well-conditioned designs.

    Returns:
        Mapping ``{axis_name: vector of shape (d,)}``. Each vector
        satisfies ``W @ vector approx e_a``.
    """
    z, c = _validate_latents_labels(latents, labels)
    n, d = z.shape
    if n < 2:
        raise ValueError(f"need at least 2 samples to fit a linear probe; got {n}")

    # Mean-centre so the bias term is absorbed.
    z_mu = z.mean(axis=0, keepdims=True)
    c_mu = c.mean(axis=0, keepdims=True)
    z_c = z - z_mu
    c_c = c - c_mu

    # Solve (Z^T Z + alpha I) W^T = Z^T C in R^{d x 3}, then transpose.
    gram_z = z_c.T @ z_c + ridge_alpha * np.eye(d)
    rhs = z_c.T @ c_c                                          # (d, 3)
    w_t = np.linalg.solve(gram_z, rhs)                         # (d, 3)
    w_mat = w_t.T                                              # (3, d)

    # Per-axis concept vectors via (W W^T + alpha I)^-1 W.
    output_gram = w_mat @ w_mat.T + ridge_alpha * np.eye(3)    # (3, 3)
    concept_mat = np.linalg.solve(output_gram, w_mat)          # (3, d)

    return {name: concept_mat[i].reshape(d) for i, name in enumerate(axis_names)}


def linear_extrapolate(
    z0: np.ndarray,
    c0: np.ndarray,
    c_target: np.ndarray,
    concept_vectors: dict[str, Optional[np.ndarray]],
    axis_names: tuple[str, str, str] = _DEFAULT_AXES,
) -> np.ndarray:
    """Linear extrapolation in the concept-vector basis.

    Computes ``z_pred = z0 + sum_a v_a * (c_target[a] - c0[a])`` for the
    three axes named in ``axis_names``. Axes whose concept vector is
    ``None`` are skipped (the corresponding label delta is treated as
    contributing zero), so a caller can still extrapolate along the two
    axes that have enough matched pairs even when the third lacks them.

    Args:
        z0: ``(d,)`` base latent.
        c0: ``(3,)`` base labels.
        c_target: ``(3,)`` target labels.
        concept_vectors: Mapping from axis name to ``(d,)`` concept
            vector (or ``None`` for axes to skip).
        axis_names: Display names for the three columns of ``c0`` /
            ``c_target`` matching the keys of ``concept_vectors``.

    Returns:
        ``(d,)`` extrapolated latent.
    """
    z0 = np.asarray(z0, dtype=np.float64).reshape(-1)
    c0 = np.asarray(c0, dtype=np.float64).reshape(-1)
    c_target = np.asarray(c_target, dtype=np.float64).reshape(-1)
    if c0.shape != (3,) or c_target.shape != (3,):
        raise ValueError(
            f"c0 and c_target must be length 3; got {c0.shape}, {c_target.shape}"
        )

    z_pred = z0.copy()
    delta_c = c_target - c0
    for axis_idx, name in enumerate(axis_names):
        v_a = concept_vectors.get(name)
        if v_a is None:
            continue
        v_a = np.asarray(v_a, dtype=np.float64).reshape(-1)
        if v_a.shape != z0.shape:
            raise ValueError(
                f"concept vector for axis {name} has shape {v_a.shape}, "
                f"expected {z0.shape}"
            )
        z_pred = z_pred + v_a * delta_c[axis_idx]
    return z_pred


def extrapolation_error(
    z_pred: np.ndarray,
    z_true: np.ndarray,
) -> dict[str, float]:
    """L2 / cosine error between a predicted and a true latent.

    Args:
        z_pred: ``(d,)`` predicted latent.
        z_true: ``(d,)`` reference latent (encoded ground truth).

    Returns:
        Dict with keys

        - ``rel_l2``: ``||z_pred - z_true|| / max(||z_true||, eps)``.
        - ``abs_l2``: ``||z_pred - z_true||``.
        - ``cosine_sim``: ``z_pred . z_true /
          max(||z_pred|| ||z_true||, eps)``. Defaults to ``1.0`` when
          either norm is below ``eps`` (zero-vector convention).
    """
    z_pred = np.asarray(z_pred, dtype=np.float64).reshape(-1)
    z_true = np.asarray(z_true, dtype=np.float64).reshape(-1)
    if z_pred.shape != z_true.shape:
        raise ValueError(
            f"z_pred and z_true must have matching shape; got "
            f"{z_pred.shape} vs {z_true.shape}"
        )

    eps = 1e-12
    diff = z_pred - z_true
    abs_l2 = float(np.linalg.norm(diff))
    true_norm = float(np.linalg.norm(z_true))
    pred_norm = float(np.linalg.norm(z_pred))
    rel_l2 = abs_l2 / max(true_norm, eps)
    if pred_norm < eps or true_norm < eps:
        cosine_sim = 1.0 if abs_l2 < eps else 0.0
    else:
        cosine_sim = float(np.dot(z_pred, z_true) / (pred_norm * true_norm))
    return {"rel_l2": rel_l2, "abs_l2": abs_l2, "cosine_sim": cosine_sim}


def consistency_check(
    averaging_vectors: dict[str, Optional[np.ndarray]],
    jacobian_vectors: dict[str, np.ndarray],
) -> dict[str, dict[str, object]]:
    """Compare the two concept-vector estimators on a per-axis basis.

    Args:
        averaging_vectors: Output of
            :func:`construct_concept_vectors_averaging`.
        jacobian_vectors: Output of
            :func:`construct_concept_vectors_jacobian`.

    Returns:
        Mapping ``{axis_name: {axis_name: str, cosine_similarity: float,
        norm_ratio: float}}`` with one entry per axis present in both
        inputs. ``cosine_similarity`` close to ``1`` means the two
        estimators identify the same direction; ``norm_ratio`` is
        ``||averaging|| / max(||jacobian||, eps)`` and should be of
        order unity when the latent geometry is roughly linear in the
        labels. Axes for which ``averaging_vectors`` is ``None`` are
        omitted from the result.
    """
    out: dict[str, dict[str, object]] = {}
    eps = 1e-12
    for name, v_avg in averaging_vectors.items():
        if v_avg is None:
            continue
        if name not in jacobian_vectors:
            continue
        v_jac = jacobian_vectors[name]
        v_avg = np.asarray(v_avg, dtype=np.float64).reshape(-1)
        v_jac = np.asarray(v_jac, dtype=np.float64).reshape(-1)
        if v_avg.shape != v_jac.shape:
            raise ValueError(
                f"axis {name}: averaging vector shape {v_avg.shape} does "
                f"not match jacobian vector shape {v_jac.shape}"
            )
        n_avg = float(np.linalg.norm(v_avg))
        n_jac = float(np.linalg.norm(v_jac))
        if n_avg < eps or n_jac < eps:
            cos = 0.0
        else:
            cos = float(np.dot(v_avg, v_jac) / (n_avg * n_jac))
        norm_ratio = n_avg / max(n_jac, eps)
        out[name] = {
            "axis_name": name,
            "cosine_similarity": cos,
            "norm_ratio": norm_ratio,
        }
    return out
