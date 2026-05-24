"""Off-diagonal covariance penalty as a JEPA-native total-correlation surrogate.

Session 12 Direction F: SIGReg constrains the marginal distributions of the
projected latent ``z`` toward a standard Gaussian along random Cramer-Wold
projections, but it does NOT directly penalise cross-coordinate correlation
between latent dimensions. The Session 11 W0_C_lam100 encoder reached a
participation ratio of PR(z) = 11.66 on a ``d = 32`` latent budget, leaving
roughly two-thirds of the available capacity unused. Adding an off-diagonal
covariance penalty pushes the empirical covariance ``Cov(z)`` toward diagonal,
broadening the effective dimensionality and (the hypothesis) improving wake
reconstruction.

Reference (motivation, not formulation):
    Wang, Tirelli, Discetti, Ianiro. "Disentangled latent representations of
    turbulent flows." arXiv:2604.18059, April 2026 (decorrelation as a
    regulariser of turbulent-flow latents).

This is NOT the VAE-style total-correlation penalty
``TC(z) = KL[q(z) || prod_i q(z_i)]`` from Chen et al. (NeurIPS 2018,
beta-TC-VAE) and Kim and Mnih (ICML 2018, FactorVAE), which require a full
density estimate of ``q(z)``. The off-diagonal covariance penalty here is the
zeroth-order Gaussian surrogate: it matches only the second moment of the
joint distribution, so it cannot detect higher-order dependencies, but it is
a single deterministic ``O(N d^2)`` op that composes cleanly with SIGReg and
needs no extra discriminator or sampling.

The mathematical form ``|| off_diag(Cov(z)) ||_F^2 / d`` is the same expression
that appears as the covariance term of VICReg (Bardes, Ponce, LeCun, ICLR
2022), but here it is exposed as a standalone module so Direction F can sweep
its weight independently of the anti-collapse module choice.
"""

from __future__ import annotations

import torch
from torch import Tensor


def off_diagonal_covariance_loss(z: Tensor) -> Tensor:
    """Off-diagonal covariance penalty for a batch of latent embeddings.

    Computes the empirical covariance of ``z`` over the batch, zeroes the
    diagonal, and returns the squared Frobenius norm normalised by ``d``.
    The ``/ d`` factor keeps the loss scale comparable across latent widths.

    Args:
        z: Batch of shape ``(N, d)`` with ``N >= 2`` and ``d >= 1``. Any
            floating dtype is accepted; the computation is performed in fp32
            internally to match the SIGReg / VICReg numerical-stability
            convention (HANDOFF.md D13).

    Returns:
        Scalar fp32 tensor ``|| off_diag(Cov(z)) ||_F^2 / d`` where
        ``Cov(z) = (z - z.mean(0)).T @ (z - z.mean(0)) / (N - 1)``.

    Raises:
        ValueError: If ``z`` is not 2D or has fewer than 2 rows.
    """
    if z.dim() != 2:
        raise ValueError(f"expected z of shape (N, d), got {tuple(z.shape)}")
    n, d = z.shape
    if n < 2:
        raise ValueError(f"off-diagonal covariance requires at least 2 samples, got {n}")

    with torch.amp.autocast(device_type=z.device.type, enabled=False):
        z32 = z.float()
        z_centered = z32 - z32.mean(dim=0, keepdim=True)
        cov = (z_centered.t() @ z_centered) / (n - 1)
        off_diag = cov - torch.diag(torch.diagonal(cov))
        return off_diag.pow(2).sum() / float(d)
