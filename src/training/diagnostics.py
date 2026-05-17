"""Anti-collapse diagnostics for the JEPA training loop.

Reference (CLAUDE.md "Risk-management" and HANDOFF.md D5):
    The single biggest risk on this dataset is that SIGReg fails on the
    low-intrinsic-dim (~5 to 10) physics manifold (LeWM Two-Room failure
    mode). These diagnostics give the training loop the information it
    needs to detect the failure and fire the auto-fallback to VICReg.

Functions:
    participation_ratio: PR = (sum s_i)^2 / sum s_i^2 over singular values.
        Equals d for isotropic latents, 1 for rank-1 collapse.
    linear_probe_r2: fit a linear regression z -> c on a held-out subset
        of indices, evaluate R^2 on the rest. Used to check that c
        information is not leaking into the encoder latent (high R^2 is
        a red flag: HANDOFF.md "Warnings and pitfalls").
    per_dim_variance_histogram: distribution of per-dimension variances,
        useful for visualising dimensional collapse on W&B.

All three are pure functions; the training loop calls them every
``diagnostic_every`` iterations on a held-out batch.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def participation_ratio(z_batch: Tensor) -> float:
    """Effective number of active dimensions from singular values of z.

    PR = (sum_i s_i)^2 / sum_i s_i^2

    where s_i are the singular values of ``z_batch``. PR ranges from 1
    (rank-1 / complete collapse) to ``d`` (isotropic full-rank).

    Args:
        z_batch: ``(N, d)`` real-valued tensor. Typically ``N = B * T``.

    Returns:
        Participation ratio as a Python ``float`` in ``(0, d]``.
    """
    if z_batch.dim() != 2:
        raise ValueError(f"expected z_batch of shape (N, d), got {tuple(z_batch.shape)}")
    z32 = z_batch.detach().float()
    s = torch.linalg.svdvals(z32)
    numerator = s.sum().pow(2)
    denominator = s.pow(2).sum().clamp_min(torch.finfo(torch.float32).eps)
    return (numerator / denominator).item()


def linear_probe_r2(
    z: Tensor,
    c: Tensor,
    fit_indices: Tensor,
    eval_indices: Tensor,
) -> dict[str, float]:
    """Closed-form linear probe ``z -> c`` and per-coordinate R^2.

    Solves the least-squares fit on ``fit_indices`` and evaluates on
    ``eval_indices``. The probe is *fit per call*, so this is a cheap
    diagnostic, not a learned head with separate state.

    Args:
        z: ``(N, d)`` latents.
        c: ``(N, c_dim)`` targets. ``c_dim`` is 3 for ``(G, D, Y)`` per D16.
        fit_indices: 1D long tensor of indices into ``[0, N)``.
        eval_indices: 1D long tensor disjoint from ``fit_indices``.

    Returns:
        ``{'r2_overall': float, 'r2_G': float, 'r2_D': float, 'r2_Y': float}``.
        ``r2_overall`` is the unweighted mean of the per-coordinate R^2.
        Names beyond ``r2_G/r2_D/r2_Y`` are emitted only if ``c_dim >= 3``;
        for smaller ``c_dim`` the per-coordinate keys are ``r2_0, r2_1, ...``.
        R^2 can be negative when the probe predicts worse than the mean.
    """
    if z.dim() != 2 or c.dim() != 2:
        raise ValueError(
            f"z and c must be 2D; got z={tuple(z.shape)}, c={tuple(c.shape)}"
        )
    if z.shape[0] != c.shape[0]:
        raise ValueError(f"z and c must agree on N; got {z.shape[0]} vs {c.shape[0]}")

    z32 = z.detach().float()
    c32 = c.detach().float()
    z_fit = z32[fit_indices]
    c_fit = c32[fit_indices]
    z_eval = z32[eval_indices]
    c_eval = c32[eval_indices]

    n_fit, d = z_fit.shape
    bias = torch.ones(n_fit, 1, device=z_fit.device, dtype=z_fit.dtype)
    X_fit = torch.cat([z_fit, bias], dim=1)
    sol = torch.linalg.lstsq(X_fit, c_fit)
    W = sol.solution

    n_eval = z_eval.shape[0]
    X_eval = torch.cat([z_eval, torch.ones(n_eval, 1, device=z_eval.device, dtype=z_eval.dtype)], dim=1)
    c_pred = X_eval @ W

    ss_res = (c_eval - c_pred).pow(2).sum(dim=0)
    c_mean = c_eval.mean(dim=0, keepdim=True)
    ss_tot = (c_eval - c_mean).pow(2).sum(dim=0).clamp_min(torch.finfo(torch.float32).eps)
    r2_per_dim = 1.0 - ss_res / ss_tot

    c_dim = c.shape[1]
    coord_names = ["G", "D", "Y"] if c_dim == 3 else [str(i) for i in range(c_dim)]
    out: dict[str, float] = {f"r2_{name}": float(r2_per_dim[i].item()) for i, name in enumerate(coord_names)}
    out["r2_overall"] = float(r2_per_dim.mean().item())
    return out


def per_dim_variance_histogram(
    z_batch: Tensor,
    n_bins: int = 20,
) -> tuple[Tensor, Tensor]:
    """Histogram of per-dimension variances over a batch.

    Useful as a W&B image: a healthy run has a unimodal distribution
    away from zero; dimensional collapse shows up as a spike at zero.

    Args:
        z_batch: ``(N, d)`` real-valued tensor.
        n_bins: Number of histogram bins.

    Returns:
        ``(counts, bin_edges)``. ``counts`` is ``(n_bins,)``,
        ``bin_edges`` is ``(n_bins + 1,)``, both fp32 CPU tensors.
    """
    if z_batch.dim() != 2:
        raise ValueError(f"expected z_batch of shape (N, d), got {tuple(z_batch.shape)}")
    var = z_batch.detach().float().var(dim=0, unbiased=True).cpu()
    counts, bin_edges = np.histogram(var.numpy(), bins=n_bins)
    return torch.from_numpy(counts).float(), torch.from_numpy(bin_edges).float()
