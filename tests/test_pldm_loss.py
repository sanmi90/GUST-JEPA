"""Tests for ``src/baselines/pldm.py``.

The PLDM loss follows arXiv:2502.14819 (Sobal, Zhang, Cho, Balestriero,
Rudner, LeCun, Feb 2025), Section 3.3 + Appendix D.1.1. The paper's
formulation has **five** terms, not seven as D8 originally claimed:

    L_JEPA = L_sim + alpha * L_var + beta * L_cov
                   + delta * L_time-sim + omega * L_IDM

where ``L_sim`` is the multi-step rollout MSE (the paper's L_sim is what
this codebase calls open-loop rollout from a single anchor frame), the
variance and covariance terms follow Bardes et al. ICLR 2022 (VICReg),
``L_time-sim`` is the temporal-smoothness MSE ``||z_t - z_{t+1}||^2``,
and ``L_IDM`` is the inverse-dynamics regression that predicts the
per-step action from ``(z_t, z_{t+1})``. There are NO var/cov terms on
the temporal-difference signal ``dz`` (the D8 reading was incorrect).

For the vortex-jepa adaptation, the IDM head predicts the static episode
descriptor ``c = (G, D, Y)`` instead of a per-step action (HANDOFF.md
D8, corrected via D32 + D30).
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from src.baselines.pldm import PLDMLoss


def _gauss(n: int, d: int, seed: int = 0) -> torch.Tensor:
    """Standard normal (N, d) tensor with a fixed seed."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g)


def test_pldm_shape_contract() -> None:
    """All loss components are zero-dim scalars; output dict has the seven
    expected keys (5 terms + L_total + L_sim alias)."""
    torch.manual_seed(0)
    loss = PLDMLoss(d=32, c_dim=3)
    z = torch.randn(4, 8, 32)
    z_hat = torch.randn(4, 8, 32)
    c = torch.randn(4, 3)
    out = loss(z, z_hat, c)
    expected = {"L_total", "L_sim", "L_var", "L_cov", "L_time_sim", "L_idm"}
    assert expected.issubset(set(out.keys())), f"missing keys: {expected - set(out.keys())}"
    for k in expected:
        assert out[k].dim() == 0, f"{k} must be a scalar"
        assert torch.isfinite(out[k]), f"{k} is not finite: {out[k]}"


def test_pldm_sim_zero_on_perfect_rollout() -> None:
    """If z_hat exactly equals z, L_sim is zero. The other four terms are
    nonzero and depend only on z, so they are unchanged."""
    torch.manual_seed(0)
    loss = PLDMLoss(d=32, c_dim=3, lambda_var=0.0, lambda_cov=0.0,
                    lambda_time_sim=0.0, lambda_idm=0.0)
    z = torch.randn(4, 8, 32)
    out = loss(z, z.clone(), torch.randn(4, 3))
    assert out["L_sim"].item() < 1e-6, f"L_sim should be zero on perfect rollout, got {out['L_sim'].item()}"


def test_pldm_var_low_on_unit_variance_gaussian() -> None:
    """z ~ N(0, I_32) gives L_var below 0.1 (hinge is satisfied at gamma=1).

    This is the VICReg variance hinge: max(0, gamma - sqrt(var + eps))
    averaged over (time, dim). Standard Gaussian latents satisfy the
    hinge essentially everywhere.
    """
    torch.manual_seed(0)
    loss = PLDMLoss(d=32, c_dim=3, gamma=1.0)
    z = torch.randn(32, 8, 32)
    out = loss(z, z.clone(), torch.randn(32, 3))
    assert out["L_var"].item() < 0.1, f"L_var should be ~0 for unit Gaussian, got {out['L_var'].item()}"


def test_pldm_var_high_on_collapsed_z() -> None:
    """z = zeros gives L_var = gamma - sqrt(eps) ~= gamma = 1.0 (default).

    Matches the analogous test in tests/test_vicreg.py.
    """
    torch.manual_seed(0)
    loss = PLDMLoss(d=32, c_dim=3, gamma=1.0)
    z = torch.zeros(8, 8, 32)
    out = loss(z, z.clone(), torch.randn(8, 3))
    assert 0.9 < out["L_var"].item() < 1.05, f"L_var should be ~1.0 for collapsed z, got {out['L_var'].item()}"


def test_pldm_cov_low_on_independent_dims() -> None:
    """Standard Gaussian latents have small off-diagonal covariance.

    Empirical floor with N=1024 samples per time slice: each off-diag
    entry has variance ~1/N=1e-3, summed over d*(d-1)=992 off-diag
    entries and divided by d=32 gives ~0.03 per time slice. Threshold
    0.1 is loose enough to pass under torch's seed but tight enough to
    catch a broken implementation that returns the FULL covariance
    norm (which would be ~1.0).
    """
    torch.manual_seed(0)
    loss = PLDMLoss(d=32, c_dim=3)
    z = torch.randn(1024, 8, 32)
    out = loss(z, z.clone(), torch.randn(1024, 3))
    assert out["L_cov"].item() < 0.1, f"L_cov should be ~0 for independent dims, got {out['L_cov'].item()}"


def test_pldm_cov_high_on_correlated_dims() -> None:
    """If every dim is a copy of one base signal, off-diagonal covariance is large."""
    torch.manual_seed(0)
    loss = PLDMLoss(d=32, c_dim=3)
    base = torch.randn(64, 8, 1)
    z = base.expand(-1, -1, 32).contiguous()
    out = loss(z, z.clone(), torch.randn(64, 3))
    assert out["L_cov"].item() > 0.5, f"L_cov should be large for perfectly correlated dims, got {out['L_cov'].item()}"


def test_pldm_time_sim_zero_on_static_z() -> None:
    """If z[:, t, :] is identical for all t, the temporal-smoothness term is 0.

    Note this is "perfectly smooth but pathological" (the latent does not
    capture dynamics). The PLDM paper does NOT penalize this on its own;
    the L_sim term against ground truth z would catch it elsewhere.
    """
    torch.manual_seed(0)
    loss = PLDMLoss(d=32, c_dim=3)
    z_one = torch.randn(8, 1, 32)
    z = z_one.expand(-1, 8, -1).contiguous()
    out = loss(z, z.clone(), torch.randn(8, 3))
    assert out["L_time_sim"].item() < 1e-6, f"L_time_sim should be 0 for static z, got {out['L_time_sim'].item()}"


def test_pldm_time_sim_high_on_random_z() -> None:
    """White-noise z has L_time_sim ~ 2.0 (this impl mean-reduces over all axes).

    Per-element diff is N(0, 2) so the mean of (diff)^2 is 2. The paper
    sums over (t, d) and normalises by batch only, which would give
    2 * (T-1) * d; we mean-reduce for numerical stability and absorb
    the scale into the lambda weights (HANDOFF.md D30).
    """
    torch.manual_seed(0)
    loss = PLDMLoss(d=32, c_dim=3)
    z = torch.randn(32, 8, 32)
    out = loss(z, z.clone(), torch.randn(32, 3))
    assert 1.5 < out["L_time_sim"].item() < 2.5, (
        f"L_time_sim for white-noise should be ~2.0 (mean of N(0,2)^2), got {out['L_time_sim'].item()}"
    )


def test_pldm_idm_finite_at_init() -> None:
    """L_idm is finite at random init; an untrained MLP predicting c from
    (z_t, z_{t+1}) on random latents produces a bounded MSE."""
    torch.manual_seed(0)
    loss = PLDMLoss(d=32, c_dim=3)
    z = torch.randn(8, 8, 32)
    c = torch.randn(8, 3)
    out = loss(z, z.clone(), c)
    assert torch.isfinite(out["L_idm"]), f"L_idm not finite: {out['L_idm']}"
    assert out["L_idm"].item() > 0.0, "L_idm should be positive at random init"


def test_pldm_idm_decreases_with_training() -> None:
    """A few Adam steps on L_idm alone make the IDM MLP improve."""
    torch.manual_seed(0)
    loss = PLDMLoss(d=32, c_dim=3, lambda_var=0.0, lambda_cov=0.0,
                    lambda_time_sim=0.0, lambda_idm=1.0)
    z = torch.randn(16, 8, 32)
    c = torch.randn(16, 3)
    opt = torch.optim.Adam(loss.idm.parameters(), lr=1e-2)
    L0 = loss(z, z.clone(), c)["L_idm"].item()
    for _ in range(20):
        out = loss(z, z.clone(), c)
        opt.zero_grad(set_to_none=True)
        out["L_idm"].backward()
        opt.step()
    L1 = loss(z, z.clone(), c)["L_idm"].item()
    assert L1 < L0, f"L_idm should decrease after IDM training: {L0:.4f} -> {L1:.4f}"


def test_pldm_total_equals_weighted_sum() -> None:
    """L_total == L_sim + a*L_var + b*L_cov + d*L_time_sim + o*L_idm."""
    torch.manual_seed(0)
    a, b, dd, oo = 4.0, 6.9, 0.75, 1.0
    loss = PLDMLoss(d=32, c_dim=3, lambda_var=a, lambda_cov=b,
                    lambda_time_sim=dd, lambda_idm=oo)
    z = torch.randn(4, 8, 32)
    z_hat = torch.randn(4, 8, 32)
    c = torch.randn(4, 3)
    out = loss(z, z_hat, c)
    expected = (
        out["L_sim"].item()
        + a * out["L_var"].item()
        + b * out["L_cov"].item()
        + dd * out["L_time_sim"].item()
        + oo * out["L_idm"].item()
    )
    assert math.isclose(out["L_total"].item(), expected, rel_tol=1e-5, abs_tol=1e-5), (
        f"L_total {out['L_total'].item()} != weighted sum {expected}"
    )


def test_pldm_gradient_flows_to_z() -> None:
    """Each of the five terms produces a nonzero gradient on z (the encoder side)."""
    torch.manual_seed(0)
    z = torch.randn(8, 8, 32, requires_grad=True)
    z_hat = torch.randn(8, 8, 32, requires_grad=True)
    c = torch.randn(8, 3)
    for key in ["L_sim", "L_var", "L_cov", "L_time_sim", "L_idm"]:
        loss = PLDMLoss(d=32, c_dim=3)
        out = loss(z, z_hat, c)
        if z.grad is not None:
            z.grad.zero_()
        if z_hat.grad is not None:
            z_hat.grad.zero_()
        out[key].backward(retain_graph=False)
        # L_sim depends on z and z_hat; everything else only on z.
        if key == "L_sim":
            assert z.grad is not None and z.grad.abs().sum().item() > 0.0
            assert z_hat.grad is not None and z_hat.grad.abs().sum().item() > 0.0
        else:
            assert z.grad is not None and z.grad.abs().sum().item() > 0.0
        # Re-zero so the next iteration computes a fresh graph
        z = z.detach().requires_grad_(True)
        z_hat = z_hat.detach().requires_grad_(True)


def test_pldm_dtype_promotion_under_bf16() -> None:
    """Inputs in bf16 are promoted to fp32 inside the loss (same convention
    as src/models/sigreg.py and src/models/vicreg.py)."""
    torch.manual_seed(0)
    loss = PLDMLoss(d=32, c_dim=3)
    z = torch.randn(4, 8, 32, dtype=torch.bfloat16)
    z_hat = torch.randn(4, 8, 32, dtype=torch.bfloat16)
    c = torch.randn(4, 3, dtype=torch.bfloat16)
    out = loss(z, z_hat, c)
    for k in ["L_sim", "L_var", "L_cov", "L_time_sim", "L_idm", "L_total"]:
        assert out[k].dtype == torch.float32, f"{k} dtype is {out[k].dtype}, expected float32"
        assert torch.isfinite(out[k]), f"{k} not finite under bf16: {out[k]}"
