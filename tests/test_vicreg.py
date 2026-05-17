"""Tests for ``src.models.vicreg.VICReg``."""

from __future__ import annotations

import math

import pytest
import torch

from src.models.vicreg import VICReg


def test_vicreg_low_on_isotropic_unit_variance_gaussian() -> None:
    """N(0, I_32) with N=1024: variance hinge ~0 (std~1), covariance term small."""
    torch.manual_seed(0)
    z = torch.randn(1024, 32)
    vic = VICReg(d=32)
    loss = vic(z).item()
    assert math.isfinite(loss)
    assert 0.0 <= loss < 1.0


def test_vicreg_high_on_collapsed_z() -> None:
    """All-zero z: std = sqrt(eps) ~ 0.01, hinge ~ gamma - sqrt(eps) ~ 0.99.

    Variance loss is mu * 0.99 ~ 24.75. Covariance is 0. Total in [24, 26].
    """
    torch.manual_seed(0)
    z = torch.zeros(1024, 32)
    vic = VICReg(d=32, mu=25.0, nu=1.0, gamma=1.0, eps=1e-4)
    loss = vic(z).item()
    assert math.isfinite(loss)
    expected = 25.0 * (1.0 - math.sqrt(1e-4))
    assert abs(loss - expected) < 0.5
    assert 24.0 <= loss <= 26.0


def test_vicreg_high_on_low_rank_z() -> None:
    """Only the first 4 of 32 dims have variance ~1; the other 28 are zero.

    Variance hinge fires on 28 dims (each contributes ~0.99); 4 dims
    contribute ~0. Mean over d = 28 * 0.99 / 32 ~ 0.866. mu * 0.866 ~ 21.65.
    """
    torch.manual_seed(0)
    z_active = torch.randn(1024, 4)
    z_zero = torch.zeros(1024, 28)
    z = torch.cat([z_active, z_zero], dim=1)
    vic = VICReg(d=32, mu=25.0, nu=1.0, gamma=1.0, eps=1e-4)
    loss = vic(z).item()
    assert math.isfinite(loss)
    assert 20.0 <= loss <= 23.0

    gaussian = VICReg(d=32)(torch.randn(1024, 32)).item()
    assert loss > 10.0 * gaussian


def test_vicreg_high_on_correlated_z() -> None:
    """All 32 dims equal a single random vector u(N).

    Variance is fine (each dim has the same std as u). Covariance is
    maximal: every off-diagonal equals the on-diagonal. With N=1024 and
    32 dims, the covariance term dominates.
    """
    torch.manual_seed(0)
    u = torch.randn(1024, 1)
    z = u.expand(-1, 32).contiguous()
    vic = VICReg(d=32, mu=25.0, nu=1.0, gamma=1.0)
    loss = vic(z).item()
    assert math.isfinite(loss)
    assert loss > 5.0


def test_vicreg_gradient_flows() -> None:
    """Backward pass produces non-zero gradients on z."""
    torch.manual_seed(0)
    z = torch.randn(64, 32, requires_grad=True)
    vic = VICReg(d=32)
    loss = vic(z)
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert z.grad.abs().sum().item() > 0.0


def test_vicreg_dtype_promotion() -> None:
    """Under autocast bf16, the output is fp32 and gradients are finite.

    Mirrors the SIGReg numerical-stability convention.
    """
    torch.manual_seed(0)
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    z = torch.randn(64, 32, device=device_type, requires_grad=True)
    vic = VICReg(d=32).to(device_type)
    with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
        loss = vic(z)
    assert loss.dtype == torch.float32
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


def test_vicreg_lambda_argument_is_inert_without_second_view() -> None:
    """D22: ``lambda_`` is kept in the API but does not affect the forward
    pass without a second view. Loss is identical for any lambda_."""
    torch.manual_seed(0)
    z = torch.randn(256, 32)
    a = VICReg(d=32, lambda_=0.0)(z).item()
    b = VICReg(d=32, lambda_=25.0)(z).item()
    c = VICReg(d=32, lambda_=1e6)(z).item()
    assert a == pytest.approx(b)
    assert a == pytest.approx(c)
