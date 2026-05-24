"""Tests for ``src.models.total_correlation.off_diagonal_covariance_loss``.

Session 12 Direction F: off-diagonal covariance penalty on the SIGReg-projected
latent z. The threshold magnitudes below follow VICReg's covariance-term
convention (Bardes, Ponce, LeCun, ICLR 2022): the loss is in normalised-second-
moment units and a Gaussian batch with finite ``N`` should sit near zero while
a degenerate batch should dominate.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.models.total_correlation import off_diagonal_covariance_loss


def test_tc_zero_on_identity_covariance() -> None:
    """A batch with ~identity sample covariance yields a small TC penalty.

    Drawing ``z ~ N(0, I_d)`` with a large batch makes the empirical
    off-diagonal entries shrink at rate ``1 / sqrt(N)``. With ``N = 4096``
    and ``d = 32`` the off-diagonal entries are O(1/sqrt(N)) ~ 0.015, so
    ``sum(off_diag**2) / d`` is at most ``31 * 32 * (3 * 0.015)**2 / 32 ~ 0.02``.
    """
    torch.manual_seed(0)
    z = torch.randn(4096, 32)
    loss = off_diagonal_covariance_loss(z).item()
    assert math.isfinite(loss)
    assert 0.0 <= loss < 0.05, f"expected near-zero for iid Gaussian, got {loss}"


def test_tc_positive_on_correlated() -> None:
    """Two columns made identical produce a strictly positive TC.

    With ``z[:, 1] = z[:, 0]`` (two columns equal up to sign) and the rest
    drawn iid Gaussian, the (0, 1) and (1, 0) off-diagonals equal the
    on-diagonal of column 0 (~1.0). The penalty must dominate the iid
    baseline computed in ``test_tc_zero_on_identity_covariance``.
    """
    torch.manual_seed(0)
    z = torch.randn(4096, 32)
    z[:, 1] = z[:, 0]
    loss = off_diagonal_covariance_loss(z).item()
    assert math.isfinite(loss)
    baseline = off_diagonal_covariance_loss(torch.randn(4096, 32)).item()
    assert loss > baseline + 0.01, (
        f"correlated batch ({loss}) should exceed iid baseline ({baseline})"
    )
    # The (0, 1) and (1, 0) entries together contribute ~2 * 1.0**2 / d = 0.0625.
    assert loss > 0.05, f"expected > 0.05 for perfect column duplication, got {loss}"


def test_tc_gradient_flow() -> None:
    """Backward pass produces a non-zero gradient on z."""
    torch.manual_seed(0)
    z = torch.randn(256, 32, requires_grad=True)
    loss = off_diagonal_covariance_loss(z)
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert z.grad.abs().sum().item() > 0.0


def test_tc_rejects_non_2d_input() -> None:
    """Non-2D input is a ValueError (mirrors the SIGReg / VICReg contract)."""
    z3 = torch.randn(4, 16, 32)
    with pytest.raises(ValueError):
        off_diagonal_covariance_loss(z3)


def test_tc_rejects_single_row() -> None:
    """A single-row batch has no defined (N-1)-normalised covariance."""
    z1 = torch.randn(1, 32)
    with pytest.raises(ValueError):
        off_diagonal_covariance_loss(z1)


def test_tc_d_normalisation() -> None:
    """The ``/ d`` factor keeps the loss scale comparable across latent widths.

    Constructing two batches where each off-diagonal entry has the same
    magnitude (here: every off-diagonal of the empirical covariance equals
    a fixed ``rho``) and varying ``d`` should leave the loss roughly
    proportional to ``d`` once the ``/ d`` factor is applied. We use the
    fully-correlated batch ``z[:, i] = u`` for which ``Cov(z) = std(u)**2 * 1
    1^T``, so every off-diagonal entry equals the same constant. The number
    of off-diagonals is ``d * (d - 1)``, hence ``loss = (d - 1) * c**2``
    (constant in ``d`` to leading order rather than quadratic).
    """
    torch.manual_seed(0)
    u = torch.randn(2048, 1)
    z32 = u.expand(-1, 32).contiguous()
    z64 = u.expand(-1, 64).contiguous()
    loss32 = off_diagonal_covariance_loss(z32).item()
    loss64 = off_diagonal_covariance_loss(z64).item()
    # The ratio is ``(d64 - 1) / (d32 - 1) = 63 / 31 ~ 2.03``: linear in d, not d^2.
    assert 1.5 < loss64 / loss32 < 2.5, (
        f"d-normalisation should keep ratio near (d-1)/(d-1); got {loss64 / loss32}"
    )


def test_tc_dtype_promotion_bf16() -> None:
    """Under bf16 autocast, the output is fp32 and grads stay finite."""
    torch.manual_seed(0)
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    z = torch.randn(256, 32, device=device_type, requires_grad=True)
    with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16):
        loss = off_diagonal_covariance_loss(z)
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss).item()
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
