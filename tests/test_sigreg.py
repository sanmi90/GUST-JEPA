"""Unit tests for src/models/sigreg.py.

Threshold calibration: the magnitudes below use the LeWM Appendix A definition
of the Epps-Pulley statistic (no N multiplier) integrated over the spec's knot
range ``(0.2, 4.0)``. See HANDOFF.md D13 for the decision record.
"""

from __future__ import annotations

import torch

from src.models.sigreg import SIGReg


def test_sigreg_low_on_isotropic_gaussian() -> None:
    """A batch from N(0, I_32) yields a small SIGReg statistic."""
    torch.manual_seed(0)
    z = torch.randn(4096, 32)
    loss = SIGReg(dim=32)(z)
    assert loss.item() < 0.01, f"expected < 0.01, got {loss.item()}"


def test_sigreg_high_on_heavy_tailed() -> None:
    """A batch from Student-t df=2 yields a clearly larger SIGReg statistic."""
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    # Student-t(df=2) via Cauchy-style construction: N(0, 1) / sqrt(chi^2_2 / 2).
    normal = torch.randn(4096, 32, generator=gen)
    chi2 = torch.randn(4096, 32, generator=gen) ** 2 + torch.randn(4096, 32, generator=gen) ** 2
    z = normal / (chi2 / 2.0).sqrt()
    loss = SIGReg(dim=32)(z)
    assert loss.item() > 0.05, f"expected > 0.05, got {loss.item()}"


def test_sigreg_high_on_uniform() -> None:
    """A batch from Uniform(-1, 1) yields a moderately large SIGReg statistic.

    A unit projection of d=32 iid uniforms is approximately N(0, 1/3) by the
    CLT, which deviates from the N(0, 1) target the regularizer compares
    against.
    """
    torch.manual_seed(0)
    z = torch.rand(4096, 32) * 2.0 - 1.0
    loss = SIGReg(dim=32)(z)
    assert loss.item() > 0.02, f"expected > 0.02, got {loss.item()}"


def test_sigreg_invariant_to_projection_count() -> None:
    """SIGReg on an isotropic Gaussian batch agrees within 20% across M values."""
    torch.manual_seed(0)
    z = torch.randn(4096, 32)
    vals: list[float] = []
    for m in (64, 256, 1024):
        torch.manual_seed(m)
        vals.append(SIGReg(dim=32, num_projections=m)(z).item())
    mean_val = sum(vals) / len(vals)
    spread = max(vals) - min(vals)
    assert spread < 0.2 * mean_val, f"vals={vals}, spread={spread}, 20%mean={0.2 * mean_val}"


def test_sigreg_gradient_flows() -> None:
    """Backward pass produces non-zero gradients on z."""
    torch.manual_seed(0)
    z = torch.randn(4096, 32, requires_grad=True)
    loss = SIGReg(dim=32)(z)
    loss.backward()
    assert z.grad is not None
    assert torch.any(z.grad != 0)


def test_sigreg_dtype_promotion() -> None:
    """bf16 input under autocast yields fp32 output; grad on the input is bf16."""
    torch.manual_seed(0)
    z = torch.randn(4096, 32, dtype=torch.bfloat16, requires_grad=True)
    sigreg = SIGReg(dim=32)
    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss = sigreg(z)
    assert loss.dtype == torch.float32, f"expected fp32 output, got {loss.dtype}"
    assert torch.isfinite(loss).item()
    loss.backward()
    assert z.grad is not None
    assert z.grad.dtype == torch.bfloat16, f"expected bf16 grad, got {z.grad.dtype}"
