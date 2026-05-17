"""Tests for ``src.training.diagnostics``."""

from __future__ import annotations

import math

import pytest
import torch

from src.training.diagnostics import (
    linear_probe_r2,
    participation_ratio,
    per_dim_variance_histogram,
)


def test_pr_isotropic_gaussian_is_close_to_d() -> None:
    """For N=8192, d=32 isotropic samples, PR concentrates near d.

    With N >> d the Marcenko-Pastur band is tight enough that PR > 0.85 * d.
    """
    torch.manual_seed(0)
    z = torch.randn(8192, 32)
    pr = participation_ratio(z)
    assert math.isfinite(pr)
    assert pr > 0.85 * 32


def test_pr_complete_collapse_is_one() -> None:
    """A rank-1 matrix has PR = 1 (only one nonzero singular value)."""
    torch.manual_seed(0)
    u = torch.randn(8192, 1)
    v = torch.randn(1, 32)
    z = u @ v
    pr = participation_ratio(z)
    assert math.isfinite(pr)
    assert pr == pytest.approx(1.0, abs=1e-3)


def test_pr_partial_collapse_low() -> None:
    """Rank-4 z plus tiny noise has PR close to 4."""
    torch.manual_seed(0)
    z_low_rank = torch.randn(8192, 4) @ torch.randn(4, 32)
    noise = 1e-6 * torch.randn(8192, 32)
    z = z_low_rank + noise
    pr = participation_ratio(z)
    assert math.isfinite(pr)
    assert 3.5 <= pr <= 5.0


def test_linear_probe_perfect_on_linear_data() -> None:
    """When c = A @ z + b + tiny noise, the probe is essentially exact."""
    torch.manual_seed(0)
    N, d, c_dim = 1024, 32, 3
    z = torch.randn(N, d)
    A = torch.randn(d, c_dim)
    b = torch.randn(c_dim)
    c = z @ A + b + 0.01 * torch.randn(N, c_dim)
    fit = torch.arange(0, 768)
    ev = torch.arange(768, N)
    r2 = linear_probe_r2(z, c, fit, ev)
    assert r2["r2_overall"] > 0.95
    assert r2["r2_G"] > 0.95


def test_linear_probe_zero_on_independent_data() -> None:
    """When c is independent of z, R^2 is near zero (slightly negative is OK)."""
    torch.manual_seed(0)
    N, d, c_dim = 4096, 32, 3
    z = torch.randn(N, d)
    c = torch.randn(N, c_dim)
    fit = torch.arange(0, 3000)
    ev = torch.arange(3000, N)
    r2 = linear_probe_r2(z, c, fit, ev)
    assert math.isfinite(r2["r2_overall"])
    assert -0.5 < r2["r2_overall"] < 0.1


def test_linear_probe_returns_named_keys() -> None:
    """For c_dim == 3 the keys are r2_G, r2_D, r2_Y, r2_overall."""
    torch.manual_seed(0)
    z = torch.randn(512, 16)
    c = torch.randn(512, 3)
    fit = torch.arange(0, 256)
    ev = torch.arange(256, 512)
    r2 = linear_probe_r2(z, c, fit, ev)
    assert set(r2.keys()) == {"r2_G", "r2_D", "r2_Y", "r2_overall"}


def test_variance_histogram_collapsed_z_concentrates_at_zero() -> None:
    """Most dims have zero variance; the histogram piles up in the first bin."""
    torch.manual_seed(0)
    z_active = torch.randn(1024, 4)
    z_zero = torch.zeros(1024, 28)
    z = torch.cat([z_active, z_zero], dim=1)
    counts, edges = per_dim_variance_histogram(z, n_bins=20)
    assert counts.shape == (20,)
    assert edges.shape == (21,)
    assert counts.sum().item() == 32
    assert counts[0].item() >= 28


def test_variance_histogram_isotropic_spreads_out() -> None:
    """For unit-variance Gaussian the bin near 1.0 has the most counts."""
    torch.manual_seed(0)
    z = torch.randn(8192, 32)
    counts, edges = per_dim_variance_histogram(z, n_bins=20)
    assert counts.sum().item() == 32
    centers = 0.5 * (edges[:-1] + edges[1:])
    mode_center = centers[int(torch.argmax(counts).item())].item()
    assert 0.7 <= mode_center <= 1.3
