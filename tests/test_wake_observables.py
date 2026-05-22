"""Tests for ``src/data/wake_observables.py``.

Verifies shape contracts, mode dimensionality, wake-mask containment of all
modes (no contribution from outside the wake ROI), and the standardization
roundtrip via :class:`WakeObservableStats`.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.wake_observables import (
    WakeObservableStats,
    compute_standardization_from_targets,
    compute_wake_observable,
    enstrophy_scalar_target,
    get_wake_mask_tensor,
    mode_output_dim,
    patch_signed_spectrum_target,
    patch_signed_target,
    radial_wake_spectrum_target,
    wake_coarse_pool_target,
)


def test_mode_output_dim_table() -> None:
    assert mode_output_dim("enstrophy_scalar") == 1
    assert mode_output_dim("patch_signed") == 64
    assert mode_output_dim("patch_signed_spectrum") == 80
    assert mode_output_dim("wake_coarse_pool") == 288
    with pytest.raises(ValueError, match="unknown wake mode"):
        mode_output_dim("nonsense")


def test_enstrophy_scalar_shape() -> None:
    x = torch.randn(8, 192, 96)
    y = enstrophy_scalar_target(x)
    assert y.shape == (8, 1)
    x5 = torch.randn(2, 4, 192, 96)
    y5 = enstrophy_scalar_target(x5)
    assert y5.shape == (2, 4, 1)
    assert (y5 >= 0).all(), "enstrophy_scalar must be non-negative (log1p of non-negative)"


def test_enstrophy_scalar_zero_outside_wake_does_not_change_target() -> None:
    """Pixels outside the wake mask do not enter the enstrophy."""
    H, W = 192, 96
    mask = get_wake_mask_tensor(H, W)
    x = torch.zeros(1, H, W)
    x_off = x.clone()
    x_off[..., :8, :] = 5.0  # near LE, fully outside wake (x < 0)
    x_off[..., :, :4] = 5.0  # bottom rows, outside wake (|y| >= 1.25)
    y = enstrophy_scalar_target(x)
    y_off = enstrophy_scalar_target(x_off)
    assert torch.allclose(y, y_off, atol=1e-6)


def test_patch_signed_shape_and_signs() -> None:
    x = torch.randn(5, 192, 96)
    y = patch_signed_target(x)
    assert y.shape == (5, 64)
    # First 32 dims come from positive vorticity energies, second 32 from negative.
    # Both halves must be non-negative (log1p of non-negative averages).
    assert (y >= 0).all()


def test_patch_signed_zero_on_zero_input() -> None:
    x = torch.zeros(3, 192, 96)
    y = patch_signed_target(x)
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-6)


def test_radial_wake_spectrum_shape_and_zero() -> None:
    x = torch.zeros(2, 192, 96)
    y = radial_wake_spectrum_target(x, n_bins=16)
    assert y.shape == (2, 16)
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-6)
    x = torch.randn(2, 192, 96)
    y = radial_wake_spectrum_target(x, n_bins=16)
    assert y.shape == (2, 16)
    assert (y >= 0).all()


def test_patch_signed_spectrum_concats_correctly() -> None:
    x = torch.randn(3, 192, 96)
    y = patch_signed_spectrum_target(x)
    assert y.shape == (3, 80)
    ps = patch_signed_target(x)
    rs = radial_wake_spectrum_target(x, n_bins=16)
    assert torch.allclose(y[:, :64], ps, atol=1e-6)
    assert torch.allclose(y[:, 64:], rs, atol=1e-6)


def test_wake_coarse_pool_shape_and_sign_preserved() -> None:
    x = torch.zeros(4, 192, 96)
    x[..., 60:70, 30:50] = +2.0  # positive blob inside the wake ROI
    x[..., 100:110, 30:50] = -2.0  # negative blob inside the wake ROI
    y = wake_coarse_pool_target(x)
    assert y.shape == (4, 288)
    # No log1p, so signs should be preserved -- the field has both + and -.
    assert (y > 0).any() and (y < 0).any()


def test_wake_coarse_pool_zero_outside_wake_does_not_change_target() -> None:
    x = torch.zeros(1, 192, 96)
    x[..., 60:70, 30:50] = 1.0
    x_off = x.clone()
    x_off[..., :20, :] = 9.9  # outside wake (x < 0)
    y = wake_coarse_pool_target(x)
    y_off = wake_coarse_pool_target(x_off)
    assert torch.allclose(y, y_off, atol=1e-6)


def test_compute_wake_observable_dispatches() -> None:
    x = torch.randn(2, 4, 192, 96)
    assert compute_wake_observable(x, "enstrophy_scalar").shape == (2, 4, 1)
    assert compute_wake_observable(x, "patch_signed").shape == (2, 4, 64)
    assert compute_wake_observable(x, "patch_signed_spectrum").shape == (2, 4, 80)
    assert compute_wake_observable(x, "wake_coarse_pool").shape == (2, 4, 288)
    with pytest.raises(ValueError, match="unknown wake mode"):
        compute_wake_observable(x, "bogus")


def test_standardization_roundtrip() -> None:
    rng = np.random.default_rng(0)
    targets = [rng.normal(loc=2.0, scale=3.0, size=(20, 8)).astype(np.float32)
               for _ in range(5)]
    stats = compute_standardization_from_targets(targets, mode="dummy", eps=1e-6)
    assert stats.mean.shape == (8,)
    assert stats.std.shape == (8,)
    # Apply on a held-out batch
    y = torch.from_numpy(rng.normal(loc=2.0, scale=3.0, size=(50, 8)).astype(np.float32))
    z = stats.standardize(y)
    # After standardization with train-pool stats the held-out should be ~N(0, 1)
    assert abs(z.mean().item()) < 0.5
    assert abs(z.std().item() - 1.0) < 0.3


def test_standardization_to_dict_from_dict_roundtrip() -> None:
    stats = WakeObservableStats(
        mode="patch_signed",
        mean=np.zeros(64, dtype=np.float32),
        std=np.ones(64, dtype=np.float32),
    )
    payload = stats.to_dict()
    restored = WakeObservableStats.from_dict(payload)
    assert restored.mode == "patch_signed"
    np.testing.assert_allclose(restored.mean, stats.mean)
    np.testing.assert_allclose(restored.std, stats.std)
