"""Near-body lift-element observable targets (Session 34, Track C).

Produces per-frame 80-D near-body observable targets from the band-weighted,
scaled Chang lift-element field (see ``src.data.lift_element``):

    field = band * e / E_SCALE,   e = omega_z * (-v dphi/dx + u dphi/dy)

Mode ``nearbody_lift_element`` (80 dim) mirrors the wake Mode C
(``patch_signed_spectrum``) byte-for-byte in structure so the N and W heads
have identical capacity and loss geometry:

- 64 dim -- 8x4 = 32 patches over the near-body band bounding box; each patch
  contributes two energies ``log1p(mean(relu(+/-field)^2))`` (sign-preserving:
  lift-generating vs lift-reducing structure stays separate).
- 16 dim -- radial spatial-wavenumber spectrum of the (already band-masked)
  field, Hann-windowed, ``log1p`` compressed.

The field is a deterministic per-frame function of the raw mid-span
``(omega_z, u, v)``, so targets are precomputed once per encounter by
``scripts/session34/precompute_nearbody_observables.py`` into
``${VORTEX_JEPA_CACHE}/{partition}/nearbody_observables/`` with train-only
standardization stats (the exact machinery of the wake observables cache;
``WakeObservableStats`` is reused unchanged).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from src.data.lift_element import DEFAULT_ADJACENT_MASK_PATH, build_nearbody_band
from src.data.wake_observables import (
    WakeObservableStats,
    _ensure_4d,
    _hann_window_2d,
    _radial_bin_indices,
    _restore_leading,
    compute_standardization_from_targets,
)

__all__ = [
    "mode_output_dim",
    "get_nearbody_band_tensor",
    "nearbody_roi_window",
    "nearbody_patch_signed_target",
    "nearbody_radial_spectrum_target",
    "nearbody_lift_element_target",
    "compute_nearbody_observable",
    "WakeObservableStats",
    "compute_standardization_from_targets",
]

_MODE_DIM = {
    "nearbody_lift_element": 80,
}

_BAND_CACHE: dict[tuple[str, float], np.ndarray] = {}


def mode_output_dim(mode: str) -> int:
    """Return the output dimensionality for a near-body observable mode."""
    if mode not in _MODE_DIM:
        raise ValueError(f"unknown nearbody mode {mode!r}; choose from {list(_MODE_DIM)}")
    return _MODE_DIM[mode]


def get_nearbody_band(
    mask_path: Path = DEFAULT_ADJACENT_MASK_PATH, delta_n: float = 0.3
) -> np.ndarray:
    """Module-cached ``(H, W)`` float32 near-body band weight."""
    key = (str(mask_path), float(delta_n))
    if key not in _BAND_CACHE:
        adjacent = np.load(Path(mask_path))
        _BAND_CACHE[key] = build_nearbody_band(adjacent, delta_n=delta_n)
    return _BAND_CACHE[key]


def get_nearbody_band_tensor(
    mask_path: Path = DEFAULT_ADJACENT_MASK_PATH,
    delta_n: float = 0.3,
    device=None,
    dtype=torch.float32,
) -> Tensor:
    """Near-body band weight as a torch tensor of shape ``(H, W)``."""
    t = torch.from_numpy(get_nearbody_band(mask_path, delta_n)).to(dtype=dtype)
    if device is not None:
        t = t.to(device)
    return t


def nearbody_roi_window(band: Optional[np.ndarray] = None) -> tuple[int, int, int, int]:
    """Return ``(row_start, row_end, col_start, col_end)`` bounding the band support."""
    if band is None:
        band = get_nearbody_band()
    rows = np.where((band > 0).any(axis=1))[0]
    cols = np.where((band > 0).any(axis=0))[0]
    return int(rows[0]), int(rows[-1] + 1), int(cols[0]), int(cols[-1] + 1)


def _roi_crop(x: Tensor, band: Optional[np.ndarray] = None) -> Tensor:
    r0, r1, c0, c1 = nearbody_roi_window(band)
    return x[..., r0:r1, c0:c1]


def nearbody_patch_signed_target(field: Tensor, ph: int = 8, pw: int = 4) -> Tensor:
    """64-D sign-preserving patch energies of the band-weighted element field.

    Args:
        field: ``(T, H, W)`` or ``(B, T, H, W)`` band-weighted scaled
            lift-element field (``band * e / E_SCALE``).
        ph, pw: patch-grid resolution (default ``(8, 4)``, matching the wake
            Mode B convention).

    Returns:
        ``(..., 2 * ph * pw)`` non-negative tensor (``log1p`` applied).
    """
    leading = field.shape[:-2]
    x, was_5d = _ensure_4d(field)
    x_roi = _roi_crop(x)
    pos = F.relu(x_roi) ** 2
    neg = F.relu(-x_roi) ** 2
    pos_p = F.adaptive_avg_pool2d(pos.unsqueeze(1), output_size=(ph, pw)).squeeze(1)
    neg_p = F.adaptive_avg_pool2d(neg.unsqueeze(1), output_size=(ph, pw)).squeeze(1)
    pos_p = torch.log1p(pos_p).reshape(pos_p.shape[0], -1)
    neg_p = torch.log1p(neg_p).reshape(neg_p.shape[0], -1)
    out = torch.cat([pos_p, neg_p], dim=-1)
    return _restore_leading(out, was_5d, leading)


def nearbody_radial_spectrum_target(field: Tensor, n_bins: int = 16) -> Tensor:
    """16-bin radial power spectrum of the (band-masked) field, Hann-windowed.

    The band weighting is already part of ``field``; only the Hann window is
    applied here (mirroring the wake spectrum, whose mask is applied at the
    same point in the pipeline).
    """
    leading = field.shape[:-2]
    x, was_5d = _ensure_4d(field)
    H, W = x.shape[-2], x.shape[-1]
    han = _hann_window_2d(H, W, device=x.device, dtype=x.dtype)
    windowed = x * han
    F2 = torch.fft.rfft2(windowed)
    power = F2.real ** 2 + F2.imag ** 2
    bins = _radial_bin_indices(H, W, n_bins).to(x.device)
    bins_flat = bins.reshape(-1)
    power_flat = power.reshape(power.shape[0], -1)
    out = power.new_zeros((power.shape[0], n_bins))
    out.scatter_add_(1, bins_flat.unsqueeze(0).expand(power.shape[0], -1), power_flat)
    counts = power.new_zeros(n_bins).scatter_add_(0, bins_flat, torch.ones_like(power_flat[0]))
    out = out / counts.clamp_min(1.0)
    out = torch.log1p(out)
    return _restore_leading(out, was_5d, leading)


def nearbody_lift_element_target(field: Tensor) -> Tensor:
    """Mode ``nearbody_lift_element``: 64-D patches + 16-bin spectrum = 80 dim."""
    ps = nearbody_patch_signed_target(field)
    rs = nearbody_radial_spectrum_target(field, n_bins=16)
    return torch.cat([ps, rs], dim=-1)


_TARGET_FNS = {
    "nearbody_lift_element": nearbody_lift_element_target,
}


def compute_nearbody_observable(field: Tensor, mode: str) -> Tensor:
    """Dispatch to the right per-frame near-body target function."""
    if mode not in _TARGET_FNS:
        raise ValueError(f"unknown nearbody mode {mode!r}; choose from {list(_TARGET_FNS)}")
    return _TARGET_FNS[mode](field)
