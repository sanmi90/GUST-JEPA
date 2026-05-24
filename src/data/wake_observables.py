"""Wake-region observable target preprocessors for Session 11.

Produces per-frame wake observable targets from pipeline-normalized
``omega_z`` (shape ``(T, H, W)`` or ``(B, T, H, W)`` with ``H=192, W=96``).

Five target modes (Session 11 plan, "Track 1" / "Track 2"; Session 12 Dir D):

- Mode A ``enstrophy_scalar``     (1 dim) -- ``log1p(mean(omega^2))`` over the wake ROI.
- Mode B ``patch_signed``        (64 dim) -- 8x4 = 32 patches over the wake ROI;
                                              each patch contributes two energies
                                              ``log1p(mean(relu(+/-omega)^2))``.
- Mode C ``patch_signed_spectrum`` (80 dim) -- Mode B plus a 16-bin radial spectrum
                                              of the wake-masked field, Hann-windowed.
- Mode D ``wake_coarse_pool``    (288 dim) -- 24x12 adaptive average-pooled wake-ROI
                                              field, flattened.
- Mode E ``wake_coarse_pool_32x16`` (512 dim) -- 32x16 adaptive average-pooled wake-ROI
                                              field, flattened (Session 12 Direction D
                                              higher-resolution sibling of Mode D).

All five modes:
1. Restrict to the wake ROI defined in ``src.evaluation.decoder_metrics``
   (x in [0, 4.5], |y| < 1.25); other pixels are zeroed.
2. Operate on NORMALIZED omega (the omega-pipeline output space) so the
   gradient scale matches the JEPA encoder's bf16 autocast regime
   (CLAUDE.md Session 9 D71 / "Omega preprocessing pipeline").
3. Are standardized at training time via train-only mean/std stats
   produced by :func:`compute_train_standardization`.

Reference: SESSION11_WAKE_RESULTS_FIRST.md "Wake observable target definitions".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from src.evaluation.decoder_metrics import wake_mask


_MODE_DIM = {
    "enstrophy_scalar": 1,
    "patch_signed": 64,
    "patch_signed_spectrum": 80,
    "wake_coarse_pool": 288,
    "wake_coarse_pool_32x16": 512,
}


def mode_output_dim(mode: str) -> int:
    """Return the output dimensionality for a wake observable mode."""
    if mode not in _MODE_DIM:
        raise ValueError(f"unknown wake mode {mode!r}; choose from {list(_MODE_DIM)}")
    return _MODE_DIM[mode]


def get_wake_mask_tensor(H: int = 192, W: int = 96, device=None, dtype=torch.float32) -> Tensor:
    """Boolean wake-ROI mask as a torch tensor of shape ``(H, W)``."""
    mask_np = wake_mask(H, W).astype(np.float32)
    t = torch.from_numpy(mask_np).to(dtype=dtype)
    if device is not None:
        t = t.to(device)
    return t


def _ensure_4d(omega_norm: Tensor) -> tuple[Tensor, bool]:
    """Promote ``(T, H, W)`` or ``(B, T, H, W)`` to ``(N=B*T, H, W)`` -- internal helper.

    Returns ``(flat, was_5d)``. The caller restores the leading shape after
    the per-frame computation.
    """
    if omega_norm.dim() == 3:
        return omega_norm, False
    if omega_norm.dim() == 4:
        B, T, H, W = omega_norm.shape
        return omega_norm.reshape(B * T, H, W), True
    raise ValueError(
        f"omega_norm must be (T, H, W) or (B, T, H, W); got {tuple(omega_norm.shape)}"
    )


def _restore_leading(out: Tensor, was_5d: bool, leading: tuple[int, ...]) -> Tensor:
    if not was_5d:
        return out
    return out.reshape(*leading, out.shape[-1])


def enstrophy_scalar_target(omega_norm: Tensor) -> Tensor:
    """Mode A: ``log1p(mean(omega^2))`` over the wake ROI.

    Args:
        omega_norm: ``(T, H, W)`` or ``(B, T, H, W)`` normalized omega.

    Returns:
        ``(T, 1)`` or ``(B, T, 1)`` tensor with a single non-negative scalar
        per frame.
    """
    leading = omega_norm.shape[:-2]
    x, was_5d = _ensure_4d(omega_norm)
    H, W = x.shape[-2], x.shape[-1]
    mask = get_wake_mask_tensor(H, W, device=x.device, dtype=x.dtype)
    sq = (x * mask) ** 2
    mean = sq.sum(dim=(-2, -1)) / max(mask.sum().item(), 1.0)
    out = torch.log1p(mean).unsqueeze(-1)
    return _restore_leading(out, was_5d, leading)


def _wake_roi_window() -> tuple[int, int, int, int]:
    """Return (row_start, row_end, col_start, col_end) for the wake ROI.

    The wake mask defined in ``src.evaluation.decoder_metrics`` selects
    rows where ``x > 0`` (~rows 48..191) and cols where ``|y| < 1.25``
    (~cols 8..87). We use the exact mask bounding box for patch slicing
    so all four modes share the same physical extent.
    """
    H, W = 192, 96
    m = wake_mask(H, W)
    rows = np.where(m.any(axis=1))[0]
    cols = np.where(m.any(axis=0))[0]
    return int(rows[0]), int(rows[-1] + 1), int(cols[0]), int(cols[-1] + 1)


def _roi_crop(x: Tensor) -> Tensor:
    """Crop ``(N, H, W)`` to the wake ROI bounding box."""
    r0, r1, c0, c1 = _wake_roi_window()
    return x[..., r0:r1, c0:c1]


def patch_signed_target(omega_norm: Tensor, ph: int = 8, pw: int = 4) -> Tensor:
    """Mode B: ``2 * ph * pw`` patch energies (positive + negative).

    The wake ROI is cropped to a ``(roi_h, roi_w)`` box, mean-pool reduced
    to ``(ph, pw)`` for each sign separately via ``adaptive_avg_pool2d``
    on ``relu(+omega)^2`` and ``relu(-omega)^2``.

    Args:
        omega_norm: ``(T, H, W)`` or ``(B, T, H, W)`` normalized omega.
        ph, pw: Patch-grid resolution. Default ``(8, 4)`` per the spec.

    Returns:
        ``(..., 2 * ph * pw)`` non-negative tensor (``log1p`` applied).
    """
    leading = omega_norm.shape[:-2]
    x, was_5d = _ensure_4d(omega_norm)
    x_roi = _roi_crop(x)
    pos = F.relu(x_roi) ** 2
    neg = F.relu(-x_roi) ** 2
    pos_p = F.adaptive_avg_pool2d(pos.unsqueeze(1), output_size=(ph, pw)).squeeze(1)
    neg_p = F.adaptive_avg_pool2d(neg.unsqueeze(1), output_size=(ph, pw)).squeeze(1)
    pos_p = torch.log1p(pos_p).reshape(pos_p.shape[0], -1)
    neg_p = torch.log1p(neg_p).reshape(neg_p.shape[0], -1)
    out = torch.cat([pos_p, neg_p], dim=-1)
    return _restore_leading(out, was_5d, leading)


def _radial_bin_indices(h: int, w: int, n_bins: int) -> Tensor:
    """Per-pixel radial bin index in [0, n_bins) on the rfft grid (h, w//2+1)."""
    ky = torch.fft.fftfreq(h) * h
    kx = torch.fft.rfftfreq(w) * w
    kyy, kxx = torch.meshgrid(ky, kx, indexing="ij")
    r = torch.sqrt(kxx ** 2 + kyy ** 2)
    r_max = float(r.max().item())
    if r_max <= 0:
        return torch.zeros_like(r, dtype=torch.long)
    bins = torch.clamp((r / r_max * n_bins).long(), max=n_bins - 1)
    return bins


def _hann_window_2d(h: int, w: int, device, dtype) -> Tensor:
    hy = torch.hann_window(h, periodic=False, device=device, dtype=dtype)
    hx = torch.hann_window(w, periodic=False, device=device, dtype=dtype)
    return hy.unsqueeze(1) * hx.unsqueeze(0)


def radial_wake_spectrum_target(omega_norm: Tensor, n_bins: int = 16) -> Tensor:
    """Mode C component: ``log1p`` of the ``n_bins``-bin radial power spectrum
    of the wake-masked field (Hann-windowed before FFT).

    Returns:
        ``(..., n_bins)`` tensor.
    """
    leading = omega_norm.shape[:-2]
    x, was_5d = _ensure_4d(omega_norm)
    H, W = x.shape[-2], x.shape[-1]
    mask = get_wake_mask_tensor(H, W, device=x.device, dtype=x.dtype)
    han = _hann_window_2d(H, W, device=x.device, dtype=x.dtype)
    field = x * mask * han
    F2 = torch.fft.rfft2(field)
    power = (F2.real ** 2 + F2.imag ** 2)
    bins = _radial_bin_indices(H, W, n_bins).to(x.device)
    bins_flat = bins.reshape(-1)
    power_flat = power.reshape(power.shape[0], -1)
    out = power.new_zeros((power.shape[0], n_bins))
    out.scatter_add_(1, bins_flat.unsqueeze(0).expand(power.shape[0], -1), power_flat)
    counts = power.new_zeros(n_bins).scatter_add_(0, bins_flat, torch.ones_like(power_flat[0]))
    out = out / counts.clamp_min(1.0)
    out = torch.log1p(out)
    return _restore_leading(out, was_5d, leading)


def patch_signed_spectrum_target(omega_norm: Tensor) -> Tensor:
    """Mode C: Mode B 64-dim plus 16-bin radial wake spectrum = 80 dim."""
    ps = patch_signed_target(omega_norm)
    rs = radial_wake_spectrum_target(omega_norm, n_bins=16)
    return torch.cat([ps, rs], dim=-1)


def wake_coarse_pool_target(omega_norm: Tensor, out_h: int = 24, out_w: int = 12) -> Tensor:
    """Mode D: wake-ROI omega downsampled to ``(out_h, out_w)`` via mean pooling.

    The wake mask is applied first, then the masked field is pooled. Returns
    a signed tensor (no relu) so the network has access to vorticity sign;
    no log1p so the dynamic range is preserved.
    """
    leading = omega_norm.shape[:-2]
    x, was_5d = _ensure_4d(omega_norm)
    H, W = x.shape[-2], x.shape[-1]
    mask = get_wake_mask_tensor(H, W, device=x.device, dtype=x.dtype)
    field = x * mask
    field_roi = _roi_crop(field)
    pooled = F.adaptive_avg_pool2d(field_roi.unsqueeze(1), output_size=(out_h, out_w)).squeeze(1)
    out = pooled.reshape(pooled.shape[0], -1)
    return _restore_leading(out, was_5d, leading)


def wake_coarse_pool_32x16_target(
    omega_norm: Tensor, out_h: int = 32, out_w: int = 16
) -> Tensor:
    """Mode E: wake-ROI omega downsampled to ``(32, 16) = 512`` via mean pooling.

    Higher-resolution sibling of :func:`wake_coarse_pool_target` introduced by
    Session 12 Direction D to test whether a 512-D pool target lets the
    encoder encode more spatial wake detail. Mirrors Mode D exactly: wake
    mask applied first, then ``adaptive_avg_pool2d`` over the ROI bounding
    box; signed (no relu) and no log1p to preserve dynamic range.
    """
    leading = omega_norm.shape[:-2]
    x, was_5d = _ensure_4d(omega_norm)
    H, W = x.shape[-2], x.shape[-1]
    mask = get_wake_mask_tensor(H, W, device=x.device, dtype=x.dtype)
    field = x * mask
    field_roi = _roi_crop(field)
    pooled = F.adaptive_avg_pool2d(field_roi.unsqueeze(1), output_size=(out_h, out_w)).squeeze(1)
    out = pooled.reshape(pooled.shape[0], -1)
    return _restore_leading(out, was_5d, leading)


_TARGET_FNS = {
    "enstrophy_scalar": enstrophy_scalar_target,
    "patch_signed": patch_signed_target,
    "patch_signed_spectrum": patch_signed_spectrum_target,
    "wake_coarse_pool": wake_coarse_pool_target,
    "wake_coarse_pool_32x16": wake_coarse_pool_32x16_target,
}


def compute_wake_observable(omega_norm: Tensor, mode: str) -> Tensor:
    """Dispatch to the right per-frame wake target function."""
    if mode not in _TARGET_FNS:
        raise ValueError(f"unknown wake mode {mode!r}; choose from {list(_TARGET_FNS)}")
    return _TARGET_FNS[mode](omega_norm)


@dataclass
class WakeObservableStats:
    """Train-only standardization stats for a wake observable mode."""

    mode: str
    mean: np.ndarray
    std: np.ndarray
    eps: float = 1e-6

    def standardize(self, y: Tensor) -> Tensor:
        """Apply ``(y - mean) / (std + eps)`` along the last dim."""
        mean_t = torch.as_tensor(self.mean, dtype=y.dtype, device=y.device)
        std_t = torch.as_tensor(self.std, dtype=y.dtype, device=y.device)
        return (y - mean_t) / (std_t + self.eps)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "eps": self.eps,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "WakeObservableStats":
        return cls(
            mode=payload["mode"],
            mean=np.asarray(payload["mean"], dtype=np.float32),
            std=np.asarray(payload["std"], dtype=np.float32),
            eps=float(payload.get("eps", 1e-6)),
        )


def compute_standardization_from_targets(
    targets: list[np.ndarray], mode: str, eps: float = 1e-6
) -> WakeObservableStats:
    """Compute per-dim mean / std across a list of target tensors.

    Each tensor in ``targets`` is shaped ``(T_k, out_dim)`` -- the per-frame
    targets of one training encounter. Stats are pooled across all frames
    of all encounters.
    """
    stacked = np.concatenate(targets, axis=0).astype(np.float64)
    return WakeObservableStats(
        mode=mode,
        mean=stacked.mean(axis=0).astype(np.float32),
        std=stacked.std(axis=0).astype(np.float32),
        eps=eps,
    )
