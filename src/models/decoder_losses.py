"""Loss functions for the multiscale visualisation decoder.

All losses are computed in NORMALISED space per the Session 9 omega
pipeline discipline (see CLAUDE.md "Omega preprocessing pipeline").
The training entrypoint un-normalises only for evaluation metrics
and figures.

Building blocks
---------------
- :func:`charbonnier` -- robust L1 surrogate (LapSRN, arXiv:1704.03915).
- :func:`region_weight` -- soft active-pixel mask + wake-ROI weight,
  used by the weighted MSE and the spatial enstrophy / circulation
  losses.
- :func:`weighted_mse` -- pixel-space MSE weighted by ``region_weight``.
- :func:`pyramid_residual_loss` -- per-level Charbonnier loss on the
  pyramid emitted by :class:`src.models.lap_film_decoder.LapFiLMDecoder`.
- :func:`local_focal_frequency_loss` -- patch-FFT focal frequency loss
  (Jiang et al. arXiv:2012.12821).
- :func:`enstrophy_field_loss` -- spatial L2 of pointwise enstrophy
  fields (D71: spatial-field comparison, NOT scalar-mean comparison).
- :func:`circulation_density_loss` -- spatial L1 of pointwise vorticity
  (signed) for circulation density.
- :func:`region_pyr_ffl_loss` -- combined loss used by the production
  runs E1, E2, and the optional E_noFiLM ablation.

Physics-correct enstrophy and circulation (D71)
-----------------------------------------------
The original collaborator proposal compared the SCALAR-MEAN enstrophy
(``pred.pow(2).mean()`` vs ``target.pow(2).mean()``), which is a global
integral that a model can satisfy trivially with uniform noise of the
right total energy. The correct physics constraint is the SPATIAL
enstrophy and circulation FIELDS, point by point. This module
implements the field-wise form; the unit test
``tests/test_decoder_losses.py::test_enstrophy_field_loss_nonzero_on_uniform_noise``
is the explicit regression check.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def charbonnier(x: Tensor, eps: float = 0.05) -> Tensor:
    """Charbonnier penalty ``sqrt(x^2 + eps^2) - eps``.

    Robust L1 surrogate that is differentiable at zero and behaves
    quadratically for ``|x| << eps`` and linearly for ``|x| >> eps``.
    Used in LapSRN (Lai et al., arXiv:1704.03915) for per-level
    super-resolution residuals.
    """
    return torch.sqrt(x * x + eps * eps) - eps


def _ensure_image_shape(t: Tensor) -> tuple[Tensor, tuple[int, ...]]:
    """Return ``(t.view(N, 1, H, W), original_shape)``.

    Accepts ``(B, 1, H, W)``, ``(B, T, 1, H, W)``, or ``(B, T, H, W)``.
    Reshapes everything to a 4D contiguous tensor for the pixel-level
    losses, returning the original shape so the caller can restore it
    if needed.
    """
    orig = t.shape
    if t.dim() == 4:
        return t, orig
    if t.dim() == 5:
        B, T, C, H, W = orig
        return t.reshape(B * T, C, H, W), orig
    if t.dim() == 3:
        B, H, W = orig
        return t.reshape(B, 1, H, W), orig
    raise ValueError(f"unsupported tensor shape {tuple(orig)}")


def region_weight(
    target_norm: Tensor,
    coord: Optional[dict[str, Tensor]] = None,
    solid_or_airfoil_mask: Optional[Tensor] = None,
    inactive_weight: float = 0.05,
    wake_weight: float = 0.50,
    active_tau: float = 0.10,
    active_softness: float = 0.03,
    wake_x_min: float = 0.0,
    wake_x_max: float = 4.5,
    wake_y_max: float = 1.25,
    physical_extent: tuple[float, float, float, float] = (-1.5, 4.5, -1.5, 1.5),
) -> Tensor:
    """Build a soft per-pixel weight map for region-weighted losses.

    The weight combines two cues:

    1. **Active-pixel soft mask.** ``sigmoid((|target| - tau) / softness)``
       in normalised space. Pixels with non-trivial vorticity get
       weight close to 1; freestream pixels get the ``inactive_weight``
       floor (default 0.05, never zero, since a hard mask makes the
       freestream diverge per Session 9 D60).
    2. **Wake ROI bonus.** A flat ``+wake_weight`` added inside the
       rectangular wake region defined by ``x in (wake_x_min, wake_x_max)``
       and ``|y| < wake_y_max``. Coordinates are inferred from
       ``physical_extent = (x_min, x_max, y_min, y_max)`` unless an
       explicit ``coord`` dict is supplied.

    If ``solid_or_airfoil_mask`` is provided (1 inside solid / adjacent,
    0 elsewhere), those pixels are zeroed in the output weight.

    The output is normalised so its non-zero mean is ~1 (the global
    mean across all pixels, including zeroed solid cells, may differ
    when a solid mask is provided).

    Returns a tensor with the same spatial shape as ``target_norm`` and
    one channel (broadcast-ready). ``coord`` and ``solid_or_airfoil_mask``
    must be broadcast-compatible with the spatial dims of ``target_norm``.
    """
    t4, orig = _ensure_image_shape(target_norm)
    N, _, H, W = t4.shape

    abs_t = t4.abs()
    active_soft = torch.sigmoid((abs_t - active_tau) / max(active_softness, 1e-6))
    weight = active_soft * (1.0 - inactive_weight) + inactive_weight

    if coord is None:
        x_min, x_max, y_min, y_max = physical_extent
        xs = torch.linspace(x_min, x_max, W, device=t4.device, dtype=t4.dtype)
        ys = torch.linspace(y_min, y_max, H, device=t4.device, dtype=t4.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        x_grid = xx[None, None, :, :]
        y_grid = yy[None, None, :, :]
    else:
        x_grid = coord["x"]
        y_grid = coord["y"]

    in_wake = (
        (x_grid > wake_x_min)
        & (x_grid < wake_x_max)
        & (y_grid.abs() < wake_y_max)
    ).to(weight.dtype)
    weight = weight + in_wake * wake_weight

    if solid_or_airfoil_mask is not None:
        mask = solid_or_airfoil_mask.to(weight.dtype)
        if mask.dim() == 2:
            mask = mask[None, None]
        elif mask.dim() == 3:
            mask = mask[None]
        weight = weight * (1.0 - mask)

    mean = weight.mean().clamp_min(1e-8)
    weight = weight / mean

    if len(orig) == 5:
        B, T = orig[0], orig[1]
        weight = weight.reshape(B, T, 1, H, W)
    return weight


def weighted_mse(pred: Tensor, target: Tensor, weight: Tensor) -> Tensor:
    """``mean(weight * (pred - target)^2)``."""
    return (weight * (pred - target).pow(2)).mean()


def pyramid_residual_loss(
    pred_pyr: list[Tensor],
    target: Tensor,
    eps: float = 0.05,
    level_weights: Optional[tuple[float, ...]] = None,
) -> Tensor:
    """Sum of Charbonnier losses on per-level predictions vs downsampled targets.

    Level weights default to ``[0.10, 0.20, 0.40, 0.80, 1.00]`` coarse-to-fine
    for the canonical 5-level pyramid; ``(1.0,)`` for single-level pyramids
    (e.g. when this is called on a non-pyramid decoder such as the
    coordinate-MLP audit); and a geometric ``2^-(n-1-k)`` schedule for
    other lengths.
    """
    n = len(pred_pyr)
    if level_weights is None:
        canonical = (0.10, 0.20, 0.40, 0.80, 1.00)
        if n == len(canonical):
            level_weights = canonical
        elif n == 1:
            level_weights = (1.0,)
        else:
            level_weights = tuple(2.0 ** -(n - 1 - k) for k in range(n))
    if len(level_weights) != n:
        raise ValueError(
            f"level_weights len {len(level_weights)} != pyramid depth {n}"
        )

    target_4d, _ = _ensure_image_shape(target)
    losses: list[Tensor] = []
    for pred_k, w_k in zip(pred_pyr, level_weights):
        pred_k_4d, _ = _ensure_image_shape(pred_k)
        h_k, w_k_dim = pred_k_4d.shape[-2:]
        target_k = F.adaptive_avg_pool2d(target_4d, output_size=(h_k, w_k_dim))
        residual = pred_k_4d - target_k
        losses.append(w_k * charbonnier(residual, eps).mean())
    return sum(losses)


def local_focal_frequency_loss(
    pred: Tensor,
    target: Tensor,
    patch: int = 32,
    alpha: float = 1.0,
    eps: float = 1e-8,
) -> Tensor:
    """Per-patch Focal Frequency Loss (Jiang et al., arXiv:2012.12821).

    For each non-overlapping spatial patch of size ``patch``:

        F_pred  = FFT(pred_patch);   F_target = FFT(target_patch)
        diff    = F_pred - F_target  (complex)
        w       = (|diff| ** alpha).detach()         -- focal weight
        w       = w / w.mean()                       -- per-patch normalise
        loss    = mean( w * |diff|^2 )

    The detach ensures the focal weight does not propagate gradients
    (the "focus" is a re-weighting, not a moving target). The per-patch
    normalisation keeps the loss scale stable when the patch mean is
    very small (eg uniform freestream patches), with a floor at ``eps``.
    """
    pred_4d, _ = _ensure_image_shape(pred)
    target_4d, _ = _ensure_image_shape(target)
    N, C, H, W = pred_4d.shape
    if H % patch != 0 or W % patch != 0:
        raise ValueError(
            f"patch {patch} does not tile spatial dims ({H}, {W}) exactly"
        )
    nH, nW = H // patch, W // patch

    pred_p = pred_4d.reshape(N, C, nH, patch, nW, patch).permute(
        0, 2, 4, 1, 3, 5
    ).reshape(N * nH * nW, C, patch, patch)
    target_p = target_4d.reshape(N, C, nH, patch, nW, patch).permute(
        0, 2, 4, 1, 3, 5
    ).reshape(N * nH * nW, C, patch, patch)

    F_pred = torch.fft.fft2(pred_p, norm="ortho")
    F_target = torch.fft.fft2(target_p, norm="ortho")
    diff = F_pred - F_target
    diff_mag2 = diff.real.pow(2) + diff.imag.pow(2)
    diff_mag = diff_mag2.clamp_min(eps).sqrt()

    with torch.no_grad():
        w = diff_mag.pow(alpha)
        w_mean = w.mean(dim=(-2, -1), keepdim=True).clamp_min(eps)
        w = w / w_mean

    return (w * diff_mag2).mean()


def enstrophy_field_loss(
    pred: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
) -> Tensor:
    """Spatial L2 of pointwise enstrophy fields (D71).

    Enstrophy density is ``omega^2`` at each point. The loss is the
    weighted mean of ``(pred^2 - target^2)^2`` over the field. This is
    NOT the same as ``(mean(pred^2) - mean(target^2))^2`` (the scalar
    integral comparison the collaborator's proposal used by mistake): a
    model producing uniform noise of the right total enstrophy would
    pass the integral comparison but fail this field comparison.

    ``weight``, if provided, is the output of :func:`region_weight` and
    restricts the constraint to the wake / active region (where the
    enstrophy is physically meaningful). Freestream pixels then enter
    with the small ``inactive_weight`` floor and the loss is not
    dominated by their tiny but ubiquitous numerical noise.
    """
    diff = pred.pow(2) - target.pow(2)
    if weight is None:
        return diff.pow(2).mean()
    return (weight * diff.pow(2)).mean()


def circulation_density_loss(
    pred: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
) -> Tensor:
    """Spatial L1 of pointwise vorticity (signed; D71).

    Circulation density is just the signed vorticity field at each
    point: ``Gamma = integral(omega dA)`` over a small region tends to
    the local omega value as the region shrinks. The L1 metric is
    appropriate because the sign matters (clockwise vs counter-clockwise
    vortex cores cancel under L2 but not L1).

    Like :func:`enstrophy_field_loss`, this is a per-pixel field
    comparison; comparing the integrated ``mean(pred)`` to ``mean(target)``
    would let a model satisfy the constraint with uniform noise.
    """
    diff = pred - target
    if weight is None:
        return diff.abs().mean()
    return (weight * diff.abs()).mean()


def region_pyr_ffl_loss(
    pred_pyr: list[Tensor],
    target: Tensor,
    coord: Optional[dict[str, Tensor]] = None,
    solid_or_airfoil_mask: Optional[Tensor] = None,
    lambda_region: float = 1.0,
    lambda_pyramid: float = 0.4,
    lambda_ffl: float = 0.05,
    lambda_enstrophy: float = 0.02,
    lambda_circulation: float = 0.01,
    ffl_alpha: float = 1.0,
    ffl_patch: int = 32,
    ffl_warmup_factor: float = 1.0,
    charbonnier_eps: float = 0.05,
    region_kwargs: Optional[dict] = None,
) -> dict[str, Tensor]:
    """Combined loss used by Session 10 production runs.

    Args:
        pred_pyr: List of per-pyramid-level predictions from
            :class:`LapFiLMDecoder`. The last entry is the 192x96 final
            prediction.
        target: Ground-truth omega in normalised space, shape
            ``(B, 1, H, W)`` or ``(B, T, 1, H, W)``.
        coord: Optional ``{"x": ..., "y": ...}`` physical-coordinate
            grids in convective-time units. Passed to ``region_weight``
            for the wake-ROI determination.
        solid_or_airfoil_mask: Optional 2D float mask of the
            airfoil-adjacent cells. Same convention as
            ``outputs/data_pipeline/v1/airfoil_adjacent_mask.npy``.
        lambda_region, ...: Loss component weights. The defaults
            match the Session 10 plan Step 2.
        ffl_warmup_factor: External multiplier on the FFL component,
            ramped from 0 to 1 by the training loop over the warmup
            window so the decoder learns the gust core before being
            asked to match high-frequency wake structure.
        charbonnier_eps: Smoothing constant for the per-level pyramid
            Charbonnier residuals.
        region_kwargs: Optional overrides for :func:`region_weight`.

    Returns: dict with keys ``L_total``, ``L_region``, ``L_pyramid``,
    ``L_ffl``, ``L_enstrophy``, ``L_circulation``. ``L_ffl`` is the
    UNWEIGHTED FFL value (the ``lambda_ffl * ffl_warmup_factor``
    contribution to ``L_total`` is computed inside).
    """
    final_pred = pred_pyr[-1]
    rk = region_kwargs or {}
    weight = region_weight(target, coord=coord,
                           solid_or_airfoil_mask=solid_or_airfoil_mask, **rk)

    L_region = weighted_mse(final_pred, target, weight)
    L_pyramid = pyramid_residual_loss(pred_pyr, target, eps=charbonnier_eps)
    L_ffl = local_focal_frequency_loss(final_pred, target,
                                       patch=ffl_patch, alpha=ffl_alpha)
    L_enstrophy = enstrophy_field_loss(final_pred, target, weight=weight)
    L_circulation = circulation_density_loss(final_pred, target, weight=weight)

    L_total = (
        lambda_region * L_region
        + lambda_pyramid * L_pyramid
        + lambda_ffl * ffl_warmup_factor * L_ffl
        + lambda_enstrophy * L_enstrophy
        + lambda_circulation * L_circulation
    )

    return {
        "L_total": L_total,
        "L_region": L_region,
        "L_pyramid": L_pyramid,
        "L_ffl": L_ffl,
        "L_enstrophy": L_enstrophy,
        "L_circulation": L_circulation,
    }
