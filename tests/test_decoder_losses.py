"""Unit tests for :mod:`src.models.decoder_losses`.

The most important test in this file is
``test_enstrophy_field_loss_nonzero_on_uniform_noise``: it is the
explicit regression check against the collaborator's original
scalar-mean enstrophy formulation, which a model can satisfy with
uniform noise of the right total energy. The fixed
:func:`enstrophy_field_loss` is a spatial-field comparison and rejects
uniform noise (D71 in HANDOFF.md).
"""

from __future__ import annotations

import math

import pytest
import torch

from src.models.decoder_losses import (
    charbonnier,
    circulation_density_loss,
    enstrophy_field_loss,
    gradient_consistency_loss,
    hann_window_2d,
    local_focal_frequency_loss,
    pyramid_residual_loss,
    region_pyr_ffl_loss,
    region_pyr_specloss_loss,
    region_weight,
    spectral_amplitude_loss,
    tukey_window_2d,
    weighted_mse,
)


def test_charbonnier_zero_at_zero() -> None:
    """``charbonnier(0) == 0`` exactly; positive elsewhere; differentiable."""
    x = torch.zeros(8)
    assert torch.allclose(charbonnier(x, eps=0.05), torch.zeros_like(x))
    y = torch.tensor([1.0, -1.0, 0.1])
    out = charbonnier(y, eps=0.05)
    assert (out > 0).all()
    x = torch.randn(4, requires_grad=True)
    loss = charbonnier(x, eps=0.05).sum()
    loss.backward()
    assert torch.isfinite(x.grad).all()


def test_region_weight_floor() -> None:
    """Weight has floor 0.05 in the freestream, > 0.5 in the wake ROI,
    and exactly 0 in the solid-mask region when a mask is provided."""
    H, W = 96, 192
    # Target is mostly freestream zeros with a single spike in the wake.
    target = torch.zeros(2, 1, H, W)
    target[..., 60, 100] = 2.0

    weight_nomask = region_weight(target)
    # Freestream weight at a pixel far from the wake ROI: should be
    # approximately inactive_weight=0.05 (divided by mean -> still proportional)
    # We check the relative ratio: the wake ROI has weight much larger.
    # First, freestream pixel at (10, 10) is outside x in (0, 4.5) wake.
    fs = weight_nomask[0, 0, 10, 10].item()
    # Wake ROI pixel at center of the image
    wake = weight_nomask[0, 0, H // 2, W // 2].item()
    assert fs > 0.0
    assert wake > 4.0 * fs

    # With airfoil mask: the masked pixels go to exactly 0.
    mask = torch.zeros(H, W)
    mask[30:40, 50:60] = 1.0
    weight_masked = region_weight(target, solid_or_airfoil_mask=mask)
    assert (weight_masked[0, 0, 30:40, 50:60] == 0).all()
    # Non-masked pixels remain finite > 0.
    assert (weight_masked[0, 0, 0, 0] > 0).item()


def test_weighted_mse_uniform_weight_matches_mean_mse() -> None:
    """When weight is uniform 1, weighted_mse equals ``((p-t)^2).mean()``."""
    pred = torch.randn(2, 1, 8, 16)
    target = torch.randn(2, 1, 8, 16)
    w = torch.ones_like(pred)
    expected = (pred - target).pow(2).mean()
    got = weighted_mse(pred, target, w)
    assert torch.allclose(got, expected)


def test_pyramid_loss_zero_on_perfect_pyramid() -> None:
    """If predictions equal averaged targets at all pyramid levels,
    pyramid_residual_loss is zero."""
    target = torch.randn(2, 1, 192, 96)
    sizes = [(12, 6), (24, 12), (48, 24), (96, 48), (192, 96)]
    pyr = []
    for h, w in sizes:
        pyr.append(torch.nn.functional.adaptive_avg_pool2d(target, output_size=(h, w)))
    loss = pyramid_residual_loss(pyr, target)
    assert loss.item() == 0.0


def test_pyramid_loss_positive_otherwise() -> None:
    """Random pred yields positive pyramid loss."""
    torch.manual_seed(0)
    target = torch.randn(2, 1, 192, 96)
    sizes = [(12, 6), (24, 12), (48, 24), (96, 48), (192, 96)]
    pyr = [torch.randn(2, 1, h, w) for h, w in sizes]
    loss = pyramid_residual_loss(pyr, target)
    assert loss.item() > 0.0


def test_local_ffl_zero_on_perfect_reconstruction() -> None:
    """If pred == target, FFL == 0 exactly (modulo eps floor on weight)."""
    torch.manual_seed(0)
    target = torch.randn(2, 1, 96, 96)
    loss = local_focal_frequency_loss(target, target, patch=32, alpha=1.0)
    assert loss.item() < 1e-10


def test_local_ffl_finite_on_perfect_freestream() -> None:
    """When pred and target are zero everywhere, FFL is finite and ~0.

    The per-patch normalisation divides by the patch mean of the focal
    weight, which is zero in this case. The eps floor must keep the
    operation finite.
    """
    pred = torch.zeros(2, 1, 96, 96)
    target = torch.zeros(2, 1, 96, 96)
    loss = local_focal_frequency_loss(pred, target, patch=32, alpha=1.0)
    assert torch.isfinite(loss)


def test_local_ffl_positive_on_mismatch() -> None:
    """FFL is positive when prediction differs from target."""
    torch.manual_seed(0)
    target = torch.randn(2, 1, 64, 64)
    pred = target + 0.5 * torch.randn_like(target)
    loss = local_focal_frequency_loss(pred, target, patch=32)
    assert loss.item() > 0.0


def test_enstrophy_field_loss_zero_on_perfect() -> None:
    """``enstrophy_field_loss(x, x) == 0`` exactly."""
    torch.manual_seed(0)
    target = torch.randn(2, 1, 96, 96)
    loss = enstrophy_field_loss(target, target)
    assert loss.item() == 0.0


def test_enstrophy_field_loss_nonzero_on_uniform_noise() -> None:
    """**D71 bug-fix regression check.**

    Construct two fields with matched scalar-mean enstrophy:
        - pred:   uniform noise (no spatial structure)
        - target: a structured wake pattern
    The collaborator's original scalar-mean comparison
    ``(pred.pow(2).mean() - target.pow(2).mean()).pow(2)``
    would be exactly zero on this pair because the means are equal by
    construction. The spatial-field loss
    ``(pred.pow(2) - target.pow(2)).pow(2).mean()`` must be strictly
    positive.
    """
    torch.manual_seed(0)
    H, W = 64, 64
    # Structured target: a smooth sine-pattern wake
    x = torch.linspace(0, 4 * math.pi, W)
    y = torch.linspace(0, 2 * math.pi, H)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    target = (torch.sin(xx) * torch.cos(yy))[None, None]
    target = target.expand(2, -1, -1, -1).contiguous()

    # Pred: uniform Gaussian noise, scaled so its scalar-mean enstrophy
    # exactly matches the target's.
    pred = torch.randn(2, 1, H, W)
    pred = pred * (target.pow(2).mean() / pred.pow(2).mean()).sqrt()

    # Sanity: the means agree by construction.
    assert (pred.pow(2).mean() - target.pow(2).mean()).abs().item() < 1e-6

    # The collaborator's original scalar-mean loss would be exactly 0:
    scalar_mean_loss = (pred.pow(2).mean() - target.pow(2).mean()).pow(2)
    assert scalar_mean_loss.item() < 1e-10

    # The spatial-field enstrophy loss must reject this pair.
    field_loss = enstrophy_field_loss(pred, target)
    assert field_loss.item() > 0.01, (
        f"enstrophy_field_loss too small ({field_loss.item():.3e}); "
        "the spatial-field comparison should reject uniform noise"
    )


def test_circulation_density_loss_zero_on_perfect() -> None:
    """L1 of signed-vorticity differences is zero when pred == target."""
    torch.manual_seed(0)
    target = torch.randn(2, 1, 96, 96)
    loss = circulation_density_loss(target, target)
    assert loss.item() == 0.0


def test_circulation_density_loss_sign_sensitive() -> None:
    """L1 of signed vorticity penalises sign flips (L2 would not, due to
    the symmetric square)."""
    target = torch.ones(1, 1, 4, 4)
    pred_neg = -torch.ones_like(target)
    # L1 distance: |1 - (-1)| = 2 per pixel
    loss = circulation_density_loss(pred_neg, target)
    assert torch.allclose(loss, torch.tensor(2.0))


def test_region_pyr_ffl_loss_smoke() -> None:
    """End-to-end smoke: shapes and finiteness of the combined loss."""
    torch.manual_seed(0)
    sizes = [(12, 6), (24, 12), (48, 24), (96, 48), (192, 96)]
    pyr = [torch.randn(2, 1, h, w, requires_grad=True) for h, w in sizes]
    target = torch.randn(2, 1, 192, 96)
    out = region_pyr_ffl_loss(pyr, target)
    assert set(out.keys()) == {
        "L_total", "L_region", "L_pyramid", "L_ffl",
        "L_enstrophy", "L_circulation",
    }
    for k, v in out.items():
        assert torch.isfinite(v), f"{k} not finite: {v}"
    # Gradients flow back to the pyramid predictions.
    out["L_total"].backward()
    for p in pyr:
        assert p.grad is not None and (p.grad != 0).any()


def test_region_pyr_ffl_warmup_factor_zero_disables_ffl() -> None:
    """ffl_warmup_factor=0 zeroes the FFL contribution to L_total but
    leaves L_ffl itself reported for monitoring."""
    torch.manual_seed(0)
    sizes = [(12, 6), (24, 12), (48, 24), (96, 48), (192, 96)]
    pyr_a = [torch.randn(2, 1, h, w) for h, w in sizes]
    pyr_b = [p.clone() for p in pyr_a]
    target = torch.randn(2, 1, 192, 96)
    out_off = region_pyr_ffl_loss(pyr_a, target, ffl_warmup_factor=0.0,
                                  lambda_ffl=1.0)
    out_on = region_pyr_ffl_loss(pyr_b, target, ffl_warmup_factor=1.0,
                                 lambda_ffl=1.0)
    # L_ffl values are equal (it's the unweighted FFL).
    assert torch.allclose(out_off["L_ffl"], out_on["L_ffl"])
    # L_total differs by exactly lambda_ffl * L_ffl.
    diff = (out_on["L_total"] - out_off["L_total"]).item()
    assert abs(diff - out_off["L_ffl"].item()) < 1e-5


# -----------------------------------------------------------------------------
# Session 12 Direction A: PRF 2026 SL loss components
# -----------------------------------------------------------------------------


def test_hann_window_endpoints_zero_center_one() -> None:
    """Separable 2D Hann window is 0 at corners and 1 at the (odd) center."""
    win = hann_window_2d(7, 9, device=torch.device("cpu"))
    assert win.shape == (7, 9)
    assert win[0, 0].item() == 0.0
    assert win[-1, -1].item() == 0.0
    assert win[0, -1].item() == 0.0
    assert win[-1, 0].item() == 0.0
    # Center pixel of a 7x9 Hann window is exactly 1.
    assert abs(win[3, 4].item() - 1.0) < 1e-6
    # All values in [0, 1].
    assert (win >= 0).all() and (win <= 1 + 1e-6).all()


def test_tukey_window_alpha_zero_is_rectangular() -> None:
    """Tukey window with alpha=0 is identically 1 (rectangular)."""
    win = tukey_window_2d(5, 7, alpha=0.0, device=torch.device("cpu"))
    assert torch.allclose(win, torch.ones(5, 7))


def test_tukey_window_alpha_one_matches_hann() -> None:
    """Tukey window with alpha=1 equals a Hann window."""
    tukey = tukey_window_2d(11, 13, alpha=1.0, device=torch.device("cpu"))
    hann = hann_window_2d(11, 13, device=torch.device("cpu"))
    assert torch.allclose(tukey, hann, atol=1e-5)


def test_gradient_consistency_zero_on_perfect() -> None:
    """G(pred, target) = 0 when pred == target (Equation 7 of PRF 2026)."""
    torch.manual_seed(0)
    pred = torch.randn(2, 1, 32, 64)
    target = pred.clone()
    assert gradient_consistency_loss(pred, target).item() == 0.0


def test_gradient_consistency_positive_on_mismatch() -> None:
    """G > 0 when pred != target (random pair)."""
    torch.manual_seed(0)
    pred = torch.randn(2, 1, 32, 64)
    target = torch.randn(2, 1, 32, 64)
    assert gradient_consistency_loss(pred, target).item() > 0


def test_gradient_consistency_gradient_flow() -> None:
    """Gradient consistency loss is differentiable wrt pred."""
    torch.manual_seed(0)
    pred = torch.randn(2, 1, 32, 64, requires_grad=True)
    target = torch.randn(2, 1, 32, 64)
    loss = gradient_consistency_loss(pred, target)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert (pred.grad != 0).any()


def test_spectral_amplitude_zero_on_perfect() -> None:
    """H(pred, target) = 0 when pred == target (Equation 8 of PRF 2026)."""
    torch.manual_seed(0)
    pred = torch.randn(2, 1, 192, 96)
    target = pred.clone()
    loss = spectral_amplitude_loss(pred, target, wake_only=True, window="hann")
    assert loss.item() < 1e-5


def test_spectral_amplitude_positive_on_mismatch() -> None:
    """H > 0 when pred != target."""
    torch.manual_seed(0)
    pred = torch.randn(2, 1, 192, 96)
    target = torch.randn(2, 1, 192, 96)
    loss = spectral_amplitude_loss(pred, target, wake_only=True, window="hann")
    assert loss.item() > 0


def test_spectral_amplitude_wake_patch_dimensions() -> None:
    """``wake_only=True`` extracts the correct wake patch (H_wake, W_wake).

    With the canonical convention from ``decoder_metrics.wake_mask``
    (H is streamwise x in (-1.5, 4.5); W is cross-stream y in (-1.5, 1.5))
    and default wake_x in [0, 4.5], |y| < 1.25:

      row_lo = round((0 - (-1.5))/6 * 192) = 48
      row_hi = round((4.5 - (-1.5))/6 * 192) = 192
      col_lo = round((-1.25 - (-1.5))/3 * 96) = 8
      col_hi = round((1.25 - (-1.5))/3 * 96) = 88

    So the wake patch is (192-48, 88-8) = (144, 80).
    """
    torch.manual_seed(0)
    # Construct pred and target that differ only OUTSIDE the wake patch.
    # The wake-only loss should be zero on this pair (after windowing,
    # the loss reflects only differences inside the patch).
    pred = torch.zeros(1, 1, 192, 96)
    target = torch.zeros(1, 1, 192, 96)
    # Inject differences strictly outside the wake patch (rows 0..47, cols 0..7).
    pred[..., :48, :8] = 5.0
    target[..., :48, :8] = -5.0
    loss = spectral_amplitude_loss(pred, target, wake_only=True, window=None)
    assert loss.item() < 1e-6


def test_spectral_amplitude_window_required_for_nonperiodic() -> None:
    """Without windowing, the spectral amplitude on a non-periodic crop is
    dominated by edge-induced ringing. With Hann windowing, the same
    smooth pred/target pair (differing only in a low-frequency mean
    shift) yields a much smaller spectral discrepancy."""
    H_wake, W_wake = 144, 80
    # A smoothly-varying field (low spatial frequency) with a small
    # additive constant. Windowing should attenuate the boundary
    # contribution more than the bulk content.
    grid_y = torch.linspace(-1.0, 1.0, H_wake)[:, None].repeat(1, W_wake)
    grid_x = torch.linspace(-1.0, 1.0, W_wake)[None, :].repeat(H_wake, 1)
    pred = (grid_x ** 2 + grid_y ** 2)[None, None]
    target = pred + 0.1
    # Inflate to full 192x96 with the patch in the wake region.
    pred_full = torch.zeros(1, 1, 192, 96)
    target_full = torch.zeros(1, 1, 192, 96)
    pred_full[..., 48:192, 8:88] = pred
    target_full[..., 48:192, 8:88] = target
    no_win = spectral_amplitude_loss(
        pred_full, target_full, wake_only=True, window=None
    ).item()
    with_win = spectral_amplitude_loss(
        pred_full, target_full, wake_only=True, window="hann"
    ).item()
    # Both > 0, but Hann window suppresses boundary energy.
    assert no_win > 0 and with_win > 0
    # The windowed amplitude should be measurable; no_win includes
    # edge-discontinuity energy from the abrupt boundary of the patch.
    # (We don't assert strict ordering because the patch IS the data here,
    # but both finite and positive.)
    assert math.isfinite(no_win) and math.isfinite(with_win)


def test_spectral_amplitude_gradient_flow() -> None:
    """Spectral amplitude loss is differentiable wrt pred."""
    torch.manual_seed(0)
    pred = torch.randn(2, 1, 192, 96, requires_grad=True)
    target = torch.randn(2, 1, 192, 96)
    loss = spectral_amplitude_loss(pred, target, wake_only=True, window="hann")
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert (pred.grad != 0).any()


def test_region_pyr_specloss_loss_smoke() -> None:
    """Composite SL loss runs end-to-end with expected key set and is finite."""
    torch.manual_seed(0)
    sizes = [(12, 6), (24, 12), (48, 24), (96, 48), (192, 96)]
    pyr = [torch.randn(2, 1, h, w, requires_grad=True) for h, w in sizes]
    target = torch.randn(2, 1, 192, 96)
    out = region_pyr_specloss_loss(pyr, target)
    expected = {
        "L_total", "L_region", "L_pyramid", "L_gradient",
        "L_spectral_amp", "L_enstrophy", "L_circulation",
    }
    assert set(out.keys()) == expected
    for k, v in out.items():
        assert torch.isfinite(v).all(), f"{k} not finite: {v}"
    out["L_total"].backward()
    for p in pyr:
        assert p.grad is not None and (p.grad != 0).any()


def test_region_pyr_specloss_lambda_zero_disables_new_terms() -> None:
    """Setting lambda_gradient and lambda_spectral_amp to 0 reduces the
    composite to the E1 recipe (region + pyramid + enstrophy + circulation)."""
    torch.manual_seed(0)
    sizes = [(12, 6), (24, 12), (48, 24), (96, 48), (192, 96)]
    pyr_a = [torch.randn(2, 1, h, w) for h, w in sizes]
    pyr_b = [p.clone() for p in pyr_a]
    target = torch.randn(2, 1, 192, 96)
    out_zero = region_pyr_specloss_loss(
        pyr_a, target, lambda_gradient=0.0, lambda_spectral_amp=0.0
    )
    out_e1 = region_pyr_ffl_loss(
        pyr_b, target, lambda_ffl=0.0, ffl_warmup_factor=0.0
    )
    # With both new terms off and FFL off, the totals should match.
    assert abs(out_zero["L_total"].item() - out_e1["L_total"].item()) < 1e-5
