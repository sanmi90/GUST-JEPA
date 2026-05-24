"""WakeRefiner: residual GAN refiner for the frozen E1 decoder output.

Session 12 Direction B (see ``SESSION12_CRISP_WAKE.md``). The Session 11
encoder (W0_C_lam100) and E1 LapFiLM decoder produce a "blurry but
present" wake reconstruction. This module is the generator half of a
GAN refinement pipeline: it takes the frozen E1 decoder output
:math:`\\hat{\\omega}_{E1}` and predicts a residual map
:math:`\\Delta \\hat{\\omega}` such that the refined field is

    omega_refined = omega_E1 + WakeRefiner(omega_E1, wake_mask)

The encoder and the E1 decoder are FROZEN during refiner training; only
the refiner and the patchGAN discriminator (see
:mod:`src.models.discriminator`) receive gradients. The discriminator
conditions on the wake ROI mask so the adversarial signal is wake-
localized rather than unconditional across the whole field.

Architecture
------------
6-block ResNet stack at 64 channels throughout, GroupNorm(8) + SiLU
activation, Conv3x3 inside each block. The last conv layer is
zero-initialised so the refiner is exactly identity at init (residual
= 0); this lets the refiner kick in gradually as the adversarial
gradient builds, and matches the LapSRN-style stable-init discipline
used by the E1 decoder heads (see
``src.models.lap_film_decoder.LapFiLMDecoder``).

Optional wake-mask conditioning concatenates the boolean wake ROI
mask (see :func:`src.evaluation.decoder_metrics.wake_mask`) as an
extra input channel so the refiner knows where to spend its capacity.

References
----------
- Johnson, Alahi, Fei-Fei. arXiv:1603.08155. Perceptual losses with
  ResNet generator (the original "9-block ResNet" pattern for
  image-to-image translation; we use 6 blocks at smaller widths).
- LapSRN: Lai, Huang, Ahuja, Yang. arXiv:1704.03915. Zero-init heads
  for stable training of residual networks.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn


def _group_norm(ch: int, n_groups: int = 8) -> nn.GroupNorm:
    """GroupNorm with a sensible ``num_groups`` for ``ch`` channels."""
    groups = min(n_groups, ch)
    if ch % groups != 0:
        groups = math.gcd(ch, n_groups) or 1
    return nn.GroupNorm(groups, ch)


class ResBlock(nn.Module):
    """Single GN -> SiLU -> Conv3x3 -> + skip residual block.

    The single-conv-per-block design keeps the parameter budget around
    the ~200k target for a 6-block stack at 64 channels (a standard
    two-conv block would put the refiner well over 400k).
    """

    def __init__(self, ch: int, n_groups: int = 8) -> None:
        super().__init__()
        self.norm = _group_norm(ch, n_groups=n_groups)
        self.act = nn.SiLU()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.conv(self.act(self.norm(x)))


class WakeRefiner(nn.Module):
    """6-block ResNet refiner that predicts a residual on top of the E1 output.

    Args:
        in_channels: Number of input channels in the frozen E1 decoder
            prediction. Always 1 for our omega_z task.
        channels: Internal feature width. Default 64 keeps the model
            around 200k params with 6 blocks.
        n_blocks: Number of ResNet blocks in the trunk. Default 6.
        use_wake_mask: If ``True``, concatenate the wake ROI mask as an
            extra input channel so the refiner sees where it is
            expected to act.
        n_groups: ``num_groups`` for the internal ``GroupNorm`` layers.

    Forward signature::

        forward(x: Tensor[B, 1, H, W], wake_mask: Optional[Tensor[H, W]] = None)
            -> Tensor[B, 1, H, W]

    The output is the RESIDUAL map (not the refined field). The caller
    is responsible for adding it to the input::

        residual = refiner(x_e1, wake_mask)
        x_refined = x_e1 + residual

    The last conv is zero-initialised so ``residual == 0`` at the start
    of training (identity at init).
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: int = 64,
        n_blocks: int = 6,
        use_wake_mask: bool = True,
        n_groups: int = 8,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.channels = channels
        self.n_blocks = n_blocks
        self.use_wake_mask = use_wake_mask

        stem_in = in_channels + (1 if use_wake_mask else 0)
        self.stem = nn.Conv2d(stem_in, channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [ResBlock(channels, n_groups=n_groups) for _ in range(n_blocks)]
        )
        self.head_norm = _group_norm(channels, n_groups=n_groups)
        self.head_act = nn.SiLU()
        self.head = nn.Conv2d(channels, in_channels, kernel_size=3, padding=1)

        # Identity-at-init: last conv zeroed so output residual is exactly
        # zero on the first forward pass, regardless of input.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _maybe_concat_mask(self, x: Tensor, wake_mask: Optional[Tensor]) -> Tensor:
        if not self.use_wake_mask:
            return x
        if wake_mask is None:
            mask_ch = torch.zeros(
                x.shape[0],
                1,
                x.shape[2],
                x.shape[3],
                device=x.device,
                dtype=x.dtype,
            )
        else:
            if wake_mask.dim() != 2:
                raise ValueError(f"wake_mask must be 2D (H, W); got shape {tuple(wake_mask.shape)}")
            if wake_mask.shape != x.shape[-2:]:
                raise ValueError(
                    f"wake_mask shape {tuple(wake_mask.shape)} does not match "
                    f"input spatial shape {tuple(x.shape[-2:])}"
                )
            mask_ch = (
                wake_mask.to(device=x.device, dtype=x.dtype)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(x.shape[0], 1, -1, -1)
            )
        return torch.cat([x, mask_ch], dim=1)

    def forward(
        self,
        x: Tensor,
        wake_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Predict a residual to add to ``x``.

        Args:
            x: E1 decoder prediction, shape ``(B, in_channels, H, W)``.
            wake_mask: Optional boolean / float wake ROI mask of shape
                ``(H, W)``. If ``self.use_wake_mask`` is True and
                ``wake_mask`` is ``None``, an all-zero mask channel is
                used instead.

        Returns:
            Residual tensor of shape ``(B, in_channels, H, W)``. Add this
            to ``x`` to obtain the refined prediction.
        """
        if x.dim() != 4:
            raise ValueError(f"x must be (B, C, H, W); got shape {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"x has {x.shape[1]} channels; expected in_channels={self.in_channels}"
            )

        feat = self._maybe_concat_mask(x, wake_mask)
        feat = self.stem(feat)
        for block in self.blocks:
            feat = block(feat)
        feat = self.head_act(self.head_norm(feat))
        residual = self.head(feat)
        return residual
