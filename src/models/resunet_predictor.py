"""Convolutional ResUNet predictor for the spatial latent (Session 31 Track B.3).

STUB for Track D. The ResUNet rolls a SPATIAL latent forward one step: it
consumes the channel-wise concatenation of the previous and current latent maps
``(B, context_length * d, h, w)`` and predicts the next latent map
``(B, d, h, w)``. This is the gray-scott / eb_jepa native predictor for the
spatial latent (a convolutional U-Net with skip connections), the spatial-tap
counterpart to the project's pooled-latent autoregressive transformer.

Scope here is the module plus a shape contract; training and evaluation are
Track D and are deliberately out of scope. Downsampling uses stride-2 convs and
upsampling uses ``F.interpolate`` to the exact skip resolution, so non-power-of-2
grids (e.g. the default 24x12 token grid, or odd dims) round-trip cleanly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class _ResBlock(nn.Module):
    """Conv -> GroupNorm -> GELU -> Conv -> GroupNorm with a residual add.

    A 1x1 projection on the skip handles channel mismatch. GroupNorm keeps the
    block batch-size agnostic, which matters for the small latent batches used
    in rollout training.
    """

    def __init__(self, in_ch: int, out_ch: int, n_groups: int = 8) -> None:
        super().__init__()
        groups = min(n_groups, out_ch)
        while out_ch % groups != 0:
            groups -= 1
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.norm1 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.act = nn.GELU()
        self.skip: nn.Module = (
            nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + self.skip(x))


class ResUNetPredictor(nn.Module):
    """Roll the spatial latent forward by one step with a ResUNet.

    Args:
        latent_dim: Channel dimension ``d`` of the spatial latent.
        context_length: Number of stacked input frames (default 2:
            previous + current). Input channels are ``context_length * d``.
        base_channels: Channel width at the top (finest) U-Net level.
        depth: Number of downsampling levels (default 2 -> bottleneck at
            ``h/4 x w/4``).
    """

    def __init__(
        self,
        latent_dim: int = 32,
        context_length: int = 2,
        base_channels: int = 64,
        depth: int = 2,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if context_length <= 0:
            raise ValueError(f"context_length must be positive, got {context_length}")
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        self.latent_dim = int(latent_dim)
        self.context_length = int(context_length)
        self.in_channels = self.context_length * self.latent_dim
        self.depth = int(depth)

        self.stem = _ResBlock(self.in_channels, base_channels)

        enc_blocks = []
        downs = []
        ch = base_channels
        for _ in range(depth):
            downs.append(nn.Conv2d(ch, ch * 2, kernel_size=3, stride=2, padding=1, bias=True))
            enc_blocks.append(_ResBlock(ch * 2, ch * 2))
            ch *= 2
        self.downs = nn.ModuleList(downs)
        self.enc_blocks = nn.ModuleList(enc_blocks)

        self.bottleneck = _ResBlock(ch, ch)

        ups = []
        dec_blocks = []
        for _ in range(depth):
            skip_ch = ch // 2
            ups.append(nn.Conv2d(ch, skip_ch, kernel_size=3, padding=1, bias=True))
            # After upsample + concat with the encoder skip: skip_ch + skip_ch.
            dec_blocks.append(_ResBlock(skip_ch * 2, skip_ch))
            ch = skip_ch
        self.ups = nn.ModuleList(ups)
        self.dec_blocks = nn.ModuleList(dec_blocks)

        self.head = nn.Conv2d(base_channels, latent_dim, kernel_size=1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        """Predict the next latent map.

        Args:
            x: ``(B, context_length * d, h, w)`` stacked latent context.

        Returns:
            ``(B, d, h, w)`` predicted next latent map.
        """
        if x.dim() != 4:
            raise ValueError(f"expected (B, C, H, W), got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"channel dim must be context_length*latent_dim={self.in_channels}; "
                f"got {x.shape[1]}"
            )

        h = self.stem(x)
        skips: list[Tensor] = []
        for down, block in zip(self.downs, self.enc_blocks):
            skips.append(h)
            h = block(down(h))

        h = self.bottleneck(h)

        for up, block, skip in zip(self.ups, self.dec_blocks, reversed(skips)):
            h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = up(h)
            h = block(torch.cat([h, skip], dim=1))

        return self.head(h)

    def forward_context(self, prev: Tensor, curr: Tensor) -> Tensor:
        """Roll forward from separate ``prev`` and ``curr`` latent maps.

        Args:
            prev: ``(B, d, h, w)`` previous latent map.
            curr: ``(B, d, h, w)`` current latent map.

        Returns:
            ``(B, d, h, w)`` predicted next latent map.
        """
        if self.context_length != 2:
            raise ValueError(f"forward_context expects context_length=2, got {self.context_length}")
        return self.forward(torch.cat([prev, curr], dim=1))
