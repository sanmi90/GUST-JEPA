"""LapFiLMDecoder: Laplacian-pyramid visualisation decoder with FiLM conditioning.

Five-level pyramid 12x6 -> 24x12 -> 48x24 -> 96x48 -> 192x96. At each level
the decoder concatenates the previous feature map (after PixelShuffle 2x
upsampling) with coordinate channels, Fourier features, and an optional
airfoil-adjacent mask channel, projects back to that level's channel
count, applies a stack of FiLM-conditioned residual blocks (GroupNorm +
SiLU), and emits a 1-channel residual added to the upsampled prediction
from the previous level (LapSRN-style).

References
----------
- LapSRN: Lai, Huang, Ahuja, Yang. arXiv:1704.03915. Laplacian pyramid
  super-resolution with per-level residual prediction.
- FiLM: Perez, Strub, de Vries, Dumoulin, Courville. arXiv:1709.07871.
  Feature-wise linear modulation.
- PixelShuffle / sub-pixel CNN: Shi et al. arXiv:1609.05158. Learned
  upsampling via channel-to-space rearrangement.
- CoordConv: Liu et al. arXiv:1807.03247. Coordinate channels.
- Fourier Features: Tancik et al. arXiv:2006.10739. High-frequency
  encoding for coordinate-conditioned networks.

The architecture spec is locked in SESSION10_MULTISCALE_DECODER.md Step 1.
Training entrypoint: scripts/session9_train_decoder.py with
``--decoder-type lapfilm``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


REPO_ROOT = Path(__file__).resolve().parents[2]


def _build_coord_grid(h: int, w: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Build a (2, h, w) coordinate grid in [-1, 1].

    Channel 0 is x (column direction), channel 1 is y (row direction).
    Matches the (B, C, H, W) convention where H is rows (y) and W is
    columns (x).
    """
    ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=0)


def _fourier_features(coord: Tensor, n_bands: int) -> Tensor:
    """(2, h, w) -> (4 * n_bands, h, w) via sin/cos at geometric frequencies.

    Frequencies are ``pi, 2 pi, 4 pi, 8 pi, ...`` (geometric base 2). For
    each frequency, both x and y get sin and cos features, so the channel
    count is ``2 (coords) * 2 (sin, cos) * n_bands = 4 * n_bands``.
    """
    if n_bands == 0:
        return torch.empty(0, coord.shape[1], coord.shape[2],
                           device=coord.device, dtype=coord.dtype)
    freqs = (2.0 ** torch.arange(n_bands, device=coord.device, dtype=coord.dtype)) * math.pi
    # coord: (2, h, w); freqs: (n_bands,)
    ang = coord[:, None, :, :] * freqs[None, :, None, None]
    # ang: (2, n_bands, h, w) -> (2*n_bands, h, w) per sin / cos
    sin_feat = ang.sin().reshape(-1, coord.shape[1], coord.shape[2])
    cos_feat = ang.cos().reshape(-1, coord.shape[1], coord.shape[2])
    return torch.cat([sin_feat, cos_feat], dim=0)


class FiLMResBlock(nn.Module):
    """Residual block with optional FiLM modulation conditioned on ``z``.

    Structure (FiLM enabled)::

        x -> GroupNorm -> FiLM(gamma1, beta1) -> SiLU -> Conv3x3
          -> GroupNorm -> FiLM(gamma2, beta2) -> SiLU -> Conv3x3 -> + x

    FiLM linears are zero-initialized so ``gamma = beta = 0`` and the
    block is exactly identity-on-residual at init: ``(1 + 0) * x + 0 = x``.

    When ``cond_dim`` is None, the FiLM linear is omitted and the block
    behaves as a plain pre-norm ResBlock (the ``no_film`` ablation
    pathway, with the latent concatenated as input channels by the
    decoder).
    """

    def __init__(
        self,
        ch: int,
        cond_dim: Optional[int],
        n_groups: int = 8,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        groups = min(n_groups, ch)
        if ch % groups != 0:
            groups = math.gcd(ch, n_groups) or 1
        self.norm1 = nn.GroupNorm(groups, ch)
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, ch)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.act = nn.SiLU() if activation == "silu" else nn.GELU()
        self.use_film = cond_dim is not None
        self.ch = ch
        if self.use_film:
            self.film = nn.Linear(cond_dim, 4 * ch)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

    def forward(self, x: Tensor, z: Optional[Tensor] = None) -> Tensor:
        h = self.norm1(x)
        if self.use_film:
            if z is None:
                raise ValueError("FiLMResBlock has FiLM enabled but z is None")
            params = self.film(z)
            g1, b1, g2, b2 = params.chunk(4, dim=-1)
            h = h * (1.0 + g1[:, :, None, None]) + b1[:, :, None, None]
        h = self.act(h)
        h = self.conv1(h)
        h = self.norm2(h)
        if self.use_film:
            h = h * (1.0 + g2[:, :, None, None]) + b2[:, :, None, None]
        h = self.act(h)
        h = self.conv2(h)
        return x + h


class _PixelShuffleUp(nn.Module):
    """Conv1x1 -> PixelShuffle(2)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 4, kernel_size=1)
        self.ps = nn.PixelShuffle(2)

    def forward(self, x: Tensor) -> Tensor:
        return self.ps(self.conv(x))


class _BilinearConvUp(nn.Module):
    """F.interpolate(bilinear) -> Conv3x3."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv(x)


class LapFiLMDecoder(nn.Module):
    """5-level Laplacian-pyramid decoder with FiLM conditioning on ``z``.

    Args:
        latent_dim: Dimension of the per-frame latent ``z`` (32 by default).
        base_hw: Spatial size of the coarsest pyramid level.
        channels: Per-level channel counts, coarse to fine.
        resblocks_per_level: Number of FiLM residual blocks at each level.
        norm: Normalisation type. Only ``"groupnorm"`` is supported.
        activation: ``"silu"`` (default) or ``"gelu"``.
        upsample: ``"pixelshuffle"`` (default) or ``"bilinear_conv"``.
        fourier_bands: Number of geometric Fourier feature bands per coord.
        use_coord_channels: Include raw (x, y) and Fourier features.
        use_airfoil_mask_channel: Include the airfoil-adjacent mask as an
            extra channel at every pyramid level (downsampled via
            adaptive max-pool).
        use_film: If ``False``, the FiLM linears are omitted and the
            latent ``z`` is broadcast as constant channels at every
            level instead (the ``no_film`` ablation pathway).
        airfoil_mask_path: Path to the 192x96 boolean mask file. Defaults
            to ``outputs/data_pipeline/v1/airfoil_adjacent_mask.npy``.
        out_h, out_w: Final output spatial size; must equal
            ``base_hw * 2^(n_levels - 1)``.
        final_activation: Optional activation applied to the final 192x96
            prediction. ``None`` (default), ``"tanh"``, or ``"sigmoid"``.

    Forward signature: takes ``z`` of shape ``(B, latent_dim)`` or
    ``(B, T, latent_dim)`` and returns::

        {
            "pred": <final prediction (B, 1, H, W) or (B, T, 1, H, W)>,
            "pyramid": [level0_pred, level1_pred, ..., level4_pred],
        }

    where each level's prediction has the corresponding pyramid spatial
    resolution. The pyramid loss in :mod:`src.models.decoder_losses`
    consumes both fields.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        base_hw: tuple[int, int] = (12, 6),
        channels: tuple[int, ...] = (64, 64, 48, 32, 24),
        resblocks_per_level: int = 2,
        norm: str = "groupnorm",
        activation: str = "silu",
        upsample: str = "pixelshuffle",
        fourier_bands: int = 4,
        use_coord_channels: bool = True,
        use_airfoil_mask_channel: bool = True,
        use_film: bool = True,
        airfoil_mask_path: Optional[str] = None,
        out_h: int = 192,
        out_w: int = 96,
        final_activation: Optional[str] = None,
    ) -> None:
        super().__init__()

        if norm != "groupnorm":
            raise ValueError(f"only groupnorm supported, got {norm!r}")
        if upsample not in ("pixelshuffle", "bilinear_conv"):
            raise ValueError(f"unknown upsample mode {upsample!r}")

        n_levels = len(channels)
        if n_levels < 2:
            raise ValueError("need at least 2 pyramid levels")

        base_h, base_w = base_hw
        expected_h = base_h * (2 ** (n_levels - 1))
        expected_w = base_w * (2 ** (n_levels - 1))
        if expected_h != out_h or expected_w != out_w:
            raise ValueError(
                f"base_hw {base_hw} * 2^{n_levels - 1} = ({expected_h}, {expected_w})"
                f" != out ({out_h}, {out_w})"
            )

        self.latent_dim = latent_dim
        self.base_hw = base_hw
        self.channels = tuple(channels)
        self.resblocks_per_level = resblocks_per_level
        self.fourier_bands = fourier_bands
        self.use_coord_channels = use_coord_channels
        self.use_airfoil_mask_channel = use_airfoil_mask_channel
        self.use_film = use_film
        self.out_h = out_h
        self.out_w = out_w
        self.final_activation = final_activation
        self.activation_name = activation

        cond_dim = latent_dim if use_film else None

        coord_extra = 0
        if use_coord_channels:
            coord_extra += 2
            coord_extra += 4 * fourier_bands
        if use_airfoil_mask_channel:
            coord_extra += 1
        if not use_film:
            coord_extra += latent_dim
        self.coord_extra = coord_extra

        base_ch = channels[0]
        self.init_proj = nn.Linear(latent_dim, base_ch * base_h * base_w)

        self.input_projs = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.heads = nn.ModuleList()
        self.ups = nn.ModuleList()

        for k, ch in enumerate(channels):
            self.input_projs.append(
                nn.Conv2d(ch + coord_extra, ch, kernel_size=1)
            )
            level_blocks = nn.ModuleList(
                [FiLMResBlock(ch, cond_dim, activation=activation)
                 for _ in range(resblocks_per_level)]
            )
            self.blocks.append(level_blocks)
            self.heads.append(nn.Conv2d(ch, 1, kernel_size=1))
            if k < n_levels - 1:
                next_ch = channels[k + 1]
                if upsample == "pixelshuffle":
                    self.ups.append(_PixelShuffleUp(ch, next_ch))
                else:
                    self.ups.append(_BilinearConvUp(ch, next_ch))

        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

        if use_airfoil_mask_channel:
            if airfoil_mask_path is None:
                airfoil_mask_path = str(
                    REPO_ROOT / "outputs" / "data_pipeline" / "v1"
                    / "airfoil_adjacent_mask.npy"
                )
            mask_np = np.load(airfoil_mask_path)
            full_mask = torch.from_numpy(mask_np.astype(np.float32))[None, None]
            for k in range(n_levels):
                h_k = base_h * (2 ** k)
                w_k = base_w * (2 ** k)
                level_mask = F.adaptive_max_pool2d(full_mask, output_size=(h_k, w_k))
                self.register_buffer(f"airfoil_mask_l{k}", level_mask, persistent=False)

    def _build_coord_features(
        self, h: int, w: int, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        """Build (1, coord_ch, h, w) coord + Fourier features (no batch dim)."""
        if not self.use_coord_channels:
            return torch.zeros(1, 0, h, w, device=device, dtype=dtype)
        coord = _build_coord_grid(h, w, device, dtype)
        if self.fourier_bands > 0:
            ff = _fourier_features(coord, self.fourier_bands)
            cat = torch.cat([coord, ff], dim=0)
        else:
            cat = coord
        return cat.unsqueeze(0)

    def forward(self, z: Tensor) -> dict[str, Tensor | list[Tensor]]:
        squeeze_T = False
        B_out = z.shape[0]
        T_out = 1
        if z.dim() == 3:
            B_out, T_out, _ = z.shape
            z = z.reshape(B_out * T_out, -1)
            squeeze_T = True

        N = z.shape[0]
        base_h, base_w = self.base_hw
        base_ch = self.channels[0]

        feat = self.init_proj(z).view(N, base_ch, base_h, base_w)

        pyramid: list[Tensor] = []
        prev_pred_up: Optional[Tensor] = None

        for k, level_ch in enumerate(self.channels):
            h_k = base_h * (2 ** k)
            w_k = base_w * (2 ** k)

            extras: list[Tensor] = []
            if self.use_coord_channels:
                coord_feat = self._build_coord_features(h_k, w_k, z.device, feat.dtype)
                extras.append(coord_feat.expand(N, -1, -1, -1))
            if self.use_airfoil_mask_channel:
                mask = getattr(self, f"airfoil_mask_l{k}").to(dtype=feat.dtype)
                extras.append(mask.expand(N, -1, -1, -1))
            if not self.use_film:
                z_chan = z[:, :, None, None].expand(-1, -1, h_k, w_k).to(dtype=feat.dtype)
                extras.append(z_chan)

            if extras:
                feat = torch.cat([feat, *extras], dim=1)

            feat = self.input_projs[k](feat)

            for block in self.blocks[k]:
                feat = block(feat, z if self.use_film else None)

            delta = self.heads[k](feat)
            if prev_pred_up is None:
                level_pred = delta
            else:
                level_pred = prev_pred_up + delta
            pyramid.append(level_pred)

            if k < len(self.channels) - 1:
                feat = self.ups[k](feat)
                prev_pred_up = F.interpolate(
                    level_pred, scale_factor=2, mode="bilinear", align_corners=False
                )

        final_pred = pyramid[-1]
        if self.final_activation == "tanh":
            final_pred = torch.tanh(final_pred)
        elif self.final_activation == "sigmoid":
            final_pred = torch.sigmoid(final_pred)
        elif self.final_activation is not None:
            raise ValueError(f"unknown final_activation {self.final_activation!r}")

        if squeeze_T:
            final_pred = final_pred.view(B_out, T_out, *final_pred.shape[-3:])
            pyramid = [p.view(B_out, T_out, *p.shape[-3:]) for p in pyramid]

        return {"pred": final_pred, "pyramid": pyramid}
