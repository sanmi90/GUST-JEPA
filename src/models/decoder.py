"""Visualisation decoder for the frozen JEPA encoder.

Mirror image of :class:`src.models.encoder.HybridCNNViTEncoder`. The decoder
takes a per-frame latent ``z`` in ``R^d`` and reconstructs the mid-plane
vorticity field ``omega_z`` of shape ``(192, 96)`` in a single forward pass.

The decoder is NEVER part of the JEPA loss (CLAUDE.md "Things to NOT do").
It is trained AFTER the JEPA encoder has been frozen, on the same train
encounters as the JEPA, with a simple per-frame MSE loss on ``omega_z``.

Architecture
------------
::

    z (B, d)
      |
      Linear "back-projection"  ->  (B, num_spatial_tokens * vit_hidden)
      reshape to (B, num_spatial_tokens, vit_hidden)
      + learned positional embedding
      |
      6 pre-norm ViT blocks   (hidden 256, 8 heads)
      LayerNorm
      |
      reshape to feature map (B, vit_hidden, 24, 12)
      |
      3 conv upsample stages   (24x12 -> 48x24 -> 96x48 -> 192x96)
      |
      final 1x1 conv to single channel
      |
      omega_z_hat (B, 1, 192, 96)

The default 3-stage stem produces 24x12 = 288 spatial tokens to match the
encoder. The conv upsampling stages use ``nn.PixelShuffle(2)`` instead of
transposed conv to avoid the well-known checkerboard artifacts.

References
----------
The architecture follows the standard "ViT decoder + conv head"
pattern used in masked image autoencoders (He et al., CVPR 2022) and
in the Solera-Rico et al. Nat. Commun. 2024 decoder for the same
gust-airfoil dataset family.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.models.encoder import _ViTBlock, _sin_cos_2d_pos_embed


def _upsample_block(in_ch: int, out_ch: int, n_groups: int = 8) -> nn.Sequential:
    """PixelShuffle 2x upsample + Conv2d + GroupNorm + GELU."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, padding=1, bias=True),
        nn.PixelShuffle(2),
        nn.GroupNorm(n_groups, out_ch),
        nn.GELU(),
    )


class HybridViTConvDecoder(nn.Module):
    """Decode a per-frame latent ``z`` back to vorticity ``omega_z``.

    Mirror image of :class:`HybridCNNViTEncoder`. The output spatial
    resolution is fixed at ``(out_h, out_w) = (192, 96)`` to match the
    encoder cache.

    Args:
        latent_dim: Dimension of ``z`` (32 by default, D2).
        vit_depth: Number of ViT blocks (default 6 to match the encoder).
        vit_hidden: Hidden width inside the ViT (default 256).
        vit_heads: Number of attention heads (default 8).
        vit_mlp_ratio: MLP expansion ratio inside the ViT (default 4.0).
        dropout: Dropout in the ViT blocks (default 0.0).
        out_h: Reconstruction height (default 192).
        out_w: Reconstruction width (default 96).
        feature_h: Feature-map height after the linear back-projection
            (default 24, == out_h // 8 to match the encoder's 3-stage stem).
        feature_w: Feature-map width after the linear back-projection
            (default 12, == out_w // 8).
        n_upsample_stages: Number of 2x upsample stages (default 3).
        c_mid: Channel dimensions at intermediate upsample stages
            (default ``(128, 64, 32)``).
    """

    def __init__(
        self,
        latent_dim: int = 32,
        vit_depth: int = 6,
        vit_hidden: int = 256,
        vit_heads: int = 8,
        vit_mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        out_h: int = 192,
        out_w: int = 96,
        feature_h: int = 24,
        feature_w: int = 12,
        n_upsample_stages: int = 3,
        c_mid: tuple[int, int, int] = (128, 64, 32),
    ) -> None:
        super().__init__()
        if feature_h * (2 ** n_upsample_stages) != out_h:
            raise ValueError(
                f"feature_h {feature_h} * 2^{n_upsample_stages} != out_h {out_h}"
            )
        if feature_w * (2 ** n_upsample_stages) != out_w:
            raise ValueError(
                f"feature_w {feature_w} * 2^{n_upsample_stages} != out_w {out_w}"
            )
        if len(c_mid) != n_upsample_stages:
            raise ValueError(
                f"c_mid must have {n_upsample_stages} entries, got {len(c_mid)}"
            )

        self.latent_dim = latent_dim
        self.feature_h = feature_h
        self.feature_w = feature_w
        self.vit_hidden = vit_hidden
        self.num_spatial_tokens = feature_h * feature_w

        self.back_proj = nn.Linear(latent_dim, self.num_spatial_tokens * vit_hidden)

        pos_embed = _sin_cos_2d_pos_embed(feature_h, feature_w, vit_hidden)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0), persistent=False)

        self.vit = nn.ModuleList(
            [_ViTBlock(vit_hidden, vit_heads, vit_mlp_ratio, dropout) for _ in range(vit_depth)]
        )
        self.norm = nn.LayerNorm(vit_hidden)

        ch_in = vit_hidden
        stages = []
        for ch_out in c_mid:
            stages.append(_upsample_block(ch_in, ch_out))
            ch_in = ch_out
        self.upsample = nn.Sequential(*stages)

        self.to_omega = nn.Conv2d(c_mid[-1], 1, kernel_size=1, bias=True)

    def forward(self, z: Tensor) -> Tensor:
        """Decode latents to vorticity.

        Args:
            z: Tensor of shape ``(B, latent_dim)`` or ``(B, T, latent_dim)``.

        Returns:
            ``omega_z_hat`` of shape ``(B, 1, out_h, out_w)`` (or
            ``(B, T, 1, out_h, out_w)`` when ``z`` was 3-D).
        """
        squeeze_T = False
        if z.dim() == 3:
            B, T, _ = z.shape
            z = z.reshape(B * T, -1)
            squeeze_T = True
        else:
            B, T = z.shape[0], 1

        h = self.back_proj(z)
        h = h.view(z.shape[0], self.num_spatial_tokens, self.vit_hidden)
        h = h + self.pos_embed
        for block in self.vit:
            h = block(h)
        h = self.norm(h)

        h = h.transpose(1, 2).reshape(z.shape[0], self.vit_hidden, self.feature_h, self.feature_w)
        h = self.upsample(h)
        x_hat = self.to_omega(h)

        if squeeze_T:
            x_hat = x_hat.view(B, T, *x_hat.shape[-3:])
        return x_hat
