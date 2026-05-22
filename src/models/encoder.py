"""Hybrid CNN + ViT encoder for vortex-gust JEPA.

Reference:
    Maes, Le Lidec, Scieur, LeCun, Balestriero. "LeWorldModel: Stable
    End-to-End Joint-Embedding Predictive Architecture from Pixels."
    arXiv:2603.19312, 2026, Section 3.1 (projection-with-BatchNorm
    rationale; see HANDOFF.md D17 for the LeJEPA caveat).

The encoder is unconditional by design (HANDOFF.md D6); the static episode
descriptor ``c = (G, D, Y)`` enters only the predictor.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _sin_cos_2d_pos_embed(h: int, w: int, dim: int) -> Tensor:
    """Standard 2D sinusoidal positional embedding for an (h, w) feature map.

    Half the channels encode the y coordinate, half encode x. Within each
    half, channels alternate ``sin / cos`` pairs at geometrically spaced
    frequencies. Returns a ``(h * w, dim)`` tensor in fp32.

    Args:
        h: Grid height.
        w: Grid width.
        dim: Embedding dimension. Must be divisible by 4.

    Returns:
        Tensor of shape ``(h * w, dim)`` with deterministic sin/cos values.

    Raises:
        ValueError: If ``dim`` is not divisible by 4.
    """
    if dim % 4 != 0:
        raise ValueError(f"dim must be divisible by 4 for 2D sin-cos, got {dim}")
    half = dim // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, 2, dtype=torch.float32) / half))
    y_ang = torch.arange(h, dtype=torch.float32)[:, None] * inv_freq[None, :]
    x_ang = torch.arange(w, dtype=torch.float32)[:, None] * inv_freq[None, :]
    y_pe = torch.stack([y_ang.sin(), y_ang.cos()], dim=-1).flatten(-2)
    x_pe = torch.stack([x_ang.sin(), x_ang.cos()], dim=-1).flatten(-2)
    y_grid = y_pe[:, None, :].expand(h, w, half)
    x_grid = x_pe[None, :, :].expand(h, w, half)
    return torch.cat([y_grid, x_grid], dim=-1).reshape(h * w, dim)


def _conv_block(
    in_ch: int,
    out_ch: int,
    kernel: int = 3,
    stride: int = 1,
    n_groups: int = 8,
) -> nn.Sequential:
    """Conv2d -> GroupNorm -> GELU building block."""
    pad = kernel // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=True),
        nn.GroupNorm(n_groups, out_ch),
        nn.GELU(),
    )


class _ViTBlock(nn.Module):
    """Pre-norm transformer encoder block (LayerNorm -> MHA -> residual ->
    LayerNorm -> MLP -> residual)."""

    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class HybridCNNViTEncoder(nn.Module):
    """Hybrid CNN stem followed by a small ViT, with [CLS] readout and a
    BatchNorm-projected MLP head to the latent dimension ``d``.

    Reference architecture for the projection-with-BatchNorm choice:
        Maes et al., "LeWorldModel: Stable End-to-End Joint-Embedding
        Predictive Architecture from Pixels", arXiv:2603.19312, Section 3.1.

    Attributes:
        stem, block1, down1, block2, down2, block3: CNN stages producing a
            ``(B*T, c3, 24, 12)`` feature map from ``(B*T, 1, 192, 96)`` input.
        cls_token: Learnable ``(1, 1, vit_hidden)`` token prepended to each
            frame's spatial token sequence.
        pos_embed: Non-persistent buffer of shape ``(1, 288, vit_hidden)``
            holding the deterministic 2D sin-cos positional embedding.
        vit: ``vit_depth`` pre-norm transformer blocks at width ``vit_hidden``.
        norm: Final LayerNorm before the [CLS] readout.
        proj: Linear -> BatchNorm1d head producing the latent embedding (the
            BatchNorm is the LeWM-specific layer; see HANDOFF.md D17).
    """

    def __init__(
        self,
        in_channels: int = 1,
        cnn_channels: tuple[int, int, int] = (64, 128, 256),
        vit_depth: int = 6,
        vit_hidden: int = 256,
        vit_heads: int = 8,
        vit_mlp_ratio: float = 4.0,
        latent_dim: int = 32,
        dropout: float = 0.0,
        projection_norm: str = "batchnorm",
    ) -> None:
        super().__init__()
        if projection_norm not in ("batchnorm", "layernorm"):
            raise ValueError(
                f"projection_norm must be 'batchnorm' or 'layernorm', got {projection_norm!r}"
            )
        self.projection_norm = projection_norm
        c1, c2, c3 = cnn_channels

        # CNN stem (192x96 -> 96x48 -> 48x24 -> 24x12).
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=7, stride=2, padding=3, bias=True),
            nn.GroupNorm(8, c1),
            nn.GELU(),
        )
        self.block1 = nn.Sequential(
            _conv_block(c1, c1, kernel=3, stride=1),
            _conv_block(c1, c1, kernel=3, stride=1),
        )
        self.down1 = _conv_block(c1, c2, kernel=3, stride=2)
        self.block2 = nn.Sequential(
            _conv_block(c2, c2, kernel=3, stride=1),
            _conv_block(c2, c2, kernel=3, stride=1),
        )
        self.down2 = _conv_block(c2, c3, kernel=3, stride=2)
        self.block3 = nn.Sequential(
            _conv_block(c3, c3, kernel=3, stride=1),
            _conv_block(c3, c3, kernel=3, stride=1),
        )

        # 288 spatial tokens of dim c3 after the stem on a (192, 96) input.
        h_feat, w_feat = 192 // 8, 96 // 8
        self._num_spatial_tokens = h_feat * w_feat

        # Lift channels into the ViT hidden dim (identity if equal).
        self.token_proj: nn.Module = (
            nn.Identity() if c3 == vit_hidden else nn.Linear(c3, vit_hidden)
        )

        pos_embed = _sin_cos_2d_pos_embed(h_feat, w_feat, vit_hidden)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0), persistent=False)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, vit_hidden))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.vit = nn.ModuleList(
            [_ViTBlock(vit_hidden, vit_heads, vit_mlp_ratio, dropout) for _ in range(vit_depth)]
        )
        self.norm = nn.LayerNorm(vit_hidden)

        # Projection head. BatchNorm is the LeWM-specific default (HANDOFF.md D17).
        # The Session 5 ``--projection-norm layernorm`` switch wires LayerNorm here
        # as the first diagnostic intervention if SIGReg collapses on physics data
        # (Session 5 Run B). See HANDOFF.md D25.
        proj_norm: nn.Module = (
            nn.BatchNorm1d(latent_dim)
            if projection_norm == "batchnorm"
            else nn.LayerNorm(latent_dim)
        )
        self.proj = nn.Sequential(
            nn.Linear(vit_hidden, latent_dim),
            proj_norm,
        )

    @property
    def num_spatial_tokens(self) -> int:
        """288 for the default 3-stage stem on a (192, 96) input."""
        return self._num_spatial_tokens

    def forward(self, x: Tensor) -> Tensor:
        """Encode a sub-trajectory of vorticity frames into per-frame latents.

        Args:
            x: Tensor of shape ``(B, T, C, H, W)`` with ``C = 1``, ``H = 192``,
                ``W = 96``.

        Returns:
            ``z`` of shape ``(B, T, latent_dim)``.
        """
        B, T = x.shape[0], x.shape[1]
        x_flat = x.flatten(0, 1)

        h = self.stem(x_flat)
        h = self.block1(h)
        h = self.down1(h)
        h = self.block2(h)
        h = self.down2(h)
        h = self.block3(h)

        h = h.flatten(2).transpose(1, 2)
        h = self.token_proj(h)
        h = h + self.pos_embed

        cls = self.cls_token.expand(B * T, -1, -1)
        h = torch.cat([cls, h], dim=1)
        for block in self.vit:
            h = block(h)
        h = self.norm(h)

        z = self.proj(h[:, 0, :])
        return z.view(B, T, -1)


class PatchPoolEncoder(nn.Module):
    """Tiny baseline encoder used for the Track 0.1 LapFiLM upper-bound test.

    Mean-pools the input omega field over fixed 16x16 patches (192x96 -> 12x6),
    then mixes the single input channel into ``out_channels`` channels via a 1x1
    convolution. Output is flattened to a 4608-dim vector per frame so it
    drops in to LapFiLMDecoder with ``spatial_init=True`` (which reshapes
    flat ``base_ch*base_h*base_w = 64*12*6`` to the level-0 feature map).

    Purpose: bypass the JEPA encoder to test what the visualisation decoder
    can reconstruct given near-raw spatial information. If LapFiLM's Test B
    SSIM stays low here, H2 (decoder-architecture-limited) is supported; if
    it improves substantially, H1 (encoder-bottleneck-limited) is supported.

    SESSION11_WAKE_RESULTS_FIRST.md Track 0.1.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 64,
        patch_h: int = 16,
        patch_w: int = 16,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.patch_h = int(patch_h)
        self.patch_w = int(patch_w)
        self.pool = nn.AvgPool2d(kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w))
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """Encode ``(B, T, C, H, W)`` -> flat ``(B, T, out_channels * H/ph * W/pw)``.

        Also accepts ``(B, C, H, W)`` and returns ``(B, out_channels * H/ph * W/pw)``.
        """
        squeeze_T = False
        if x.dim() == 5:
            B, T = x.shape[0], x.shape[1]
            x = x.flatten(0, 1)
            squeeze_T = True
        elif x.dim() == 4:
            B = x.shape[0]
            T = 1
        else:
            raise ValueError(
                f"PatchPoolEncoder expects 4D (B, C, H, W) or 5D (B, T, C, H, W); "
                f"got {tuple(x.shape)}"
            )
        h = self.pool(x)
        h = self.proj(h)
        h = h.flatten(1)
        if squeeze_T:
            h = h.view(B, T, -1)
        return h
