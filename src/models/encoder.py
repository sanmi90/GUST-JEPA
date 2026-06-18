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
import torch.nn.functional as F
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


class _CausalConv3dBlock(nn.Module):
    """Causal Conv3d -> per-frame GroupNorm -> GELU.

    The time axis is left-padded by ``t_kernel - 1`` with zeros and uses
    temporal stride 1, so output frame ``t`` depends only on input frames
    ``<= t``. GroupNorm is applied per frame (reshape to ``(B*T, C, H, W)``)
    so the normalization statistics never mix across time, which would
    otherwise leak the future into the past and break causality.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        t_kernel: int,
        s_kernel: int = 3,
        spatial_stride: int = 1,
        n_groups: int = 8,
    ) -> None:
        super().__init__()
        self._t_pad = t_kernel - 1
        s_pad = s_kernel // 2
        self.conv = nn.Conv3d(
            in_ch,
            out_ch,
            kernel_size=(t_kernel, s_kernel, s_kernel),
            stride=(1, spatial_stride, spatial_stride),
            padding=(0, s_pad, s_pad),
            bias=True,
        )
        self.norm = nn.GroupNorm(n_groups, out_ch)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, T, H, W). Causal left-pad on the time axis only.
        x = F.pad(x, (0, 0, 0, 0, self._t_pad, 0))
        x = self.conv(x)
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.act(self.norm(x))
        return x.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)


class SpatioTemporalCNNViTEncoder(nn.Module):
    """Causal 3D-conv tubelet stem + per-frame ViT, emitting per-frame latents.

    Same ``(B, T, 1, H, W) -> (B, T, d)`` contract as
    ``HybridCNNViTEncoder``, but the stem mixes a causal window of frames so
    each ``z_t`` integrates frames ``<= t`` instead of a single snapshot. The
    temporal receptive field is ``1 + (t_kernel - 1) * 3`` frames (the stem and
    the two spatial-downsampling convs are temporal; the residual blocks are
    spatial-only at temporal kernel 1). The ViT, ``[CLS]`` readout, and
    BatchNorm projection are identical to ``HybridCNNViTEncoder``.
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
        temporal_kernel: int = 3,
    ) -> None:
        super().__init__()
        if projection_norm not in ("batchnorm", "layernorm"):
            raise ValueError(
                f"projection_norm must be 'batchnorm' or 'layernorm', got {projection_norm!r}"
            )
        if temporal_kernel < 1:
            raise ValueError(f"temporal_kernel must be >= 1, got {temporal_kernel}")
        self.projection_norm = projection_norm
        self.temporal_kernel = temporal_kernel
        c1, c2, c3 = cnn_channels
        tk = temporal_kernel

        # Causal 3D stem: temporal in stem + the two downsamples (RF = 1+(tk-1)*3);
        # residual blocks are spatial-only (t_kernel=1).
        self.stem = _CausalConv3dBlock(in_channels, c1, t_kernel=tk, s_kernel=7, spatial_stride=2)
        self.block1 = nn.Sequential(
            _CausalConv3dBlock(c1, c1, t_kernel=1),
            _CausalConv3dBlock(c1, c1, t_kernel=1),
        )
        self.down1 = _CausalConv3dBlock(c1, c2, t_kernel=tk, spatial_stride=2)
        self.block2 = nn.Sequential(
            _CausalConv3dBlock(c2, c2, t_kernel=1),
            _CausalConv3dBlock(c2, c2, t_kernel=1),
        )
        self.down2 = _CausalConv3dBlock(c2, c3, t_kernel=tk, spatial_stride=2)
        self.block3 = nn.Sequential(
            _CausalConv3dBlock(c3, c3, t_kernel=1),
            _CausalConv3dBlock(c3, c3, t_kernel=1),
        )

        h_feat, w_feat = 192 // 8, 96 // 8
        self._num_spatial_tokens = h_feat * w_feat

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
        proj_norm: nn.Module = (
            nn.BatchNorm1d(latent_dim)
            if projection_norm == "batchnorm"
            else nn.LayerNorm(latent_dim)
        )
        self.proj = nn.Sequential(nn.Linear(vit_hidden, latent_dim), proj_norm)

    @property
    def num_spatial_tokens(self) -> int:
        return self._num_spatial_tokens

    def forward(self, x: Tensor) -> Tensor:
        """Encode a sub-trajectory into causal-window-aware per-frame latents.

        Args:
            x: ``(B, T, C, H, W)`` with ``C = 1``, ``H = 192``, ``W = 96``.

        Returns:
            ``z`` of shape ``(B, T, latent_dim)``.
        """
        if x.dim() != 5:
            raise ValueError(f"x must be (B, T, C, H, W), got {tuple(x.shape)}")
        b, t = x.shape[0], x.shape[1]
        h = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        h = self.stem(h)
        h = self.block1(h)
        h = self.down1(h)
        h = self.block2(h)
        h = self.down2(h)
        h = self.block3(h)  # (B, c3, T, 24, 12)

        c3, hf, wf = h.shape[1], h.shape[3], h.shape[4]
        h = h.permute(0, 2, 1, 3, 4).reshape(b * t, c3, hf, wf)
        h = h.flatten(2).transpose(1, 2)  # (B*T, 288, c3)
        h = self.token_proj(h)
        h = h + self.pos_embed
        cls = self.cls_token.expand(b * t, -1, -1)
        h = torch.cat([cls, h], dim=1)
        for block in self.vit:
            h = block(h)
        h = self.norm(h)
        z = self.proj(h[:, 0, :])
        return z.view(b, t, -1)


class CNNOnlyEncoder(nn.Module):
    """CNN-only ablation of :class:`HybridCNNViTEncoder` (the ViT removed).

    Shares the exact 3-stage CNN stem of the hybrid encoder, then global
    average pools the ``(B*T, c3, 24, 12)`` feature map to ``(B*T, c3)`` and
    applies the SAME ``Linear -> BatchNorm1d`` projection head to the latent
    dimension ``d``. The only architectural difference from
    :class:`HybridCNNViTEncoder` is the absence of the transformer, which is
    precisely the "CNN vs CNN+ViT" axis isolated by the Session 20 Track A
    2x2 controls (A2/A4 vs A1/A3). The output contract ``(B, T, d)`` and the
    BatchNorm latent boundary (required by SIGReg, CLAUDE.md) are identical,
    so it drops into the JEPA wrapper unchanged.
    """

    def __init__(
        self,
        in_channels: int = 1,
        cnn_channels: tuple[int, int, int] = (64, 128, 256),
        latent_dim: int = 32,
        projection_norm: str = "batchnorm",
    ) -> None:
        super().__init__()
        if projection_norm not in ("batchnorm", "layernorm"):
            raise ValueError(
                f"projection_norm must be 'batchnorm' or 'layernorm', got {projection_norm!r}"
            )
        self.projection_norm = projection_norm
        c1, c2, c3 = cnn_channels

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

        proj_norm: nn.Module = (
            nn.BatchNorm1d(latent_dim)
            if projection_norm == "batchnorm"
            else nn.LayerNorm(latent_dim)
        )
        self.proj = nn.Sequential(
            nn.Linear(c3, latent_dim),
            proj_norm,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Encode ``(B, T, C, H, W)`` with ``C=1, H=192, W=96`` into ``(B, T, d)``."""
        B, T = x.shape[0], x.shape[1]
        x_flat = x.flatten(0, 1)

        h = self.stem(x_flat)
        h = self.block1(h)
        h = self.down1(h)
        h = self.block2(h)
        h = self.down2(h)
        h = self.block3(h)

        h = h.mean(dim=(2, 3))  # global average pool -> (B*T, c3)
        z = self.proj(h)
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
