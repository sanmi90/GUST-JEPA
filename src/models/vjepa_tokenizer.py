"""3D-conv tubelet tokenizer for V-JEPA (arXiv:2404.08471, Sec. 3)."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _sin_cos_1d(n: int, dim: int) -> Tensor:
    """(n, dim) 1D sin-cos table; dim must be even."""
    if dim % 2 != 0:
        raise ValueError(f"dim must be even, got {dim}")
    pos = torch.arange(n, dtype=torch.float32)[:, None]
    inv = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    ang = pos * inv[None, :]
    return torch.stack([ang.sin(), ang.cos()], dim=-1).flatten(1)


def sincos_3d(grid: tuple[int, int, int], dim: int) -> Tensor:
    """Factorized 3D sin-cos pos embed of shape (prod(grid), dim). dim % 6 == 0."""
    if dim % 6 != 0:
        raise ValueError(f"dim must be divisible by 6 for 3D sin-cos, got {dim}")
    gt, gh, gw = grid
    d3 = dim // 3
    et = _sin_cos_1d(gt, d3)[:, None, None, :].expand(gt, gh, gw, d3)
    eh = _sin_cos_1d(gh, d3)[None, :, None, :].expand(gt, gh, gw, d3)
    ew = _sin_cos_1d(gw, d3)[None, None, :, :].expand(gt, gh, gw, d3)
    return torch.cat([et, eh, ew], dim=-1).reshape(gt * gh * gw, dim)


class TubeletEmbed(nn.Module):
    """Embed an omega clip into space-time tubelet tokens.

    Input ``(B, T, 1, H, W)`` -> ``(B, N, D)`` where the tubelet is
    ``(t, p, p)`` with stride = tubelet size, ``N = (T/t)(H/p)(W/p)``.
    """

    def __init__(
        self,
        in_channels: int = 1,
        t_tubelet: int = 2,
        p_tubelet: int = 16,
        hidden: int = 384,
        clip_len: int = 32,
        height: int = 192,
        width: int = 96,
    ) -> None:
        super().__init__()
        if hidden % 6 != 0:
            raise ValueError(f"hidden must be divisible by 6, got {hidden}")
        self.proj = nn.Conv3d(
            in_channels,
            hidden,
            kernel_size=(t_tubelet, p_tubelet, p_tubelet),
            stride=(t_tubelet, p_tubelet, p_tubelet),
        )
        self.grid = (clip_len // t_tubelet, height // p_tubelet, width // p_tubelet)
        self.num_tokens = self.grid[0] * self.grid[1] * self.grid[2]
        pos = sincos_3d(self.grid, hidden).unsqueeze(0)
        self.register_buffer("pos_embed", pos, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 5:
            raise ValueError(f"x must be (B,T,1,H,W), got {tuple(x.shape)}")
        x = x.permute(0, 2, 1, 3, 4)  # (B,1,T,H,W) for Conv3d
        h = self.proj(x)  # (B, D, T/t, H/p, W/p)
        tok = h.flatten(2).transpose(1, 2)  # (B, N, D)
        return tok + self.pos_embed
