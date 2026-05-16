"""Rotary Position Embeddings for 1D temporal sequences.

Reference:
    Su, Lu, Pan, Murtadha, Wen, Liu. "RoFormer: Enhanced Transformer with
    Rotary Position Embedding." arXiv:2104.09864, 2021, sections 3.3 to 3.4.

Used in the JEPA predictor along the time axis only (the encoder uses 2D
sinusoidal embeddings for spatial tokens).
"""

from __future__ import annotations

import torch
from torch import Tensor


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor]:
    """Precompute cos/sin angles for rotary position embedding.

    Args:
        seq_len: Number of temporal positions.
        head_dim: Per-head dimension. Must be even.
        base: Base of the inverse-frequency geometric series. Default 10000.
        dtype: Output dtype for cos and sin caches.
        device: Output device. ``None`` defaults to CPU.

    Returns:
        ``(cos, sin)`` each of shape ``(seq_len, head_dim // 2)`` and dtype
        ``dtype`` on ``device``. Entry ``[t, i]`` holds ``cos`` (or ``sin``) of
        ``t / base ** (2 i / head_dim)``.

    Raises:
        ValueError: If ``head_dim`` is odd.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")

    device = device if device is not None else torch.device("cpu")
    half = head_dim // 2
    pair_idx = torch.arange(half, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (base ** (2.0 * pair_idx / head_dim))
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)
    angles = positions[:, None] * inv_freq[None, :]
    return angles.cos().to(dtype), angles.sin().to(dtype)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary position embedding to a query or key tensor.

    Pairs of adjacent feature dimensions are rotated by the precomputed angles.
    For pair ``i`` at position ``t`` with angle ``theta``::

        x'_{2i}   = x_{2i}   * cos(theta) - x_{2i+1} * sin(theta)
        x'_{2i+1} = x_{2i}   * sin(theta) + x_{2i+1} * cos(theta)

    Args:
        x: Tensor of shape ``(B, n_heads, seq_len, head_dim)``.
        cos: Cosine cache of shape ``(seq_len, head_dim // 2)``.
        sin: Sine cache of shape ``(seq_len, head_dim // 2)``.

    Returns:
        Tensor of the same shape, dtype, and device as ``x`` with each adjacent
        feature pair rotated.
    """
    cos = cos.to(dtype=x.dtype, device=x.device)
    sin = sin.to(dtype=x.dtype, device=x.device)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    return torch.stack((out_even, out_odd), dim=-1).flatten(-2)
