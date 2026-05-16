"""Unit tests for src/models/rope.py."""

from __future__ import annotations

import pytest
import torch

from src.models.rope import apply_rope, build_rope_cache


def test_rope_identity_at_position_zero() -> None:
    """apply_rope at position 0 returns the input unchanged (cos=1, sin=0)."""
    torch.manual_seed(0)
    head_dim = 64
    x = torch.randn(2, 4, 1, head_dim)
    cos, sin = build_rope_cache(seq_len=16, head_dim=head_dim)
    out = apply_rope(x, cos[0:1], sin[0:1])
    assert torch.allclose(out, x, atol=1e-6)


def test_rope_preserves_dot_product_relative_to_offset() -> None:
    """dot(rope(q, t1), rope(k, t2)) depends only on (t1 - t2)."""
    torch.manual_seed(0)
    head_dim = 64
    q = torch.randn(1, 1, 1, head_dim)
    k = torch.randn(1, 1, 1, head_dim)
    cos, sin = build_rope_cache(seq_len=32, head_dim=head_dim)

    q35 = apply_rope(q, cos[3:4], sin[3:4])
    k35 = apply_rope(k, cos[5:6], sin[5:6])
    dot_35 = (q35 * k35).sum()

    q1012 = apply_rope(q, cos[10:11], sin[10:11])
    k1012 = apply_rope(k, cos[12:13], sin[12:13])
    dot_1012 = (q1012 * k1012).sum()

    assert torch.allclose(dot_35, dot_1012, atol=1e-5)


def test_rope_cache_shapes() -> None:
    """build_rope_cache returns tensors of shape (seq_len, head_dim // 2)."""
    seq_len = 32
    head_dim = 64
    cos, sin = build_rope_cache(seq_len=seq_len, head_dim=head_dim)
    assert cos.shape == (seq_len, head_dim // 2)
    assert sin.shape == (seq_len, head_dim // 2)


def test_rope_cache_dtypes() -> None:
    """build_rope_cache respects the requested dtype and device."""
    cos_fp32, sin_fp32 = build_rope_cache(seq_len=8, head_dim=16, dtype=torch.float32)
    assert cos_fp32.dtype == torch.float32
    assert sin_fp32.dtype == torch.float32

    cos_bf16, sin_bf16 = build_rope_cache(seq_len=8, head_dim=16, dtype=torch.bfloat16)
    assert cos_bf16.dtype == torch.bfloat16
    assert sin_bf16.dtype == torch.bfloat16

    cpu = torch.device("cpu")
    cos_cpu, sin_cpu = build_rope_cache(seq_len=8, head_dim=16, device=cpu)
    assert cos_cpu.device == cpu
    assert sin_cpu.device == cpu


def test_rope_rejects_odd_head_dim() -> None:
    """build_rope_cache raises ValueError for odd head_dim."""
    with pytest.raises(ValueError):
        build_rope_cache(seq_len=8, head_dim=7)
