from __future__ import annotations
import torch
from torch import nn

from src.models.vjepa_tokenizer import TubeletEmbed


def test_tubelet_token_count_and_shape() -> None:
    """(B,T,1,H,W) -> (B, N, D) with N = (T/2)(H/16)(W/16) = 1152, D=384."""
    torch.manual_seed(0)
    tok = TubeletEmbed()
    x = torch.randn(2, 32, 1, 192, 96)
    z = tok(x)
    assert z.shape == (2, 1152, 384)
    assert tok.grid == (16, 12, 6)
    assert tok.num_tokens == 1152


def test_tubelet_pos_embed_added_and_deterministic() -> None:
    """Pos embed is a fixed (1,N,D) buffer added to tokens; two passes match."""
    torch.manual_seed(0)
    tok = TubeletEmbed().eval()
    x = torch.randn(1, 32, 1, 192, 96)
    with torch.no_grad():
        a, b = tok(x), tok(x)
    assert torch.allclose(a, b)
    assert tok.pos_embed.shape == (1, 1152, 384)


from src.models.vjepa_masking import MultiBlockMask


def test_mask_shape_and_constant_visible_count() -> None:
    """Mask is (B,N) bool; every row masks the same count (so visible is constant)."""
    torch.manual_seed(0)
    m = MultiBlockMask(grid=(16, 12, 6), mask_ratio=0.8)
    mask = m.sample(batch_size=4)
    assert mask.shape == (4, 1152) and mask.dtype == torch.bool
    counts = mask.sum(dim=1)
    assert (counts == counts[0]).all(), "visible count must be constant across batch"
    assert 0 < int(counts[0]) < 1152


def test_mask_leaves_some_visible_and_varies_per_row() -> None:
    """At least ~10% visible, and rows are not all identical."""
    torch.manual_seed(1)
    m = MultiBlockMask(grid=(16, 12, 6), mask_ratio=0.8)
    mask = m.sample(batch_size=8)
    vis_frac = 1.0 - mask.float().mean().item()
    assert vis_frac >= 0.10
    assert not torch.equal(mask[0], mask[1])
