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
