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


from src.models.vjepa import VJEPA


def _batch(b=2):
    return torch.randn(b, 32, 1, 192, 96)


def test_vjepa_forward_returns_scalar_loss() -> None:
    torch.manual_seed(0)
    model = VJEPA(depth=2, pred_depth=2)
    out = model(_batch())
    assert out["loss"].ndim == 0 and torch.isfinite(out["loss"])


def test_vjepa_target_encoder_has_no_grad() -> None:
    model = VJEPA(depth=2, pred_depth=2)
    assert all(not p.requires_grad for p in model.target_encoder.parameters())
    assert any(p.requires_grad for p in model.context_encoder.parameters())


def test_vjepa_ema_moves_target_toward_context() -> None:
    torch.manual_seed(0)
    model = VJEPA(depth=2, pred_depth=2)
    with torch.no_grad():
        for p in model.context_encoder.parameters():
            p.add_(1.0)
    tgt_before = next(iter(model.target_encoder.parameters())).clone()
    ctx = next(iter(model.context_encoder.parameters())).clone()
    model.ema_update(momentum=0.9)
    tgt_after = next(iter(model.target_encoder.parameters()))
    assert torch.norm(tgt_after - ctx) < torch.norm(tgt_before - ctx)


def test_vjepa_encode_tokens_shape() -> None:
    model = VJEPA(depth=2, pred_depth=2).eval()
    with torch.no_grad():
        tok = model.encode_tokens(_batch(1))
    assert tok.shape == (1, 1152, 384)


def test_vjepa_overfits_one_batch_cpu() -> None:
    """Loss on a fixed batch + mask must drop substantially after a few steps."""
    torch.manual_seed(0)
    model = VJEPA(depth=2, pred_depth=2)
    x = _batch(2)
    mask = model.masker.sample(2)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    losses = []
    for _ in range(30):
        opt.zero_grad()
        out = model(x, mask=mask)
        out["loss"].backward()
        opt.step()
        model.ema_update(momentum=0.99)
        losses.append(float(out["loss"].detach()))
    assert losses[-1] < 0.6 * losses[0], f"no overfit: {losses[0]:.4f} -> {losses[-1]:.4f}"


from src.models.vjepa_pool import frame_mean_pool


def test_frame_mean_pool_shape() -> None:
    """(B,N,D) tokens with grid (16,12,6) -> (B,16,D) frame means."""
    tok = torch.randn(2, 1152, 384)
    fp = frame_mean_pool(tok, grid=(16, 12, 6))
    assert fp.shape == (2, 16, 384)


def test_frame_mean_pool_is_mean_over_spatial() -> None:
    """Frame f's pooled vector equals the mean of that frame's gh*gw tokens."""
    tok = torch.randn(1, 1152, 384)
    fp = frame_mean_pool(tok, grid=(16, 12, 6))
    # frame 0 occupies tokens [0 : 12*6]
    expected0 = tok[0, 0:72].mean(dim=0)
    assert torch.allclose(fp[0, 0], expected0, atol=1e-5)
