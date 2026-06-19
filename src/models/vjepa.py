"""V-JEPA objective: masked space-time feature prediction with an EMA target
encoder (Bardes et al., arXiv:2404.08471). Reuses the project's _ViTBlock."""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.models.encoder import _ViTBlock
from src.models.vjepa_masking import MultiBlockMask
from src.models.vjepa_tokenizer import TubeletEmbed, sincos_3d


class _Encoder(nn.Module):
    """ViT stack + final LayerNorm operating on a token sequence (B, L, D)."""

    def __init__(
        self, depth: int, hidden: int, heads: int, mlp_ratio: float, dropout: float
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [_ViTBlock(hidden, heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: Tensor) -> Tensor:
        for b in self.blocks:
            x = b(x)
        return self.norm(x)


class VJEPA(nn.Module):
    def __init__(
        self,
        hidden: int = 384,
        depth: int = 8,
        heads: int = 6,
        mlp_ratio: float = 4.0,
        pred_hidden: int = 192,
        pred_depth: int = 6,
        pred_heads: int = 6,
        dropout: float = 0.0,
        mask_ratio: float = 0.8,
    ) -> None:
        super().__init__()
        self.tokenizer = TubeletEmbed(hidden=hidden)
        self.grid = self.tokenizer.grid
        self.num_tokens = self.tokenizer.num_tokens
        self.context_encoder = _Encoder(depth, hidden, heads, mlp_ratio, dropout)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.masker = MultiBlockMask(grid=self.grid, mask_ratio=mask_ratio)
        self.pred_embed = nn.Linear(hidden, pred_hidden)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_hidden))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.pred_blocks = nn.ModuleList(
            [_ViTBlock(pred_hidden, pred_heads, mlp_ratio, dropout) for _ in range(pred_depth)]
        )
        self.pred_norm = nn.LayerNorm(pred_hidden)
        self.pred_proj = nn.Linear(pred_hidden, hidden)
        pred_pos = sincos_3d(self.grid, pred_hidden).unsqueeze(0)
        self.register_buffer("pred_pos", pred_pos, persistent=False)

    @torch.no_grad()
    def ema_update(self, momentum: float) -> None:
        for pt, pc in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            pt.mul_(momentum).add_(pc.detach(), alpha=1.0 - momentum)
        for bt, bc in zip(self.target_encoder.buffers(), self.context_encoder.buffers()):
            bt.copy_(bc)

    def encode_tokens(self, omega: Tensor) -> Tensor:
        """Eval helper: full-clip context-encoder tokens (B, N, D)."""
        return self.context_encoder(self.tokenizer(omega))

    def forward(self, omega: Tensor, mask: Tensor | None = None) -> dict[str, Tensor]:
        tok = self.tokenizer(omega)  # (B,N,D)
        b, n, d = tok.shape
        if mask is None:
            mask = self.masker.sample(b).to(tok.device)  # (B,N) True=masked
        vis = ~mask
        n_vis = int(vis[0].sum())
        n_mask = n - n_vis
        vis_idx = vis.nonzero(as_tuple=False)[:, 1].view(b, n_vis)
        mask_idx = mask.nonzero(as_tuple=False)[:, 1].view(b, n_mask)
        # context: encode visible tokens only
        ctx_tok = torch.gather(tok, 1, vis_idx[:, :, None].expand(b, n_vis, d))
        ctx = self.context_encoder(ctx_tok)  # (B, n_vis, D)
        # target: full clip through EMA encoder, stop-grad, per-token LayerNorm
        with torch.no_grad():
            tgt_full = self.target_encoder(self.tokenizer(omega))  # (B,N,D)
            tgt_full = F.layer_norm(tgt_full, (d,))
        tgt = torch.gather(tgt_full, 1, mask_idx[:, :, None].expand(b, n_mask, d))
        # predictor: context tokens + mask tokens, each stamped with its position
        ph = self.pred_pos.shape[-1]
        pos = self.pred_pos.expand(b, n, -1)
        vis_pos = torch.gather(pos, 1, vis_idx[:, :, None].expand(b, n_vis, ph))
        mask_pos = torch.gather(pos, 1, mask_idx[:, :, None].expand(b, n_mask, ph))
        ctx_p = self.pred_embed(ctx) + vis_pos
        mtok = self.mask_token.expand(b, n_mask, -1) + mask_pos
        seq = torch.cat([ctx_p, mtok], dim=1)
        for blk in self.pred_blocks:
            seq = blk(seq)
        seq = self.pred_norm(seq)
        pred_mask = self.pred_proj(seq[:, n_vis:, :])  # (B, n_mask, D)
        loss = F.smooth_l1_loss(pred_mask, tgt, beta=0.5)
        return {"loss": loss, "mask": mask}
