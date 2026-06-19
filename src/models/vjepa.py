"""V-JEPA objective: masked space-time feature prediction with an EMA target
encoder (Bardes et al., arXiv:2404.08471). Reuses the project's _ViTBlock."""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.models.encoder import _ViTBlock
from src.models.vjepa_masking import MultiBlockMask
from src.models.vjepa_pool import frame_mean_pool
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
        n_lift: int = 1,
        wake_dim: int = 80,
        wake_head_hidden: int = 128,
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
        gt, gh, gw = self.grid
        coords = (
            torch.stack(
                torch.meshgrid(torch.arange(gt), torch.arange(gh), torch.arange(gw), indexing="ij"),
                dim=-1,
            )
            .reshape(-1, 3)
            .float()
        )  # (N, 3) token (t,h,w) coords in TubeletEmbed flatten order
        self.register_buffer("grid_coords", coords, persistent=False)
        self.lift_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, n_lift)
        )
        self.wake_head = nn.Sequential(
            nn.Linear(hidden, wake_head_hidden), nn.GELU(), nn.Linear(wake_head_hidden, wake_dim)
        )

    @torch.no_grad()
    def ema_update(self, momentum: float) -> None:
        for pt, pc in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            pt.mul_(momentum).add_(pc.detach(), alpha=1.0 - momentum)
        for bt, bc in zip(self.target_encoder.buffers(), self.context_encoder.buffers()):
            bt.copy_(bc)

    def encode_tokens(self, omega: Tensor) -> Tensor:
        """Eval helper: full-clip context-encoder tokens (B, N, D)."""
        return self.context_encoder(self.tokenizer(omega))

    def forward(
        self,
        omega: Tensor,
        mask: Tensor | None = None,
        lam_ctx: float = 0.0,
        cl_future: Tensor | None = None,
        wake_target: Tensor | None = None,
        lift_w: float = 0.0,
        wake_w: float = 0.0,
    ) -> dict[str, Tensor]:
        tok = self.tokenizer(omega)  # (B,N,D)
        b, n, d = tok.shape
        if mask is None:
            mask = self.masker.sample(b).to(tok.device)
        vis = ~mask
        n_vis = int(vis[0].sum())
        n_mask = n - n_vis
        vis_idx = vis.nonzero(as_tuple=False)[:, 1].view(b, n_vis)
        mask_idx = mask.nonzero(as_tuple=False)[:, 1].view(b, n_mask)
        ctx_tok = torch.gather(tok, 1, vis_idx[:, :, None].expand(b, n_vis, d))
        ctx = self.context_encoder(ctx_tok)
        with torch.no_grad():
            tgt_full = self.target_encoder(self.tokenizer(omega))
            tgt_full = F.layer_norm(tgt_full, (d,))
        tgt_m = torch.gather(tgt_full, 1, mask_idx[:, :, None].expand(b, n_mask, d))
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
        pred_full = self.pred_proj(seq)  # (b, n_vis + n_mask, d)
        pred_m = pred_full[:, n_vis:, :]
        l_pred = F.smooth_l1_loss(pred_m, tgt_m, beta=0.5)
        out = {"loss": l_pred, "l_pred": l_pred.detach(), "mask": mask}
        if lam_ctx > 0:
            pred_v = pred_full[:, :n_vis, :]
            tgt_v = torch.gather(tgt_full, 1, vis_idx[:, :, None].expand(b, n_vis, d))
            vis_c = self.grid_coords[vis_idx]  # (b, n_vis, 3)
            mask_c = self.grid_coords[mask_idx]  # (b, n_mask, 3)
            dmin = torch.cdist(vis_c, mask_c).min(dim=-1).values  # (b, n_vis)
            w = lam_ctx / torch.sqrt(dmin.clamp(min=1.0))  # (b, n_vis)
            l_elem = F.smooth_l1_loss(pred_v, tgt_v, beta=0.5, reduction="none").mean(
                -1
            )  # (b,n_vis)
            l_ctx = (w * l_elem).mean()
            out["loss"] = l_pred + l_ctx
            out["l_ctx"] = l_ctx.detach()
        if lift_w > 0.0 or wake_w > 0.0:
            feat_full = self.context_encoder(tok)  # full-clip, with grad (B,N,hidden)
            fp = frame_mean_pool(feat_full, self.grid)  # (B, gt, hidden)
            gt = fp.shape[1]
            if lift_w > 0.0 and cl_future is not None:
                if cl_future.dim() == 2:
                    cl_future = cl_future.unsqueeze(-1)
                cl_ds = cl_future[:, ::2, :][:, :gt, :]  # (B,32,n)->(B,gt,n)
                l_lift = F.smooth_l1_loss(self.lift_head(fp), cl_ds, beta=0.5)
                out["loss"] = out["loss"] + lift_w * l_lift
                out["l_lift"] = l_lift.detach()
            if wake_w > 0.0 and wake_target is not None:
                if wake_target.dim() == 2:
                    wake_target = wake_target.unsqueeze(-1)
                wk_ds = wake_target[:, ::2, :][:, :gt, :]
                l_wake = F.smooth_l1_loss(self.wake_head(fp), wk_ds, beta=0.5)
                out["loss"] = out["loss"] + wake_w * l_wake
                out["l_wake"] = l_wake.detach()
        return out
