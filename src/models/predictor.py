"""Autoregressive transformer predictor for vortex-gust JEPA.

References:
    Maes, Le Lidec, Scieur, LeCun, Balestriero. "LeWorldModel: Stable
    End-to-End Joint-Embedding Predictive Architecture from Pixels."
    arXiv:2603.19312, 2026, Section 3.1 and Appendix D (predictor wiring
    and BatchNorm projection).
    Peebles, Xie. "Scalable Diffusion Models with Transformers."
    arXiv:2212.09748, 2023, figure 3 (DiT AdaLN-Zero block: two AdaLN
    modules per block, one before attention and one before the MLP).
    Su, Lu, Pan, Murtadha, Wen, Liu. "RoFormer: Enhanced Transformer
    with Rotary Position Embedding." arXiv:2104.09864, 2021, Section 3.4
    (RoPE applied to query and key only, never to value).

Conditioning is the static episode descriptor ``c = (G, D, Y)``
(``cond_dim = 3``); the time-varying phase variable phi_t is not part of
the default (HANDOFF.md D16) but ``cond_dim = 4`` is a one-line switch.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.models.adaln import AdaLN
from src.models.rope import apply_rope, build_rope_cache


class CausalSelfAttentionWithRoPE(nn.Module):
    """Multi-head causal self-attention with RoPE on Q and K.

    Values are not rotated; only the query and key tensors pass through
    ``apply_rope`` (RoFormer Section 3.4). The causal mask is set via
    ``is_causal=True`` to ``F.scaled_dot_product_attention``.
    """

    def __init__(self, hidden_dim: int, heads: int, dropout: float, max_seq_len: int) -> None:
        """Sets up the QKV / output projections and the precomputed RoPE cache.

        Args:
            hidden_dim: Residual stream width.
            heads: Number of attention heads. ``hidden_dim`` must be divisible
                by ``heads`` and the per-head dim must be even (RoPE requirement).
            dropout: Attention dropout probability (used only in train mode).
            max_seq_len: Largest sequence length the cached RoPE angles support.
        """
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})")
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.dropout = dropout
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        cos, sin = build_rope_cache(max_seq_len, self.head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """Applies causal self-attention with RoPE on Q and K only.

        Args:
            x: ``(B, T, hidden_dim)`` residual stream.

        Returns:
            ``(B, T, hidden_dim)`` post-projection tensor.
        """
        B, T, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = apply_rope(q, self.rope_cos[:T], self.rope_sin[:T])
        k = apply_rope(k, self.rope_cos[:T], self.rope_sin[:T])
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True
        )
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.proj(out)


class PredictorBlock(nn.Module):
    """DiT-style AdaLN-Zero block with causal RoPE attention.

    Two AdaLN modules per block: one before attention, one before the MLP.
    The LayerNorms have ``elementwise_affine=False`` so the AdaLN
    ``(shift, scale, gate)`` triples are the only affine modulation; with a
    learnable affine inside LayerNorm, the AdaLN-Zero identity property at
    init would no longer hold. This follows DiT (arXiv:2212.09748).
    """

    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        mlp_ratio: float,
        dropout: float,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.adaln1 = AdaLN(hidden_dim=hidden_dim, cond_dim=hidden_dim)
        self.attn = CausalSelfAttentionWithRoPE(
            hidden_dim=hidden_dim,
            heads=heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.adaln2 = AdaLN(hidden_dim=hidden_dim, cond_dim=hidden_dim)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, c_seq: Tensor) -> Tensor:
        """Applies one DiT-style block.

        Args:
            x: ``(B, T, hidden_dim)`` residual stream.
            c_seq: ``(B, T, hidden_dim)`` conditioning sequence (constant in
                time when only ``c = (G, D, Y)`` is used).

        Returns:
            ``(B, T, hidden_dim)`` updated residual stream.
        """
        shift1, scale1, gate1 = self.adaln1(c_seq)
        h = self.norm1(x) * (1 + scale1) + shift1
        x = x + gate1 * self.attn(h)
        shift2, scale2, gate2 = self.adaln2(c_seq)
        h = self.norm2(x) * (1 + scale2) + shift2
        x = x + gate2 * self.mlp(h)
        return x


class AutoregressivePredictor(nn.Module):
    """Autoregressive transformer over latent trajectories.

    AdaLN-Zero conditioning on the static episode descriptor ``c = (G, D, Y)``
    (HANDOFF.md D16), RoPE temporal positions on Q and K only, causal
    attention mask, and a BatchNorm-projected output head matching the
    encoder projector so predicted and target embeddings live in the same
    space (LeWM Section 3.1; HANDOFF.md D17).

    Attributes:
        embed: ``Linear(latent_dim, hidden_dim)`` input projection.
        cond_mlp: 2-layer MLP producing the conditioning vector that feeds
            every AdaLN inside the block stack.
        blocks: ``depth`` ``PredictorBlock`` instances.
        out_proj: ``Linear(hidden_dim, latent_dim)`` -> ``BatchNorm1d``
            output head (BatchNorm matches the encoder projector).
    """

    def __init__(
        self,
        latent_dim: int = 32,
        cond_dim: int = 3,
        hidden_dim: int = 384,
        depth: int = 6,
        heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_seq_len: int = 32,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        self.embed = nn.Linear(latent_dim, hidden_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [
                PredictorBlock(
                    hidden_dim=hidden_dim,
                    heads=heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    max_seq_len=max_seq_len,
                )
                for _ in range(depth)
            ]
        )
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
        )

    def forward(self, z: Tensor, cond: Tensor) -> Tensor:
        """Teacher-forced next-step prediction over a latent sub-trajectory.

        Args:
            z: ``(B, T, latent_dim)`` encoder latents.
            cond: ``(B, cond_dim)`` static episode descriptor; broadcast
                internally to ``(B, T, hidden_dim)``.

        Returns:
            ``z_hat`` of shape ``(B, T, latent_dim)``. ``z_hat[:, t, :]`` is
            the next-step prediction of ``z[:, t + 1, :]`` from ``z[:, :t + 1, :]``.
            The last position ``z_hat[:, T - 1, :]`` is the prediction of the
            would-be ``(T + 1)``-th frame, which is the natural rollout step.
        """
        B, T, _ = z.shape
        if T > self.max_seq_len:
            raise ValueError(f"sequence length {T} exceeds max_seq_len={self.max_seq_len}")
        x = self.embed(z)
        c_seq = self.cond_mlp(cond).unsqueeze(1).expand(-1, T, -1)
        for block in self.blocks:
            x = block(x, c_seq)
        out = self.out_proj(x.flatten(0, 1))
        return out.view(B, T, -1)

    def rollout(self, z_init: Tensor, cond: Tensor, steps: int) -> Tensor:
        """Open-loop autoregressive rollout from a seed sub-trajectory.

        Args:
            z_init: ``(B, T_init, latent_dim)`` seed latents. ``T_init`` may
                be 1 (rollout from a single frame) or larger (warm-start with
                several ground-truth frames).
            cond: ``(B, cond_dim)`` static episode descriptor.
            steps: Number of additional frames to predict beyond ``T_init``.

        Returns:
            ``z_full`` of shape ``(B, T_init + steps, latent_dim)``. The first
            ``T_init`` positions equal ``z_init`` exactly; subsequent positions
            are the rolled-out predictions.
        """
        z_full = z_init
        for _ in range(steps):
            z_hat = self.forward(z_full, cond)
            z_full = torch.cat([z_full, z_hat[:, -1:, :]], dim=1)
        return z_full
