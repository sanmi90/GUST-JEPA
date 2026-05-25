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


class UnconditionedPredictorBlock(nn.Module):
    """Standard pre-norm transformer block (LayerNorm + attention + MLP).

    Used when ``cond_dim = 0``: there is no conditioning vector, so the
    AdaLN-Zero path collapses to the identity (gate stays 0 forever),
    which would freeze the predictor. This block uses learnable affine
    LayerNorms instead. Same residual structure as ``PredictorBlock``.

    Inherits the F-NC factorial variant motivation from
    SESSION6_FACTORIAL_DIAGNOSTIC.md Step 3, F-NC.
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
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=True)
        self.attn = CausalSelfAttentionWithRoPE(
            hidden_dim=hidden_dim,
            heads=heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=True)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, c_seq: Tensor | None = None) -> Tensor:
        """``c_seq`` is accepted for API symmetry with ``PredictorBlock`` and ignored."""
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


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
        if cond_dim < 0:
            raise ValueError(f"cond_dim must be >= 0; got {cond_dim}")
        self.latent_dim = latent_dim
        self.cond_dim = int(cond_dim)
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        self.embed = nn.Linear(latent_dim, hidden_dim)
        if self.cond_dim > 0:
            self.cond_mlp: nn.Module = nn.Sequential(
                nn.Linear(cond_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            block_cls = PredictorBlock
        else:
            self.cond_mlp = nn.Identity()
            block_cls = UnconditionedPredictorBlock
        self.blocks = nn.ModuleList(
            [
                block_cls(
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

    def forward(self, z: Tensor, cond: Tensor | None = None) -> Tensor:
        """Teacher-forced next-step prediction over a latent sub-trajectory.

        Args:
            z: ``(B, T, latent_dim)`` encoder latents.
            cond: ``(B, cond_dim)`` static episode descriptor; broadcast
                internally to ``(B, T, hidden_dim)``. Ignored entirely when
                the predictor was constructed with ``cond_dim = 0`` (F-NC
                variant); pass ``None`` or any tensor in that case.

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
        if self.cond_dim > 0:
            if cond is None:
                raise ValueError("cond is required when predictor cond_dim > 0")
            c_seq = self.cond_mlp(cond).unsqueeze(1).expand(-1, T, -1)
        else:
            c_seq = None
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


class ReversePredictor(nn.Module):
    """Reverse-factorisation transformer: forces -> encoder latents.

    Maps a time series of per-frame force coefficients ``(C_L(t), C_D(t))``
    plus the static episode descriptor ``c = (G, D, Y)`` to the corresponding
    frozen-encoder latent ``z_t in R^latent_dim``. The encoder ``E`` is held
    frozen during training; the target is ``z_t = E(omega_t)`` (Session 14
    Thrust 5).

    Architecturally identical to :class:`AutoregressivePredictor` (DiT-style
    AdaLN-Zero blocks, RoPE on Q/K, causal mask) except:

    - the input embedding is ``Linear(input_dim=2, hidden_dim)`` so the
      transformer ingests force time-series instead of latent time-series, and
    - the BatchNorm on the output head is OPTIONAL via ``output_norm``. The
      reverse predictor regresses against a (BatchNorm-projected) target
      latent so passing the prediction through another BatchNorm is
      unnecessary; keep the default ``output_norm="none"``.

    The causal mask remains: prediction of ``z_t`` only sees
    ``(C_L, C_D)[<=t]`` (and the static ``c``), matching the JEPA
    encoder-as-causal-observer semantics used in the forward predictor.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        input_dim: int = 2,
        cond_dim: int = 3,
        hidden_dim: int = 384,
        depth: int = 6,
        heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_seq_len: int = 32,
        output_norm: str = "none",
    ) -> None:
        super().__init__()
        if cond_dim < 0:
            raise ValueError(f"cond_dim must be >= 0; got {cond_dim}")
        if input_dim <= 0:
            raise ValueError(f"input_dim must be > 0; got {input_dim}")
        if output_norm not in ("none", "batchnorm"):
            raise ValueError(
                f"output_norm must be 'none' or 'batchnorm'; got {output_norm!r}"
            )
        self.latent_dim = int(latent_dim)
        self.input_dim = int(input_dim)
        self.cond_dim = int(cond_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_seq_len = int(max_seq_len)
        self.output_norm = output_norm

        self.embed = nn.Linear(input_dim, hidden_dim)
        if self.cond_dim > 0:
            self.cond_mlp: nn.Module = nn.Sequential(
                nn.Linear(cond_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            block_cls = PredictorBlock
        else:
            self.cond_mlp = nn.Identity()
            block_cls = UnconditionedPredictorBlock
        self.blocks = nn.ModuleList(
            [
                block_cls(
                    hidden_dim=hidden_dim,
                    heads=heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    max_seq_len=max_seq_len,
                )
                for _ in range(depth)
            ]
        )
        layers: list[nn.Module] = [nn.Linear(hidden_dim, latent_dim)]
        if output_norm == "batchnorm":
            layers.append(nn.BatchNorm1d(latent_dim))
        self.out_proj = nn.Sequential(*layers)

    def forward(self, forces: Tensor, cond: Tensor | None = None) -> Tensor:
        """Predicts the latent trajectory from the force trajectory.

        Args:
            forces: ``(B, T, input_dim)`` per-frame (C_L, C_D) (default
                ``input_dim = 2``). Already standardised by the dataset.
            cond: ``(B, cond_dim)`` static episode descriptor; broadcast
                internally to ``(B, T, hidden_dim)``. Required when
                ``cond_dim > 0``.

        Returns:
            ``z_hat`` of shape ``(B, T, latent_dim)``. ``z_hat[:, t, :]`` is
            the predicted encoder latent at frame ``t``, depending causally
            on ``forces[:, :t + 1, :]`` and the static ``cond``.
        """
        B, T, F = forces.shape
        if F != self.input_dim:
            raise ValueError(
                f"forces.shape[-1]={F} does not match input_dim={self.input_dim}"
            )
        if T > self.max_seq_len:
            raise ValueError(
                f"sequence length {T} exceeds max_seq_len={self.max_seq_len}"
            )
        x = self.embed(forces)
        if self.cond_dim > 0:
            if cond is None:
                raise ValueError("cond is required when ReversePredictor cond_dim > 0")
            c_seq = self.cond_mlp(cond).unsqueeze(1).expand(-1, T, -1)
        else:
            c_seq = None
        for block in self.blocks:
            x = block(x, c_seq)
        out = self.out_proj(x.flatten(0, 1))
        return out.view(B, T, -1)
