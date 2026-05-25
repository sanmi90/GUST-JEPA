"""Conditional diffusion refiner that takes the SL-decoder's omega output
and learns to refine it toward the DNS target.

Design
------
- The refiner is an image-to-image conditional U-Net trained as a DDPM
  epsilon-predictor. The "starting image" the refiner improves is the
  SL decoder's predicted omega; the JEPA latent ``z`` provides global
  context to discriminate cases.
- Forward pass: ``predict_noise(x_t, t, sl_omega, z)``. The U-Net takes
  ``x_t`` (the noised DNS) concatenated with ``sl_omega`` as a 2-channel
  input. The timestep ``t`` enters via sinusoidal embedding + MLP. The
  latent ``z`` enters via FiLM modulation in every ResBlock.
- Inference: DDIM sampling starting from ``x_T = noise``. Conditioning
  the reverse process on ``(sl_omega, z)`` produces a refined omega.
- The objective is residual: the network only has to model the
  reconstruction *delta* the SL decoder is missing, since ``sl_omega``
  is in the input channel of every step.

References
----------
- Ho, Jain, Abbeel 2020 (DDPM); Song et al. 2020 (DDIM).
- Saharia et al. 2022 ``Image Super-Resolution via Iterative Refinement``
  (SR3) for the image-to-image conditioning template that this module
  follows.
- PRF 2026 (Balasubramanian et al.) "Conclusions" section explicitly
  recommends diffusion as the next-step decoder refinement.

The architecture stays small (~5 M params) so it can train end-to-end on
one RTX 6000 in ~4 to 6 h. It is NOT a Stable-Diffusion-scale model;
this is a single-channel grayscale refinement task.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def sinusoidal_timestep_embedding(t: Tensor, dim: int, max_period: int = 10_000) -> Tensor:
    """Standard sinusoidal timestep embedding (Vaswani et al. 2017).

    Args:
        t: ``(B,)`` integer timesteps.
        dim: embedding width.
        max_period: largest period in the sinusoidal basis.

    Returns:
        ``(B, dim)`` float tensor.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class FiLM(nn.Module):
    """FiLM (Perez et al. 2018) per-channel affine modulation."""

    def __init__(self, cond_dim: int, n_channels: int) -> None:
        super().__init__()
        self.to_scale_shift = nn.Linear(cond_dim, 2 * n_channels)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Apply scale and shift derived from ``cond`` to ``x``.

        Args:
            x: ``(B, C, H, W)`` feature map.
            cond: ``(B, cond_dim)`` conditioning vector.

        Returns:
            ``(B, C, H, W)`` modulated feature map.
        """
        ss = self.to_scale_shift(cond)
        scale, shift = ss.chunk(2, dim=1)
        return x * (1.0 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)


class ResBlock(nn.Module):
    """GroupNorm + SiLU + Conv2D ResBlock with FiLM conditioning.

    Standard DDPM residual block: two conv layers, GroupNorm before each,
    FiLM on the first activation, skip connection through a 1x1 conv when
    channel widths differ.
    """

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int,
                 n_groups: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(min(n_groups, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.film = FiLM(cond_dim, out_ch)
        self.norm2 = nn.GroupNorm(min(n_groups, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = self.film(h, cond)
        h = F.silu(self.norm2(h))
        h = self.drop(h)
        h = self.conv2(h)
        return h + self.skip(x)


class SelfAttention2D(nn.Module):
    """Lightweight 2D self-attention at the U-Net bottleneck.

    Only used at the smallest spatial resolution (24x12) so the
    quadratic cost in tokens stays manageable.
    """

    def __init__(self, n_channels: int, n_heads: int = 4) -> None:
        super().__init__()
        if n_channels % n_heads != 0:
            raise ValueError(f"n_channels {n_channels} must be divisible by n_heads {n_heads}")
        self.heads = n_heads
        self.scale = (n_channels // n_heads) ** -0.5
        self.norm = nn.GroupNorm(min(8, n_channels), n_channels)
        self.qkv = nn.Conv2d(n_channels, 3 * n_channels, 1)
        self.proj = nn.Conv2d(n_channels, n_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.heads, C // self.heads, H * W)
        q, k, v = qkv.unbind(dim=1)
        attn = (q.transpose(-1, -2) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        out = v @ attn.transpose(-1, -2)
        out = out.reshape(B, C, H, W)
        return x + self.proj(out)


class Downsample(nn.Module):
    """2x spatial downsampling via strided conv."""

    def __init__(self, n_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, 3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """2x spatial upsampling via nearest-neighbor + conv (no checkerboard)."""

    def __init__(self, n_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class DiffusionRefiner(nn.Module):
    """Image-to-image conditional U-Net diffusion refiner.

    Args:
        in_channels: number of input channels (1 for omega).
        cond_image_channels: extra channels carrying conditioning images
            (1 for the SL decoder output -- concatenated to ``x_t``).
        z_dim: JEPA latent dimension (64 in production).
        base_channels: U-Net stem width.
        ch_mult: per-stage channel multipliers (3 entries = 3 down/up stages).
        n_resblocks: ResBlocks per stage.
        attn_bottleneck: enable self-attention at the bottleneck.
        cond_emb_dim: combined time + z embedding width.
        dropout: dropout in ResBlocks.
    """

    def __init__(
        self,
        in_channels: int = 1,
        cond_image_channels: int = 1,
        z_dim: int = 64,
        base_channels: int = 32,
        ch_mult: tuple[int, ...] = (1, 2, 4),
        n_resblocks: int = 2,
        attn_bottleneck: bool = True,
        cond_emb_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.cond_image_channels = cond_image_channels
        self.z_dim = z_dim
        self.cond_emb_dim = cond_emb_dim

        # Time + latent embedders
        self.time_mlp = nn.Sequential(
            nn.Linear(cond_emb_dim, cond_emb_dim),
            nn.SiLU(),
            nn.Linear(cond_emb_dim, cond_emb_dim),
        )
        self.z_mlp = nn.Sequential(
            nn.Linear(z_dim, cond_emb_dim),
            nn.SiLU(),
            nn.Linear(cond_emb_dim, cond_emb_dim),
        )

        # Stem
        self.stem = nn.Conv2d(in_channels + cond_image_channels, base_channels, 3, padding=1)

        # Down path
        widths = [base_channels * m for m in (1, *ch_mult)]
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(len(ch_mult)):
            in_w, out_w = widths[i], widths[i + 1]
            stage = nn.ModuleList(
                [ResBlock(in_w if j == 0 else out_w, out_w, cond_emb_dim, dropout=dropout)
                 for j in range(n_resblocks)]
            )
            self.down_blocks.append(stage)
            self.downsamples.append(Downsample(out_w))

        # Bottleneck (no downsample, two ResBlocks + optional self-attn)
        bot_w = widths[-1]
        self.bot_res1 = ResBlock(bot_w, bot_w, cond_emb_dim, dropout=dropout)
        self.bot_attn = SelfAttention2D(bot_w) if attn_bottleneck else nn.Identity()
        self.bot_res2 = ResBlock(bot_w, bot_w, cond_emb_dim, dropout=dropout)

        # Up path: mirror of down, with skip-concat
        # At up-stage i (iterating reversed range(len(ch_mult))):
        #   - h enters with widths[i+1] channels (from the prior up or bottleneck)
        #   - upsample preserves channel count
        #   - skip has widths[i+1] channels (output of the matching down stage)
        #   - after concat, the first ResBlock sees 2 * widths[i+1] channels and emits widths[i]
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i in reversed(range(len(ch_mult))):
            in_w, out_w = widths[i + 1], widths[i]
            self.upsamples.append(Upsample(in_w))
            stage = nn.ModuleList()
            for j in range(n_resblocks):
                in_for_block = 2 * in_w if j == 0 else out_w
                stage.append(ResBlock(in_for_block, out_w, cond_emb_dim, dropout=dropout))
            self.up_blocks.append(stage)

        # Output head
        self.out_norm = nn.GroupNorm(min(8, base_channels), base_channels)
        self.out_conv = nn.Conv2d(base_channels, in_channels, 3, padding=1)

    def cond_embed(self, t: Tensor, z: Tensor) -> Tensor:
        """Combine timestep + JEPA latent into a single conditioning vector."""
        t_emb = sinusoidal_timestep_embedding(t, self.cond_emb_dim)
        t_emb = self.time_mlp(t_emb)
        z_emb = self.z_mlp(z)
        return t_emb + z_emb

    def forward(self, x_t: Tensor, t: Tensor, sl_omega: Tensor, z: Tensor) -> Tensor:
        """Predict the noise residual for the DDPM reverse step.

        Args:
            x_t: ``(B, in_channels, H, W)`` noised input.
            t: ``(B,)`` integer timesteps.
            sl_omega: ``(B, cond_image_channels, H, W)`` SL decoder output
                that the refiner is conditioning on.
            z: ``(B, z_dim)`` JEPA latent for the same sample.

        Returns:
            ``(B, in_channels, H, W)`` predicted noise.
        """
        cond = self.cond_embed(t, z)

        # Stem with concatenated condition image
        h = self.stem(torch.cat([x_t, sl_omega], dim=1))

        skips: list[Tensor] = []
        # Down path
        for stage, down in zip(self.down_blocks, self.downsamples):
            for block in stage:
                h = block(h, cond)
            skips.append(h)
            h = down(h)

        # Bottleneck
        h = self.bot_res1(h, cond)
        h = self.bot_attn(h)
        h = self.bot_res2(h, cond)

        # Up path
        for up, stage, skip in zip(self.upsamples, self.up_blocks, reversed(skips)):
            h = up(h)
            # Concatenate with skip on the first block of this stage
            for j, block in enumerate(stage):
                if j == 0:
                    h = torch.cat([h, skip], dim=1)
                h = block(h, cond)

        h = F.silu(self.out_norm(h))
        return self.out_conv(h)


class NoiseSchedule:
    """Linear beta schedule + precomputed alpha-bar values for DDPM/DDIM."""

    def __init__(self, n_timesteps: int = 1000, beta_start: float = 1e-4,
                 beta_end: float = 0.02) -> None:
        self.n_timesteps = n_timesteps
        betas = torch.linspace(beta_start, beta_end, n_timesteps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

    def to(self, device: torch.device) -> "NoiseSchedule":
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        self.sqrt_alpha_bars = self.sqrt_alpha_bars.to(device)
        self.sqrt_one_minus_alpha_bars = self.sqrt_one_minus_alpha_bars.to(device)
        return self

    def q_sample(self, x_0: Tensor, t: Tensor, noise: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        """Forward diffusion: sample ``x_t = sqrt(a_bar_t) x_0 + sqrt(1-a_bar_t) eps``.

        Returns ``(x_t, noise)``.
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        sa = self.sqrt_alpha_bars[t].view(-1, 1, 1, 1)
        so = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1)
        x_t = sa * x_0 + so * noise
        return x_t, noise


@torch.no_grad()
def ddim_sample(
    model: DiffusionRefiner,
    schedule: NoiseSchedule,
    sl_omega: Tensor,
    z: Tensor,
    n_steps: int = 50,
    eta: float = 0.0,
    init_from_sl: bool = True,
) -> Tensor:
    """DDIM reverse sampling for refinement.

    Args:
        model: trained DiffusionRefiner.
        schedule: noise schedule used at training time.
        sl_omega: ``(B, 1, H, W)`` SL decoder output, used as conditioning
            and (when ``init_from_sl=True``) as the starting point with
            a small initial noise level.
        z: ``(B, z_dim)`` JEPA latent.
        n_steps: number of DDIM denoising steps (typical 20 to 50).
        eta: DDIM stochasticity (0 = deterministic, 1 = full DDPM).
        init_from_sl: if True, start at ``x_T = q_sample(sl_omega, T_start)``
            with a moderate ``T_start`` (a fraction of the full schedule)
            rather than pure noise. This is the image-to-image refinement
            convention (SR3) and converges much faster than starting from
            ``N(0, I)``.

    Returns:
        ``(B, 1, H, W)`` refined omega.
    """
    device = sl_omega.device
    B = sl_omega.shape[0]
    T_total = schedule.n_timesteps

    if init_from_sl:
        # Start at moderate noise level (T_start = 60 % of full schedule)
        t_start = int(0.6 * T_total) - 1
        t_start_b = torch.full((B,), t_start, device=device, dtype=torch.long)
        x_t, _ = schedule.q_sample(sl_omega, t_start_b)
        time_steps = torch.linspace(t_start, 0, n_steps + 1, dtype=torch.long, device=device)
    else:
        x_t = torch.randn_like(sl_omega)
        time_steps = torch.linspace(T_total - 1, 0, n_steps + 1, dtype=torch.long, device=device)

    for i in range(n_steps):
        t_now = time_steps[i]
        t_next = time_steps[i + 1]
        t_b = torch.full((B,), t_now.item(), device=device, dtype=torch.long)
        eps = model(x_t, t_b, sl_omega, z)
        alpha_bar_now = schedule.alpha_bars[t_now]
        alpha_bar_next = schedule.alpha_bars[t_next] if t_next >= 0 else torch.tensor(1.0, device=device)
        x0_pred = (x_t - torch.sqrt(1 - alpha_bar_now) * eps) / torch.sqrt(alpha_bar_now)
        # DDIM step
        sigma = eta * torch.sqrt((1 - alpha_bar_next) / (1 - alpha_bar_now)) * torch.sqrt(1 - alpha_bar_now / alpha_bar_next)
        dir_xt = torch.sqrt(torch.clamp(1 - alpha_bar_next - sigma ** 2, min=0.0)) * eps
        noise = torch.randn_like(x_t) if eta > 0 else 0.0
        x_t = torch.sqrt(alpha_bar_next) * x0_pred + dir_xt + sigma * noise

    return x_t


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
