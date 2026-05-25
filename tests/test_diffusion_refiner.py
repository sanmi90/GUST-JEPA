"""Unit tests for src/models/diffusion_refiner.py."""

from __future__ import annotations

import math

import torch

from src.models.diffusion_refiner import (
    DiffusionRefiner,
    NoiseSchedule,
    SelfAttention2D,
    count_parameters,
    ddim_sample,
    sinusoidal_timestep_embedding,
)


def test_sinusoidal_timestep_embedding_shape() -> None:
    """Embedding has the requested width and is finite."""
    t = torch.tensor([0, 100, 500, 999], dtype=torch.long)
    emb = sinusoidal_timestep_embedding(t, dim=128)
    assert emb.shape == (4, 128)
    assert torch.isfinite(emb).all()


def test_diffusion_refiner_forward_shape() -> None:
    """Refiner accepts (B, 1, 192, 96) + (B, 1, 192, 96) condition + (B, 64) latent."""
    model = DiffusionRefiner(in_channels=1, cond_image_channels=1, z_dim=64,
                              base_channels=16, ch_mult=(1, 2, 4))
    x_t = torch.randn(2, 1, 192, 96)
    t = torch.tensor([100, 500], dtype=torch.long)
    sl_omega = torch.randn(2, 1, 192, 96)
    z = torch.randn(2, 64)
    out = model(x_t, t, sl_omega, z)
    assert out.shape == (2, 1, 192, 96)
    assert torch.isfinite(out).all()


def test_diffusion_refiner_param_count() -> None:
    """At base_channels=32 and the default ch_mult=(1,2,4), param count is ~5 M."""
    model = DiffusionRefiner(base_channels=32, ch_mult=(1, 2, 4))
    n = count_parameters(model)
    assert 1_000_000 < n < 15_000_000, f"unexpected param count {n}"


def test_diffusion_refiner_backward_flows() -> None:
    """Loss has non-zero gradients on the refiner weights."""
    model = DiffusionRefiner(base_channels=16, ch_mult=(1, 2, 4))
    x_t = torch.randn(2, 1, 192, 96, requires_grad=False)
    t = torch.tensor([200, 800], dtype=torch.long)
    sl_omega = torch.randn(2, 1, 192, 96)
    z = torch.randn(2, 64)
    out = model(x_t, t, sl_omega, z)
    loss = (out ** 2).mean()
    loss.backward()
    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None and (p.grad != 0).any())
    assert n_with_grad > 5, f"too few parameters received gradients ({n_with_grad})"


def test_noise_schedule_q_sample_at_t0_equals_x0() -> None:
    """At t=0, q_sample returns essentially the original signal (1% noise).

    Linear schedule from beta=1e-4 puts alpha_bar_0 ~ 0.9999 so the noise
    contribution per element is ~sqrt(1e-4) ~ 0.01. Test mean abs delta,
    not pointwise tolerance, because small-magnitude x_0 elements can
    flip sign under 1% additive noise.
    """
    sched = NoiseSchedule(n_timesteps=1000)
    x_0 = torch.randn(4, 1, 8, 8)
    t = torch.zeros(4, dtype=torch.long)
    x_t, _ = sched.q_sample(x_0, t)
    mean_abs_delta = (x_t - x_0).abs().mean().item()
    assert mean_abs_delta < 0.05, f"mean |x_t - x_0| at t=0 should be tiny, got {mean_abs_delta:.4f}"
    # Cosine similarity should be essentially 1
    cos = (x_t.flatten() * x_0.flatten()).sum() / (x_t.norm() * x_0.norm() + 1e-9)
    assert cos.item() > 0.999, f"x_t and x_0 should be highly correlated at t=0, got {cos.item():.4f}"


def test_noise_schedule_q_sample_at_tmax_is_noisy() -> None:
    """At t=T-1, q_sample is dominated by noise."""
    sched = NoiseSchedule(n_timesteps=1000)
    x_0 = torch.randn(4, 1, 8, 8)
    t = torch.full((4,), 999, dtype=torch.long)
    x_t, noise = sched.q_sample(x_0, t)
    # The signal portion sqrt(alpha_bar_T) ~ 0 so x_t ~= noise
    # noise std ~ 1, x_0 std ~ 1; the correlation between x_t and x_0 should be small
    corr = (x_t.flatten() * x_0.flatten()).mean() / (x_t.std() * x_0.std() + 1e-9)
    assert abs(corr.item()) < 0.2, f"x_t and x_0 should be roughly uncorrelated at t=T, got {corr.item()}"


def test_self_attention_preserves_shape() -> None:
    """Self-attention at the bottleneck maintains (B, C, H, W) shape."""
    attn = SelfAttention2D(32, n_heads=4)
    x = torch.randn(2, 32, 24, 12)
    out = attn(x)
    assert out.shape == x.shape


def test_ddim_sample_returns_correct_shape() -> None:
    """DDIM sampling produces a tensor with the same shape as sl_omega."""
    model = DiffusionRefiner(base_channels=16, ch_mult=(1, 2, 4))
    model.eval()
    sched = NoiseSchedule(n_timesteps=50)
    sl_omega = torch.randn(2, 1, 192, 96)
    z = torch.randn(2, 64)
    refined = ddim_sample(model, sched, sl_omega, z, n_steps=5, init_from_sl=True)
    assert refined.shape == sl_omega.shape
    assert torch.isfinite(refined).all()


def test_ddim_sample_init_from_noise_works() -> None:
    """init_from_sl=False (pure-noise start) also returns the right shape."""
    model = DiffusionRefiner(base_channels=16, ch_mult=(1, 2, 4))
    model.eval()
    sched = NoiseSchedule(n_timesteps=50)
    sl_omega = torch.randn(2, 1, 192, 96)
    z = torch.randn(2, 64)
    refined = ddim_sample(model, sched, sl_omega, z, n_steps=5, init_from_sl=False)
    assert refined.shape == sl_omega.shape
    assert torch.isfinite(refined).all()


def test_refiner_conditions_on_sl_omega() -> None:
    """Changing the SL conditioning image changes the model output."""
    torch.manual_seed(0)
    model = DiffusionRefiner(base_channels=16, ch_mult=(1, 2, 4))
    model.eval()
    x_t = torch.randn(1, 1, 192, 96)
    t = torch.tensor([100], dtype=torch.long)
    z = torch.randn(1, 64)
    sl_a = torch.randn(1, 1, 192, 96)
    sl_b = torch.randn(1, 1, 192, 96)
    with torch.no_grad():
        out_a = model(x_t, t, sl_a, z)
        out_b = model(x_t, t, sl_b, z)
    assert not torch.allclose(out_a, out_b, atol=1e-3), \
        "refiner output should differ when the SL conditioning image changes"


def test_refiner_conditions_on_z() -> None:
    """Changing the JEPA latent z changes the model output."""
    torch.manual_seed(0)
    model = DiffusionRefiner(base_channels=16, ch_mult=(1, 2, 4))
    model.eval()
    x_t = torch.randn(1, 1, 192, 96)
    t = torch.tensor([100], dtype=torch.long)
    sl = torch.randn(1, 1, 192, 96)
    z_a = torch.randn(1, 64)
    z_b = torch.randn(1, 64)
    with torch.no_grad():
        out_a = model(x_t, t, sl, z_a)
        out_b = model(x_t, t, sl, z_b)
    assert not torch.allclose(out_a, out_b, atol=1e-3), \
        "refiner output should differ when z changes"
