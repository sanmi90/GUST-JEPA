"""Tests for the Solera-Rico 2024 beta-VAE port (T8, AD3; L5 literature pin).

Convention pinned 2026-06-11 (L5 check + author confirmation; Carlos is the
second author of the 2024 paper): the CANONICAL beta-VAE objective, i.e. KL
summed over latent dimensions and averaged over batch x time (Higgins et al.
2017; the paper's Eq. (4)). The released KTH-FlowAI code averages the KL over
dimensions as well; that is a known typo, not the intended objective. Beta
values quoted under the code's mean convention must be divided by d before
reuse here (published production 0.05 at d = 20 -> 2.5e-3 canonical).

CPU-friendly (no CUDA path exercised).
"""

from __future__ import annotations

import torch
import pytest

from src.baselines.solera_rico import BetaVAEWrapper, kl_divergence


class TestKLConvention:
    def test_kl_matches_canonical_form(self) -> None:
        torch.manual_seed(0)
        mu = torch.randn(8, 16)
        logvar = torch.randn(8, 16).clamp(-2, 2)
        expected = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(dim=-1).mean()
        assert torch.allclose(kl_divergence(mu, logvar), expected)

    def test_kl_sums_over_latent_dims(self) -> None:
        # mu = 1, logvar = 0 gives per-dimension KL = 0.5; the canonical
        # convention returns d/2 (the released-code mean convention would
        # return 0.5 at any d, which is the typo we are NOT reproducing).
        for d in (4, 64):
            mu = torch.ones(3, d)
            logvar = torch.zeros(3, d)
            assert torch.allclose(kl_divergence(mu, logvar), torch.tensor(d / 2.0))

    def test_kl_zero_at_standard_normal(self) -> None:
        mu = torch.zeros(5, 8)
        logvar = torch.zeros(5, 8)
        assert torch.allclose(kl_divergence(mu, logvar), torch.tensor(0.0))

    def test_kl_mean_over_time_frames(self) -> None:
        # (B, T, d) input: sum over d, mean over B and T jointly.
        mu = torch.ones(2, 3, 4)
        logvar = torch.zeros(2, 3, 4)
        assert torch.allclose(kl_divergence(mu, logvar), torch.tensor(2.0))


class TestBetaWarmup:
    def test_linear_warmup_from_beta_init(self) -> None:
        m = BetaVAEWrapper(latent_dim=4, beta=2.5e-3, beta_init=1e-4, beta_warmup_steps=100)
        assert m.current_beta() == pytest.approx(1e-4)
        m._train_forwards += 50
        assert m.current_beta() == pytest.approx(1e-4 + 0.5 * (2.5e-3 - 1e-4))
        m._train_forwards += 50
        assert m.current_beta() == pytest.approx(2.5e-3)
        m._train_forwards += 10  # holds at target after warmup
        assert m.current_beta() == pytest.approx(2.5e-3)

    def test_no_warmup_returns_target(self) -> None:
        m = BetaVAEWrapper(latent_dim=4, beta=2.5e-3, beta_warmup_steps=0)
        assert m.current_beta() == pytest.approx(2.5e-3)


class TestWrapperForward:
    def test_forward_l_kl_uses_canonical_convention(self) -> None:
        torch.manual_seed(0)
        m = BetaVAEWrapper(latent_dim=4, beta=2.5e-3, lambda_lift=0.0)
        m.eval()
        omega = 50.0 * torch.randn(1, 2, 1, 192, 96)  # (B, T, C, H, W)
        out = m({"omega": omega})
        omega_norm = m._maybe_clip(omega) / m.omega_scale
        mu, logvar = m.encoder.forward_dist(omega_norm)
        assert torch.allclose(out["L_kl"], kl_divergence(mu, logvar), atol=1e-6)

    def test_eval_forward_returns_mu_and_does_not_count(self) -> None:
        torch.manual_seed(0)
        m = BetaVAEWrapper(latent_dim=4, beta=2.5e-3, lambda_lift=0.0)
        m.eval()
        omega = 50.0 * torch.randn(1, 2, 1, 192, 96)  # (B, T, C, H, W)
        out = m({"omega": omega})
        omega_norm = m._maybe_clip(omega) / m.omega_scale
        mu, _ = m.encoder.forward_dist(omega_norm)
        assert torch.allclose(out["z"], mu)
        assert int(m._train_forwards.item()) == 0
