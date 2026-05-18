"""Tests for ``src/training/sanity_checks.py``.

The five Session 5 sanity checks gate the variant smoke runs. These
unit tests verify the check functions themselves behave correctly on
small synthetic inputs. Checks 4 and 5 are integration-only (they
require the partition v1 cache and are exercised by the script
entrypoint ``python -m src.training.sanity_checks --all``), so they
are not unit-tested here.

Reference: SESSION5_MEANINGFUL_SMOKE_5K.md "Pre-run sanity checks".
"""

from __future__ import annotations

import torch

from src.models.encoder import HybridCNNViTEncoder
from src.models.jepa import JEPA
from src.models.predictor import AutoregressivePredictor
from src.models.sigreg import SIGReg
from src.training.sanity_checks import (
    SanityCheckResult,
    check_1_batchnorm_running_stats,
    check_2_predictor_identity_then_moves,
    check_3_sigreg_gradient_reaches_encoder,
)


def _tiny_jepa(latent_dim: int = 32) -> JEPA:
    """Build a small JEPA for sanity-check unit tests (depth=2, dropout=0)."""
    encoder = HybridCNNViTEncoder(latent_dim=latent_dim)
    predictor = AutoregressivePredictor(
        latent_dim=latent_dim,
        cond_dim=3,
        hidden_dim=384,
        depth=2,
        heads=8,
        dropout=0.0,
        max_seq_len=32,
    )
    sigreg = SIGReg(dim=latent_dim, num_projections=64, num_knots=9)
    return JEPA(encoder, predictor, sigreg, H_roll=2, rollout_weight=0.5)


def test_check_1_pass_on_mixed_random_inputs() -> None:
    """Mixed random inputs leave the projection BN running stats in a
    reasonable range; check_1 reports PASS.
    """
    torch.manual_seed(0)
    encoder = HybridCNNViTEncoder(latent_dim=32)
    batches = [torch.randn(2, 4, 1, 192, 96) for _ in range(5)]
    result = check_1_batchnorm_running_stats(encoder, batches)
    assert isinstance(result, SanityCheckResult)
    assert result.name == "check_1"
    assert result.passed, f"check_1 unexpectedly FAILED: {result.message}"


def test_check_1_fail_on_nan_input() -> None:
    """NaN-poisoned inputs send the BN running stats to NaN; check_1 FAILS."""
    torch.manual_seed(0)
    encoder = HybridCNNViTEncoder(latent_dim=32)
    bad = torch.full((2, 4, 1, 192, 96), float("nan"))
    result = check_1_batchnorm_running_stats(encoder, [bad, bad])
    assert isinstance(result, SanityCheckResult)
    assert not result.passed
    assert "NaN" in result.message or "Inf" in result.message or "not finite" in result.message.lower()


def test_check_2_pass_on_fresh_jepa() -> None:
    """A fresh JEPA: L_pred at init in [0.01, 10], AdaLN gates move after one
    Adam step, and L_pred decreases over a few overfitting steps.
    """
    torch.manual_seed(0)
    jepa = _tiny_jepa()
    batch = {"omega": torch.randn(2, 8, 1, 192, 96), "c": torch.randn(2, 3)}
    result = check_2_predictor_identity_then_moves(jepa, batch, n_overfitting_steps=4)
    assert isinstance(result, SanityCheckResult)
    assert result.name == "check_2"
    assert result.passed, f"check_2 unexpectedly FAILED: {result.message}"


def test_check_2_fail_when_predictor_is_frozen() -> None:
    """If the predictor is frozen, AdaLN gates do not move; check_2 FAILS."""
    torch.manual_seed(0)
    jepa = _tiny_jepa()
    for p in jepa.predictor.parameters():
        p.requires_grad = False
    batch = {"omega": torch.randn(2, 8, 1, 192, 96), "c": torch.randn(2, 3)}
    result = check_2_predictor_identity_then_moves(jepa, batch, n_overfitting_steps=4)
    assert isinstance(result, SanityCheckResult)
    assert not result.passed


def test_check_3_pass_on_real_encoder_output() -> None:
    """SIGReg gradient on real encoder output reaches the projection BN bias."""
    torch.manual_seed(0)
    encoder = HybridCNNViTEncoder(latent_dim=32)
    sigreg = SIGReg(dim=32, num_projections=64, num_knots=9)
    omega = torch.randn(2, 8, 1, 192, 96)
    result = check_3_sigreg_gradient_reaches_encoder(encoder, sigreg, omega)
    assert isinstance(result, SanityCheckResult)
    assert result.name == "check_3"
    assert result.passed, f"check_3 unexpectedly FAILED: {result.message}"


def test_check_3_fail_when_encoder_frozen() -> None:
    """If the encoder is frozen, BN bias has no grad after backward; check_3 FAILS."""
    torch.manual_seed(0)
    encoder = HybridCNNViTEncoder(latent_dim=32)
    for p in encoder.parameters():
        p.requires_grad = False
    sigreg = SIGReg(dim=32, num_projections=64, num_knots=9)
    omega = torch.randn(2, 8, 1, 192, 96)
    result = check_3_sigreg_gradient_reaches_encoder(encoder, sigreg, omega)
    assert isinstance(result, SanityCheckResult)
    assert not result.passed
