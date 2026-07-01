"""Shape/plumbing tests for the Session 31 Track B.3 ResUNet predictor stub.

The ResUNet rolls the SPATIAL latent forward one step: it consumes the
concatenation of the current and previous latent maps ``(B, 2d, h, w)`` and
predicts the next latent map ``(B, d, h, w)``. This is the gray-scott / eb_jepa
native predictor for the spatial latent. Scope here is the module + shape test;
training and evaluation are Track D.
"""

from __future__ import annotations

import pytest
import torch

from src.models.resunet_predictor import ResUNetPredictor


def test_resunet_shape_contract() -> None:
    """``(B, 2d, h, w)`` -> ``(B, d, h, w)`` on the default 24x12 grid."""
    torch.manual_seed(0)
    d = 32
    net = ResUNetPredictor(latent_dim=d, context_length=2)
    x = torch.randn(2, 2 * d, 24, 12)
    y = net(x)
    assert y.shape == (2, d, 24, 12)


def test_resunet_accepts_separate_context_frames() -> None:
    """Convenience ``forward_context(prev, curr)`` concatenates then rolls."""
    torch.manual_seed(0)
    d = 16
    net = ResUNetPredictor(latent_dim=d, context_length=2)
    prev = torch.randn(3, d, 24, 12)
    curr = torch.randn(3, d, 24, 12)
    y = net.forward_context(prev, curr)
    assert y.shape == (3, d, 24, 12)


def test_resunet_rejects_wrong_channel_count() -> None:
    net = ResUNetPredictor(latent_dim=32, context_length=2)
    with pytest.raises(ValueError, match="channel"):
        net(torch.randn(2, 32, 24, 12))  # only d, expected 2d


def test_resunet_gradient_flow() -> None:
    """Backward reaches every ResUNet parameter."""
    torch.manual_seed(0)
    d = 8
    net = ResUNetPredictor(latent_dim=d, context_length=2)
    x = torch.randn(2, 2 * d, 24, 12)
    net(x).pow(2).mean().backward()
    for n, p in net.named_parameters():
        assert p.grad is not None, f"no grad for {n}"


def test_resunet_handles_odd_grid() -> None:
    """Upsampling matches skip sizes so odd spatial dims do not crash."""
    torch.manual_seed(0)
    d = 8
    net = ResUNetPredictor(latent_dim=d, context_length=2)
    y = net(torch.randn(1, 2 * d, 13, 7))
    assert y.shape == (1, d, 13, 7)
