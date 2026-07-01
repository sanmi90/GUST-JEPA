"""Unit tests for the Session 31 Track B.2 frozen-probe harness.

The defining property of the harness: probes train on a FROZEN encoder with a
provable stop-gradient on the latent. These CPU tests assert, for each of the
three probe heads (field decoder, observable readout, pressure), that a backward
through the probe loss reaches NO encoder parameter (every encoder
``param.grad`` is ``None``), and that the decoder probe produces a
non-degenerate field (spatial variance > 0) on random input.

The harness keeps the encoder in ``eval()`` and ``requires_grad_(False)`` and
detaches the latent before the probe head, so neither the BatchNorm running
stats nor the encoder weights can be updated by probe training.
"""

from __future__ import annotations

import pytest
import torch

from src.models.encoder import HybridCNNViTEncoder
from src.probes import (
    DecoderProbe,
    FrozenProbe,
    ObservableHeadProbe,
    PressureProbe,
    freeze_encoder,
)


def _encoder(latent_mode: str, latent_dim: int = 16) -> HybridCNNViTEncoder:
    torch.manual_seed(0)
    return HybridCNNViTEncoder(latent_mode=latent_mode, latent_dim=latent_dim)


def _assert_no_encoder_grad(encoder: HybridCNNViTEncoder) -> None:
    for name, p in encoder.named_parameters():
        assert (
            p.grad is None or float(p.grad.abs().sum()) == 0.0
        ), f"gradient leaked into frozen encoder param {name!r}"


def _adversarially_unfreeze(encoder: HybridCNNViTEncoder) -> None:
    """Re-enable grad and train() on the encoder AFTER the probe froze it.

    This is the adversarial setup for the "detach is load-bearing" tests: with
    ``requires_grad=True`` restored, the ONLY remaining barrier is the
    ``no_grad`` + ``.detach()`` in ``FrozenProbe.encode``. A positive control:
    if the probe called ``self.encoder(x)`` directly (no ``encode`` barrier),
    gradient WOULD leak into these now-trainable encoder parameters.
    """
    encoder.train()
    for p in encoder.parameters():
        p.requires_grad_(True)


def test_freeze_encoder_sets_eval_and_requires_grad_false() -> None:
    enc = _encoder("spatial")
    freeze_encoder(enc)
    assert not enc.training
    assert all(not p.requires_grad for p in enc.parameters())


def test_frozen_probe_freezes_on_construction() -> None:
    enc = _encoder("spatial")
    probe = FrozenProbe(enc)
    assert all(not p.requires_grad for p in probe.encoder.parameters())
    # The encoder must not appear among the trainable parameters.
    trainable_ids = {id(p) for p in probe.trainable_parameters()}
    assert all(id(p) not in trainable_ids for p in probe.encoder.parameters())


@pytest.mark.parametrize("latent_mode", ["spatial", "pooled"])
def test_decoder_probe_stopgrad(latent_mode: str) -> None:
    enc = _encoder(latent_mode)
    probe = DecoderProbe(enc, latent_dim=16)
    x = torch.randn(2, 3, 1, 192, 96)
    field = probe(x)
    assert field.shape == (2, 3, 1, 192, 96)
    target = torch.randn_like(field)
    loss = (field - target).pow(2).mean()
    loss.backward()
    _assert_no_encoder_grad(enc)
    # The probe's own decoder head DID receive gradient.
    head_grads = [p.grad for p in probe.trainable_parameters() if p.grad is not None]
    assert head_grads, "probe head received no gradient"


@pytest.mark.parametrize("latent_mode", ["spatial", "pooled"])
def test_decoder_probe_nondegenerate_field(latent_mode: str) -> None:
    enc = _encoder(latent_mode)
    probe = DecoderProbe(enc, latent_dim=16).eval()
    with torch.no_grad():
        field = probe(torch.randn(2, 2, 1, 192, 96))
    assert torch.isfinite(field).all()
    var = field.flatten(-2).var(dim=-1)
    assert (var > 0).all(), f"degenerate (constant) field, var={var}"


@pytest.mark.parametrize("latent_mode", ["spatial", "pooled"])
def test_observable_probe_stopgrad(latent_mode: str) -> None:
    enc = _encoder(latent_mode)
    probe = ObservableHeadProbe(enc, latent_dim=16, n_observables=5)
    x = torch.randn(2, 4, 1, 192, 96)
    obs = probe(x)
    assert obs.shape == (2, 4, 5)
    target = torch.randn_like(obs)
    (obs - target).pow(2).mean().backward()
    _assert_no_encoder_grad(enc)
    # The readout head must have received gradient (guards against a test that
    # passes vacuously with a disconnected head).
    assert any(p.grad is not None for p in probe.readout.parameters())


@pytest.mark.parametrize("latent_mode", ["spatial", "pooled"])
def test_pressure_probe_stopgrad(latent_mode: str) -> None:
    enc = _encoder(latent_mode)
    probe = PressureProbe(enc, latent_dim=16, n_surface=192, n_observables=5)
    x = torch.randn(2, 3, 1, 192, 96)
    p_wall = torch.randn(2, 3, 192)
    out = probe(x, p_wall)
    assert out["field"].shape == (2, 3, 1, 192, 96)
    assert out["observables"].shape == (2, 3, 5)
    assert out["pred_latent"].shape == out["target_latent"].shape
    # A combined readout + latent-matching loss.
    loss = (
        out["field"].mean()
        + out["observables"].pow(2).mean()
        + (out["pred_latent"] - out["target_latent"]).pow(2).mean()
    )
    loss.backward()
    _assert_no_encoder_grad(enc)
    # The pressure->latent map received gradient.
    assert any(p.grad is not None for p in probe.p_to_latent.parameters())


def test_pressure_probe_frozen_readout_blocks_decoder_grad() -> None:
    """``freeze_readout=True`` trains only the pressure->latent map."""
    enc = _encoder("spatial")
    probe = PressureProbe(enc, latent_dim=16, freeze_readout=True)
    x = torch.randn(1, 2, 1, 192, 96)
    p_wall = torch.randn(1, 2, 192)
    out = probe(x, p_wall)
    out["field"].mean().backward()
    _assert_no_encoder_grad(enc)
    for p in probe.field_decoder.parameters():
        assert p.grad is None or float(p.grad.abs().sum()) == 0.0


# ---------------------------------------------------------------------------
# "detach is load-bearing": prove the no_grad/detach barrier (not just
# requires_grad=False) stops the gradient, by adversarially re-enabling
# encoder grad + train() before backprop.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("latent_mode", ["spatial", "pooled"])
def test_decoder_probe_detach_is_load_bearing(latent_mode: str) -> None:
    enc = _encoder(latent_mode)
    probe = DecoderProbe(enc, latent_dim=16)
    _adversarially_unfreeze(enc)
    field = probe(torch.randn(2, 3, 1, 192, 96))
    field.pow(2).mean().backward()
    _assert_no_encoder_grad(enc)


@pytest.mark.parametrize("latent_mode", ["spatial", "pooled"])
def test_observable_probe_detach_is_load_bearing(latent_mode: str) -> None:
    enc = _encoder(latent_mode)
    probe = ObservableHeadProbe(enc, latent_dim=16, n_observables=5)
    _adversarially_unfreeze(enc)
    obs = probe(torch.randn(2, 4, 1, 192, 96))
    obs.pow(2).mean().backward()
    _assert_no_encoder_grad(enc)


@pytest.mark.parametrize("latent_mode", ["spatial", "pooled"])
def test_pressure_probe_detach_is_load_bearing(latent_mode: str) -> None:
    enc = _encoder(latent_mode)
    probe = PressureProbe(enc, latent_dim=16, n_surface=192, n_observables=5)
    _adversarially_unfreeze(enc)
    out = probe(torch.randn(2, 3, 1, 192, 96), torch.randn(2, 3, 192))
    loss = (
        out["field"].mean()
        + out["observables"].pow(2).mean()
        + (out["pred_latent"] - out["target_latent"]).pow(2).mean()
    )
    loss.backward()
    _assert_no_encoder_grad(enc)


# ---------------------------------------------------------------------------
# Decode-floor capacity must be GENUINELY identical for pooled vs spatial.
# ---------------------------------------------------------------------------


def test_decoder_probe_trainable_capacity_identical_pooled_vs_spatial() -> None:
    """The pooled and spatial decoder probes have identical trainable capacity.

    The pooled->spatial lift is parameter-free (broadcast/tile), so the only
    trainable parameters in either probe are the SHARED field decoder. This is
    the central decode-floor comparison: any difference here would bias it.
    """
    probe_spatial = DecoderProbe(_encoder("spatial"), latent_dim=16)
    probe_pooled = DecoderProbe(_encoder("pooled"), latent_dim=16)
    n_spatial = sum(p.numel() for p in probe_spatial.trainable_parameters())
    n_pooled = sum(p.numel() for p in probe_pooled.trainable_parameters())
    assert n_pooled == n_spatial, f"capacity mismatch: pooled {n_pooled} vs spatial {n_spatial}"
