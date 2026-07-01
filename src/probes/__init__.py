"""Frozen-encoder probe harness (Session 31 Track B.2).

Probes read out from a FROZEN encoder with a provable stop-gradient on the
latent: the encoder is held in ``eval()`` with ``requires_grad=False`` and the
latent is detached before any probe head, so training a probe never updates the
encoder weights or its BatchNorm running statistics. Three readout heads share
this discipline:

    * :class:`DecoderProbe` -- latent -> vorticity field (shared decoder, the
      only trainable module; pooled latents are lifted by a parameter-free
      broadcast so the trainable capacity is identical to the spatial path).
    * :class:`ObservableHeadProbe` -- latent -> five scalar observables.
    * :class:`PressureProbe` -- wall pressure -> latent -> {field, observables}.
"""

from __future__ import annotations

from src.probes.base import FrozenProbe, freeze_encoder, latent_to_vector
from src.probes.decoder_probe import DecoderProbe, PooledToSpatialAdapter
from src.probes.observable_probe import OBSERVABLE_NAMES, ObservableHeadProbe
from src.probes.pressure_probe import PressureProbe

__all__ = [
    "FrozenProbe",
    "freeze_encoder",
    "latent_to_vector",
    "DecoderProbe",
    "PooledToSpatialAdapter",
    "ObservableHeadProbe",
    "OBSERVABLE_NAMES",
    "PressureProbe",
]
