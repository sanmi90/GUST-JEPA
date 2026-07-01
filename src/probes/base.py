"""Frozen-encoder probe harness (Session 31 Track B.2).

The defining property of a probe in this project is that it reads out from a
FROZEN encoder with a provable stop-gradient on the latent: training the probe
head must never update the encoder weights, nor (because the encoder is held in
``eval()``) its BatchNorm running statistics. This module provides the small
shared base that enforces that discipline so the three readout heads (field
decoder, observable readout, pressure) all inherit identical guarantees.

Two mechanisms are layered for belt-and-suspenders safety:
    1. :func:`freeze_encoder` sets the encoder to ``eval()`` and flips
       ``requires_grad`` to ``False`` on every encoder parameter, so autograd
       never allocates a gradient for them.
    2. :meth:`FrozenProbe.encode` runs the encoder under ``torch.no_grad`` and
       explicitly ``.detach()``-es the latent, so no graph edge connects the
       probe loss back to the encoder even if a caller re-enables grads.
"""

from __future__ import annotations

from typing import Iterator

import torch
from torch import Tensor, nn


def freeze_encoder(encoder: nn.Module) -> nn.Module:
    """Put ``encoder`` into frozen-probe mode (``eval`` + no grad).

    WARNING: this mutates ``encoder`` IN PLACE -- it calls ``encoder.eval()``
    and flips ``requires_grad`` to ``False`` on every parameter. Constructing a
    :class:`FrozenProbe` (or any subclass) freezes the caller's encoder object
    by side effect, so a reference held elsewhere is frozen too. Pass a copy if
    you need the original to stay trainable.

    Args:
        encoder: The encoder whose parameters should be frozen.

    Returns:
        The same ``encoder`` instance, for call chaining.
    """
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


def latent_to_vector(z: Tensor) -> Tensor:
    """Collapse a (possibly spatial) latent to a per-frame ``d``-vector.

    A pooled latent ``(B, T, d)`` is returned unchanged. A spatial latent
    ``(B, T, d, h, w)`` is global-average-pooled over the ``(h, w)`` grid to
    ``(B, T, d)`` so the observable readout sees the same vector interface for
    every model.

    Args:
        z: Pooled ``(B, T, d)`` or spatial ``(B, T, d, h, w)`` latent.

    Returns:
        ``(B, T, d)`` per-frame latent vector.
    """
    if z.dim() == 5:
        return z.mean(dim=(-2, -1))
    if z.dim() == 3:
        return z
    raise ValueError(f"expected (B, T, d) or (B, T, d, h, w) latent, got {tuple(z.shape)}")


class FrozenProbe(nn.Module):
    """Base class for a readout head on a frozen encoder.

    The encoder is registered as a submodule (so ``.to(device)`` and
    ``state_dict`` move it with the probe) but is frozen on construction and
    forced back into ``eval()`` on every :meth:`encode` call, which also detaches
    the latent. Subclasses build their trainable head and call :meth:`encode`.

    Attributes:
        encoder: The frozen encoder (``eval`` mode, ``requires_grad=False``).
    """

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        freeze_encoder(self.encoder)

    def encode(self, x: Tensor) -> Tensor:
        """Run the frozen encoder and return a detached latent (stop-gradient).

        Args:
            x: Encoder input ``(B, T, C, H, W)``.

        Returns:
            The detached latent (``(B, T, d)`` pooled or ``(B, T, d, h, w)``
            spatial), with no graph edge back to the encoder.
        """
        self.encoder.eval()
        with torch.no_grad():
            z = self.encoder(x)
        return z.detach()

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Yield the probe-head parameters (everything except the encoder)."""
        return (p for p in self.parameters() if p.requires_grad)
