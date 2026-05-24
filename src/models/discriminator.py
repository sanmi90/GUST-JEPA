"""PatchGAN discriminator for Session 12 Direction B wake refinement.

This is the discriminator half of the GAN refinement pipeline
documented in :mod:`src.models.refiner` (Session 12 Direction B).
The generator is :class:`src.models.refiner.WakeRefiner`; together
they sharpen the wake region of the frozen E1 decoder output. The
discriminator output is a patch decision map (one decision per
overlapping patch in the field) rather than a single global real /
fake scalar, which is the original patchGAN design from
Isola et al. (CVPR 2017).

Conditioning
------------
The discriminator concatenates the wake ROI mask (see
:func:`src.evaluation.decoder_metrics.wake_mask`) as an extra input
channel so the adversarial signal is wake-localized. This is what
stops the refiner from spending capacity hallucinating detail in the
freestream where there is nothing to sharpen.

Spectral normalisation
----------------------
Every conv weight is wrapped in
``torch.nn.utils.parametrizations.spectral_norm`` (the new
parametrization-based API; the older ``torch.nn.utils.spectral_norm``
hook still works but is deprecated). Spectral normalisation enforces
a Lipschitz constraint on the discriminator, which combined with the
hinge loss gives the most stable GAN training in our experience and
matches the SAGAN / BigGAN convention (Miyato et al., arXiv:1802.05957).

Architecture
------------
4-layer Conv2d -> LeakyReLU(0.2) stack with channel progression
``(input_channels=2) -> 32 -> 64 -> 128 -> 1`` (the (32, 64, 128)
width is smaller than the canonical pix2pix (64, 128, 256) so the
parameter budget lands near the 150k target instead of ~660k).
Strides ``(2, 2, 2, 1)`` so the receptive-field walk is
``192x96 -> 96x48 -> 48x24 -> 24x12 -> 24x12`` (the last layer is
stride-1 with kernel-size 3, padding 1 to preserve the 24x12 patch
decision map exactly; canonical patchGAN uses kernel-size 4 stride 1
padding 1 which would give 23x11 instead).

References
----------
- Isola, Zhu, Zhou, Efros. "Image-to-Image Translation with
  Conditional Adversarial Networks." CVPR 2017, arXiv:1611.07004.
  Original patchGAN.
- Miyato, Kataoka, Koyama, Yoshida. "Spectral Normalization for
  Generative Adversarial Networks." ICLR 2018, arXiv:1802.05957.
- Lim, Ye. "Geometric GAN." arXiv:1705.02894. Hinge loss for GANs
  (which is the loss used by the training loop that consumes this
  module).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.utils.parametrizations import spectral_norm


class PatchGANDiscriminator(nn.Module):
    """4-layer spectrally-normalised patchGAN discriminator.

    The discriminator takes the (real or refined) omega field plus the
    wake ROI mask, both at native ``(192, 96)`` resolution, and emits a
    ``(24, 12)`` patch decision map. Hinge loss is computed by the
    training loop:

    - Discriminator: ``relu(1 - D(real, mask)).mean() +
      relu(1 + D(refined.detach(), mask)).mean()``.
    - Generator (adversarial term only):
      ``-D(refined, mask).mean()``.

    Args:
        in_channels: Channel count of the omega field input (1 for our
            single-channel ``omega_z`` task).
        mask_channels: Channel count of the conditioning mask input
            (1 for a single boolean ROI mask).
        channels: Channel widths after each of the first three convs.
            Default ``(32, 64, 128)`` lands at ~150k parameters.
        leaky_slope: Negative slope for ``LeakyReLU``. Default 0.2.

    Forward signature::

        forward(x: Tensor[B, in_channels, H, W],
                wake_mask: Tensor[H, W]) -> Tensor[B, 1, h_out, w_out]

    with ``(h_out, w_out) == (24, 12)`` for the production ``(192, 96)``
    input.
    """

    def __init__(
        self,
        in_channels: int = 1,
        mask_channels: int = 1,
        channels: tuple[int, int, int] = (32, 64, 128),
        leaky_slope: float = 0.2,
    ) -> None:
        super().__init__()
        if len(channels) != 3:
            raise ValueError(
                f"channels must have length 3 (one per non-final layer); " f"got {len(channels)}"
            )

        self.in_channels = in_channels
        self.mask_channels = mask_channels
        self.leaky_slope = leaky_slope

        c1, c2, c3 = channels
        c_in = in_channels + mask_channels

        # Layer 1: 192x96 -> 96x48
        self.conv1 = spectral_norm(nn.Conv2d(c_in, c1, kernel_size=4, stride=2, padding=1))
        # Layer 2: 96x48 -> 48x24
        self.conv2 = spectral_norm(nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1))
        # Layer 3: 48x24 -> 24x12
        self.conv3 = spectral_norm(nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1))
        # Layer 4: 24x12 -> 24x12 (size-preserving stride-1 patch head;
        # kernel 3 padding 1 keeps the 24x12 grid exactly).
        self.conv4 = spectral_norm(nn.Conv2d(c3, 1, kernel_size=3, stride=1, padding=1))

        self.act = nn.LeakyReLU(negative_slope=leaky_slope, inplace=False)

    def _broadcast_mask(self, x: Tensor, wake_mask: Tensor) -> Tensor:
        """Tile a ``(H, W)`` mask up to ``(B, mask_channels, H, W)``."""
        if wake_mask.dim() != 2:
            raise ValueError(f"wake_mask must be 2D (H, W); got shape {tuple(wake_mask.shape)}")
        if wake_mask.shape != x.shape[-2:]:
            raise ValueError(
                f"wake_mask shape {tuple(wake_mask.shape)} does not match "
                f"input spatial shape {tuple(x.shape[-2:])}"
            )
        mask = wake_mask.to(device=x.device, dtype=x.dtype)
        mask = mask.unsqueeze(0).unsqueeze(0).expand(x.shape[0], self.mask_channels, -1, -1)
        return mask

    def forward(self, x: Tensor, wake_mask: Tensor) -> Tensor:
        """Compute the patch decision map for ``(x, wake_mask)``.

        Args:
            x: Omega field, shape ``(B, in_channels, H, W)``.
            wake_mask: Wake ROI mask, shape ``(H, W)``.

        Returns:
            Tensor of shape ``(B, 1, H/8, W/8)`` (i.e. ``(B, 1, 24, 12)``
            for the production ``(192, 96)`` input). Values are NOT
            squashed through a sigmoid; the hinge loss is computed on
            the raw logits.
        """
        if x.dim() != 4:
            raise ValueError(f"x must be (B, C, H, W); got shape {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"x has {x.shape[1]} channels; expected in_channels=" f"{self.in_channels}"
            )

        mask = self._broadcast_mask(x, wake_mask)
        feat = torch.cat([x, mask], dim=1)
        feat = self.act(self.conv1(feat))
        feat = self.act(self.conv2(feat))
        feat = self.act(self.conv3(feat))
        out = self.conv4(feat)
        return out
