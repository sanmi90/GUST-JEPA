"""Field-decoder probe for the frozen encoder (Session 31 Track B.2).

The decoder probe gives every model an IDENTICAL-capacity field readout: the
shared :class:`src.models.decoder.SpatialLatentFieldDecoder` is the ONLY
trainable module, and it decodes the frozen spatial latent ``(B, T, d, h, w)``
back to the vorticity field. Pooled-latent models (the ``jepa_pool`` ablation
path) are lifted to ``(B, T, d, h, w)`` by a PARAMETER-FREE broadcast/tile
(:class:`PooledToSpatialAdapter`) before the same decoder, so the decode floor
is measured with an identical trainable-parameter budget for pooled and spatial
latents alike. The pooled latent therefore must carry all structure through its
``d`` channels and gains nothing the spatial latent does not, which faithfully
represents "a pooled latent has no spatial layout" (the gray-scott point).
"""

from __future__ import annotations

from torch import Tensor, nn

from src.models.decoder import SpatialLatentFieldDecoder
from src.probes.base import FrozenProbe


class PooledToSpatialAdapter(nn.Module):
    """Parameter-free lift of a pooled latent to a spatial map by broadcast/tile.

    The ``d``-vector is tiled to the IDENTICAL value at every ``(h, w)`` cell.
    There are NO learned parameters: a pooled latent has no spatial structure,
    so it carries all information through its ``d`` channels and adds zero
    capacity. This keeps the :class:`DecoderProbe` trainable-parameter count
    identical to the spatial path, so the decode-floor comparison is unbiased.

    Args:
        latent_dim: Latent channel dimension ``d``.
        feature_h: Spatial-latent grid height (default 24).
        feature_w: Spatial-latent grid width (default 12).
    """

    def __init__(self, latent_dim: int, feature_h: int = 24, feature_w: int = 12) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.feature_h = int(feature_h)
        self.feature_w = int(feature_w)

    def forward(self, z: Tensor) -> Tensor:
        """``(B, d)`` or ``(B, T, d)`` -> ``(B, d, h, w)`` or ``(B, T, d, h, w)``."""
        if z.dim() == 3:
            B, T, d = z.shape
            return z[:, :, :, None, None].expand(B, T, d, self.feature_h, self.feature_w)
        if z.dim() == 2:
            n, d = z.shape
            return z[:, :, None, None].expand(n, d, self.feature_h, self.feature_w)
        raise ValueError(f"expected (B, d) or (B, T, d), got {tuple(z.shape)}")


class DecoderProbe(FrozenProbe):
    """Decode the frozen latent to a vorticity field with a shared decoder.

    Args:
        encoder: A frozen ``HybridCNNViTEncoder`` (pooled or spatial mode).
        latent_dim: Latent channel dimension ``d``.
        feature_h: Spatial-latent grid height (default 24).
        feature_w: Spatial-latent grid width (default 12).
        decoder: Optional pre-built shared field decoder. If ``None`` a fresh
            :class:`SpatialLatentFieldDecoder` is constructed.
        adapter: Optional pre-built pooled->spatial adapter (pooled mode only).
    """

    def __init__(
        self,
        encoder: nn.Module,
        latent_dim: int = 32,
        feature_h: int = 24,
        feature_w: int = 12,
        decoder: nn.Module | None = None,
        adapter: nn.Module | None = None,
    ) -> None:
        super().__init__(encoder)
        self.latent_mode = getattr(encoder, "latent_mode", "pooled")
        self.field_decoder = decoder or SpatialLatentFieldDecoder(
            latent_dim=latent_dim, feature_h=feature_h, feature_w=feature_w
        )
        if self.latent_mode == "pooled":
            self.adapter: nn.Module | None = adapter or PooledToSpatialAdapter(
                latent_dim, feature_h, feature_w
            )
        else:
            self.adapter = None

    def to_spatial(self, z: Tensor) -> Tensor:
        """Return a spatial latent, lifting a pooled latent via the adapter."""
        if self.adapter is not None:
            return self.adapter(z)
        return z

    def forward(self, x: Tensor) -> Tensor:
        """Encode ``x`` (frozen, detached) and decode to a field.

        Args:
            x: Encoder input ``(B, T, 1, 192, 96)``.

        Returns:
            ``(B, T, 1, 192, 96)`` reconstructed vorticity field.
        """
        z = self.encode(x)
        return self.field_decoder(self.to_spatial(z))
