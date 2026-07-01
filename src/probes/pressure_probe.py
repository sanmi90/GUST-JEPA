"""Wall-pressure probe for the frozen encoder (Session 31 Track B.2).

Inverts the sensing chain: a small trainable net maps the wall-pressure sensor
vector ``p_wall`` to the (frozen) encoder latent space, and the shared field
decoder / observable readout then read out ``{field, observables}`` from that
PREDICTED latent. The frozen encoder supplies the target latent so the
pressure->latent map can be supervised; backward through the probe never
reaches the encoder weights (it is frozen and the target latent is detached).

The harness is intentionally small: the point is the stop-gradient discipline
plus a uniform readout interface, not a tuned pressure estimator.
"""

from __future__ import annotations

from torch import Tensor, nn

from src.models.decoder import SpatialLatentFieldDecoder
from src.models.observable_head import ObservableHead
from src.probes.base import FrozenProbe, freeze_encoder, latent_to_vector
from src.probes.decoder_probe import PooledToSpatialAdapter


class PressureProbe(FrozenProbe):
    """Map wall pressure to the frozen latent, then read out field/observables.

    Args:
        encoder: A frozen ``HybridCNNViTEncoder`` (pooled or spatial mode).
        latent_dim: Latent channel dimension ``d``.
        n_surface: Number of wall-pressure sensors (default 192).
        feature_h: Spatial-latent grid height (default 24).
        feature_w: Spatial-latent grid width (default 12).
        n_observables: Number of scalar observables to read out (default 5).
        hidden_dim: Width of the pressure->latent MLP hidden layer (default 128).
        freeze_readout: If ``True``, freeze the field decoder and observable
            readout so only the pressure->latent map trains.
    """

    def __init__(
        self,
        encoder: nn.Module,
        latent_dim: int = 32,
        n_surface: int = 192,
        feature_h: int = 24,
        feature_w: int = 12,
        n_observables: int = 5,
        hidden_dim: int = 128,
        freeze_readout: bool = False,
    ) -> None:
        super().__init__(encoder)
        self.latent_mode = getattr(encoder, "latent_mode", "pooled")
        self.latent_dim = int(latent_dim)
        self.feature_h = int(feature_h)
        self.feature_w = int(feature_w)

        if self.latent_mode == "spatial":
            self._latent_out = latent_dim * feature_h * feature_w
        else:
            self._latent_out = latent_dim
        self.p_to_latent = nn.Sequential(
            nn.Linear(n_surface, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self._latent_out),
        )

        self.field_decoder = SpatialLatentFieldDecoder(
            latent_dim=latent_dim, feature_h=feature_h, feature_w=feature_w
        )
        self.observable_head = ObservableHead(latent_dim=latent_dim, n_deltas=n_observables)
        if self.latent_mode == "pooled":
            self.adapter: nn.Module | None = PooledToSpatialAdapter(
                latent_dim, feature_h, feature_w
            )
        else:
            self.adapter = None

        if freeze_readout:
            freeze_encoder(self.field_decoder)
            freeze_encoder(self.observable_head)
            if self.adapter is not None:
                freeze_encoder(self.adapter)

    def predict_latent(self, p_wall: Tensor) -> Tensor:
        """Map ``p_wall`` ``(B, T, n_surface)`` to a latent matching the encoder.

        Returns ``(B, T, d)`` in pooled mode, ``(B, T, d, h, w)`` in spatial mode.
        """
        if p_wall.dim() != 3:
            raise ValueError(f"expected p_wall (B, T, n_surface), got {tuple(p_wall.shape)}")
        b, t = p_wall.shape[0], p_wall.shape[1]
        z = self.p_to_latent(p_wall)  # (B, T, _latent_out)
        if self.latent_mode == "spatial":
            return z.view(b, t, self.latent_dim, self.feature_h, self.feature_w)
        return z

    def _to_spatial(self, z: Tensor) -> Tensor:
        if self.adapter is not None:
            return self.adapter(z)
        return z

    def forward(self, x: Tensor, p_wall: Tensor) -> dict[str, Tensor]:
        """Predict the latent from pressure and read out field + observables.

        Args:
            x: Encoder input ``(B, T, 1, 192, 96)`` (supplies the target latent).
            p_wall: Wall pressure ``(B, T, n_surface)``.

        Returns:
            Dict with ``pred_latent``, ``target_latent`` (detached, frozen),
            ``field`` ``(B, T, 1, 192, 96)`` and ``observables``
            ``(B, T, n_observables)``.
        """
        target_latent = self.encode(x)
        pred_latent = self.predict_latent(p_wall)
        field = self.field_decoder(self._to_spatial(pred_latent))
        observables = self.observable_head(latent_to_vector(pred_latent))
        return {
            "pred_latent": pred_latent,
            "target_latent": target_latent,
            "field": field,
            "observables": observables,
        }
