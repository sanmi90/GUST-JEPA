"""Observable-readout probe for the frozen encoder (Session 31 Track B.2).

Maps the frozen latent to the five scalar observables
``(C_L, C_D, circ_pos, circ_neg, wake_enstrophy)`` via a small frozen-encoder
readout (the :class:`src.models.observable_head.ObservableHead` MLP). Spatial
latents are global-average-pooled to a per-frame ``d``-vector first, so the
readout sees the same interface for pooled and spatial models.
"""

from __future__ import annotations

from torch import Tensor, nn

from src.models.observable_head import ObservableHead
from src.probes.base import FrozenProbe, latent_to_vector

# Column order of the five observables this probe predicts.
OBSERVABLE_NAMES: tuple[str, ...] = (
    "C_L",
    "C_D",
    "circ_pos",
    "circ_neg",
    "wake_enstrophy",
)


class ObservableHeadProbe(FrozenProbe):
    """Readout the five flow observables from a frozen latent.

    Args:
        encoder: A frozen ``HybridCNNViTEncoder`` (pooled or spatial mode).
        latent_dim: Latent channel dimension ``d``.
        n_observables: Number of scalar observables (default 5).
        hidden_dim: Width of the readout MLP hidden layer (default 64).
    """

    def __init__(
        self,
        encoder: nn.Module,
        latent_dim: int = 32,
        n_observables: int = 5,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__(encoder)
        self.latent_mode = getattr(encoder, "latent_mode", "pooled")
        self.n_observables = int(n_observables)
        self.readout = ObservableHead(
            latent_dim=latent_dim, hidden_dim=hidden_dim, n_deltas=n_observables
        )

    def forward(self, x: Tensor) -> Tensor:
        """Encode ``x`` (frozen, detached) and read out the observables.

        Args:
            x: Encoder input ``(B, T, 1, 192, 96)``.

        Returns:
            ``(B, T, n_observables)`` predicted observables.
        """
        z = self.encode(x)
        return self.readout(latent_to_vector(z))
