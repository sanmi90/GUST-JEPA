"""Canonical-model adapter for the Session 31 spatial latent (Track C.1).

This module is the glue between the spatial encoder/predictor/decoder built in
Track B and the loss kit built in Track A. It owns the four PURE pieces of the
trainer migration, each unit-tested in isolation (``tests/test_canonical_model.py``):

- :func:`flatten_spatial_latent`: ``[B, T, d, h, w] -> [B*T*h*w, d]`` so SIGReg /
  VICReg see the latent distribution over the batch across cases, frames AND
  spatial cells, never over time within a clip (the SESSION 31 anti-collapse
  note; a time-variance term collapses on quasi-static episodes).
- :func:`global_average_pool_latent`: ``[B, T, d, h, w] -> [B, T, d]`` so the
  scalar lift / wake heads (which expect a ``d``-vector per frame) read the
  spatial latent through a parameter-free GAP adapter.
- :func:`assemble_spatial_rollout`: a single multi-step open-loop rollout with
  the :class:`~src.models.resunet_predictor.ResUNetPredictor` (NO teacher
  forcing). From the first ``context_length`` encoded frames it rolls ``horizon``
  steps autoregressively; targets are the online encoder outputs DETACHED
  (gray-scott online-target, NO EMA network, per ``configs/_kit.yaml``).
- :func:`build_outputs`: assembles the named-tensor ``outputs`` dict that
  :func:`src.losses.kit.compute_total_loss` consumes, choosing recon / pred /
  supervision tensors by the model's active terms.

:class:`CanonicalModel` is the ``nn.Module`` that wires a spatial
``HybridCNNViTEncoder`` to the right downstream modules for a resolved kit
config and returns the ``outputs`` dict from one forward pass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from src.data.wake_observables import mode_output_dim
from src.models.decoder import SpatialLatentFieldDecoder
from src.models.encoder import HybridCNNViTEncoder
from src.models.observable_head import ObservableHead, WakeObservableHead
from src.models.resunet_predictor import ResUNetPredictor

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.config.kit_config import ResolvedKitConfig

__all__ = [
    "flatten_spatial_latent",
    "global_average_pool_latent",
    "assemble_spatial_rollout",
    "build_outputs",
    "CanonicalModel",
]


# ---------------------------------------------------------------------------
# Pure latent adapters
# ---------------------------------------------------------------------------
def flatten_spatial_latent(z: Tensor) -> Tensor:
    """Flatten a latent to ``(N, d)`` for the anti-collapse statistic.

    The SIGReg / VICReg statistic is computed over the batch distribution. For a
    spatial latent every ``(batch, frame, cell)`` triple is an independent
    sample of the ``d``-dim latent, so the flatten pools all three leading axes.

    Args:
        z: ``(B, T, d, h, w)`` spatial latent, ``(B, T, d)`` pooled latent, or an
            already-flat ``(N, d)`` tensor.

    Returns:
        ``(B*T*h*w, d)`` (spatial), ``(B*T, d)`` (pooled), or the input
        unchanged (already 2-D). ``d`` is always the trailing dimension.
    """
    if z.dim() == 5:
        b, t, d, h, w = z.shape
        # Move d to the trailing axis, then collapse (B, T, h, w) into N.
        return z.permute(0, 1, 3, 4, 2).reshape(b * t * h * w, d)
    if z.dim() == 3:
        return z.flatten(0, 1)
    if z.dim() == 2:
        return z
    raise ValueError(
        f"expected z of shape (B, T, d, h, w), (B, T, d) or (N, d); got {tuple(z.shape)}"
    )


def global_average_pool_latent(z: Tensor) -> Tensor:
    """Global-average-pool a spatial latent over its ``(h, w)`` grid.

    The lift and wake heads expect a ``d``-vector per frame; the spatial latent
    is reduced to that vector by a parameter-free mean over the spatial cells.

    Args:
        z: ``(B, T, d, h, w)`` spatial latent, or ``(B, T, d)`` already pooled.

    Returns:
        ``(B, T, d)``.
    """
    if z.dim() == 5:
        return z.mean(dim=(3, 4))
    if z.dim() == 3:
        return z
    raise ValueError(f"expected z of shape (B, T, d, h, w) or (B, T, d); got {tuple(z.shape)}")


# ---------------------------------------------------------------------------
# Pure rollout-target assembly
# ---------------------------------------------------------------------------
def assemble_spatial_rollout(
    predictor: nn.Module,
    z: Tensor,
    horizon: int,
) -> tuple[Tensor, Tensor]:
    """Open-loop ``horizon``-step rollout of the spatial latent (NO teacher forcing).

    The teacher-forced one-step term of the v2.1 trainer is dropped; the kit
    consolidates prediction to a single rollout sequence (SESSION 31). From the
    first ``predictor.context_length`` encoded frames the predictor rolls
    ``horizon`` steps autoregressively, feeding its own predictions back in. The
    targets are the online encoder outputs at the matching frames, DETACHED
    (gray-scott online target; the encoder gets gradient only through the rollout
    *context*, never through the targets, which is what stops representation
    collapse without an EMA network).

    Args:
        predictor: A :class:`ResUNetPredictor`-style module mapping
            ``(B, context_length * d, h, w) -> (B, d, h, w)`` and exposing a
            ``context_length`` attribute.
        z: ``(B, T, d, h, w)`` encoder latents, autograd-attached. Requires
            ``T >= context_length + horizon``.
        horizon: Number of rollout steps ``H``.

    Returns:
        ``(pred_seq, target_seq)`` each ``(B, H, d, h, w)``. ``pred_seq`` is
        autograd-attached (the rollout); ``target_seq`` is detached.
    """
    if z.dim() != 5:
        raise ValueError(f"expected z of shape (B, T, d, h, w); got {tuple(z.shape)}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    cl = int(getattr(predictor, "context_length", 2))
    T = z.shape[1]
    if T < cl + horizon:
        raise ValueError(
            f"sub-trajectory T={T} too short for context_length={cl} + horizon={horizon}; "
            f"need T >= {cl + horizon}"
        )

    # Seed window: the first cl encoded frames (grad-attached, so the encoder is
    # trained through the rollout context).
    context: list[Tensor] = [z[:, i] for i in range(cl)]
    preds: list[Tensor] = []
    for _ in range(horizon):
        nxt = predictor(torch.cat(context, dim=1))
        preds.append(nxt)
        context = context[1:] + [nxt]

    pred_seq = torch.stack(preds, dim=1)
    target_seq = torch.stack([z[:, cl + s] for s in range(horizon)], dim=1).detach()
    return pred_seq, target_seq


# ---------------------------------------------------------------------------
# Pure outputs-dict builder
# ---------------------------------------------------------------------------
def build_outputs(
    z_spatial: Tensor,
    batch: dict[str, Tensor],
    *,
    objective: str,
    horizon: int,
    predictor: nn.Module | None = None,
    decoder: nn.Module | None = None,
    lift_head: nn.Module | None = None,
    wake_head: nn.Module | None = None,
) -> dict[str, Tensor]:
    """Assemble the named-tensor ``outputs`` dict for :func:`compute_total_loss`.

    Only the tensors the active terms require are produced. The anti-collapse
    latent ``z`` is always included pre-flattened to ``(N, d)``; the kit ignores
    it when anti-collapse is off, so including it is harmless. ``_z_spatial`` is
    carried through for diagnostics and is never consumed by the loss.

    Args:
        z_spatial: ``(B, T, d, h, w)`` encoder latent (autograd-attached).
        batch: The training batch (must carry ``cl_future`` when ``lift_head`` is
            set and ``wake_target`` when ``wake_head`` is set).
        objective: ``"recon"``, ``"pred"`` or ``"none"`` (``cfg.objective``).
        horizon: Rollout horizon for the ``"pred"`` objective.
        predictor: The rollout predictor (required for ``"pred"``).
        decoder: The field decoder (required for ``"recon"``).
        lift_head: Optional current-frame ``C_L`` head.
        wake_head: Optional wake-observable head.

    Returns:
        A dict with ``z`` (flattened latent) and ``_z_spatial`` always, plus the
        subset of ``recon_field``/``target_field``, ``pred_seq``/``target_seq``,
        ``cl_pred``/``cl_true``, ``wake_pred``/``wake_true`` selected by the
        active terms.
    """
    outputs: dict[str, Tensor] = {
        "z": flatten_spatial_latent(z_spatial),
        "_z_spatial": z_spatial,
    }

    if objective == "recon":
        if decoder is None:
            raise ValueError("objective 'recon' requires a decoder")
        outputs["recon_field"] = decoder(z_spatial)
        outputs["target_field"] = batch["omega"]
    elif objective == "pred":
        if predictor is None:
            raise ValueError("objective 'pred' requires a predictor")
        pred_seq, target_seq = assemble_spatial_rollout(predictor, z_spatial, horizon)
        outputs["pred_seq"] = pred_seq
        outputs["target_seq"] = target_seq
    elif objective != "none":
        raise ValueError(f"unknown objective {objective!r}; expected recon|pred|none")

    if lift_head is not None:
        if "cl_future" not in batch:
            raise KeyError("lift head configured but batch has no 'cl_future' tensor")
        z_vec = global_average_pool_latent(z_spatial)
        outputs["cl_pred"] = lift_head(z_vec)
        outputs["cl_true"] = batch["cl_future"]

    if wake_head is not None:
        if "wake_target" not in batch:
            raise KeyError("wake head configured but batch has no 'wake_target' tensor")
        z_vec = global_average_pool_latent(z_spatial)
        outputs["wake_pred"] = wake_head(z_vec)
        outputs["wake_true"] = batch["wake_target"]

    return outputs


# ---------------------------------------------------------------------------
# The canonical model
# ---------------------------------------------------------------------------
class CanonicalModel(nn.Module):
    """One spatial-latent model whose graph is a switch-set over the kit config.

    Builds a spatial :class:`HybridCNNViTEncoder` and, by the resolved config's
    active terms, the matching downstream modules: a
    :class:`ResUNetPredictor` for the predictive objective, a
    :class:`SpatialLatentFieldDecoder` for the reconstruction objective, an
    :class:`ObservableHead` for the lift supervision and a
    :class:`WakeObservableHead` for the wake supervision. The forward pass
    returns the ``outputs`` dict for :func:`compute_total_loss`.

    Args:
        cfg: A rule-validated :class:`ResolvedKitConfig`.
        latent_dim: Latent channel dimension ``d`` (default 32, CLAUDE.md).
        projection_norm: Encoder latent-boundary norm (default ``batchnorm``;
            SIGReg requires BatchNorm, CLAUDE.md).
        wake_out_dim: Output width of the wake head. Defaults to the
            ``patch_signed_spectrum`` mode width (80).
    """

    def __init__(
        self,
        cfg: "ResolvedKitConfig",
        latent_dim: int = 32,
        projection_norm: str = "batchnorm",
        wake_out_dim: int | None = None,
    ) -> None:
        super().__init__()
        encoder_kind = cfg.model.get("encoder", "cnn_vit")
        latent_kind = cfg.model.get("latent", "spatial")
        if encoder_kind != "cnn_vit":
            raise NotImplementedError(
                f"CanonicalModel (Track C.1) supports encoder 'cnn_vit'; got "
                f"{encoder_kind!r}. The cnn_only / cnn_vit_temporal ablation "
                f"encoders are a later task."
            )
        if latent_kind != "spatial":
            raise NotImplementedError(
                f"CanonicalModel (Track C.1) supports the spatial latent; got " f"{latent_kind!r}."
            )

        self.objective = cfg.objective
        terms = cfg.active_terms()
        self.lift_on = "lift" in terms
        self.wake_on = "wake" in terms
        self.latent_dim = int(latent_dim)
        self.horizon = int(cfg.representation_objective["pred"]["horizon"])

        self.encoder = HybridCNNViTEncoder(
            latent_dim=latent_dim,
            projection_norm=projection_norm,
            latent_mode="spatial",
        )
        feature_h, feature_w = self.encoder.latent_grid

        self.predictor: nn.Module | None = None
        self.decoder: nn.Module | None = None
        if self.objective == "pred":
            self.predictor = ResUNetPredictor(latent_dim=latent_dim, context_length=2)
        elif self.objective == "recon":
            self.decoder = SpatialLatentFieldDecoder(
                latent_dim=latent_dim, feature_h=feature_h, feature_w=feature_w
            )

        self.lift_head: nn.Module | None = None
        if self.lift_on:
            # Current-frame C_L augmentation (Fukami-standard); one delta (0).
            self.lift_head = ObservableHead(latent_dim=latent_dim, n_deltas=1)

        self.wake_head: nn.Module | None = None
        if self.wake_on:
            out_dim = (
                wake_out_dim
                if wake_out_dim is not None
                else mode_output_dim("patch_signed_spectrum")
            )
            self.wake_head = WakeObservableHead(latent_dim=latent_dim, out_dim=out_dim)

    def downstream_parameters(self) -> list[nn.Parameter]:
        """Every trainable parameter that is NOT in the encoder.

        These take the predictor learning rate in the optimiser (predictor,
        decoder, lift head, wake head).
        """
        params: list[nn.Parameter] = []
        for module in (self.predictor, self.decoder, self.lift_head, self.wake_head):
            if module is not None:
                params += list(module.parameters())
        return params

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Encode ``omega`` and assemble the kit ``outputs`` dict.

        Args:
            batch: ``{'omega': (B, T, 1, 192, 96), ...}`` with ``cl_future`` /
                ``wake_target`` present when the lift / wake heads are on.

        Returns:
            The ``outputs`` dict from :func:`build_outputs`.
        """
        if "omega" not in batch:
            raise KeyError(f"batch must contain 'omega'; got {list(batch.keys())}")
        z = self.encoder(batch["omega"])
        return build_outputs(
            z,
            batch,
            objective=self.objective,
            horizon=self.horizon,
            predictor=self.predictor,
            decoder=self.decoder,
            lift_head=self.lift_head,
            wake_head=self.wake_head,
        )
