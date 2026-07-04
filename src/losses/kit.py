"""The Session 31 loss kit: one menu, every model is a switch-set over it.

This module owns the only place where a loss term is defined for the canonical
v2.2 comparison. Each term is a pure-ish function over named tensors returning a
scalar, and :func:`compute_total_loss` assembles the active terms (and only the
active terms) from a :class:`~src.config.kit_config.ResolvedKitConfig`.

The semantics mirror the v2.1 trainer (``src/models/jepa.py``):

- recon: plain MSE between the decoded field and the pipeline-normalised target
  field (matched autoencoders only).
- pred: multi-step rollout MSE in latent space. The teacher-forced one-step term
  of v2.1 is dropped; the kit consolidates prediction to a single rollout
  sequence loss (SESSION 31, gray-scott ``SquareLossSeq``). The reduction is the
  mean over all ``(batch, horizon, latent)`` entries, matching
  ``src.training.scheduled_sampling.open_loop_rollout_loss``.
- anti_collapse: SIGReg (default) or VICReg (ablation) on the flattened latent
  ``z``, weighted by the pinned ``lambda``. Computed over the batch, never over
  time within a clip (a time-variance term collapses on quasi-static episodes).
- lift_head: smooth-L1 on the current-frame ``C_L`` augmentation (Fukami
  lineage standard); held on across the controlled matrix.
- wake_head: smooth-L1 on the ``patch_signed_spectrum`` wake observable, the one
  supervision axis flipped with/without. The smooth-L1 transition is ``beta``
  (default 0.5, mirroring ``smooth_l1_observable_loss``); the term is scaled by
  its ``weight``.

Reading note on the wake term: the kit schema separates ``beta`` (the smooth-L1
transition point passed into the Huber loss, mirroring the v2.1
``wake_loss_beta``) from ``weight`` (the multiplicative scale on the term). This
follows the existing trainer semantics exactly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.models.observable_head import smooth_l1_observable_loss
from src.models.sigreg import SIGReg
from src.models.vicreg import VICReg

if TYPE_CHECKING:  # avoid a runtime import cycle (kit_config imports nothing here)
    from src.config.kit_config import ResolvedKitConfig

__all__ = [
    "recon_mse_loss",
    "pred_mse_seq_loss",
    "build_anticollapse",
    "anti_collapse_loss",
    "lift_head_loss",
    "wake_head_loss",
    "nearbody_head_loss",
    "compute_total_loss",
]


# ---------------------------------------------------------------------------
# Representation-objective terms (mutually exclusive across models)
# ---------------------------------------------------------------------------
def recon_mse_loss(recon_field: Tensor, target_field: Tensor) -> Tensor:
    """Plain MSE between a decoded field and its pipeline-normalised target.

    Args:
        recon_field: Decoder output, any shape.
        target_field: Ground-truth field in the SAME shape, in
            ``omega_pipeline_norm`` (3-sigma) space.

    Returns:
        Scalar fp32 mean-squared error.
    """
    if recon_field.shape != target_field.shape:
        raise ValueError(
            "recon and target shapes must match; got "
            f"{tuple(recon_field.shape)} vs {tuple(target_field.shape)}"
        )
    return F.mse_loss(recon_field.float(), target_field.float(), reduction="mean")


def pred_mse_seq_loss(pred_seq: Tensor, target_seq: Tensor) -> Tensor:
    """Multi-step rollout MSE in latent space (``mse_seq``).

    Args:
        pred_seq: Rolled-out latents, shape ``(B, H, d)``.
        target_seq: Ground-truth latents at the same horizons, same shape.

    Returns:
        Scalar fp32 mean over all ``(B, H, d)`` entries.
    """
    if pred_seq.shape != target_seq.shape:
        raise ValueError(
            "pred and target shapes must match; got "
            f"{tuple(pred_seq.shape)} vs {tuple(target_seq.shape)}"
        )
    return F.mse_loss(pred_seq.float(), target_seq.float(), reduction="mean")


# ---------------------------------------------------------------------------
# Anti-collapse term
# ---------------------------------------------------------------------------
def build_anticollapse(
    method: str,
    dim: int,
    sigreg: dict,
    vicreg: dict,
) -> nn.Module:
    """Construct the configured anti-collapse module for a latent of width ``dim``.

    SIGReg and VICReg both carry no trainable parameters, so building one per
    call inside the kit is cheap and keeps the term self-contained.

    Args:
        method: ``"sigreg"`` or ``"vicreg"``.
        dim: Latent dimension ``d`` (``z.shape[-1]``).
        sigreg: Mapping with ``projections``, ``knots``, ``knot_range``.
        vicreg: Mapping with ``std`` (-> ``mu``) and ``cov`` (-> ``nu``).

    Returns:
        A SIGReg or VICReg module mapping ``(N, dim) -> scalar``.
    """
    if method == "sigreg":
        return SIGReg(
            dim=dim,
            num_projections=int(sigreg.get("projections", 256)),
            num_knots=int(sigreg.get("knots", 17)),
            knot_range=tuple(sigreg.get("knot_range", (0.2, 4.0))),  # type: ignore[arg-type]
        )
    if method == "vicreg":
        return VICReg(
            d=dim,
            mu=float(vicreg.get("std", 25.0)),
            nu=float(vicreg.get("cov", 1.0)),
        )
    raise ValueError(f"unknown anti_collapse method {method!r}; expected 'sigreg' or 'vicreg'")


def anti_collapse_loss(z: Tensor, module: nn.Module) -> Tensor:
    """Apply an anti-collapse module to the latent, flattening ``(B, T, d)``.

    Mirrors ``src/models/jepa.py`` which calls the regulariser on
    ``z.flatten(0, 1)`` so the statistic is over the batch of all frames.

    Args:
        z: Latent batch, ``(N, d)`` or ``(B, T, d)``.
        module: A SIGReg or VICReg instance.

    Returns:
        Scalar fp32 anti-collapse statistic.
    """
    if z.dim() == 3:
        z = z.flatten(0, 1)
    return module(z)


# ---------------------------------------------------------------------------
# Supervision terms
# ---------------------------------------------------------------------------
def lift_head_loss(cl_pred: Tensor, cl_true: Tensor, beta: float = 1.0) -> Tensor:
    """Smooth-L1 on the current-frame lift augmentation (Fukami-standard).

    Args:
        cl_pred: Predicted ``C_L`` (and optionally ``C_D``), any shape.
        cl_true: Ground truth, same shape.
        beta: Smooth-L1 transition point (default 1.0, standard Huber).

    Returns:
        Scalar fp32 smooth-L1 loss.
    """
    return smooth_l1_observable_loss(cl_pred, cl_true, beta=beta)


def wake_head_loss(wake_pred: Tensor, wake_true: Tensor, beta: float = 0.5) -> Tensor:
    """Smooth-L1 on the wake observable (``patch_signed_spectrum``).

    Args:
        wake_pred: Predicted wake observable, any shape.
        wake_true: Ground truth, same shape.
        beta: Smooth-L1 transition point (default 0.5, SESSION 11 spec).

    Returns:
        Scalar fp32 smooth-L1 loss (unscaled; the term ``weight`` is applied by
        :func:`compute_total_loss`).
    """
    return smooth_l1_observable_loss(wake_pred, wake_true, beta=beta)


def nearbody_head_loss(nb_pred: Tensor, nb_true: Tensor, beta: float = 0.5) -> Tensor:
    """Smooth-L1 on the near-body observable (``nearbody_lift_element``).

    Mirrors :func:`wake_head_loss` exactly (Session 34 Track C: identical loss
    geometry for the N and W heads is the controlled-comparison keystone).

    Args:
        nb_pred: Predicted near-body observable, any shape.
        nb_true: Ground truth, same shape.
        beta: Smooth-L1 transition point (default 0.5, matching the wake head).

    Returns:
        Scalar fp32 smooth-L1 loss (unscaled; the term ``weight`` is applied by
        :func:`compute_total_loss`).
    """
    return smooth_l1_observable_loss(nb_pred, nb_true, beta=beta)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def _require(outputs: dict[str, Tensor], *keys: str) -> None:
    missing = [k for k in keys if k not in outputs]
    if missing:
        raise KeyError(
            f"compute_total_loss is missing required output tensor(s) {missing}; "
            f"got keys {sorted(outputs)}"
        )


def compute_total_loss(
    cfg: "ResolvedKitConfig",
    outputs: dict[str, Tensor],
    anticollapse_module: nn.Module | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Sum the active loss terms for a resolved kit config.

    Only the terms enabled in ``cfg`` are computed and summed. The returned
    components dict carries the WEIGHTED contribution of each active term plus a
    ``"total"`` key; the per-term contributions sum to ``total``.

    Required ``outputs`` keys, by active term:
        recon: ``recon_field``, ``target_field``
        pred: ``pred_seq``, ``target_seq``
        anti_collapse: ``z``
        lift: ``cl_pred``, ``cl_true``
        wake: ``wake_pred``, ``wake_true``

    Args:
        cfg: A :class:`ResolvedKitConfig` (already rule-validated by the loader).
        outputs: Named tensors produced by the model forward pass.
        anticollapse_module: Optional pre-built SIGReg / VICReg module reused
            across calls. SIGReg and VICReg carry no trainable parameters, so a
            one-off build is correct for an eval; but a training loop calls this
            every iteration, where rebuilding the module (and its quadrature
            grid / projection buffers) per call is wasteful. Pass the module
            built once at trainer setup (with ``dim == z.shape[-1]`` and on the
            right device) to skip the rebuild. When ``None`` (default) the module
            is built from ``cfg.anti_collapse`` exactly as before, so callers
            that do not pass it are unaffected. Used only when anti-collapse is
            on; ignored otherwise. The auto-fallback swaps it for a VICReg
            module by passing the new module here.

    Returns:
        ``(total, components)`` where ``total`` is an autograd-attached scalar and
        ``components`` maps each active term name (plus ``"total"``) to a float.
    """
    ro = cfg.representation_objective
    ac = cfg.anti_collapse
    sup = cfg.supervision

    total: Tensor = torch.zeros((), dtype=torch.float32)
    components: dict[str, float] = {}

    if ro["recon"]["on"]:
        _require(outputs, "recon_field", "target_field")
        term = recon_mse_loss(outputs["recon_field"], outputs["target_field"])
        total = total + term
        components["recon"] = term.detach().item()

    if ro["pred"]["on"]:
        _require(outputs, "pred_seq", "target_seq")
        term = pred_mse_seq_loss(outputs["pred_seq"], outputs["target_seq"])
        total = total + term
        components["pred"] = term.detach().item()

    if ac["on"]:
        _require(outputs, "z")
        z = outputs["z"]
        zf = z.flatten(0, 1) if z.dim() == 3 else z
        if anticollapse_module is not None:
            module = anticollapse_module
        else:
            module = build_anticollapse(ac["method"], zf.shape[-1], ac["sigreg"], ac["vicreg"])
        term = float(ac["lambda"]) * anti_collapse_loss(zf, module)
        total = total + term
        components["anti_collapse"] = term.detach().item()

    lift = sup["lift_head"]
    if lift["on"]:
        _require(outputs, "cl_pred", "cl_true")
        beta = float(lift.get("beta", 1.0))
        term = float(lift["weight"]) * lift_head_loss(
            outputs["cl_pred"], outputs["cl_true"], beta=beta
        )
        total = total + term
        components["lift"] = term.detach().item()

    wake = sup["wake_head"]
    if wake["on"]:
        _require(outputs, "wake_pred", "wake_true")
        term = float(wake["weight"]) * wake_head_loss(
            outputs["wake_pred"], outputs["wake_true"], beta=float(wake["beta"])
        )
        total = total + term
        components["wake"] = term.detach().item()

    nearbody = sup.get("nearbody_head", {"on": False})
    if nearbody["on"]:
        _require(outputs, "nearbody_pred", "nearbody_true")
        term = float(nearbody["weight"]) * nearbody_head_loss(
            outputs["nearbody_pred"], outputs["nearbody_true"],
            beta=float(nearbody["beta"]),
        )
        total = total + term
        components["nearbody"] = term.detach().item()

    components["total"] = total.detach().item()
    return total, components
