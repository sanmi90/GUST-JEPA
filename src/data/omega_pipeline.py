"""Reusable omega-field preprocessing pipeline.

Three-stage transformation applied identically to every encounter
(train + test_a + test_b + test_c, plus any future cases):

    1. Spatial mask:    zero cells inside-solid + 1-cell-adjacent.
                        Removes leading-edge finite-difference artifacts
                        geometrically. Fixed mask, same for all encounters.

    2. Per-encounter clip:  clip |omega| above its own p99.99 (computed
                            AFTER the spatial mask). Catches residual
                            statistical outliers in a data-driven way:
                            sparse spikes (top 0.01%) get clipped; dense
                            physical features (e.g., 5% of pixels at the
                            same magnitude) sit below p99.99 and are kept.

    3. Z-score normalize:   (omega - mu_train) / sigma_train where
                            mu_train and sigma_train are computed over
                            all train encounters AFTER steps 1 and 2.
                            sigma_train is the natural omega scale of
                            this dataset (3.67 for partition v1).

The pipeline is reversible (unnormalize for visualization). New cases
are added via a one-shot script that recomputes per-encounter p99.99
and appends to the manifest; train statistics stay frozen so test-time
preprocessing is deterministic.

The manifest schema (JSON) is::

    {
        "version": "v1",
        "partition": "v1.2",
        "mask_path": "airfoil_adjacent_mask.npy",
        "train_stats": {
            "mean": 0.053,
            "std":  3.67,
            "n_pixels": 302915520,
            "computed_at": "2026-05-20T..."
        },
        "thresholds": {
            "<case_id>": {"<encounter_index>": p99_99, ...},
            ...
        }
    }

Usage in training::

    from src.data.omega_pipeline import OmegaPipeline
    pipeline = OmegaPipeline.from_manifest("outputs/data_pipeline/v1.json")
    omega_norm = pipeline(omega, case_id="G+1.00_D0.50_Y+0.10",
                          encounter_index=0)
    # ... model.forward(omega_norm) ...
    omega_recon = pipeline.unnormalize(model.decode(z))

Build the manifest with::

    python scripts/build_omega_pipeline.py --output outputs/data_pipeline/v1.json
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import torch


ArrayLike = Union[np.ndarray, torch.Tensor]


@dataclass
class OmegaTrainStats:
    """Frozen train-set statistics for normalization."""

    mean: float
    std: float
    n_pixels: int

    def to_dict(self) -> dict:
        return {"mean": float(self.mean), "std": float(self.std),
                "n_pixels": int(self.n_pixels)}


class OmegaPipeline:
    """Three-stage omega-field preprocessing pipeline.

    Stateless transformation parameterized by:

    * a fixed spatial mask ``mask`` of shape ``(H, W)`` (boolean),
    * a dict of per-encounter clip thresholds ``thresholds``,
    * frozen train-set ``mean`` and ``std`` for the final normalization.

    Stage 1 is geometric (mask), Stage 2 is statistical (per-encounter clip),
    Stage 3 is train-distribution normalization. All three commute with
    autograd, so the pipeline can be used inside a forward pass.

    The pipeline is split into ``preprocess_raw`` (Stages 1 + 2) and
    ``normalize`` (Stage 3) so the wrapper can hold the normalized
    inputs but compute reconstruction losses on the masked-and-clipped
    raw scale (which is what the per-pixel MSE comparison against the
    case-mean noise floor uses).
    """

    def __init__(
        self,
        mask: ArrayLike,
        thresholds: dict[str, dict[str, float]],
        train_stats: OmegaTrainStats,
        version: str = "v1",
    ) -> None:
        # mask: (H, W) bool
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        self.mask = mask.bool()
        self.thresholds = thresholds
        self.train_stats = train_stats
        self.version = version

    @classmethod
    def from_manifest(cls, manifest_path: str | Path) -> "OmegaPipeline":
        manifest_path = Path(manifest_path)
        with open(manifest_path) as f:
            m = json.load(f)
        mask_path = manifest_path.parent / m["mask_path"]
        mask = np.load(mask_path)
        ts = m["train_stats"]
        return cls(
            mask=mask,
            thresholds=m["thresholds"],
            train_stats=OmegaTrainStats(
                mean=ts["mean"], std=ts["std"], n_pixels=ts["n_pixels"],
            ),
            version=m.get("version", "v1"),
        )

    def to_dict(self, mask_path: str = "airfoil_adjacent_mask.npy") -> dict:
        """Serialize the pipeline to a JSON-friendly dict.

        The mask itself is stored as a sidecar ``.npy`` (referenced by
        ``mask_path``) since JSON does not handle 2D arrays gracefully.
        Callers should save the mask separately.
        """
        return {
            "version": self.version,
            "mask_path": mask_path,
            "train_stats": self.train_stats.to_dict(),
            "thresholds": self.thresholds,
        }

    def get_threshold(self, case_id: str, encounter_index: int) -> float:
        """Per-encounter p99.99 clip threshold. Returns float('inf') if
        the (case_id, encounter_index) pair is not in the manifest (the
        encounter will pass through Stage 2 unchanged)."""
        case_thresh = self.thresholds.get(case_id)
        if case_thresh is None:
            return float("inf")
        t = case_thresh.get(str(encounter_index))
        if t is None:
            return float("inf")
        return float(t)

    def preprocess_raw(
        self,
        omega: ArrayLike,
        case_id: str,
        encounter_index: int,
    ) -> ArrayLike:
        """Stages 1 + 2: spatial mask + per-encounter clip.

        Returns omega on the raw scale (units of 1 / convective time)
        with artifacts removed. The output is ready for the
        case-mean MSE-ratio + SSIM evaluation pipeline.
        """
        threshold = self.get_threshold(case_id, encounter_index)
        if isinstance(omega, np.ndarray):
            out = omega.copy()
            # Mask: assumes mask is (H, W) and broadcasts onto the last two
            # axes of out (which can be (T, H, W) or (1, T, 1, H, W) etc.)
            mask_np = self.mask.cpu().numpy()
            out[..., mask_np] = 0.0
            if threshold != float("inf"):
                out = np.clip(out, -threshold, threshold)
            return out
        # torch path
        if self.mask.device != omega.device:
            self.mask = self.mask.to(omega.device)
        out = torch.where(self.mask, torch.zeros_like(omega), omega)
        if threshold != float("inf"):
            out = out.clamp(-threshold, threshold)
        return out

    def normalize(self, omega_raw: ArrayLike) -> ArrayLike:
        """Stage 3: z-score normalize by train-set statistics."""
        m = self.train_stats.mean
        s = self.train_stats.std
        if isinstance(omega_raw, np.ndarray):
            return (omega_raw - m) / s
        return (omega_raw - m) / s

    def unnormalize(self, omega_norm: ArrayLike) -> ArrayLike:
        """Inverse of Stage 3: bring a decoder output back to raw scale."""
        m = self.train_stats.mean
        s = self.train_stats.std
        if isinstance(omega_norm, np.ndarray):
            return omega_norm * s + m
        return omega_norm * s + m

    def __call__(
        self,
        omega: ArrayLike,
        case_id: str,
        encounter_index: int,
    ) -> ArrayLike:
        """Full pipeline: Stages 1 + 2 + 3."""
        return self.normalize(self.preprocess_raw(omega, case_id, encounter_index))
