"""Fukami and Taira 2023 lift-augmented convolutional autoencoder.

Reference:
    Fukami, K. and Taira, K. "Grasping extreme aerodynamics on a
    low-dimensional manifold." J. Fluid Mech. 1018, A22 (2023);
    arXiv:2305.18394. See supplementary material Table S.1 for the
    network specification this module follows.

The Fukami architecture is a strictly convolutional autoencoder with
two reconstruction paths sharing a single latent:

    omega_z  ->  CNN encoder  ->  z (R^d)  ->  CNN decoder  ->  omega_z_hat
                                   |
                                   +-------->  MLP lift head  ->  C_L_hat

Joint training objective (single optimizer step on all three components):

    L = lambda_recon * MSE(omega_z, omega_z_hat)
      + lambda_lift  * MSE(C_L,    C_L_hat)

Default lambda_recon = lambda_lift = 1.0. Fukami's original paper used
a latent dimension d=3 for their 240x120 input; we match our
matched-capacity production at d=32 for the comparison against the
SIGReg + OBS + BN JEPA. The single-frame output of the Fukami lift
head is generalised here to a multi-horizon CL_future target with
``n_deltas`` shift offsets to match the JEPA's ``ObservableHead``
(D37; we predict C_L at t + delta for delta in {8, 16, 24} frames,
the same as the JEPA observable head).

Spatial dimension adaptation
----------------------------
Fukami's input is (240, 120, 1) and reaches a (12, 6, 4) bottleneck
via three maxpool stages with ratios (2, 2, 5). Our DNS cache stores
the mid-plane vorticity at (192, 96, 1), so the (12, 6, 4) bottleneck
is reached via four maxpool stages with ratios (2, 2, 2, 2). The FC
chain after flattening the (12, 6, 4) feature map matches Fukami's
Table S.1 columns: 256 -> 64 -> 32 -> 16 -> d. The decoder mirrors
the encoder.

Smoke-tested at d=32 on (B=2, 1, 192, 96) random input.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _conv_block(in_ch: int, out_ch: int, n_groups: int = 4) -> nn.Sequential:
    """Conv2D 3x3 + ReLU. Fukami used plain conv + ReLU; we keep that.

    For numerical stability at bf16 we use GroupNorm with a small
    number of groups (the original Fukami model was trained at fp32
    on CPU/single-GPU; we run bf16 on the RTX 6000 Blackwell, and
    GroupNorm makes that path more stable than vanilla Conv-ReLU).
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True),
        nn.GroupNorm(min(n_groups, out_ch), out_ch),
        nn.ReLU(inplace=True),
    )


class FukamiCNNEncoder(nn.Module):
    """Fukami Table S.1 encoder.

    Channel progression matches Fukami exactly: 1 -> 32 -> 16 -> 8 -> 4.
    FC chain after the (12, 6, 4) bottleneck is Fukami's exact
    `288 -> 256 -> 64 -> 32 -> 16 -> latent_dim`. The default
    `latent_dim = 3` reproduces Fukami's published configuration. The
    spatial pooling layout is adapted for our (192, 96) input via four
    2x maxpools instead of Fukami's (2, 2, 5) at (240, 120); same final
    bottleneck shape (12, 6, 4).
    """

    def __init__(self, latent_dim: int = 3) -> None:
        super().__init__()
        # Channel progression follows Fukami: 1 -> 32 -> 16 -> 8 -> 4.
        self.stage1 = nn.Sequential(_conv_block(1, 32), _conv_block(32, 32))
        self.pool1 = nn.MaxPool2d(2)  # (192, 96) -> (96, 48)
        self.stage2 = nn.Sequential(_conv_block(32, 16), _conv_block(16, 16))
        self.pool2 = nn.MaxPool2d(2)  # (96, 48) -> (48, 24)
        self.stage3 = nn.Sequential(_conv_block(16, 8), _conv_block(8, 8))
        self.pool3 = nn.MaxPool2d(2)  # (48, 24) -> (24, 12)
        self.stage4 = nn.Sequential(_conv_block(8, 4, n_groups=2),
                                    _conv_block(4, 4, n_groups=2))
        self.pool4 = nn.MaxPool2d(2)  # (24, 12) -> (12, 6)

        # Fukami's exact FC chain: 288 -> 256 -> 64 -> 32 -> 16 -> latent_dim.
        self.fc = nn.Sequential(
            nn.Linear(288, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 16), nn.ReLU(inplace=True),
            nn.Linear(16, latent_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        squeeze_T = False
        if x.dim() == 5:
            B, T = x.shape[0], x.shape[1]
            x = x.reshape(B * T, *x.shape[-3:])
            squeeze_T = True

        h = self.pool1(self.stage1(x))
        h = self.pool2(self.stage2(h))
        h = self.pool3(self.stage3(h))
        h = self.pool4(self.stage4(h))
        h = h.flatten(1)
        z = self.fc(h)

        if squeeze_T:
            z = z.view(B, T, -1)
        return z


class FukamiCNNDecoder(nn.Module):
    """Mirror-image decoder of FukamiCNNEncoder.

    FC chain reverses Fukami's encoder chain: latent_dim -> 16 -> 32 ->
    64 -> 256 -> 288, then reshape to (12, 6, 4) and four upsample stages
    back to (192, 96, 1). Default latent_dim = 3 to reproduce Fukami's
    published configuration.
    """

    def __init__(self, latent_dim: int = 3) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(inplace=True),
            nn.Linear(16, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 288), nn.ReLU(inplace=True),
        )

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = nn.Sequential(_conv_block(4, 4, n_groups=2),
                                  _conv_block(4, 8))
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = nn.Sequential(_conv_block(8, 8), _conv_block(8, 16))
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = nn.Sequential(_conv_block(16, 16), _conv_block(16, 32))
        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec4 = nn.Sequential(_conv_block(32, 32), _conv_block(32, 32))
        self.to_omega = nn.Conv2d(32, 1, kernel_size=1, bias=True)

    def forward(self, z: Tensor) -> Tensor:
        squeeze_T = False
        if z.dim() == 3:
            B, T = z.shape[0], z.shape[1]
            z = z.reshape(B * T, -1)
            squeeze_T = True

        h = self.fc(z)
        h = h.view(-1, 4, 12, 6)
        h = self.dec1(self.up1(h))
        h = self.dec2(self.up2(h))
        h = self.dec3(self.up3(h))
        h = self.dec4(self.up4(h))
        x_hat = self.to_omega(h)

        if squeeze_T:
            x_hat = x_hat.view(B, T, *x_hat.shape[-3:])
        return x_hat


class FukamiLiftHead(nn.Module):
    """MLP head from latent to predicted C_L (multi-horizon).

    Follows Fukami's Table S.1 lift-decoder column: a 3-hidden-layer
    MLP (32 -> 64 -> 32 -> output). Fukami's original output was a
    single instantaneous CL scalar; we generalise to multi-horizon
    CL_future by setting the output dimension to ``n_deltas``, so the
    same model can be compared to the JEPA's ObservableHead at the
    matched delta in {8, 16, 24} frames offset (D37).
    """

    def __init__(self, latent_dim: int = 3, n_deltas: int = 3) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 32), nn.ReLU(inplace=True),
            nn.Linear(32, n_deltas),
        )

    def forward(self, z: Tensor) -> Tensor:
        squeeze_T = False
        if z.dim() == 3:
            B, T = z.shape[0], z.shape[1]
            z = z.reshape(B * T, -1)
            squeeze_T = True

        cl_hat = self.fc(z)

        if squeeze_T:
            cl_hat = cl_hat.view(B, T, -1)
        return cl_hat


class FukamiAEWrapper(nn.Module):
    """Full lift-augmented autoencoder: encoder + decoder + lift head.

    Forward signature is designed to slot into the JEPA training data
    pipeline: ``batch`` is the dict produced by ``jepa_collate``.

    Input normalization. Fukami's supplementary Figure S.1 shows
    vorticity in roughly [-0.6, +0.6]; the dataset is normalized before
    training. Our raw omega_z lives in roughly [-1000, +1000]. The
    ``omega_scale`` argument (default 1000) divides the input by that
    scalar before encoding and multiplies the output by the same scalar
    after decoding, so the reconstruction loss is computed in the same
    raw-omega frame as the per-case-mean noise floor used by the
    evaluation pipeline.
    """

    def __init__(
        self,
        latent_dim: int = 3,
        n_deltas: int = 3,
        lambda_recon: float = 1.0,
        lambda_lift: float = 1.0,
        omega_scale: float = 1000.0,
        omega_clip: float | None = None,
        omega_clip_pct: float | None = None,
        airfoil_mask: Tensor | None = None,
        omega_pipeline=None,
    ) -> None:
        super().__init__()
        self.encoder = FukamiCNNEncoder(latent_dim)
        self.decoder = FukamiCNNDecoder(latent_dim)
        self.lift_head = FukamiLiftHead(latent_dim, n_deltas)
        self.latent_dim = latent_dim
        self.n_deltas = n_deltas
        self.lambda_recon = lambda_recon
        self.lambda_lift = lambda_lift
        self.omega_scale = float(omega_scale)
        # omega_pipeline: when not None, supersedes the omega_scale / clip /
        # mask parameters. The pipeline implements the canonical three-stage
        # transform (spatial mask -> per-encounter clip -> z-score normalize)
        # documented in src/data/omega_pipeline.py. The batch must carry
        # ``case_ids`` (list[str]) and ``encounter_indices`` (Tensor[int])
        # so the per-encounter clip thresholds can be looked up.
        self.omega_pipeline = omega_pipeline
        if omega_pipeline is not None:
            self.register_buffer(
                "_pipeline_mask",
                omega_pipeline.mask.clone().bool(),
                persistent=False,
            )
        # airfoil_mask: boolean (H, W) tensor; True where omega should be
        # zeroed (inside-solid + 1-cell-adjacent). Removes LE artifact
        # geometrically: 93-100% of |omega| > 500 pixels live in this layer
        # across our 246 encounters. Built by
        # scripts/compute_omega_clip_thresholds.py.
        if airfoil_mask is not None:
            self.register_buffer(
                "airfoil_mask", airfoil_mask.bool(), persistent=False,
            )
        else:
            self.airfoil_mask = None
        # Artifact suppression. Two complementary options:
        #   omega_clip (fixed): clamp |omega| <= omega_clip globally.
        #   omega_clip_pct (adaptive): per-sample, clip |omega| above its
        #     own p_X percentile. p99.99 is the natural cutoff for our DNS
        #     because the physical tail grows smoothly p99 -> p99.9 -> p99.99
        #     by ~3-4x, then jumps 3-30x from p99.99 to max -- the
        #     leading-edge finite-difference artifact lives in that final
        #     jump. Density-aware: if 5% of pixels have |omega| = 100, they
        #     sit below p99.99 and are kept; if 0.01% of pixels at the LE
        #     have |omega| = 1000, they get clipped to the per-sample p99.99
        #     value.
        self.omega_clip = float(omega_clip) if omega_clip is not None else None
        self.omega_clip_pct = float(omega_clip_pct) if omega_clip_pct is not None else None

    def _maybe_clip(self, omega: Tensor) -> Tensor:
        """Apply the configured artifact suppression to a raw-scale omega."""
        # Spatial mask: zero out inside-solid + 1-cell-adjacent. Applied
        # first so subsequent percentile thresholds reflect the cleaned
        # distribution (otherwise the artifact spikes dominate the tail).
        if self.airfoil_mask is not None:
            # omega: (B, T, 1, H, W) or (B, 1, H, W)
            mask = self.airfoil_mask  # (H, W) bool
            # Broadcast to match omega's last two dims
            omega = torch.where(mask, torch.zeros_like(omega), omega)
        if self.omega_clip is not None:
            omega = omega.clamp(-self.omega_clip, self.omega_clip)
        if self.omega_clip_pct is not None:
            # Per-sample (batch element) adaptive clipping at the chosen
            # percentile of |omega|. The "sample" is the leading dimension;
            # for (B, T, 1, H, W) we threshold each B-element independently.
            B = omega.shape[0]
            flat_abs = omega.reshape(B, -1).abs()
            q = torch.quantile(flat_abs, self.omega_clip_pct / 100.0, dim=1)
            view = [B] + [1] * (omega.dim() - 1)
            q = q.view(*view)
            omega = torch.where(omega.abs() > q, torch.sign(omega) * q, omega)
        return omega

    def encode(self, omega: Tensor) -> Tensor:
        """omega: (B, T, 1, H, W) or (B, 1, H, W) in RAW units.

        The wrapper optionally clips ``|omega|`` (LE artifact suppression),
        then normalises by ``omega_scale`` before passing through the CNN
        encoder. Returns ``z`` in matching leading-batch shape.
        """
        return self.encoder(self._maybe_clip(omega) / self.omega_scale)

    def decode(self, z: Tensor) -> Tensor:
        """Decode ``z`` and de-normalise back to the raw omega frame."""
        return self.decoder(z) * self.omega_scale

    def predict_lift(self, z: Tensor) -> Tensor:
        return self.lift_head(z)

    def _preprocess_with_pipeline(self, omega: Tensor, batch: dict) -> Tensor:
        """Apply OmegaPipeline (mask + per-encounter clip) to the batch.

        Returns omega on the raw scale with artifacts removed. Caller
        decides whether to z-score normalize before the encoder.
        """
        pipe = self.omega_pipeline
        case_ids = batch.get("case_ids")
        enc_idx = batch.get("encounter_indices")
        if case_ids is None or enc_idx is None:
            raise KeyError("omega_pipeline requires batch to carry 'case_ids' "
                           "(list[str]) and 'encounter_indices' (Tensor[int])")
        # Stage 1: spatial mask. Buffer is pre-loaded on the right device.
        mask = self._pipeline_mask
        omega = torch.where(mask, torch.zeros_like(omega), omega)
        # Stage 2: per-encounter clip threshold lookup.
        thresholds = torch.tensor(
            [pipe.get_threshold(cid, int(enc_idx[i].item()))
             for i, cid in enumerate(case_ids)],
            dtype=omega.dtype, device=omega.device,
        )
        view = [omega.shape[0]] + [1] * (omega.dim() - 1)
        thresholds = thresholds.view(*view)
        omega = torch.where(omega.abs() > thresholds,
                            torch.sign(omega) * thresholds, omega)
        return omega

    def forward(self, batch: dict) -> dict:
        """Compute the joint loss on a batch from the JEPA pipeline.

        Args:
            batch: ``{"omega": (B, T, 1, H, W), "cl_future": (B, T, n_deltas)}``
                where ``cl_future`` is the multi-horizon CL target (built by
                ``jepa_collate`` when the observable head is enabled in JEPA).
                Accepts ``"omega_z"`` as an alias of ``"omega"``.
                When ``omega_pipeline`` is set, the batch must also include
                ``case_ids`` and ``encounter_indices`` for per-encounter
                clip-threshold lookup.

        Returns:
            Dict with the loss components.
        """
        omega = batch.get("omega", batch.get("omega_z"))
        if omega is None:
            raise KeyError("batch must contain 'omega' (JEPA) or 'omega_z' (cache)")
        # Apply artifact suppression to the target as well so the
        # reconstruction loss does not penalize the model for failing to
        # reproduce the LE artifact spikes.
        if self.omega_pipeline is not None:
            omega = self._preprocess_with_pipeline(omega, batch)
            # Standard AE training: loss computed in NORMALIZED space. The
            # decoder learns to predict the normalized target directly;
            # un-normalization happens only at evaluation / visualization
            # time. This is what Fukami and every standard AE paper do.
            # Computing the loss on the unnormalized (raw) scale would
            # inflate gradients by (3*sigma)^2 ~ 116x at sigma=3.585 and
            # destabilize training at the usual lr=1e-3.
            omega_norm = self.omega_pipeline.normalize(omega)
            z = self.encoder(omega_norm)
            omega_hat_norm = self.decoder(z)
            L_recon = ((omega_norm - omega_hat_norm) ** 2).mean()
        else:
            omega = self._maybe_clip(omega)
            z = self.encoder(omega / self.omega_scale)
            omega_hat = self.decode(z)
            L_recon = ((omega - omega_hat) ** 2).mean()

        if "cl_future" in batch and self.lambda_lift > 0:
            cl_target = batch["cl_future"]
            cl_hat = self.predict_lift(z)
            mask = torch.isfinite(cl_target)
            if mask.any():
                L_lift = ((cl_hat - cl_target) ** 2 * mask).sum() / mask.sum().clamp_min(1)
            else:
                L_lift = torch.zeros((), device=omega.device, dtype=omega.dtype)
        else:
            L_lift = torch.zeros((), device=omega.device, dtype=omega.dtype)

        L_total = self.lambda_recon * L_recon + self.lambda_lift * L_lift
        return {
            "L_total": L_total,
            "L_recon": L_recon.detach(),
            "L_lift": L_lift.detach(),
            "z": z.detach(),
        }
