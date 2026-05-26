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
import torch.nn.functional as F
from torch import Tensor, nn


def _act(activation: str) -> nn.Module:
    """Activation factory. 'relu' (default; our bf16 default) or 'tanh'
    (strict-Fukami: matches the original 2023 JFM paper)."""
    if activation == "tanh":
        return nn.Tanh()
    return nn.ReLU(inplace=True)


def _conv_block(in_ch: int, out_ch: int, n_groups: int = 4,
                activation: str = "relu", use_norm: bool = True) -> nn.Sequential:
    """Conv2D 3x3 (+ optional GroupNorm) + activation.

    Defaults: ReLU + GroupNorm with a small number of groups. Fukami's
    original is plain Conv + tanh in fp32 on CPU/single-GPU; we add
    GroupNorm because the RTX 6000 Blackwell training runs in bf16
    autocast and the vanilla path is less stable in that precision.
    Pass ``activation='tanh'`` and ``use_norm=False`` for the strict
    Fukami match (then run in fp32).
    """
    layers: list[nn.Module] = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True),
    ]
    if use_norm:
        layers.append(nn.GroupNorm(min(n_groups, out_ch), out_ch))
    layers.append(_act(activation))
    return nn.Sequential(*layers)


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

    def __init__(self, latent_dim: int = 3, activation: str = "relu",
                 use_norm: bool = True) -> None:
        super().__init__()
        cb = lambda i, o, g=4: _conv_block(i, o, n_groups=g,
                                           activation=activation,
                                           use_norm=use_norm)
        self.stage1 = nn.Sequential(cb(1, 32), cb(32, 32))
        self.pool1 = nn.MaxPool2d(2)  # (192, 96) -> (96, 48)
        self.stage2 = nn.Sequential(cb(32, 16), cb(16, 16))
        self.pool2 = nn.MaxPool2d(2)  # (96, 48) -> (48, 24)
        self.stage3 = nn.Sequential(cb(16, 8), cb(8, 8))
        self.pool3 = nn.MaxPool2d(2)  # (48, 24) -> (24, 12)
        self.stage4 = nn.Sequential(cb(8, 4, g=2), cb(4, 4, g=2))
        self.pool4 = nn.MaxPool2d(2)  # (24, 12) -> (12, 6)

        # Fukami's exact FC chain: 288 -> 256 -> 64 -> 32 -> 16 -> latent_dim.
        self.fc = nn.Sequential(
            nn.Linear(288, 256), _act(activation),
            nn.Linear(256, 64), _act(activation),
            nn.Linear(64, 32), _act(activation),
            nn.Linear(32, 16), _act(activation),
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

    def __init__(self, latent_dim: int = 3, activation: str = "relu",
                 use_norm: bool = True) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 16), _act(activation),
            nn.Linear(16, 32), _act(activation),
            nn.Linear(32, 64), _act(activation),
            nn.Linear(64, 256), _act(activation),
            nn.Linear(256, 288), _act(activation),
        )

        cb = lambda i, o, g=4: _conv_block(i, o, n_groups=g,
                                           activation=activation,
                                           use_norm=use_norm)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = nn.Sequential(cb(4, 4, g=2), cb(4, 8))
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = nn.Sequential(cb(8, 8), cb(8, 16))
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = nn.Sequential(cb(16, 16), cb(16, 32))
        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec4 = nn.Sequential(cb(32, 32), cb(32, 32))
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

    def __init__(self, latent_dim: int = 3, n_deltas: int = 3,
                 activation: str = "relu") -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 32), _act(activation),
            nn.Linear(32, 64), _act(activation),
            nn.Linear(64, 32), _act(activation),
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
        recon_loss_type: str = "mse",
        charbonnier_epsilon: float = 0.05,
        recon_active_threshold: float = 0.0,
        recon_inactive_weight: float = 0.0,
        activation: str = "relu",
        use_conv_norm: bool = True,
        wake_observable_head: "nn.Module | None" = None,
        wake_observable_weight: float = 0.0,
        wake_loss_kind: str = "smooth_l1",
        wake_loss_beta: float = 0.5,
    ) -> None:
        super().__init__()
        self.encoder = FukamiCNNEncoder(latent_dim, activation=activation,
                                        use_norm=use_conv_norm)
        self.decoder = FukamiCNNDecoder(latent_dim, activation=activation,
                                        use_norm=use_conv_norm)
        self.lift_head = FukamiLiftHead(latent_dim, n_deltas, activation=activation)
        # Session 11 ablation: Fukami AE + wake observable head, applied
        # at the latent z just like in the JEPA pipeline. Lets us test
        # whether the wake loss alone explains the wake reconstruction
        # win (D84) or whether the JEPA encoder architecture also matters.
        self.wake_observable_head = wake_observable_head
        self.wake_observable_weight = float(wake_observable_weight)
        if wake_loss_kind not in ("smooth_l1", "mse"):
            raise ValueError(
                f"wake_loss_kind must be 'smooth_l1' or 'mse'; got {wake_loss_kind!r}"
            )
        self.wake_loss_kind = wake_loss_kind
        self.wake_loss_beta = float(wake_loss_beta)
        self.latent_dim = latent_dim
        self.n_deltas = n_deltas
        self.lambda_recon = lambda_recon
        self.lambda_lift = lambda_lift
        self.omega_scale = float(omega_scale)
        if recon_loss_type not in {"mse", "l1", "charbonnier", "multiscale", "l2norm"}:
            raise ValueError(f"Unknown recon_loss_type {recon_loss_type!r}; "
                             "choose mse / l1 / charbonnier / multiscale / l2norm.")
        self.recon_loss_type = recon_loss_type
        self.charbonnier_epsilon = float(charbonnier_epsilon)
        # Active-pixel mask (training only). When > 0, the per-pixel loss is
        # weighted by an active-pixel mask: |target| > threshold gets weight 1,
        # rest gets weight ``recon_inactive_weight`` (default 0 = hard mask).
        # Inference / metric evaluation does NOT apply this mask; the decoder
        # is judged on the full field. Hard mask (weight 0) tends to let the
        # freestream diverge into noise; weight ~0.05 keeps it constrained.
        self.recon_active_threshold = float(recon_active_threshold)
        self.recon_inactive_weight = float(recon_inactive_weight)
        # multiscale: Charbonnier on omega + lambda_grad * Charbonnier on Sobel-grad
        # Forces wake-band high-frequency content to be encoded (cores have low
        # gradient magnitude relative to pixel magnitude; wake structure has high
        # spatial gradients).
        self.lambda_grad = 1.0
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0],
                                [-2.0, 0.0, 2.0],
                                [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3) / 8.0
        sobel_y = sobel_x.transpose(-1, -2).clone()
        self.register_buffer("_sobel_x", sobel_x, persistent=False)
        self.register_buffer("_sobel_y", sobel_y, persistent=False)
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

    def _recon_loss(self, target: Tensor, pred: Tensor) -> Tensor:
        """Per-pixel reconstruction loss according to ``recon_loss_type``.

        - ``mse``: standard mean squared error. Strong gradient on
          high-magnitude errors but bulk-zero pixels can dominate via
          their sheer count (the failure mode that pushed our d=3 decoder
          toward "predict zero" with our sparse-vortical DNS).

        - ``l1``: mean absolute error. Unit-magnitude gradient per pixel
          regardless of error size; rare high-magnitude pixels carry the
          same per-pixel gradient as bulk-zero pixels, but their sign is
          consistent (vs symmetric noise from bulk-zero) so they dominate
          the optimizer's direction. Robust to sparse signals.

        - ``charbonnier``: ``sqrt((target - pred)^2 + eps^2) - eps``.
          Smooth L1: quadratic below eps (well-conditioned near optimum),
          linear above eps (L1-like robustness for sparse high-magnitude
          features). Single eps hyperparameter, set to roughly the noise
          floor of the normalized data.

        - ``l2norm``: paper-faithful Fukami loss form (un-squared L2
          norm per frame, averaged over batch and time). Eqn 6 of
          arXiv:2305.08024 writes ``||q - q_hat||_2`` literally. Per
          frame: ``sqrt(sum_pixels (target - pred)^2)``. Per batch:
          ``mean over (B, T) of per-frame L2 norm``. This DIFFERS from
          MSE in gradient magnitude and effective beta scaling; expect
          the L-curve elbow to shift versus MSE.
        """
        err = target - pred
        # l2norm path: per-frame L2 norm averaged over batch+time. Independent
        # of the active-pixel weighting (paper does not specify one).
        if self.recon_loss_type == "l2norm":
            sum_axes = tuple(range(2, err.dim()))  # (C, H, W) for (B, T, C, H, W)
            sq_per_frame = (err ** 2).sum(dim=sum_axes)  # (B, T)
            return sq_per_frame.clamp_min(1e-12).sqrt().mean()
        tau = self.recon_active_threshold
        if tau > 0.0:
            # Active-pixel weighted loss: |target| > tau gets weight 1,
            # rest gets weight ``recon_inactive_weight`` (0 = hard mask;
            # 0.05 = soft mask preserving freestream supervision).
            active = (target.abs() > tau).to(err.dtype)
            w_in = self.recon_inactive_weight
            weight = active + (1.0 - active) * w_in
            denom = weight.sum().clamp_min(1.0)
            if self.recon_loss_type == "mse":
                return ((err ** 2) * weight).sum() / denom
            if self.recon_loss_type == "l1":
                return (err.abs() * weight).sum() / denom
            eps = self.charbonnier_epsilon
            char_pp = torch.sqrt(err * err + eps * eps) - eps
            char = (char_pp * weight).sum() / denom
            if self.recon_loss_type == "charbonnier":
                return char
        else:
            if self.recon_loss_type == "mse":
                return (err ** 2).mean()
            if self.recon_loss_type == "l1":
                return err.abs().mean()
            eps = self.charbonnier_epsilon
            char = (torch.sqrt(err * err + eps * eps) - eps).mean()
            if self.recon_loss_type == "charbonnier":
                return char
        # multiscale: Charbonnier(omega) + lambda_grad * Charbonnier(grad omega)
        # Reduce target/pred to (B*T, 1, H, W) for the Sobel convolution.
        if target.dim() == 5:
            B, T, C, H, W = target.shape
            tgt = target.reshape(B * T, C, H, W)
            prd = pred.reshape(B * T, C, H, W)
        else:
            tgt, prd = target, pred
        # Sobel grad in (x, y) for both target and pred; concatenate channels.
        sx, sy = self._sobel_x.to(tgt.dtype), self._sobel_y.to(tgt.dtype)
        tx = F.conv2d(tgt, sx, padding=1)
        ty = F.conv2d(tgt, sy, padding=1)
        px = F.conv2d(prd, sx, padding=1)
        py = F.conv2d(prd, sy, padding=1)
        grad_err = torch.cat([tx - px, ty - py], dim=1)
        grad_char = (torch.sqrt(grad_err * grad_err + eps * eps) - eps).mean()
        return char + self.lambda_grad * grad_char

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
            omega_norm = self.omega_pipeline.normalize(omega)
            z = self.encoder(omega_norm)
            omega_hat_norm = self.decoder(z)
            L_recon = self._recon_loss(omega_norm, omega_hat_norm)
        else:
            omega = self._maybe_clip(omega)
            z = self.encoder(omega / self.omega_scale)
            omega_hat = self.decode(z)
            L_recon = self._recon_loss(omega, omega_hat)

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

        if (
            self.wake_observable_head is not None
            and self.wake_observable_weight > 0.0
        ):
            if "wake_target" not in batch:
                raise KeyError(
                    "wake_observable_head configured but batch has no 'wake_target' tensor"
                )
            wake_pred = self.wake_observable_head(z)
            wake_target = batch["wake_target"]
            if self.wake_loss_kind == "mse":
                L_wake = ((wake_pred.float() - wake_target.float()) ** 2).mean()
            else:
                L_wake = torch.nn.functional.smooth_l1_loss(
                    wake_pred.float(), wake_target.float(), beta=self.wake_loss_beta
                )
        else:
            L_wake = torch.zeros((), device=omega.device, dtype=omega.dtype)

        L_total = (
            self.lambda_recon * L_recon
            + self.lambda_lift * L_lift
            + self.wake_observable_weight * L_wake
        )
        return {
            "L_total": L_total,
            "L_recon": L_recon.detach(),
            "L_lift": L_lift.detach(),
            "L_wake": L_wake.detach(),
            "z": z.detach(),
        }
