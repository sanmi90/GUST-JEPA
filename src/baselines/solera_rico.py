"""Solera-Rico et al. 2024 beta-variational autoencoder baseline (T8, AD3).

Reference:
    Solera-Rico, A., Sanmiguel Vila, C., Gomez-Lopez, M., Wang, Y.,
    Almashjary, A., Dawson, S. T. M. and Vinuesa, R. "beta-Variational
    autoencoders and transformers for reduced-order modelling of fluid
    flows." Nat. Commun. 15, 1361 (2024); arXiv:2304.03571.

The 2024 pipeline is a beta-VAE compression stage plus a transformer
temporal predictor on the latent. Here we port the COMPRESSION stage as the
fourth baseline family at matched latent dimension; the temporal stage is
supplied by the shared B1 predictor-on-top protocol (one matched predictor
per family), which is the fairness convention of the whole comparison.

Architecture: the same CNN body as our Fukami AE baseline
(:class:`src.baselines.fukami_ae.FukamiCNNEncoder` / ``FukamiCNNDecoder``)
with the FC chain widened to output (mu, logvar); this isolates the
VARIATIONAL objective axis against the deterministic AE at identical
architecture and capacity, mirroring how the 2x2 control isolates the
objective axis. The Solera-Rico-faithful element is the beta-VAE objective:

    L = lambda_recon * MSE(omega, omega_hat)
      + beta * KL( q(z|omega) || N(0, I) )
      [+ lambda_lift * MSE(C_L) + lambda_wake * wake]   (cell-dependent heads)

KL is summed over latent dimensions and averaged over batch x time frames
(the Nat. Commun. convention); beta is linearly warmed up over
``beta_warmup_steps`` training forwards (KL annealing), final value pinned
per the L5 literature check before any T8 cell launches.

`BetaVAEEncoder.forward` returns MU, so every downstream consumer (the
final_eval path, latent-encoding scripts, probes) sees the deterministic
embedding; the sampled path is only used inside ``BetaVAEWrapper.forward``
via ``forward_dist``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.baselines.fukami_ae import (
    FukamiAEWrapper,
    FukamiCNNDecoder,
    FukamiCNNEncoder,
)


class BetaVAEEncoder(nn.Module):
    """Fukami CNN body emitting (mu, logvar); plain forward returns mu."""

    def __init__(self, latent_dim: int, activation: str = "relu",
                 use_norm: bool = True) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.body = FukamiCNNEncoder(2 * latent_dim, activation=activation,
                                     use_norm=use_norm)

    def forward_dist(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.body(x)
        mu, logvar = h.chunk(2, dim=-1)
        # clamp keeps exp(logvar) finite under bf16 autocast
        return mu, logvar.clamp(-10.0, 10.0)

    def forward(self, x: Tensor) -> Tensor:
        mu, _ = self.forward_dist(x)
        return mu


class BetaVAEWrapper(FukamiAEWrapper):
    """beta-VAE baseline: FukamiAEWrapper plumbing + reparameterised latent + KL.

    Accepts every FukamiAEWrapper argument plus ``beta`` and
    ``beta_warmup_steps``. ``encoder_kind`` must stay 'cnn' (the baseline is
    defined at the Fukami CNN architecture; the architecture axis lives in
    the 2x2 control, not here).
    """

    def __init__(self, *args, beta: float = 1e-3, beta_warmup_steps: int = 0,
                 **kwargs) -> None:
        if kwargs.get("encoder_kind", "cnn") != "cnn":
            raise ValueError("BetaVAEWrapper supports encoder_kind='cnn' only")
        super().__init__(*args, **kwargs)
        latent_dim = self.latent_dim
        activation = kwargs.get("activation", "relu")
        use_norm = kwargs.get("use_conv_norm", True)
        self.encoder = BetaVAEEncoder(latent_dim, activation=activation,
                                      use_norm=use_norm)
        # decoder / lift head / wake head from the parent are reused as-is;
        # rebuild the decoder only to keep init RNG ordering deterministic
        # under the seeded constructor (parent built one already).
        self.decoder = FukamiCNNDecoder(latent_dim, activation=activation,
                                        use_norm=use_norm)
        self.beta = float(beta)
        self.beta_warmup_steps = int(beta_warmup_steps)
        self.register_buffer("_train_forwards", torch.zeros((), dtype=torch.long),
                             persistent=True)

    def current_beta(self) -> float:
        if self.beta_warmup_steps <= 0:
            return self.beta
        frac = min(1.0, float(self._train_forwards.item()) / self.beta_warmup_steps)
        return self.beta * frac

    def forward(self, batch: dict) -> dict:
        omega = batch.get("omega", batch.get("omega_z"))
        if omega is None:
            raise KeyError("batch must contain 'omega' (JEPA) or 'omega_z' (cache)")
        if self.omega_pipeline is not None:
            omega_clean = self._preprocess_with_pipeline(omega, batch)
            omega_norm = self.omega_pipeline.normalize(omega_clean)
        else:
            omega_norm = self._maybe_clip(omega) / self.omega_scale

        mu, logvar = self.encoder.forward_dist(omega_norm)
        if self.training:
            std = (0.5 * logvar).exp()
            z = mu + std * torch.randn_like(std)
            self._train_forwards += 1
        else:
            z = mu
        omega_hat_norm = self.decoder(z)
        L_recon = self._recon_loss(omega_norm, omega_hat_norm)
        # KL per frame: sum over latent dims, mean over (B, T) frames.
        kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)
        L_kl = kl.mean()
        beta_now = self.current_beta()

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
            and "wake_target" in batch
        ):
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
            + beta_now * L_kl
            + self.lambda_lift * L_lift
            + self.wake_observable_weight * L_wake
        )
        return {
            "L_total": L_total,
            "L_recon": L_recon.detach(),
            "L_kl": L_kl.detach(),
            "beta_now": torch.as_tensor(beta_now),
            "L_lift": L_lift.detach(),
            "L_wake": L_wake.detach(),
            "z": mu.detach(),
        }
