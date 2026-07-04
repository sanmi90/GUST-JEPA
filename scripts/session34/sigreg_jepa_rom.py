"""SIGReg-JEPA-ROM adapted to vortex-jepa data (Session 34, "last no-lift attempt").

Verbatim port of Carlos's reference skeleton (uploads/a6e2d22b-sigreg_jepa_rom.py)
with ONLY data-shape adaptations: single-channel pipeline-normalized omega_z at
(192, 96), r defaults to 32 (the paper's pooled dimension), and a participation-
ratio diagnostic added so latent health is comparable with the Track C
PR >= 0.3 d = 9.6 floor. No methodological modifications: POD-anchored decoder,
causal AdaLN-Zero predictor with horizon conditioning, quadrature SIGReg,
stop-gradient same-encoder targets, persistence-normalized horizon weights,
staged losses -- all as in the uploaded design.

Design summary (from the original)
----------------------------------
Explicit ROM state a_t in R^r (the only propagated object), POD-anchored
decoder, causal transformer predictor with AdaLN-Zero horizon conditioning,
and SIGReg (isotropic Gaussian matching) as the single latent-geometry
regularizer. No projector heads, no EMA target encoder: targets are
stop-gradient outputs of the same encoder; reconstruction plus SIGReg
prevent collapse.

Shapes: fields x (B, C, H, W); latents a (B, r); context A (B, L, r).

Stages (activate loss terms progressively):
  1  warm start: E regresses standardized POD coefficients; decoder residual
     is zero-initialized, so the full model starts exactly at POD.
  2  SIGReg-AE: L_rec + L_sig (this model is itself an ablation arm).
  3  JEPA core: L_rec + L_pred + L_sig  (this file implements this step).
  4  + rollout curriculum (noise injection, optional pushforward).
  5  + statistical fine-tune for the chaotic case (latent variance and PSD).

Smoke test at the bottom runs on CPU with random tensors at the vortex-jepa
field shape.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Encoder (capacity-matched to the beta-VAE baseline trunk, deterministic head)
# --------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride, 1)
        self.norm = nn.GroupNorm(8, cout)

    def forward(self, x):
        return F.silu(self.norm(self.conv(x)))


class Encoder(nn.Module):
    """x -> a in R^r. Same trunk class as the beta-VAE baseline; the (mu, sigma)
    head is replaced by a single deterministic linear head. SIGReg shapes the
    aggregate distribution of a instead of a per-sample KL."""

    def __init__(self, c_in=1, r=32, width=32, n_down=4, hw=(192, 96)):
        super().__init__()
        chs = [c_in] + [width * 2 ** i for i in range(n_down)]
        self.blocks = nn.Sequential(
            *[ConvBlock(chs[i], chs[i + 1]) for i in range(n_down)]
        )
        h, w = hw[0] // 2 ** n_down, hw[1] // 2 ** n_down
        self.head = nn.Linear(chs[-1] * h * w, r)

    def forward(self, x):
        return self.head(self.blocks(x).flatten(1))


# --------------------------------------------------------------------------
# POD-anchored decoder
# --------------------------------------------------------------------------

class PODAnchoredDecoder(nn.Module):
    """x_hat = (W_pod + dW) a + R_psi(a), with W_pod = Phi_r diag(sqrt(lambda))
    frozen. Since SIGReg keeps a near unit variance, the sqrt(lambda) scaling
    maps unit-variance coordinates back to physical modal energy. dW starts at
    zero and the residual output conv is zero-initialized, so after the Stage-1
    encoder warm start the model reproduces POD exactly."""

    def __init__(self, phi_r, lam_r, out_shape, r=32, width=32, n_up=4):
        super().__init__()
        C, H, W = out_shape
        self.out_shape = out_shape
        w_pod = phi_r * lam_r.sqrt()[None, :]          # (N, r), N = C*H*W
        self.register_buffer("W_pod", w_pod)
        self.dW = nn.Parameter(torch.zeros_like(w_pod))

        h, w = H // 2 ** n_up, W // 2 ** n_up
        c0 = width * 2 ** (n_up - 1)
        self._hw0 = (c0, h, w)
        self.fc = nn.Linear(r, c0 * h * w)
        ups, c = [], c0
        for _ in range(n_up):
            cn = max(c // 2, width)
            ups += [nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(c, cn, 3, 1, 1), nn.GroupNorm(8, cn), nn.SiLU()]
            c = cn
        self.ups = nn.Sequential(*ups)
        self.out = nn.Conv2d(c, C, 3, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, a):
        B = a.shape[0]
        lin = (a @ (self.W_pod + self.dW).T).view(B, *self.out_shape)
        res = self.out(self.ups(self.fc(a).view(B, *self._hw0)))
        return lin + res


# --------------------------------------------------------------------------
# Predictor: causal transformer with AdaLN-Zero horizon conditioning
# --------------------------------------------------------------------------

class AdaLNZeroBlock(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.n1 = nn.LayerNorm(d, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2 = nn.LayerNorm(d, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.SiLU(), nn.Linear(4 * d, d))
        self.mod = nn.Linear(d, 6 * d)
        nn.init.zeros_(self.mod.weight)                # identity block at init
        nn.init.zeros_(self.mod.bias)

    def forward(self, x, cond, attn_mask):
        s1, b1, g1, s2, b2, g2 = self.mod(cond).chunk(6, dim=-1)
        h = self.n1(x) * (1 + s1) + b1
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + g1 * h
        h = self.mlp(self.n2(x) * (1 + s2) + b2)
        return x + g2 * h


class Predictor(nn.Module):
    """A_t = (a_{t-L+1..t}) plus horizon tau -> a_hat_{t+tau}. Residual head:
    a_hat = a_t + delta, zero-initialized, so the predictor starts exactly at
    persistence, which pairs with the persistence-normalized horizon weights."""

    def __init__(self, r=32, L=64, d=128, depth=4, heads=4,
                 taus=(1, 2, 5, 10, 20)):
        super().__init__()
        self.in_proj = nn.Linear(r, d)
        self.pos = nn.Parameter(torch.zeros(1, L, d))
        self.tau_index = {int(t): i for i, t in enumerate(taus)}
        self.tau_emb = nn.Embedding(len(taus), d)
        self.blocks = nn.ModuleList(AdaLNZeroBlock(d, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, r)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.register_buffer(
            "mask", torch.triu(torch.ones(L, L, dtype=torch.bool), 1)
        )

    def forward(self, A, tau: int):
        B, L, _ = A.shape
        idx = torch.full((B,), self.tau_index[int(tau)],
                         dtype=torch.long, device=A.device)
        cond = self.tau_emb(idx)[:, None]              # (B, 1, d)
        x = self.in_proj(A) + self.pos[:, :L]
        for blk in self.blocks:
            x = blk(x, cond, self.mask[:L, :L])
        return A[:, -1] + self.head(self.norm(x[:, -1]))


# --------------------------------------------------------------------------
# SIGReg: sketched isotropic-Gaussian regularization on encoder outputs
# --------------------------------------------------------------------------

def sigreg(a, n_dirs: int = 128, t_grid=None):
    """Characteristic-function distance of random 1D projections of the batch
    embeddings to N(0, 1), Epps-Pulley style, numerical quadrature version."""
    B, r = a.shape
    u = F.normalize(torch.randn(r, n_dirs, device=a.device), dim=0)
    s = a @ u                                          # (B, M)
    if t_grid is None:
        t_grid = torch.linspace(0.1, 4.0, 16, device=a.device)
    st = s[..., None] * t_grid                         # (B, M, T)
    re, im = torch.cos(st).mean(0), torch.sin(st).mean(0)
    target = torch.exp(-0.5 * t_grid ** 2)             # CF of N(0, 1)
    w = target                                         # normal weighting
    return (((re - target) ** 2 + im ** 2) * w).mean()


def sigreg_moments(a):
    """Weaker moment-matching surrogate (zero mean, identity covariance)."""
    mu = a.mean(0)
    ac = a - mu
    C = ac.T @ ac / max(a.shape[0] - 1, 1)
    I = torch.eye(a.shape[1], device=a.device)
    return mu.pow(2).mean() + (C - I).pow(2).mean()


# --------------------------------------------------------------------------
# Persistence-normalized horizon weights
# --------------------------------------------------------------------------

class HorizonWeights:
    """w_tau = clamp(1 / rho_tau, w_max), rho_tau = EMA of the persistence
    error (1/r)||a_t - a_{t+tau}||^2. Downweights horizons that are close to
    decorrelated (rho -> 2 for unit-variance coordinates), i.e. the loss does
    not force prediction of the unpredictable. Weights are renormalized to
    mean 1 across horizons."""

    def __init__(self, taus, w_max=10.0, momentum=0.99, eps=1e-4):
        self.rho = {int(t): None for t in taus}
        self.w_max, self.m, self.eps = w_max, momentum, eps

    def update(self, tau: int, persistence_err: float):
        r = self.rho[int(tau)]
        self.rho[int(tau)] = (persistence_err if r is None
                              else self.m * r + (1 - self.m) * persistence_err)

    def normalized(self):
        ws = {t: (1.0 if r is None
                  else min(1.0 / (r + self.eps), self.w_max))
              for t, r in self.rho.items()}
        mean_w = sum(ws.values()) / len(ws)
        return {t: w / mean_w for t, w in ws.items()}


# --------------------------------------------------------------------------
# Stage 3 training step: L = lam_rec L_rec + lam_pred L_pred + lam_sig L_sig
# --------------------------------------------------------------------------

def stage3_step(models, batch, hweights, opt,
                lam=dict(rec=1.0, pred=1.0, sig=0.1)):
    """batch: {'context': (B, L, C, H, W), 'targets': {tau: (B, C, H, W)}}."""
    E, D, P = models["enc"], models["dec"], models["pred"]
    xc, xt = batch["context"], batch["targets"]
    B, L = xc.shape[:2]

    a_ctx = E(xc.flatten(0, 1)).view(B, L, -1)
    a_last = a_ctx[:, -1]

    # Reconstruction on the last context frame.
    L_rec = F.mse_loss(D(a_last), xc[:, -1])

    # Stop-gradient targets from the SAME encoder (no EMA). Update the
    # persistence EMA first, then form weighted prediction losses.
    a_tgt = {}
    for tau, x_tau in xt.items():
        with torch.no_grad():
            a_tgt[tau] = E(x_tau)
        hweights.update(tau, F.mse_loss(a_last.detach(), a_tgt[tau]).item())
    ws = hweights.normalized()

    L_pred = a_last.new_zeros(())
    for tau in xt:
        L_pred = L_pred + ws[int(tau)] * F.mse_loss(P(a_ctx, tau), a_tgt[tau])

    # SIGReg on encoder outputs only (context frames carry the gradient).
    L_sig = sigreg(a_ctx.reshape(B * L, -1))

    loss = lam["rec"] * L_rec + lam["pred"] * L_pred + lam["sig"] * L_sig
    opt.zero_grad(set_to_none=True)
    loss.backward()
    params = [p for m in models.values() for p in m.parameters()]
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    return {"rec": L_rec.item(), "pred": L_pred.item(), "sig": L_sig.item()}


# --------------------------------------------------------------------------
# Stage 4: rollout loss with noise injection (optionally pushforward)
# --------------------------------------------------------------------------

def rollout_loss(models, a_ctx, x_future, K=10, gamma=0.95,
                 sigma_inj=0.02, mu_dec=0.1, pushforward=False):
    """a_ctx: (B, L, r) encoded context (grad ok). x_future: (B, K, C, H, W)."""
    E, D, P = models["enc"], models["dec"], models["pred"]
    A, loss = a_ctx, a_ctx.new_zeros(())
    for k in range(1, K + 1):
        a_hat = P(A, 1)
        with torch.no_grad():
            a_ref = E(x_future[:, k - 1])
        loss = loss + gamma ** k * (
            F.mse_loss(a_hat, a_ref)
            + mu_dec * F.mse_loss(D(a_hat), x_future[:, k - 1])
        )
        fed = a_hat + sigma_inj * torch.randn_like(a_hat)
        if pushforward:
            fed = fed.detach()
        A = torch.cat([A[:, 1:], fed[:, None]], dim=1)
    return loss


@torch.no_grad()
def free_run(P, A0, n_steps: int):
    """Autoregressive free run for Stage-5 statistics and evaluation."""
    A, out = A0.clone(), []
    for _ in range(n_steps):
        a = P(A, 1)
        out.append(a)
        A = torch.cat([A[:, 1:], a[:, None]], dim=1)
    return torch.stack(out, dim=1)                     # (B, n_steps, r)


# --------------------------------------------------------------------------
# Diagnostics and optimizer
# --------------------------------------------------------------------------

@torch.no_grad()
def effective_rank(a):
    a = a - a.mean(0)
    ev = torch.linalg.eigvalsh(a.T @ a / max(len(a) - 1, 1)).clamp_min(1e-12)
    p = ev / ev.sum()
    return torch.exp(-(p * p.log()).sum()).item()


@torch.no_grad()
def participation_ratio(a):
    """PR = (sum s_i)^2 / sum s_i^2 of the latent covariance eigenvalues --
    the vortex-jepa collapse diagnostic (floor 0.3 d). Added for comparability
    with the Track C cells; not part of any loss."""
    a = a - a.mean(0)
    ev = torch.linalg.eigvalsh(a.T @ a / max(len(a) - 1, 1)).clamp_min(0)
    return float((ev.sum() ** 2 / (ev ** 2).sum().clamp_min(1e-12)).item())


@torch.no_grad()
def contraction_ratio(a_hat, a):
    """tr Cov(a_hat) / tr Cov(a) per horizon; << 1 signals mean collapse."""
    v = lambda z: (z - z.mean(0)).pow(2).sum(0).mean()
    return (v(a_hat) / v(a).clamp_min(1e-12)).item()


def build_optimizer(models, lr_pred=3e-4, lr_enc=3e-5, lr_dec=1e-4, wd=1e-4):
    """Encoder learns ~10x slower than the predictor in Stage 3+ so the
    predictive gradient reshapes the representation without churning it."""
    groups = [
        {"params": models["enc"].parameters(), "lr": lr_enc},
        {"params": models["dec"].parameters(), "lr": lr_dec},
        {"params": models["pred"].parameters(), "lr": lr_pred},
    ]
    return torch.optim.AdamW(groups, weight_decay=wd)


# --------------------------------------------------------------------------
# Smoke test (vortex-jepa field shape, CPU)
# --------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, H, W, r, L = 2, 1, 192, 96, 32, 16
    taus = (1, 2, 5)

    phi = torch.linalg.qr(torch.randn(C * H * W, r)).Q
    lam = torch.linspace(1.0, 0.05, r)

    models = {
        "enc": Encoder(C, r, width=16, n_down=4, hw=(H, W)),
        "dec": PODAnchoredDecoder(phi, lam, (C, H, W), r, width=16, n_up=4),
        "pred": Predictor(r, L, d=64, depth=2, heads=4, taus=taus),
    }
    opt = build_optimizer(models)
    hw_ = HorizonWeights(taus)

    batch = {
        "context": torch.randn(B, L, C, H, W),
        "targets": {t: torch.randn(B, C, H, W) for t in taus},
    }
    logs = stage3_step(models, batch, hw_, opt)
    print("stage3 losses:", logs)

    a_ctx = models["enc"](batch["context"].flatten(0, 1)).view(B, L, r)
    x_fut = torch.randn(B, 5, C, H, W)
    print("rollout loss:", rollout_loss(models, a_ctx, x_fut, K=5).item())
    print("effective rank:", effective_rank(a_ctx.reshape(-1, r).detach()))
    print("participation ratio:", participation_ratio(a_ctx.reshape(-1, r).detach()))
