"""AeroJEPA-style full-encoder no-lift arm on v2p2 (Session 34).

Third no-lift arm, one controlled step beyond the SIGReg-JEPA-ROM arm
(run_rom_nolift.py). Two named differences, both taken from AeroJEPA
(arXiv:2605.05586v1, Giral et al. 2026), everything else identical to the ROM
arm for a controlled comparison:

1. FULL ENCODER: the project's HybridCNNViTEncoder (pooled, d=32, BatchNorm
   boundary; the flagship encoder, ~10M params) replaces the small conv trunk.
2. RECON ON PREDICTED LATENTS (AeroJEPA coupled objective, their Eq 5+8):
   L_total = lam_l * L_lat + lam_r * L_rec + lam_s * L_sig with
   L_rec = ||D(P(A, tau)) - x_tau||^2 through the PREDICTED latent, so the
   reconstruction gradient supervises predictor AND encoder jointly. AeroJEPA:
   "including the reconstruction pathway during AeroJEPA training helps
   maintain physical validity in the predicted latents while preserving their
   semantic alignment." This deliberately tests the design axis that vortex-
   jepa's locked "no recon in the JEPA loss" rule forbids in the kit; it is a
   user-approved side-arm experiment, not a kit change.

Kept from AeroJEPA's recipe shape: no EMA / no stop-gradient teacher (SIGReg
only; targets are stop-grad outputs of the same encoder, as in both designs),
conditioning-free encoder. Kept from the ROM arm (NOT AeroJEPA) for
comparability: the POD-anchored decoder (our small-2D equivalent of their INR
decoder), the quadrature SIGReg at lam_s = 0.1 (AeroJEPA used 0.01 with the
token formulation; noted, not copied, to keep the ROM comparison controlled),
the AdaLN-Zero horizon predictor, horizon weights, stage-1 POD warm start, and
all optimizer settings. AeroJEPA is steady-state geometry->field; the temporal
JEPA structure here is ours.

Run (RTX 6000):
    taskset -c 0-15 python -m scripts.session34.run_aerojepa_nolift --gpu 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.session34.rom_pod_basis import H, W, load_split_fields  # noqa: E402
from scripts.session34.run_rom_nolift import (  # noqa: E402
    CTX_L,
    TAUS,
    WindowSampler,
    save_latent_cache,
)
from scripts.session34.sigreg_jepa_rom import (  # noqa: E402
    HorizonWeights,
    PODAnchoredDecoder,
    Predictor,
    contraction_ratio,
    effective_rank,
    participation_ratio,
    sigreg,
)


class FrameEncoder(torch.nn.Module):
    """HybridCNNViTEncoder (pooled) applied frame-wise: (B, 1, H, W) -> (B, d)."""

    def __init__(self, r: int = 32) -> None:
        super().__init__()
        from src.models.encoder import HybridCNNViTEncoder

        self.enc = HybridCNNViTEncoder(
            latent_dim=r, projection_norm="batchnorm", latent_mode="pooled"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x.unsqueeze(1)).squeeze(1)  # (B,1,1,H,W) -> (B,1,d) -> (B,d)


def aerojepa_step(models, batch, hweights, opt,
                  lam=dict(lat=1.0, rec=1.0, sig=0.1, lift=0.0)):
    """AeroJEPA coupled step: L_lat + recon-through-PREDICTED-latent + SIGReg.

    Optional scalar lift head (Session 34 follow-up, user-directed): when
    ``models`` carries ``"lift"`` and ``lam["lift"] > 0``, adds the kit-style
    current-frame C_L smooth-L1 on every context latent -- the Track C
    load-bearing anchor grafted onto the AeroJEPA coupled objective."""
    E, D, P = models["enc"], models["dec"], models["pred"]
    xc, xt = batch["context"], batch["targets"]
    B, L = xc.shape[:2]

    a_ctx = E(xc.flatten(0, 1)).view(B, L, -1)

    a_tgt = {}
    a_last = a_ctx[:, -1]
    for tau, x_tau in xt.items():
        with torch.no_grad():
            a_tgt[tau] = E(x_tau)
        hweights.update(tau, F.mse_loss(a_last.detach(), a_tgt[tau]).item())
    ws = hweights.normalized()

    L_lat = a_ctx.new_zeros(())
    a_hat = {}
    for tau in xt:
        a_hat[tau] = P(a_ctx, tau)
        L_lat = L_lat + ws[int(tau)] * F.mse_loss(a_hat[tau], a_tgt[tau])

    # Reconstruction THROUGH the predicted latent (AeroJEPA Eq 8): one horizon
    # sampled per step for cost; gradient reaches P and E jointly.
    taus = list(xt.keys())
    tau_r = taus[int(torch.randint(len(taus), (1,)).item())]
    L_rec = F.mse_loss(D(a_hat[tau_r]), xt[tau_r])

    L_sig = sigreg(a_ctx.reshape(B * L, -1))

    loss = lam["lat"] * L_lat + lam["rec"] * L_rec + lam["sig"] * L_sig
    logs = {"lat": L_lat.item(), "rec": L_rec.item(), "sig": L_sig.item()}
    if lam.get("lift", 0.0) > 0 and "lift" in models:
        cl_pred = models["lift"](a_ctx).squeeze(-1)          # (B, L)
        L_lift = F.smooth_l1_loss(cl_pred, batch["cl_context"], beta=1.0)
        loss = loss + lam["lift"] * L_lift
        logs["lift"] = L_lift.item()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    params = [p for m in models.values() for p in m.parameters()]
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    return logs


@torch.no_grad()
def encode_all(enc: torch.nn.Module, fields: np.ndarray, mean_field: np.ndarray,
               device, batch: int = 128) -> np.ndarray:
    out = []
    enc.eval()
    for i in range(0, fields.shape[0], batch):
        x = torch.from_numpy(fields[i : i + batch] - mean_field[None]) \
            .unsqueeze(1).to(device)
        out.append(enc(x).cpu().numpy())
    enc.train()
    return np.concatenate(out, axis=0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="AeroJEPA-style full-encoder no-lift arm")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--warm-iters", type=int, default=2000)
    ap.add_argument("--stage3-iters", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--warm-batch", type=int, default=64)
    ap.add_argument("--diag-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--pod", default="outputs/session34/rom_pod_basis.npz")
    ap.add_argument("--sig-weight", type=float, default=0.1,
                    help="Weight of the quadrature SIGReg term.")
    ap.add_argument("--lift-weight", type=float, default=0.0,
                    help="Weight of the kit-style scalar C_L head on context "
                         "latents (0 = the original no-lift arm).")
    ap.add_argument("--out", default="outputs/runs/session34/aerojepa_nolift_s0")
    ap.add_argument("--cache-dir", default="outputs/session34/trackc_latents")
    args = ap.parse_args(argv)

    from src.utils.device import require_rtx6000

    device = require_rtx6000(gpu_index=args.gpu)
    gpu_name = torch.cuda.get_device_name(device.index)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    name = out_dir.name
    log = open(out_dir / "train.log", "a")

    def emit(msg: str) -> None:
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    pod = np.load(REPO_ROOT / args.pod)
    mean_field = pod["mean_field"].astype(np.float32)
    phi = torch.from_numpy(pod["phi"][:, : args.r]).float()
    lam_pod = torch.from_numpy(pod["lam"][: args.r]).float()
    emit(f"[aero] device={device} ({gpu_name}) r={args.r} full CNN+ViT encoder, "
         f"recon-on-predicted (AeroJEPA coupled)")

    tr = load_split_fields("train")
    tb = load_split_fields("test_b")
    sampler = WindowSampler(tr, mean_field, CTX_L, max(TAUS), args.seed,
                            return_cl=args.lift_weight > 0)

    models = {
        "enc": FrameEncoder(args.r).to(device),
        "dec": PODAnchoredDecoder(phi, lam_pod, (1, H, W), r=args.r, width=32,
                                  n_up=4).to(device),
        "pred": Predictor(r=args.r, L=CTX_L, d=128, depth=4, heads=4,
                          taus=TAUS).to(device),
    }
    if args.lift_weight > 0:
        from src.models.observable_head import ObservableHead

        models["lift"] = ObservableHead(latent_dim=args.r, n_deltas=1).to(device)
        emit(f"[aero] lift head ON (weight {args.lift_weight})")
    n_enc = sum(p.numel() for p in models["enc"].parameters())
    emit(f"[aero] encoder params: {n_enc / 1e6:.1f}M")

    rng = np.random.default_rng(0)
    diag_rows = rng.choice(tb["fields"].shape[0], size=1536, replace=False)
    diag_x = torch.from_numpy(tb["fields"][diag_rows] - mean_field[None]).unsqueeze(1)

    # ---- Stage 1 warm start (identical to the ROM arm) ----------------------
    Xc = tr["fields"].reshape(tr["fields"].shape[0], -1) - mean_field.reshape(-1)[None]
    a_pod_std = ((Xc @ pod["phi"][:, : args.r])
                 / np.sqrt(pod["lam"][: args.r])[None]).astype(np.float32)
    warm_opt = torch.optim.AdamW(models["enc"].parameters(), lr=3e-4,
                                 weight_decay=1e-4)
    t0 = time.time()
    emit(f"[aero] stage 1 warm start: {args.warm_iters} iters")
    M = Xc.shape[0]
    for it in range(1, args.warm_iters + 1):
        idx = np.random.randint(0, M, size=args.warm_batch)
        x = torch.from_numpy(tr["fields"][idx] - mean_field[None]) \
            .unsqueeze(1).to(device)
        y = torch.from_numpy(a_pod_std[idx]).to(device)
        loss = F.mse_loss(models["enc"](x), y)
        warm_opt.zero_grad(set_to_none=True)
        loss.backward()
        warm_opt.step()
        if it % 200 == 0:
            emit(f"[aero] warm {it}/{args.warm_iters} mse={loss.item():.4f} "
                 f"({time.time() - t0:.0f}s)")
    del warm_opt

    # ---- Stage 3: AeroJEPA coupled objective ---------------------------------
    groups = [
        {"params": models["enc"].parameters(), "lr": 3e-5},
        {"params": models["dec"].parameters(), "lr": 1e-4},
        {"params": models["pred"].parameters(), "lr": 3e-4},
    ]
    if "lift" in models:
        groups.append({"params": models["lift"].parameters(), "lr": 3e-4})
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    hw_ = HorizonWeights(TAUS)
    metrics_path = out_dir / "metrics.jsonl"
    emit(f"[aero] stage 3: {args.stage3_iters} iters (B={args.batch}, L={CTX_L})")
    lam = dict(lat=1.0, rec=1.0, sig=args.sig_weight, lift=args.lift_weight)
    for it in range(1, args.stage3_iters + 1):
        batch = sampler.batch(args.batch, device)
        logs = aerojepa_step(models, batch, hw_, opt, lam=lam)
        if it % args.log_every == 0:
            extra = f" lift={logs['lift']:.4f}" if "lift" in logs else ""
            emit(f"[aero] iter {it}/{args.stage3_iters} "
                 f"lat={logs['lat']:.4f} rec={logs['rec']:.4f} "
                 f"sig={logs['sig']:.5f}{extra} ({time.time() - t0:.0f}s)")
        if it % args.diag_every == 0 or it == args.stage3_iters:
            with torch.no_grad():
                a_diag = []
                for i in range(0, diag_x.shape[0], 128):
                    a_diag.append(models["enc"](diag_x[i : i + 128].to(device)))
                a_diag = torch.cat(a_diag, 0)
                pr = participation_ratio(a_diag)
                er = effective_rank(a_diag)
                b2 = sampler.batch(args.batch, device)
                a_ctx = models["enc"](b2["context"].flatten(0, 1)) \
                    .view(args.batch, CTX_L, -1)
                cr = contraction_ratio(models["pred"](a_ctx, 1),
                                       models["enc"](b2["targets"][1]))
            rec = {"step": it, "diag/pr": pr, "diag/effective_rank": er,
                   "diag/contraction_tau1": cr,
                   **{f"loss/{k}": v for k, v in logs.items()}}
            with open(metrics_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            emit(f"[aero] DIAG iter {it}: PR={pr:.1f} (floor {0.3 * args.r:.1f}) "
                 f"effrank={er:.1f} contraction(tau1)={cr:.2f}")

    torch.save({k: m.state_dict() for k, m in models.items()},
               out_dir / f"checkpoint_iter{args.stage3_iters:06d}.pt")

    # ---- Eval (identical protocol to the ROM arm) ----------------------------
    cache_dir = REPO_ROOT / args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    a_tr = encode_all(models["enc"], tr["fields"], mean_field, device)
    a_tb = encode_all(models["enc"], tb["fields"], mean_field, device)
    save_latent_cache(cache_dir / f"latents_{name}_train.npz", a_tr, tr, "train", name)
    save_latent_cache(cache_dir / f"latents_{name}_test_b.npz", a_tb, tb, "test_b", name)

    from sklearn.metrics import r2_score

    from src.evaluation.represent import fit_linear_probe

    probe = fit_linear_probe(a_tr, tr["cl"])
    cl_r2 = float(r2_score(tb["cl"], probe.predict(a_tb)))
    summary = {
        "name": name,
        "gpu_name": gpu_name,
        "seed": args.seed,
        "r": args.r,
        "encoder": "HybridCNNViTEncoder pooled (full)",
        "objective": "AeroJEPA coupled: L_lat + recon(D(P(A,tau)), x_tau) + 0.1*sigreg"
                     + (f" + {args.lift_weight}*lift" if args.lift_weight > 0 else ""),
        "lift_weight": args.lift_weight, "sig_weight": args.sig_weight,
        "warm_iters": args.warm_iters,
        "stage3_iters": args.stage3_iters,
        "final_pr_test_b": participation_ratio(torch.from_numpy(a_tb)),
        "final_effective_rank_test_b": effective_rank(torch.from_numpy(a_tb)),
        "cl_linear_probe_r2_test_b": cl_r2,
        "pr_floor": 0.3 * args.r,
        "wall_s": time.time() - t0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    emit(f"[aero] SUMMARY: PR(test_b)={summary['final_pr_test_b']:.1f} "
         f"effrank={summary['final_effective_rank_test_b']:.1f} "
         f"C_L probe R2={cl_r2:+.3f}")
    emit(f"[aero] done in {time.time() - t0:.0f}s")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
