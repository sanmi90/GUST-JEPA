"""Train the conditional diffusion refiner on top of a frozen
(encoder, SL decoder) pair.

Pipeline per training step
--------------------------
1. Sample a batch of sub-trajectories from EpisodeDataset.
2. Encode -> z (frozen E d=64 encoder).
3. Decode -> sl_omega (frozen SL decoder).
4. Sample timestep t and noise; compute x_t = q_sample(dns_omega, t, noise).
5. Predict eps_hat = refiner(x_t, t, sl_omega, z).
6. Loss = MSE(eps_hat, noise).

The refiner sees both ``sl_omega`` (as input channel) and ``z`` (as FiLM
modulation) at every step.

Saves checkpoints at ``--output-dir/diffusion_refiner_iter*.pt`` and a
``metrics.jsonl`` with per-iter loss values.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.data.episode_dataset import EpisodeDataset
from src.data.omega_pipeline import OmegaPipeline
from src.models.diffusion_refiner import DiffusionRefiner, NoiseSchedule, ddim_sample, count_parameters
from src.models.encoder import HybridCNNViTEncoder
from src.models.lap_film_decoder import LapFiLMDecoder
from src.utils.device import require_rtx6000


REPO = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train diffusion refiner on top of frozen SL decoder")
    p.add_argument("--encoder-checkpoint", type=str, required=True)
    p.add_argument("--decoder-checkpoint", type=str, required=True)
    p.add_argument("--omega-pipeline-manifest", type=str, required=True)
    p.add_argument("--partition", type=str, default="v1")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--max-iters", type=int, default=15000)
    p.add_argument("--B", type=int, default=8)
    p.add_argument("--T", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)

    p.add_argument("--refiner-base-channels", type=int, default=32)
    p.add_argument("--refiner-ch-mult", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--refiner-resblocks", type=int, default=2)
    p.add_argument("--refiner-attn-bottleneck", action="store_true", default=True)
    p.add_argument("--refiner-dropout", type=float, default=0.1)
    p.add_argument("--cond-emb-dim", type=int, default=256)

    p.add_argument("--n-diffusion-steps", type=int, default=1000)
    p.add_argument("--beta-start", type=float, default=1e-4)
    p.add_argument("--beta-end", type=float, default=0.02)

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--checkpoint-every", type=int, default=2500)
    p.add_argument("--sample-every", type=int, default=5000)
    p.add_argument("--sample-n-steps", type=int, default=30,
                   help="DDIM steps for periodic eval sampling.")

    return p.parse_args()


def cosine_with_warmup(step: int, max_steps: int, warmup_steps: int, floor: float = 0.05) -> float:
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / float(max(1, max_steps - warmup_steps))
    return floor + (1.0 - floor) * 0.5 * (1.0 + np.cos(np.pi * progress))


def load_frozen_encoder(ckpt_path: Path, latent_dim: int, device: torch.device) -> HybridCNNViTEncoder:
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    sd = ckpt["jepa_state_dict"]
    encoder_sd = {k.removeprefix("encoder."): v for k, v in sd.items() if k.startswith("encoder.")}
    enc = HybridCNNViTEncoder(latent_dim=latent_dim, projection_norm="batchnorm").to(device).eval()
    enc.load_state_dict(encoder_sd)
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


def load_frozen_decoder(ckpt_path: Path, latent_dim: int, device: torch.device) -> LapFiLMDecoder:
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    dec = LapFiLMDecoder(latent_dim=latent_dim).to(device).eval()
    dec.load_state_dict(ckpt["decoder_state_dict"])
    for p in dec.parameters():
        p.requires_grad_(False)
    return dec


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = require_rtx6000(gpu_index=args.gpu)
    print(f"[diff-train] device={device} gpu={torch.cuda.get_device_name(device.index)}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    log_path = output_dir / "train.log"

    def write_log(msg: str) -> None:
        with log_path.open("a") as f:
            f.write(msg + "\n")
        print(msg, flush=True)

    # Persist config
    with (output_dir / "args.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    # Frozen encoder + decoder
    encoder = load_frozen_encoder(Path(args.encoder_checkpoint), args.latent_dim, device)
    decoder = load_frozen_decoder(Path(args.decoder_checkpoint), args.latent_dim, device)
    pipeline = OmegaPipeline.from_manifest(args.omega_pipeline_manifest)
    write_log(f"[diff-train] loaded frozen encoder + decoder; pipeline mean={pipeline.train_stats.mean:.4f} "
              f"std={pipeline.train_stats.std:.4f}")

    # Refiner (trainable)
    refiner = DiffusionRefiner(
        in_channels=1, cond_image_channels=1, z_dim=args.latent_dim,
        base_channels=args.refiner_base_channels,
        ch_mult=tuple(args.refiner_ch_mult),
        n_resblocks=args.refiner_resblocks,
        attn_bottleneck=args.refiner_attn_bottleneck,
        cond_emb_dim=args.cond_emb_dim,
        dropout=args.refiner_dropout,
    ).to(device)
    write_log(f"[diff-train] refiner params = {count_parameters(refiner):,}")

    # Noise schedule
    schedule = NoiseSchedule(args.n_diffusion_steps, args.beta_start, args.beta_end).to(device)

    # Dataset
    ds_train = EpisodeDataset(
        partition=args.partition,
        split="train",
        subtraj_len=args.T,
        omega_pipeline_manifest=args.omega_pipeline_manifest,
        seed=args.seed,
        emit_cl_future=False,
        emit_wake_observable=False,
    )
    loader = DataLoader(
        ds_train, batch_size=args.B, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    write_log(f"[diff-train] dataset partition={args.partition} train samples={len(ds_train)} "
              f"batch_size={args.B} T={args.T}")

    # Optimizer + scheduler
    opt = AdamW(refiner.parameters(), lr=args.lr, betas=(0.9, 0.95),
                weight_decay=args.weight_decay)
    warmup = max(1, int(args.warmup_frac * args.max_iters))
    sched = LambdaLR(opt, lr_lambda=lambda s: cosine_with_warmup(s, args.max_iters, warmup))

    # Config event in metrics.jsonl
    with metrics_path.open("a") as f:
        f.write(json.dumps({"event": "config", **vars(args)}) + "\n")

    encoder.eval()
    decoder.eval()
    refiner.train()

    iteration = 0
    t_start = time.time()
    loader_iter = iter(loader)
    while iteration < args.max_iters:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        omega = batch["omega_z"].to(device, non_blocking=True)  # (B, T, H, W) normalized
        B, T, H, W = omega.shape
        # Encoder expects (B, T, 1, H, W)
        omega_btchw = omega.unsqueeze(2)

        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            z = encoder(omega_btchw.bfloat16())  # (B, T, latent_dim)
            # Flatten time axis: per-frame samples
            z_flat = z.reshape(B * T, args.latent_dim)
            # Decoder expects (N, latent_dim) -> dict with "pred" (N, 1, H, W)
            dec_out = decoder(z_flat)
            sl_omega = dec_out["pred"].float()  # (B*T, 1, H, W)

        # Target: DNS omega per-frame, flat
        dns_omega = omega.reshape(B * T, 1, H, W).float()

        # Sample t and noise
        t = torch.randint(0, args.n_diffusion_steps, (B * T,), device=device, dtype=torch.long)
        x_t, noise = schedule.q_sample(dns_omega, t)

        # Predict
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            eps_hat = refiner(x_t, t, sl_omega, z_flat.float())
            loss = ((eps_hat.float() - noise) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), args.grad_clip)
        opt.step()
        sched.step()

        if iteration % args.log_every == 0:
            wall = time.time() - t_start
            log_entry = {
                "event": "log", "iter": iteration, "step": iteration,
                "loss_mse": float(loss.item()),
                "lr": float(sched.get_last_lr()[0]),
                "wall_s": wall,
            }
            with metrics_path.open("a") as f:
                f.write(json.dumps(log_entry) + "\n")
            write_log(f"[diff-train] iter {iteration}/{args.max_iters} loss={loss.item():.5f} "
                      f"lr={sched.get_last_lr()[0]:.2e} wall={wall:.0f}s")

        if iteration > 0 and iteration % args.checkpoint_every == 0:
            ckpt_path = output_dir / f"diffusion_refiner_iter{iteration:06d}.pt"
            torch.save({"iteration": iteration, "refiner_state_dict": refiner.state_dict(),
                        "args": vars(args)}, ckpt_path)
            write_log(f"[diff-train] checkpoint -> {ckpt_path}")

        if args.sample_every > 0 and iteration > 0 and iteration % args.sample_every == 0:
            refiner.eval()
            with torch.no_grad():
                # Cast inputs to fp32 for the DDIM sampler -- the refiner
                # weights stay in fp32 outside the autocast context, and z /
                # sl_omega may be left in bf16 by the encoder/decoder calls.
                z_one = z_flat[:1].float()
                sl_one = sl_omega[:1].float()
                dns_one = dns_omega[:1].float()
                refined = ddim_sample(refiner, schedule, sl_one, z_one,
                                       n_steps=args.sample_n_steps, init_from_sl=True)
                sl_mse = ((sl_one - dns_one) ** 2).mean().item()
                ref_mse = ((refined - dns_one) ** 2).mean().item()
                write_log(f"[diff-train] iter {iteration} ddim sample: "
                          f"sl_mse_vs_dns={sl_mse:.4f} refined_mse_vs_dns={ref_mse:.4f}")
                sample_entry = {"event": "ddim_sample", "iter": iteration,
                                "sl_mse_vs_dns": sl_mse, "refined_mse_vs_dns": ref_mse}
                with metrics_path.open("a") as f:
                    f.write(json.dumps(sample_entry) + "\n")
            refiner.train()

        iteration += 1

    final_ckpt = output_dir / f"diffusion_refiner_iter{iteration:06d}.pt"
    torch.save({"iteration": iteration, "refiner_state_dict": refiner.state_dict(),
                "args": vars(args)}, final_ckpt)
    write_log(f"[diff-train] final checkpoint -> {final_ckpt}")
    write_log(f"[diff-train] done in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
