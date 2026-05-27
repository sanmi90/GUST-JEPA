"""Session 18 B1 Part (c): train a transformer predictor on precomputed
baseline latents under the JEPA fairness protocol.

Inputs: precomputed latents from ``encode_baseline_latents.py`` at
``outputs/session18/exp_b1/latents_{baseline}_d{d}/{split}.npz``.

Predictor: ``src.models.predictor.AutoregressivePredictor`` with
``hidden_dim=384, depth=6, heads=16, mlp_ratio=4, dropout=0.1,
max_seq_len=32`` and AdaLN-Zero conditioning on ``(G, D, Y)``. Same
recipe as the JEPA's predictor (CLAUDE.md "Locked decisions").

Loss: ``L = L_teacher_forced + 0.5 * L_open_loop_rollout`` with
``H_roll=8`` and ``rollout_start_strategy='uniform_random'``. Same as the
JEPA's training loop.

Optimizer: AdamW, betas=(0.9, 0.95), weight_decay=0.05, lr=5e-4 peak,
warmup 5%, cosine to 5%. 20000 iters. bf16 mixed precision.

Sampling: impact-aware (70% impact-aware, 30% uniform) with
sub-trajectory length T=32. The impact-aware sampler picks a start
frame within [max(0, t_impact - 16), min(120 - 32, t_impact - 16)]; the
uniform sampler picks a start frame uniformly in [0, 120 - 32].

Usage:
    python scripts/session18/train_baseline_predictor.py \\
        --latents-dir outputs/session18/exp_b1/latents_fukami_d64 \\
        --tag fukami_d64

The script is hardware-locked to RTX 6000 via ``require_rtx6000``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.models.predictor import AutoregressivePredictor  # noqa: E402
from src.training.scheduled_sampling import (  # noqa: E402
    open_loop_rollout_loss,
    teacher_forced_prediction_loss,
)
from src.utils.device import require_rtx6000  # noqa: E402


# B1 fairness defaults (do not tune per baseline).
PROTOCOL_DEFAULTS = dict(
    hidden_dim=384,
    depth=6,
    heads=16,
    mlp_ratio=4.0,
    dropout=0.1,
    max_seq_len=32,
    cond_dim=3,
    lr=5e-4,
    weight_decay=0.05,
    warmup_frac=0.05,
    grad_clip=1.0,
    max_iters=20_000,
    B=16,
    T=32,
    H_roll=8,
    impact_aware_fraction=0.7,
)


class LatentSubTrajectoryDataset(Dataset):
    """Sample sub-trajectories of length T from precomputed per-frame
    latents using the JEPA's impact-aware sampler.

    Latents are z-score normalised per-dim using train-data statistics
    so the predictor sees ~N(0,1) inputs (matches the JEPA's BatchNorm-
    normalised encoder output). Stats are exposed as ``self.mean`` and
    ``self.std`` so callers can un-normalise rollouts."""

    def __init__(
        self,
        z_full: np.ndarray,
        G: np.ndarray,
        D: np.ndarray,
        Y: np.ndarray,
        impact_frame: np.ndarray,
        T: int,
        impact_aware_fraction: float,
        seed: int = 0,
        n_samples_per_epoch: int = 4096,
    ) -> None:
        if z_full.ndim != 3:
            raise ValueError(f"z_full must be (n_enc, T_total, d); got {z_full.shape}")
        flat = z_full.reshape(-1, z_full.shape[-1])
        self.mean = flat.mean(axis=0).astype(np.float32)
        self.std = flat.std(axis=0).clip(min=1e-6).astype(np.float32)
        z_norm = ((z_full - self.mean) / self.std).astype(np.float32)
        self.z = torch.from_numpy(z_norm).float()
        self.G = torch.from_numpy(G).float()
        self.D = torch.from_numpy(D).float()
        self.Y = torch.from_numpy(Y).float()
        self.impact = impact_frame.astype(np.int64)
        self.n_enc, self.T_total, self.d = self.z.shape
        if T > self.T_total:
            raise ValueError(f"T={T} > T_total={self.T_total}")
        self.T = T
        self.fraction = float(impact_aware_fraction)
        self.rng = np.random.default_rng(seed)
        self.n_samples = int(n_samples_per_epoch)

    def __len__(self) -> int:
        return self.n_samples

    def _sample_start(self, enc_idx: int) -> int:
        use_impact = self.rng.random() < self.fraction
        t_impact = int(self.impact[enc_idx])
        T_max = self.T_total - self.T
        if T_max <= 0:
            return 0
        if use_impact:
            target = t_impact - self.T // 2
            lo = max(0, target - 8)
            hi = min(T_max, target + 8)
            if hi <= lo:
                return max(0, min(T_max, target))
            return int(self.rng.integers(lo, hi + 1))
        return int(self.rng.integers(0, T_max + 1))

    def __getitem__(self, _: int) -> dict[str, torch.Tensor]:
        i = int(self.rng.integers(0, self.n_enc))
        t0 = self._sample_start(i)
        sub_z = self.z[i, t0 : t0 + self.T, :]
        cond = torch.stack([self.G[i], self.D[i], self.Y[i]])
        return {"z": sub_z, "cond": cond}


def collate(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "z": torch.stack([s["z"] for s in samples]),
        "cond": torch.stack([s["cond"] for s in samples]),
    }


def build_lr_lambda(max_iters: int, warmup_frac: float, floor: float = 0.05):
    warmup = max(1, int(warmup_frac * max_iters))

    def fn(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(warmup)
        prog = min(1.0, (step - warmup) / max(1, max_iters - warmup))
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * prog))

    return fn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train baseline predictor (B1 Part c)")
    p.add_argument(
        "--latents-dir",
        type=Path,
        required=True,
        help="Directory containing train.npz / test_a.npz / test_b.npz / test_c.npz.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: outputs/session18/exp_b1/predictor_{tag}.",
    )
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--checkpoint-every", type=int, default=5000)
    p.add_argument("--max-iters", type=int, default=PROTOCOL_DEFAULTS["max_iters"])
    p.add_argument("--B", type=int, default=PROTOCOL_DEFAULTS["B"])
    p.add_argument("--T", type=int, default=PROTOCOL_DEFAULTS["T"])
    p.add_argument("--H-roll", type=int, default=PROTOCOL_DEFAULTS["H_roll"])
    p.add_argument(
        "--no-output-bn", action="store_true",
        help="Replace the predictor's output BatchNorm1d with Identity. Test 1 for the "
             "B1 BN-running-stats-mismatch hypothesis on JEPA latents.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = require_rtx6000(gpu_index=args.gpu)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.output_dir is None:
        args.output_dir = (
            REPO / "outputs" / "session18" / "exp_b1" / f"predictor_{args.tag}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train.log"
    metrics_path = args.output_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    def log(msg: str) -> None:
        line = msg if msg.endswith("\n") else msg + "\n"
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(line)

    log(f"[predictor-train] tag={args.tag}  device={device}  output={args.output_dir}")

    train_npz = np.load(args.latents_dir / "train.npz")
    z_full = train_npz["z_full"]
    d = int(z_full.shape[-1])
    log(
        f"[predictor-train] loaded latents: z_full={z_full.shape}  "
        f"d={d}  n_enc={z_full.shape[0]}"
    )

    dataset = LatentSubTrajectoryDataset(
        z_full=z_full,
        G=train_npz["G"],
        D=train_npz["D"],
        Y=train_npz["Y"],
        impact_frame=train_npz["impact_frame"],
        T=args.T,
        impact_aware_fraction=PROTOCOL_DEFAULTS["impact_aware_fraction"],
        seed=args.seed,
        n_samples_per_epoch=args.B * 256,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.B,
        num_workers=args.num_workers,
        collate_fn=collate,
        shuffle=False,
        drop_last=True,
    )

    predictor = AutoregressivePredictor(
        latent_dim=d,
        cond_dim=PROTOCOL_DEFAULTS["cond_dim"],
        hidden_dim=PROTOCOL_DEFAULTS["hidden_dim"],
        depth=PROTOCOL_DEFAULTS["depth"],
        heads=PROTOCOL_DEFAULTS["heads"],
        mlp_ratio=PROTOCOL_DEFAULTS["mlp_ratio"],
        dropout=PROTOCOL_DEFAULTS["dropout"],
        max_seq_len=args.T,
    ).to(device)
    if args.no_output_bn:
        # B1 Test 1: replace the output BN with Identity. The Linear stays.
        out_lin = predictor.out_proj[0]
        predictor.out_proj = nn.Sequential(out_lin, nn.Identity()).to(device)
        log("[predictor-train] --no-output-bn: replaced out_proj BatchNorm1d with Identity")
    n_params = sum(p.numel() for p in predictor.parameters())
    log(f"[predictor-train] AutoregressivePredictor params={n_params:,} ({n_params/1e6:.2f}M)")

    optimizer = AdamW(
        predictor.parameters(),
        lr=PROTOCOL_DEFAULTS["lr"],
        betas=(0.9, 0.95),
        weight_decay=PROTOCOL_DEFAULTS["weight_decay"],
    )
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=build_lr_lambda(args.max_iters, PROTOCOL_DEFAULTS["warmup_frac"]),
    )

    run_config = {
        "tag": args.tag,
        "latents_dir": str(args.latents_dir),
        "d": d,
        "seed": args.seed,
        "max_iters": args.max_iters,
        "B": args.B, "T": args.T, "H_roll": args.H_roll,
        "latent_norm": {
            "mean": dataset.mean.tolist(),
            "std": dataset.std.tolist(),
        },
        "predictor_config": {
            k: PROTOCOL_DEFAULTS[k]
            for k in (
                "hidden_dim",
                "depth",
                "heads",
                "mlp_ratio",
                "dropout",
                "max_seq_len",
                "cond_dim",
            )
        },
        "optimizer_config": {
            "lr": PROTOCOL_DEFAULTS["lr"],
            "weight_decay": PROTOCOL_DEFAULTS["weight_decay"],
            "warmup_frac": PROTOCOL_DEFAULTS["warmup_frac"],
            "grad_clip": PROTOCOL_DEFAULTS["grad_clip"],
            "betas": [0.9, 0.95],
        },
        "n_params": n_params,
        "gpu_name": torch.cuda.get_device_name(device.index),
    }
    with open(metrics_path, "w") as f:
        f.write(json.dumps({"event": "config", **run_config}) + "\n")
    log(f"[predictor-train] run_config: {json.dumps(run_config, indent=2)}")

    def infinite(loader: DataLoader):
        while True:
            for b in loader:
                yield b

    train_iter = infinite(loader)
    predictor.train()
    rng_rollout = np.random.default_rng(args.seed + 1)

    for it in range(args.max_iters):
        batch = next(train_iter)
        z_target = batch["z"].to(device, non_blocking=True)
        cond = batch["cond"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            z_hat = predictor(z_target, cond)
            L_tf = teacher_forced_prediction_loss(z_target, z_hat)

            start_max = args.T - args.H_roll - 1
            start_t = int(rng_rollout.integers(0, max(1, start_max + 1)))
            L_roll = open_loop_rollout_loss(
                predictor, z_target, cond, start_t=start_t, horizon=args.H_roll
            )
            L = L_tf + 0.5 * L_roll

        if not math.isfinite(L.item()):
            raise RuntimeError(
                f"non-finite loss at iter {it}: L={L.item()} "
                f"(L_tf={L_tf.item()}, L_roll={L_roll.item()})"
            )

        optimizer.zero_grad(set_to_none=True)
        L.backward()
        nn.utils.clip_grad_norm_(predictor.parameters(), PROTOCOL_DEFAULTS["grad_clip"])
        optimizer.step()
        scheduler.step()

        if it % args.log_every == 0:
            entry = {
                "event": "log",
                "iter": it,
                "L_total": float(L.item()),
                "L_tf": float(L_tf.item()),
                "L_roll": float(L_roll.item()),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
            with open(metrics_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
            log(
                f"[iter {it}/{args.max_iters}] L={L.item():.4f} "
                f"(tf={L_tf.item():.4f} roll={L_roll.item():.4f}) "
                f"lr={entry['lr']:.2e}"
            )

        if it > 0 and it % args.checkpoint_every == 0:
            ckpt = args.output_dir / f"checkpoint_iter{it:06d}.pt"
            torch.save(
                {
                    "predictor_state_dict": predictor.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "iteration": it,
                    "run_config": run_config,
                    "latent_mean": dataset.mean,
                    "latent_std": dataset.std,
                },
                ckpt,
            )
            log(f"[checkpoint] wrote {ckpt}")

    final_ckpt = args.output_dir / f"checkpoint_iter{args.max_iters:06d}.pt"
    torch.save(
        {
            "predictor_state_dict": predictor.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "iteration": args.max_iters,
            "run_config": run_config,
            "latent_mean": dataset.mean,
            "latent_std": dataset.std,
        },
        final_ckpt,
    )
    log(f"[predictor-train] FINAL checkpoint at {final_ckpt}")


if __name__ == "__main__":
    main()
