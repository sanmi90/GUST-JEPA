"""Argparse training entrypoint for the JEPA wrapper.

Session 4 deliverable (SESSION4_JEPA_WRAPPER_AND_TRAINING_SCAFFOLD.md):
    Compose encoder + predictor + SIGReg + scheduled-sampling + diagnostics +
    auto-fallback into a runnable training loop, with bf16 autocast and W&B
    logging that satisfies the four required keys plus the seven paper-grade
    keys (CLAUDE.md "Logging").

Reference recipes:
    Maes et al. arXiv:2603.19312 (loss composition).
    Bardes, Ponce, LeCun arXiv:2105.04906 (VICReg fallback coefficients).
    Assran et al. arXiv:2506.09985 (scheduled-sampling).

Hardware (CLAUDE.md "Hardware", HANDOFF.md D19): the first call inside
``main()`` is ``require_rtx6000()`` and the script refuses to run if the
RTX 6000 Blackwell is not visible to torch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import yaml
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.data.episode_dataset import EpisodeDataset
from src.models.encoder import HybridCNNViTEncoder
from src.models.jepa import JEPA
from src.models.predictor import AutoregressivePredictor
from src.models.sigreg import SIGReg
from src.models.vicreg import VICReg
from src.training.auto_fallback import AutoFallbackController
from src.training.diagnostics import (
    linear_probe_r2,
    participation_ratio,
    per_dim_variance_histogram,
)
from src.utils.device import require_rtx6000


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the JEPA on partition v1")
    p.add_argument("--partition", type=str, default="v1")
    p.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help="Subset of case_ids to train on. Default: all train cases.",
    )
    p.add_argument("--max-iters", type=int, default=80_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--B", type=int, default=16)
    p.add_argument("--T", type=int, default=32)
    p.add_argument("--H-roll", type=int, default=8)
    p.add_argument("--lambda-sigreg", type=float, default=0.1)
    p.add_argument("--lr-encoder", type=float, default=1.5e-4)
    p.add_argument("--lr-predictor", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output-dir", type=str, default="outputs/runs/smoke")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--diagnostic-every", type=int, default=1000)
    p.add_argument("--checkpoint-every", type=int, default=10_000)
    p.add_argument(
        "--wandb-mode",
        type=str,
        choices=["online", "offline", "disabled"],
        default="online",
    )
    return p.parse_args()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        )
        return out.strip().decode()
    except Exception:
        return "unknown"


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def jepa_collate(samples: list[dict[str, Any]]) -> dict[str, Tensor]:
    omega = torch.stack([s["omega_z"].unsqueeze(1) for s in samples])
    c = torch.tensor(
        [[s["G"], s["D"], s["Y"]] for s in samples],
        dtype=torch.float32,
    )
    return {"omega": omega, "c": c}


def make_train_loader(args: argparse.Namespace) -> DataLoader:
    ds = EpisodeDataset(partition=args.partition, split="train", subtraj_len=args.T)
    if args.cases is not None:
        wanted = set(args.cases)
        ds.samples = [s for s in ds.samples if s[0] in wanted]
        if not ds.samples:
            raise RuntimeError(
                f"No training samples after filtering to cases={args.cases}. "
                f"Check spelling and that cases are in 'train' split."
            )
    drop_last = len(ds) >= args.B
    return DataLoader(
        ds,
        batch_size=args.B,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=jepa_collate,
        drop_last=drop_last,
        persistent_workers=args.num_workers > 0,
    )


def infinite_iter(loader: DataLoader) -> Iterator[dict[str, Tensor]]:
    """Yield batches indefinitely, re-iterating (and re-shuffling) per epoch.

    ``itertools.cycle`` would cache the first iteration's batches and
    return them forever, so shuffling never refreshes. This generator
    calls ``iter(loader)`` each epoch instead.
    """
    while True:
        for batch in loader:
            yield batch


def make_test_b_loader(args: argparse.Namespace) -> DataLoader:
    ds = EpisodeDataset(partition=args.partition, split="test_b", subtraj_len=args.T)
    return DataLoader(
        ds,
        batch_size=args.B,
        shuffle=False,
        num_workers=0,
        collate_fn=jepa_collate,
        drop_last=False,
    )


def build_lr_lambda(args: argparse.Namespace) -> "callable[[int], float]":
    warmup_iters = max(1, int(args.warmup_frac * args.max_iters))
    floor = 0.05

    def lr_lambda(step: int) -> float:
        if step < warmup_iters:
            return float(step + 1) / float(warmup_iters)
        progress = (step - warmup_iters) / max(1, args.max_iters - warmup_iters)
        progress = min(max(progress, 0.0), 1.0)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


def move_batch(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def run_diagnostics(
    jepa: JEPA,
    test_b_iter: "itertools.cycle[dict[str, Tensor]]",
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    """Compute PR + linear probe R^2 + variance histogram on a Test B batch."""
    batch = move_batch(next(test_b_iter), device)
    was_training = jepa.training
    jepa.eval()
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            z = jepa.encoder(batch["omega"])
    if was_training:
        jepa.train()

    z32 = z.detach().float().flatten(0, 1)
    c_static = batch["c"].detach().float()
    c_repeated = c_static.unsqueeze(1).expand(-1, z.shape[1], -1).reshape(-1, c_static.shape[1])

    n = z32.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    perm = torch.randperm(n, generator=g)
    fit = perm[: max(2, int(0.75 * n))]
    ev = perm[max(2, int(0.75 * n)) :]
    if ev.numel() < 2:
        fit = perm[: n // 2]
        ev = perm[n // 2 :]

    pr = participation_ratio(z32)
    r2 = linear_probe_r2(z32, c_repeated, fit, ev)
    counts, edges = per_dim_variance_histogram(z32)
    return {
        "pr": pr,
        **r2,
        "var_hist_counts_zero_bin": counts[0].item(),
        "var_hist_max_edge": edges[-1].item(),
    }


def main() -> None:
    args = parse_args()
    device = require_rtx6000()
    set_all_seeds(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_path = REPO_ROOT / "configs" / "splits" / f"split_{args.partition}.json"
    with open(split_path) as f:
        split_manifest = json.load(f)
    with open(REPO_ROOT / "configs" / "preprocessing.yaml") as f:
        preprocessing_cfg = yaml.safe_load(f)

    gpu_name = torch.cuda.get_device_name(device.index)
    if "RTX" not in gpu_name or "6000" not in gpu_name:
        raise RuntimeError(
            f"Hardware policy violation (CLAUDE.md): gpu_name={gpu_name!r} does not "
            "contain both 'RTX' and '6000'. Run aborted."
        )

    run_config = {
        "preprocessing_version": preprocessing_cfg["preprocessing_version"],
        "partition_version": args.partition,
        "lambda_sigreg": args.lambda_sigreg,
        "seed": args.seed,
        "split_sha256": file_sha256(split_path),
        "inventory_sha256": split_manifest["source_inventory"]["sha256"],
        "code_sha256": git_commit_hash(),
        "auto_fallback_triggered": False,
        "gpu_name": gpu_name,
        "max_iters": args.max_iters,
        "B": args.B,
        "T": args.T,
        "d": args.d,
        "H_roll": args.H_roll,
        "lr_encoder": args.lr_encoder,
        "lr_predictor": args.lr_predictor,
        "weight_decay": args.weight_decay,
        "warmup_frac": args.warmup_frac,
        "cases": args.cases,
    }

    import wandb

    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "vortex-jepa"),
        group=f"partition_{args.partition}",
        tags=["hybrid_cnn_vit", "sigreg"],
        mode=args.wandb_mode,
        config=run_config,
        dir=str(output_dir),
    )
    wandb.run.summary["wandb_run_id"] = wandb.run.id

    train_loader = make_train_loader(args)
    test_b_loader = make_test_b_loader(args)
    train_iter = infinite_iter(train_loader)
    test_b_iter = infinite_iter(test_b_loader)

    encoder = HybridCNNViTEncoder(latent_dim=args.d)
    predictor = AutoregressivePredictor(
        latent_dim=args.d,
        cond_dim=3,
        max_seq_len=args.T,
    )
    sigreg = SIGReg(dim=args.d)
    jepa = JEPA(
        encoder=encoder,
        predictor=predictor,
        anticollapse=sigreg,
        lambda_anticollapse=args.lambda_sigreg,
        rollout_weight=0.5,
        H_roll=args.H_roll,
        rollout_start_strategy="uniform_random",
    ).to(device)

    optimizer = AdamW(
        [
            {"params": list(jepa.encoder.parameters()), "lr": args.lr_encoder},
            {"params": list(jepa.predictor.parameters()), "lr": args.lr_predictor},
        ],
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    scheduler = LambdaLR(optimizer, lr_lambda=build_lr_lambda(args))
    controller = AutoFallbackController(d=args.d)

    jepa.train()
    print(
        f"[train_jepa] device={device} gpu={gpu_name} "
        f"n_train_samples={len(train_loader.dataset)} "
        f"n_test_b_samples={len(test_b_loader.dataset)}",
        flush=True,
    )

    last_loss_total = float("nan")
    for iteration in range(args.max_iters):
        batch = move_batch(next(train_iter), device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = jepa(batch)

        loss_total = out["loss_total"]
        last_loss_total = float(loss_total.item())
        if not math.isfinite(last_loss_total):
            raise RuntimeError(
                f"non-finite loss at iter {iteration}: loss_total={last_loss_total} "
                f"(pred={out['loss_pred'].item()}, roll={out['loss_roll'].item()}, "
                f"anti={out['loss_anticollapse'].item()})"
            )

        optimizer.zero_grad(set_to_none=True)
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(jepa.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        if iteration % args.log_every == 0:
            wandb.log(
                {
                    "iter": iteration,
                    "loss_total": last_loss_total,
                    "loss_pred": out["loss_pred"].item(),
                    "loss_roll": out["loss_roll"].item(),
                    "loss_anticollapse": out["loss_anticollapse"].item(),
                    "lr_encoder": optimizer.param_groups[0]["lr"],
                    "lr_predictor": optimizer.param_groups[1]["lr"],
                },
                step=iteration,
            )
            print(
                f"[iter {iteration}/{args.max_iters}] loss={last_loss_total:.4f} "
                f"(pred={out['loss_pred'].item():.4f}, "
                f"roll={out['loss_roll'].item():.4f}, "
                f"anti={out['loss_anticollapse'].item():.4f})",
                flush=True,
            )

        if iteration % args.diagnostic_every == 0:
            diag = run_diagnostics(jepa, test_b_iter, device, args.seed)
            wandb.log({f"diag/{k}": v for k, v in diag.items()}, step=iteration)
            print(
                f"[diag iter {iteration}] PR={diag['pr']:.2f} "
                f"r2_overall={diag['r2_overall']:.3f}",
                flush=True,
            )
            fired = controller.step(iteration, diag["pr"], diag["r2_overall"])
            if fired:
                print(
                    f"[AUTO-FALLBACK] firing SIGReg->VICReg at iter {iteration} "
                    f"(PR={diag['pr']:.2f}, r2={diag['r2_overall']:.3f})",
                    flush=True,
                )
                vicreg = VICReg(d=args.d).to(device)
                jepa.set_anticollapse(vicreg)
                wandb.run.summary["auto_fallback_triggered"] = True
                wandb.run.summary["auto_fallback_iter"] = iteration

        if iteration > 0 and iteration % args.checkpoint_every == 0:
            ckpt_path = output_dir / f"checkpoint_iter{iteration:06d}.pt"
            torch.save(
                {
                    "iteration": iteration,
                    "jepa_state_dict": jepa.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "args": vars(args),
                    "run_config": run_config,
                },
                ckpt_path,
            )
            print(f"[checkpoint] wrote {ckpt_path}", flush=True)

    final_path = output_dir / f"checkpoint_iter{args.max_iters:06d}.pt"
    torch.save(
        {
            "iteration": args.max_iters,
            "jepa_state_dict": jepa.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "args": vars(args),
            "run_config": run_config,
        },
        final_path,
    )
    print(f"[final checkpoint] wrote {final_path}", flush=True)
    wandb.run.summary["loss_total_final"] = last_loss_total
    wandb.run.summary["final_iter"] = args.max_iters
    wandb.finish()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[train_jepa] interrupted", flush=True)
        sys.exit(130)
