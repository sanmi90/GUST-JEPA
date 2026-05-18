"""Argparse training entrypoint for the matched-capacity baselines.

Session 5.PLDM deliverable (SESSION5_PLDM_BASELINE.md): a parallel
training entrypoint for the PLDM baseline that reuses the JEPA's
encoder, predictor, data loader, diagnostics, and W&B contract; only
the loss composition changes.

Dispatching on ``--baseline``:
    pldm           -> :class:`src.models.pldm_wrapper.PLDMWrapper` with
                       :class:`src.baselines.pldm.PLDMLoss`.
    fukami_ae      -> not yet implemented.
    solera_rico    -> not yet implemented.
    pod            -> not a training run; use ``scripts/run_pod_baseline.py``.

Hardware (CLAUDE.md "Hardware", HANDOFF.md D19): the first call inside
``main()`` is ``require_rtx6000()``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.baselines.pldm import PLDMLoss
from src.data.episode_dataset import EpisodeDataset
from src.models.encoder import HybridCNNViTEncoder
from src.models.pldm_wrapper import PLDMWrapper
from src.models.predictor import AutoregressivePredictor
from src.training.diagnostics import (
    linear_probe_r2,
    participation_ratio,
    per_dim_variance_histogram,
)
from src.training.train_jepa import (
    REPO_ROOT,
    build_lr_lambda,
    file_sha256,
    git_commit_hash,
    infinite_iter,
    jepa_collate,
    move_batch,
    resolve_cases,
    set_all_seeds,
)
from src.utils.device import require_rtx6000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a matched-capacity baseline")
    p.add_argument("--baseline", type=str, choices=["pldm", "fukami_ae", "solera_rico", "pod"],
                   required=True)
    p.add_argument("--partition", type=str, default="v1")
    p.add_argument("--cases", nargs="+", default=None)
    p.add_argument("--cases-from", type=str, default=None)
    p.add_argument("--max-iters", type=int, default=80_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--B", type=int, default=16)
    p.add_argument("--T", type=int, default=32)
    p.add_argument("--H-roll", type=int, default=8,
                   help="PLDM prediction horizon (paper's H). Matches the JEPA H_roll for the comparison.")
    p.add_argument("--projection-norm", type=str, choices=["batchnorm", "layernorm"],
                   default="batchnorm")
    p.add_argument("--lr-encoder", type=float, default=1.5e-4)
    p.add_argument("--lr-predictor", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output-dir", type=str, default="outputs/runs/baseline")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--diagnostic-every", type=int, default=1000)
    p.add_argument("--checkpoint-every", type=int, default=10_000)
    p.add_argument("--wandb-mode", type=str, choices=["online", "offline", "disabled"],
                   default="online")
    p.add_argument("--tag-suffix", type=str, default="")
    # PLDM weights (defaults match the all-1.0 placeholders in PLDMLoss; the
    # paper's Two-Rooms / PointMaze / Ant-U-Maze defaults vary widely,
    # see arXiv:2502.14819 Appendix J.2).
    p.add_argument("--lambda-var", type=float, default=1.0)
    p.add_argument("--lambda-cov", type=float, default=1.0)
    p.add_argument("--lambda-time-sim", type=float, default=1.0)
    p.add_argument("--lambda-idm", type=float, default=1.0)
    p.add_argument("--pldm-gamma", type=float, default=1.0)
    return p.parse_args()


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
        ds, batch_size=args.B, shuffle=True, num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(), collate_fn=jepa_collate,
        drop_last=drop_last, persistent_workers=args.num_workers > 0,
    )


def make_test_b_loader(args: argparse.Namespace) -> DataLoader:
    ds = EpisodeDataset(partition=args.partition, split="test_b", subtraj_len=args.T)
    return DataLoader(
        ds, batch_size=args.B, shuffle=False, num_workers=0,
        collate_fn=jepa_collate, drop_last=False,
    )


def run_diagnostics(
    wrapper: PLDMWrapper,
    test_b_iter,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    """Compute PR + linear probe R^2 + variance histogram on a Test B batch.

    Mirrors ``src.training.train_jepa.run_diagnostics`` but reads
    ``wrapper.encoder`` instead of ``jepa.encoder``.
    """
    batch = move_batch(next(test_b_iter), device)
    was_training = wrapper.training
    wrapper.eval()
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            z = wrapper.encoder(batch["omega"])
    if was_training:
        wrapper.train()

    z32 = z.detach().float().flatten(0, 1)
    c_static = batch["c"].detach().float()
    c_repeated = c_static.unsqueeze(1).expand(-1, z.shape[1], -1).reshape(-1, c_static.shape[1])

    n = z32.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    perm = torch.randperm(n, generator=g)
    fit = perm[: max(2, int(0.75 * n))]
    ev = perm[max(2, int(0.75 * n)):]
    if ev.numel() < 2:
        fit = perm[: n // 2]
        ev = perm[n // 2:]

    pr = participation_ratio(z32)
    r2 = linear_probe_r2(z32, c_repeated, fit, ev)
    counts, edges = per_dim_variance_histogram(z32)
    return {
        "pr": pr,
        **r2,
        "var_hist_counts_zero_bin": counts[0].item(),
        "var_hist_max_edge": edges[-1].item(),
    }


def build_pldm(args: argparse.Namespace, device: torch.device) -> PLDMWrapper:
    encoder = HybridCNNViTEncoder(latent_dim=args.d, projection_norm=args.projection_norm)
    predictor = AutoregressivePredictor(
        latent_dim=args.d, cond_dim=3, max_seq_len=args.T,
    )
    loss = PLDMLoss(
        d=args.d, c_dim=3,
        lambda_var=args.lambda_var,
        lambda_cov=args.lambda_cov,
        lambda_time_sim=args.lambda_time_sim,
        lambda_idm=args.lambda_idm,
        gamma=args.pldm_gamma,
    )
    return PLDMWrapper(
        encoder=encoder, predictor=predictor, loss=loss,
        prediction_horizon=args.H_roll,
    ).to(device)


def main() -> None:
    args = parse_args()
    args.cases = resolve_cases(args)
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
            f"Hardware policy violation: gpu_name={gpu_name!r} does not "
            "contain both 'RTX' and '6000'."
        )

    if args.baseline != "pldm":
        raise NotImplementedError(
            f"baseline {args.baseline!r} not yet implemented; only 'pldm' lands in Session 5.PLDM."
        )

    run_config = {
        "preprocessing_version": preprocessing_cfg["preprocessing_version"],
        "partition_version": args.partition,
        "baseline": args.baseline,
        # PLDM has no single "lambda_sigreg" key; the W&B contract uses the
        # four PLDM lambdas. We keep `lambda_sigreg=None` so the four required
        # keys (CLAUDE.md "Logging") are still emitted, just with a null value
        # for the field that does not apply to this objective.
        "lambda_sigreg": None,
        "lambda_var": args.lambda_var,
        "lambda_cov": args.lambda_cov,
        "lambda_time_sim": args.lambda_time_sim,
        "lambda_idm": args.lambda_idm,
        "pldm_gamma": args.pldm_gamma,
        "seed": args.seed,
        "split_sha256": file_sha256(split_path),
        "inventory_sha256": split_manifest["source_inventory"]["sha256"],
        "code_sha256": git_commit_hash(),
        "auto_fallback_triggered": False,
        "gpu_name": gpu_name,
        "max_iters": args.max_iters,
        "B": args.B, "T": args.T, "d": args.d, "H_roll": args.H_roll,
        "lr_encoder": args.lr_encoder, "lr_predictor": args.lr_predictor,
        "weight_decay": args.weight_decay, "warmup_frac": args.warmup_frac,
        "cases": args.cases, "projection_norm": args.projection_norm,
        "tag_suffix": args.tag_suffix,
    }

    import wandb

    wandb_tags = ["hybrid_cnn_vit", "pldm_5term"]
    if args.tag_suffix:
        wandb_tags.append(f"run:{args.tag_suffix}")

    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "vortex-jepa"),
        group=f"partition_{args.partition}",
        tags=wandb_tags, mode=args.wandb_mode, config=run_config, dir=str(output_dir),
    )
    wandb.run.summary["wandb_run_id"] = wandb.run.id

    metrics_jsonl = output_dir / "metrics.jsonl"
    with open(metrics_jsonl, "w") as f:
        f.write(json.dumps({"event": "config", "wandb_run_id": wandb.run.id, **run_config}) + "\n")

    def _log_metrics(payload: dict[str, Any], step: int) -> None:
        wandb.log(payload, step=step)
        with open(metrics_jsonl, "a") as fh:
            fh.write(json.dumps({"event": "log", "step": int(step), **payload}) + "\n")

    train_loader = make_train_loader(args)
    test_b_loader = make_test_b_loader(args)
    train_iter = infinite_iter(train_loader)
    test_b_iter = infinite_iter(test_b_loader)

    wrapper = build_pldm(args, device)

    # IDM-MLP parameters go in the predictor group (closest semantically; both
    # are part of the latent-dynamics machinery). Encoder keeps its own group.
    optimizer = AdamW(
        [
            {"params": list(wrapper.encoder.parameters()), "lr": args.lr_encoder},
            {"params": list(wrapper.predictor.parameters()) + list(wrapper.loss.idm.parameters()),
             "lr": args.lr_predictor},
        ],
        betas=(0.9, 0.95), weight_decay=args.weight_decay,
    )
    scheduler = LambdaLR(optimizer, lr_lambda=build_lr_lambda(args))

    wrapper.train()
    print(
        f"[train_baseline] baseline=pldm device={device} gpu={gpu_name} "
        f"n_train_samples={len(train_loader.dataset)} "
        f"n_test_b_samples={len(test_b_loader.dataset)}",
        flush=True,
    )

    last_loss_total = float("nan")
    for iteration in range(args.max_iters):
        batch = move_batch(next(train_iter), device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = wrapper(batch)

        L_total = out["L_total"]
        last_loss_total = float(L_total.item())
        if not math.isfinite(last_loss_total):
            raise RuntimeError(
                f"non-finite loss at iter {iteration}: L_total={last_loss_total} "
                f"(sim={out['L_sim'].item()}, var={out['L_var'].item()}, "
                f"cov={out['L_cov'].item()}, time={out['L_time_sim'].item()}, "
                f"idm={out['L_idm'].item()})"
            )

        optimizer.zero_grad(set_to_none=True)
        L_total.backward()
        torch.nn.utils.clip_grad_norm_(wrapper.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        if iteration % args.log_every == 0:
            _log_metrics(
                {
                    "iter": iteration,
                    "loss_total": last_loss_total,
                    "L_sim": out["L_sim"].item(),
                    "L_var": out["L_var"].item(),
                    "L_cov": out["L_cov"].item(),
                    "L_time_sim": out["L_time_sim"].item(),
                    "L_idm": out["L_idm"].item(),
                    "lr_encoder": optimizer.param_groups[0]["lr"],
                    "lr_predictor": optimizer.param_groups[1]["lr"],
                },
                step=iteration,
            )
            print(
                f"[iter {iteration}/{args.max_iters}] L={last_loss_total:.4f} "
                f"(sim={out['L_sim'].item():.4f}, var={out['L_var'].item():.4f}, "
                f"cov={out['L_cov'].item():.4f}, time={out['L_time_sim'].item():.4f}, "
                f"idm={out['L_idm'].item():.4f})",
                flush=True,
            )

        if iteration % args.diagnostic_every == 0:
            diag = run_diagnostics(wrapper, test_b_iter, device, args.seed)
            _log_metrics({f"diag/{k}": v for k, v in diag.items()}, step=iteration)
            print(
                f"[diag iter {iteration}] PR={diag['pr']:.2f} "
                f"r2_overall={diag['r2_overall']:.3f}",
                flush=True,
            )

        if iteration > 0 and iteration % args.checkpoint_every == 0:
            ckpt_path = output_dir / f"checkpoint_iter{iteration:06d}.pt"
            torch.save(
                {
                    "iteration": iteration,
                    "wrapper_state_dict": wrapper.state_dict(),
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
            "wrapper_state_dict": wrapper.state_dict(),
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
        print("[train_baseline] interrupted", flush=True)
        sys.exit(130)
