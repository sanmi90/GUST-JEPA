"""Reference-baseline trainer: the published Fukami lift-augmented AE on v2.2.

SESSION 31 Track C (references). Unlike the shared-backbone spine
(``train_canonical.py``), the references use their OWN architectures and are
trained on their own recipe. This entrypoint trains the Fukami/Taira lineage
autoencoder (:class:`~src.baselines.fukami_ae.FukamiAEWrapper`, native CNN
encoder/decoder + MLP lift head) under MSE reconstruction + ``beta * C_L`` (the
``fukami`` config), optionally with the wake-spectrum head (``fukami_wake``). No
anti-collapse (the field pins the latent).

The eval reads the trained encoder through the SAME frozen probes as the spine
(:func:`src.evaluation.rom_eval.load_reference_model`), so training here only has
to produce a checkpoint the loader can rebuild: ``model_state_dict`` is the full
:class:`FukamiAEWrapper` state and ``args['config']`` records the reference config.

POD (the linear reference) is NOT trained here; it has no neural parameters, see
``scripts/session31/fit_pod.py``.

Key normalisation fact: :class:`EpisodeDataset` with an omega pipeline manifest
returns 3-sigma-normalised omega per worker (D85), so the Fukami wrapper runs in
normalised space (``omega_scale=1.0``, no internal pipeline/clip) and its
reconstruction MSE is in normalised units, matching the matched AE spine and the
"loss in normalised space" rule (CLAUDE.md).

Hardware (CLAUDE.md, strict): ``require_rtx6000(gpu_index=args.gpu)`` first; the
run refuses to start unless an RTX 6000 Blackwell is visible.
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

from src.config.kit_config import load_model_config
from src.baselines.fukami_ae import FukamiAEWrapper
from src.data.episode_dataset import EpisodeDataset
from src.data.omega_pipeline import OmegaPipeline
from src.data.wake_observables import mode_output_dim
from src.models.observable_head import WakeObservableHead
from src.training.canonical_model import global_average_pool_latent  # noqa: F401 (kept for parity)
from src.training.diagnostics import linear_probe_r2, participation_ratio
from src.training.train_jepa import (
    file_sha256,
    git_commit_hash,
    infinite_iter,
    jepa_collate,
    move_batch,
    set_all_seeds,
)
from src.utils.device import require_rtx6000

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reference-baseline trainer (Fukami AE, Session 31)")
    p.add_argument(
        "--config",
        type=str,
        required=True,
        help="Reference kit config, e.g. configs/reference/fukami.yaml.",
    )
    p.add_argument("--partition", type=str, default="v2p2")
    p.add_argument("--split", type=str, default="configs/splits/split_v2p2.json")
    p.add_argument(
        "--pipeline-manifest",
        "--omega-pipeline-manifest",
        dest="pipeline_manifest",
        type=str,
        default="outputs/data_pipeline/v2p2/manifest.json",
    )
    p.add_argument("--max-iters", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--d", "--latent-dim", dest="d", type=int, default=32)
    p.add_argument("--B", type=int, default=16)
    p.add_argument("--T", "--sub-trajectory-length", dest="T", type=int, default=32)
    p.add_argument("--lr-encoder", type=float, default=1.5e-4)
    p.add_argument(
        "--lr-decoder",
        type=float,
        default=5e-4,
        help="LR for decoder + heads (the non-encoder Fukami parameters).",
    )
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--out",
        "--output-dir",
        dest="output_dir",
        type=str,
        default="outputs/runs/session31/fukami",
    )
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--diagnostic-every", type=int, default=1000)
    p.add_argument("--checkpoint-every", type=int, default=2500)
    p.add_argument(
        "--wandb-mode", type=str, choices=["online", "offline", "disabled"], default="offline"
    )
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Explicit torch device, bypassing require_rtx6000 (non-paper runs only).",
    )
    return p.parse_args()


def build_lr_lambda(max_iters: int, warmup_frac: float):
    warmup_iters = max(1, int(warmup_frac * max_iters))
    floor = 0.05

    def lr_lambda(step: int) -> float:
        if step < warmup_iters:
            return float(step + 1) / float(warmup_iters)
        progress = (step - warmup_iters) / max(1, max_iters - warmup_iters)
        progress = min(max(progress, 0.0), 1.0)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


def make_loader(args, cfg, split, batch_size, shuffle, num_workers) -> DataLoader:
    terms = cfg.active_terms()
    lift_on = "lift" in terms
    wake_on = "wake" in terms
    ds = EpisodeDataset(
        partition=args.partition,
        split=split,
        subtraj_len=args.T,
        emit_cl_future=lift_on,
        cl_future_deltas=(0,),  # current-frame C_L (Fukami eq 2.4 lift augmentation)
        emit_wake_observable=wake_on,
        wake_observable_type="patch_signed_spectrum",
        omega_pipeline_manifest=args.pipeline_manifest,
        split_manifest_path=args.split,
    )
    drop_last = shuffle and len(ds) >= batch_size
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=jepa_collate,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )


def build_wrapper(args, cfg, device) -> FukamiAEWrapper:
    terms = cfg.active_terms()
    wake_on = "wake" in terms
    wake_head = (
        WakeObservableHead(latent_dim=args.d, out_dim=mode_output_dim("patch_signed_spectrum"))
        if wake_on
        else None
    )
    wrapper = FukamiAEWrapper(
        latent_dim=args.d,
        n_deltas=1,
        lambda_recon=1.0,
        lambda_lift=1.0,  # beta (Fukami eq 2.4)
        omega_scale=1.0,  # dataset already returns 3-sigma normalised omega
        recon_loss_type="mse",  # MSE reconstruction (faithful Fukami)
        encoder_kind="cnn",  # native Fukami CNN encoder/decoder
        wake_observable_head=wake_head,
        wake_observable_weight=1.0 if wake_on else 0.0,
        wake_loss_kind="smooth_l1",
        wake_loss_beta=0.5,
    )
    return wrapper.to(device)


@torch.no_grad()
def run_diagnostics(wrapper, test_b_iter, device, seed) -> dict[str, float]:
    """PR + linear-probe R^2 for (G, D, Y) on a Test B batch (convergence sanity)."""
    batch = move_batch(next(test_b_iter), device)
    was_training = wrapper.training
    wrapper.eval()
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        z = wrapper.encoder(batch["omega"] / wrapper.omega_scale)
    if was_training:
        wrapper.train()
    z32 = z.detach().float().flatten(0, 1)
    c_static = batch["c"].detach().float()
    c_rep = c_static.unsqueeze(1).expand(-1, z.shape[1], -1).reshape(-1, c_static.shape[1])
    n = z32.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    perm = torch.randperm(n, generator=g)
    cut = max(2, int(0.75 * n))
    fit, ev = perm[:cut], perm[cut:]
    if ev.numel() < 2:
        half = n // 2
        fit, ev = perm[:half], perm[half:]
    pr = participation_ratio(z32)
    r2 = linear_probe_r2(z32, c_rep, fit, ev)
    return {"pr": pr, **r2}


def main() -> None:
    args = parse_args()
    device = (
        torch.device(args.device)
        if getattr(args, "device", None)
        else require_rtx6000(gpu_index=args.gpu)
    )

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    cfg = load_model_config(config_path)
    if cfg.model.get("encoder") != "fukami_native":
        raise ValueError(
            f"train_reference only trains the Fukami native AE; config {cfg.name!r} has "
            f"model.encoder={cfg.model.get('encoder')!r}. POD is fit by "
            f"scripts/session31/fit_pod.py."
        )
    set_all_seeds(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_path = Path(args.split)
    if not split_path.is_absolute():
        split_path = REPO_ROOT / split_path
    with open(split_path) as f:
        split_manifest = json.load(f)
    with open(REPO_ROOT / "configs" / "preprocessing.yaml") as f:
        preprocessing_cfg = yaml.safe_load(f)

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device.index)
        _allow_non = os.environ.get("VORTEX_JEPA_ALLOW_NON_RTX6000", "").strip()
        if not _allow_non and ("RTX" not in gpu_name or "6000" not in gpu_name):
            raise RuntimeError(
                f"Hardware policy violation (CLAUDE.md): gpu_name={gpu_name!r} does not "
                "contain both 'RTX' and '6000'. Run aborted."
            )
    else:
        gpu_name = str(device)

    # Confirm the pipeline manifest is the v2.2 one and print its stats.
    manifest_path = Path(args.pipeline_manifest)
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    pipe = OmegaPipeline.from_manifest(manifest_path)
    print(
        f"[train_reference] pipeline {manifest_path} mask={int(pipe.mask.sum().item())} "
        f"train_std={pipe.train_stats.std:.4f}",
        flush=True,
    )

    wrapper = build_wrapper(args, cfg, device)

    run_config = {
        "preprocessing_version": preprocessing_cfg["preprocessing_version"],
        "partition_version": args.partition,
        "lambda_sigreg": 0.0,
        "seed": args.seed,
        "split_sha256": file_sha256(split_path),
        "inventory_sha256": split_manifest["source_inventory"]["sha256"],
        "code_sha256": git_commit_hash(),
        "auto_fallback_triggered": False,
        "gpu_name": gpu_name,
        "model_name": cfg.name,
        "objective": cfg.objective,
        "active_terms": sorted(cfg.active_terms()),
        "reference": True,
        "encoder": "fukami_native",
        "max_iters": args.max_iters,
        "B": args.B,
        "T": args.T,
        "d": args.d,
        "lr_encoder": args.lr_encoder,
        "lr_decoder": args.lr_decoder,
        "weight_decay": args.weight_decay,
        "warmup_frac": args.warmup_frac,
        "pipeline_manifest": str(args.pipeline_manifest),
        "config_path": str(config_path),
    }

    import wandb

    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "vortex-jepa"),
        group=f"partition_{args.partition}",
        tags=[cfg.name, "fukami_ae", "none"],
        mode=args.wandb_mode,
        config=run_config,
        dir=str(output_dir),
    )
    wandb.run.summary["wandb_run_id"] = wandb.run.id

    metrics_jsonl = output_dir / "metrics.jsonl"
    with open(metrics_jsonl, "w") as f:
        f.write(json.dumps({"event": "config", "wandb_run_id": wandb.run.id, **run_config}) + "\n")

    def _log(payload: dict[str, Any], step: int) -> None:
        wandb.log(payload, step=step)
        with open(metrics_jsonl, "a") as fh:
            fh.write(json.dumps({"event": "log", "step": int(step), **payload}) + "\n")

    train_loader = make_loader(args, cfg, "train", args.B, True, args.num_workers)
    test_b_loader = make_loader(args, cfg, "test_b", args.B, False, 0)
    train_iter = infinite_iter(train_loader)
    test_b_iter = infinite_iter(test_b_loader)

    enc_params = list(wrapper.encoder.parameters())
    downstream = list(wrapper.decoder.parameters()) + list(wrapper.lift_head.parameters())
    if wrapper.wake_observable_head is not None:
        downstream += list(wrapper.wake_observable_head.parameters())
    optimizer = AdamW(
        [
            {"params": enc_params, "lr": args.lr_encoder},
            {"params": downstream, "lr": args.lr_decoder},
        ],
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    scheduler = LambdaLR(optimizer, lr_lambda=build_lr_lambda(args.max_iters, args.warmup_frac))

    wrapper.train()
    print(
        f"[train_reference] model={cfg.name} terms={sorted(cfg.active_terms())} "
        f"device={device} gpu={gpu_name} n_train={len(train_loader.dataset)} "
        f"n_test_b={len(test_b_loader.dataset)} B={args.B} T={args.T}",
        flush=True,
    )

    last_total = float("nan")
    for iteration in range(args.max_iters):
        batch = move_batch(next(train_iter), device)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            out = wrapper(batch)
        loss_total = out["L_total"]
        last_total = float(loss_total.item())
        if not math.isfinite(last_total):
            raise RuntimeError(
                f"non-finite loss at iter {iteration}: {last_total} "
                f"(recon={out['L_recon'].item()}, lift={out['L_lift'].item()}, "
                f"wake={out['L_wake'].item()})"
            )
        optimizer.zero_grad(set_to_none=True)
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(wrapper.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        if iteration % args.log_every == 0:
            _log(
                {
                    "iter": iteration,
                    "loss_total": last_total,
                    "loss_recon": out["L_recon"].item(),
                    "loss_lift": out["L_lift"].item(),
                    "loss_wake": out["L_wake"].item(),
                    "lr_encoder": optimizer.param_groups[0]["lr"],
                    "lr_decoder": optimizer.param_groups[1]["lr"],
                },
                step=iteration,
            )
            print(
                f"[iter {iteration}/{args.max_iters}] L={last_total:.4f} "
                f"(recon={out['L_recon'].item():.4f}, lift={out['L_lift'].item():.4f}, "
                f"wake={out['L_wake'].item():.4f})",
                flush=True,
            )

        if iteration % args.diagnostic_every == 0:
            diag = run_diagnostics(wrapper, test_b_iter, device, args.seed)
            _log({f"diag/{k}": v for k, v in diag.items()}, step=iteration)
            print(
                f"[diag iter {iteration}] PR={diag['pr']:.2f} r2_overall={diag['r2_overall']:.3f}",
                flush=True,
            )

        if iteration > 0 and iteration % args.checkpoint_every == 0:
            ckpt = output_dir / f"checkpoint_iter{iteration:06d}.pt"
            torch.save(
                {
                    "iteration": iteration,
                    "model_state_dict": wrapper.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "args": vars(args),
                    "run_config": run_config,
                },
                ckpt,
            )
            print(f"[checkpoint] wrote {ckpt}", flush=True)

    final = output_dir / f"checkpoint_iter{args.max_iters:06d}.pt"
    torch.save(
        {
            "iteration": args.max_iters,
            "model_state_dict": wrapper.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "args": vars(args),
            "run_config": run_config,
        },
        final,
    )
    print(f"[final checkpoint] wrote {final}", flush=True)
    wandb.run.summary["loss_total_final"] = last_total
    wandb.run.summary["final_iter"] = args.max_iters
    wandb.finish()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[train_reference] interrupted", flush=True)
        sys.exit(130)
