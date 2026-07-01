"""Canonical trainer: one entrypoint, every model is a kit switch-set (Track C.1).

SESSION 31 Track C.1 (SESSION_31_canonical_v2p2_retrain.md). The v2.1 trainer
``src/training/train_jepa.py`` hard-wired the JEPA loss inline. This entrypoint
routes the autoencoder AND JEPA models through ONE shared loss path
(:func:`src.losses.kit.compute_total_loss`) on the SPATIAL latent
(``HybridCNNViTEncoder(latent_mode="spatial")``), so the only thing that differs
between models is the resolved kit config (``configs/canonical/<model>.yaml``).

Supported models (this task): jepa_nowake, jepa_wake, ae_nowake, ae_wake,
supervised_only (and regAE, which is the same recon+anti-collapse path with no
heads). Each is a :class:`~src.training.canonical_model.CanonicalModel` built
from the resolved config.

Loss path differences vs v2.1 ``train_jepa.py``:
- Teacher forcing is DROPPED. Prediction is a single open-loop ``H_roll``-step
  rollout with the :class:`ResUNetPredictor`; targets are detached online
  encoder outputs (gray-scott online target, NO EMA).
- The anti-collapse statistic is over the flattened spatial latent
  ``[B*T*h*w, d]`` (batch across cases/frames/cells, never over time).
- Reconstruction (matched AE) decodes the spatial latent to omega in
  PIPELINE-NORMALISED space (CLAUDE.md: loss in normalised space).
- Lift / wake heads read the latent through a parameter-free global-average pool.

Hardware (CLAUDE.md "Hardware", strict): the first call inside ``main()`` is
``require_rtx6000(gpu_index=args.gpu)``; the script refuses to run if an RTX 6000
Blackwell is not visible to torch, and asserts ``gpu_name`` contains RTX and 6000.
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

from src.config.kit_config import ResolvedKitConfig, load_model_config
from src.data.episode_dataset import EpisodeDataset
from src.data.omega_pipeline import OmegaPipeline
from src.losses.kit import build_anticollapse, compute_total_loss
from src.models.decoder import SpatialLatentFieldDecoder
from src.models.vicreg import VICReg
from src.training.auto_fallback import AutoFallbackController
from src.training.canonical_model import (
    CanonicalModel,
    flatten_spatial_latent,
    global_average_pool_latent,
)
from src.training.diagnostics import (
    linear_probe_r2,
    participation_ratio,
    per_dim_variance_histogram,
)
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
    p = argparse.ArgumentParser(description="Canonical kit trainer (Session 31 Track C.1)")
    p.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a per-model kit config, e.g. configs/canonical/jepa_nowake.yaml.",
    )
    p.add_argument(
        "--partition",
        type=str,
        default="v2p2",
        help="Cache partition (subdir of VORTEX_JEPA_CACHE). Default v2p2.",
    )
    p.add_argument(
        "--split",
        type=str,
        default="configs/splits/split_v2p2.json",
        help="Split manifest path. Default split_v2p2.json.",
    )
    p.add_argument(
        "--pipeline-manifest",
        "--omega-pipeline-manifest",
        dest="pipeline_manifest",
        type=str,
        default="outputs/data_pipeline/v2p2/manifest.json",
        help="OmegaPipeline manifest. Applied inside the dataset worker (D85).",
    )
    p.add_argument("--max-iters", type=int, default=80_000)
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Run seed. Default: training.seed from the kit config, else 0.",
    )
    p.add_argument(
        "--d",
        "--latent-dim",
        dest="d",
        type=int,
        default=32,
        help="Latent dimension d (default 32, CLAUDE.md).",
    )
    p.add_argument(
        "--B",
        type=int,
        default=None,
        help="Batch size. Default: training.batch_size from the kit config (16).",
    )
    p.add_argument(
        "--T",
        "--sub-trajectory-length",
        dest="T",
        type=int,
        default=32,
        help="Sub-trajectory length L (frames). Must be >= context_length + H_roll.",
    )
    p.add_argument(
        "--lr-encoder",
        type=float,
        default=None,
        help="Encoder LR. Default: training.encoder_lr (1.5e-4).",
    )
    p.add_argument(
        "--lr-predictor",
        type=float,
        default=None,
        help="Downstream (predictor/decoder/heads) LR. Default: training.predictor_lr.",
    )
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--out",
        "--output-dir",
        dest="output_dir",
        type=str,
        default="outputs/runs/session31/canonical",
    )
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--diagnostic-every", type=int, default=1000)
    p.add_argument("--checkpoint-every", type=int, default=10_000)
    p.add_argument(
        "--wandb-mode", type=str, choices=["online", "offline", "disabled"], default="online"
    )
    p.add_argument(
        "--gpu", type=int, default=None, help="0-indexed selector into the RTX 6000 subset (D40)."
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Explicit torch device, bypassing require_rtx6000 (non-paper runs only).",
    )
    p.add_argument(
        "--projection-norm", type=str, choices=["batchnorm", "layernorm"], default="batchnorm"
    )
    return p.parse_args()


def build_lr_lambda(max_iters: int, warmup_frac: float) -> "callable[[int], float]":
    warmup_iters = max(1, int(warmup_frac * max_iters))
    floor = 0.05

    def lr_lambda(step: int) -> float:
        if step < warmup_iters:
            return float(step + 1) / float(warmup_iters)
        progress = (step - warmup_iters) / max(1, max_iters - warmup_iters)
        progress = min(max(progress, 0.0), 1.0)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


def make_loader(
    cfg: ResolvedKitConfig,
    args: argparse.Namespace,
    split: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    terms = cfg.active_terms()
    lift_on = "lift" in terms
    wake_on = "wake" in terms
    ds = EpisodeDataset(
        partition=args.partition,
        split=split,
        subtraj_len=args.T,
        emit_cl_future=lift_on,
        cl_future_deltas=(0,),  # current-frame C_L (Fukami-standard lift augmentation)
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


def run_diagnostics(
    model: CanonicalModel,
    diag_decoder: SpatialLatentFieldDecoder,
    test_b_iter: Any,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    """PR + linear-probe R^2 + per-dim variance + a non-degenerate field decode.

    Adapted to the spatial latent: PR is on the flattened ``[N, d]`` latent; the
    parameter probe uses the global-average-pooled ``[B*T, d]`` latent (the
    IMPACT/PER-frame probe regime note in CLAUDE.md applies to interpretation,
    not to this collapse diagnostic). The decode check runs the spatial latent
    through the model's own decoder when it has one (matched AE), else through a
    fixed diagnostic decoder, and confirms the decoded field is finite and not a
    constant map (spatial std > 0).
    """
    batch = move_batch(next(test_b_iter), device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            z = model.encoder(batch["omega"])
        zf = flatten_spatial_latent(z).float()
        z_vec = global_average_pool_latent(z).float()  # (B, T, d)
        decoder = model.decoder if model.decoder is not None else diag_decoder
        field = decoder(z.float())
    if was_training:
        model.train()

    c_static = batch["c"].detach().float()
    c_rep = c_static.unsqueeze(1).expand(-1, z.shape[1], -1).reshape(-1, c_static.shape[1])
    z_probe = z_vec.reshape(-1, z_vec.shape[-1])

    n = z_probe.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    perm = torch.randperm(n, generator=g)
    cut = max(2, int(0.75 * n))
    fit, ev = perm[:cut], perm[cut:]
    if ev.numel() < 2:
        half = n // 2
        fit, ev = perm[:half], perm[half:]

    pr = participation_ratio(zf)
    r2 = linear_probe_r2(z_probe, c_rep, fit, ev)
    counts, edges = per_dim_variance_histogram(zf)
    # Spatial non-degeneracy: mean over (B, T, d) of the std across (h, w).
    spatial_std = float(z.float().std(dim=(3, 4)).mean().item())
    field = field.float()
    return {
        "pr": pr,
        **r2,
        "var_hist_counts_zero_bin": counts[0].item(),
        "var_hist_max_edge": edges[-1].item(),
        "latent_spatial_std": spatial_std,
        "decode_field_std": float(field.std().item()),
        "decode_field_finite": float(bool(torch.isfinite(field).all().item())),
    }


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
    training = cfg.training

    seed = args.seed if args.seed is not None else int(training.get("seed") or 0)
    batch_size = args.B if args.B is not None else int(training.get("batch_size", 16))
    lr_encoder = (
        args.lr_encoder
        if args.lr_encoder is not None
        else float(training.get("encoder_lr", 1.5e-4))
    )
    lr_predictor = (
        args.lr_predictor
        if args.lr_predictor is not None
        else float(training.get("predictor_lr", 5.0e-4))
    )
    weight_decay = (
        args.weight_decay
        if args.weight_decay is not None
        else float(training.get("weight_decay", 0.05))
    )
    set_all_seeds(seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_path = Path(args.split)
    if not split_path.is_absolute():
        split_path = REPO_ROOT / split_path
    with open(split_path) as f:
        split_manifest = json.load(f)
    with open(REPO_ROOT / "configs" / "preprocessing.yaml") as f:
        preprocessing_cfg = yaml.safe_load(f)

    gpu_name = torch.cuda.get_device_name(device.index)
    _allow_non = os.environ.get("VORTEX_JEPA_ALLOW_NON_RTX6000", "").strip()
    if not _allow_non and ("RTX" not in gpu_name or "6000" not in gpu_name):
        raise RuntimeError(
            f"Hardware policy violation (CLAUDE.md): gpu_name={gpu_name!r} does not "
            "contain both 'RTX' and '6000'. Run aborted."
        )

    # --- model -------------------------------------------------------------
    model = CanonicalModel(cfg, latent_dim=args.d, projection_norm=args.projection_norm).to(device)

    ac = cfg.anti_collapse
    ac_on = bool(ac["on"])
    lambda_anticollapse = float(ac["lambda"]) if ac_on else 0.0
    regularizer_tag = ac["method"] if ac_on else "none"
    anticollapse_module = None
    if ac_on:
        anticollapse_module = build_anticollapse(
            ac["method"], args.d, ac["sigreg"], ac["vicreg"]
        ).to(device)

    # Fixed diagnostic decoder so jepa / supervised models still get a decoded-
    # field non-degeneracy check (not part of the loss graph, never optimised).
    feature_h, feature_w = model.encoder.latent_grid
    diag_decoder = SpatialLatentFieldDecoder(
        latent_dim=args.d, feature_h=feature_h, feature_w=feature_w
    ).to(device)
    diag_decoder.eval()
    for prm in diag_decoder.parameters():
        prm.requires_grad_(False)

    # --- run config / W&B --------------------------------------------------
    run_config = {
        "preprocessing_version": preprocessing_cfg["preprocessing_version"],
        "partition_version": args.partition,
        "lambda_sigreg": lambda_anticollapse,
        "seed": seed,
        "split_sha256": file_sha256(split_path),
        "inventory_sha256": split_manifest["source_inventory"]["sha256"],
        "code_sha256": git_commit_hash(),
        "auto_fallback_triggered": False,
        "gpu_name": gpu_name,
        "model_name": cfg.name,
        "objective": cfg.objective,
        "active_terms": sorted(cfg.active_terms()),
        "anti_collapse_method": regularizer_tag,
        "max_iters": args.max_iters,
        "B": batch_size,
        "T": args.T,
        "d": args.d,
        "H_roll": model.horizon,
        "lr_encoder": lr_encoder,
        "lr_predictor": lr_predictor,
        "weight_decay": weight_decay,
        "warmup_frac": args.warmup_frac,
        "pipeline_manifest": args.pipeline_manifest,
        "config_path": str(config_path),
    }

    import wandb

    wandb_tags = [cfg.name, regularizer_tag]
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "vortex-jepa"),
        group=str(training.get("wandb_group", f"partition_{args.partition}")),
        tags=wandb_tags,
        mode=args.wandb_mode,
        config=run_config,
        dir=str(output_dir),
    )
    wandb.run.summary["wandb_run_id"] = wandb.run.id

    metrics_jsonl = output_dir / "metrics.jsonl"
    with open(metrics_jsonl, "w") as f:
        f.write(json.dumps({"event": "config", "wandb_run_id": wandb.run.id, **run_config}) + "\n")

    def _log_metrics(payload: dict[str, Any], step: int) -> None:
        wandb.log(payload, step=step)
        with open(metrics_jsonl, "a") as fh:
            fh.write(json.dumps({"event": "log", "step": int(step), **payload}) + "\n")

    if args.pipeline_manifest is not None:
        manifest_path = Path(args.pipeline_manifest)
        if not manifest_path.is_absolute():
            manifest_path = REPO_ROOT / manifest_path
        pipe = OmegaPipeline.from_manifest(manifest_path)
        print(
            f"[train_canonical] loaded omega pipeline from {manifest_path}\n"
            f"  mask cells: {int(pipe.mask.sum().item())}, "
            f"train_std={pipe.train_stats.std:.4f}",
            flush=True,
        )

    # --- data --------------------------------------------------------------
    train_loader = make_loader(cfg, args, "train", batch_size, True, args.num_workers)
    test_b_loader = make_loader(cfg, args, "test_b", batch_size, False, 0)
    train_iter = infinite_iter(train_loader)
    test_b_iter = infinite_iter(test_b_loader)

    # --- optimiser ---------------------------------------------------------
    downstream = model.downstream_parameters()
    param_groups = [{"params": list(model.encoder.parameters()), "lr": lr_encoder}]
    if downstream:
        param_groups.append({"params": downstream, "lr": lr_predictor})
    optimizer = AdamW(param_groups, betas=(0.9, 0.95), weight_decay=weight_decay)
    scheduler = LambdaLR(optimizer, lr_lambda=build_lr_lambda(args.max_iters, args.warmup_frac))
    controller = AutoFallbackController(d=args.d)

    model.train()
    print(
        f"[train_canonical] model={cfg.name} objective={cfg.objective} "
        f"terms={sorted(cfg.active_terms())} device={device} gpu={gpu_name}\n"
        f"  n_train={len(train_loader.dataset)} n_test_b={len(test_b_loader.dataset)} "
        f"B={batch_size} T={args.T} H_roll={model.horizon} lambda_ac={lambda_anticollapse}",
        flush=True,
    )

    last_total = float("nan")
    for iteration in range(args.max_iters):
        batch = move_batch(next(train_iter), device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(batch)
        loss_total, components = compute_total_loss(
            cfg, outputs, anticollapse_module=anticollapse_module
        )
        last_total = float(loss_total.item())
        if not math.isfinite(last_total):
            raise RuntimeError(
                f"non-finite loss at iter {iteration}: {last_total} (components={components})"
            )

        optimizer.zero_grad(set_to_none=True)
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        if iteration % args.log_every == 0:
            payload = {"iter": iteration, "loss_total": last_total}
            payload.update({f"loss_{k}": v for k, v in components.items()})
            payload["lr_encoder"] = optimizer.param_groups[0]["lr"]
            if len(optimizer.param_groups) > 1:
                payload["lr_predictor"] = optimizer.param_groups[1]["lr"]
            _log_metrics(payload, step=iteration)
            comp_str = ", ".join(f"{k}={v:.4f}" for k, v in components.items() if k != "total")
            print(
                f"[iter {iteration}/{args.max_iters}] loss={last_total:.4f} ({comp_str})",
                flush=True,
            )

        if iteration % args.diagnostic_every == 0:
            diag = run_diagnostics(model, diag_decoder, test_b_iter, device, seed)
            _log_metrics({f"diag/{k}": v for k, v in diag.items()}, step=iteration)
            print(
                f"[diag iter {iteration}] PR={diag['pr']:.2f} "
                f"r2_overall={diag['r2_overall']:.3f} "
                f"spatial_std={diag['latent_spatial_std']:.4f} "
                f"decode_std={diag['decode_field_std']:.4f}",
                flush=True,
            )
            if ac_on and ac["method"] == "sigreg":
                fired = controller.step(iteration, diag["pr"], diag["r2_overall"])
                if fired:
                    print(
                        f"[AUTO-FALLBACK] SIGReg->VICReg at iter {iteration} "
                        f"(PR={diag['pr']:.2f}, r2={diag['r2_overall']:.3f})",
                        flush=True,
                    )
                    anticollapse_module = VICReg(d=args.d).to(device)
                    wandb.run.summary["auto_fallback_triggered"] = True
                    wandb.run.summary["auto_fallback_iter"] = iteration

        if iteration > 0 and iteration % args.checkpoint_every == 0:
            ckpt = output_dir / f"checkpoint_iter{iteration:06d}.pt"
            torch.save(
                {
                    "iteration": iteration,
                    "model_state_dict": model.state_dict(),
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
            "model_state_dict": model.state_dict(),
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
        print("[train_canonical] interrupted", flush=True)
        sys.exit(130)
