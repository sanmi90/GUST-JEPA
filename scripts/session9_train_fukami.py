"""Session 9 Step 3 Ablation A11: train the Fukami lift-augmented autoencoder.

Reference: Fukami and Taira, "Grasping extreme aerodynamics on a low-
dimensional manifold," J. Fluid Mech. 1018, A22 (2023);
arXiv:2305.18394. The supplementary material's Table S.1 specifies the
architecture; the present script implements the adapted version for
our (192, 96) mid-plane vorticity input at matched d=32 latent (vs
Fukami's d=3 at (240, 120)).

Training objective: L = lambda_recon * MSE(omega, omega_hat) +
lambda_lift * MSE(CL_future, CL_future_hat). Defaults lambda_recon =
lambda_lift = 1.0.

Final evaluation against the SIGReg + OBS JEPA: per-encounter MSE and
SSIM (Wang et al. 2004; Fukami's primary reconstruction metric) on
Test A, Test B, and Test C, plus the downstream r2(CL_future) on
Test B for the head-to-head with the JEPA's Section 5.5 production
Test B delta.

Usage:
    python scripts/session9_train_fukami.py \\
        --gpu 1 \\
        --partition v1 --all-train --max-iters 20000 --seed 0 \\
        --latent-dim 32 \\
        --observable-head cl_future --observable-head-deltas 8 16 24 \\
        --output-dir outputs/runs/session9/run_a11_fukami_ae \\
        --wandb-mode offline

Hardware: RTX 6000 Blackwell only (require_rtx6000 at entry).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.baselines.fukami_ae import FukamiAEWrapper  # noqa: E402
from src.training.train_jepa import (  # noqa: E402
    REPO_ROOT,
    file_sha256,
    git_commit_hash,
    infinite_iter,
    jepa_collate,
    make_test_b_loader,
    make_train_loader,
    move_batch,
    resolve_cases,
    set_all_seeds,
)


def fukami_collate(samples):
    """Like jepa_collate, but also carry case_ids + encounter_indices.

    Needed by FukamiAEWrapper when an OmegaPipeline is attached, since the
    wrapper looks up per-encounter clip thresholds at forward time.
    """
    batch = jepa_collate(samples)
    batch["case_ids"] = [s["case_id"] for s in samples]
    batch["encounter_indices"] = torch.tensor(
        [int(s["encounter_index"]) for s in samples], dtype=torch.long,
    )
    return batch


def move_batch_mixed(batch, device):
    """Like move_batch but tolerates non-tensor values (e.g., case_ids list)."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out
from src.utils.device import require_rtx6000  # noqa: E402


PREVENT = Path(os.environ.get("PREVENT_ROOT", "/home/carlos/PREVENT"))
CACHE = Path(os.environ.get("VORTEX_JEPA_CACHE", PREVENT / "data" / "processed" / "vortex-jepa"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Session 9 Ablation A11: Fukami AE")
    p.add_argument("--partition", type=str, default="v1")
    p.add_argument("--cases", nargs="+", default=None)
    p.add_argument("--cases-from", type=str, default=None)
    p.add_argument("--all-train", action="store_true")
    p.add_argument("--max-iters", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--d", "--latent-dim", dest="d", type=int, default=32)
    p.add_argument("--B", type=int, default=16)
    p.add_argument("--T", type=int, default=32)
    p.add_argument("--observable-head", type=str, choices=["none", "cl_future"],
                   default="cl_future")
    p.add_argument("--observable-head-weight", type=float, default=1.0,
                   help="Fukami's lambda_lift (default 1.0 per the paper).")
    p.add_argument("--observable-head-deltas", type=int, nargs="+", default=[8, 16, 24])
    p.add_argument("--lambda-recon", type=float, default=1.0)
    p.add_argument("--lambda-lift", type=float, default=1.0)
    p.add_argument("--omega-scale", type=float, default=1000.0,
                   help="Divide omega by this before CNN encoder; matches Fukami's "
                        "normalized [-0.6, 0.6] input range when set to ~1000.")
    p.add_argument("--omega-clip", type=float, default=None,
                   help="If set, clip |omega| to this value before encoder. "
                        "Default None (no clipping).")
    p.add_argument("--omega-clip-pct", type=float, default=None,
                   help="If set, per-sample percentile clipping: clip |omega| "
                        "above its own p_X percentile. Data-driven artifact "
                        "suppression. Suggested 99.99 (clips top 0.01%% which "
                        "removes the leading-edge artifact spikes while keeping "
                        "all dense physical features).")
    p.add_argument("--airfoil-mask-path", type=str, default=None,
                   help="If set, load a (192, 96) boolean mask from this .npy "
                        "file and zero omega at masked cells before the encoder. "
                        "Use scripts/compute_omega_clip_thresholds.py to build "
                        "outputs/runs/session9/airfoil_adjacent_mask.npy.")
    p.add_argument("--omega-pipeline-manifest", type=str, default=None,
                   help="If set, load OmegaPipeline (mask + per-encounter clip + "
                        "train-stats z-score) from this JSON manifest. Overrides "
                        "omega-scale / omega-clip / omega-clip-pct / "
                        "airfoil-mask-path. Build the manifest with "
                        "scripts/build_omega_pipeline.py.")
    p.add_argument("--lr", type=float, default=1.0e-3,
                   help="Fukami used Adam with lr around 1e-3; we keep that.")
    p.add_argument("--weight-decay", type=float, default=0.0,
                   help="Fukami used no weight decay; defaults to 0.")
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--diagnostic-every", type=int, default=500)
    p.add_argument("--checkpoint-every", type=int, default=2000)
    p.add_argument("--wandb-mode", type=str, choices=["online", "offline", "disabled"],
                   default="offline")
    p.add_argument("--tag-suffix", type=str, default="")
    p.add_argument("--gpu", type=int, default=None)
    return p.parse_args()


def build_lr_lambda(max_iters: int, warmup_frac: float):
    warmup = max(1, int(warmup_frac * max_iters))
    floor = 0.05

    def fn(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(warmup)
        prog = min(1.0, (step - warmup) / max(1, max_iters - warmup))
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * prog))
    return fn


def _ssim(x: np.ndarray, y: np.ndarray, c1: float = 0.16, c2: float = 1.44) -> float:
    """Fukami's SSIM definition with C1, C2 from arXiv:2305.18394 Eq. (1).

    Operates on a single (H, W) array pair. Mean, std, and covariance
    are global, not windowed; this matches the supplementary text's
    definition rather than the classical Wang et al. windowed SSIM.
    """
    mu_x, mu_y = x.mean(), y.mean()
    var_x, var_y = x.var(), y.var()
    cov_xy = ((x - mu_x) * (y - mu_y)).mean()
    num = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
    return float(num / max(den, 1e-12))


def gather_eval_encounters(split: str) -> list[dict]:
    with open(REPO / "configs" / "splits" / "split_v1.json") as f:
        manifest = json.load(f)
    out = []
    for cid, case in manifest["cases"].items():
        if split == "test_a" and case["split"] == "train":
            ks = case["test_a_encounter_indices"]
        elif split == "test_b" and case["split"] == "test_b":
            ks = list(range(case["n_encounters_full"]))
        elif split == "test_c" and case["split"] == "test_c":
            ks = list(range(case["n_encounters_full"]))
        else:
            continue
        for k in ks:
            path = CACHE / "v1" / cid / f"encounter_{k:02d}.h5"
            if not path.exists():
                continue
            out.append({"case_id": cid, "k": int(k), "path": str(path)})
    return out


def _l2_relative_error(q: np.ndarray, q_hat: np.ndarray, eps: float = 1.0) -> float:
    """Fukami's L_2 relative reconstruction error.

    eps = || q - q_hat ||_2 / max(|| q ||_2, eps)

    Where || . ||_2 is the L_2 norm. eps floor of 1.0 (raw vorticity units)
    prevents the metric from exploding on near-zero baseline frames where
    || q ||_2 is essentially noise-level. Used in Fukami arXiv:2305.18394 /
    J. Fluid Mech. 1018, A22 (2023) Figures 15-18 to report per-snapshot
    reconstruction quality; we report it at both per-frame and per-volume
    granularity.
    """
    num = float(np.sqrt(((q - q_hat) ** 2).sum()))
    den = float(np.sqrt((q ** 2).sum()))
    return num / max(den, eps)


def evaluate_split(
    wrapper: FukamiAEWrapper,
    encs: list[dict],
    device: torch.device,
) -> dict:
    """Per-encounter MSE + Fukami L2-relative-error + SSIM + case-mean floor.

    When the wrapper has an attached OmegaPipeline, the evaluation applies
    Stages 1 + 2 (mask + per-encounter clip) to the raw omega BEFORE
    encoding, and computes all metrics on that masked-clipped raw scale
    (matching the training loss). The encoder receives the normalized
    omega via pipeline.normalize and the decoder output is un-normalized
    via pipeline.unnormalize, both internal to the wrapper.forward path
    which we re-use here.
    """
    pipe = wrapper.omega_pipeline
    case_to_arr: dict[str, list[np.ndarray]] = {}
    for e in encs:
        with h5py.File(e["path"], "r") as f:
            omega = np.asarray(f["omega_z"], dtype=np.float32)
        # Pre-apply the pipeline's spatial mask + per-encounter clip to the
        # case-mean inputs as well; the floor should compare against the
        # cleaned omega, not the artifact-laden raw.
        if pipe is not None:
            omega = pipe.preprocess_raw(omega, e["case_id"], int(e["k"]))
        case_to_arr.setdefault(e["case_id"], []).append(omega)
    case_mean = {cid: np.stack(arrs).mean(axis=0) for cid, arrs in case_to_arr.items()}

    mses, floors, ssims, eps_frames_mean, eps_volume = [], [], [], [], []
    wrapper.eval()
    with torch.no_grad():
        for e in encs:
            with h5py.File(e["path"], "r") as f:
                omega = np.asarray(f["omega_z"], dtype=np.float32)
            if pipe is not None:
                omega = pipe.preprocess_raw(omega, e["case_id"], int(e["k"]))
            x = torch.from_numpy(omega).unsqueeze(0).unsqueeze(2).to(device)  # (1, T, 1, H, W)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                if pipe is not None:
                    x_norm = pipe.normalize(x)
                    z = wrapper.encoder(x_norm)
                    x_hat_norm = wrapper.decoder(z)
                    x_hat = pipe.unnormalize(x_hat_norm)
                else:
                    z = wrapper.encode(x)
                    x_hat = wrapper.decode(z)
            x_hat = x_hat.float().squeeze(0).squeeze(1).cpu().numpy()  # (T, H, W)
            mse = float(((omega - x_hat) ** 2).mean())
            floor = float(((omega - case_mean[e["case_id"]]) ** 2).mean())
            ssim_frames = [_ssim(omega[t], x_hat[t]) for t in range(omega.shape[0])]
            eps_per_frame = [_l2_relative_error(omega[t], x_hat[t])
                             for t in range(omega.shape[0])]
            eps_frames_mean.append(float(np.mean(eps_per_frame)))
            eps_volume.append(_l2_relative_error(omega, x_hat))
            mses.append(mse)
            floors.append(floor)
            ssims.append(float(np.mean(ssim_frames)))
    return {
        "mse_mean": float(np.mean(mses)),
        "mse_median": float(np.median(mses)),
        "floor_mean": float(np.mean(floors)),
        "ratio_mean": float(np.mean(mses) / max(np.mean(floors), 1e-12)),
        "ssim_mean": float(np.mean(ssims)),
        "ssim_median": float(np.median(ssims)),
        "eps_per_frame_mean": float(np.mean(eps_frames_mean)),
        "eps_per_frame_median": float(np.median(eps_frames_mean)),
        "eps_volume_mean": float(np.mean(eps_volume)),
        "eps_volume_median": float(np.median(eps_volume)),
        "n_encounters": len(encs),
    }


def main() -> None:
    args = parse_args()
    args.cases = resolve_cases(args)
    device = require_rtx6000(gpu_index=args.gpu)
    set_all_seeds(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    metrics_path = out_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    def log(msg: str) -> None:
        line = msg if msg.endswith("\n") else msg + "\n"
        with open(log_path, "a") as f:
            f.write(line)
        print(msg, flush=True)

    gpu_name = torch.cuda.get_device_name(device.index)
    if "RTX" not in gpu_name or "6000" not in gpu_name:
        raise RuntimeError(
            f"Hardware policy violation: gpu_name={gpu_name!r} does not "
            "contain both 'RTX' and '6000'."
        )
    log(f"[fukami-train] device={device} gpu={gpu_name}")

    split_path = REPO_ROOT / "configs" / "splits" / f"split_{args.partition}.json"
    with open(split_path) as f:
        split_manifest = json.load(f)
    with open(REPO_ROOT / "configs" / "preprocessing.yaml") as f:
        preprocessing_cfg = yaml.safe_load(f)

    run_config = {
        "preprocessing_version": preprocessing_cfg["preprocessing_version"],
        "partition_version": args.partition,
        "baseline": "fukami_ae",
        "lambda_sigreg": None,
        "lambda_recon": args.lambda_recon,
        "lambda_lift": args.lambda_lift,
        "seed": args.seed,
        "split_sha256": file_sha256(split_path),
        "inventory_sha256": split_manifest["source_inventory"]["sha256"],
        "code_sha256": git_commit_hash(),
        "auto_fallback_triggered": False,
        "gpu_name": gpu_name,
        "max_iters": args.max_iters,
        "B": args.B, "T": args.T, "d": args.d,
        "lr": args.lr,
        "weight_decay": args.weight_decay, "warmup_frac": args.warmup_frac,
        "cases": args.cases, "all_train": bool(getattr(args, "all_train", False)),
        "tag_suffix": args.tag_suffix,
        "observable_head": args.observable_head,
        "observable_head_weight": args.observable_head_weight,
        "observable_head_deltas": list(args.observable_head_deltas),
    }

    import wandb
    wandb_tags = ["fukami_ae", "section7_a11"]
    if args.tag_suffix:
        wandb_tags.append(f"run:{args.tag_suffix}")
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "vortex-jepa"),
        group=f"partition_{args.partition}",
        tags=wandb_tags, mode=args.wandb_mode, config=run_config, dir=str(out_dir),
    )
    wandb.run.summary["wandb_run_id"] = wandb.run.id
    with open(metrics_path, "w") as f:
        f.write(json.dumps({"event": "config", "wandb_run_id": wandb.run.id, **run_config}) + "\n")

    def _log_metrics(payload: dict[str, Any], step: int) -> None:
        wandb.log(payload, step=step)
        with open(metrics_path, "a") as fh:
            fh.write(json.dumps({"event": "log", "step": int(step), **payload}) + "\n")

    # Load the omega pipeline (if requested) BEFORE constructing data loaders,
    # so we can set num_workers=0 to avoid persistent-worker fork issues
    # caused by swapping the collate_fn.
    airfoil_mask = None
    if args.airfoil_mask_path is not None:
        mask_np = np.load(args.airfoil_mask_path)
        airfoil_mask = torch.from_numpy(mask_np).bool()
        log(f"[fukami-train] loaded airfoil mask from {args.airfoil_mask_path} "
            f"({mask_np.sum()} cells masked of {mask_np.size})")

    omega_pipeline = None
    if args.omega_pipeline_manifest is not None:
        from src.data.omega_pipeline import OmegaPipeline
        omega_pipeline = OmegaPipeline.from_manifest(args.omega_pipeline_manifest)
        log(f"[fukami-train] loaded omega pipeline from {args.omega_pipeline_manifest}")
        log(f"  mask: {int(omega_pipeline.mask.sum().item())} cells")
        log(f"  thresholds: {sum(len(v) for v in omega_pipeline.thresholds.values())} encounters")
        log(f"  train_stats: mean={omega_pipeline.train_stats.mean:.4f}, "
            f"std={omega_pipeline.train_stats.std:.4f}")

    if omega_pipeline is not None:
        # Force num_workers=0 to avoid persistent-worker fork issues when we
        # swap the collate_fn after loader construction.
        args.num_workers = 0
    train_loader = make_train_loader(args)
    test_b_loader = make_test_b_loader(args)
    # Swap to the Fukami collate (carries case_ids + encounter_indices) when
    # an OmegaPipeline is attached so the wrapper can look up per-encounter
    # clip thresholds.
    if omega_pipeline is not None:
        train_loader.collate_fn = fukami_collate
        test_b_loader.collate_fn = fukami_collate
    train_iter = infinite_iter(train_loader)
    test_b_iter = infinite_iter(test_b_loader)

    wrapper = FukamiAEWrapper(
        latent_dim=args.d, n_deltas=len(args.observable_head_deltas),
        lambda_recon=args.lambda_recon, lambda_lift=args.lambda_lift,
        omega_scale=args.omega_scale,
        omega_clip=args.omega_clip,
        omega_clip_pct=args.omega_clip_pct,
        airfoil_mask=airfoil_mask,
        omega_pipeline=omega_pipeline,
    ).to(device)
    n_params = sum(p.numel() for p in wrapper.parameters())
    log(f"[fukami-train] params={n_params:,} ({n_params/1e6:.2f}M)")

    optimizer = AdamW(wrapper.parameters(), lr=args.lr, betas=(0.9, 0.95),
                      weight_decay=args.weight_decay)
    scheduler = LambdaLR(optimizer, lr_lambda=build_lr_lambda(args.max_iters, args.warmup_frac))

    wrapper.train()
    log(f"[fukami-train] n_train_samples={len(train_loader.dataset)} "
        f"n_test_b_samples={len(test_b_loader.dataset)}")

    for iteration in range(args.max_iters):
        batch = move_batch_mixed(next(train_iter), device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = wrapper(batch)
        L_total = out["L_total"]
        if not math.isfinite(L_total.item()):
            raise RuntimeError(
                f"non-finite loss at iter {iteration}: L_total={L_total.item()} "
                f"(recon={out['L_recon'].item()}, lift={out['L_lift'].item()})"
            )

        optimizer.zero_grad(set_to_none=True)
        L_total.backward()
        nn.utils.clip_grad_norm_(wrapper.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        if iteration % args.log_every == 0:
            _log_metrics({
                "iter": iteration,
                "loss_total": float(L_total.item()),
                "L_recon": float(out["L_recon"].item()),
                "L_lift": float(out["L_lift"].item()),
                "lr": optimizer.param_groups[0]["lr"],
            }, step=iteration)
            log(f"[iter {iteration}/{args.max_iters}] L={L_total.item():.4f} "
                f"(recon={out['L_recon'].item():.4f}, lift={out['L_lift'].item():.4f})")

        if iteration % args.diagnostic_every == 0 and iteration > 0:
            wrapper.eval()
            with torch.no_grad():
                tb = move_batch_mixed(next(test_b_iter), device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    tb_out = wrapper(tb)
                z_b = tb_out["z"].float()
                pr = float((z_b.var(dim=(0, 1)).sum() ** 2) /
                           (z_b.var(dim=(0, 1)) ** 2).sum().clamp_min(1e-8))
            log(f"[diag iter {iteration}] PR={pr:.2f} L_recon_test_b="
                f"{tb_out['L_recon'].item():.4f}")
            _log_metrics({"diag/pr_test_b": pr,
                          "diag/L_recon_test_b": float(tb_out["L_recon"].item())},
                         step=iteration)
            wrapper.train()

        if iteration > 0 and iteration % args.checkpoint_every == 0:
            ckpt = out_dir / f"checkpoint_iter{iteration:06d}.pt"
            torch.save({
                "wrapper_state_dict": wrapper.state_dict(),
                "args": vars(args),
                "iteration": iteration,
                "run_config": run_config,
            }, ckpt)
            log(f"[checkpoint] wrote {ckpt}")

    final_ckpt = out_dir / f"checkpoint_iter{args.max_iters:06d}.pt"
    torch.save({
        "wrapper_state_dict": wrapper.state_dict(),
        "args": vars(args),
        "iteration": args.max_iters,
        "run_config": run_config,
    }, final_ckpt)
    log(f"[fukami-train] final checkpoint at {final_ckpt}")

    # Final evaluation with MSE + SSIM
    log("[fukami-train] evaluating Test A / B / C with MSE + SSIM")
    encs_a = gather_eval_encounters("test_a")
    encs_b = gather_eval_encounters("test_b")
    encs_c = gather_eval_encounters("test_c")
    ev_a = evaluate_split(wrapper, encs_a, device)
    ev_b = evaluate_split(wrapper, encs_b, device)
    ev_c = evaluate_split(wrapper, encs_c, device)
    summary = {
        "test_a": ev_a, "test_b": ev_b, "test_c": ev_c,
        "ratio_a_within_2x": ev_a["ratio_mean"] < 2.0,
    }
    with open(out_dir / "final_eval.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"[fukami-train] FINAL eval written to {out_dir / 'final_eval.json'}")
    for split, ev in (("test_a", ev_a), ("test_b", ev_b), ("test_c", ev_c)):
        log(f"  {split}: MSE_mean={ev['mse_mean']:.4f} "
            f"floor={ev['floor_mean']:.4f} ratio={ev['ratio_mean']:.3f} "
            f"SSIM_mean={ev['ssim_mean']:.4f} "
            f"eps_per_frame_mean={ev['eps_per_frame_mean']:.4f} "
            f"eps_volume_mean={ev['eps_volume_mean']:.4f}")


if __name__ == "__main__":
    main()
