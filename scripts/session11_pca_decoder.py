"""Train a LapFiLM decoder on PCA-truncated JEPA latents.

Tests the hypothesis: JEPA's PR=11.66 (effective d) means most of the
useful information lives in ~12 principal components. If we train a
decoder on top-k PCs only (with latent_dim=k, no zero-padding back to
32), does Test B SSIM hold?

Pipeline:
  1. Load the JEPA encoder (frozen) and OmegaPipeline.
  2. Encode all train encounters once: z_train shape (N, 32).
  3. Center + SVD → top-k orthonormal columns P ∈ R^{32×k} and mean ∈ R^32.
  4. Build LapFiLMDecoder with latent_dim=k (replaces the 32-d head).
  5. Train decoder with the standard E1 recipe; before each forward,
     project z to k dims with (z - mean) @ P.
  6. Final eval reports Test A/B/C SSIM + Session 10 metric bundle.

Usage::

    python scripts/session11_pca_decoder.py \\
        --encoder-run outputs/runs/session11/W0_C_lam100 \\
        --k 12 \\
        --output-dir outputs/runs/session11/W0_C_lam100/decoder_pca_k12 \\
        --gpu 1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.omega_pipeline import OmegaPipeline  # noqa: E402
from src.evaluation.decoder_metrics import (  # noqa: E402
    aggregate_split_metrics,
    compute_encounter_metrics,
)
from src.models.encoder import HybridCNNViTEncoder  # noqa: E402
from src.models.lap_film_decoder import LapFiLMDecoder  # noqa: E402
from src.models.decoder_losses import region_pyr_ffl_loss  # noqa: E402
from src.utils.device import require_rtx6000  # noqa: E402


PREVENT = Path(os.environ.get("PREVENT_ROOT", "/home/carlos/PREVENT"))
CACHE = Path(os.environ.get("VORTEX_JEPA_CACHE", PREVENT / "data" / "processed" / "vortex-jepa"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PCA-truncated decoder retrain")
    p.add_argument("--encoder-run", required=True, type=str)
    p.add_argument("--k", required=True, type=int)
    p.add_argument("--output-dir", required=True, type=str)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--B", type=int, default=16)
    p.add_argument("--T", type=int, default=32)
    p.add_argument("--max-iters", type=int, default=20000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--checkpoint-every", type=int, default=2000)
    p.add_argument(
        "--omega-pipeline-manifest", type=str,
        default="outputs/data_pipeline/v1/manifest.json",
    )
    p.add_argument("--decoder-base-ch", type=int, default=64)
    p.add_argument("--lambda-region", type=float, default=1.0)
    p.add_argument("--lambda-pyramid", type=float, default=0.4)
    p.add_argument("--lambda-ffl", type=float, default=0.0)
    p.add_argument("--ffl-warmup-iters", type=int, default=2000)
    p.add_argument("--ffl-ramp-iters", type=int, default=1000)
    p.add_argument("--ffl-alpha", type=float, default=1.0)
    p.add_argument("--ffl-patch", type=int, default=32)
    p.add_argument("--lambda-enstrophy", type=float, default=0.02)
    p.add_argument("--lambda-circulation", type=float, default=0.01)
    p.add_argument("--active-tau", type=float, default=0.10)
    p.add_argument("--active-softness", type=float, default=0.03)
    p.add_argument("--inactive-weight", type=float, default=0.05)
    p.add_argument("--wake-weight", type=float, default=0.5)
    return p.parse_args()


def gather_encs(split: str) -> list[dict]:
    with open(REPO / "configs" / "splits" / "split_v1.json") as f:
        m = json.load(f)
    out = []
    for cid, case in m["cases"].items():
        if split == "train" and case["split"] == "train":
            ks = case["train_encounter_indices"]
        elif split == "test_a" and case["split"] == "train":
            ks = case["test_a_encounter_indices"]
        elif split in ("test_b", "test_c") and case["split"] == split:
            ks = list(range(case["n_encounters_full"]))
        else:
            continue
        for k in ks:
            path = CACHE / "v1" / cid / f"encounter_{k:02d}.h5"
            if path.exists():
                out.append({"case_id": cid, "k": int(k), "path": str(path)})
    return out


def load_encoder(encoder_run: Path, device: torch.device) -> tuple[HybridCNNViTEncoder, int]:
    cands = sorted(encoder_run.glob("checkpoint_iter*.pt"))
    if not cands:
        raise FileNotFoundError(f"No checkpoint under {encoder_run}")
    ckpt_path = cands[-1]
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = blob["args"]
    enc = HybridCNNViTEncoder(
        latent_dim=int(args["d"]),
        projection_norm=args.get("projection_norm", "batchnorm"),
    )
    state = {
        k.removeprefix("encoder."): v
        for k, v in blob["jepa_state_dict"].items()
        if k.startswith("encoder.")
    }
    enc.load_state_dict(state, strict=False)
    enc.eval().to(device)
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc, int(args["d"])


def compute_pca_basis(
    enc: HybridCNNViTEncoder,
    pipe: OmegaPipeline,
    device: torch.device,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Encode all train encounters → SVD → return (mean (d,), P (d, k), singular_values)."""
    encs = gather_encs("train")
    chunks = []
    t0 = time.time()
    for i, e in enumerate(encs):
        with h5py.File(e["path"], "r") as f:
            omega_raw = np.asarray(f["omega_z"], dtype=np.float32)
        omega_clean = pipe.preprocess_raw(omega_raw, e["case_id"], int(e["k"]))
        omega_norm = pipe.normalize(torch.from_numpy(omega_clean)).to(device)
        x = omega_norm.unsqueeze(0).unsqueeze(2)  # (1, T, 1, H, W)
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            z = enc(x).float().squeeze(0)
        chunks.append(z.cpu())
        if (i + 1) % 30 == 0:
            print(f"[pca] encoded {i+1}/{len(encs)} ({time.time()-t0:.1f}s)", flush=True)
    Z = torch.cat(chunks, dim=0)
    print(f"[pca] z shape {tuple(Z.shape)}; total {time.time()-t0:.1f}s", flush=True)
    mean = Z.mean(dim=0)
    Zc = Z - mean
    U, S, Vh = torch.linalg.svd(Zc, full_matrices=False)
    print(f"[pca] singular values (top 5 / bottom 3): "
          f"{S[:5].tolist()} ... {S[-3:].tolist()}", flush=True)
    energy_cum = (S ** 2).cumsum(0) / (S ** 2).sum()
    print(f"[pca] cumulative energy at k={k}: {energy_cum[k-1].item():.4f}", flush=True)
    P = Vh[:k].T  # (d, k); each column an orthonormal basis vector
    return mean, P, S.numpy()


class EncounterFrameDataset(torch.utils.data.Dataset):
    def __init__(self, encs: list[dict], pipe: OmegaPipeline, T: int = 32, seed: int = 0):
        self.encs = encs
        self.pipe = pipe
        self.T = T
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.encs)

    def __getitem__(self, idx: int) -> dict:
        e = self.encs[idx]
        with h5py.File(e["path"], "r") as f:
            omega = np.asarray(f["omega_z"], dtype=np.float32)
        T_full = omega.shape[0]
        start = 0 if T_full <= self.T else int(self.rng.integers(0, T_full - self.T + 1))
        x = omega[start:start + self.T]
        x = self.pipe.preprocess_raw(x, e["case_id"], int(e["k"]))
        x = self.pipe.normalize(torch.from_numpy(x))
        return {"omega": x}


def collate(samples):
    return {"omega": torch.stack([s["omega"] for s in samples], dim=0)}


def build_lr_lambda(max_iters: int, warmup_frac: float):
    warmup = max(1, int(warmup_frac * max_iters))
    def fn(step: int) -> float:
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, max_iters - warmup)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * prog))
    return fn


def ffl_warmup_factor(step: int, warmup_iters: int, ramp_iters: int) -> float:
    if step < warmup_iters:
        return 0.0
    if ramp_iters <= 0:
        return 1.0
    return max(0.0, min(1.0, (step - warmup_iters) / ramp_iters))


def _ssim_fukami(x: np.ndarray, y: np.ndarray) -> float:
    c1, c2 = 0.16, 1.44
    mu_x, mu_y = x.mean(), y.mean()
    var_x, var_y = x.var(), y.var()
    cov = ((x - mu_x) * (y - mu_y)).mean()
    num = (2 * mu_x * mu_y + c1) * (2 * cov + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
    return float(num / max(den, 1e-12))


def main():
    args = parse_args()
    device = require_rtx6000(gpu_index=args.gpu)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "decoder_train.log"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    log(f"[pca-decoder] device={device} k={args.k}")

    pipe_path = Path(args.omega_pipeline_manifest)
    if not pipe_path.is_absolute():
        pipe_path = REPO / pipe_path
    pipe = OmegaPipeline.from_manifest(pipe_path)
    enc, d = load_encoder(Path(args.encoder_run), device)
    log(f"[pca-decoder] encoder d={d}, encoder_run={args.encoder_run}")

    if args.k > d:
        raise SystemExit(f"--k {args.k} exceeds encoder latent_dim {d}")

    mean, P, S = compute_pca_basis(enc, pipe, device, args.k)
    mean_d = mean.to(device)
    P_d = P.to(device)
    np.savez(out_dir / "pca_basis.npz",
             mean=mean.numpy(), P=P.numpy(), singular_values=S,
             k=args.k, encoder_d=d)
    log(f"[pca-decoder] saved PCA basis: shape mean={mean.shape}, P={P.shape}")

    # Build LapFiLM with latent_dim=k.
    bc = args.decoder_base_ch
    channels = (bc, bc, int(bc * 0.75), int(bc * 0.5), int(bc * 0.375))
    dec = LapFiLMDecoder(
        latent_dim=args.k,
        channels=channels,
        upsample="pixelshuffle",
        fourier_bands=4,
        use_film=True,
    ).to(device)
    log(f"[pca-decoder] decoder params={sum(p.numel() for p in dec.parameters()):,}")

    train_encs = gather_encs("train")
    test_a_encs = gather_encs("test_a")
    test_b_encs = gather_encs("test_b")
    test_c_encs = gather_encs("test_c")
    log(f"[pca-decoder] train={len(train_encs)} test_a={len(test_a_encs)} "
        f"test_b={len(test_b_encs)} test_c={len(test_c_encs)}")

    ds = EncounterFrameDataset(train_encs, pipe, T=args.T, seed=args.seed)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.B, shuffle=True, num_workers=args.num_workers,
        collate_fn=collate, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    it = iter(loader)

    airfoil_mask_np = np.load(REPO / "outputs" / "data_pipeline" / "v1" / "airfoil_adjacent_mask.npy")
    airfoil_mask = torch.from_numpy(airfoil_mask_np.astype(np.float32)).to(device)

    opt = torch.optim.AdamW(dec.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=build_lr_lambda(args.max_iters, args.warmup_frac))

    metrics_path = out_dir / "decoder_metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    region_kwargs = dict(
        active_tau=args.active_tau, active_softness=args.active_softness,
        inactive_weight=args.inactive_weight, wake_weight=args.wake_weight,
    )

    def project(z: torch.Tensor) -> torch.Tensor:
        # z: (B, T, d) → (B, T, k) via z @ P after centering
        return (z - mean_d) @ P_d

    for step in range(args.max_iters + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        x = batch["omega"].to(device).unsqueeze(2)  # (B, T, 1, H, W)
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            z = enc(x).float()
            z_proj = project(z)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            dec_out = dec(z_proj)
            pred = dec_out["pred"].float()
            pyramid = [p.float() for p in dec_out["pyramid"]]
            target = x.float()
        warmup_f = ffl_warmup_factor(step, args.ffl_warmup_iters, args.ffl_ramp_iters)
        loss_out = region_pyr_ffl_loss(
            pred_pyr=pyramid,
            target=target,
            lambda_region=args.lambda_region,
            lambda_pyramid=args.lambda_pyramid,
            lambda_ffl=args.lambda_ffl,
            lambda_enstrophy=args.lambda_enstrophy,
            lambda_circulation=args.lambda_circulation,
            ffl_alpha=args.ffl_alpha, ffl_patch=args.ffl_patch,
            ffl_warmup_factor=warmup_f,
            solid_or_airfoil_mask=airfoil_mask,
            region_kwargs=region_kwargs,
        )
        loss = loss_out["L_total"]
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(dec.parameters(), args.grad_clip)
        opt.step()
        sched.step()
        if step % args.log_every == 0:
            comps = ' '.join(f'{k}={float(v):.4f}' for k, v in loss_out.items())
            log(f"[iter {step}/{args.max_iters}] {comps} ffl_w={warmup_f:.2f} "
                f"lr={sched.get_last_lr()[0]:.2e}")
        if step > 0 and step % args.checkpoint_every == 0:
            ckpt = out_dir / f"decoder_iter{step:06d}.pt"
            torch.save({"decoder_state_dict": dec.state_dict(),
                        "iter": step, "args": vars(args),
                        "encoder_run": str(args.encoder_run),
                        "k": args.k}, ckpt)
            log(f"[checkpoint] saved {ckpt}")

    # Final eval on all three splits with Session 10 metric bundle.
    log("[pca-decoder] final evaluation on test_a/b/c")
    dec.eval()
    summary = {"k": args.k, "encoder_d": d, "encoder_run": str(args.encoder_run)}
    for name, encs in (("test_a", test_a_encs), ("test_b", test_b_encs),
                      ("test_c", test_c_encs)):
        per_enc = []
        for e in encs:
            with h5py.File(e["path"], "r") as f:
                omega_raw = np.asarray(f["omega_z"], dtype=np.float32)
            omega_clean = pipe.preprocess_raw(omega_raw, e["case_id"], int(e["k"]))
            x = pipe.normalize(torch.from_numpy(omega_clean)).unsqueeze(0).unsqueeze(2).to(device)
            with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                z = enc(x).float()
                z_proj = project(z)
                pred_norm = dec(z_proj)["pred"].float().squeeze(0).squeeze(1)
                pred_raw = pipe.unnormalize(pred_norm.unsqueeze(0)).squeeze(0).cpu().numpy()
            per_enc.append(compute_encounter_metrics(omega_clean, pred_raw))
        agg = aggregate_split_metrics(per_enc)
        summary[name] = agg
        log(f"[pca-decoder] {name}: "
            f"SSIM med={agg.get('ssim_mean_median', float('nan')):.4f} "
            f"mean={agg.get('ssim_mean_mean', float('nan')):.4f} "
            f"eps_vol med={agg.get('eps_volume_median', float('nan')):.4f} "
            f"ens_wake med={agg.get('enstrophy_rel_err_wake_median', float('nan')):.4f}")
    with open(out_dir / "decoder_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"[pca-decoder] wrote {out_dir / 'decoder_summary.json'}")


if __name__ == "__main__":
    main()
