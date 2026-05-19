"""Session 9 Step 2: train the visualisation decoder on a frozen JEPA encoder.

The decoder is a separate :class:`HybridViTConvDecoder` that maps
``z in R^d`` (the JEPA encoder's projection-head output) back to a
mid-plane vorticity field ``omega_z`` of shape ``(192, 96)``. It is
NEVER part of the JEPA loss; the encoder weights stay frozen here.

Training loop
-------------
- 10k iterations, AdamW (0.9, 0.95), lr=1e-4, wd=0.05.
- Per-frame MSE on ``omega_z``, summed over (T, H, W).
- bf16 mixed precision on the RTX 6000 Blackwell.
- Cosine LR with 5% linear warmup.
- Batch B = 16 sub-trajectories of T = 32 frames (matches the encoder
  training data layout).
- W&B logging: encoder checkpoint hash, decoder seed, train MSE,
  iter-2000 / 4000 / 6000 / 8000 / 10000 evaluation on Test A.

Pass criterion (Session 9 plan, Section 5.6 of the architecture spec):
the Test A reconstruction MSE must be within 2x the per-case-mean noise
floor (= the MSE of the per-case-mean ``omega_z`` field on Test A).

Usage:
    python -m scripts.session9_train_decoder \\
        --jepa-checkpoint outputs/runs/session9/run_f1_lam0p001_seed0/checkpoint_iter020000.pt \\
        --output-dir outputs/runs/session9/decoder \\
        --gpu 0 --max-iters 10000

Hardware: RTX 6000 Blackwell only (require_rtx6000 at entry).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.models.decoder import HybridViTConvDecoder  # noqa: E402
from src.models.encoder import HybridCNNViTEncoder  # noqa: E402
from src.utils.device import require_rtx6000  # noqa: E402


PREVENT = Path(os.environ.get("PREVENT_ROOT", "/home/carlos/PREVENT"))
CACHE = Path(os.environ.get("VORTEX_JEPA_CACHE", PREVENT / "data" / "processed" / "vortex-jepa"))


def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_encoder(ckpt_path: Path, device: torch.device) -> tuple[HybridCNNViTEncoder, int]:
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
    missing, unexpected = enc.load_state_dict(state, strict=False)
    if unexpected:
        print(f"[decoder-train] WARNING: unexpected encoder keys ignored: {unexpected}",
              flush=True)
    enc.eval().to(device)
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc, int(args["d"])


def gather_train_encounters() -> list[dict]:
    with open(REPO / "configs" / "splits" / "split_v1.json") as f:
        manifest = json.load(f)
    out = []
    for cid, case in manifest["cases"].items():
        if case["split"] != "train":
            continue
        # All non-test_a encounters in the case
        test_a_idx = set(case["test_a_encounter_indices"])
        for k in range(case["n_encounters_full"]):
            if k in test_a_idx:
                continue
            path = CACHE / "v1" / cid / f"encounter_{k:02d}.h5"
            if not path.exists():
                continue
            out.append({"case_id": cid, "k": int(k), "path": str(path)})
    return out


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


class EncounterFrameDataset(torch.utils.data.Dataset):
    """Yields a random T-frame sub-trajectory of omega_z for one encounter."""

    def __init__(self, encs: list[dict], T: int = 32, seed: int = 0) -> None:
        self.encs = encs
        self.T = T
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.encs)

    def __getitem__(self, idx: int) -> torch.Tensor:
        e = self.encs[idx]
        with h5py.File(e["path"], "r") as f:
            omega = np.asarray(f["omega_z"], dtype=np.float32)
        T_full = omega.shape[0]
        if T_full <= self.T:
            start = 0
        else:
            start = int(self.rng.integers(0, T_full - self.T + 1))
        x = omega[start : start + self.T]
        return torch.from_numpy(x)  # (T, H, W)


def collate(batch: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(batch, dim=0)  # (B, T, H, W)


def encode_batch(enc: HybridCNNViTEncoder, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Run frozen encoder. x: (B, T, H, W). Returns z: (B, T, d)."""
    with torch.no_grad():
        x = x.to(device).unsqueeze(2)  # (B, T, 1, H, W)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            z = enc(x)
    return z.float()


def build_lr_lambda(max_iters: int, warmup_frac: float):
    warmup = max(1, int(warmup_frac * max_iters))
    def fn(step: int) -> float:
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, max_iters - warmup)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * prog))
    return fn


def _ssim(x: np.ndarray, y: np.ndarray, c1: float = 0.16, c2: float = 1.44) -> float:
    """Fukami's SSIM definition (arXiv:2305.18394 Eq. 1) on a (H, W) pair."""
    mu_x, mu_y = x.mean(), y.mean()
    var_x, var_y = x.var(), y.var()
    cov_xy = ((x - mu_x) * (y - mu_y)).mean()
    num = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
    return float(num / max(den, 1e-12))


def evaluate_split(
    enc: HybridCNNViTEncoder,
    dec: HybridViTConvDecoder,
    encs: list[dict],
    device: torch.device,
) -> dict:
    """Per-encounter reconstruction MSE + SSIM on a split + case-mean noise floor."""
    case_to_arr: dict[str, list[np.ndarray]] = {}
    for e in encs:
        with h5py.File(e["path"], "r") as f:
            omega = np.asarray(f["omega_z"], dtype=np.float32)
        case_to_arr.setdefault(e["case_id"], []).append(omega)
    case_mean = {cid: np.stack(arrs, axis=0).mean(axis=0) for cid, arrs in case_to_arr.items()}

    mses = []
    floors = []
    ssims = []
    dec.eval()
    with torch.no_grad():
        for e in encs:
            with h5py.File(e["path"], "r") as f:
                omega = np.asarray(f["omega_z"], dtype=np.float32)
            T = omega.shape[0]
            x = torch.from_numpy(omega).unsqueeze(0).unsqueeze(2).to(device)  # (1, T, 1, H, W)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                z = enc(x)
                x_hat = dec(z)
            x_hat = x_hat.float().squeeze(0).squeeze(1).cpu().numpy()  # (T, H, W)
            mse = float(((omega - x_hat) ** 2).mean())
            floor = float(((omega - case_mean[e["case_id"]]) ** 2).mean())
            ssim_t = float(np.mean([_ssim(omega[t], x_hat[t]) for t in range(T)]))
            mses.append(mse)
            floors.append(floor)
            ssims.append(ssim_t)
    return {
        "mse_mean": float(np.mean(mses)),
        "mse_median": float(np.median(mses)),
        "ssim_mean": float(np.mean(ssims)),
        "ssim_median": float(np.median(ssims)),
        "floor_mean": float(np.mean(floors)),
        "ratio_mean": float(np.mean(mses) / max(np.mean(floors), 1e-12)),
        "n_encounters": len(encs),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Session 9 Step 2: train visualisation decoder")
    p.add_argument("--jepa-checkpoint", required=True, type=str)
    p.add_argument("--output-dir", required=True, type=str)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--B", type=int, default=16)
    p.add_argument("--T", type=int, default=32)
    p.add_argument("--max-iters", type=int, default=10000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--checkpoint-every", type=int, default=2000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = require_rtx6000(gpu_index=args.gpu)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "decoder_train.log"

    def log(msg: str) -> None:
        line = msg if msg.endswith("\n") else msg + "\n"
        with open(log_path, "a") as f:
            f.write(line)
        print(msg, flush=True)

    log(f"[decoder-train] device={device} gpu={torch.cuda.get_device_name(device.index)}")
    ckpt_path = Path(args.jepa_checkpoint).resolve()
    log(f"[decoder-train] jepa_checkpoint={ckpt_path}")
    log(f"[decoder-train] jepa_checkpoint sha256={file_sha256(ckpt_path)}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    enc, d = load_encoder(ckpt_path, device)
    log(f"[decoder-train] encoder loaded, d={d}, params="
        f"{sum(p.numel() for p in enc.parameters()):,} (FROZEN)")

    dec = HybridViTConvDecoder(latent_dim=d).to(device)
    log(f"[decoder-train] decoder params={sum(p.numel() for p in dec.parameters()):,}")

    train_encs = gather_train_encounters()
    test_a_encs = gather_eval_encounters("test_a")
    test_b_encs = gather_eval_encounters("test_b")
    test_c_encs = gather_eval_encounters("test_c")
    log(f"[decoder-train] train={len(train_encs)} encs, "
        f"test_a={len(test_a_encs)}, test_b={len(test_b_encs)}, test_c={len(test_c_encs)}")

    ds = EncounterFrameDataset(train_encs, T=args.T, seed=args.seed)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.B, shuffle=True, num_workers=args.num_workers,
        collate_fn=collate, pin_memory=True, drop_last=True, persistent_workers=True,
    )
    it = iter(loader)

    opt = torch.optim.AdamW(dec.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=build_lr_lambda(args.max_iters, args.warmup_frac),
    )

    metrics_path = out_dir / "decoder_metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    for step in range(args.max_iters + 1):
        try:
            x = next(it)
        except StopIteration:
            it = iter(loader)
            x = next(it)

        z = encode_batch(enc, x, device)  # (B, T, d)
        target = x.to(device).unsqueeze(2)  # (B, T, 1, H, W)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            x_hat = dec(z)
        loss = F.mse_loss(x_hat.float(), target.float())

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(dec.parameters(), args.grad_clip)
        opt.step()
        sched.step()

        if step % args.log_every == 0:
            log(f"[iter {step}/{args.max_iters}] loss={loss.item():.4f} "
                f"lr={sched.get_last_lr()[0]:.2e}")

        if step > 0 and step % args.eval_every == 0:
            ev_a = evaluate_split(enc, dec, test_a_encs[:8], device)  # cheap subset during training
            log(f"[eval iter {step}] test_a (subset 8): "
                f"mse_mean={ev_a['mse_mean']:.4f} floor_mean={ev_a['floor_mean']:.4f} "
                f"ratio={ev_a['ratio_mean']:.3f}")
            dec.train()
            with open(metrics_path, "a") as f:
                f.write(json.dumps({
                    "iter": step, "train_loss": float(loss.item()),
                    "test_a_subset8": ev_a,
                }) + "\n")

        if step > 0 and step % args.checkpoint_every == 0:
            ckpt_out = out_dir / f"decoder_iter{step:06d}.pt"
            torch.save({
                "decoder_state_dict": dec.state_dict(),
                "iter": step,
                "args": vars(args),
            }, ckpt_out)
            log(f"[checkpoint] saved {ckpt_out}")

    # Final full evaluation
    log("[decoder-train] final evaluation on Test A / B / C")
    ev_a = evaluate_split(enc, dec, test_a_encs, device)
    ev_b = evaluate_split(enc, dec, test_b_encs, device)
    ev_c = evaluate_split(enc, dec, test_c_encs, device)

    summary = {
        "jepa_checkpoint": str(ckpt_path),
        "jepa_checkpoint_sha256": file_sha256(ckpt_path),
        "latent_dim": d,
        "iters": args.max_iters,
        "test_a": ev_a,
        "test_b": ev_b,
        "test_c": ev_c,
        "pass_test_a_within_2x_floor": ev_a["ratio_mean"] < 2.0,
    }
    summary_path = out_dir / "decoder_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"[decoder-train] wrote {summary_path}")
    log(f"[decoder-train] FINAL test_a mse={ev_a['mse_mean']:.4f} floor={ev_a['floor_mean']:.4f} "
        f"ratio={ev_a['ratio_mean']:.3f} PASS={summary['pass_test_a_within_2x_floor']}")
    log(f"[decoder-train] FINAL test_b mse={ev_b['mse_mean']:.4f} floor={ev_b['floor_mean']:.4f} "
        f"ratio={ev_b['ratio_mean']:.3f}")
    log(f"[decoder-train] FINAL test_c mse={ev_c['mse_mean']:.4f} floor={ev_c['floor_mean']:.4f} "
        f"ratio={ev_c['ratio_mean']:.3f}")


if __name__ == "__main__":
    main()
