"""Session 18 B1 Part (d), step 1: run Markov-only and Full-context
rollouts for a (baseline, d) pair on Test B and Test C using precomputed
latents + the baseline's trained transformer predictor.

Output:
    outputs/session18/exp_b1/rollouts_{tag}/{split}.npz
    keys:
        z_dns      (n_enc, 120, d)         ground-truth per-frame latents
        z_markov   (n_enc, 120, d)         Markov-only rollout from z_impact
        z_full     (n_enc, 120, d)         Full-context rollout from z[:impact+1]
        G, D, Y, case_ids, encounter_indices, impact_frame

The Markov rollout uses the attention monkey-patch from Session 17 exp2:
``CausalSelfAttentionWithRoPE.forward`` is replaced inside a context
manager with a variant that masks attention to (z_0, self) only.

Physical observables and per-baseline probes are computed downstream by
``physical_metrics_from_rollouts.py`` (B1 Part d, step 2).

Usage:
    python scripts/session18/eval_baseline_rollouts.py \\
        --latents-dir outputs/session18/exp_b1/latents_fukami_d64 \\
        --predictor outputs/session18/exp_b1/predictor_fukami_d64/checkpoint_iter020000.pt \\
        --tag fukami_d64

The script is hardware-locked to RTX 6000 via ``require_rtx6000``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.models.predictor import (  # noqa: E402
    AutoregressivePredictor,
    CausalSelfAttentionWithRoPE,
)
from src.models.rope import apply_rope  # noqa: E402
from src.utils.device import require_rtx6000  # noqa: E402


def make_markov_attention_forward():
    """Return a replacement ``forward`` that masks attention to (z_0, self)."""

    def markov_forward(self: CausalSelfAttentionWithRoPE, x: Tensor) -> Tensor:
        B, T, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = apply_rope(q, self.rope_cos[:T], self.rope_sin[:T])
        k = apply_rope(k, self.rope_cos[:T], self.rope_sin[:T])
        mask = torch.full((T, T), float("-inf"), device=x.device, dtype=q.dtype)
        mask[:, 0] = 0.0
        mask.fill_diagonal_(0.0)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
        )
        return self.proj(out.transpose(1, 2).reshape(B, T, -1))

    return markov_forward


@contextmanager
def markov_attention(model: AutoregressivePredictor) -> Iterator[None]:
    """Context manager that swaps every attention module to Markov-mask mode."""
    new_forward = make_markov_attention_forward()
    originals: list = []
    for module in model.modules():
        if isinstance(module, CausalSelfAttentionWithRoPE):
            originals.append((module, module.forward))
            module.forward = new_forward.__get__(module, type(module))
    try:
        yield
    finally:
        for module, original in originals:
            module.forward = original


@torch.no_grad()
def rollout_markov(
    pred: AutoregressivePredictor,
    z_init: Tensor,
    cond: Tensor,
    steps: int,
    device: torch.device,
) -> Tensor:
    """Markov-only rollout. z_init: (B, 1, d). Returns (B, 1 + steps, d)."""
    max_seq = int(pred.max_seq_len)
    z_full = z_init.clone()
    with markov_attention(pred):
        for _ in range(steps):
            ctx = z_full[:, -max_seq:, :]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                z_hat = pred(ctx, cond)
            z_full = torch.cat([z_full, z_hat[:, -1:, :].float()], dim=1)
    return z_full


@torch.no_grad()
def rollout_full(
    pred: AutoregressivePredictor,
    z_seed: Tensor,
    cond: Tensor,
    steps: int,
    device: torch.device,
) -> Tensor:
    """Full-context rollout. z_seed: (B, T_init, d) with T_init <= max_seq."""
    max_seq = int(pred.max_seq_len)
    z_full = z_seed.clone()
    for _ in range(steps):
        ctx = z_full[:, -max_seq:, :]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            z_hat = pred(ctx, cond)
        z_full = torch.cat([z_full, z_hat[:, -1:, :].float()], dim=1)
    return z_full


def load_predictor(checkpoint: Path, d: int, device: torch.device):
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = blob.get("run_config", {})
    pcfg = cfg.get("predictor_config", {})
    pred = AutoregressivePredictor(
        latent_dim=d,
        cond_dim=int(pcfg.get("cond_dim", 3)),
        hidden_dim=int(pcfg.get("hidden_dim", 384)),
        depth=int(pcfg.get("depth", 6)),
        heads=int(pcfg.get("heads", 16)),
        mlp_ratio=float(pcfg.get("mlp_ratio", 4.0)),
        dropout=float(pcfg.get("dropout", 0.1)),
        max_seq_len=int(pcfg.get("max_seq_len", 32)),
    ).to(device)
    state = blob["predictor_state_dict"]
    if "out_proj.1.weight" not in state and "out_proj.1.running_mean" not in state:
        # B1 Test 1: checkpoint trained with --no-output-bn (out_proj is
        # Sequential(Linear, Identity)). Patch the predictor to match.
        from torch import nn as _nn
        out_lin = pred.out_proj[0]
        pred.out_proj = _nn.Sequential(out_lin, _nn.Identity()).to(device)
        print(f"[load] patched out_proj to Identity (no-output-bn checkpoint)")
    pred.load_state_dict(state)
    pred.eval()
    for p in pred.parameters():
        p.requires_grad_(False)
    mean = blob.get("latent_mean")
    std = blob.get("latent_std")
    if mean is None or std is None:
        raise RuntimeError(
            f"checkpoint {checkpoint} missing latent_mean / latent_std; "
            "predictor was trained without z-score normalisation. Retrain."
        )
    mean_t = torch.tensor(np.asarray(mean), dtype=torch.float32, device=device)
    std_t = torch.tensor(np.asarray(std), dtype=torch.float32, device=device)
    return pred, mean_t, std_t


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline rollouts for B1 Part d")
    p.add_argument("--latents-dir", type=Path, required=True)
    p.add_argument("--predictor", type=Path, required=True)
    p.add_argument("--tag", type=str, required=True)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: outputs/session18/exp_b1/rollouts_{tag}.",
    )
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument(
        "--splits",
        nargs="+",
        default=["test_b", "test_c"],
    )
    p.add_argument(
        "--max-seq",
        type=int,
        default=32,
        help="Maximum seed length for Full-context rollout (capped at predictor.max_seq_len).",
    )
    return p.parse_args()


def _get(blob, *names):
    """Return the first present field across name aliases (handles Session 14
    singular 'case_id'/'encounter_index' vs Session 18 plural variants)."""
    for n in names:
        if n in blob.files:
            return blob[n]
    raise KeyError(f"none of {names} present in npz")


def rollout_split(
    npz_path: Path,
    pred: AutoregressivePredictor,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
    device: torch.device,
    max_seq: int,
    T_total: int = 120,
) -> dict:
    blob = np.load(npz_path, allow_pickle=True)
    z_full_dns_raw = blob["z_full"].astype(np.float32)  # (n_enc, 120, d)
    G = blob["G"].astype(np.float32)
    D = blob["D"].astype(np.float32)
    Y = blob["Y"].astype(np.float32)
    impact = _get(blob, "impact_frame").astype(np.int64)
    case_ids = _get(blob, "case_ids", "case_id")
    enc_idx = _get(blob, "encounter_indices", "encounter_index")

    n_enc, _, d = z_full_dns_raw.shape
    z_dns_raw = torch.from_numpy(z_full_dns_raw).to(device)
    z_dns_norm = (z_dns_raw - mean_t) / std_t
    cond_t = torch.from_numpy(np.stack([G, D, Y], axis=1)).to(device)

    z_markov_norm = torch.zeros_like(z_dns_norm)
    z_full_norm = torch.zeros_like(z_dns_norm)

    pred_max_seq = int(pred.max_seq_len)
    full_seed_len = min(max_seq, pred_max_seq)

    for i in range(n_enc):
        ti = int(impact[i])
        c = cond_t[i : i + 1]

        z_markov_norm[i, : ti + 1] = z_dns_norm[i, : ti + 1]
        z_init = z_dns_norm[i, ti : ti + 1].unsqueeze(0)
        steps = T_total - ti - 1
        if steps > 0:
            z_m = rollout_markov(pred, z_init, c, steps=steps, device=device)
            z_markov_norm[i, ti + 1 : T_total] = z_m[0, 1 : steps + 1].float()

        seed_start = max(0, ti + 1 - full_seed_len)
        z_seed = z_dns_norm[i, seed_start : ti + 1].unsqueeze(0)
        z_full_norm[i, : ti + 1] = z_dns_norm[i, : ti + 1]
        if steps > 0:
            z_f = rollout_full(pred, z_seed, c, steps=steps, device=device)
            z_full_norm[i, ti + 1 : T_total] = z_f[0, -steps:].float()

        if (i + 1) % 10 == 0 or (i + 1) == n_enc:
            print(f"   rollout {i + 1}/{n_enc}")

    # Un-normalise back to raw latent space for downstream probes.
    z_markov_raw = z_markov_norm * std_t + mean_t
    z_full_raw = z_full_norm * std_t + mean_t

    return {
        "z_dns": z_dns_raw.cpu().numpy(),
        "z_markov": z_markov_raw.cpu().numpy(),
        "z_full": z_full_raw.cpu().numpy(),
        "G": G,
        "D": D,
        "Y": Y,
        "case_ids": case_ids,
        "encounter_indices": enc_idx,
        "impact_frame": impact,
    }


def main() -> None:
    args = parse_args()
    device = require_rtx6000(gpu_index=args.gpu)

    train_npz = np.load(args.latents_dir / "train.npz")
    d = int(train_npz["z_full"].shape[-1])

    if args.output_dir is None:
        args.output_dir = (
            REPO / "outputs" / "session18" / "exp_b1" / f"rollouts_{args.tag}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pred, mean_t, std_t = load_predictor(args.predictor, d, device)
    print(
        f"[eval-rollouts] tag={args.tag}  d={d}  predictor={args.predictor.name}"
        f"  device={device}"
    )

    for split in args.splits:
        in_path = args.latents_dir / f"{split}.npz"
        if not in_path.exists():
            print(f"[eval-rollouts] {split}: latent npz missing at {in_path}; skipping")
            continue
        print(f"[eval-rollouts] {split}: rolling out from {in_path}")
        out = rollout_split(in_path, pred, mean_t, std_t, device, max_seq=args.max_seq)
        out_path = args.output_dir / f"{split}.npz"
        np.savez(out_path, **out)
        print(
            f"[eval-rollouts] {split}: wrote {out_path} "
            f"({out['z_dns'].nbytes / 1e6:.2f} MB)"
        )

    print("[eval-rollouts] DONE")


if __name__ == "__main__":
    main()
