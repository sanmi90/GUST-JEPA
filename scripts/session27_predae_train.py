"""Predictive autoencoder (PredAE): Fukami encoder -> LSTM in latent -> decoder,
trained with a PIXEL-space next-frame prediction loss (the loss-space control vs
JEPA's latent-space prediction). Unconditioned. Encoder is co-shaped by the
predictive-decode objective. Saves encoder+predictor+decoder for a later closure
eval (roll the LSTM, probe wake enstrophy).

Loss = L_tf (teacher-forced next-frame recon) + 0.5 * L_roll (open-loop H_roll-step
rollout, decode each step, pixel loss vs future frames). smooth_l1 on normalised omega.

Usage: python scripts/session27_predae_train.py --iters 15000 --device cuda:1
"""
from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.baselines.fukami_ae import FukamiCNNEncoder, FukamiCNNDecoder  # noqa: E402
from src.models.predictor import LSTMLatentPredictor  # noqa: E402
from src.training.train_jepa import make_train_loader, infinite_iter  # noqa: E402

H_ROLL = 4


def lr_lambda(it, iters, warm=0.05, floor=0.05):
    w = max(1, int(warm * iters))
    if it < w:
        return (it + 1) / w
    import math
    p = min(1.0, (it - w) / max(1, iters - w))
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=15000)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--T", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/session27/predae_d64")
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = REPO / args.out; out.mkdir(parents=True, exist_ok=True)

    # omega sub-trajectory loader (same as JEPA training, observable heads off)
    la = Namespace(partition="v1", T=args.T, B=args.B, num_workers=3, cases=None, split=None,
                   observable_head="none", observable_head_deltas=[0],
                   wake_observable_type="patch_signed_spectrum", wake_observables_root=None,
                   omega_pipeline_manifest=str(REPO / "outputs/data_pipeline/v1/manifest.json"))
    loader = make_train_loader(la)
    it_loader = infinite_iter(loader)

    enc = FukamiCNNEncoder(args.d).to(device)
    dec = FukamiCNNDecoder(args.d).to(device)
    lstm = LSTMLatentPredictor(args.d, cond_dim=0, hidden_dim=256, num_layers=3, max_seq_len=args.T).to(device)
    params = list(enc.parameters()) + list(dec.parameters()) + list(lstm.parameters())
    n_par = sum(p.numel() for p in params)
    print(f"[predae] device={device} params={n_par/1e6:.2f}M (enc+lstm+dec)", flush=True)
    opt = AdamW(params, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.05)
    sch = LambdaLR(opt, lr_lambda=lambda it: lr_lambda(it, args.iters))
    rng = np.random.default_rng(args.seed + 1)

    def encode_seq(omega):                          # (B,L,1,H,W) -> (B,L,d)
        B, L, C, Hh, Ww = omega.shape
        z = enc(omega.reshape(B * L, C, Hh, Ww))
        return z.reshape(B, L, -1)

    def decode_seq(z):                              # (B,M,d) -> (B,M,1,H,W)
        B, M, d = z.shape
        x = dec(z.reshape(B * M, d))
        return x.reshape(B, M, *x.shape[1:])

    enc.train(); dec.train(); lstm.train()
    for it in range(args.iters):
        batch = next(it_loader)
        omega = batch["omega"].to(device, non_blocking=True)   # (B,L,1,H,W), normalised
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = encode_seq(omega)                              # (B,L,d)
            zhat = lstm(z)                                     # next-step latent preds
            pred_tf = decode_seq(zhat[:, :-1])                 # predicted omega(t+1)
            L_tf = F.smooth_l1_loss(pred_tf, omega[:, 1:], beta=0.5)
            start = int(rng.integers(0, max(1, args.T - H_ROLL)))
            zroll = lstm.rollout(z[:, :start + 1], None, H_ROLL)        # (B, start+1+H_ROLL, d)
            pred_roll = decode_seq(zroll[:, start + 1:start + 1 + H_ROLL])
            L_roll = F.smooth_l1_loss(pred_roll, omega[:, start + 1:start + 1 + H_ROLL], beta=0.5)
            loss = L_tf + 0.5 * L_roll
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sch.step()
        if it % 200 == 0:
            print(f"[iter {it}/{args.iters}] loss={loss.item():.4f} (tf={L_tf.item():.4f} roll={L_roll.item():.4f})", flush=True)
        if it > 0 and it % 5000 == 0:
            torch.save({"enc": enc.state_dict(), "dec": dec.state_dict(), "lstm": lstm.state_dict(),
                        "iter": it, "d": args.d}, out / f"checkpoint_iter{it:06d}.pt")
    torch.save({"enc": enc.state_dict(), "dec": dec.state_dict(), "lstm": lstm.state_dict(),
                "iter": args.iters, "d": args.d}, out / f"checkpoint_iter{args.iters:06d}.pt")
    print(f"[predae] DONE -> {out}/checkpoint_iter{args.iters:06d}.pt", flush=True)


if __name__ == "__main__":
    main()
