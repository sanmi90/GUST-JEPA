"""Train a faithful V-JEPA on the gust omega clips. RTX 6000 only.

Reuses train_jepa's data loader (make_train_loader/infinite_iter); adds EMA
target update, multi-block masking, masked smooth-L1 loss, and a collapse
diagnostic (token feature std + participation ratio) since there is no SIGReg.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from src.models.vjepa import VJEPA
from src.training.train_jepa import make_train_loader, infinite_iter
from src.utils.device import require_rtx6000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--partition", default="v2p1")
    p.add_argument("--split", default="configs/splits/split_v2p1.json")
    p.add_argument("--omega-pipeline-manifest", default="outputs/data_pipeline/v2p1/manifest.json")
    p.add_argument("--B", type=int, default=16)
    p.add_argument("--T", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=3)
    p.add_argument("--max-iters", type=int, default=20000)
    p.add_argument("--lr-encoder", type=float, default=1.5e-4)
    p.add_argument("--lr-predictor", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--mask-ratio", type=float, default=0.8)
    p.add_argument("--ema-base", type=float, default=0.996)
    p.add_argument("--ema-final", type=float, default=1.0)
    p.add_argument("--hidden", type=int, default=384)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--pred-depth", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--diagnostic-every", type=int, default=2000)
    p.add_argument("--checkpoint-every", type=int, default=10000)
    p.add_argument("--wandb-mode", default="offline")
    p.add_argument("--tag-suffix", default="")
    p.add_argument("--output-dir", required=True)
    # gating args so make_train_loader's _emit_cl/_emit_wake both return False
    # (V-JEPA needs only omega clips). Defaults match "off".
    p.add_argument("--observable-head", default="none")
    p.add_argument("--observable-head-weight", type=float, default=0.01)
    p.add_argument("--observable-head-deltas", type=int, nargs="+", default=[8, 16, 24])
    p.add_argument("--wake-observable-type", default="none")
    p.add_argument("--lambda-wake", type=float, default=0.0)
    p.add_argument("--wake-observables-root", default=None)
    p.add_argument("--cases", nargs="+", default=None)
    p.add_argument("--all-train", action="store_true")
    p.add_argument(
        "--lam-ctx",
        type=float,
        default=0.5,
        help="V-JEPA 2.1 dense context-loss base weight (0 = original masked-only).",
    )
    p.add_argument(
        "--lam-ctx-warmup-frac",
        type=float,
        default=0.25,
        help="Fraction of training over which lam_ctx warms 0 -> lam_ctx.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = require_rtx6000(gpu_index=args.gpu)
    name = torch.cuda.get_device_name(device.index)
    assert "RTX" in name and "6000" in name, f"not RTX 6000: {name}"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    loader = make_train_loader(args)
    it = infinite_iter(loader)
    model = VJEPA(
        hidden=args.hidden, depth=args.depth, pred_depth=args.pred_depth, mask_ratio=args.mask_ratio
    ).to(device)
    enc_params = list(model.tokenizer.parameters()) + list(model.context_encoder.parameters())
    pred_params = (
        list(model.pred_embed.parameters())
        + [model.mask_token]
        + list(model.pred_blocks.parameters())
        + list(model.pred_norm.parameters())
        + list(model.pred_proj.parameters())
    )
    opt = torch.optim.AdamW(
        [
            {"params": enc_params, "lr": args.lr_encoder},
            {"params": pred_params, "lr": args.lr_predictor},
        ],
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    warmup = int(args.warmup_frac * args.max_iters)
    base_lrs = [args.lr_encoder, args.lr_predictor]

    def lr_scale(i: int) -> float:
        if i < warmup:
            return (i + 1) / max(1, warmup)
        prog = (i - warmup) / max(1, args.max_iters - warmup)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * prog))

    def ema_m(i: int) -> float:
        prog = i / max(1, args.max_iters)
        return args.ema_base + (args.ema_final - args.ema_base) * prog

    log = open(out / "train.log", "w")
    for i in range(args.max_iters):
        batch = next(it)
        omega = batch["omega"].to(device)
        if omega.dim() == 4:
            omega = omega.unsqueeze(2)
        s = lr_scale(i)
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * s
        opt.zero_grad(set_to_none=True)
        lam_w = args.lam_ctx * min(1.0, i / max(1.0, args.lam_ctx_warmup_frac * args.max_iters))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out_d = model(omega, lam_ctx=lam_w)
        out_d["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        model.ema_update(ema_m(i))
        if i % args.log_every == 0:
            lp = float(out_d.get("l_pred", out_d["loss"]))
            lc = float(out_d["l_ctx"]) if "l_ctx" in out_d else 0.0
            print(
                f"[iter {i}/{args.max_iters}] L={float(out_d['loss'].detach()):.4f} "
                f"Lpred={lp:.4f} Lctx={lc:.4f} lam={lam_w:.3f} "
                f"lr={opt.param_groups[0]['lr']:.2e} ema={ema_m(i):.4f}",
                file=log,
                flush=True,
            )
        if i % args.diagnostic_every == 0:
            with torch.no_grad():
                tok = model.encode_tokens(omega).float()
                z = tok.reshape(-1, tok.shape[-1])
                sv = torch.linalg.svdvals(z - z.mean(0))
                pr = float((sv.sum() ** 2) / (sv**2).sum())
            print(
                f"[diag iter {i}] token_std={float(tok.std()):.4f} " f"PR={pr:.1f}/{tok.shape[-1]}",
                file=log,
                flush=True,
            )
        if (i + 1) % args.checkpoint_every == 0 or i + 1 == args.max_iters:
            torch.save(
                {"vjepa_state_dict": model.state_dict(), "args": vars(args), "iter": i + 1},
                out / f"checkpoint_iter{i + 1:06d}.pt",
            )
    log.close()


if __name__ == "__main__":
    main()
