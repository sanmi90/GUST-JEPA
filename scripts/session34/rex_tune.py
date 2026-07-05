"""latent-REX tuning with an honest protocol (Session 34, user-directed).

Fixes two things about the ad-hoc latent-REX: (1) its defaults were never
selected; (2) the filter band-scale was previously swept LOOKING AT test_b
(test peeking). Here every choice is made on a GROUPED VALIDATION split
(15% of train CASES held out, the D252 fit_lstm_tuned pattern); test_b is
touched exactly once, afterwards, by the separate eval/filter scripts.

Grid: backbone {lstm, xlstm} x hidden {256, 512} x n_quantiles {3, 9}.
xlstm = mLSTM-only xLSTMBlockStack (NX-AI xlstm 2.0.0; a 2-line shim works
around its import-time use of the cpp_extension API removed in torch 2.12;
mLSTM path is pure torch, no kernel compilation). Selection: val decoded-C_L
R^2 (probe fit on the train part only). Also calibrates the band scale c* so
the val one-step 10-90 coverage hits 0.80 (this becomes the filter's
band-scale, replacing the test-peeked 4.0). Winner is retrained on the FULL
train pool and saved as latent_rex_model_tuned.pt + tuned_meta.json.

Run (RTX 6000): taskset -c 8-15 python -m scripts.session34.rex_tune --gpu 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# xlstm 2.0.0 import-time shim for torch >= 2.12 (include_paths lost 'cuda' kwarg)
import torch.utils.cpp_extension as _cpp  # noqa: E402

_orig_include_paths = _cpp.include_paths
_cpp.include_paths = lambda *a, **k: _orig_include_paths()

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from scripts.session34.latent_rex import QUANTILES as _Q3  # noqa: E402
from scripts.session34.trackc_lift_eval import group_encounters, load_cache  # noqa: E402

CACHE = REPO_ROOT / "outputs/session34/trackc_latents"
Q9 = tuple(np.round(np.arange(0.1, 0.91, 0.1), 1))


def make_quantiles(nq: int):
    return _Q3 if nq == 3 else Q9


class RexBackbone(nn.Module):
    """LSTM or mLSTM-stack backbone under the shared REX wrapper."""

    def __init__(self, kind: str, d: int, hidden: int, horizon: int, nq: int) -> None:
        super().__init__()
        self.kind, self.horizon, self.d, self.nq = kind, horizon, d, nq
        if kind == "lstm":
            self.core = nn.LSTM(d, hidden, num_layers=2, batch_first=True)
        else:
            from xlstm import (mLSTMBlockConfig, mLSTMLayerConfig,
                               xLSTMBlockStack, xLSTMBlockStackConfig)

            self.in_proj = nn.Linear(d, hidden)
            self.core = xLSTMBlockStack(xLSTMBlockStackConfig(
                mlstm_block=mLSTMBlockConfig(mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4)),
                context_length=32, num_blocks=2, embedding_dim=hidden))
        self.head = nn.Sequential(
            nn.Linear(hidden, 4 * hidden), nn.GELU(),
            nn.Linear(4 * hidden, horizon * d * nq))

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        mu = ctx.mean(dim=1, keepdim=True)
        sd = ctx.std(dim=1, keepdim=True).clamp_min(1e-3)
        x = torch.asinh((ctx - mu) / sd)
        if self.kind == "lstm":
            _, (h, _) = self.core(x)
            feat = h[-1]
        else:
            feat = self.core(self.in_proj(x))[:, -1]
        out = self.head(feat).view(-1, self.horizon, self.d, self.nq)
        return torch.sinh(out) * sd[:, :, :, None] + mu[:, :, :, None]


def pinball(pred, target, quantiles):
    loss = 0.0
    for i, q in enumerate(quantiles):
        e = target - pred[..., i]
        loss = loss + torch.maximum(q * e, (q - 1) * e).mean()
    return loss / len(quantiles)


def train_one(Zt, device, kind, hidden, nq, iters, seed, horizon=40, batch=64):
    quantiles = make_quantiles(nq)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_ep, T, d = Zt.shape
    model = RexBackbone(kind, d, hidden, horizon, nq).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    for _ in range(iters):
        ep = rng.integers(0, n_ep, size=batch)
        cl_ = int(rng.integers(16, 31))
        s = rng.integers(0, T - cl_ - horizon + 1, size=batch)
        idx = s[:, None] + np.arange(cl_)[None]
        ctx = Zt[torch.from_numpy(ep).long()[:, None], torch.from_numpy(idx).long()]
        tgt_idx = (s + cl_)[:, None] + np.arange(horizon)[None]
        tgt = Zt[torch.from_numpy(ep).long()[:, None], torch.from_numpy(tgt_idx).long()]
        loss = pinball(model(ctx), tgt, quantiles)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    return model


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--run", default="jepa_pool_vec")
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--out", default="outputs/session34/rex_tune.json")
    args = ap.parse_args(argv)

    from src.utils.device import require_rtx6000

    device = require_rtx6000(gpu_index=args.gpu)
    tr = load_cache(CACHE, args.run, "train")
    encs = group_encounters(tr)
    cases = sorted(set(e["case_id"] for e in encs))
    rng = np.random.default_rng(0)
    val_cases = set(rng.choice(cases, size=max(2, int(len(cases) * args.val_frac)),
                               replace=False).tolist())
    tr_encs = [e for e in encs if e["case_id"] not in val_cases]
    va_encs = [e for e in encs if e["case_id"] in val_cases]
    Ztr = np.stack([tr["z_gap"][e["rows"]] for e in tr_encs])
    Zva = np.stack([tr["z_gap"][e["rows"]] for e in va_encs])
    CLva = np.stack([tr["cl"][e["rows"]] for e in va_encs])
    Zt = torch.from_numpy(Ztr).float().to(device)
    print(f"[tune] {len(tr_encs)} train / {len(va_encs)} val encounters "
          f"({len(val_cases)} val cases)", flush=True)

    from src.evaluation.represent import fit_linear_probe

    Xp = np.concatenate([tr["z_gap"][e["rows"]] for e in tr_encs])
    yp = np.concatenate([tr["cl"][e["rows"]] for e in tr_encs])
    probe = fit_linear_probe(Xp, yp)

    CTX, H = 25, 40
    rows = []
    for kind in ("lstm", "xlstm"):
        for hidden in (256, 512):
            for nq in (3, 9):
                t0 = time.time()
                model = train_one(Zt, device, kind, hidden, nq, args.iters, args.seed)
                model.eval()
                with torch.no_grad():
                    pred = model(torch.from_numpy(Zva[:, :CTX]).float().to(device)) \
                        .cpu().numpy()
                mid = pred.shape[-1] // 2
                roll = pred[..., mid]
                zt = Zva[:, CTX:CTX + H]
                lat = 1 - ((zt - roll) ** 2).sum() / ((zt - zt.reshape(-1, zt.shape[-1])
                                                       .mean(0)) ** 2).sum()
                ct = CLva[:, CTX:CTX + H].ravel()
                cp = probe.predict(roll.reshape(-1, roll.shape[-1]))
                cl_r2 = 1 - ((ct - cp) ** 2).sum() / ((ct - ct.mean()) ** 2).sum()
                # one-step band calibration factor on val (coverage -> 0.80)
                with torch.no_grad():
                    origins = list(range(16, 100, 6))
                    res_ratio = []
                    for t0_ in origins:
                        p1 = model(torch.from_numpy(Zva[:, max(0, t0_ - 25):t0_])
                                   .float().to(device)).cpu().numpy()
                        band = np.clip(p1[:, 0, :, -1] - p1[:, 0, :, 0], 1e-6, None)
                        resid = np.abs(Zva[:, t0_] - p1[:, 0, :, mid])
                        res_ratio.append((2 * resid / band).ravel())
                    c_star = float(np.quantile(np.concatenate(res_ratio), 0.80))
                n_par = sum(p.numel() for p in model.parameters()) / 1e6
                rows.append({"kind": kind, "hidden": hidden, "nq": nq,
                             "val_latent_r2": float(lat), "val_cl_r2": float(cl_r2),
                             "band_c_star": c_star, "params_m": n_par,
                             "train_s": time.time() - t0})
                print(f"[tune] {kind} h{hidden} q{nq}: val C_L R2={cl_r2:+.3f} "
                      f"latent={lat:+.3f} c*={c_star:.2f} "
                      f"({n_par:.1f}M, {rows[-1]['train_s']:.0f}s)", flush=True)
                del model
                torch.cuda.empty_cache()

    best = max(rows, key=lambda r: r["val_cl_r2"])
    print(f"[tune] WINNER: {best['kind']} h{best['hidden']} q{best['nq']} "
          f"(val C_L {best['val_cl_r2']:+.3f}, c*={best['band_c_star']:.2f})", flush=True)

    # retrain winner on FULL train pool
    Zfull = torch.from_numpy(np.stack([tr["z_gap"][e["rows"]] for e in encs])) \
        .float().to(device)
    final = train_one(Zfull, device, best["kind"], best["hidden"], best["nq"],
                      args.iters, args.seed)
    torch.save(final.state_dict(),
               REPO_ROOT / "outputs/session34/latent_rex_model_tuned.pt")
    meta = {"grid": rows, "winner": best, "run": args.run, "iters": args.iters,
            "val_cases": sorted(val_cases)}
    (REPO_ROOT / "outputs/session34/rex_tune.json").write_text(
        json.dumps(meta, indent=1))
    print("[tune] saved latent_rex_model_tuned.pt + rex_tune.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
