"""latent-REX: TiRex-lessons recurrent quantile forecaster on frozen latents.

UNCONDITIONED temporal model for the d=32 pooled latent (Session 34,
user-directed), taking the transferable lessons of TiRex/TiRex-2
(arXiv 2607.01204) without the foundation-model machinery:

- recurrent backbone (2-layer LSTM; the project's own rebuttal work already
  showed recurrence beats attention in-envelope on this data),
- per-window instance normalization + arcsinh scaling (robust to the |G|
  envelope's scale shifts; inverted at output),
- DIRECT multi-horizon output (one shot, H steps x d dims x 3 quantiles),
  no autoregressive feedback, so rollout error cannot compound,
- pinball (quantile) loss at q = {0.1, 0.5, 0.9}; the (q90 - q10) spread is a
  calibrated per-step predictive band (candidate state-dependent EnKF Q).

Trains on the frozen flagship's cached train latents in minutes; scored with
the identical protocol as trackc_forecast / tirex_forecast (context 25,
horizon 40, latent R^2 + decoded C_L R^2 via the frozen affine probe).

Run: taskset -c 0-15 python -m scripts.session34.latent_rex --gpu 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.session34.trackc_lift_eval import group_encounters, load_cache  # noqa: E402

CACHE = REPO_ROOT / "outputs/session34/trackc_latents"
RUN = "jepa_pool_vec"
QUANTILES = (0.1, 0.5, 0.9)


class LatentRex(nn.Module):
    def __init__(self, d: int = 32, hidden: int = 256, layers: int = 2,
                 horizon: int = 40, nq: int = 3) -> None:
        super().__init__()
        self.horizon, self.d, self.nq = horizon, d, nq
        self.lstm = nn.LSTM(d, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, 4 * hidden), nn.GELU(),
            nn.Linear(4 * hidden, horizon * d * nq),
        )

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        """ctx (B, T, d) RAW latents -> quantile forecasts (B, H, d, nq) RAW."""
        mu = ctx.mean(dim=1, keepdim=True)
        sd = ctx.std(dim=1, keepdim=True).clamp_min(1e-3)
        x = torch.asinh((ctx - mu) / sd)                      # instance norm + arcsinh
        _, (h, _) = self.lstm(x)
        out = self.head(h[-1]).view(-1, self.horizon, self.d, self.nq)
        return torch.sinh(out) * sd[:, :, :, None] + mu[:, :, :, None]


def pinball(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """pred (B, H, d, nq), target (B, H, d)."""
    loss = 0.0
    for i, q in enumerate(QUANTILES):
        e = target - pred[..., i]
        loss = loss + torch.maximum(q * e, (q - 1) * e).mean()
    return loss / len(QUANTILES)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ctx", type=int, default=25)
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--out", default="outputs/session34/latent_rex.json")
    args = ap.parse_args()

    from src.utils.device import require_rtx6000

    device = require_rtx6000(gpu_index=args.gpu)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    tr = load_cache(CACHE, RUN, "train")
    tb = load_cache(CACHE, RUN, "test_b")
    encs_tr = group_encounters(tr)
    Ztr = np.stack([tr["z"] if False else tr["z_gap"][e["rows"]] for e in encs_tr])
    n_ep, T, d = Ztr.shape
    Zt = torch.from_numpy(Ztr).float().to(device)

    model = LatentRex(d=d, horizon=args.horizon).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters)
    print(f"[rex] {n_par/1e6:.2f}M params, {n_ep} train episodes, device={device}",
          flush=True)

    t0 = time.time()
    for it in range(1, args.iters + 1):
        ep = rng.integers(0, n_ep, size=args.batch)
        ctx_len = int(rng.integers(16, args.ctx + 6))          # 16..30, robustness
        s = rng.integers(0, T - ctx_len - args.horizon + 1, size=args.batch)
        idx = s[:, None] + np.arange(ctx_len)[None]
        ctx = Zt[torch.from_numpy(ep).long()[:, None], torch.from_numpy(idx).long()]
        tgt_idx = (s + ctx_len)[:, None] + np.arange(args.horizon)[None]
        tgt = Zt[torch.from_numpy(ep).long()[:, None], torch.from_numpy(tgt_idx).long()]
        loss = pinball(model(ctx), tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % 500 == 0:
            print(f"[rex] iter {it}/{args.iters} pinball={loss.item():.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    # ---- eval: identical protocol -------------------------------------------
    blob = np.load(REPO_ROOT / "outputs/session34/tirex_input.npz", allow_pickle=True)
    Z, CL = blob["Z_test"].astype(np.float32), blob["CL_test"]
    w, b = blob["probe_w"], float(blob["probe_b"])
    CTX, H = 25, args.horizon
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(Z[:, :CTX]).float().to(device)).cpu().numpy()
    roll = pred[..., 1]                                        # median (42, H, d)
    spread = (pred[..., 2] - pred[..., 0])                     # q90 - q10
    zt = Z[:, CTX:CTX + H]
    lat = 1 - ((zt - roll) ** 2).sum() / ((zt - zt.reshape(-1, d).mean(0)) ** 2).sum()
    pers = np.repeat(Z[:, CTX - 1][:, None], H, 1)
    lat_pers = 1 - ((zt - roll) ** 2).sum() / ((zt - pers) ** 2).sum()
    ct = CL[:, CTX:CTX + H].ravel()
    cp = roll.reshape(-1, d) @ w + b
    cl_r2 = 1 - ((ct - cp) ** 2).sum() / ((ct - ct.mean()) ** 2).sum()
    # calibration of the 10-90 band
    inside = float(((zt >= pred[..., 0]) & (zt <= pred[..., 2])).mean())
    print(f"[rex] EVAL: latent R2={lat:+.3f} (vs persist {lat_pers:+.3f})  "
          f"decoded C_L R2={cl_r2:+.3f}  band(10-90) coverage={inside:.2f}",
          flush=True)
    Path(REPO_ROOT / args.out).write_text(json.dumps({
        "params_m": n_par / 1e6, "iters": args.iters,
        "latent_r2": float(lat), "latent_r2_vs_persistence": float(lat_pers),
        "decoded_cl_r2": float(cl_r2), "band_coverage_10_90": inside,
        "mean_spread": float(spread.mean()),
        "wall_s": time.time() - t0,
    }, indent=1))
    torch.save(model.state_dict(),
               REPO_ROOT / "outputs/session34/latent_rex_model.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
