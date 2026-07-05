"""latent-REX-2: TiRex-2 lessons as our OWN trainable predictor (Session 34).

User-directed: no TiRex-2 weights; adapt the ARCHITECTURE idea instead. The
v2-specific lesson is FUTURE-KNOWN COVARIATES with strict target-causality
(their asymmetric variate mixer). Small-scale analogue here: a covariate
encoder (MLP over the episode parameters (G, D, Y) and the context-end phase
t0) FiLM-modulates the forecast head. Covariates never see the targets
(trivially causal); constants are fine because nothing erases them (no
foundation-model instance-norm on covariates).

DEPLOYMENT FRAMING: (G, D, Y) are unknown in real deployment, so the
conditioned variant is an ORACLE DIAGNOSTIC quantifying what knowing the gust
is worth for latent forecasting (the question the TiRex-2 zero-shot ladder
was meant to answer, now answered in-house with a trainable model). The
phase covariate t0 alone IS deployable.

Arms trained on the FULL train pool with the tuned winner's architecture
(LSTM h512 q9, 6000 iters), no new selection, one-shot test_b compare:
  none        : tuned baseline (= latent_rex_model_tuned, retrained here for
                seed parity)
  phase       : + t0 phase only (deployable)
  phase_gdy   : + (G, D, Y) oracle

Run (RTX 6000): taskset -c 8-15 python -m scripts.session34.rex2_cov --gpu 1
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

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from scripts.session34.rex_tune import make_quantiles, pinball  # noqa: E402
from scripts.session34.trackc_lift_eval import group_encounters, load_cache  # noqa: E402

CACHE = REPO_ROOT / "outputs/session34/trackc_latents"
CTX_EVAL, H = 25, 40


class Rex2(nn.Module):
    """LSTM h512 q9 REX with a FiLM covariate pathway on the forecast head."""

    def __init__(self, d: int = 32, hidden: int = 512, horizon: int = 40,
                 nq: int = 9, n_cov: int = 0) -> None:
        super().__init__()
        self.horizon, self.d, self.nq, self.n_cov = horizon, d, nq, n_cov
        self.lstm = nn.LSTM(d, hidden, num_layers=2, batch_first=True)
        self.head_in = nn.Linear(hidden, 4 * hidden)
        self.head_out = nn.Linear(4 * hidden, horizon * d * nq)
        if n_cov:
            self.cov_enc = nn.Sequential(
                nn.Linear(n_cov, hidden), nn.GELU(),
                nn.Linear(hidden, 2 * 4 * hidden),
            )
            nn.init.zeros_(self.cov_enc[-1].weight)   # identity FiLM at init
            nn.init.zeros_(self.cov_enc[-1].bias)

    def forward(self, ctx: torch.Tensor, cov: torch.Tensor | None = None) -> torch.Tensor:
        mu = ctx.mean(dim=1, keepdim=True)
        sd = ctx.std(dim=1, keepdim=True).clamp_min(1e-3)
        x = torch.asinh((ctx - mu) / sd)
        _, (h, _) = self.lstm(x)
        f = torch.nn.functional.gelu(self.head_in(h[-1]))
        if self.n_cov and cov is not None:
            scale, shift = self.cov_enc(cov).chunk(2, dim=-1)
            f = f * (1 + scale) + shift
        out = self.head_out(f).view(-1, self.horizon, self.d, self.nq)
        return torch.sinh(out) * sd[:, :, :, None] + mu[:, :, :, None]


def build_cov(mode: str, gdy: np.ndarray, t0: np.ndarray) -> np.ndarray | None:
    phase = (t0[:, None].astype(np.float32) * 0.05)
    if mode == "none":
        return None
    if mode == "phase":
        return phase
    return np.concatenate([phase, gdy.astype(np.float32)], axis=1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run", default="jepa_pool_vec")
    ap.add_argument("--out", default="outputs/session34/rex2_cov.json")
    args = ap.parse_args(argv)

    from src.utils.device import require_rtx6000

    device = require_rtx6000(gpu_index=args.gpu)
    quantiles = make_quantiles(9)

    tr = load_cache(CACHE, args.run, "train")
    tb = load_cache(CACHE, args.run, "test_b")
    split = json.loads((REPO_ROOT / "configs/splits/split_v2p2.json").read_text())["cases"]
    encs_tr = group_encounters(tr)
    encs_tb = group_encounters(tb)
    Ztr = np.stack([tr["z_gap"][e["rows"]] for e in encs_tr])
    gdy_tr = np.array([[split[e["case_id"]][k] for k in "GDY"] for e in encs_tr])
    Zt = torch.from_numpy(Ztr).float().to(device)
    Gt = torch.from_numpy(gdy_tr.astype(np.float32)).to(device)
    n_ep, T, d = Ztr.shape

    from src.evaluation.represent import fit_linear_probe

    probe = fit_linear_probe(tr["z_gap"], tr["cl"])
    Zte = np.stack([tb["z_gap"][e["rows"]] for e in encs_tb]).astype(np.float32)
    CLte = np.stack([tb["cl"][e["rows"]] for e in encs_tb])
    gdy_te = np.array([[split[e["case_id"]][k] for k in "GDY"] for e in encs_tb])

    results = {}
    for mode, n_cov in (("none", 0), ("phase", 1), ("phase_gdy", 4)):
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        model = Rex2(d=d, horizon=H, n_cov=n_cov).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters)
        t0c = time.time()
        for _ in range(args.iters):
            ep = rng.integers(0, n_ep, size=64)
            cl_ = int(rng.integers(16, 31))
            s = rng.integers(0, T - cl_ - H + 1, size=64)
            idx = s[:, None] + np.arange(cl_)[None]
            ctx = Zt[torch.from_numpy(ep).long()[:, None], torch.from_numpy(idx).long()]
            tgt_idx = (s + cl_)[:, None] + np.arange(H)[None]
            tgt = Zt[torch.from_numpy(ep).long()[:, None], torch.from_numpy(tgt_idx).long()]
            cov = None
            if n_cov:
                phase = torch.from_numpy(((s + cl_) * 0.05).astype(np.float32)) \
                    .to(device)[:, None]
                cov = phase if mode == "phase" else torch.cat(
                    [phase, Gt[torch.from_numpy(ep).long()]], dim=1)
            loss = pinball(model(ctx, cov), tgt, quantiles)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        model.eval()
        cov_te = build_cov(mode, gdy_te, np.full(len(encs_tb), CTX_EVAL))
        with torch.no_grad():
            pred = model(
                torch.from_numpy(Zte[:, :CTX_EVAL]).float().to(device),
                None if cov_te is None else torch.from_numpy(cov_te).to(device),
            ).cpu().numpy()
        mid = pred.shape[-1] // 2
        roll = pred[..., mid]
        zt = Zte[:, CTX_EVAL:CTX_EVAL + H]
        lat = 1 - ((zt - roll) ** 2).sum() / ((zt - zt.reshape(-1, d).mean(0)) ** 2).sum()
        ct = CLte[:, CTX_EVAL:CTX_EVAL + H].ravel()
        cp = probe.predict(roll.reshape(-1, d))
        cl_r2 = 1 - ((ct - cp) ** 2).sum() / ((ct - ct.mean()) ** 2).sum()
        results[mode] = {"latent_r2": float(lat), "decoded_cl_r2": float(cl_r2),
                         "train_s": time.time() - t0c}
        print(f"[rex2] {mode:10s}: latent R2={lat:+.3f} decoded C_L R2={cl_r2:+.3f} "
              f"({results[mode]['train_s']:.0f}s)", flush=True)
        torch.save(model.state_dict(),
                   REPO_ROOT / f"outputs/session34/rex2_{mode}.pt")
        del model
        torch.cuda.empty_cache()

    Path(REPO_ROOT / args.out).write_text(json.dumps(results, indent=1))
    print(f"[rex2] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
