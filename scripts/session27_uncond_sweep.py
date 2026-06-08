"""Can an UNCONDITIONED temporal predictor match the conditioned forecast?

Holds the production encoder FROZEN (jepa_d64_test1_noBN latents) and trains a
range of predictor architectures on the d=64 latent trajectories, with and
without the gust conditioning c=(G,D,Y). Scores each with the canonical closure
probe (exp_closure_r2): held-out wake-enstrophy R^2 at impact+H=16.

Two rollout modes per model:
  - markov   : seed the single impact-frame latent (matches the conditioned 0.449
               protocol). With c this is the published number; without c it is the
               known no-c collapse.
  - fullctx  : seed the pre-impact window z[:impact+1] (last max_seq frames). This
               is where a Takens-style temporal window can let an UNCONDITIONED
               model infer the forcing from history.

Models: transformer (cond3 = harness validator, cond0), GRU, LSTM, delay-embedding
MLP (explicit Takens window), and an echo-state reservoir (ridge readout).

Usage:  python scripts/session27_uncond_sweep.py --iters 20000 --gpu 0
        python scripts/session27_uncond_sweep.py --iters 400 --models tf_cond3 gru   # smoke
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "session20"))
sys.path.insert(0, str(REPO / "scripts" / "session18"))
import exp_closure_r2 as cr  # noqa: E402
from src.models.predictor import AutoregressivePredictor  # noqa: E402
from src.utils.device import require_rtx6000  # noqa: E402
from src.training.scheduled_sampling import (  # noqa: E402
    teacher_forced_prediction_loss, open_loop_rollout_loss,
)
from train_baseline_predictor import (  # noqa: E402
    LatentSubTrajectoryDataset, collate, build_lr_lambda, PROTOCOL_DEFAULTS as PD,
)

LATDIR = REPO / "outputs/session18/exp_b1/latents_jepa_d64_test1_noBN"
OUT = REPO / "outputs/session27/uncond_sweep.csv"
MAX_SEQ = 32
H = 16
HMAX = 18


# --------------------------- models (forward + rollout) ---------------------------
class _RollMixin:
    def rollout(self, z_init, cond, steps):  # used by open_loop_rollout_loss
        zf = z_init
        for _ in range(steps):
            zf = torch.cat([zf, self.forward(zf, cond)[:, -1:, :]], dim=1)
        return zf


class GRUPredictor(_RollMixin, nn.Module):
    def __init__(self, d, hidden=384, layers=2, dropout=0.1, **_):
        super().__init__()
        self.rnn = nn.GRU(d, hidden, layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, d)
        self.max_seq_len = MAX_SEQ

    def forward(self, z, cond=None):
        out, _ = self.rnn(z)
        return self.head(out)


class LSTMPredictor(_RollMixin, nn.Module):
    def __init__(self, d, hidden=384, layers=2, dropout=0.1, **_):
        super().__init__()
        self.rnn = nn.LSTM(d, hidden, layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, d)
        self.max_seq_len = MAX_SEQ

    def forward(self, z, cond=None):
        out, _ = self.rnn(z)
        return self.head(out)


class DelayMLP(_RollMixin, nn.Module):
    """Explicit Takens delay embedding: predict z(t+1) from the window z(t-W+1..t)."""
    def __init__(self, d, W=8, hidden=512, **_):
        super().__init__()
        self.W = W
        self.net = nn.Sequential(
            nn.Linear(W * d, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, d),
        )
        self.max_seq_len = MAX_SEQ

    def forward(self, z, cond=None):
        B, T, d = z.shape
        zp = torch.cat([z[:, :1].expand(B, self.W - 1, d), z], dim=1)   # left-pad
        win = zp.unfold(1, self.W, 1).reshape(B, T, d * self.W)         # (B,T,d*W)
        return self.net(win)


def build_model(name, d, device):
    if name in ("tf_cond3", "tf_cond0", "tf_cond0_big"):
        cond_dim = 3 if name == "tf_cond3" else 0
        big = name == "tf_cond0_big"
        m = AutoregressivePredictor(
            latent_dim=d, cond_dim=cond_dim,
            hidden_dim=512 if big else PD["hidden_dim"],
            depth=10 if big else PD["depth"],
            heads=16, mlp_ratio=PD["mlp_ratio"], dropout=PD["dropout"], max_seq_len=MAX_SEQ,
        )
        out_lin = m.out_proj[0]
        m.out_proj = nn.Sequential(out_lin, nn.Identity())   # --no-output-bn (B1 Test1)
        return m.to(device)
    if name == "gru":
        return GRUPredictor(d).to(device)
    if name == "lstm":
        return LSTMPredictor(d).to(device)
    if name == "delaymlp":
        return DelayMLP(d, W=8).to(device)
    raise ValueError(name)


# --------------------------- training (backprop models) ---------------------------
def train_model(model, dataset, iters, device, seed=0):
    loader = DataLoader(dataset, batch_size=PD["B"], num_workers=2,
                        collate_fn=collate, shuffle=False, drop_last=True)
    opt = AdamW(model.parameters(), lr=PD["lr"], betas=(0.9, 0.95), weight_decay=PD["weight_decay"])
    sch = LambdaLR(opt, lr_lambda=build_lr_lambda(iters, PD["warmup_frac"]))
    rng = np.random.default_rng(seed + 1)
    it = 0
    model.train()
    while it < iters:
        for b in loader:
            if it >= iters:
                break
            z = b["z"].to(device); cond = b["cond"].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z_hat = model(z, cond)
                L_tf = teacher_forced_prediction_loss(z, z_hat)
                start = int(rng.integers(0, max(1, PD["T"] - PD["H_roll"])))
                L_roll = open_loop_rollout_loss(model, z, cond, start_t=start, horizon=PD["H_roll"])
                L = L_tf + 0.5 * L_roll
            opt.zero_grad(set_to_none=True)
            L.backward()
            nn.utils.clip_grad_norm_(model.parameters(), PD["grad_clip"])
            opt.step(); sch.step(); it += 1
            if it % 2000 == 0:
                print(f"    iter {it}/{iters}  L={L.item():.4f} (tf={L_tf.item():.4f} roll={L_roll.item():.4f})", flush=True)
    model.eval()
    return model


# --------------------------- reservoir / ESN (no backprop) ---------------------------
def esn_fit_eval(ztr_norm, splits_norm, mean, std, probe, dns, device,
                 n_res=1200, spectral=0.95, leak=0.5, in_scale=0.6, ridge=1e-2, seed=0):
    rng = np.random.default_rng(seed)
    d = ztr_norm.shape[-1]
    Win = (rng.standard_normal((n_res, d)) * in_scale).astype(np.float32)
    Wr = rng.standard_normal((n_res, n_res)).astype(np.float32)
    Wr *= spectral / (np.abs(np.linalg.eigvals(Wr)).max() + 1e-9)

    def run_states(zseq):                          # zseq (T,d) -> states (T,n_res)
        h = np.zeros(n_res, np.float32); S = np.zeros((zseq.shape[0], n_res), np.float32)
        for t in range(zseq.shape[0]):
            h = (1 - leak) * h + leak * np.tanh(Win @ zseq[t] + Wr @ h)
            S[t] = h
        return S

    # collect teacher-forced (state_t, input_t) -> z_{t+1} over train trajectories
    X, Y = [], []
    for i in range(ztr_norm.shape[0]):
        S = run_states(ztr_norm[i])
        feat = np.concatenate([S[:-1], ztr_norm[i][:-1]], axis=1)
        X.append(feat); Y.append(ztr_norm[i][1:])
    X = np.concatenate(X); Y = np.concatenate(Y)
    A = X.T @ X + ridge * np.eye(X.shape[1])
    Wout = np.linalg.solve(A, X.T @ Y).astype(np.float32)     # (n_res+d, d)

    def free_run(zseed, steps):                    # zseed (Tseed,d) -> (Tseed+steps,d)
        h = np.zeros(n_res, np.float32); zf = list(zseed)
        for t in range(len(zseed)):
            h = (1 - leak) * h + leak * np.tanh(Win @ zseed[t] + Wr @ h)
        for _ in range(steps):
            nxt = np.concatenate([h, zf[-1]]) @ Wout
            zf.append(nxt.astype(np.float32))
            h = (1 - leak) * h + leak * np.tanh(Win @ zf[-1] + Wr @ h)
        return np.array(zf)

    res = {}
    for split, (zsplit_norm, imp, di) in splits_norm.items():
        for mode in ("markov", "fullctx"):
            yp, yt = [], []
            for i in range(zsplit_norm.shape[0]):
                if mode == "markov":
                    seed = zsplit_norm[i, imp[i]:imp[i] + 1]; k = H
                else:
                    lo = max(0, imp[i] + 1 - MAX_SEQ); seed = zsplit_norm[i, lo:imp[i] + 1]; k = len(seed) - 1 + H
                roll = free_run(seed, HMAX)
                z_raw = roll[k] * std + mean
                yp.append(float(cr.apply_probe(z_raw[None], probe)[0]))
                yt.append(float(dns[f"{split}_wake_enstrophy"][di[i], imp[i] + H]))
            res[(split, mode)] = _r2(np.array(yp), np.array(yt))
    return res


# --------------------------- rollout eval (backprop models) ---------------------------
def _r2(yp, yt):
    ss_res = float(((yp - yt) ** 2).sum()); ss_tot = float(((yt - yt.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


@torch.no_grad()
def free_run_torch(model, z_seed, cond, steps):
    zf = z_seed
    for _ in range(steps):
        ctx = zf[:, -MAX_SEQ:, :]
        zh = model(ctx, cond)[:, -1:, :]
        zf = torch.cat([zf, zh], dim=1)
    return zf


@torch.no_grad()
def eval_model(model, splits_norm, mean_t, std_t, probe, dns, device, cond_dim):
    res = {}
    mean = mean_t.cpu().numpy(); std = std_t.cpu().numpy()
    for split, (z_norm, imp, di) in splits_norm.items():
        zt = torch.from_numpy(z_norm).to(device)
        n = zt.shape[0]
        for mode in ("markov", "fullctx"):
            yp, yt = [], []
            for i in range(n):
                cond = torch.zeros(1, cond_dim, device=device) if cond_dim else torch.zeros(1, 0, device=device)
                if cond_dim == 3:
                    cond = torch.tensor(_c_of[split][i], device=device).float().reshape(1, 3)
                if mode == "markov":
                    seed = zt[i, imp[i]:imp[i] + 1].unsqueeze(0); k = H
                else:
                    lo = max(0, imp[i] + 1 - MAX_SEQ); seed = zt[i, lo:imp[i] + 1].unsqueeze(0); k = seed.shape[1] - 1 + H
                roll = free_run_torch(model, seed, cond, HMAX)
                z_raw = roll[0, k].cpu().numpy() * std + mean
                yp.append(float(cr.apply_probe(z_raw[None], probe)[0]))
                yt.append(float(dns[f"{split}_wake_enstrophy"][di[i], imp[i] + H]))
            res[(split, mode)] = _r2(np.array(yp), np.array(yt))
    return res


_c_of = {}


def load_split_norm(split, mean, std, dns):
    b = np.load(LATDIR / f"{split}.npz", allow_pickle=True)
    z = b["z_full"].astype(np.float32)
    cid = cr._get(b, "case_ids", "case_id"); ei = cr._get(b, "encounter_indices", "encounter_index")
    imp = b["impact_frame"].astype(int)
    di = cr.match_index(cid, ei, dns[f"{split}_case_id"], dns[f"{split}_encounter_index"])
    keep = di >= 0
    z = z[keep]; imp = imp[keep]; di = di[keep]
    _c_of[split] = np.c_[b["G"], b["D"], b["Y"]][keep].astype(np.float32)
    z_norm = (z - mean) / std
    return z_norm.astype(np.float32), imp, di


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--models", nargs="+",
                    default=["tf_cond3", "tf_cond0", "tf_cond0_big", "gru", "lstm", "delaymlp", "esn"])
    args = ap.parse_args()
    device = require_rtx6000(gpu_index=args.gpu)
    print("device", device, torch.cuda.get_device_name(device.index))

    dns = np.load(cr.DNS_METRICS_PATH, allow_pickle=True)
    probe = cr.fit_probes(LATDIR, dns)["wake_enstrophy"]

    tr = np.load(LATDIR / "train.npz", allow_pickle=True)
    z_full = tr["z_full"].astype(np.float32)
    d = z_full.shape[-1]
    dataset = LatentSubTrajectoryDataset(
        z_full=z_full, G=tr["G"], D=tr["D"], Y=tr["Y"], impact_frame=tr["impact_frame"],
        T=PD["T"], impact_aware_fraction=PD["impact_aware_fraction"], seed=0, n_samples_per_epoch=PD["B"] * 256)
    mean = dataset.mean; std = dataset.std
    mean_t = torch.from_numpy(mean).to(device); std_t = torch.from_numpy(std).to(device)

    splits_norm = {s: load_split_norm(s, mean, std, dns) for s in ("test_b", "test_c")}
    ztr_norm = ((z_full - mean) / std).astype(np.float32)

    rows = []
    for name in args.models:
        print(f"\n=== {name} ===", flush=True)
        if name == "esn":
            res = esn_fit_eval(ztr_norm, splits_norm, mean, std, probe, dns, device)
            cond_dim = 0
        else:
            model = build_model(name, d, device)
            np_params = sum(p.numel() for p in model.parameters())
            print(f"  params={np_params/1e6:.2f}M", flush=True)
            model = train_model(model, dataset, args.iters, device)
            cond_dim = 3 if name == "tf_cond3" else 0
            res = eval_model(model, splits_norm, mean_t, std_t, probe, dns, device, cond_dim)
        for (split, mode), r2 in res.items():
            rows.append(dict(model=name, cond=("c" if name == "tf_cond3" else "none"),
                             split=split, mode=mode, horizon=H, wake_r2=round(r2, 4)))
            print(f"  {split:7s} {mode:8s} wake R2@{H} = {r2:+.3f}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {OUT}")
    print("\nReference: conditioned z_markov wake R2@16 = 0.449 (test_b), 0.325 (test_c); z_dns ceiling 0.754.")


if __name__ == "__main__":
    main()
