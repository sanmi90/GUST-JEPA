"""Context-length (T) sweep for the unconditioned LSTM closure predictor.

Varies the training sub-trajectory length T AND the eval rollout context cap
(MAX_SEQ = T) together, so it is a clean "how much latent history does the
unconditioned LSTM need?" test. Fixed architecture (default 384-hidden / 2-layer
unless overridden). 1 seed per T (coarse), then 3 seeds on the best T.

Scored with the canonical wake-enstrophy closure probe on the frozen production
encoder (same self-contained eval as session27_uncond_sweep; ~0.08 below the
canonical scale but identical for every T).

Usage: python scripts/session27_lstm_T_sweep.py --gpu 1 --hidden 384 --layers 2
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import session27_uncond_sweep as S  # noqa: E402
import exp_closure_r2 as cr  # noqa: E402
from train_baseline_predictor import LatentSubTrajectoryDataset, PROTOCOL_DEFAULTS as PD  # noqa: E402

OUT = REPO / "outputs/session27/lstm_T_sweep.csv"
T_GRID = [8, 16, 24, 32, 48, 64]


def run_T(T, hidden, layers, iters, seed, mean, std, mean_t, std_t, splits, probe, dns, zf, device):
    PD["T"] = T
    PD["H_roll"] = min(8, max(2, T - 2))        # open_loop_rollout_loss needs start_t+H_roll < T
    S.MAX_SEQ = T                               # eval rollout context cap = training context
    dataset = LatentSubTrajectoryDataset(
        z_full=zf, G=dns_dummy_G, D=dns_dummy_D, Y=dns_dummy_Y, impact_frame=dns_dummy_imp,
        T=T, impact_aware_fraction=PD["impact_aware_fraction"], seed=seed,
        n_samples_per_epoch=PD["B"] * 256)
    torch.manual_seed(seed); np.random.seed(seed)
    model = S.LSTMPredictor(dataset.d, hidden=hidden, layers=layers, dropout=0.1).to(device)
    npar = sum(p.numel() for p in model.parameters())
    model = S.train_model(model, dataset, iters, device, seed=seed)
    res = S.eval_model(model, splits, mean_t, std_t, probe, dns, device, cond_dim=0)
    return npar, res


def main():
    global dns_dummy_G, dns_dummy_D, dns_dummy_Y, dns_dummy_imp
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--iters", type=int, default=15000)
    ap.add_argument("--iters-best", type=int, default=20000)
    args = ap.parse_args()
    device = S.require_rtx6000(gpu_index=args.gpu)
    print("device", device, torch.cuda.get_device_name(device.index), flush=True)

    dns = np.load(cr.DNS_METRICS_PATH, allow_pickle=True)
    probe = cr.fit_probes(S.LATDIR, dns)["wake_enstrophy"]
    tr = np.load(S.LATDIR / "train.npz", allow_pickle=True)
    zf = tr["z_full"].astype(np.float32)
    dns_dummy_G, dns_dummy_D, dns_dummy_Y, dns_dummy_imp = tr["G"], tr["D"], tr["Y"], tr["impact_frame"]
    flat = zf.reshape(-1, zf.shape[-1]); mean = flat.mean(0).astype(np.float32); std = flat.std(0).clip(1e-6).astype(np.float32)
    mean_t = torch.from_numpy(mean).to(device); std_t = torch.from_numpy(std).to(device)
    splits = {s: S.load_split_norm(s, mean, std, dns) for s in ("test_b", "test_c")}

    rows = []
    print(f"\n=== T context sweep, LSTM h={args.hidden} l={args.layers} (1 seed, {args.iters} iters) ===", flush=True)
    for T in T_GRID:
        npar, res = run_T(T, args.hidden, args.layers, args.iters, 0, mean, std, mean_t, std_t, splits, probe, dns, zf, device)
        tbm, tbf = res[("test_b", "markov")], res[("test_b", "fullctx")]
        tcm, tcf = res[("test_c", "markov")], res[("test_c", "fullctx")]
        rows.append(dict(stage=1, T=T, hidden=args.hidden, layers=args.layers, seed=0, params=npar,
                         tb_markov=round(tbm, 4), tb_fullctx=round(tbf, 4),
                         tc_markov=round(tcm, 4), tc_fullctx=round(tcf, 4), tb_best=round(max(tbm, tbf), 4)))
        print(f"  T={T:3d}  test_b markov/fullctx = {tbm:+.3f}/{tbf:+.3f}   test_c = {tcm:+.3f}/{tcf:+.3f}", flush=True)

    best = max([r for r in rows if r["stage"] == 1], key=lambda r: r["tb_best"])
    print(f"\n=== best T={best['T']} -> 3 seeds @ {args.iters_best} iters ===", flush=True)
    acc = {k: [] for k in ("tb_markov", "tb_fullctx", "tc_markov", "tc_fullctx")}
    for seed in (0, 1, 2):
        npar, res = run_T(best["T"], args.hidden, args.layers, args.iters_best, seed, mean, std, mean_t, std_t, splits, probe, dns, zf, device)
        acc["tb_markov"].append(res[("test_b", "markov")]); acc["tb_fullctx"].append(res[("test_b", "fullctx")])
        acc["tc_markov"].append(res[("test_c", "markov")]); acc["tc_fullctx"].append(res[("test_c", "fullctx")])
        rows.append(dict(stage=2, T=best["T"], hidden=args.hidden, layers=args.layers, seed=seed, params=npar,
                         tb_markov=round(res[("test_b", "markov")], 4), tb_fullctx=round(res[("test_b", "fullctx")], 4),
                         tc_markov=round(res[("test_c", "markov")], 4), tc_fullctx=round(res[("test_c", "fullctx")], 4),
                         tb_best=round(max(res[("test_b", "markov")], res[("test_b", "fullctx")]), 4)))
        print(f"  seed {seed}: test_b markov/fullctx = {res[('test_b','markov')]:+.3f}/{res[('test_b','fullctx')]:+.3f}", flush=True)
    print(f"\n  best T={best['T']} (3 seeds): "
          f"tb_markov {np.mean(acc['tb_markov']):+.3f}+/-{np.std(acc['tb_markov']):.3f}  "
          f"tb_fullctx {np.mean(acc['tb_fullctx']):+.3f}+/-{np.std(acc['tb_fullctx']):.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
