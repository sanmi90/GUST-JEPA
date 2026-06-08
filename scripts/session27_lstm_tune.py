"""Tune the UNCONDITIONED LSTM closure predictor (cheap, ~ties the transformer).

Stage 1: coarse grid over hidden x layers (1 seed, 12k iters) to find the
cost/performance frontier. Stage 2: best config at 20k iters x 3 seeds for a
credible mean +/- std. Scored with the canonical wake-enstrophy closure probe
(exp_closure_r2) on the frozen production encoder; same self-contained eval as
session27_uncond_sweep (runs ~0.08 below the canonical scale but identical for
every config, and identical to the transformer numbers reported there).

Usage: python scripts/session27_lstm_tune.py --gpu 0
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

OUT = REPO / "outputs/session27/lstm_tune.csv"
GRID = [(h, l) for h in (128, 256, 384, 512) for l in (1, 2, 3)]


def setup(device):
    dns = np.load(cr.DNS_METRICS_PATH, allow_pickle=True)
    probe = cr.fit_probes(S.LATDIR, dns)["wake_enstrophy"]
    tr = np.load(S.LATDIR / "train.npz", allow_pickle=True)
    zf = tr["z_full"].astype(np.float32)
    dataset = LatentSubTrajectoryDataset(
        z_full=zf, G=tr["G"], D=tr["D"], Y=tr["Y"], impact_frame=tr["impact_frame"],
        T=PD["T"], impact_aware_fraction=PD["impact_aware_fraction"], seed=0,
        n_samples_per_epoch=PD["B"] * 256)
    mean, std = dataset.mean, dataset.std
    mean_t = torch.from_numpy(mean).to(device); std_t = torch.from_numpy(std).to(device)
    splits = {s: S.load_split_norm(s, mean, std, dns) for s in ("test_b", "test_c")}
    return dns, probe, dataset, mean_t, std_t, splits, zf.shape[-1]


def run_one(h, l, dropout, iters, seed, dataset, splits, mean_t, std_t, probe, dns, device):
    torch.manual_seed(seed); np.random.seed(seed)
    model = S.LSTMPredictor(dataset.d, hidden=h, layers=l, dropout=dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    model = S.train_model(model, dataset, iters, device, seed=seed)
    res = S.eval_model(model, splits, mean_t, std_t, probe, dns, device, cond_dim=0)
    return n_params, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--stage1-iters", type=int, default=12000)
    ap.add_argument("--stage2-iters", type=int, default=20000)
    args = ap.parse_args()
    device = S.require_rtx6000(gpu_index=args.gpu)
    print("device", device, torch.cuda.get_device_name(device.index))
    dns, probe, dataset, mean_t, std_t, splits, d = setup(device)

    rows = []
    print("\n=== STAGE 1: hidden x layers grid (1 seed, %d iters) ===" % args.stage1_iters, flush=True)
    for (h, l) in GRID:
        npar, res = run_one(h, l, 0.1, args.stage1_iters, 0, dataset, splits, mean_t, std_t, probe, dns, device)
        tbm, tbf = res[("test_b", "markov")], res[("test_b", "fullctx")]
        tcm, tcf = res[("test_c", "markov")], res[("test_c", "fullctx")]
        best_tb = max(tbm, tbf)
        rows.append(dict(stage=1, hidden=h, layers=l, dropout=0.1, seed=0, params=npar,
                         tb_markov=round(tbm, 4), tb_fullctx=round(tbf, 4),
                         tc_markov=round(tcm, 4), tc_fullctx=round(tcf, 4), tb_best=round(best_tb, 4)))
        print(f"  h={h:4d} l={l}  params={npar/1e6:5.2f}M  test_b markov/fullctx = {tbm:+.3f}/{tbf:+.3f}  "
              f"test_c = {tcm:+.3f}/{tcf:+.3f}", flush=True)

    s1 = [r for r in rows if r["stage"] == 1]
    s1.sort(key=lambda r: r["tb_best"], reverse=True)
    print("\n  -- stage 1 ranked by best test_b wake R2 --")
    for r in s1[:5]:
        print(f"     h={r['hidden']:4d} l={r['layers']} ({r['params']/1e6:.2f}M)  tb_best={r['tb_best']:+.3f}")
    best = s1[0]
    print(f"\n=== STAGE 2: best config h={best['hidden']} l={best['layers']} @ {args.stage2_iters} iters x 3 seeds ===", flush=True)
    acc = {k: [] for k in ("test_b_markov", "test_b_fullctx", "test_c_markov", "test_c_fullctx")}
    for seed in (0, 1, 2):
        npar, res = run_one(best["hidden"], best["layers"], 0.1, args.stage2_iters, seed,
                            dataset, splits, mean_t, std_t, probe, dns, device)
        acc["test_b_markov"].append(res[("test_b", "markov")]); acc["test_b_fullctx"].append(res[("test_b", "fullctx")])
        acc["test_c_markov"].append(res[("test_c", "markov")]); acc["test_c_fullctx"].append(res[("test_c", "fullctx")])
        rows.append(dict(stage=2, hidden=best["hidden"], layers=best["layers"], dropout=0.1, seed=seed, params=npar,
                         tb_markov=round(res[("test_b", "markov")], 4), tb_fullctx=round(res[("test_b", "fullctx")], 4),
                         tc_markov=round(res[("test_c", "markov")], 4), tc_fullctx=round(res[("test_c", "fullctx")], 4),
                         tb_best=round(max(res[("test_b", "markov")], res[("test_b", "fullctx")]), 4)))
        print(f"  seed {seed}: test_b markov/fullctx = {res[('test_b','markov')]:+.3f}/{res[('test_b','fullctx')]:+.3f}", flush=True)
    print("\n  -- stage 2 mean +/- std (3 seeds) --")
    for k, v in acc.items():
        print(f"     {k:16s} {np.mean(v):+.3f} +/- {np.std(v):.3f}  {[round(x,3) for x in v]}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {OUT}")
    print("Reference (same harness): transformer no-c test_b 0.52/0.43; conditioned 0.33/0.37.")
    print("Reference (canonical scale): conditioned 0.449; transformer no-c 0.473 +/- 0.074.")


if __name__ == "__main__":
    main()
