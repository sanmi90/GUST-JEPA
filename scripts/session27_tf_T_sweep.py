"""Context-length (T) sweep for the UNCONDITIONED TRANSFORMER closure predictor.

Mirror of session27_lstm_T_sweep.py but for the cond-0 transformer
(AutoregressivePredictor, no-output-bn). The transformer is rebuilt with
max_seq_len = T each time (RoPE positions are bounded), and the training
sub-trajectory length and eval rollout context cap both equal T. 1 seed per T,
then best T at 3 seeds. Wake-enstrophy closure R^2 @16 on the frozen production
encoder (same self-contained eval as session27_uncond_sweep).

Usage: python scripts/session27_tf_T_sweep.py --gpu 0
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

OUT = REPO / "outputs/session27/tf_T_sweep.csv"
T_GRID = [8, 16, 24, 32, 48, 64]


def run_T(T, iters, seed, zf, mean, std, mean_t, std_t, splits, probe, dns, device):
    PD["T"] = T
    PD["H_roll"] = min(8, max(2, T - 2))
    S.MAX_SEQ = T                                       # rebuilds tf with max_seq_len=T + caps eval context
    dataset = LatentSubTrajectoryDataset(
        z_full=zf, G=Gv, D=Dv, Y=Yv, impact_frame=impv, T=T,
        impact_aware_fraction=PD["impact_aware_fraction"], seed=seed,
        n_samples_per_epoch=PD["B"] * 256)
    torch.manual_seed(seed); np.random.seed(seed)
    model = S.build_model("tf_cond0", dataset.d, device)   # uses S.MAX_SEQ=T for max_seq_len
    npar = sum(p.numel() for p in model.parameters())
    model = S.train_model(model, dataset, iters, device, seed=seed)
    res = S.eval_model(model, splits, mean_t, std_t, probe, dns, device, cond_dim=0)
    return npar, res


def main():
    global Gv, Dv, Yv, impv
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--iters", type=int, default=15000)
    ap.add_argument("--iters-best", type=int, default=20000)
    args = ap.parse_args()
    device = S.require_rtx6000(gpu_index=args.gpu)
    print("device", device, torch.cuda.get_device_name(device.index), flush=True)

    dns = np.load(cr.DNS_METRICS_PATH, allow_pickle=True)
    probe = cr.fit_probes(S.LATDIR, dns)["wake_enstrophy"]
    tr = np.load(S.LATDIR / "train.npz", allow_pickle=True)
    zf = tr["z_full"].astype(np.float32)
    Gv, Dv, Yv, impv = tr["G"], tr["D"], tr["Y"], tr["impact_frame"]
    flat = zf.reshape(-1, zf.shape[-1]); mean = flat.mean(0).astype(np.float32); std = flat.std(0).clip(1e-6).astype(np.float32)
    mean_t = torch.from_numpy(mean).to(device); std_t = torch.from_numpy(std).to(device)
    splits = {s: S.load_split_norm(s, mean, std, dns) for s in ("test_b", "test_c")}

    rows = []
    print(f"\n=== TRANSFORMER (no c) T context sweep (1 seed, {args.iters} iters) ===", flush=True)
    for T in T_GRID:
        npar, res = run_T(T, args.iters, 0, zf, mean, std, mean_t, std_t, splits, probe, dns, device)
        tbm, tbf = res[("test_b", "markov")], res[("test_b", "fullctx")]
        tcm, tcf = res[("test_c", "markov")], res[("test_c", "fullctx")]
        rows.append(dict(stage=1, T=T, seed=0, params=npar, tb_markov=round(tbm, 4), tb_fullctx=round(tbf, 4),
                         tc_markov=round(tcm, 4), tc_fullctx=round(tcf, 4), tb_best=round(max(tbm, tbf), 4)))
        print(f"  T={T:3d} ({npar/1e6:.1f}M)  test_b markov/fullctx = {tbm:+.3f}/{tbf:+.3f}   test_c = {tcm:+.3f}/{tcf:+.3f}", flush=True)

    best = max([r for r in rows if r["stage"] == 1], key=lambda r: r["tb_best"])
    print(f"\n=== best T={best['T']} -> 3 seeds @ {args.iters_best} iters ===", flush=True)
    acc = {k: [] for k in ("tb_markov", "tb_fullctx", "tc_markov", "tc_fullctx")}
    for seed in (0, 1, 2):
        npar, res = run_T(best["T"], args.iters_best, seed, zf, mean, std, mean_t, std_t, splits, probe, dns, device)
        acc["tb_markov"].append(res[("test_b", "markov")]); acc["tb_fullctx"].append(res[("test_b", "fullctx")])
        acc["tc_markov"].append(res[("test_c", "markov")]); acc["tc_fullctx"].append(res[("test_c", "fullctx")])
        rows.append(dict(stage=2, T=best["T"], seed=seed, params=npar,
                         tb_markov=round(res[("test_b", "markov")], 4), tb_fullctx=round(res[("test_b", "fullctx")], 4),
                         tc_markov=round(res[("test_c", "markov")], 4), tc_fullctx=round(res[("test_c", "fullctx")], 4),
                         tb_best=round(max(res[("test_b", "markov")], res[("test_b", "fullctx")]), 4)))
        print(f"  seed {seed}: test_b markov/fullctx = {res[('test_b','markov')]:+.3f}/{res[('test_b','fullctx')]:+.3f}", flush=True)
    print(f"\n  best T={best['T']} (3 seeds): tb_markov {np.mean(acc['tb_markov']):+.3f}+/-{np.std(acc['tb_markov']):.3f}  "
          f"tb_fullctx {np.mean(acc['tb_fullctx']):+.3f}+/-{np.std(acc['tb_fullctx']):.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
