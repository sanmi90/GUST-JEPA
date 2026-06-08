"""Apply the BEST unconditioned LSTM to each encoder family (matched predictor).

Fairness check for the predictability claim: train the same tuned LSTM-no-c on
JEPA, POD and Fukami frozen latents and compare wake-enstrophy closure R^2 @16.
If JEPA still dominates POD/Fukami, the forward-closure advantage is a property of
the representation, not of the (transformer) predictor or the conditioning.

Same self-contained harness/eval as session27_uncond_sweep (consistent scale
across families; the JEPA row re-confirms the best config). Pass the winning
config from the hidden x layers and T sweeps.

Usage: python scripts/session27_lstm_crossfamily.py --hidden 256 --layers 3 --T 24 --gpu 0
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

FAMILIES = [
    ("jepa", "latents_jepa_d64_test1_noBN"),
    ("pod", "latents_pod_d64_noBN"),
    ("fukami", "latents_fukami_d64_noBN"),
]
OUT = REPO / "outputs/session27/lstm_crossfamily.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--T", type=int, default=24)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--device", type=str, default=None,
                    help="explicit torch device (e.g. cuda:0 for an L40S); bypasses require_rtx6000")
    args = ap.parse_args()
    device = torch.device(args.device) if args.device else S.require_rtx6000(gpu_index=args.gpu)
    print(f"device {device} {torch.cuda.get_device_name(device.index)}  "
          f"LSTM h={args.hidden} l={args.layers} T={args.T}", flush=True)
    dns = np.load(cr.DNS_METRICS_PATH, allow_pickle=True)
    PD["T"] = args.T
    PD["H_roll"] = min(8, max(2, args.T - 2))
    S.MAX_SEQ = args.T

    rows = []
    for fam, latname in FAMILIES:
        S.LATDIR = REPO / "outputs/session18/exp_b1" / latname
        if not (S.LATDIR / "train.npz").exists():
            print(f"  SKIP {fam}: {S.LATDIR} missing"); continue
        probe = cr.fit_probes(S.LATDIR, dns)["wake_enstrophy"]
        tr = np.load(S.LATDIR / "train.npz", allow_pickle=True)
        zf = tr["z_full"].astype(np.float32)
        dataset = LatentSubTrajectoryDataset(
            z_full=zf, G=tr["G"], D=tr["D"], Y=tr["Y"], impact_frame=tr["impact_frame"],
            T=args.T, impact_aware_fraction=PD["impact_aware_fraction"], seed=0,
            n_samples_per_epoch=PD["B"] * 256)
        mean, std = dataset.mean, dataset.std
        mean_t = torch.from_numpy(mean).to(device); std_t = torch.from_numpy(std).to(device)
        splits = {s: S.load_split_norm(s, mean, std, dns) for s in ("test_b", "test_c")}
        acc = {k: [] for k in ("tb_markov", "tb_fullctx", "tc_markov", "tc_fullctx")}
        for seed in args.seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            model = S.LSTMPredictor(dataset.d, hidden=args.hidden, layers=args.layers, dropout=0.1).to(device)
            model = S.train_model(model, dataset, args.iters, device, seed=seed)
            res = S.eval_model(model, splits, mean_t, std_t, probe, dns, device, cond_dim=0)
            acc["tb_markov"].append(res[("test_b", "markov")]); acc["tb_fullctx"].append(res[("test_b", "fullctx")])
            acc["tc_markov"].append(res[("test_c", "markov")]); acc["tc_fullctx"].append(res[("test_c", "fullctx")])
            rows.append(dict(family=fam, seed=seed, hidden=args.hidden, layers=args.layers, T=args.T,
                             tb_markov=round(res[("test_b", "markov")], 4), tb_fullctx=round(res[("test_b", "fullctx")], 4),
                             tc_markov=round(res[("test_c", "markov")], 4), tc_fullctx=round(res[("test_c", "fullctx")], 4)))
        print(f"  {fam:7s}  test_b markov {np.mean(acc['tb_markov']):+.3f}+/-{np.std(acc['tb_markov']):.3f}  "
              f"fullctx {np.mean(acc['tb_fullctx']):+.3f}+/-{np.std(acc['tb_fullctx']):.3f}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {OUT}")
    print("If JEPA >> POD/Fukami here, the forward-closure advantage is in the representation, "
          "robust to predictor + conditioning.")


if __name__ == "__main__":
    main()
