"""Roll a retrained JEPA's OWN predictor and score wake-enstrophy closure.

Encodes the JEPA's own latents (its encoder), loads its co-trained predictor
(transformer or LSTM) from the checkpoint, rolls it from impact in markov and
full-context (history) seed modes, and reports held-out wake R^2@16 via the
canonical probe. Latents are fed RAW (the JEPA predictor was trained on the
encoder's BatchNorm-normalised output, not z-scored), so mean=0/std=1.

Usage:
  python scripts/session27_eval_own_predictor.py --jepa-ckpt <ckpt> \
     --ptype lstm --hidden 256 --layers 3 --tag jepa_noc_lstm --device cuda:1
  python scripts/session27_eval_own_predictor.py --jepa-ckpt <ckpt> \
     --ptype transformer --tag jepa_noc_tf --device cuda:1
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "session20"))
import session27_uncond_sweep as S  # noqa: E402
import exp_closure_r2 as cr  # noqa: E402
from src.models.predictor import AutoregressivePredictor, LSTMLatentPredictor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jepa-ckpt", required=True)
    ap.add_argument("--ptype", choices=["transformer", "lstm"], required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    args = ap.parse_args()
    device = torch.device(args.device)
    S.MAX_SEQ = 32

    latdir = REPO / f"outputs/session27/latents_own_{args.tag}"
    if not (latdir / "test_b.npz").exists():
        print(f"[own-eval] encoding latents -> {latdir}", flush=True)
        subprocess.run([sys.executable, str(REPO / "scripts/session18/encode_baseline_latents.py"),
                        "--baseline", "jepa", "--d", "64", "--checkpoint", args.jepa_ckpt,
                        "--output-dir", str(latdir), "--device", args.device], check=True)
    S.LATDIR = latdir

    dns = np.load(cr.DNS_METRICS_PATH, allow_pickle=True)
    probe = cr.fit_probes(S.LATDIR, dns)["wake_enstrophy"]
    d = 64
    mean = np.zeros(d, np.float32); std = np.ones(d, np.float32)            # feed RAW latents
    mean_t = torch.from_numpy(mean).to(device); std_t = torch.from_numpy(std).to(device)
    splits = {s: S.load_split_norm(s, mean, std, dns) for s in ("test_b", "test_c")}

    ck = torch.load(args.jepa_ckpt, map_location="cpu", weights_only=False)
    sd = ck["jepa_state_dict"]
    psd = {k[len("predictor."):]: v for k, v in sd.items() if k.startswith("predictor.")}
    if args.ptype == "lstm":
        pred = LSTMLatentPredictor(64, cond_dim=0, hidden_dim=args.hidden,
                                   num_layers=args.layers, max_seq_len=32)
    else:
        pred = AutoregressivePredictor(latent_dim=64, cond_dim=0, max_seq_len=32)
    pred.load_state_dict(psd, strict=True)
    pred.eval().to(device)
    n_params = sum(p.numel() for p in pred.parameters())
    print(f"[own-eval] {args.tag} {args.ptype} own predictor loaded ({n_params/1e6:.2f}M)", flush=True)

    res = S.eval_model(pred, splits, mean_t, std_t, probe, dns, device, cond_dim=0)
    print(f"\n=== {args.tag}: own-predictor wake R2@16 ===")
    for (split, mode), r2 in res.items():
        print(f"  {split:7s} {mode:8s} {r2:+.3f}")
    print("Reference: conditioned (separate closure predictor) test_b 0.449; "
          "transformer-no-c closure-predictor markov 0.038.")


if __name__ == "__main__":
    main()
