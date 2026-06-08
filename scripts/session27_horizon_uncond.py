"""Forward-closure R^2 vs horizon for the UNCONDITIONED JEPAs, each rolling its
OWN predictor (no c). Rolls once to H=26 per encounter and probes wake-enstrophy
R^2 at H in {1,2,4,8,12,16,20,24} on test_b, markov + fullctx, vs the z_dns ceiling.
Tier-1 headline figure (unconditioned analogue of fig_horizon_sweep).

Usage: python scripts/session27_horizon_uncond.py --device cuda:2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "session20"))
import session27_uncond_sweep as S  # noqa: E402
import exp_closure_r2 as cr  # noqa: E402
from src.models.predictor import AutoregressivePredictor, LSTMLatentPredictor  # noqa: E402

HZ = [1, 2, 4, 8, 12, 16, 20, 24]
HMAX_MAX = max(HZ) + 2
S.MAX_SEQ = 32
SPLIT = "test_b"


def r2(yp, yt):
    yp, yt = np.asarray(yp), np.asarray(yt)
    return 1.0 - float(((yp - yt) ** 2).sum()) / max(float(((yt - yt.mean()) ** 2).sum()), 1e-12)


def load_pred(tag, device):
    ck = torch.load(REPO / f"outputs/session27/JEPA_d64_noc_{tag}/checkpoint_iter020000.pt",
                    map_location="cpu", weights_only=False)
    a = ck["args"]; ptype = a.get("predictor_type", "transformer")
    psd = {k[len("predictor."):]: v for k, v in ck["jepa_state_dict"].items() if k.startswith("predictor.")}
    if ptype == "lstm":
        p = LSTMLatentPredictor(64, cond_dim=0, hidden_dim=int(a["predictor_hidden"]),
                                num_layers=int(a["predictor_layers"]), max_seq_len=32)
    else:
        p = AutoregressivePredictor(latent_dim=64, cond_dim=0, max_seq_len=32)
    p.load_state_dict(psd, strict=True); p.eval().to(device)
    return p, ptype


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:2")
    args = ap.parse_args()
    device = torch.device(args.device)
    dns = np.load(cr.DNS_METRICS_PATH, allow_pickle=True)
    mean = np.zeros(64, np.float32); std = np.ones(64, np.float32)            # feed RAW latents
    cond0 = torch.zeros(1, 0, device=device)
    curves = {}

    for tag in ("tf", "lstm"):
        latdir = REPO / f"outputs/session27/latents_own_jepa_noc_{tag}"
        S.LATDIR = latdir
        probe = cr.fit_probes(latdir, dns)["wake_enstrophy"]
        pred, ptype = load_pred(tag, device)
        z_norm, imp, di = S.load_split_norm(SPLIT, mean, std, dns)
        zt = torch.from_numpy(z_norm).to(device)
        print(f"[{tag}] ptype={ptype} n={len(imp)}", flush=True)
        # z_dns ceiling once
        cyt = {H: [] for H in HZ}; cyp = {H: [] for H in HZ}
        for mode in ("markov", "fullctx"):
            rolls = []
            with torch.no_grad():
                for i in range(len(imp)):
                    if mode == "markov":
                        seed = zt[i, imp[i]:imp[i] + 1].unsqueeze(0)
                    else:
                        lo = max(0, imp[i] + 1 - S.MAX_SEQ); seed = zt[i, lo:imp[i] + 1].unsqueeze(0)
                    rolls.append((S.free_run_torch(pred, seed, cond0, HMAX_MAX), seed.shape[1]))
            for H in HZ:
                yp, yt = [], []
                for i, (roll, sl) in enumerate(rolls):
                    k = H if mode == "markov" else sl - 1 + H
                    yp.append(float(cr.apply_probe(roll[0, k].cpu().numpy()[None], probe)[0]))
                    yt.append(float(dns[f"{SPLIT}_wake_enstrophy"][di[i], imp[i] + H]))
                    if mode == "markov":  # ceiling uses true latent at imp+H (compute once)
                        cyp[H].append(float(cr.apply_probe(zt[i, imp[i] + H].cpu().numpy()[None], probe)[0]))
                        cyt[H].append(float(dns[f"{SPLIT}_wake_enstrophy"][di[i], imp[i] + H]))
                curves[(tag, mode)] = curves.get((tag, mode), {}); curves[(tag, mode)][H] = r2(yp, yt)
        curves[(tag, "ceiling")] = {H: r2(cyp[H], cyt[H]) for H in HZ}

    print("\n=== wake R2 vs horizon (test_b, own predictor) ===")
    for key in sorted(curves, key=lambda k: (k[0], k[1])):
        print(f"  {key[0]:5s} {key[1]:8s} " + "  ".join(f"H{H}:{curves[key][H]:+.2f}" for H in HZ))

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    sty = {("tf", "markov"): ("C0", "-", "tf-no-c markov"),
           ("tf", "fullctx"): ("C0", "--", "tf-no-c fullctx"),
           ("lstm", "markov"): ("C1", "-", "lstm-no-c markov"),
           ("lstm", "fullctx"): ("C1", "--", "lstm-no-c fullctx"),
           ("tf", "ceiling"): ("C0", ":", "tf z_dns ceiling"),
           ("lstm", "ceiling"): ("C1", ":", "lstm z_dns ceiling")}
    for key, (c, ls, lab) in sty.items():
        ys = [curves[key][H] for H in HZ]
        ax.plot(HZ, ys, ls, color=c, label=lab, lw=1.8 if "ceiling" not in lab else 1.0,
                alpha=1.0 if "ceiling" not in lab else 0.5)
    ax.axhline(0, color="k", lw=0.6, alpha=0.4)
    ax.set_xlabel("forecast horizon H (frames after impact)")
    ax.set_ylabel("wake-enstrophy closure $R^2$ (test_b)")
    ax.set_title("Unconditioned JEPA: forward closure vs horizon (own predictor)", fontsize=9)
    ax.legend(fontsize=6.5, ncol=2); ax.grid(alpha=0.25)
    fig.tight_layout()
    out = REPO / "outputs/session27/horizon_uncond.pdf"
    fig.savefig(out, bbox_inches="tight"); print(f"\n[horizon] figure -> {out}")


if __name__ == "__main__":
    main()
