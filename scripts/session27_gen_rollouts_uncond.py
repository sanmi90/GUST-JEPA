"""Generate full-trajectory rollouts for the UNCONDITIONED JEPAs in the production
rollouts format (z_dns, z_markov, z_full + metadata), so the existing Tier-2
analyses (topology, traces, drift, physical metrics) can consume them.

Protocol matches scripts/session18/eval_baseline_rollouts.py rollout_split:
  z_markov[:impact+1] = z_dns[:impact+1] (true latents up to impact),
  z_markov[impact+1:] = markov rollout seeded at the impact-frame latent.
  z_full[:impact+1]   = z_dns; z_full[impact+1:] = full-context rollout from the
                        pre-impact window.
Differences vs the B1 generator: own co-trained predictor (cond_dim=0, no z-score;
fed RAW latents = the encoder BatchNorm output), and LSTM support (free-run, no
markov-attention mask). Writes to outputs/session27/rollouts_noc_{tag}/ (NO clobber
of the production session18 rollouts).

Usage: python scripts/session27_gen_rollouts_uncond.py --device cuda:2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "session17"))
import exp2_rollout_decode_metrics as E2  # noqa: E402
from src.models.predictor import AutoregressivePredictor, LSTMLatentPredictor  # noqa: E402

T_TOTAL = 120


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


@torch.no_grad()
def gen_split(latnpz, pred, ptype, device):
    b = np.load(latnpz, allow_pickle=True)
    zf = b["z_full"].astype(np.float32)                      # (n, 120, d) raw latents = z_dns
    impact = (b["impact_frame"]).astype(np.int64)
    n, T, d = zf.shape
    z_dns = torch.from_numpy(zf).to(device)
    z_markov = z_dns.clone()
    z_fullc = z_dns.clone()
    cond0 = torch.zeros(1, 0, device=device)
    for i in range(n):
        ti = int(impact[i]); steps = T_TOTAL - ti - 1
        if steps <= 0:
            continue
        z_init = z_dns[i, ti:ti + 1]                         # (1, d)
        if ptype == "transformer":
            zm = E2.rollout_markov_only(pred, z_init, cond0, steps, device)
        else:
            zm = E2.rollout_autoregressive(pred, z_init, cond0, steps, device)
        z_markov[i, ti + 1:T_TOTAL] = zm[0, 1:steps + 1].float()
        lo = max(0, ti + 1 - int(pred.max_seq_len)); seed = z_dns[i, lo:ti + 1]
        zfu = E2.rollout_autoregressive(pred, seed, cond0, steps, device)
        z_fullc[i, ti + 1:T_TOTAL] = zfu[0, -steps:].float()
    out = {"z_dns": zf, "z_markov": z_markov.cpu().numpy(), "z_full": z_fullc.cpu().numpy()}
    for k in ("G", "D", "Y", "impact_frame"):
        out[k] = b[k]
    out["case_ids"] = b["case_ids"] if "case_ids" in b.files else b["case_id"]
    out["encounter_indices"] = b["encounter_indices"] if "encounter_indices" in b.files else b["encounter_index"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--tags", nargs="+", default=["tf", "lstm"])
    ap.add_argument("--splits", nargs="+", default=["test_b", "test_c"])
    args = ap.parse_args()
    device = torch.device(args.device)
    for tag in args.tags:
        pred, ptype = load_pred(tag, device)
        outdir = REPO / f"outputs/session27/rollouts_noc_{tag}"; outdir.mkdir(parents=True, exist_ok=True)
        for split in args.splits:
            latnpz = REPO / f"outputs/session27/latents_own_jepa_noc_{tag}/{split}.npz"
            res = gen_split(latnpz, pred, ptype, device)
            np.savez_compressed(outdir / f"{split}.npz", **res)
            print(f"[{tag}/{ptype}] {split}: z_markov{res['z_markov'].shape} -> {outdir}/{split}.npz", flush=True)


if __name__ == "__main__":
    main()
