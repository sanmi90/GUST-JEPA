"""B1 Test 2: rollouts using the PRODUCTION JEPA's jointly-trained predictor
on the Session 14 precomputed JEPA latents.

If H=16 instability disappears here vs the generic-predictor result in B1, the
issue is that my from-scratch predictor's BatchNorm running stats are
miscalibrated relative to the JEPA encoder's actual latent distribution
(the production predictor's BN stats were trained jointly).

Output:
    outputs/session18/exp_b1_test3/rollouts_jepa_d64_test2_prod/{test_b,test_c}.npz
        z_dns, z_markov, z_full, G, D, Y, case_ids, encounter_indices, impact_frame
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.models.predictor import AutoregressivePredictor  # noqa: E402
from src.utils.device import require_rtx6000  # noqa: E402
from scripts.session18.eval_baseline_rollouts import (  # noqa: E402
    _get, rollout_markov, rollout_full,
)


PROD_CKPT = REPO / "outputs/runs/session12/S12_E_d64/encoder/checkpoint_iter020000.pt"
LATENTS_DIR = REPO / "outputs/session18/exp_b1/latents_jepa_d64"
OUT_DIR = REPO / "outputs/session18/exp_b1_test3/rollouts_jepa_d64_test2_prod"


def load_production_predictor(device: torch.device) -> AutoregressivePredictor:
    blob = torch.load(PROD_CKPT, map_location="cpu", weights_only=False)
    args = blob["args"]
    pred = AutoregressivePredictor(
        latent_dim=int(args["d"]),
        cond_dim=int(args.get("predictor_cond_dim", 3)),
        max_seq_len=int(args.get("T", 32)),
    ).to(device)
    state = blob["jepa_state_dict"]
    pred_state = {k.removeprefix("predictor."): v for k, v in state.items() if k.startswith("predictor.")}
    pred.load_state_dict(pred_state, strict=False)
    pred.eval()
    for p in pred.parameters():
        p.requires_grad_(False)
    print(f"[test2] loaded production predictor: hidden={pred.hidden_dim}, "
          f"max_seq_len={pred.max_seq_len}, cond_dim={pred.cond_dim}")
    return pred


def main() -> None:
    p = argparse.ArgumentParser(description="B1 Test 2: production JEPA predictor")
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--splits", nargs="+", default=["test_b", "test_c"])
    args = p.parse_args()

    device = require_rtx6000(gpu_index=args.gpu)
    pred = load_production_predictor(device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        npz_path = LATENTS_DIR / f"{split}.npz"
        if not npz_path.exists():
            print(f"[test2] {split}: latents npz missing at {npz_path}; skipping")
            continue
        print(f"[test2] {split}: rolling out with production predictor")
        blob = np.load(npz_path, allow_pickle=True)
        z_full_dns = blob["z_full"].astype(np.float32)  # (n_enc, 120, d)
        G = blob["G"].astype(np.float32)
        D = blob["D"].astype(np.float32)
        Y = blob["Y"].astype(np.float32)
        impact = _get(blob, "impact_frame").astype(np.int64)
        case_ids = _get(blob, "case_ids", "case_id")
        enc_idx = _get(blob, "encounter_indices", "encounter_index")

        n_enc, T_total, d = z_full_dns.shape
        # Production predictor expects RAW (BN-normalized by encoder) latents.
        # NO additional standardization (this is what Session 17 D119 did).
        z_dns = torch.from_numpy(z_full_dns).to(device)
        cond_t = torch.from_numpy(np.stack([G, D, Y], axis=1)).to(device)
        max_seq = pred.max_seq_len

        z_markov = torch.zeros_like(z_dns)
        z_full_out = torch.zeros_like(z_dns)

        for i in range(n_enc):
            ti = int(impact[i])
            c = cond_t[i : i + 1]
            z_markov[i, : ti + 1] = z_dns[i, : ti + 1]
            steps = T_total - ti - 1
            if steps > 0:
                z_init = z_dns[i, ti : ti + 1].unsqueeze(0)
                z_m = rollout_markov(pred, z_init, c, steps=steps, device=device)
                z_markov[i, ti + 1 : T_total] = z_m[0, 1 : steps + 1].float()
            seed_start = max(0, ti + 1 - max_seq)
            z_seed = z_dns[i, seed_start : ti + 1].unsqueeze(0)
            z_full_out[i, : ti + 1] = z_dns[i, : ti + 1]
            if steps > 0:
                z_f = rollout_full(pred, z_seed, c, steps=steps, device=device)
                z_full_out[i, ti + 1 : T_total] = z_f[0, -steps:].float()
            if (i + 1) % 10 == 0 or (i + 1) == n_enc:
                print(f"   rollout {i + 1}/{n_enc}")

        out_path = OUT_DIR / f"{split}.npz"
        np.savez(
            out_path,
            z_dns=z_dns.cpu().numpy(),
            z_markov=z_markov.cpu().numpy(),
            z_full=z_full_out.cpu().numpy(),
            G=G, D=D, Y=Y,
            case_ids=case_ids,
            encounter_indices=enc_idx,
            impact_frame=impact,
        )
        print(f"[test2] {split}: wrote {out_path}")

    print("[test2] DONE")


if __name__ == "__main__":
    main()
