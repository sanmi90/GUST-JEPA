"""Session 16, Exp 4 follow-up: predictor with cond=0 (AdaLN-Zero conditioning
disabled at inference).

Tests whether z_impact carries enough parameter information that the AdaLN-Zero
conditioning on c = (G, D, Y) is REDUNDANT at inference. If Markov-only rollout
still works with cond=zeros, the predictor relies on z_impact's parameter
content. If it breaks, the predictor depends critically on c.

Three rollout modes per encounter, each with cond=zeros:
    Markov-only with cond=0
    AR-from-z_impact with cond=0
    Full-context with cond=0

Compared to the original Exp 4 (cond=true_params) results.

Output:
    outputs/session16/exp4/cond_ablation.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import Tensor

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.data.omega_pipeline import OmegaPipeline  # noqa: E402
from scripts.session16.exp4_markov_closure import (  # noqa: E402
    ENCODER_CKPT, OMEGA_MANIFEST, DEFAULT_IMPACT_FRAME,
    load_encoder_predictor, gather_split_encounters, encode_full,
    rollout_markov_only, rollout_autoregressive, per_step_latent_rmse, horizon_summary,
)
from src.utils.device import require_rtx6000  # noqa: E402

OUT = REPO / "outputs" / "session16" / "exp4"


def evaluate_encounter_cond_zero(encounter, enc, pred, pipeline, device, cond_mode: str):
    case_id = encounter["case_id"]
    k = int(encounter["k"])
    G, D, Y = encounter["G"], encounter["D"], encounter["Y"]
    with h5py.File(encounter["path"], "r") as f:
        omega_raw = np.asarray(f["omega_z"], dtype=np.float32)
        impact = int(f.attrs.get("impact_frame_estimate", DEFAULT_IMPACT_FRAME))
    omega_clean = pipeline.preprocess_raw(omega_raw, case_id, k)
    omega_norm = pipeline.normalize(omega_clean).astype(np.float32)
    T_full = int(omega_norm.shape[0])
    if impact >= T_full - 1:
        return None
    z_dns = encode_full(enc, omega_norm, device)

    if cond_mode == "zero":
        cond = torch.zeros(1, 3, dtype=torch.float32, device=device)
    elif cond_mode == "true":
        cond = torch.tensor([[G, D, Y]], dtype=torch.float32, device=device)
    else:
        raise ValueError(cond_mode)

    H_max = T_full - impact - 1

    z_impact = z_dns[impact:impact + 1]
    z_markov = rollout_markov_only(pred, z_impact, cond, steps=H_max, device=device)
    rmse_markov = per_step_latent_rmse(z_markov.squeeze(0)[1:, :], z_dns[impact + 1:])

    z_ar = rollout_autoregressive(pred, z_impact, cond, steps=H_max, device=device)
    rmse_ar = per_step_latent_rmse(z_ar.squeeze(0)[1:, :], z_dns[impact + 1:])

    seed_len = min(32, impact + 1)
    z_seed = z_dns[impact + 1 - seed_len:impact + 1]
    z_fc = rollout_autoregressive(pred, z_seed, cond, steps=H_max, device=device)
    rmse_fc = per_step_latent_rmse(z_fc.squeeze(0)[seed_len:, :], z_dns[impact + 1:])

    return {
        "case_id": case_id, "encounter_index": k, "G": G, "D": D, "Y": Y,
        "impact_frame": impact, "n_post_impact": H_max,
        "rmse_markov": rmse_markov.tolist(),
        "rmse_ar_from_impact": rmse_ar.tolist(),
        "rmse_full_context": rmse_fc.tolist(),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    device = require_rtx6000(gpu_index=args.gpu)
    print(f"[exp4-cond0] device={device}")

    enc, pred = load_encoder_predictor(device)
    pipeline = OmegaPipeline.from_manifest(OMEGA_MANIFEST)

    HORIZONS = [1, 2, 4, 8, 16, 32, 64, 79]
    splits = ["test_b", "test_c"]
    summary: dict = {"horizons": HORIZONS, "per_split": {}}
    t0 = time.time()
    for cond_mode in ("zero", "true"):
        print(f"\n[exp4-cond0] cond_mode = {cond_mode}")
        for split in splits:
            encs = gather_split_encounters(split)
            records = []
            for e in encs:
                rec = evaluate_encounter_cond_zero(e, enc, pred, pipeline, device, cond_mode)
                if rec is not None:
                    records.append(rec)
            h_sum = horizon_summary(records, HORIZONS)
            summary["per_split"].setdefault(split, {})[cond_mode] = h_sum
            print(f"  {split} cond={cond_mode} horizon mean RMSE:")
            print(f"  {'H':>4s} {'markov':>10s} {'ar_imp':>10s} {'full_ctx':>10s}")
            for i, H in enumerate(HORIZONS):
                m = h_sum['markov']['mean_by_horizon'][i]
                a = h_sum['ar_from_impact']['mean_by_horizon'][i]
                f = h_sum['full_context']['mean_by_horizon'][i]
                print(f"  {H:>4d} {m:>10.3f} {a:>10.3f} {f:>10.3f}")

    print(f"\n[exp4-cond0] total: {time.time()-t0:.1f}s")
    save = OUT / "cond_ablation.json"
    save.write_text(json.dumps(summary, indent=2))
    print(f"[exp4-cond0] wrote {save.relative_to(REPO)}")


if __name__ == "__main__":
    main()
