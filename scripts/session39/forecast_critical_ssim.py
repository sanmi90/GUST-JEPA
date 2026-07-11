#!/usr/bin/env python3
"""Decoded-FORECAST SSIM at the critical instants (Carlos, 2026-07-11): the
forecast companion to tab:critical_ssim. Forecast the field h = 8 steps ahead
*to* each critical instant of the encounter (pre-impact / impact / peak lift),
decode through each family's frozen decoder, and SSIM against the DNS truth.

For a target at position p, the context ends at p - h and the shared operator
(identical to Track M1 / T5) rolls h steps to the target; the h-step forecast
latent is decoded. Full field and near-body band, averaged over test_b.

GPU (RTX-6000). Run:
    OMP_NUM_THREADS=8 taskset -c 0-15 .venv/bin/python \\
        scripts/session39/forecast_critical_ssim.py --gpu 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO))

from scripts.session36.rex_families_m1 import (  # noqa: E402
    load_split, group, train_operator, CTX, MEDIAN_IDX,
)

CACHE = REPO / "outputs/session34/trackc_latents"
DECODERS = REPO / "outputs/session34/trackc_decoders"
OUT = REPO / "outputs/session39/forecast_critical_ssim.json"
IMPACT, PRE, H = 40, 30, 8
MIN_CTX = 16
FAMILIES = [
    ("predictive", "jepa_pool_vec", "jepa_pool_vec"),
    ("AE (wake)", "ae_wake_pool", "ae_wake_pool"),
    ("Fukami (wake)", "fukami_wake", "fukami_wake"),
    ("POD", "pod_d32", "pod_d32"),  # matched latent+decoder (latents_pod is 4x scaled)
]
INSTANTS = ("preimpact", "impact", "peaklift")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from src.utils.device import require_rtx6000
    from src.models.decoder import SpatialLatentFieldDecoder
    from src.data.omega_pipeline import ssim_data_range
    from src.data.nearbody_observables import get_nearbody_band
    from src.data.wake_observables import get_wake_mask_tensor
    from scripts.session34.trackc_region_ssim import masked_ssim_per_frame

    device = require_rtx6000(gpu_index=args.gpu)
    ssim_L = float(ssim_data_range(
        str(REPO / "outputs/data_pipeline/v2p2/manifest.json")))
    band = get_nearbody_band()
    masks = {"nearbody": band > 0, "wake": get_wake_mask_tensor().numpy() > 0,
             "full": np.ones_like(band, dtype=bool)}

    ff = np.load(CACHE / "fields_test_b.npz", allow_pickle=True)
    omega = ff["omega_norm"].astype(np.float32)
    cl = ff["target_C_L"]
    fcase = np.asarray([str(c) for c in ff["case_id"]])
    fenc = np.asarray(ff["encounter_index"])
    ffr = np.asarray(ff["frame"])
    fkey = {(fcase[i], int(fenc[i]), int(ffr[i])): i for i in range(len(fcase))}
    # critical target frame per encounter (from the fields C_L)
    encs: dict = {}
    for i in range(len(fcase)):
        encs.setdefault((fcase[i], int(fenc[i])), []).append(i)
    crit_frame = {}
    for key, idxs in encs.items():
        idxs = np.asarray(idxs)
        fr = ffr[idxs]
        crit_frame[key] = {
            "preimpact": int(fr[np.argmin(np.abs(fr - PRE))]),
            "impact": int(fr[np.argmin(np.abs(fr - IMPACT))]),
            "peaklift": int(fr[np.argmax(np.abs(cl[idxs]))]),
        }

    def tile(z):
        return np.repeat(np.repeat(z[:, :, None, None], 24, 2), 12, 3)

    results = {"_provenance": {"script": "scripts/session39/forecast_critical_ssim.py",
                              "metric": "decoded h=8 forecast SSIM at critical instants",
                              "horizon": H, "instants": {"preimpact": PRE,
                              "impact": IMPACT, "peaklift": "argmax|C_L|"},
                              "ssim_L": ssim_L, "seed": args.seed,
                              "gpu_name": torch.cuda.get_device_name(device.index)},
               "families": {}}

    for label, run, dec_stem in FAMILIES:
        tr, tb = load_split(run, "train"), load_split(run, "test_b")
        encs_tr = group(tr)
        Zt = torch.from_numpy(
            np.stack([tr["z_gap"][e["rows"]] for e in encs_tr])).float().to(device)
        model = train_operator(Zt, args.seed, args.iters, device)
        dec = SpatialLatentFieldDecoder(latent_dim=Zt.shape[-1], feature_h=24,
                                        feature_w=12).to(device).eval()
        dec.load_state_dict(torch.load(DECODERS / f"decoder_{dec_stem}.pt",
                                       map_location="cpu"))
        # per-encounter latent trajectories (sorted by frame)
        acc = {inst: {m: [] for m in masks} for inst in INSTANTS}
        for e in group(tb):
            rows = e["rows"]
            Z = tb["z_gap"][rows]
            frames = tb["frame"][rows]
            case, enc = e["case_id"], int(tb["encounter_index"][rows[0]])
            pos_of = {int(f): p for p, f in enumerate(frames)}
            for inst in INSTANTS:
                tgt_f = crit_frame.get((str(case), enc), {}).get(inst)
                if tgt_f is None or tgt_f not in pos_of:
                    continue
                p_tgt = pos_of[tgt_f]
                p_anc = p_tgt - H
                if p_anc < MIN_CTX - 1:            # not enough context
                    continue
                ctx = Z[max(0, p_anc - CTX + 1):p_anc + 1][None]   # (1,ctx,d)
                with torch.no_grad():
                    pred = model(torch.from_numpy(ctx).float().to(device)).cpu().numpy()
                zf = pred[0, H - 1, :, MEDIAN_IDX][None]           # (1,d)
                with torch.no_grad():
                    df = dec(torch.from_numpy(tile(zf)).float().to(device)
                             ).float().squeeze(1).cpu().numpy()
                trow = fkey[(str(case), enc, tgt_f)]
                ss = masked_ssim_per_frame(df, omega[trow:trow + 1], ssim_L, masks)
                for m in masks:
                    acc[inst][m].append(float(ss[m][0]))
        fam_rec = {}
        for inst in INSTANTS:
            for m in masks:
                v = acc[inst][m]
                if v:
                    fam_rec[f"{inst}_{m}"] = {"ssim_mean": float(np.mean(v)),
                                              "n": len(v)}
        results["families"][label] = fam_rec
        line = "  ".join(f"{inst}={fam_rec.get(f'{inst}_full',{}).get('ssim_mean',float('nan')):.3f}"
                         for inst in INSTANTS)
        print(f"[fcrit] {label:16s} (full,h={H}): {line}", flush=True)
        del model, dec
        torch.cuda.empty_cache()
    OUT.write_text(json.dumps(results, indent=1))
    print(f"[fcrit] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
