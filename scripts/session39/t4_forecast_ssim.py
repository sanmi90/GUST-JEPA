#!/usr/bin/env python3
"""T4 (paper_redesign.md section 6): decoded-forecast SSIM by phase.

Rolls the shared direct forecaster (identical to Track M1 / T5) on each family's
frozen latents, decodes the h-step forecast through that family's frozen decoder,
and scores SSIM against the DNS truth field, split by phase (pre / through / post
impact). Answers the SSIM half of the central question: how well is the *field*
(not just the latent) forecast, before and through the impact transient.

Three central families (predictive / matched-supervision AE / linear), horizons
{4, 8, 16}, near-body / wake / full masks at the pinned v2p2 Wang SSIM range.

GPU (RTX-6000). Run:
    OMP_NUM_THREADS=8 taskset -c 0-15 .venv/bin/python \\
        scripts/session39/t4_forecast_ssim.py --gpu 1
"""
from __future__ import annotations

import argparse
import json
import time
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
OUT = REPO / "outputs/session39/t4_forecast_ssim.json"
IMPACT = 40
HORIZONS = (4, 8, 16)
HROLL = 40
# paper label -> (latent run, decoder file stem)
FAMILIES = {
    "JepaWake": ("jepa_pool_vec", "jepa_pool_vec"),
    "AeWake": ("ae_wake_pool", "ae_wake_pool"),
    "Pod": ("pod_d32", "pod_d32"),  # matched latent+decoder (latents_pod is 4x scaled)
}


def phase_of(ftgt, fanc):
    if ftgt < IMPACT:
        return "pre"
    if fanc < IMPACT <= ftgt:
        return "through"
    return "post"


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

    # truth-field lookup: (case, enc, frame) -> omega_norm row
    ff = np.load(CACHE / "fields_test_b.npz", allow_pickle=True)
    omega = ff["omega_norm"].astype(np.float32)
    fkey = {(str(c), int(e), int(fr)): i for i, (c, e, fr)
            in enumerate(zip(ff["case_id"], ff["encounter_index"], ff["frame"]))}

    def tile(z):  # (B,d) -> (B,d,24,12) broadcast, decoder input
        return np.repeat(np.repeat(z[:, :, None, None], 24, 2), 12, 3)

    results = {"_provenance": {"script": "scripts/session39/t4_forecast_ssim.py",
                              "impact_frame": IMPACT, "horizons": list(HORIZONS),
                              "ssim_L": ssim_L, "seed": args.seed,
                              "gpu_name": torch.cuda.get_device_name(device.index),
                              "note": "single operator seed; decoded-forecast SSIM"},
               "families": {}}
    t0 = time.time()
    for label, (run, dec_stem) in FAMILIES.items():
        tr, tb = load_split(run, "train"), load_split(run, "test_b")
        encs_tr, encs_tb = group(tr), group(tb)
        Zt = torch.from_numpy(
            np.stack([tr["z_gap"][e["rows"]] for e in encs_tr])).float().to(device)
        model = train_operator(Zt, args.seed, args.iters, device)

        dec = SpatialLatentFieldDecoder(latent_dim=Zt.shape[-1],
                                        feature_h=24, feature_w=12).to(device).eval()
        dec.load_state_dict(torch.load(DECODERS / f"decoder_{dec_stem}.pt",
                                       map_location="cpu"))

        def decode(z):  # (B,d)->(B,192,96)
            outs = []
            for i in range(0, len(z), 128):
                zb = torch.from_numpy(tile(z[i:i + 128])).float().to(device)
                with torch.no_grad():
                    outs.append(dec(zb).float().squeeze(1).cpu().numpy())
            return np.concatenate(outs) if outs else np.empty((0, 192, 96))

        # accumulate SSIM per (horizon, phase, mask)
        acc = {h: {p: {k: [] for k in masks} for p in ("pre", "through", "post")}
               for h in HORIZONS}
        for e in encs_tb:
            rows = e["rows"]
            Z = tb["z_gap"][rows]
            wm = tb["window_mask"][rows]
            frames = tb["frame"][rows]
            T = Z.shape[0]
            case = e["case_id"]
            enc = int(tb["encounter_index"][rows[0]])
            anchors = np.arange(CTX - 1, T - 1)
            ctx = np.stack([Z[a - CTX + 1:a + 1] for a in anchors])
            with torch.no_grad():
                pred = model(torch.from_numpy(ctx).float().to(device)).cpu().numpy()
            roll = pred[..., MEDIAN_IDX]  # (A, HROLL, d)
            for h in HORIZONS:
                tgt = anchors + h
                ok = (tgt <= T - 1) & wm[np.clip(tgt, 0, T - 1)]
                if not ok.any():
                    continue
                zf = roll[ok, h - 1]                       # forecast latent
                tfr = frames[tgt[ok]]                      # target frames
                afr = frames[anchors[ok]]
                # truth fields at target frames
                idx = [fkey.get((str(case), enc, int(f))) for f in tfr]
                keep = [j for j, ii in enumerate(idx) if ii is not None]
                if not keep:
                    continue
                zf = zf[keep]
                truth = omega[[idx[j] for j in keep]]
                dfields = decode(zf)
                ss = masked_ssim_per_frame(dfields, truth, ssim_L, masks)
                for j, jj in enumerate(keep):
                    p = phase_of(int(tfr[jj]), int(afr[jj]))
                    for k in masks:
                        acc[h][p][k].append(float(ss[k][j]))
        fam_rec = {}
        for h in HORIZONS:
            for p in ("pre", "through", "post"):
                for k in masks:
                    v = acc[h][p][k]
                    if v:
                        fam_rec[f"h{h}_{p}_{k}"] = {"ssim_mean": float(np.mean(v)),
                                                    "n": len(v)}
        results["families"][label] = fam_rec
        OUT.write_text(json.dumps(results, indent=1))
        line = " ".join(f"h{h}/{p}={fam_rec.get(f'h{h}_{p}_full',{}).get('ssim_mean',float('nan')):.3f}"
                        for h in (8,) for p in ("pre", "through", "post"))
        print(f"[t4] {label} DONE (full-mask): {line}  ({time.time()-t0:.0f}s)",
              flush=True)
        del model, dec
        torch.cuda.empty_cache()
    print(f"[t4] all done ({time.time()-t0:.0f}s) -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
