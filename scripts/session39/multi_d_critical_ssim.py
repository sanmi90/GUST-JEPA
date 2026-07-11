#!/usr/bin/env python3
"""Decode SSIM at the critical instants (pre-impact / impact / peak lift) ACROSS
DIMENSION d in {4, 8, 16, 32} (Carlos, 2026-07-11): the dimension sweep of
tab:critical_ssim. Decode floor (true latent -> field, no forecast), per family
per d, full field and near-body band, averaged over test_b.

Families with multi-d decoders: predictive (JEPA), Fukami (wake), POD across all
four d; AE (wake) at d=32 only. GPU. Run:
    OMP_NUM_THREADS=8 taskset -c 0-15 .venv/bin/python \\
        scripts/session39/multi_d_critical_ssim.py --gpu 1
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

CACHE = REPO / "outputs/session34/trackc_latents"
DECODERS = REPO / "outputs/session34/trackc_decoders"
OUT = REPO / "outputs/session39/multi_d_critical_ssim.json"
IMPACT, PRE = 40, 30
DIMS = (4, 8, 16, 32)
INSTANTS = ("preimpact", "impact", "peaklift")
# family -> {d: (latent run, decoder stem)}
FAMILIES = {
    "predictive": {4: ("jepa_pool_vec_d4", "jepa_pool_vec_d4"),
                   8: ("jepa_pool_vec_d8", "jepa_pool_vec_d8"),
                   16: ("jepa_pool_vec_d16", "jepa_pool_vec_d16"),
                   32: ("jepa_pool_vec", "jepa_pool_vec")},
    "predictive (lift)": {32: ("jepa_pool_ln_s0", "jepa_pool_ln_s0")},  # CLN, d32
    "AE (wake)": {32: ("ae_wake_pool", "ae_wake_pool")},
    "Fukami (wake)": {4: ("fukami_wake_d4", "fukami_wake_d4"),
                      8: ("fukami_wake_d8", "fukami_wake_d8"),
                      16: ("fukami_wake_d16", "fukami_wake_d16"),
                      32: ("fukami_wake", "fukami_wake")},
    "POD": {4: ("pod_d4", "pod_d4"), 8: ("pod_d8", "pod_d8"),
            16: ("pod_d16", "pod_d16"), 32: ("pod_d32", "pod_d32")},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
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
    encs: dict = {}
    for i in range(len(fcase)):
        encs.setdefault((fcase[i], int(fenc[i])), []).append(i)
    crit = {}
    for key, idxs in encs.items():
        idxs = np.asarray(idxs)
        fr = ffr[idxs]
        crit[key] = {"preimpact": idxs[np.argmin(np.abs(fr - PRE))],
                     "impact": idxs[np.argmin(np.abs(fr - IMPACT))],
                     "peaklift": idxs[np.argmax(np.abs(cl[idxs]))]}

    def tile(z):
        return np.repeat(np.repeat(z[:, :, None, None], 24, 2), 12, 3)

    results = {"_provenance": {"script": "scripts/session39/multi_d_critical_ssim.py",
                              "metric": "decode-floor SSIM at critical instants vs d",
                              "ssim_L": ssim_L, "gpu_name": torch.cuda.get_device_name(device.index)},
               "families": {}}
    for label, dmap in FAMILIES.items():
        results["families"][label] = {}
        for d, (run, dec_stem) in dmap.items():
            lp = CACHE / f"latents_{run}_test_b.npz"
            if not lp.exists():
                continue
            lat = np.load(lp, allow_pickle=True)
            z = lat["z_gap"].astype(np.float32)
            lkey = {(str(c), int(e), int(fr)): i for i, (c, e, fr)
                    in enumerate(zip(lat["case_id"], lat["encounter_index"], lat["frame"]))}
            dec = SpatialLatentFieldDecoder(latent_dim=z.shape[1], feature_h=24,
                                            feature_w=12).to(device).eval()
            dec.load_state_dict(torch.load(DECODERS / f"decoder_{dec_stem}.pt",
                                           map_location="cpu"))
            acc = {inst: {m: [] for m in masks} for inst in INSTANTS}
            for (case, enc), pick in crit.items():
                for inst, frow in pick.items():
                    fr = int(ffr[frow])
                    li = lkey.get((case, enc, fr))
                    if li is None:
                        continue
                    with torch.no_grad():
                        df = dec(torch.from_numpy(tile(z[li:li + 1])).float().to(device)
                                 ).float().squeeze(1).cpu().numpy()
                    ss = masked_ssim_per_frame(df, omega[frow:frow + 1], ssim_L, masks)
                    for m in masks:
                        acc[inst][m].append(float(ss[m][0]))
            rec = {}
            for inst in INSTANTS:
                for m in masks:
                    v = acc[inst][m]
                    if v:
                        rec[f"{inst}_{m}"] = float(np.mean(v))
            results["families"][label][str(d)] = rec
            print(f"[md] {label:16s} d={d:2d} (full): "
                  + " ".join(f"{i}={rec.get(f'{i}_full',float('nan')):.3f}" for i in INSTANTS),
                  flush=True)
    OUT.write_text(json.dumps(results, indent=1))
    print(f"[md] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
