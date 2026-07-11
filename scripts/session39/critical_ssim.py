#!/usr/bin/env python3
"""Decode SSIM at CRITICAL INSTANTS (Carlos, 2026-07-11): instead of the
window-averaged decode-floor SSIM, score each family at three instants of the
gust encounter -- pre-impact, impact, and peak lift -- to see how faithfully the
field is rendered when it matters. Decode-floor (true latent -> field, no
forecast), per family, averaged over the test_b encounters.

Instants per encounter (frames within the encounter):
  pre-impact : the frame nearest impact - 10 (approaching vortex)
  impact     : the frame nearest the nominal impact (40)
  peak-lift  : argmax |C_L| (the peak-load instant)

GPU (RTX-6000). Run:
    OMP_NUM_THREADS=8 taskset -c 0-15 .venv/bin/python \\
        scripts/session39/critical_ssim.py --gpu 1
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
OUT = REPO / "outputs/session39/critical_ssim.json"
IMPACT = 40
PRE = 30  # impact - 10
# paper label -> (latent run, decoder stem)
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

    # per encounter: pick the three critical field-rows
    encs: dict = {}
    for i in range(len(fcase)):
        encs.setdefault((fcase[i], int(fenc[i])), []).append(i)
    crit_rows = {}  # (case,enc) -> {instant: field_row}
    for key, idxs in encs.items():
        idxs = np.asarray(idxs)
        frames = ffr[idxs]
        pick = {
            "preimpact": idxs[np.argmin(np.abs(frames - PRE))],
            "impact": idxs[np.argmin(np.abs(frames - IMPACT))],
            "peaklift": idxs[np.argmax(np.abs(cl[idxs]))],
        }
        crit_rows[key] = pick

    def tile(z):
        return np.repeat(np.repeat(z[:, :, None, None], 24, 2), 12, 3)

    results = {"_provenance": {"script": "scripts/session39/critical_ssim.py",
                              "metric": "decode-floor SSIM (true latent -> field)",
                              "instants": {"preimpact": f"nearest frame {PRE}",
                                           "impact": f"nearest frame {IMPACT}",
                                           "peaklift": "argmax |C_L|"},
                              "ssim_L": ssim_L, "n_encounters": len(crit_rows),
                              "gpu_name": torch.cuda.get_device_name(device.index)},
               "families": {}}

    for label, run, dec_stem in FAMILIES:
        lp = CACHE / f"latents_{run}_test_b.npz"
        if not lp.exists():
            print(f"[crit] {label}: latents missing, skip")
            continue
        lat = np.load(lp, allow_pickle=True)
        z = lat["z_gap"].astype(np.float32)
        lkey = {(str(c), int(e), int(fr)): i for i, (c, e, fr)
                in enumerate(zip(lat["case_id"], lat["encounter_index"],
                                 lat["frame"]))}
        dec = SpatialLatentFieldDecoder(latent_dim=z.shape[1], feature_h=24,
                                        feature_w=12).to(device).eval()
        dec.load_state_dict(torch.load(DECODERS / f"decoder_{dec_stem}.pt",
                                       map_location="cpu"))
        # gather the critical latents + truth fields, tagged by instant
        acc = {inst: {m: [] for m in masks} for inst in INSTANTS}
        for (case, enc), pick in crit_rows.items():
            for inst, frow in pick.items():
                fr = int(ffr[frow])
                li = lkey.get((case, enc, fr))
                if li is None:
                    continue
                zt = z[li:li + 1]
                with torch.no_grad():
                    df = dec(torch.from_numpy(tile(zt)).float().to(device)
                             ).float().squeeze(1).cpu().numpy()
                ss = masked_ssim_per_frame(df, omega[frow:frow + 1], ssim_L, masks)
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
        print(f"[crit] {label:16s} (full): {line}", flush=True)

    OUT.write_text(json.dumps(results, indent=1))
    print(f"[crit] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
