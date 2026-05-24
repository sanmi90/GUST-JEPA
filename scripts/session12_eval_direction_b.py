"""Direction B extended evaluation: wraps the WakeRefiner around the frozen
W0_C_lam100 + E1 decoder pair so the standard session10_evaluate.py pipeline
works. Writes the same extended_metrics.json format as the other configs.

Usage:
    python scripts/session12_eval_direction_b.py --gpu 0

Output:
    outputs/runs/session12/S12_B_gan_refine/eval/extended_metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.session10_evaluate import gather_encounters  # noqa: E402
from scripts.session9_decoder_fig3_pipeline import (  # noqa: E402
    _extract_pred,
    load_decoder,
    load_encoder,
    resolve_encoder_ckpt,
)
from src.evaluation.decoder_metrics import (  # noqa: E402
    aggregate_split_metrics,
    compute_encounter_metrics,
    wake_2d_premult_spectrum_series,
    wake_mask,
)
from src.models.refiner import WakeRefiner  # noqa: E402
from src.utils.device import require_rtx6000  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Direction B extended eval")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--encoder-run", type=str,
                   default="outputs/runs/session11/W0_C_lam100")
    p.add_argument(
        "--decoder-checkpoint", type=str,
        default="outputs/runs/session11/W0_C_lam100/decoder_E1_recipe/decoder_iter020000.pt",
    )
    p.add_argument(
        "--refiner-checkpoint", type=str,
        default="outputs/runs/session12/S12_B_gan_refine/refiner_iter020000.pt",
    )
    p.add_argument(
        "--output-json", type=str,
        default="outputs/runs/session12/S12_B_gan_refine/eval/extended_metrics.json",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = require_rtx6000(gpu_index=args.gpu)

    enc_ckpt = resolve_encoder_ckpt(None, args.encoder_run)
    enc, d, pipe = load_encoder(enc_ckpt, device)
    print(f"[eval-B] encoder loaded d={d}, pipeline={'yes' if pipe else 'no'}",
          flush=True)
    dec = load_decoder(Path(args.decoder_checkpoint), d, device)
    print(f"[eval-B] decoder loaded", flush=True)

    blob = torch.load(args.refiner_checkpoint, map_location="cpu",
                      weights_only=False)
    refiner_args = blob.get("args", {}) or {}
    channels = int(refiner_args.get("refiner_channels", 64))
    blocks = int(refiner_args.get("refiner_blocks", 6))
    refiner = WakeRefiner(channels=channels, n_blocks=blocks).to(device)
    state = blob.get("refiner_state_dict") or blob.get("refiner") or blob.get(
        "state_dict"
    )
    refiner.load_state_dict(state, strict=False)
    refiner.eval()
    for p in refiner.parameters():
        p.requires_grad_(False)
    wm = torch.tensor(wake_mask(192, 96), dtype=torch.float32, device=device)
    print(f"[eval-B] refiner loaded ({sum(p.numel() for p in refiner.parameters()):,} params)",
          flush=True)

    out = {
        "encoder_checkpoint": str(enc_ckpt),
        "decoder_checkpoint": args.decoder_checkpoint,
        "refiner_checkpoint": args.refiner_checkpoint,
        "decoder_type": "lapfilm+refiner",
        "latent_dim": d,
    }

    for split in ("test_a", "test_b", "test_c"):
        encs = gather_encounters(split)
        print(f"[eval-B] evaluating {split} ({len(encs)} encounters)...",
              flush=True)
        per_enc = []
        per_2d = []
        for i, e in enumerate(encs):
            with h5py.File(e["path"], "r") as f:
                omega = np.asarray(f["omega_z"], dtype=np.float32)
            if pipe is not None:
                omega = pipe.preprocess_raw(omega, e["case_id"], int(e["k"]))
            x = torch.from_numpy(omega).unsqueeze(0).unsqueeze(2).to(device)
            with torch.no_grad():
                with torch.autocast(device_type=device.type,
                                    dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                    x_in = pipe.normalize(x) if pipe is not None else x
                    z = enc(x_in)
                    dec_out = dec(z)
                    coarse = _extract_pred(dec_out)
                    # coarse can be (B,T,C,H,W) or (B*T,C,H,W); reshape
                    if coarse.dim() == 5:
                        cf = coarse.reshape(-1, *coarse.shape[-3:])
                    else:
                        cf = coarse
                    residual = refiner(cf, wm)
                    refined = cf + residual
                    if coarse.dim() == 5:
                        refined = refined.reshape(*coarse.shape)
                    x_hat = refined
                    if pipe is not None:
                        x_hat = pipe.unnormalize(x_hat)
            x_hat = x_hat.float().squeeze(0).squeeze(1).cpu().numpy()
            m = compute_encounter_metrics(target=omega, pred=x_hat,
                                          active_tau_raw=1.0,
                                          radial_n_bins=32)
            per_enc.append(m)
            sp = wake_2d_premult_spectrum_series(pred=x_hat, target=omega)
            per_2d.append({
                "max_wavelength_ratio": float(sp["max_wavelength_ratio"]),
                "mean_contour_iou": float(sp["mean_contour_iou"]),
                "contour_iou": [float(v) for v in sp["contour_iou"]],
            })
            if (i + 1) % 8 == 0 or (i + 1) == len(encs):
                print(f"  ... {i + 1}/{len(encs)}", flush=True)
        agg = aggregate_split_metrics(per_enc)
        mwr = np.array([
            p["max_wavelength_ratio"] for p in per_2d
            if np.isfinite(p["max_wavelength_ratio"])
        ])
        iou = np.array([p["mean_contour_iou"] for p in per_2d])
        per_level = np.array([p["contour_iou"] for p in per_2d])
        agg["spectrum2d_max_wavelength_ratio_mean"] = (
            float(mwr.mean()) if mwr.size else float("nan")
        )
        agg["spectrum2d_max_wavelength_ratio_median"] = (
            float(np.median(mwr)) if mwr.size else float("nan")
        )
        agg["spectrum2d_mean_contour_iou_mean"] = float(iou.mean())
        agg["spectrum2d_mean_contour_iou_median"] = float(np.median(iou))
        for li, lvl in enumerate((0.10, 0.50, 0.90)):
            agg[f"spectrum2d_contour_iou_level_{int(lvl*100):02d}_median"] = (
                float(np.median(per_level[:, li]))
            )
        out[split] = agg
        print(f"  {split}: SSIM={agg['ssim_mean_mean']:.3f}  "
              f"eps_vol={agg['eps_volume_mean']:.3f}  "
              f"mse_wake={agg['mse_wake_mean']:.2f}  "
              f"enstrophy_wake_rel={agg['enstrophy_rel_err_wake_mean']:.3f}  "
              f"radial_L2={agg['radial_spectrum_l2_wake_mean']:.3f}  "
              f"spec2d_iou={agg['spectrum2d_mean_contour_iou_median']:.3f}  "
              f"spec2d_lam_ratio={agg['spectrum2d_max_wavelength_ratio_median']:.3f}",
              flush=True)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[eval-B] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
