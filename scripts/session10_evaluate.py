"""Session 10 extended evaluation: per-decoder physics-grounded metrics.

Loads a frozen JEPA encoder + a decoder checkpoint, reconstructs every
encounter in Test A / B / C, and writes a ``decoder_summary_extended.json``
file with the Session 10 metric bundle (mse full/active/inactive/wake,
SSIM, eps_vol, enstrophy relative error, circulation absolute error,
local FFT error, radial spectrum L2 in the wake ROI). The metrics are
computed in raw scale after the omega pipeline's unnormalisation.

Usage::

    python -m scripts.session10_evaluate \\
        --encoder-run outputs/runs/session9/run_jepa_pipeline_lam0p01_seed42 \\
        --decoder-run outputs/runs/session10/E1_jepa_lapfilm_pyr_noffl \\
        --gpu 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.omega_pipeline import OmegaPipeline  # noqa: E402
from src.evaluation.decoder_metrics import (  # noqa: E402
    aggregate_split_metrics,
    compute_encounter_metrics,
    wake_2d_premult_spectrum_series,
)
from src.utils.device import require_rtx6000  # noqa: E402
from scripts.session9_decoder_fig3_pipeline import (  # noqa: E402
    build_decoder_from_ckpt,
    load_encoder,
    resolve_encoder_ckpt,
    _extract_pred,
)


PREVENT = Path(os.environ.get("PREVENT_ROOT", "/home/carlos/PREVENT"))
CACHE = Path(os.environ.get("VORTEX_JEPA_CACHE", PREVENT / "data" / "processed" / "vortex-jepa"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Session 10 extended decoder evaluation")
    enc_group = p.add_mutually_exclusive_group(required=True)
    enc_group.add_argument("--jepa-checkpoint", type=str)
    enc_group.add_argument("--encoder-run", type=str)
    p.add_argument("--decoder-run", required=True, type=str,
                   help="Decoder run dir containing decoder_iter*.pt files. "
                        "Uses the largest-iter checkpoint by default.")
    p.add_argument("--decoder-checkpoint", type=str, default=None,
                   help="Override the auto-selected decoder checkpoint.")
    p.add_argument("--decoder-type", type=str, default=None,
                   choices=["fukami", "lapfilm", "coord_mlp"])
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--active-tau-raw", type=float, default=1.0)
    p.add_argument("--n-radial-bins", type=int, default=32)
    p.add_argument("--output-json", type=str, default=None,
                   help="Defaults to <decoder-run>/decoder_summary_extended.json.")
    return p.parse_args()


def resolve_decoder_ckpt(decoder_run: str, override: str | None) -> Path:
    if override is not None:
        return Path(override).resolve()
    run_dir = Path(decoder_run).resolve()
    candidates = sorted(run_dir.glob("decoder_iter*.pt"))
    if not candidates:
        raise FileNotFoundError(f"no decoder_iter*.pt under {run_dir}")
    return candidates[-1]


def gather_encounters(split: str) -> list[dict]:
    with open(REPO / "configs" / "splits" / "split_v1.json") as f:
        manifest = json.load(f)
    out = []
    for cid, case in manifest["cases"].items():
        if split == "test_a" and case["split"] == "train":
            ks = case["test_a_encounter_indices"]
        elif split == "test_b" and case["split"] == "test_b":
            ks = list(range(case["n_encounters_full"]))
        elif split == "test_c" and case["split"] == "test_c":
            ks = list(range(case["n_encounters_full"]))
        else:
            continue
        for k in ks:
            path = CACHE / "v1" / cid / f"encounter_{k:02d}.h5"
            if not path.exists():
                continue
            out.append({
                "case_id": cid, "k": int(k), "path": str(path),
                "G": float(case.get("G", 0.0)),
                "D": float(case.get("D", 0.0)),
                "Y": float(case.get("Y", 0.0)),
            })
    return out


def evaluate_split_extended(
    enc, dec, encs, device, pipe, active_tau_raw, n_radial_bins, log,
) -> dict:
    per_enc = []
    per_enc_2d_spec = []  # Session 12 D90: per-encounter 2D premult spectrum agreement
    for i, e in enumerate(encs):
        with h5py.File(e["path"], "r") as f:
            omega = np.asarray(f["omega_z"], dtype=np.float32)
        if pipe is not None:
            omega = pipe.preprocess_raw(omega, e["case_id"], int(e["k"]))
        x = torch.from_numpy(omega).unsqueeze(0).unsqueeze(2).to(device)
        if pipe is not None:
            x_in = pipe.normalize(x)
        else:
            x_in = x
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                z = enc(x_in)
                dec_out = dec(z)
                x_hat = _extract_pred(dec_out)
                if pipe is not None:
                    x_hat = pipe.unnormalize(x_hat)
        x_hat = x_hat.float().squeeze(0).squeeze(1).cpu().numpy()  # (T, H, W)
        metrics = compute_encounter_metrics(
            target=omega, pred=x_hat,
            active_tau_raw=active_tau_raw,
            radial_n_bins=n_radial_bins,
        )
        per_enc.append(metrics)
        spec_out = wake_2d_premult_spectrum_series(pred=x_hat, target=omega)
        per_enc_2d_spec.append({
            "max_wavelength_ratio": float(spec_out["max_wavelength_ratio"]),
            "mean_contour_iou": float(spec_out["mean_contour_iou"]),
            "contour_iou": [float(v) for v in spec_out["contour_iou"]],
            "median_wavelength_ratio": [
                float(v) for v in spec_out["median_wavelength_ratio"]
            ],
        })
        if (i + 1) % 8 == 0 or i + 1 == len(encs):
            log(f"  ... {i + 1}/{len(encs)} encounters")
    agg = aggregate_split_metrics(per_enc)
    # Aggregate 2D spectrum metrics (mean / median across encounters).
    if per_enc_2d_spec:
        mwr = np.array([
            p["max_wavelength_ratio"] for p in per_enc_2d_spec
            if np.isfinite(p["max_wavelength_ratio"])
        ])
        iou = np.array([p["mean_contour_iou"] for p in per_enc_2d_spec])
        agg["spectrum2d_max_wavelength_ratio_mean"] = (
            float(mwr.mean()) if mwr.size else float("nan")
        )
        agg["spectrum2d_max_wavelength_ratio_median"] = (
            float(np.median(mwr)) if mwr.size else float("nan")
        )
        agg["spectrum2d_mean_contour_iou_mean"] = float(iou.mean())
        agg["spectrum2d_mean_contour_iou_median"] = float(np.median(iou))
        # Per-level contour IoU (10%, 50%, 90%) median across encounters
        contour_levels = (0.10, 0.50, 0.90)
        per_level = np.array([p["contour_iou"] for p in per_enc_2d_spec])
        for li, lvl in enumerate(contour_levels):
            agg[f"spectrum2d_contour_iou_level_{int(lvl*100):02d}_median"] = (
                float(np.median(per_level[:, li]))
            )
    return agg


def main() -> None:
    args = parse_args()
    device = require_rtx6000(gpu_index=args.gpu)

    decoder_ckpt = resolve_decoder_ckpt(args.decoder_run, args.decoder_checkpoint)
    encoder_ckpt = resolve_encoder_ckpt(args.jepa_checkpoint, args.encoder_run)

    def log(msg: str) -> None:
        print(msg, flush=True)

    log(f"[eval] device={device}")
    log(f"[eval] encoder_ckpt={encoder_ckpt}")
    log(f"[eval] decoder_ckpt={decoder_ckpt}")

    enc, d, pipe = load_encoder(encoder_ckpt, device)
    log(f"[eval] encoder d={d}, pipeline={'yes' if pipe is not None else 'no'}")
    dec, decoder_type = build_decoder_from_ckpt(
        decoder_ckpt, d, device, override_type=args.decoder_type,
    )
    log(f"[eval] decoder_type={decoder_type}")

    out = {
        "encoder_checkpoint": str(encoder_ckpt),
        "decoder_checkpoint": str(decoder_ckpt),
        "decoder_type": decoder_type,
        "latent_dim": d,
        "active_tau_raw": args.active_tau_raw,
        "radial_n_bins": args.n_radial_bins,
    }
    for split in ("test_a", "test_b", "test_c"):
        encs = gather_encounters(split)
        log(f"[eval] evaluating {split} ({len(encs)} encounters)...")
        out[split] = evaluate_split_extended(
            enc, dec, encs, device, pipe,
            active_tau_raw=args.active_tau_raw,
            n_radial_bins=args.n_radial_bins, log=log,
        )

    json_path = Path(args.output_json) if args.output_json else (
        Path(args.decoder_run) / "decoder_summary_extended.json"
    )
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    log(f"[eval] wrote {json_path}")

    # Brief summary on stdout
    for split in ("test_a", "test_b", "test_c"):
        s = out[split]
        log(f"  {split}: SSIM={s['ssim_mean_mean']:.3f}  eps_vol={s['eps_volume_mean']:.3f}  "
            f"mse_wake={s['mse_wake_mean']:.2f}  "
            f"enstrophy_wake_rel={s['enstrophy_rel_err_wake_mean']:.3f}  "
            f"radial_L2={s['radial_spectrum_l2_wake_mean']:.3f}  "
            f"spec2d_iou={s.get('spectrum2d_mean_contour_iou_median', float('nan')):.3f}  "
            f"spec2d_lam_ratio={s.get('spectrum2d_max_wavelength_ratio_median', float('nan')):.3f}")


if __name__ == "__main__":
    main()
