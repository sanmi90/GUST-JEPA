"""Session 11 Track 0.3: latent perturbation probe on the E2 LapFiLM decoder.

Pure inference. Loads the Session 10 E2 decoder + Session 9 JEPA encoder
and evaluates Test B reconstruction quality as a function of additive
Gaussian noise injected into the encoder latent ``z``:

    z_perturbed[t] = z[t] + sigma * eps,  eps ~ N(0, I_d)

If the wake reconstruction quality stays robust through sigma=0.1, the
wake information is encoded in broad latent directions (good signal-to-
noise), suggesting the encoder is fine and the bottleneck is elsewhere.
If wake quality collapses already at sigma=0.05, the wake info lives in
narrow directions, justifying encoder retraining (Tracks 1-3).

Sigma is interpreted in the same units as the JEPA latent z. The per-
sigma summary reports SSIM, eps_volume, wake-enstrophy relative error,
and the radial-spectrum L2 error on Test B.

Reference: SESSION11_WAKE_RESULTS_FIRST.md "Track 0.3".

Usage::

    python scripts/session11_perturbation_probe.py \\
        --decoder-checkpoint outputs/runs/session10/E2_jepa_lapfilm_pyr_ffl/decoder_iter020000.pt \\
        --output-dir outputs/runs/session11/T0_3_perturbation_probe \\
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
)
from src.models.encoder import HybridCNNViTEncoder  # noqa: E402
from src.models.lap_film_decoder import LapFiLMDecoder  # noqa: E402
from src.utils.device import require_rtx6000  # noqa: E402


PREVENT = Path(os.environ.get("PREVENT_ROOT", "/home/carlos/PREVENT"))
CACHE = Path(os.environ.get("VORTEX_JEPA_CACHE", PREVENT / "data" / "processed" / "vortex-jepa"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Session 11 Track 0.3 latent perturbation probe")
    p.add_argument("--decoder-checkpoint", required=True, type=str)
    p.add_argument("--output-dir", required=True, type=str)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--sigmas",
        type=float,
        nargs="+",
        default=[0.0, 0.01, 0.05, 0.1, 0.5],
        help="Noise levels. 0.0 is the unperturbed reference.",
    )
    p.add_argument(
        "--omega-pipeline-manifest",
        type=str,
        default="outputs/data_pipeline/v1/manifest.json",
    )
    p.add_argument(
        "--encoder-run-override",
        type=str,
        default=None,
    )
    return p.parse_args()


def gather_test_b_encounters() -> list[dict]:
    with open(REPO / "configs" / "splits" / "split_v2.json") as f:
        manifest = json.load(f)
    out = []
    for cid, case in manifest["cases"].items():
        if case["split"] != "test_b":
            continue
        for k in range(case["n_encounters_full"]):
            path = CACHE / "v1" / cid / f"encounter_{k:02d}.h5"
            if path.exists():
                out.append({"case_id": cid, "k": int(k), "path": str(path)})
    return out


def resolve_encoder_run(decoder_args: dict, override: str | None) -> Path:
    if override is not None:
        return Path(override).resolve()
    enc_run = decoder_args.get("encoder_run")
    if enc_run is None:
        raise SystemExit(
            "decoder checkpoint has no encoder_run; pass --encoder-run-override"
        )
    p = Path(enc_run)
    if not p.is_absolute():
        p = REPO / p
    return p.resolve()


def load_jepa_encoder(ckpt_path: Path, device: torch.device) -> tuple[HybridCNNViTEncoder, int]:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = blob["args"]
    enc = HybridCNNViTEncoder(
        latent_dim=int(args["d"]),
        projection_norm=args.get("projection_norm", "batchnorm"),
    )
    state = {
        k.removeprefix("encoder."): v
        for k, v in blob["jepa_state_dict"].items()
        if k.startswith("encoder.")
    }
    enc.load_state_dict(state, strict=False)
    enc.eval().to(device)
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc, int(args["d"])


def build_lapfilm_from_args(decoder_args: dict, latent_dim: int) -> LapFiLMDecoder:
    bc = int(decoder_args.get("decoder_base_ch", 64))
    channels = (bc, bc, int(bc * 0.75), int(bc * 0.5), int(bc * 0.375))
    return LapFiLMDecoder(
        latent_dim=latent_dim,
        channels=channels,
        resblocks_per_level=int(decoder_args.get("decoder_resblocks_per_level", 2)),
        upsample=decoder_args.get("decoder_upsample", "pixelshuffle"),
        fourier_bands=int(decoder_args.get("decoder_fourier_bands") or 4),
        use_film=bool(decoder_args.get("decoder_use_film", True)),
        airfoil_mask_path=decoder_args.get("airfoil_mask_path"),
    )


def evaluate_sigma(
    encs: list[dict],
    enc: HybridCNNViTEncoder,
    dec: LapFiLMDecoder,
    device: torch.device,
    omega_pipeline: OmegaPipeline,
    sigma: float,
    seed: int,
) -> dict:
    """Per-encounter metrics under additive Gaussian noise of stddev ``sigma``."""
    generator = torch.Generator(device=device).manual_seed(seed)
    per_encounter = []
    for e in encs:
        with h5py.File(e["path"], "r") as f:
            omega_raw = np.asarray(f["omega_z"], dtype=np.float32)
        omega_clean = omega_pipeline.preprocess_raw(omega_raw, e["case_id"], int(e["k"]))
        x = torch.from_numpy(omega_clean).unsqueeze(0).unsqueeze(2).to(device)
        x = omega_pipeline.normalize(x)
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            z = enc(x).float()  # (1, T, d)
            if sigma > 0.0:
                noise = torch.randn(
                    z.shape, dtype=torch.float32, device=device, generator=generator
                ) * sigma
                z_noisy = z + noise
            else:
                z_noisy = z
            dec_out = dec(z_noisy)
            pred = dec_out["pred"] if isinstance(dec_out, dict) else dec_out
            pred_norm = pred.float().squeeze(0).squeeze(1)
            pred_raw = omega_pipeline.unnormalize(pred_norm.unsqueeze(0)).squeeze(0)
        per_encounter.append(compute_encounter_metrics(omega_clean, pred_raw.cpu().numpy()))
    return aggregate_split_metrics(per_encounter)


def main() -> None:
    args = parse_args()
    device = require_rtx6000(gpu_index=args.gpu)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "perturbation_probe.log"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    log(f"[pert-probe] device={device} gpu={torch.cuda.get_device_name(device.index)}")
    dec_ckpt_path = Path(args.decoder_checkpoint).resolve()
    log(f"[pert-probe] decoder_checkpoint={dec_ckpt_path}")

    dec_blob = torch.load(dec_ckpt_path, map_location="cpu", weights_only=False)
    dec_args = dec_blob["args"]
    enc_run = resolve_encoder_run(dec_args, args.encoder_run_override)
    cands = sorted(enc_run.glob("checkpoint_iter*.pt"))
    if not cands:
        raise SystemExit(f"no JEPA checkpoint found under {enc_run}")
    enc_ckpt = cands[-1]
    log(f"[pert-probe] jepa_checkpoint={enc_ckpt}")

    enc, d = load_jepa_encoder(enc_ckpt, device)
    log(f"[pert-probe] encoder loaded, d={d}")

    dec = build_lapfilm_from_args(dec_args, latent_dim=d).to(device)
    dec.load_state_dict(dec_blob["decoder_state_dict"])
    dec.eval()
    for p in dec.parameters():
        p.requires_grad_(False)
    log(f"[pert-probe] decoder loaded; params="
        f"{sum(p.numel() for p in dec.parameters()):,}")

    manifest_path = Path(args.omega_pipeline_manifest)
    if not manifest_path.is_absolute():
        manifest_path = REPO / manifest_path
    omega_pipeline = OmegaPipeline.from_manifest(manifest_path)

    encs = gather_test_b_encounters()
    log(f"[pert-probe] test_b: {len(encs)} encounters; sigmas={args.sigmas}")

    results: dict[str, dict] = {}
    for sigma in args.sigmas:
        log(f"[pert-probe] sigma={sigma}: evaluating...")
        agg = evaluate_sigma(encs, enc, dec, device, omega_pipeline, sigma, args.seed)
        results[f"sigma_{sigma:g}"] = agg
        log(
            f"[pert-probe] sigma={sigma:g}: "
            f"ssim_median={agg.get('ssim_mean_median', float('nan')):.4f} "
            f"eps_vol_median={agg.get('eps_volume_median', float('nan')):.4f} "
            f"wake_enstrophy_rel_err_median={agg.get('enstrophy_rel_err_wake_median', float('nan')):.4f} "
            f"radial_spectrum_l2_wake_median={agg.get('radial_spectrum_l2_wake_median', float('nan')):.4f}"
        )

    summary = {
        "decoder_checkpoint": str(dec_ckpt_path),
        "jepa_checkpoint": str(enc_ckpt),
        "test_b_n_encounters": len(encs),
        "sigmas": args.sigmas,
        "results_by_sigma": results,
    }
    base = results[f"sigma_{args.sigmas[0]:g}"].get("ssim_mean_median", float("nan"))
    summary["ssim_median_by_sigma"] = {
        f"{s:g}": results[f"sigma_{s:g}"].get("ssim_mean_median", float("nan"))
        for s in args.sigmas
    }
    s05 = results.get("sigma_0.05", {}).get("ssim_mean_median", float("nan"))
    s1 = results.get("sigma_0.1", {}).get("ssim_mean_median", float("nan"))
    summary["narrow_directions_flag_if_sigma_0p05_drop_geq_50pct"] = bool(
        np.isfinite(base) and np.isfinite(s05) and (base - s05) / max(abs(base), 1e-6) >= 0.5
    )
    summary["robust_through_sigma_0p1_if_drop_lt_25pct"] = bool(
        np.isfinite(base) and np.isfinite(s1) and (base - s1) / max(abs(base), 1e-6) < 0.25
    )
    with open(out_dir / "perturbation_probe.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"[pert-probe] wrote {out_dir / 'perturbation_probe.json'}")


if __name__ == "__main__":
    main()
