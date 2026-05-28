"""Session 11 Track 0.2: temporal-window probe on the E2 LapFiLM decoder.

Pure inference. Loads the Session 10 E2 decoder + the Session 9 JEPA encoder
and evaluates Test B reconstruction quality under three temporal aggregation
modes of the latent ``z``:

- ``single``: ``decode(z_t)`` (the production setup).
- ``temporal_mean``: ``decode(mean(z_{t-2..t+2}))`` (5-frame symmetric window).
- ``future_window``: ``decode(mean(z_t..z_{t+5}))`` (6-frame future window).

If the future-window mode boosts Test B SSIM by >= 0.05 over the single-frame
baseline, H3 (the encoder per-frame latent lacks future wake context) is
supported and Track 4 should include a temporal-aware decoder.

Reference: SESSION11_WAKE_RESULTS_FIRST.md "Track 0.2".

Usage::

    python scripts/session11_temporal_probe.py \\
        --decoder-checkpoint outputs/runs/session10/E2_jepa_lapfilm_pyr_ffl/decoder_iter020000.pt \\
        --output-dir outputs/runs/session11/T0_2_temporal_probe \\
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
    p = argparse.ArgumentParser(description="Session 11 Track 0.2 temporal-window probe")
    p.add_argument("--decoder-checkpoint", required=True, type=str)
    p.add_argument("--output-dir", required=True, type=str)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument(
        "--omega-pipeline-manifest",
        type=str,
        default="outputs/data_pipeline/v1/manifest.json",
    )
    p.add_argument(
        "--encoder-run-override",
        type=str,
        default=None,
        help="Override the encoder_run path stored in the decoder checkpoint args.",
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


def aggregate_modes(z: torch.Tensor, mode: str) -> torch.Tensor:
    """Aggregate the per-frame latent ``z`` (T, d) into a per-frame z' (T, d).

    ``single``       -> z[t]
    ``temporal_mean``-> mean(z[t-2..t+2]), edge-clipped
    ``future_window``-> mean(z[t..t+5]), edge-clipped
    """
    T, d = z.shape
    out = torch.empty_like(z)
    if mode == "single":
        return z.clone()
    if mode == "temporal_mean":
        # Window of 5 centred at t: indices [t-2, t-1, t, t+1, t+2].
        idx = torch.arange(T)
        lo = torch.clamp(idx - 2, min=0)
        hi = torch.clamp(idx + 2, max=T - 1)
        for t in range(T):
            out[t] = z[lo[t]:hi[t] + 1].mean(dim=0)
        return out
    if mode == "future_window":
        # Window of 6 forward from t: indices [t, t+1, ..., t+5].
        idx = torch.arange(T)
        lo = idx
        hi = torch.clamp(idx + 5, max=T - 1)
        for t in range(T):
            out[t] = z[lo[t]:hi[t] + 1].mean(dim=0)
        return out
    raise ValueError(f"unknown mode {mode!r}")


def evaluate_mode(
    encs: list[dict],
    enc: HybridCNNViTEncoder,
    dec: LapFiLMDecoder,
    device: torch.device,
    omega_pipeline: OmegaPipeline,
    mode: str,
) -> dict:
    """Compute aggregate test_b metrics for one temporal-aggregation mode."""
    per_encounter = []
    for e in encs:
        with h5py.File(e["path"], "r") as f:
            omega_raw = np.asarray(f["omega_z"], dtype=np.float32)
        omega_clean = omega_pipeline.preprocess_raw(omega_raw, e["case_id"], int(e["k"]))
        x = torch.from_numpy(omega_clean).unsqueeze(0).unsqueeze(2).to(device)
        x = omega_pipeline.normalize(x)  # (1, T, 1, H, W)
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            z = enc(x).float().squeeze(0)  # (T, d)
            z_agg = aggregate_modes(z, mode).unsqueeze(0)  # (1, T, d)
            dec_out = dec(z_agg)
            pred = dec_out["pred"] if isinstance(dec_out, dict) else dec_out
            pred_norm = pred.float().squeeze(0).squeeze(1)  # (T, H, W)
            pred_raw = omega_pipeline.unnormalize(pred_norm.unsqueeze(0)).squeeze(0)
        pred_np = pred_raw.cpu().numpy()
        em = compute_encounter_metrics(omega_clean, pred_np)
        per_encounter.append(em)
    return aggregate_split_metrics(per_encounter)


def main() -> None:
    args = parse_args()
    device = require_rtx6000(gpu_index=args.gpu)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "temporal_probe.log"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    log(f"[temporal-probe] device={device} gpu={torch.cuda.get_device_name(device.index)}")
    dec_ckpt_path = Path(args.decoder_checkpoint).resolve()
    log(f"[temporal-probe] decoder_checkpoint={dec_ckpt_path}")

    dec_blob = torch.load(dec_ckpt_path, map_location="cpu", weights_only=False)
    dec_args = dec_blob["args"]
    enc_run = resolve_encoder_run(dec_args, args.encoder_run_override)
    cands = sorted(enc_run.glob("checkpoint_iter*.pt"))
    if not cands:
        raise SystemExit(f"no JEPA checkpoint found under {enc_run}")
    enc_ckpt = cands[-1]
    log(f"[temporal-probe] jepa_checkpoint={enc_ckpt}")

    enc, d = load_jepa_encoder(enc_ckpt, device)
    log(f"[temporal-probe] encoder loaded, d={d}")

    dec = build_lapfilm_from_args(dec_args, latent_dim=d).to(device)
    dec.load_state_dict(dec_blob["decoder_state_dict"])
    dec.eval()
    for p in dec.parameters():
        p.requires_grad_(False)
    log(f"[temporal-probe] decoder loaded; type=lapfilm, params="
        f"{sum(p.numel() for p in dec.parameters()):,}")

    manifest_path = Path(args.omega_pipeline_manifest)
    if not manifest_path.is_absolute():
        manifest_path = REPO / manifest_path
    omega_pipeline = OmegaPipeline.from_manifest(manifest_path)
    log(f"[temporal-probe] omega pipeline loaded from {manifest_path}")

    encs = gather_test_b_encounters()
    log(f"[temporal-probe] test_b: {len(encs)} encounters")

    results: dict[str, dict] = {}
    for mode in ("single", "temporal_mean", "future_window"):
        log(f"[temporal-probe] mode={mode}: evaluating...")
        agg = evaluate_mode(encs, enc, dec, device, omega_pipeline, mode)
        results[mode] = agg
        log(
            f"[temporal-probe] mode={mode}: "
            f"ssim_median={agg.get('ssim_mean_median', float('nan')):.4f} "
            f"ssim_mean={agg.get('ssim_mean_mean', float('nan')):.4f} "
            f"eps_vol_median={agg.get('eps_volume_median', float('nan')):.4f} "
            f"wake_enstrophy_rel_err_median={agg.get('enstrophy_rel_err_wake_median', float('nan')):.4f} "
            f"radial_spectrum_l2_wake_median={agg.get('radial_spectrum_l2_wake_median', float('nan')):.4f}"
        )

    summary = {
        "decoder_checkpoint": str(dec_ckpt_path),
        "jepa_checkpoint": str(enc_ckpt),
        "test_b_n_encounters": len(encs),
        "results_by_mode": results,
    }
    delta_ssim = (
        results["future_window"].get("ssim_mean_median", 0.0)
        - results["single"].get("ssim_mean_median", 0.0)
    )
    summary["delta_ssim_future_minus_single"] = float(delta_ssim)
    summary["H3_supported_if_delta_ssim_geq_0p05"] = bool(delta_ssim >= 0.05)
    with open(out_dir / "temporal_probe.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"[temporal-probe] delta_ssim_future_minus_single={delta_ssim:+.4f} "
        f"(>= 0.05 -> H3 supported, temporal-aware decoder is the right Track 4)")
    log(f"[temporal-probe] wrote {out_dir / 'temporal_probe.json'}")


if __name__ == "__main__":
    main()
