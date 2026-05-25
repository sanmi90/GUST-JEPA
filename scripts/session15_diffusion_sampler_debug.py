"""Sweep DDIM sampler hyperparameters for the trained diffusion refiner.

Loads a refiner checkpoint + frozen encoder + frozen SL decoder, runs DDIM
sampling for every (t_start, n_steps, eta) combination on a few held-out
test_b encounters, and reports per-config sl_mse vs refined_mse / sl_ssim
vs refined_ssim.

Goal: find a sampling recipe where refined < sl (any improvement is a
proof-of-concept; the production setting in the train script started at
0.6 * T which was too noisy).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.omega_pipeline import OmegaPipeline
from src.evaluation.decoder_metrics import _ssim_single
from src.models.diffusion_refiner import DiffusionRefiner, NoiseSchedule, ddim_sample
from src.models.encoder import HybridCNNViTEncoder
from src.models.lap_film_decoder import LapFiLMDecoder
from src.utils.device import require_rtx6000


REPO = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--refiner-checkpoint", type=str, required=True)
    p.add_argument("--encoder-checkpoint", type=str,
                   default=str(REPO / "outputs/runs/session12/S12_E_d64/encoder/checkpoint_iter020000.pt"))
    p.add_argument("--decoder-checkpoint", type=str,
                   default=str(REPO / "outputs/runs/session12/S12_E_d64/encoder/decoder_specloss_recipe/decoder_iter012000.pt"))
    p.add_argument("--omega-pipeline-manifest", type=str,
                   default=str(REPO / "outputs/data_pipeline/v1/manifest.json"))
    p.add_argument("--split", type=str, default=str(REPO / "configs/splits/split_v1.json"))
    p.add_argument("--n-encounters", type=int, default=6, help="held-out test_b encounters to use")
    p.add_argument("--n-frames-per-enc", type=int, default=4, help="frames sampled per encounter (impact +/- offset)")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n-timesteps", type=int, default=1000)
    p.add_argument("--out", type=str, default=str(REPO / "outputs/session15/diffusion_sampler_sweep.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = require_rtx6000(gpu_index=args.gpu)
    print(f"[sampler-debug] device={device}", flush=True)

    # Frozen models
    enc_ckpt = torch.load(args.encoder_checkpoint, weights_only=False, map_location="cpu")
    enc_sd = {k.removeprefix("encoder."): v for k, v in enc_ckpt["jepa_state_dict"].items()
              if k.startswith("encoder.")}
    encoder = HybridCNNViTEncoder(latent_dim=64, projection_norm="batchnorm").to(device).eval()
    encoder.load_state_dict(enc_sd)

    dec_ckpt = torch.load(args.decoder_checkpoint, weights_only=False, map_location="cpu")
    decoder = LapFiLMDecoder(latent_dim=64).to(device).eval()
    decoder.load_state_dict(dec_ckpt["decoder_state_dict"])

    refiner_ckpt = torch.load(args.refiner_checkpoint, weights_only=False, map_location="cpu")
    refiner_args = refiner_ckpt["args"]
    refiner = DiffusionRefiner(
        in_channels=1, cond_image_channels=1, z_dim=refiner_args["latent_dim"],
        base_channels=refiner_args["refiner_base_channels"],
        ch_mult=tuple(refiner_args["refiner_ch_mult"]),
        n_resblocks=refiner_args["refiner_resblocks"],
        attn_bottleneck=refiner_args["refiner_attn_bottleneck"],
        cond_emb_dim=refiner_args["cond_emb_dim"],
        dropout=0.0,
    ).to(device).eval()
    refiner.load_state_dict(refiner_ckpt["refiner_state_dict"])
    schedule = NoiseSchedule(args.n_timesteps).to(device)
    print(f"[sampler-debug] loaded refiner iter {refiner_ckpt['iteration']}, "
          f"{sum(p.numel() for p in refiner.parameters()):,} params", flush=True)

    pipeline = OmegaPipeline.from_manifest(args.omega_pipeline_manifest)

    # Pick held-out test_b encounters
    with open(args.split) as f:
        split = json.load(f)
    test_b = [c for c in split["cases"].values() if c["split"] == "test_b"]
    encounters: list[tuple[str, int, str]] = []
    for c in test_b[:args.n_encounters]:
        encounters.append((c["case_id"], 0, c["relative_path"]))

    # Build (z, sl_omega, dns_omega) tuples for the sampled frames
    prevent = Path(os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT")))
    cache_root = prevent / "data/processed/vortex-jepa/v1"
    samples: list[dict] = []
    for cid, enc_idx, _rel in encounters:
        cache_p = cache_root / cid / f"encounter_{enc_idx:02d}.h5"
        if not cache_p.exists():
            continue
        with h5py.File(cache_p, "r") as f:
            omega = f["omega_z"][:].astype(np.float32)  # (120, 192, 96) normalized
            impact = int(f.attrs.get("impact_frame_estimate", 40))

        # Encode the full 120-frame trajectory once
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            omega_t = torch.from_numpy(omega).unsqueeze(0).unsqueeze(2).to(device).bfloat16()
            z_full = encoder(omega_t)[0]  # (120, 64)

        # Pick n_frames frames around impact: impact, impact-8, impact+8, impact+16
        offsets = [0, -8, 8, 16][:args.n_frames_per_enc]
        for off in offsets:
            fr = impact + off
            if fr < 0 or fr >= 120:
                continue
            z_one = z_full[fr:fr+1].float()  # (1, 64)
            dns_omega = torch.from_numpy(omega[fr]).unsqueeze(0).unsqueeze(0).to(device).float()
            with torch.no_grad():
                sl_omega = decoder(z_one)["pred"].float()  # (1, 1, 192, 96)
            samples.append({"cid": cid, "enc_idx": enc_idx, "frame": fr,
                            "z": z_one, "dns": dns_omega, "sl": sl_omega})
    print(f"[sampler-debug] collected {len(samples)} (encounter, frame) samples", flush=True)

    # Baseline metrics: SL-only
    sl_mses = []
    sl_ssims = []
    for s in samples:
        sl = s["sl"][0, 0].cpu().numpy()
        dns = s["dns"][0, 0].cpu().numpy()
        sl_mses.append(float(((sl - dns) ** 2).mean()))
        sl_ssims.append(float(_ssim_single(sl, dns)))
    sl_mse_mean = float(np.mean(sl_mses))
    sl_ssim_mean = float(np.mean(sl_ssims))
    print(f"\n[sampler-debug] BASELINE (SL only):  "
          f"mse={sl_mse_mean:.5f}  ssim={sl_ssim_mean:.4f}\n", flush=True)

    # Sweep DDIM hyperparameters
    t_starts = [0.05, 0.1, 0.2, 0.4]
    n_steps_list = [30, 100]
    etas = [0.0, 0.5]

    print(f"[sampler-debug] sweeping {len(t_starts)*len(n_steps_list)*len(etas)} configs "
          f"x {len(samples)} samples", flush=True)
    print(f"{'t_start':>8}  {'n_steps':>7}  {'eta':>5}  "
          f"{'ref_mse':>8}  {'mse_delta':>10}  {'ref_ssim':>9}  {'ssim_delta':>11}", flush=True)
    results = [{"config": "BASELINE_SL_ONLY",
                "mse_mean": sl_mse_mean, "ssim_mean": sl_ssim_mean}]
    for t_start, n_steps, eta in itertools.product(t_starts, n_steps_list, etas):
        ref_mses = []
        ref_ssims = []
        # Custom DDIM sample with adjustable t_start
        for s in samples:
            with torch.no_grad():
                T_total = schedule.n_timesteps
                t_start_int = int(t_start * T_total) - 1
                if t_start_int < 1:
                    t_start_int = 1
                t_start_b = torch.full((1,), t_start_int, device=device, dtype=torch.long)
                x_t, _ = schedule.q_sample(s["sl"], t_start_b)
                time_steps = torch.linspace(t_start_int, 0, n_steps + 1,
                                             dtype=torch.long, device=device)
                for i in range(n_steps):
                    t_now = time_steps[i]
                    t_next = time_steps[i + 1]
                    t_b = torch.full((1,), t_now.item(), device=device, dtype=torch.long)
                    eps = refiner(x_t, t_b, s["sl"], s["z"])
                    a_now = schedule.alpha_bars[t_now]
                    a_next = schedule.alpha_bars[t_next] if t_next >= 0 else torch.tensor(1.0, device=device)
                    x0_pred = (x_t - torch.sqrt(1 - a_now) * eps) / torch.sqrt(a_now)
                    sigma = eta * torch.sqrt((1 - a_next) / (1 - a_now)) * torch.sqrt(1 - a_now / a_next)
                    dir_xt = torch.sqrt(torch.clamp(1 - a_next - sigma ** 2, min=0.0)) * eps
                    noise = torch.randn_like(x_t) if eta > 0 else 0.0
                    x_t = torch.sqrt(a_next) * x0_pred + dir_xt + sigma * noise
                refined = x_t[0, 0].cpu().numpy()
            dns = s["dns"][0, 0].cpu().numpy()
            ref_mses.append(float(((refined - dns) ** 2).mean()))
            ref_ssims.append(float(_ssim_single(refined, dns)))
        ref_mse_mean = float(np.mean(ref_mses))
        ref_ssim_mean = float(np.mean(ref_ssims))
        mse_delta = ref_mse_mean - sl_mse_mean
        ssim_delta = ref_ssim_mean - sl_ssim_mean
        marker = "WIN" if (mse_delta < 0 or ssim_delta > 0) else "lose"
        print(f"{t_start:>8.2f}  {n_steps:>7}  {eta:>5.1f}  "
              f"{ref_mse_mean:>8.5f}  {mse_delta:>+10.5f}  "
              f"{ref_ssim_mean:>9.4f}  {ssim_delta:>+11.4f}  {marker}", flush=True)
        results.append({
            "config": f"t_start={t_start}, n_steps={n_steps}, eta={eta}",
            "t_start": t_start, "n_steps": n_steps, "eta": eta,
            "mse_mean": ref_mse_mean, "ssim_mean": ref_ssim_mean,
            "mse_delta_vs_sl": mse_delta, "ssim_delta_vs_sl": ssim_delta,
        })

    # Save
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump({
            "refiner_checkpoint": args.refiner_checkpoint,
            "refiner_iteration": refiner_ckpt["iteration"],
            "n_samples": len(samples),
            "sl_baseline_mse_mean": sl_mse_mean,
            "sl_baseline_ssim_mean": sl_ssim_mean,
            "results": results,
        }, f, indent=2)
    print(f"\n[sampler-debug] saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
