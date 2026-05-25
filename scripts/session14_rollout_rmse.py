"""Session 14 Thrust 2 -- open-loop rollout RMSE for the E d=64 JEPA stack.

Loads the production E d=64 encoder + predictor + matched SL LapFiLM decoder,
seeds the predictor with the first L=32 cache frames of each Test B / Test C
encounter, and rolls out open-loop (no teacher forcing) for the requested
horizons. At each horizon ``H`` we compare three signals:

* ``latent_rmse(H)``  := L2 distance between the rolled-out latent ``z_pred``
  and the ground-truth encoded ``z_dns`` at the same time index.
* ``raw_omega_rmse(H)`` := L2 distance between
  ``decode(z_pred)`` and ``decode(z_dns)`` -- the "decoder-readable" error
  attributable to the latent drift only.
* ``ssim(H)`` := Fukami-style SSIM between ``decode(z_pred)`` and the *DNS*
  omega field at the same time index (so it includes the decoder's own
  reconstruction error in addition to the rollout drift -- this is the
  number we will quote against Solera-Rico Figure 2).

The script also dumps the canonical Test B hero encounter rollout omega
fields to disk for the JFM hero figure later. Reference comparator:
Solera-Rico et al. Nat. Commun. 15, 1361 (2024) Figure 2 rolls a beta-VAE
+ transformer to H~200 (~10 t/c at dt=0.05).

Predictor sliding window
------------------------
The production predictor was trained at ``max_seq_len = 32``. After the
L=32 seed pass, every additional rollout step keeps only the most recent
32 frames as predictor context (sliding window) so the RoPE cache and
causal attention stay in spec.

Outputs
-------
``outputs/session14/rollout/S12_E_d64/{test_b,test_c}_rollout.json``
    Per-encounter table + per-horizon mean / median summary.
``outputs/session14/rollout/S12_E_d64/test_b_hero/omega_{frame:03d}.npy``
    Decoded rollout omega + DNS omega at the requested hero horizons.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch import Tensor, nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.omega_pipeline import OmegaPipeline  # noqa: E402
from src.evaluation.decoder_metrics import _ssim_single  # noqa: E402
from src.models.encoder import HybridCNNViTEncoder  # noqa: E402
from src.models.lap_film_decoder import LapFiLMDecoder  # noqa: E402
from src.models.predictor import AutoregressivePredictor  # noqa: E402
from src.utils.device import NoRTX6000Error, require_rtx6000  # noqa: E402


PREVENT = Path(os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT")))
CACHE = Path(
    os.environ.get("VORTEX_JEPA_CACHE", str(PREVENT / "data" / "processed" / "vortex-jepa"))
)


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------


def load_encoder_predictor(
    ckpt_path: Path, device: torch.device
) -> tuple[HybridCNNViTEncoder, AutoregressivePredictor, dict]:
    """Load encoder + predictor from a JEPA training checkpoint."""
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = blob["args"]
    enc = HybridCNNViTEncoder(
        latent_dim=int(args["d"]),
        projection_norm=args.get("projection_norm", "batchnorm"),
    )
    pred = AutoregressivePredictor(
        latent_dim=int(args["d"]),
        cond_dim=int(args.get("predictor_cond_dim", 3)),
        max_seq_len=int(args.get("T", 32)),
    )
    full_state = blob["jepa_state_dict"]
    enc_state = {
        k.removeprefix("encoder."): v
        for k, v in full_state.items()
        if k.startswith("encoder.")
    }
    pred_state = {
        k.removeprefix("predictor."): v
        for k, v in full_state.items()
        if k.startswith("predictor.")
    }
    missing_e, unexpected_e = enc.load_state_dict(enc_state, strict=False)
    missing_p, unexpected_p = pred.load_state_dict(pred_state, strict=False)
    if unexpected_e:
        print(f"[load] WARNING: unexpected encoder keys: {unexpected_e}", flush=True)
    if unexpected_p:
        print(f"[load] WARNING: unexpected predictor keys: {unexpected_p}", flush=True)
    enc.eval().to(device)
    pred.eval().to(device)
    for p in enc.parameters():
        p.requires_grad_(False)
    for p in pred.parameters():
        p.requires_grad_(False)
    return enc, pred, args


def load_lapfilm_decoder(
    ckpt_path: Path, latent_dim: int, device: torch.device
) -> LapFiLMDecoder:
    """Load a matched LapFiLM decoder, using the training-time decoder args."""
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    dec_args = blob.get("args", {})
    bc = int(dec_args.get("decoder_base_ch", 64))
    channels = (bc, bc, int(bc * 0.75), int(bc * 0.5), int(bc * 0.375))
    dec = LapFiLMDecoder(
        latent_dim=latent_dim,
        channels=channels,
        resblocks_per_level=int(dec_args.get("decoder_resblocks_per_level", 2)),
        upsample=dec_args.get("decoder_upsample", "pixelshuffle"),
        fourier_bands=int(dec_args.get("decoder_fourier_bands") or 4),
        use_film=bool(dec_args.get("decoder_use_film", True)),
        airfoil_mask_path=dec_args.get("airfoil_mask_path"),
    )
    missing, unexpected = dec.load_state_dict(blob["decoder_state_dict"], strict=False)
    if missing:
        print(f"[load] WARNING: missing decoder keys: {missing}", flush=True)
    if unexpected:
        print(f"[load] WARNING: unexpected decoder keys: {unexpected}", flush=True)
    dec.eval().to(device)
    for p in dec.parameters():
        p.requires_grad_(False)
    return dec


# -----------------------------------------------------------------------------
# Split enumeration
# -----------------------------------------------------------------------------


def gather_split_encounters(split: str, splits_path: Path) -> list[dict]:
    with open(splits_path) as f:
        manifest = json.load(f)
    out = []
    for cid, case in manifest["cases"].items():
        if split == "test_a" and case["split"] == "train":
            ks = case["test_a_encounter_indices"]
        elif split == "test_b" and case["split"] == "test_b":
            ks = list(range(int(case["n_encounters_full"])))
        elif split == "test_c" and case["split"] == "test_c":
            ks = list(range(int(case["n_encounters_full"])))
        else:
            continue
        for k in ks:
            path = CACHE / "v1" / cid / f"encounter_{k:02d}.h5"
            if not path.exists():
                continue
            out.append(
                {
                    "case_id": cid,
                    "k": int(k),
                    "path": str(path),
                    "G": float(case.get("G", 0.0)),
                    "D": float(case.get("D", 0.0)),
                    "Y": float(case.get("Y", 0.0)),
                }
            )
    return out


# -----------------------------------------------------------------------------
# Rollout
# -----------------------------------------------------------------------------


@torch.no_grad()
def encode_omega(
    enc: HybridCNNViTEncoder,
    omega_norm: np.ndarray,
    device: torch.device,
) -> Tensor:
    """Encode a full normalized omega series (T, H, W) -> (T, d).

    Returns fp32.
    """
    x = torch.from_numpy(omega_norm).to(device).unsqueeze(0).unsqueeze(2)  # (1, T, 1, H, W)
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        z = enc(x)  # (1, T, d)
    return z.float().squeeze(0)  # (T, d)


@torch.no_grad()
def rollout_sliding(
    predictor: AutoregressivePredictor,
    z_seed: Tensor,
    cond: Tensor,
    steps: int,
    device: torch.device,
) -> Tensor:
    """Open-loop rollout with a fixed-size sliding context window.

    Args:
        predictor: production predictor (max_seq_len = 32 for E d=64).
        z_seed: (B, L_seed, d) seed latents (encoded from DNS omega).
        cond: (B, cond_dim).
        steps: number of additional frames to predict.
        device: cuda device.

    Returns:
        z_full of shape (B, L_seed + steps, d) where the first L_seed
        positions equal ``z_seed`` and subsequent positions are the
        rolled-out predictions.
    """
    max_seq = int(predictor.max_seq_len)
    z_full = z_seed.clone()
    for _ in range(steps):
        ctx = z_full[:, -max_seq:, :]
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            z_hat = predictor(ctx, cond)
        next_z = z_hat[:, -1:, :].float()
        z_full = torch.cat([z_full, next_z], dim=1)
    return z_full


@torch.no_grad()
def decode_single_frame(
    decoder: LapFiLMDecoder, z_frame: Tensor, device: torch.device
) -> Tensor:
    """Decode a single (1, d) latent to (1, 1, H, W) normalized omega."""
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        out = decoder(z_frame)
    return out["pred"].float()  # (1, 1, H, W)


# -----------------------------------------------------------------------------
# Per-encounter pipeline
# -----------------------------------------------------------------------------


def evaluate_encounter(
    enc: HybridCNNViTEncoder,
    pred: AutoregressivePredictor,
    dec: LapFiLMDecoder,
    pipeline: OmegaPipeline,
    encounter: dict,
    context_length: int,
    horizons: list[int],
    device: torch.device,
    hero_horizons: Optional[list[int]] = None,
    save_hero_to: Optional[Path] = None,
) -> tuple[dict, bool]:
    """Roll out one encounter and return per-horizon metrics.

    Returns:
        (metrics_dict, decoder_nan_flag)
    """
    case_id = encounter["case_id"]
    k = int(encounter["k"])
    G, D, Y = encounter["G"], encounter["D"], encounter["Y"]

    with h5py.File(encounter["path"], "r") as f:
        omega_raw = np.asarray(f["omega_z"], dtype=np.float32)
    T_full = int(omega_raw.shape[0])

    # Pipeline (mask + clip), then normalize -> (T, H, W) fp32.
    omega_clean_raw = pipeline.preprocess_raw(omega_raw, case_id, k)
    omega_norm = pipeline.normalize(omega_clean_raw).astype(np.float32)

    # Encode the whole encounter so we have z_dns at every time index. The
    # encoder accepts (B, T, C, H, W) which is small for T=120; runs in one
    # bf16 forward.
    z_dns = encode_omega(enc, omega_norm, device)  # (T, d)

    # Seed = first ``context_length`` encoder latents.
    L = int(context_length)
    if L >= T_full:
        raise ValueError(
            f"context_length {L} >= encounter length {T_full} for {case_id}:{k}"
        )
    z_seed = z_dns[None, :L, :].contiguous()  # (1, L, d)
    cond = torch.tensor([[G, D, Y]], dtype=torch.float32, device=device)

    # Maximum rollout horizon achievable for this encounter.
    H_max = T_full - L
    valid_horizons = [int(h) for h in horizons if 1 <= int(h) <= H_max]
    if not valid_horizons:
        return (
            {
                "case_id": case_id,
                "encounter_index": k,
                "G": float(G),
                "D": float(D),
                "Y": float(Y),
                "n_frames": T_full,
                "horizons": [],
                "latent_rmse": [],
                "raw_omega_rmse": [],
                "ssim": [],
                "max_horizon": int(H_max),
            },
            False,
        )

    steps = int(max(valid_horizons))
    z_full = rollout_sliding(pred, z_seed, cond, steps, device)  # (1, L + steps, d)

    nan_flag = False
    per_h_lat = []
    per_h_raw = []
    per_h_ssim = []

    # Pre-extract dns omega in normalized space; un-normalize for SSIM/raw_omega.
    # The decoder outputs normalised omega -> un-normalise before comparing.
    s = 3.0 * pipeline.train_stats.std  # 3-sigma scale for unnormalize

    hero_dir: Optional[Path] = None
    hero_set: set[int] = set()
    if save_hero_to is not None and hero_horizons:
        hero_dir = Path(save_hero_to)
        hero_dir.mkdir(parents=True, exist_ok=True)
        hero_set = {int(h) for h in hero_horizons if 1 <= int(h) <= H_max}

    for h in valid_horizons:
        t_index = L + h - 1  # zero-indexed time of the predicted frame
        # zero-based: rollout positions in z_full are 0..L-1 (seed),
        # L (= predicted z_{L}) ... so the "+H" prediction lives at index L + h - 1.
        z_pred_h = z_full[:, t_index, :].contiguous()  # (1, d)
        z_dns_h = z_dns[t_index : t_index + 1, :]  # (1, d)

        # Latent RMSE (L2 norm of the per-frame difference vector).
        diff = (z_pred_h - z_dns_h).float()
        lat_rmse = float(torch.sqrt((diff * diff).mean()).item())

        # Decode both latents -> raw omega RMSE (decoder-readable error).
        # decode_single_frame expects (B, d) -> returns (B, 1, H, W) in
        # normalised space.
        dec_pred = decode_single_frame(dec, z_pred_h, device)  # (1, 1, H, W)
        dec_dns = decode_single_frame(dec, z_dns_h, device)  # (1, 1, H, W)

        # NaN check (the decoder occasionally explodes on out-of-distribution
        # predictor outputs).
        if (
            not torch.isfinite(dec_pred).all().item()
            or not torch.isfinite(dec_dns).all().item()
        ):
            nan_flag = True
            # Replace NaNs with 0 in raw omega space so the metric is finite;
            # SSIM is meaningless but we flag the encounter.
            dec_pred = torch.nan_to_num(dec_pred, nan=0.0, posinf=0.0, neginf=0.0)
            dec_dns = torch.nan_to_num(dec_dns, nan=0.0, posinf=0.0, neginf=0.0)

        dec_pred_raw = (dec_pred * s).squeeze(0).squeeze(0).cpu().numpy()  # (H, W)
        dec_dns_raw = (dec_dns * s).squeeze(0).squeeze(0).cpu().numpy()
        raw_rmse = float(np.sqrt(((dec_pred_raw - dec_dns_raw) ** 2).mean()))

        # SSIM between decoded rollout and DNS omega (raw scale).
        dns_raw = omega_clean_raw[t_index]  # (H, W)
        ssim_v = _ssim_single(dns_raw.astype(np.float64), dec_pred_raw.astype(np.float64))

        per_h_lat.append(lat_rmse)
        per_h_raw.append(raw_rmse)
        per_h_ssim.append(float(ssim_v))

        if hero_dir is not None and h in hero_set:
            frame = t_index
            np.save(hero_dir / f"omega_pred_H{h:03d}_frame{frame:03d}.npy", dec_pred_raw)
            np.save(hero_dir / f"omega_dns_H{h:03d}_frame{frame:03d}.npy", dns_raw)
            np.save(
                hero_dir / f"omega_dec_dns_H{h:03d}_frame{frame:03d}.npy",
                dec_dns_raw,
            )

    return (
        {
            "case_id": case_id,
            "encounter_index": k,
            "G": float(G),
            "D": float(D),
            "Y": float(Y),
            "n_frames": int(T_full),
            "horizons": valid_horizons,
            "latent_rmse": per_h_lat,
            "raw_omega_rmse": per_h_raw,
            "ssim": per_h_ssim,
            "max_horizon": int(H_max),
        },
        nan_flag,
    )


def aggregate_split(per_encounter: list[dict], horizons: list[int]) -> dict:
    """Mean / median per horizon across encounters that reached it."""
    lat_mean, lat_med = [], []
    raw_mean, raw_med = [], []
    ss_mean, ss_med = [], []
    n_per_h = []
    for h in horizons:
        lat_vals, raw_vals, ss_vals = [], [], []
        for row in per_encounter:
            if h not in row["horizons"]:
                continue
            j = row["horizons"].index(h)
            lat_vals.append(row["latent_rmse"][j])
            raw_vals.append(row["raw_omega_rmse"][j])
            ss_vals.append(row["ssim"][j])
        n_per_h.append(len(lat_vals))
        if lat_vals:
            lat_mean.append(float(np.mean(lat_vals)))
            lat_med.append(float(np.median(lat_vals)))
            raw_mean.append(float(np.mean(raw_vals)))
            raw_med.append(float(np.median(raw_vals)))
            ss_mean.append(float(np.mean(ss_vals)))
            ss_med.append(float(np.median(ss_vals)))
        else:
            lat_mean.append(float("nan"))
            lat_med.append(float("nan"))
            raw_mean.append(float("nan"))
            raw_med.append(float("nan"))
            ss_mean.append(float("nan"))
            ss_med.append(float("nan"))
    return {
        "horizons": horizons,
        "n_encounters_per_horizon": n_per_h,
        "latent_rmse_mean": lat_mean,
        "latent_rmse_median": lat_med,
        "raw_omega_rmse_mean": raw_mean,
        "raw_omega_rmse_median": raw_med,
        "ssim_mean": ss_mean,
        "ssim_median": ss_med,
    }


# -----------------------------------------------------------------------------
# Argument parsing + driver
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--encoder-checkpoint",
        type=str,
        default="outputs/runs/session12/S12_E_d64/encoder/checkpoint_iter020000.pt",
        help="JEPA checkpoint containing encoder + predictor state dicts.",
    )
    p.add_argument(
        "--decoder-checkpoint",
        type=str,
        default=(
            "outputs/runs/session12/S12_E_d64/encoder/"
            "decoder_specloss_recipe/decoder_iter012000.pt"
        ),
        help="LapFiLM SL decoder checkpoint.",
    )
    p.add_argument(
        "--omega-pipeline-manifest",
        type=str,
        default="outputs/data_pipeline/v1/manifest.json",
    )
    p.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["test_b", "test_c"],
        choices=["test_b", "test_c"],
    )
    p.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16, 32, 64, 88],
    )
    p.add_argument("--context-length", type=int, default=32)
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument(
        "--hero-encounter",
        type=str,
        default="G+1.00_D1.00_Y+0.10",
    )
    p.add_argument("--hero-encounter-index", type=int, default=0)
    p.add_argument(
        "--hero-horizons", type=int, nargs="+", default=[16, 32, 64, 88]
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="outputs/session14/rollout/S12_E_d64",
    )
    p.add_argument(
        "--splits-path",
        type=str,
        default="configs/splits/split_v1.json",
    )
    p.add_argument(
        "--max-encounters",
        type=int,
        default=0,
        help="Quick smoke; 0 means all encounters in each split.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Hardware: prefer cuda:1 (Thrust 2). Fall back to cuda:0 if busy.
    requested_gpu = int(args.gpu)
    used_gpu = requested_gpu
    try:
        device = require_rtx6000(gpu_index=requested_gpu)
    except NoRTX6000Error as e:
        if requested_gpu != 0:
            print(
                f"[rollout] requested gpu_index={requested_gpu} unavailable "
                f"({e}); falling back to gpu_index=0",
                flush=True,
            )
            device = require_rtx6000(gpu_index=0)
            used_gpu = 0
        else:
            raise
    print(
        f"[rollout] device={device} gpu_name={torch.cuda.get_device_name(device.index)} "
        f"used_gpu_index={used_gpu}",
        flush=True,
    )

    enc_ckpt = (REPO / args.encoder_checkpoint).resolve()
    dec_ckpt = (REPO / args.decoder_checkpoint).resolve()
    pipe_man = (REPO / args.omega_pipeline_manifest).resolve()
    splits_path = (REPO / args.splits_path).resolve()
    output_dir = (REPO / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[rollout] encoder_checkpoint={enc_ckpt}", flush=True)
    print(f"[rollout] decoder_checkpoint={dec_ckpt}", flush=True)
    print(f"[rollout] pipeline_manifest={pipe_man}", flush=True)
    print(f"[rollout] output_dir={output_dir}", flush=True)

    pipeline = OmegaPipeline.from_manifest(pipe_man)

    enc, predictor, enc_args = load_encoder_predictor(enc_ckpt, device)
    latent_dim = int(enc_args["d"])
    print(
        f"[rollout] encoder latent_dim={latent_dim} predictor max_seq_len="
        f"{predictor.max_seq_len} cond_dim={predictor.cond_dim}",
        flush=True,
    )
    dec = load_lapfilm_decoder(dec_ckpt, latent_dim, device)

    horizons = sorted(int(h) for h in args.horizons)
    print(f"[rollout] horizons={horizons} context_length={args.context_length}", flush=True)

    all_results: dict[str, dict] = {}
    any_nan = False
    overall_t0 = time.time()

    for split in args.splits:
        t0 = time.time()
        encs = gather_split_encounters(split, splits_path)
        if args.max_encounters > 0:
            encs = encs[: int(args.max_encounters)]
        print(f"[rollout] split={split} n_encounters={len(encs)}", flush=True)

        per_encounter = []
        for i, e in enumerate(encs):
            is_hero = (
                split == "test_b"
                and e["case_id"] == args.hero_encounter
                and int(e["k"]) == int(args.hero_encounter_index)
            )
            hero_dir = output_dir / "test_b_hero" if is_hero else None
            row, nan_flag = evaluate_encounter(
                enc=enc,
                pred=predictor,
                dec=dec,
                pipeline=pipeline,
                encounter=e,
                context_length=int(args.context_length),
                horizons=horizons,
                device=device,
                hero_horizons=list(args.hero_horizons) if is_hero else None,
                save_hero_to=hero_dir,
            )
            per_encounter.append(row)
            any_nan = any_nan or nan_flag
            tag = "HERO " if is_hero else ""
            nan_tag = " NAN" if nan_flag else ""
            print(
                f"[{split} {i + 1}/{len(encs)}] {tag}{e['case_id']}:{e['k']}{nan_tag}"
                f"  lat_rmse(H={horizons[0]})={row['latent_rmse'][0] if row['latent_rmse'] else float('nan'):.4f}"
                f"  lat_rmse(H={horizons[-1]})={row['latent_rmse'][-1] if row['latent_rmse'] else float('nan'):.4f}",
                flush=True,
            )

        # Hero fallback: if the requested hero is not in test_b, fall back to
        # the first test_b encounter and re-run that one to capture the npy.
        if split == "test_b":
            hero_present = any(
                r["case_id"] == args.hero_encounter
                and r["encounter_index"] == int(args.hero_encounter_index)
                for r in per_encounter
            )
            if not hero_present and per_encounter:
                fallback = per_encounter[0]
                print(
                    f"[rollout] WARNING: hero {args.hero_encounter}:"
                    f"{args.hero_encounter_index} not in test_b; "
                    f"falling back to {fallback['case_id']}:"
                    f"{fallback['encounter_index']}",
                    flush=True,
                )
                # Re-run the fallback to dump hero frames.
                hero_dir = output_dir / "test_b_hero"
                fallback_enc = next(
                    e
                    for e in encs
                    if e["case_id"] == fallback["case_id"]
                    and int(e["k"]) == int(fallback["encounter_index"])
                )
                _, _ = evaluate_encounter(
                    enc=enc,
                    pred=predictor,
                    dec=dec,
                    pipeline=pipeline,
                    encounter=fallback_enc,
                    context_length=int(args.context_length),
                    horizons=horizons,
                    device=device,
                    hero_horizons=list(args.hero_horizons),
                    save_hero_to=hero_dir,
                )

        summary = aggregate_split(per_encounter, horizons)
        blob = {
            "split": split,
            "encoder_checkpoint": str(enc_ckpt),
            "predictor_checkpoint": str(enc_ckpt),
            "decoder_checkpoint": str(dec_ckpt),
            "omega_pipeline_manifest": str(pipe_man),
            "context_length_L": int(args.context_length),
            "horizons": horizons,
            "latent_dim": int(latent_dim),
            "n_encounters": len(per_encounter),
            "per_encounter": per_encounter,
            "summary": summary,
            "wall_time_seconds": float(time.time() - t0),
            "any_decoder_nan": any_nan,
            "gpu_name": torch.cuda.get_device_name(device.index),
            "used_gpu_index": int(used_gpu),
        }
        out_path = output_dir / f"{split}_rollout.json"
        with open(out_path, "w") as f:
            json.dump(blob, f, indent=2)
        all_results[split] = blob
        print(
            f"[rollout] wrote {out_path} ({len(per_encounter)} encounters, "
            f"wall={time.time() - t0:.1f}s)",
            flush=True,
        )

    total_wall = time.time() - overall_t0
    print(f"[rollout] TOTAL wall={total_wall:.1f}s any_decoder_nan={any_nan}", flush=True)

    # Print a compact summary table for the operator.
    for split, blob in all_results.items():
        s = blob["summary"]
        print(f"\n[summary] split={split}", flush=True)
        print(
            f"  H        n_enc   lat_rmse_mean    lat_rmse_med    raw_rmse_mean    "
            f"ssim_mean    ssim_med",
            flush=True,
        )
        for i, h in enumerate(s["horizons"]):
            print(
                f"  {h:5d}    {s['n_encounters_per_horizon'][i]:5d}    "
                f"{s['latent_rmse_mean'][i]:13.4f}    "
                f"{s['latent_rmse_median'][i]:12.4f}    "
                f"{s['raw_omega_rmse_mean'][i]:12.4f}    "
                f"{s['ssim_mean'][i]:9.4f}    {s['ssim_median'][i]:7.4f}",
                flush=True,
            )

    # Hero file listing.
    hero_dir = output_dir / "test_b_hero"
    if hero_dir.exists():
        hero_files = sorted(hero_dir.glob("*.npy"))
        print(f"\n[hero] {len(hero_files)} npy files at {hero_dir}:", flush=True)
        for hf in hero_files:
            print(f"  {hf.name}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[rollout] interrupted", flush=True)
        sys.exit(130)
