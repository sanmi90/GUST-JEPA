"""Session 9 Step 2: train the visualisation decoder on a frozen JEPA encoder.

The decoder is a separate :class:`HybridViTConvDecoder` that maps
``z in R^d`` (the JEPA encoder's projection-head output) back to a
mid-plane vorticity field ``omega_z`` of shape ``(192, 96)``. It is
NEVER part of the JEPA loss; the encoder weights stay frozen here.

Training loop
-------------
- 10k iterations, AdamW (0.9, 0.95), lr=1e-4, wd=0.05.
- Per-frame MSE on ``omega_z``, summed over (T, H, W).
- bf16 mixed precision on the RTX 6000 Blackwell.
- Cosine LR with 5% linear warmup.
- Batch B = 16 sub-trajectories of T = 32 frames (matches the encoder
  training data layout).
- W&B logging: encoder checkpoint hash, decoder seed, train MSE,
  iter-2000 / 4000 / 6000 / 8000 / 10000 evaluation on Test A.

Pass criterion (Session 9 plan, Section 5.6 of the architecture spec):
the Test A reconstruction MSE must be within 2x the per-case-mean noise
floor (= the MSE of the per-case-mean ``omega_z`` field on Test A).

Usage:
    python -m scripts.session9_train_decoder \\
        --jepa-checkpoint outputs/runs/session9/run_f1_lam0p001_seed0/checkpoint_iter020000.pt \\
        --output-dir outputs/runs/session9/decoder \\
        --gpu 0 --max-iters 10000

Hardware: RTX 6000 Blackwell only (require_rtx6000 at entry).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.models.coord_mlp_decoder import CoordMLPDecoder  # noqa: E402
from src.models.decoder import HybridViTConvDecoder  # noqa: E402
from src.models.decoder_losses import (  # noqa: E402
    region_pyr_ffl_loss,
    region_pyr_specloss_loss,
)
from src.models.encoder import HybridCNNViTEncoder, PatchPoolEncoder  # noqa: E402
from src.models.lap_film_decoder import LapFiLMDecoder  # noqa: E402
from src.utils.device import require_rtx6000  # noqa: E402


PREVENT = Path(os.environ.get("PREVENT_ROOT", "/home/carlos/PREVENT"))
CACHE = Path(os.environ.get("VORTEX_JEPA_CACHE", PREVENT / "data" / "processed" / "vortex-jepa"))


def file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_encoder(ckpt_path: Path, device: torch.device) -> tuple[HybridCNNViTEncoder, int]:
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
    missing, unexpected = enc.load_state_dict(state, strict=False)
    if unexpected:
        print(f"[decoder-train] WARNING: unexpected encoder keys ignored: {unexpected}",
              flush=True)
    enc.eval().to(device)
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc, int(args["d"])


def gather_train_encounters() -> list[dict]:
    with open(REPO / "configs" / "splits" / "split_v1.json") as f:
        manifest = json.load(f)
    out = []
    for cid, case in manifest["cases"].items():
        if case["split"] != "train":
            continue
        # All non-test_a encounters in the case
        test_a_idx = set(case["test_a_encounter_indices"])
        for k in range(case["n_encounters_full"]):
            if k in test_a_idx:
                continue
            path = CACHE / "v1" / cid / f"encounter_{k:02d}.h5"
            if not path.exists():
                continue
            out.append({"case_id": cid, "k": int(k), "path": str(path)})
    return out


def gather_eval_encounters(split: str) -> list[dict]:
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
            out.append({"case_id": cid, "k": int(k), "path": str(path)})
    return out


class EncounterFrameDataset(torch.utils.data.Dataset):
    """Yields a random T-frame sub-trajectory of omega_z for one encounter."""

    def __init__(self, encs: list[dict], T: int = 32, seed: int = 0,
                 omega_pipeline=None) -> None:
        self.encs = encs
        self.T = T
        self.rng = np.random.default_rng(seed)
        self.omega_pipeline = omega_pipeline

    def __len__(self) -> int:
        return len(self.encs)

    def __getitem__(self, idx: int) -> torch.Tensor:
        e = self.encs[idx]
        with h5py.File(e["path"], "r") as f:
            omega = np.asarray(f["omega_z"], dtype=np.float32)
        T_full = omega.shape[0]
        if T_full <= self.T:
            start = 0
        else:
            start = int(self.rng.integers(0, T_full - self.T + 1))
        x = omega[start : start + self.T]
        if self.omega_pipeline is not None:
            x = self.omega_pipeline.preprocess_raw(x, e["case_id"], int(e["k"]))
            x = self.omega_pipeline.normalize(x)
        return torch.from_numpy(x)  # (T, H, W), normalized if pipeline set


def collate(batch: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(batch, dim=0)  # (B, T, H, W)


def encode_batch(
    enc: nn.Module,
    x: torch.Tensor,
    device: torch.device,
    train_encoder: bool = False,
) -> torch.Tensor:
    """Run encoder. x: (B, T, H, W). Returns z: (B, T, d).

    When ``train_encoder=False`` (default) the encoder runs under
    ``torch.no_grad()`` -- matches the Session 9 / 10 behaviour with a
    frozen JEPA encoder. ``train_encoder=True`` keeps the autograd graph
    so backprop reaches the encoder (Session 11 Track 0.1 omega_direct).
    """
    x = x.to(device).unsqueeze(2)  # (B, T, 1, H, W)
    if train_encoder:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            z = enc(x)
    else:
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                z = enc(x)
    return z.float()


def build_lr_lambda(max_iters: int, warmup_frac: float):
    warmup = max(1, int(warmup_frac * max_iters))
    def fn(step: int) -> float:
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, max_iters - warmup)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * prog))
    return fn


def _ssim(x: np.ndarray, y: np.ndarray, c1: float = 0.16, c2: float = 1.44) -> float:
    """Fukami's SSIM definition (arXiv:2305.18394 Eq. 1) on a (H, W) pair."""
    mu_x, mu_y = x.mean(), y.mean()
    var_x, var_y = x.var(), y.var()
    cov_xy = ((x - mu_x) * (y - mu_y)).mean()
    num = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
    return float(num / max(den, 1e-12))


def _l2_relative_error(q: np.ndarray, q_hat: np.ndarray, eps: float = 1.0) -> float:
    """Fukami's L_2 relative reconstruction error: || q - q_hat ||_2 / || q ||_2.

    eps floor of 1.0 (raw vorticity units) prevents the metric from
    exploding on near-zero baseline frames. Used in Fukami
    arXiv:2305.18394 / J. Fluid Mech. 1018, A22 (2023) Figures 15-18.
    """
    num = float(np.sqrt(((q - q_hat) ** 2).sum()))
    den = float(np.sqrt((q ** 2).sum()))
    return num / max(den, eps)


def evaluate_split(
    enc: HybridCNNViTEncoder,
    dec: HybridViTConvDecoder,
    encs: list[dict],
    device: torch.device,
    omega_scale: float = 1.0,
    omega_pipeline=None,
) -> dict:
    """Per-encounter reconstruction MSE + SSIM on a split + case-mean noise floor.

    When ``omega_pipeline`` is provided, inputs are pipeline-preprocessed
    (mask + per-encounter clip + normalize) before the encoder. The
    decoder output is unnormalized back to raw scale and metrics are
    computed against the pipeline-preprocessed target (the cleaned omega,
    NOT the artifact-laden raw — matches the training target).
    """
    case_to_arr: dict[str, list[np.ndarray]] = {}
    for e in encs:
        with h5py.File(e["path"], "r") as f:
            omega = np.asarray(f["omega_z"], dtype=np.float32)
        if omega_pipeline is not None:
            omega = omega_pipeline.preprocess_raw(omega, e["case_id"], int(e["k"]))
        case_to_arr.setdefault(e["case_id"], []).append(omega)
    case_mean = {cid: np.stack(arrs, axis=0).mean(axis=0) for cid, arrs in case_to_arr.items()}

    mses = []
    floors = []
    ssims = []
    eps_frames = []
    eps_volume = []
    dec.eval()
    with torch.no_grad():
        for e in encs:
            with h5py.File(e["path"], "r") as f:
                omega = np.asarray(f["omega_z"], dtype=np.float32)
            if omega_pipeline is not None:
                omega = omega_pipeline.preprocess_raw(omega, e["case_id"], int(e["k"]))
            T = omega.shape[0]
            x = torch.from_numpy(omega).unsqueeze(0).unsqueeze(2).to(device)  # (1, T, 1, H, W)
            if omega_pipeline is not None:
                x = omega_pipeline.normalize(x)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                z = enc(x)
                dec_out = dec(z)
                if isinstance(dec_out, dict):
                    x_hat = dec_out["pred"]
                else:
                    x_hat = dec_out
                if omega_pipeline is not None:
                    x_hat = omega_pipeline.unnormalize(x_hat)
                else:
                    x_hat = x_hat * omega_scale  # de-normalize to raw scale
            x_hat = x_hat.float().squeeze(0).squeeze(1).cpu().numpy()  # (T, H, W)
            mse = float(((omega - x_hat) ** 2).mean())
            floor = float(((omega - case_mean[e["case_id"]]) ** 2).mean())
            ssim_t = float(np.mean([_ssim(omega[t], x_hat[t]) for t in range(T)]))
            eps_t = float(np.mean([_l2_relative_error(omega[t], x_hat[t]) for t in range(T)]))
            eps_v = _l2_relative_error(omega, x_hat)
            mses.append(mse)
            floors.append(floor)
            ssims.append(ssim_t)
            eps_frames.append(eps_t)
            eps_volume.append(eps_v)
    return {
        "mse_mean": float(np.mean(mses)),
        "mse_median": float(np.median(mses)),
        "ssim_mean": float(np.mean(ssims)),
        "ssim_median": float(np.median(ssims)),
        "eps_per_frame_mean": float(np.mean(eps_frames)),
        "eps_per_frame_median": float(np.median(eps_frames)),
        "eps_volume_mean": float(np.mean(eps_volume)),
        "eps_volume_median": float(np.median(eps_volume)),
        "floor_mean": float(np.mean(floors)),
        "ratio_mean": float(np.mean(mses) / max(np.mean(floors), 1e-12)),
        "n_encounters": len(encs),
    }


def _str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes", "y", "t"):
        return True
    if v.lower() in ("false", "0", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError(f"boolean value expected, got {v!r}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Session 9 / 10 visualisation decoder")

    enc_group = p.add_mutually_exclusive_group(required=False)
    enc_group.add_argument(
        "--jepa-checkpoint", type=str,
        help="Path to a single JEPA encoder checkpoint .pt file.")
    enc_group.add_argument(
        "--encoder-run", type=str,
        help="Path to a JEPA run directory; the script picks the largest-iter "
             "checkpoint inside (matches the Session 10 plan launch commands).")
    p.add_argument(
        "--input-mode", type=str, default="latent",
        choices=["latent", "omega_direct"],
        help=(
            "latent (default): load a frozen JEPA encoder from --jepa-checkpoint "
            "or --encoder-run and train the decoder only. "
            "omega_direct (Session 11 Track 0.1): bypass the JEPA encoder and "
            "feed omega through a small trainable PatchPoolEncoder (16x16 patch "
            "pool to 12x6, 1x1 conv to base_ch channels). This is the LapFiLM "
            "upper-bound diagnostic: how well does the decoder reconstruct given "
            "richer-than-32D input? Both encoder and decoder are trained."
        ),
    )

    p.add_argument("--output-dir", required=True, type=str)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--B", type=int, default=16)
    p.add_argument("--T", type=int, default=32)
    p.add_argument("--max-iters", type=int, default=10000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--omega-scale", type=float, default=1.0,
                   help="Normalize target omega by this scale during training; "
                        "decoder learns to output omega/omega_scale. At inference "
                        "the decoder output is multiplied back by omega_scale so "
                        "eval metrics (MSE, SSIM, eps) are reported on raw scale.")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--checkpoint-every", type=int, default=2000)
    p.add_argument("--omega-pipeline-manifest", type=str, default=None,
                   help="OmegaPipeline manifest path. When set: target = "
                        "pipeline.normalize(pipeline.preprocess_raw(omega)); "
                        "decoder learns in normalized space; eval unnormalizes "
                        "before computing raw-scale MSE/SSIM/eps. Overrides "
                        "--omega-scale when set.")

    # Decoder architecture
    p.add_argument("--decoder-type", type=str, default="fukami",
                   choices=["fukami", "lapfilm", "coord_mlp"],
                   help="Visualisation decoder architecture. "
                        "fukami = HybridViTConvDecoder (Session 9 baseline). "
                        "lapfilm = 5-level Laplacian-pyramid + FiLM (Session 10). "
                        "coord_mlp = coordinate neural field audit (Session 10 E4).")
    p.add_argument("--decoder-upsample", type=str, default="pixelshuffle",
                   choices=["pixelshuffle", "bilinear_conv"])
    p.add_argument("--decoder-fourier-bands", type=int, default=None,
                   help="Fourier bands per coord. Defaults: 4 for lapfilm, 8 for coord_mlp.")
    p.add_argument("--decoder-base-ch", type=int, default=64,
                   help="Coarsest-level channel count for lapfilm.")
    p.add_argument("--decoder-resblocks-per-level", type=int, default=2,
                   help="Residual blocks per pyramid level for lapfilm.")
    p.add_argument("--decoder-use-film", type=_str2bool, default=True,
                   help="lapfilm only: enable FiLM modulation. False = concat-only "
                        "(the no_film ablation E_noFiLM).")
    p.add_argument("--decoder-mlp-hidden", type=int, default=128,
                   help="coord_mlp hidden width.")
    p.add_argument("--decoder-mlp-layers", type=int, default=5,
                   help="coord_mlp depth.")
    p.add_argument("--decoder-mlp-activation", type=str, default="sine",
                   choices=["sine", "gelu_fourier"],
                   help="coord_mlp activation (SIREN or GELU+Fourier).")
    p.add_argument("--decoder-mlp-chunk", type=int, default=4096,
                   help="coord_mlp pixels per chunk.")
    p.add_argument("--decoder-cond", type=str, default="none",
                   choices=["none", "params", "params_phase"],
                   help="Conditioning mode beyond z. Session 10 only supports "
                        "'none' (the conditioning ablation is deferred to Session 11).")

    # Loss selection
    p.add_argument("--decoder-loss", type=str, default="mse",
                   choices=["mse", "charbonnier", "region_pyr_ffl",
                            "region_pyr_specloss"],
                   help="Reconstruction loss family. region_pyr_ffl uses the "
                        "Session 10 combined region-weighted + pyramid + focal-"
                        "frequency + enstrophy + circulation objective. "
                        "region_pyr_specloss is the Session 12 Direction A "
                        "composite: region + pyramid + enstrophy + circulation + "
                        "PRF 2026 gradient consistency + PRF 2026 spectral "
                        "amplitude (Balasubramanian et al. Phys. Rev. Fluids "
                        "11, 044907, 2026, Eqs. 6-8).")
    p.add_argument("--recon-loss-type", type=str, default=None,
                   choices=["mse", "charbonnier"],
                   help="Session 9 deprecated alias for --decoder-loss; if "
                        "provided, overrides --decoder-loss for backward "
                        "compatibility with Session 9 launch commands.")
    p.add_argument("--charbonnier-epsilon", type=float, default=0.05)
    p.add_argument("--recon-active-threshold", type=float, default=0.0,
                   help="Active-pixel mask threshold (legacy mse/charbonnier loss). "
                        "0 disables. region_pyr_ffl uses --active-tau / "
                        "--inactive-weight instead.")
    p.add_argument("--recon-inactive-weight", type=float, default=0.0,
                   help="Weight on inactive pixels (legacy mse/charbonnier loss).")

    # region_pyr_ffl loss hyperparameters
    p.add_argument("--lambda-region", type=float, default=1.0)
    p.add_argument("--lambda-pyramid", type=float, default=0.4)
    p.add_argument("--lambda-ffl", type=float, default=0.05)
    p.add_argument("--lambda-enstrophy", type=float, default=0.02)
    p.add_argument("--lambda-circulation", type=float, default=0.01)
    p.add_argument("--ffl-warmup-iters", type=int, default=2000,
                   help="Iterations before FFL ramps up from 0.")
    p.add_argument("--ffl-ramp-iters", type=int, default=1000,
                   help="Iterations over which FFL ramps from 0 to 1 (linear).")
    p.add_argument("--ffl-alpha", type=float, default=1.0)
    p.add_argument("--ffl-patch", type=int, default=32)
    p.add_argument("--active-tau", type=float, default=0.10)
    p.add_argument("--active-softness", type=float, default=0.03)
    p.add_argument("--inactive-weight", type=float, default=0.05)
    p.add_argument("--wake-weight", type=float, default=0.50)
    p.add_argument("--airfoil-mask-path", type=str, default=None,
                   help="Path to the airfoil-adjacent mask .npy. Defaults to "
                        "outputs/data_pipeline/v1/airfoil_adjacent_mask.npy.")

    # PRF 2026 SL loss hyperparameters (Session 12 Direction A).
    p.add_argument("--lambda-gradient", type=float, default=1.0,
                   help="Weight on the PRF 2026 gradient consistency term "
                        "(Eq. 7). Only used with --decoder-loss "
                        "region_pyr_specloss.")
    p.add_argument("--lambda-spectral-amp", type=float, default=1.0,
                   help="Weight on the PRF 2026 spectral amplitude term "
                        "(Eq. 8). Only used with --decoder-loss "
                        "region_pyr_specloss.")
    p.add_argument("--spectral-window", type=str, default="hann",
                   choices=["hann", "tukey", "none"],
                   help="Window applied before the 2D FFT in the spectral "
                        "amplitude term. Required for our non-periodic, "
                        "airfoil-masked domain (PRF 2026 used periodic BCs).")
    p.add_argument("--spectral-tukey-alpha", type=float, default=0.5,
                   help="Tukey taper fraction; only used when "
                        "--spectral-window tukey.")
    p.add_argument("--spectral-wake-only", action="store_true", default=True,
                   help="Restrict the spectral amplitude FFT to the wake ROI "
                        "(default). When set --no-spectral-wake-only the FFT "
                        "is taken on the full field.")
    p.add_argument("--no-spectral-wake-only", action="store_false",
                   dest="spectral_wake_only")

    return p.parse_args()


def resolve_encoder_checkpoint(args: argparse.Namespace, log) -> Path:
    """Return the .pt encoder checkpoint path from --jepa-checkpoint or --encoder-run."""
    if args.jepa_checkpoint is not None:
        return Path(args.jepa_checkpoint).resolve()
    run_dir = Path(args.encoder_run).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"--encoder-run {run_dir} is not a directory")
    candidates = sorted(run_dir.glob("checkpoint_iter*.pt"))
    if not candidates:
        raise FileNotFoundError(f"no checkpoint_iter*.pt under {run_dir}")
    chosen = candidates[-1]
    log(f"[decoder-train] resolved --encoder-run {run_dir} -> {chosen.name}")
    return chosen


def build_decoder(
    args: argparse.Namespace,
    latent_dim: int,
    device: torch.device,
    spatial_init: bool = False,
) -> nn.Module:
    """Construct the visualisation decoder per --decoder-type."""
    if args.decoder_type == "fukami":
        if spatial_init:
            raise ValueError("spatial_init is only supported for --decoder-type lapfilm")
        return HybridViTConvDecoder(latent_dim=latent_dim).to(device)
    if args.decoder_type == "lapfilm":
        fb = args.decoder_fourier_bands if args.decoder_fourier_bands is not None else 4
        # The base-ch flag sets the first-level channel count; the rest follow
        # the canonical taper [base, base, base*0.75, base*0.5, base*0.375].
        bc = args.decoder_base_ch
        channels = (bc, bc, int(bc * 0.75), int(bc * 0.5), int(bc * 0.375))
        return LapFiLMDecoder(
            latent_dim=latent_dim,
            channels=channels,
            resblocks_per_level=args.decoder_resblocks_per_level,
            upsample=args.decoder_upsample,
            fourier_bands=fb,
            use_film=args.decoder_use_film,
            airfoil_mask_path=args.airfoil_mask_path,
            spatial_init=spatial_init,
        ).to(device)
    if args.decoder_type == "coord_mlp":
        fb = args.decoder_fourier_bands if args.decoder_fourier_bands is not None else 8
        return CoordMLPDecoder(
            latent_dim=latent_dim,
            hidden=args.decoder_mlp_hidden,
            layers=args.decoder_mlp_layers,
            fourier_bands=fb,
            activation=args.decoder_mlp_activation,
            chunk_pixels=args.decoder_mlp_chunk,
        ).to(device)
    raise ValueError(f"unknown decoder-type {args.decoder_type!r}")


def extract_pred(dec_out) -> tuple[torch.Tensor, Optional[list[torch.Tensor]]]:
    """Return (final_pred, pyramid_or_None) from a decoder forward output."""
    if isinstance(dec_out, dict):
        return dec_out["pred"], dec_out.get("pyramid")
    return dec_out, None


def ffl_warmup_factor(step: int, warmup_iters: int, ramp_iters: int) -> float:
    if step < warmup_iters:
        return 0.0
    if ramp_iters <= 0:
        return 1.0
    f = (step - warmup_iters) / ramp_iters
    return max(0.0, min(1.0, f))


def compute_decoder_loss(
    args: argparse.Namespace,
    pred: torch.Tensor,
    pyramid: Optional[list[torch.Tensor]],
    target: torch.Tensor,
    step: int,
    airfoil_mask: Optional[torch.Tensor],
) -> tuple[torch.Tensor, dict]:
    """Dispatch on --decoder-loss and return (loss_scalar, components_dict)."""
    err = target - pred
    if args.decoder_loss == "mse":
        tau = float(args.recon_active_threshold)
        if tau > 0.0:
            active = (target.abs() > tau).to(err.dtype)
            w_in = float(args.recon_inactive_weight)
            weight = active + (1.0 - active) * w_in
            denom = weight.sum().clamp_min(1.0)
            loss = ((err ** 2) * weight).sum() / denom
        else:
            loss = (err ** 2).mean()
        return loss, {"L_total": loss.detach()}
    if args.decoder_loss == "charbonnier":
        eps = float(args.charbonnier_epsilon)
        ch = torch.sqrt(err * err + eps * eps) - eps
        tau = float(args.recon_active_threshold)
        if tau > 0.0:
            active = (target.abs() > tau).to(err.dtype)
            w_in = float(args.recon_inactive_weight)
            weight = active + (1.0 - active) * w_in
            denom = weight.sum().clamp_min(1.0)
            loss = (ch * weight).sum() / denom
        else:
            loss = ch.mean()
        return loss, {"L_total": loss.detach()}
    if args.decoder_loss == "region_pyr_ffl":
        warmup_f = ffl_warmup_factor(step, args.ffl_warmup_iters, args.ffl_ramp_iters)
        pred_pyr = pyramid if pyramid is not None else [pred]
        region_kwargs = dict(
            inactive_weight=args.inactive_weight,
            wake_weight=args.wake_weight,
            active_tau=args.active_tau,
            active_softness=args.active_softness,
        )
        out = region_pyr_ffl_loss(
            pred_pyr, target,
            solid_or_airfoil_mask=airfoil_mask,
            lambda_region=args.lambda_region,
            lambda_pyramid=args.lambda_pyramid,
            lambda_ffl=args.lambda_ffl,
            lambda_enstrophy=args.lambda_enstrophy,
            lambda_circulation=args.lambda_circulation,
            ffl_alpha=args.ffl_alpha,
            ffl_patch=args.ffl_patch,
            ffl_warmup_factor=warmup_f,
            charbonnier_eps=args.charbonnier_epsilon,
            region_kwargs=region_kwargs,
        )
        components = {k: v.detach() for k, v in out.items()}
        components["ffl_warmup_factor"] = torch.tensor(warmup_f)
        return out["L_total"], components
    if args.decoder_loss == "region_pyr_specloss":
        pred_pyr = pyramid if pyramid is not None else [pred]
        region_kwargs = dict(
            inactive_weight=args.inactive_weight,
            wake_weight=args.wake_weight,
            active_tau=args.active_tau,
            active_softness=args.active_softness,
        )
        win = args.spectral_window
        if win == "none":
            win = None
        out = region_pyr_specloss_loss(
            pred_pyr, target,
            solid_or_airfoil_mask=airfoil_mask,
            lambda_region=args.lambda_region,
            lambda_pyramid=args.lambda_pyramid,
            lambda_gradient=args.lambda_gradient,
            lambda_spectral_amp=args.lambda_spectral_amp,
            lambda_enstrophy=args.lambda_enstrophy,
            lambda_circulation=args.lambda_circulation,
            spectral_wake_only=args.spectral_wake_only,
            spectral_window=win,
            spectral_tukey_alpha=args.spectral_tukey_alpha,
            charbonnier_eps=args.charbonnier_epsilon,
            region_kwargs=region_kwargs,
        )
        components = {k: v.detach() for k, v in out.items()}
        return out["L_total"], components
    raise ValueError(f"unknown decoder-loss {args.decoder_loss!r}")


def main() -> None:
    args = parse_args()
    device = require_rtx6000(gpu_index=args.gpu)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "decoder_train.log"

    def log(msg: str) -> None:
        line = msg if msg.endswith("\n") else msg + "\n"
        with open(log_path, "a") as f:
            f.write(line)
        print(msg, flush=True)

    log(f"[decoder-train] device={device} gpu={torch.cuda.get_device_name(device.index)}")
    log(f"[decoder-train] input_mode={args.input_mode}")
    if args.input_mode == "latent":
        if args.jepa_checkpoint is None and args.encoder_run is None:
            raise SystemExit(
                "error: --input-mode latent requires --jepa-checkpoint or --encoder-run"
            )
        ckpt_path: Optional[Path] = resolve_encoder_checkpoint(args, log)
        log(f"[decoder-train] jepa_checkpoint={ckpt_path}")
        log(f"[decoder-train] jepa_checkpoint sha256={file_sha256(ckpt_path)}")
    else:
        if args.jepa_checkpoint is not None or args.encoder_run is not None:
            raise SystemExit(
                "error: --input-mode omega_direct must not be combined with "
                "--jepa-checkpoint or --encoder-run (the encoder is built fresh)"
            )
        if args.decoder_type != "lapfilm":
            raise SystemExit(
                "error: --input-mode omega_direct currently only supports "
                "--decoder-type lapfilm (Session 11 Track 0.1)"
            )
        ckpt_path = None
        log("[decoder-train] omega_direct: building PatchPoolEncoder + LapFiLM "
            "with spatial_init")
    log(f"[decoder-train] decoder_type={args.decoder_type} "
        f"decoder_loss={args.decoder_loss}")

    if args.decoder_cond != "none":
        raise NotImplementedError(
            f"--decoder-cond {args.decoder_cond!r} is Session-11 work (E3 deferred). "
            "Session 10 only supports 'none'."
        )

    if args.recon_loss_type is not None:
        # Session 9 backwards-compatible alias for --decoder-loss.
        args.decoder_loss = args.recon_loss_type
        log(f"[decoder-train] --recon-loss-type {args.recon_loss_type!r} "
            f"mapped to --decoder-loss {args.decoder_loss!r}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.input_mode == "omega_direct":
        bc = args.decoder_base_ch
        enc = PatchPoolEncoder(in_channels=1, out_channels=bc).to(device)
        d = bc * 12 * 6
        enc.train()
        log(f"[decoder-train] PatchPoolEncoder built: out_channels={bc}, d={d}, "
            f"params={sum(p.numel() for p in enc.parameters()):,} (TRAINABLE)")
    else:
        enc, d = load_encoder(ckpt_path, device)
        log(f"[decoder-train] encoder loaded, d={d}, params="
            f"{sum(p.numel() for p in enc.parameters()):,} (FROZEN)")

    omega_pipeline = None
    if args.omega_pipeline_manifest is not None:
        from src.data.omega_pipeline import OmegaPipeline
        manifest = Path(args.omega_pipeline_manifest)
        if not manifest.is_absolute():
            manifest = REPO / manifest
        omega_pipeline = OmegaPipeline.from_manifest(manifest)
        log(f"[decoder-train] loaded omega pipeline from {manifest}")
        log(f"  mask: {int(omega_pipeline.mask.sum().item())} cells, "
            f"thresholds: {sum(len(v) for v in omega_pipeline.thresholds.values())} encs, "
            f"std={omega_pipeline.train_stats.std:.4f}")

    dec = build_decoder(
        args,
        latent_dim=d,
        device=device,
        spatial_init=(args.input_mode == "omega_direct"),
    )
    log(f"[decoder-train] decoder params={sum(p.numel() for p in dec.parameters()):,}")

    # Optional airfoil mask for region_pyr_ffl loss (the mask is also
    # consumed inside LapFiLMDecoder; here we expose it to the loss).
    airfoil_mask = None
    if args.decoder_loss == "region_pyr_ffl":
        mask_path = args.airfoil_mask_path or str(
            REPO / "outputs" / "data_pipeline" / "v1" / "airfoil_adjacent_mask.npy"
        )
        if Path(mask_path).exists():
            mask_np = np.load(mask_path).astype(np.float32)
            airfoil_mask = torch.from_numpy(mask_np).to(device)
            log(f"[decoder-train] loaded airfoil mask from {mask_path}, "
                f"{int(airfoil_mask.sum().item())} cells")

    train_encs = gather_train_encounters()
    test_a_encs = gather_eval_encounters("test_a")
    test_b_encs = gather_eval_encounters("test_b")
    test_c_encs = gather_eval_encounters("test_c")
    log(f"[decoder-train] train={len(train_encs)} encs, "
        f"test_a={len(test_a_encs)}, test_b={len(test_b_encs)}, test_c={len(test_c_encs)}")

    ds = EncounterFrameDataset(train_encs, T=args.T, seed=args.seed,
                                omega_pipeline=omega_pipeline)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.B, shuffle=True, num_workers=args.num_workers,
        collate_fn=collate, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    it = iter(loader)

    trainable_params = list(dec.parameters())
    if args.input_mode == "omega_direct":
        trainable_params += list(enc.parameters())
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=build_lr_lambda(args.max_iters, args.warmup_frac),
    )

    metrics_path = out_dir / "decoder_metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    for step in range(args.max_iters + 1):
        try:
            x = next(it)
        except StopIteration:
            it = iter(loader)
            x = next(it)

        train_encoder = args.input_mode == "omega_direct"
        z = encode_batch(enc, x, device, train_encoder=train_encoder)  # (B, T, d)
        # Target. When pipeline is set, x is already normalized; otherwise
        # divide by --omega-scale. Loss is computed in this normalized space.
        if omega_pipeline is not None:
            target = x.to(device).unsqueeze(2)  # (B, T, 1, H, W), normalized
        else:
            target = (x.to(device).unsqueeze(2) / args.omega_scale)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            dec_out = dec(z)
            pred, pyramid = extract_pred(dec_out)
        pred_f = pred.float()
        target_f = target.float()
        if pyramid is not None:
            pyramid = [p.float() for p in pyramid]
        loss, comps = compute_decoder_loss(
            args, pred_f, pyramid, target_f, step, airfoil_mask,
        )

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
        opt.step()
        sched.step()

        if step % args.log_every == 0:
            parts = [f"L_total={float(comps.get('L_total', loss)):.4f}"]
            for k in ("L_region", "L_pyramid", "L_ffl",
                      "L_enstrophy", "L_circulation", "ffl_warmup_factor"):
                if k in comps:
                    parts.append(f"{k}={float(comps[k]):.4f}")
            log(f"[iter {step}/{args.max_iters}] " + " ".join(parts)
                + f" lr={sched.get_last_lr()[0]:.2e}")

        if step > 0 and step % args.eval_every == 0:
            if args.input_mode == "omega_direct":
                enc.eval()
            ev_a = evaluate_split(enc, dec, test_a_encs[:8], device,
                                  omega_scale=args.omega_scale,
                                  omega_pipeline=omega_pipeline)
            log(f"[eval iter {step}] test_a (subset 8): "
                f"mse_mean={ev_a['mse_mean']:.4f} floor_mean={ev_a['floor_mean']:.4f} "
                f"ratio={ev_a['ratio_mean']:.3f}")
            dec.train()
            if args.input_mode == "omega_direct":
                enc.train()
            record = {
                "iter": step,
                "train_loss": float(loss.item()),
                "test_a_subset8": ev_a,
            }
            for k, v in comps.items():
                record[f"train_{k}"] = float(v) if hasattr(v, "item") else float(v)
            with open(metrics_path, "a") as f:
                f.write(json.dumps(record) + "\n")

        if step > 0 and step % args.checkpoint_every == 0:
            ckpt_out = out_dir / f"decoder_iter{step:06d}.pt"
            ckpt_blob = {
                "decoder_state_dict": dec.state_dict(),
                "iter": step,
                "args": vars(args),
            }
            if args.input_mode == "omega_direct":
                ckpt_blob["encoder_state_dict"] = enc.state_dict()
            torch.save(ckpt_blob, ckpt_out)
            log(f"[checkpoint] saved {ckpt_out}")

    # Final full evaluation
    log("[decoder-train] final evaluation on Test A / B / C")
    if args.input_mode == "omega_direct":
        enc.eval()
    ev_a = evaluate_split(enc, dec, test_a_encs, device,
                          omega_scale=args.omega_scale, omega_pipeline=omega_pipeline)
    ev_b = evaluate_split(enc, dec, test_b_encs, device,
                          omega_scale=args.omega_scale, omega_pipeline=omega_pipeline)
    ev_c = evaluate_split(enc, dec, test_c_encs, device,
                          omega_scale=args.omega_scale, omega_pipeline=omega_pipeline)

    summary = {
        "input_mode": args.input_mode,
        "jepa_checkpoint": str(ckpt_path) if ckpt_path is not None else None,
        "jepa_checkpoint_sha256": file_sha256(ckpt_path) if ckpt_path is not None else None,
        "latent_dim": d,
        "iters": args.max_iters,
        "test_a": ev_a,
        "test_b": ev_b,
        "test_c": ev_c,
        "pass_test_a_within_2x_floor": ev_a["ratio_mean"] < 2.0,
    }
    summary_path = out_dir / "decoder_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"[decoder-train] wrote {summary_path}")
    log(f"[decoder-train] FINAL test_a mse={ev_a['mse_mean']:.4f} floor={ev_a['floor_mean']:.4f} "
        f"ratio={ev_a['ratio_mean']:.3f} PASS={summary['pass_test_a_within_2x_floor']}")
    log(f"[decoder-train] FINAL test_b mse={ev_b['mse_mean']:.4f} floor={ev_b['floor_mean']:.4f} "
        f"ratio={ev_b['ratio_mean']:.3f}")
    log(f"[decoder-train] FINAL test_c mse={ev_c['mse_mean']:.4f} floor={ev_c['floor_mean']:.4f} "
        f"ratio={ev_c['ratio_mean']:.3f}")


if __name__ == "__main__":
    main()
