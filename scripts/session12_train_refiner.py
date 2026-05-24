"""Session 12 Direction B: GAN refiner training.

Trains the :class:`WakeRefiner` residual refiner (and a
:class:`PatchGANDiscriminator`) on top of the frozen Session 11
``W0_C_lam100`` encoder + E1 LapFiLM decoder. The encoder AND the E1
decoder are FROZEN during refiner training; only the refiner and the
discriminator receive gradients.

Training pipeline (per step)
----------------------------
1. ``x`` = ground-truth omega sub-trajectory ``(B, T, H, W)`` (normalised
   by the pipeline manifest).
2. ``z = encoder(x)`` -- frozen, ``(B, T, d)``.
3. ``dec_out = E1_decoder(z)`` -- frozen. ``coarse = dec_out["pred"]`` is
   the 192x96 prediction; ``pyramid = dec_out["pyramid"]`` is the list of
   five resolution levels.
4. ``residual = refiner(coarse_flat, wake_mask)`` and
   ``refined_flat = coarse_flat + residual``. The refined prediction
   REPLACES the final pyramid level only; lower levels are reused from
   the frozen decoder.
5. Generator update (refiner):
   ``L_recon = region_pyr_ffl_loss(pyramid_refined, target, E1 weights)``;
   ``L_adv = - mean(D(refined, wake_mask))`` (hinge);
   ``L_gen = L_recon + lambda_adv * L_adv``.
6. Discriminator update (hinge):
   ``L_disc = mean(relu(1 - D(target, mask)))
              + mean(relu(1 + D(refined.detach(), mask)))``.

A discriminator warmup (default 1000 iters) lets the refiner train with
reconstruction loss only before the adversarial signal turns on; this
matches the conservative-GAN protocol called out in
``SESSION12_CRISP_WAKE.md`` Direction B.

Usage::

    python -m scripts.session12_train_refiner \\
        --encoder-run outputs/runs/session11/W0_C_lam100 \\
        --decoder-checkpoint \\
            outputs/runs/session11/W0_C_lam100/decoder_E1_recipe/decoder_iter020000.pt \\
        --omega-pipeline-manifest outputs/data_pipeline/v1/manifest.json \\
        --output-dir outputs/runs/session12/S12_B_gan_refine \\
        --B 16 --T 32 --max-iters 20000 --gpu 0 --seed 42

Hardware: RTX 6000 Blackwell only (``require_rtx6000`` at entry).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.evaluation.decoder_metrics import wake_mask  # noqa: E402
from src.models.decoder_losses import region_pyr_ffl_loss  # noqa: E402
from src.models.discriminator import PatchGANDiscriminator  # noqa: E402
from src.models.encoder import HybridCNNViTEncoder  # noqa: E402
from src.models.lap_film_decoder import LapFiLMDecoder  # noqa: E402
from src.models.refiner import WakeRefiner  # noqa: E402
from src.utils.device import require_rtx6000  # noqa: E402


PREVENT = Path(os.environ.get("PREVENT_ROOT", "/home/carlos/PREVENT"))
CACHE = Path(os.environ.get("VORTEX_JEPA_CACHE", PREVENT / "data" / "processed" / "vortex-jepa"))


# -----------------------------------------------------------------------------
# checkpoint / data utilities (mirror scripts/session9_train_decoder.py)
# -----------------------------------------------------------------------------


def file_sha256(p: Path) -> str:
    """Stream a file through SHA-256 in 1 MiB chunks."""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_encoder(ckpt_path: Path, device: torch.device) -> tuple[HybridCNNViTEncoder, int]:
    """Load the frozen JEPA encoder used to produce the latent ``z``."""
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
        print(f"[refiner-train] WARNING: unexpected encoder keys ignored: {unexpected}",
              flush=True)
    enc.eval().to(device)
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc, int(args["d"])


def load_decoder(
    ckpt_path: Path,
    latent_dim: int,
    device: torch.device,
) -> LapFiLMDecoder:
    """Load the frozen E1 LapFiLM decoder checkpoint."""
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = blob["args"]
    fb = args.get("decoder_fourier_bands") if args.get("decoder_fourier_bands") is not None else 4
    bc = int(args.get("decoder_base_ch", 64))
    channels = (bc, bc, int(bc * 0.75), int(bc * 0.5), int(bc * 0.375))
    dec = LapFiLMDecoder(
        latent_dim=latent_dim,
        channels=channels,
        resblocks_per_level=int(args.get("decoder_resblocks_per_level", 2)),
        upsample=args.get("decoder_upsample", "pixelshuffle"),
        fourier_bands=fb,
        use_film=bool(args.get("decoder_use_film", True)),
        airfoil_mask_path=args.get("airfoil_mask_path"),
        spatial_init=False,
    )
    dec.load_state_dict(blob["decoder_state_dict"], strict=True)
    dec.eval().to(device)
    for p in dec.parameters():
        p.requires_grad_(False)
    return dec


def gather_train_encounters() -> list[dict]:
    """Training pool = all non-test-a encounters from ``split == train`` cases."""
    with open(REPO / "configs" / "splits" / "split_v1.json") as f:
        manifest = json.load(f)
    out = []
    for cid, case in manifest["cases"].items():
        if case["split"] != "train":
            continue
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
    """Held-out encounters for the named split (``test_a``, ``test_b``, ``test_c``)."""
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
    """Yields a random ``T``-frame omega sub-trajectory for one encounter.

    Mirrors :class:`scripts.session9_train_decoder.EncounterFrameDataset`
    -- pipeline preprocessing runs per worker (Session 11 D85 fix).
    """

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
        x = omega[start: start + self.T]
        if self.omega_pipeline is not None:
            x = self.omega_pipeline.preprocess_raw(x, e["case_id"], int(e["k"]))
            x = self.omega_pipeline.normalize(x)
        return torch.from_numpy(x)


def collate(batch: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(batch, dim=0)  # (B, T, H, W)


# -----------------------------------------------------------------------------
# evaluation
# -----------------------------------------------------------------------------


def _ssim(x: np.ndarray, y: np.ndarray, c1: float = 0.16, c2: float = 1.44) -> float:
    """Fukami's SSIM definition (arXiv:2305.18394 Eq. 1) on a (H, W) pair."""
    mu_x, mu_y = x.mean(), y.mean()
    var_x, var_y = x.var(), y.var()
    cov_xy = ((x - mu_x) * (y - mu_y)).mean()
    num = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
    return float(num / max(den, 1e-12))


def _l2_relative_error(q: np.ndarray, q_hat: np.ndarray, eps: float = 1.0) -> float:
    """Fukami's L2 relative reconstruction error with a raw-units eps floor."""
    num = float(np.sqrt(((q - q_hat) ** 2).sum()))
    den = float(np.sqrt((q ** 2).sum()))
    return num / max(den, eps)


def evaluate_split(
    enc: HybridCNNViTEncoder,
    dec: LapFiLMDecoder,
    refiner: WakeRefiner,
    encs: list[dict],
    device: torch.device,
    wake_mask_t: torch.Tensor,
    omega_pipeline=None,
) -> dict:
    """Per-encounter reconstruction quality on a split for the refined output.

    The encoder and decoder run unchanged from the frozen E1 baseline;
    the refiner residual is added to the final 192x96 prediction.
    Metrics are computed on RAW-SCALE omega (the pipeline output is
    un-normalised before computing MSE / SSIM / eps).
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
    eps_volume = []
    refiner.eval()
    dec.eval()
    with torch.no_grad():
        for e in encs:
            with h5py.File(e["path"], "r") as f:
                omega = np.asarray(f["omega_z"], dtype=np.float32)
            if omega_pipeline is not None:
                omega = omega_pipeline.preprocess_raw(omega, e["case_id"], int(e["k"]))
            T = omega.shape[0]
            x = torch.from_numpy(omega).unsqueeze(0).unsqueeze(2).to(device)
            if omega_pipeline is not None:
                x = omega_pipeline.normalize(x)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                z = enc(x)
                dec_out = dec(z)
                coarse = dec_out["pred"]  # (1, T, 1, H, W)
                B_, T_, C_, H_, W_ = coarse.shape
                coarse_flat = coarse.reshape(B_ * T_, C_, H_, W_)
                residual = refiner(coarse_flat.float(), wake_mask_t)
                refined_flat = coarse_flat.float() + residual
                x_hat = refined_flat.reshape(B_, T_, C_, H_, W_)
                if omega_pipeline is not None:
                    x_hat = omega_pipeline.unnormalize(x_hat)
            x_hat = x_hat.float().squeeze(0).squeeze(1).cpu().numpy()  # (T, H, W)
            mse = float(((omega - x_hat) ** 2).mean())
            floor = float(((omega - case_mean[e["case_id"]]) ** 2).mean())
            ssim_t = float(np.mean([_ssim(omega[t], x_hat[t]) for t in range(T)]))
            eps_v = _l2_relative_error(omega, x_hat)
            mses.append(mse)
            floors.append(floor)
            ssims.append(ssim_t)
            eps_volume.append(eps_v)
    return {
        "mse_mean": float(np.mean(mses)),
        "mse_median": float(np.median(mses)),
        "ssim_mean": float(np.mean(ssims)),
        "ssim_median": float(np.median(ssims)),
        "eps_volume_mean": float(np.mean(eps_volume)),
        "eps_volume_median": float(np.median(eps_volume)),
        "floor_mean": float(np.mean(floors)),
        "ratio_mean": float(np.mean(mses) / max(np.mean(floors), 1e-12)),
        "n_encounters": len(encs),
    }


# -----------------------------------------------------------------------------
# training utilities
# -----------------------------------------------------------------------------


def build_lr_lambda(max_iters: int, warmup_frac: float):
    """Linear warmup + cosine decay down to 5% of peak."""
    warmup = max(1, int(warmup_frac * max_iters))

    def fn(step: int) -> float:
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, max_iters - warmup)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * prog))
    return fn


def init_discriminator_weights(disc: PatchGANDiscriminator, std: float = 0.02) -> None:
    """Small-variance init for the patchGAN to avoid early saturation.

    The spectral-norm parametrisation rescales the conv weight on every
    forward pass, but the underlying ``weight_orig`` tensor (the
    learnable parameter) is still initialised with Kaiming-uniform by
    default; that produces large logits early in training. The pix2pix
    convention is ``Normal(0, 0.02)`` for the original weight, which
    keeps the discriminator output near zero at init and gives the
    hinge loss room to provide a useful gradient.
    """
    for m in disc.modules():
        if isinstance(m, nn.Conv2d):
            target = m.parametrizations.weight.original if hasattr(m, "parametrizations") \
                else m.weight
            nn.init.normal_(target, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Session 12 Direction B: GAN refiner on frozen E1 decoder.",
    )
    enc_group = p.add_mutually_exclusive_group(required=True)
    enc_group.add_argument(
        "--jepa-checkpoint", type=str,
        help="Path to a single JEPA encoder checkpoint .pt file.")
    enc_group.add_argument(
        "--encoder-run", type=str,
        help="Path to a JEPA run directory; the script picks the largest-iter "
             "checkpoint inside.")

    p.add_argument("--decoder-checkpoint", type=str, required=True,
                   help="Path to the frozen E1 LapFiLM decoder checkpoint .pt")

    p.add_argument("--output-dir", required=True, type=str)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--B", type=int, default=16)
    p.add_argument("--T", type=int, default=32)
    p.add_argument("--max-iters", type=int, default=20000)

    # Refiner / discriminator hyperparameters.
    p.add_argument("--lambda-adv", type=float, default=0.05,
                   help="Adversarial loss weight in the generator update.")
    p.add_argument("--lr-refiner", type=float, default=1e-4,
                   help="AdamW learning rate for the refiner (generator).")
    p.add_argument("--lr-disc", type=float, default=4e-4,
                   help="AdamW learning rate for the discriminator (TTUR).")
    p.add_argument("--disc-warmup-iters", type=int, default=1000,
                   help="Iterations before the discriminator joins the loop. "
                        "During warmup the refiner trains on L_recon only.")
    p.add_argument("--refiner-channels", type=int, default=64)
    p.add_argument("--refiner-blocks", type=int, default=6)

    # Optimiser / schedule shared with the decoder script.
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--checkpoint-every", type=int, default=2000)

    # Pipeline manifest (must match the frozen decoder's training pipeline).
    p.add_argument("--omega-pipeline-manifest", type=str, default=None,
                   help="OmegaPipeline manifest path (must match the frozen "
                        "decoder's training-time manifest).")

    # E1 recipe loss weights (Session 11 W0_C_lam100 / decoder_E1_recipe).
    p.add_argument("--lambda-region", type=float, default=1.0)
    p.add_argument("--lambda-pyramid", type=float, default=0.4)
    p.add_argument("--lambda-ffl", type=float, default=0.0)
    p.add_argument("--lambda-enstrophy", type=float, default=0.02)
    p.add_argument("--lambda-circulation", type=float, default=0.01)
    p.add_argument("--charbonnier-epsilon", type=float, default=0.05)
    p.add_argument("--active-tau", type=float, default=0.10)
    p.add_argument("--active-softness", type=float, default=0.03)
    p.add_argument("--inactive-weight", type=float, default=0.05)
    p.add_argument("--wake-weight", type=float, default=0.50)
    p.add_argument("--airfoil-mask-path", type=str, default=None)

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
    log(f"[refiner-train] resolved --encoder-run {run_dir} -> {chosen.name}")
    return chosen


# -----------------------------------------------------------------------------
# main loop
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    device = require_rtx6000(gpu_index=args.gpu)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "refiner_train.log"

    def log(msg: str) -> None:
        line = msg if msg.endswith("\n") else msg + "\n"
        with open(log_path, "a") as f:
            f.write(line)
        print(msg, flush=True)

    log(f"[refiner-train] device={device} gpu={torch.cuda.get_device_name(device.index)}")

    # ---- frozen encoder ----
    enc_ckpt = resolve_encoder_checkpoint(args, log)
    log(f"[refiner-train] encoder_checkpoint={enc_ckpt}")
    log(f"[refiner-train] encoder_checkpoint sha256={file_sha256(enc_ckpt)}")
    enc, latent_dim = load_encoder(enc_ckpt, device)
    log(f"[refiner-train] encoder loaded, d={latent_dim}, params="
        f"{sum(p.numel() for p in enc.parameters()):,} (FROZEN)")

    # ---- frozen decoder ----
    dec_ckpt = Path(args.decoder_checkpoint).resolve()
    if not dec_ckpt.exists():
        raise FileNotFoundError(f"decoder checkpoint not found: {dec_ckpt}")
    log(f"[refiner-train] decoder_checkpoint={dec_ckpt}")
    log(f"[refiner-train] decoder_checkpoint sha256={file_sha256(dec_ckpt)}")
    dec = load_decoder(dec_ckpt, latent_dim, device)
    log(f"[refiner-train] decoder loaded, params="
        f"{sum(p.numel() for p in dec.parameters()):,} (FROZEN)")

    # ---- pipeline (must match decoder training) ----
    omega_pipeline = None
    if args.omega_pipeline_manifest is not None:
        from src.data.omega_pipeline import OmegaPipeline
        manifest = Path(args.omega_pipeline_manifest)
        if not manifest.is_absolute():
            manifest = REPO / manifest
        omega_pipeline = OmegaPipeline.from_manifest(manifest)
        log(f"[refiner-train] loaded omega pipeline from {manifest}")

    # ---- seeds ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # ---- wake mask (cached on device, dtype float32 for autocast safety) ----
    wm_np = wake_mask(192, 96).astype(np.float32)
    wake_mask_t = torch.from_numpy(wm_np).to(device)
    log(f"[refiner-train] wake mask: {int(wake_mask_t.sum().item())} cells "
        f"of {wake_mask_t.numel()}")

    # ---- airfoil mask for the reconstruction loss ----
    airfoil_mask = None
    mask_path = args.airfoil_mask_path or str(
        REPO / "outputs" / "data_pipeline" / "v1" / "airfoil_adjacent_mask.npy"
    )
    if Path(mask_path).exists():
        mask_np = np.load(mask_path).astype(np.float32)
        airfoil_mask = torch.from_numpy(mask_np).to(device)
        log(f"[refiner-train] loaded airfoil mask from {mask_path}, "
            f"{int(airfoil_mask.sum().item())} cells")

    # ---- refiner (trainable) ----
    refiner = WakeRefiner(
        in_channels=1,
        channels=args.refiner_channels,
        n_blocks=args.refiner_blocks,
        use_wake_mask=True,
    ).to(device)
    log(f"[refiner-train] refiner params={sum(p.numel() for p in refiner.parameters()):,}")

    # ---- discriminator (trainable, small-variance init) ----
    disc = PatchGANDiscriminator(in_channels=1, mask_channels=1).to(device)
    init_discriminator_weights(disc, std=0.02)
    log(f"[refiner-train] discriminator params="
        f"{sum(p.numel() for p in disc.parameters()):,}")

    # ---- data ----
    train_encs = gather_train_encounters()
    test_a_encs = gather_eval_encounters("test_a")
    test_b_encs = gather_eval_encounters("test_b")
    test_c_encs = gather_eval_encounters("test_c")
    log(f"[refiner-train] train={len(train_encs)} encs, "
        f"test_a={len(test_a_encs)}, test_b={len(test_b_encs)}, test_c={len(test_c_encs)}")

    ds = EncounterFrameDataset(train_encs, T=args.T, seed=args.seed,
                               omega_pipeline=omega_pipeline)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.B, shuffle=True, num_workers=args.num_workers,
        collate_fn=collate, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    it = iter(loader)

    # ---- optimisers (two-time-scale: refiner < disc) ----
    opt_g = torch.optim.AdamW(
        refiner.parameters(),
        lr=args.lr_refiner, betas=(0.5, 0.999), weight_decay=args.weight_decay,
    )
    opt_d = torch.optim.AdamW(
        disc.parameters(),
        lr=args.lr_disc, betas=(0.5, 0.999), weight_decay=0.0,
    )
    sched_g = torch.optim.lr_scheduler.LambdaLR(
        opt_g, lr_lambda=build_lr_lambda(args.max_iters, args.warmup_frac),
    )
    sched_d = torch.optim.lr_scheduler.LambdaLR(
        opt_d, lr_lambda=build_lr_lambda(args.max_iters, args.warmup_frac),
    )

    metrics_path = out_dir / "refiner_metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    region_kwargs = dict(
        inactive_weight=args.inactive_weight,
        wake_weight=args.wake_weight,
        active_tau=args.active_tau,
        active_softness=args.active_softness,
    )

    for step in range(args.max_iters + 1):
        try:
            x = next(it)
        except StopIteration:
            it = iter(loader)
            x = next(it)

        # Encoder + decoder forward (frozen).
        x = x.to(device, non_blocking=True).unsqueeze(2)  # (B, T, 1, H, W)
        target = x  # already pipeline-normalised
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                z = enc(x)
                dec_out = dec(z)
                coarse = dec_out["pred"]  # (B, T, 1, H, W)
                coarse_pyr = dec_out["pyramid"]  # list of (B, T, 1, h_k, w_k)
        coarse_f = coarse.float()
        coarse_pyr_f = [p.float() for p in coarse_pyr]

        B, T, C, H, W = coarse_f.shape
        coarse_flat = coarse_f.reshape(B * T, C, H, W)
        target_flat = target.float().reshape(B * T, C, H, W)

        # ---- Generator step ----
        refiner.train()
        disc.eval()  # discriminator is queried in inference mode here so its
        # spectral-norm running estimates don't tick from the generator step;
        # they'll update in the discriminator step below.
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            residual = refiner(coarse_flat, wake_mask_t)
        residual_f = residual.float()
        refined_flat = coarse_flat + residual_f
        refined = refined_flat.reshape(B, T, C, H, W)

        # Build the refined pyramid: copy the lower levels from the frozen
        # decoder and replace ONLY the final 192x96 level with the refined
        # prediction.
        refined_pyr = list(coarse_pyr_f)
        refined_pyr[-1] = refined

        recon_out = region_pyr_ffl_loss(
            refined_pyr, target.float(),
            solid_or_airfoil_mask=airfoil_mask,
            lambda_region=args.lambda_region,
            lambda_pyramid=args.lambda_pyramid,
            lambda_ffl=args.lambda_ffl,
            lambda_enstrophy=args.lambda_enstrophy,
            lambda_circulation=args.lambda_circulation,
            ffl_warmup_factor=0.0,  # FFL disabled in the E1 recipe
            charbonnier_eps=args.charbonnier_epsilon,
            region_kwargs=region_kwargs,
        )
        L_recon = recon_out["L_total"]

        disc_active = step >= args.disc_warmup_iters
        if disc_active:
            # Hinge generator term: maximise the patch logits for the refined
            # field. The discriminator is queried in fp32 (no autocast) to keep
            # spectral-norm parametrisations stable; this matches the SAGAN /
            # BigGAN convention.
            d_logits_g = disc(refined_flat, wake_mask_t)
            L_adv = -d_logits_g.mean()
            L_gen = L_recon + args.lambda_adv * L_adv
        else:
            L_adv = torch.zeros((), device=device)
            L_gen = L_recon

        opt_g.zero_grad(set_to_none=True)
        L_gen.backward()
        nn.utils.clip_grad_norm_(refiner.parameters(), args.grad_clip)
        opt_g.step()
        sched_g.step()

        # ---- Discriminator step ----
        if disc_active:
            disc.train()
            with torch.no_grad():
                # Recompute the refined field without grad for the disc step
                # (the generator graph has been freed by L_gen.backward()).
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                    residual_eval = refiner(coarse_flat, wake_mask_t)
                refined_flat_d = coarse_flat + residual_eval.float()

            d_real = disc(target_flat, wake_mask_t)
            d_fake = disc(refined_flat_d.detach(), wake_mask_t)
            L_disc_real = F.relu(1.0 - d_real).mean()
            L_disc_fake = F.relu(1.0 + d_fake).mean()
            L_disc = L_disc_real + L_disc_fake

            opt_d.zero_grad(set_to_none=True)
            L_disc.backward()
            nn.utils.clip_grad_norm_(disc.parameters(), args.grad_clip)
            opt_d.step()
            sched_d.step()
        else:
            L_disc = torch.zeros((), device=device)
            L_disc_real = torch.zeros((), device=device)
            L_disc_fake = torch.zeros((), device=device)

        if step % args.log_every == 0:
            parts = [
                f"L_recon={float(L_recon):.4f}",
                f"L_adv={float(L_adv):.4f}",
                f"L_gen={float(L_gen):.4f}",
                f"L_disc={float(L_disc):.4f}",
            ]
            for k in ("L_region", "L_pyramid", "L_enstrophy", "L_circulation"):
                if k in recon_out:
                    parts.append(f"{k}={float(recon_out[k]):.4f}")
            parts.append(f"lr_g={sched_g.get_last_lr()[0]:.2e}")
            parts.append(f"lr_d={sched_d.get_last_lr()[0]:.2e}")
            parts.append(f"disc_active={int(disc_active)}")
            log(f"[iter {step}/{args.max_iters}] " + " ".join(parts))

        if step > 0 and step % args.eval_every == 0:
            ev_a = evaluate_split(
                enc, dec, refiner, test_a_encs[:8], device,
                wake_mask_t=wake_mask_t, omega_pipeline=omega_pipeline,
            )
            log(f"[eval iter {step}] test_a (subset 8): "
                f"mse_mean={ev_a['mse_mean']:.4f} floor_mean={ev_a['floor_mean']:.4f} "
                f"ratio={ev_a['ratio_mean']:.3f} ssim={ev_a['ssim_mean']:.3f}")
            refiner.train()
            record = {
                "iter": step,
                "L_recon": float(L_recon),
                "L_adv": float(L_adv),
                "L_gen": float(L_gen),
                "L_disc": float(L_disc),
                "L_disc_real": float(L_disc_real),
                "L_disc_fake": float(L_disc_fake),
                "disc_active": int(disc_active),
                "test_a_subset8": ev_a,
            }
            for k, v in recon_out.items():
                record[f"train_{k}"] = float(v) if hasattr(v, "item") else float(v)
            with open(metrics_path, "a") as f:
                f.write(json.dumps(record) + "\n")

        if step > 0 and step % args.checkpoint_every == 0:
            ckpt_out = out_dir / f"refiner_iter{step:06d}.pt"
            rng_state = {
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "numpy_rng": np.random.get_state(),
            }
            ckpt_blob = {
                "refiner_state_dict": refiner.state_dict(),
                "discriminator_state_dict": disc.state_dict(),
                "opt_g_state_dict": opt_g.state_dict(),
                "opt_d_state_dict": opt_d.state_dict(),
                "sched_g_state_dict": sched_g.state_dict(),
                "sched_d_state_dict": sched_d.state_dict(),
                "iter": step,
                "args": vars(args),
                "encoder_checkpoint": str(enc_ckpt),
                "decoder_checkpoint": str(dec_ckpt),
                "rng_state": rng_state,
            }
            torch.save(ckpt_blob, ckpt_out)
            log(f"[checkpoint] saved {ckpt_out}")

    # Final full evaluation on each split.
    log("[refiner-train] final evaluation on Test A / B / C")
    ev_a = evaluate_split(enc, dec, refiner, test_a_encs, device,
                          wake_mask_t=wake_mask_t, omega_pipeline=omega_pipeline)
    ev_b = evaluate_split(enc, dec, refiner, test_b_encs, device,
                          wake_mask_t=wake_mask_t, omega_pipeline=omega_pipeline)
    ev_c = evaluate_split(enc, dec, refiner, test_c_encs, device,
                          wake_mask_t=wake_mask_t, omega_pipeline=omega_pipeline)

    summary = {
        "encoder_checkpoint": str(enc_ckpt),
        "encoder_checkpoint_sha256": file_sha256(enc_ckpt),
        "decoder_checkpoint": str(dec_ckpt),
        "decoder_checkpoint_sha256": file_sha256(dec_ckpt),
        "latent_dim": latent_dim,
        "iters": args.max_iters,
        "lambda_adv": args.lambda_adv,
        "disc_warmup_iters": args.disc_warmup_iters,
        "test_a": ev_a,
        "test_b": ev_b,
        "test_c": ev_c,
    }
    summary_path = out_dir / "refiner_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"[refiner-train] wrote {summary_path}")
    log(f"[refiner-train] FINAL test_a mse={ev_a['mse_mean']:.4f} "
        f"floor={ev_a['floor_mean']:.4f} ratio={ev_a['ratio_mean']:.3f}")
    log(f"[refiner-train] FINAL test_b mse={ev_b['mse_mean']:.4f} "
        f"floor={ev_b['floor_mean']:.4f} ratio={ev_b['ratio_mean']:.3f}")
    log(f"[refiner-train] FINAL test_c mse={ev_c['mse_mean']:.4f} "
        f"floor={ev_c['floor_mean']:.4f} ratio={ev_c['ratio_mean']:.3f}")


if __name__ == "__main__":
    main()
