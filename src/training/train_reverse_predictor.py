"""Train a reverse-factorisation predictor: forces -> encoder latent.

Session 14 Thrust 5 (SESSION14_JFM_NATCOMM_PUSH.md). The reverse
predictor is a causal transformer that maps the per-frame force
coefficients ``(C_L(t), C_D(t))`` plus the static episode descriptor
``c = (G, D, Y)`` to the corresponding frozen-encoder latent
``z_t = E(omega_t)`` produced by the Session 12 Direction E (d=64)
encoder. The encoder weights are read from
``outputs/runs/session12/S12_E_d64/encoder/checkpoint_iter020000.pt``
and held frozen; the targets are loaded from the precomputed
``outputs/session14/latents/S12_E_d64/{train,test_a}.npz`` latent
caches (``z_full`` field, shape ``(N_enc, 120, 64)``).

Training data
-------------
- One sample = a sub-trajectory of length ``T`` (default 32) drawn from
  one cached encounter. The full encounter holds 120 frames at
  ``dt_tc = 0.05``.
- Start frame is sampled with the same impact-aware mixture as
  :class:`src.data.episode_dataset.EpisodeDataset`: 70 % from
  ``[8, 40]`` (intersects the impact window [25, 55] by >= 7 frames),
  30 % uniform over ``[0, n_frames - T]``.
- Inputs: ``(C_L, C_D)`` slice from the encounter cache file.
- Targets: ``z_full[start:start + T]`` from the latent cache.

Loss
----
Per-frame MSE between predicted ``z_hat`` and oracle latent ``z`` over
the sub-trajectory, summed across the latent dimension and averaged
over (batch, time). Computed in fp32 outside the bf16 autocast region
to avoid the BatchNorm float-bf16 interaction observed in the encoder
projector path (HANDOFF.md D14).

Optimiser
---------
AdamW (0.9, 0.95), lr 5e-4, weight_decay 0.05. Cosine LR with 5 %
linear warmup, decayed to 0.05 * peak. Gradient clip 1.0. bf16
autocast over the forward pass; backward in fp32. 20000 iterations
match the forward-predictor sister job in Thrust 6.

Hardware
--------
RTX 6000 Blackwell only. ``require_rtx6000(gpu_index=args.gpu)`` is the
first call in :func:`main`; CPU fallback is forbidden per CLAUDE.md
"Hardware".

Usage
-----
    python -m src.training.train_reverse_predictor \\
        --encoder-checkpoint outputs/runs/session12/S12_E_d64/encoder/checkpoint_iter020000.pt \\
        --latents-dir outputs/session14/latents/S12_E_d64 \\
        --output-dir outputs/runs/session14/thrust5_reverse \\
        --max-iters 20000 --B 16 --T 32 --seed 42 --gpu 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.models.encoder import HybridCNNViTEncoder  # noqa: E402
from src.models.predictor import ReversePredictor  # noqa: E402
from src.utils.device import require_rtx6000  # noqa: E402


PREVENT = Path(os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT")))
CACHE = Path(
    os.environ.get("VORTEX_JEPA_CACHE", PREVENT / "data" / "processed" / "vortex-jepa")
)
SPLIT_MANIFEST_PATH = REPO / "configs" / "splits" / "split_v1.json"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, stderr=subprocess.DEVNULL
        )
        return out.strip().decode()
    except Exception:
        return "unknown"


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Session 14 Thrust 5 reverse predictor (forces -> latent)"
    )
    p.add_argument(
        "--encoder-checkpoint",
        type=str,
        default=(
            "outputs/runs/session12/S12_E_d64/encoder/checkpoint_iter020000.pt"
        ),
        help="Frozen encoder checkpoint (S12_E_d64 production winner per D99).",
    )
    p.add_argument(
        "--latents-dir",
        type=str,
        default="outputs/session14/latents/S12_E_d64",
        help=(
            "Directory containing train.npz / test_a.npz with z_full latent "
            "trajectories from the frozen encoder."
        ),
    )
    p.add_argument(
        "--partition",
        type=str,
        default="v1",
        help="Partition name; resolves cache and split paths.",
    )
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--B", type=int, default=16)
    p.add_argument("--T", type=int, default=32)
    p.add_argument("--max-iters", type=int, default=20_000)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--diagnostic-every", type=int, default=500)
    p.add_argument("--checkpoint-every", type=int, default=2000)
    p.add_argument(
        "--impact-aware-fraction",
        type=float,
        default=0.7,
        help="Probability of impact-aware start sampling (D2).",
    )
    p.add_argument(
        "--impact-overlap-start-range",
        type=int,
        nargs=2,
        default=[8, 40],
        help="Inclusive [lo, hi] start frame for impact-aware sampling.",
    )
    p.add_argument(
        "--hidden-dim",
        type=int,
        default=384,
        help="ReversePredictor hidden width (matches AutoregressivePredictor).",
    )
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument(
        "--standardize-forces",
        action="store_true",
        default=True,
        help=(
            "Standardise (C_L, C_D) with train-pool stats before feeding to "
            "the predictor (helps the bf16 transformer ingest forces in "
            "[-1, 1]-ish range)."
        ),
    )
    p.add_argument(
        "--no-standardize-forces",
        action="store_false",
        dest="standardize_forces",
    )
    p.add_argument(
        "--tag-suffix",
        type=str,
        default="thrust5_reverse",
        help="W&B run-tag suffix.",
    )
    p.add_argument(
        "--wandb-mode",
        type=str,
        choices=["online", "offline", "disabled"],
        default="offline",
    )
    return p.parse_args()


def gather_train_encounters(partition: str) -> list[dict]:
    """Enumerate every encounter used by Thrust 5 training.

    Pool = train_encounter_indices over all cases tagged ``split == 'train'``
    in the partition manifest. Mirrors the JEPA encoder's train pool.
    """
    with open(SPLIT_MANIFEST_PATH) as f:
        manifest = json.load(f)
    out: list[dict] = []
    for cid, case in manifest["cases"].items():
        if case.get("split") != "train":
            continue
        for k in case["train_encounter_indices"]:
            out.append(
                {
                    "case_id": cid,
                    "k": int(k),
                    "G": float(case["G"]),
                    "D": float(case["D"]),
                    "Y": float(case["Y"]),
                }
            )
    return out


def gather_test_a_encounters(partition: str) -> list[dict]:
    """Enumerate the held-out test_a encounters used for diagnostics."""
    with open(SPLIT_MANIFEST_PATH) as f:
        manifest = json.load(f)
    out: list[dict] = []
    for cid, case in manifest["cases"].items():
        if case.get("split") != "train":
            continue
        for k in case["test_a_encounter_indices"]:
            out.append(
                {
                    "case_id": cid,
                    "k": int(k),
                    "G": float(case["G"]),
                    "D": float(case["D"]),
                    "Y": float(case["Y"]),
                }
            )
    return out


def load_latent_pool(latents_dir: Path, split: str) -> dict[tuple[str, int], np.ndarray]:
    """Map ``(case_id, encounter_index) -> z_full[120, d]`` for a split.

    The latent npz holds the columns produced by
    ``scripts/session14_encode_latents.py``: ``z_full`` of shape
    ``(N_enc, 120, latent_dim)`` aligned with the ``case_id`` and
    ``encounter_index`` columns.
    """
    path = latents_dir / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"latent cache missing: {path}")
    blob = np.load(path, allow_pickle=True)
    z_full = blob["z_full"]
    case_ids = blob["case_id"]
    enc_idx = blob["encounter_index"]
    out: dict[tuple[str, int], np.ndarray] = {}
    for i in range(z_full.shape[0]):
        out[(str(case_ids[i]), int(enc_idx[i]))] = np.asarray(z_full[i], dtype=np.float32)
    return out


def load_force_stats(
    encs: list[dict], partition: str
) -> tuple[np.ndarray, np.ndarray]:
    """Compute pooled mean/std for (C_L, C_D) over the training encounters.

    Used to z-score the predictor inputs so the bf16 transformer sees
    inputs in a reasonable range (C_L spans roughly [-2, 4] across the
    cube; C_D ~ [0, 5]).
    """
    sums = np.zeros(2, dtype=np.float64)
    sumsqs = np.zeros(2, dtype=np.float64)
    n = 0
    cache_dir = CACHE / partition
    for e in encs:
        path = cache_dir / e["case_id"] / f"encounter_{e['k']:02d}.h5"
        if not path.exists():
            continue
        with h5py.File(path, "r") as g:
            cl = np.asarray(g["C_L"][:], dtype=np.float64)
            cd = np.asarray(g["C_D"][:], dtype=np.float64)
        sums[0] += cl.sum()
        sums[1] += cd.sum()
        sumsqs[0] += (cl * cl).sum()
        sumsqs[1] += (cd * cd).sum()
        n += cl.shape[0]
    if n == 0:
        raise RuntimeError("no encounters with cached forces; cannot fit stats")
    mean = sums / n
    var = np.clip(sumsqs / n - mean * mean, 1e-12, None)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


class ReversePredictorDataset(Dataset):
    """Pair (C_L, C_D) sub-trajectories with frozen-encoder oracle latents.

    Each sample = one randomly-started length-``T`` sub-trajectory of one
    encounter. Returns a dict with::

        forces  torch.float32 (T, 2)         standardised if requested
        z       torch.float32 (T, d)         oracle latent
        c       torch.float32 (3,)           (G, D, Y)

    The latent pool is held in RAM (180 encounters * 120 frames * 64 dims
    * 4 bytes = ~5.5 MB for train); cheap. C_L / C_D are streamed from the
    HDF5 cache per __getitem__ to avoid pre-loading 180 * 120 floats * 2
    channels (still tiny but consistent with EpisodeDataset).
    """

    def __init__(
        self,
        encs: list[dict],
        latents: dict[tuple[str, int], np.ndarray],
        partition: str,
        subtraj_len: int,
        impact_aware_fraction: float,
        impact_overlap_start_range: tuple[int, int],
        n_frames_per_encounter: int = 120,
        force_mean: np.ndarray | None = None,
        force_std: np.ndarray | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.cache_dir = CACHE / partition
        self.subtraj_len = int(subtraj_len)
        self.n_frames = int(n_frames_per_encounter)
        if self.subtraj_len > self.n_frames:
            raise ValueError("subtraj_len > frames_per_encounter")
        self.impact_aware_fraction = float(impact_aware_fraction)
        max_valid_start = self.n_frames - self.subtraj_len
        lo, hi = impact_overlap_start_range
        self.impact_overlap_start_range = (int(lo), min(int(hi), max_valid_start))
        self.uniform_start_range = (0, max_valid_start)
        self.force_mean = force_mean
        self.force_std = force_std
        self._seed = seed
        self.samples: list[dict] = []
        for e in encs:
            key = (e["case_id"], e["k"])
            if key not in latents:
                continue
            path = self.cache_dir / e["case_id"] / f"encounter_{e['k']:02d}.h5"
            if not path.exists():
                continue
            entry = dict(e)
            entry["latent"] = latents[key]
            entry["path"] = path
            self.samples.append(entry)
        if not self.samples:
            raise RuntimeError("no encounters paired between latents and force cache")

    def __len__(self) -> int:
        return len(self.samples)

    def _make_rng(self, idx: int) -> np.random.Generator:
        if self._seed is None:
            return np.random.default_rng()
        return np.random.default_rng(self._seed * 100003 + idx)

    def _sample_start(self, rng: np.random.Generator) -> int:
        if rng.random() < self.impact_aware_fraction:
            lo, hi = self.impact_overlap_start_range
        else:
            lo, hi = self.uniform_start_range
        return int(rng.integers(lo, hi + 1))

    def __getitem__(self, idx: int) -> dict:
        e = self.samples[idx]
        rng = self._make_rng(idx)
        start = self._sample_start(rng)
        end = start + self.subtraj_len
        with h5py.File(e["path"], "r") as g:
            cl = g["C_L"][start:end].astype(np.float32)
            cd = g["C_D"][start:end].astype(np.float32)
        forces = np.stack([cl, cd], axis=-1)  # (T, 2)
        if self.force_mean is not None and self.force_std is not None:
            forces = (forces - self.force_mean) / self.force_std
        z = e["latent"][start:end].astype(np.float32)  # (T, d)
        c = np.asarray([e["G"], e["D"], e["Y"]], dtype=np.float32)
        return {
            "forces": torch.from_numpy(forces),
            "z": torch.from_numpy(z),
            "c": torch.from_numpy(c),
            "case_id": e["case_id"],
            "encounter_index": int(e["k"]),
            "frame_start": int(start),
        }


def reverse_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    forces = torch.stack([s["forces"] for s in samples])
    z = torch.stack([s["z"] for s in samples])
    c = torch.stack([s["c"] for s in samples])
    return {
        "forces": forces,
        "z": z,
        "c": c,
        "case_ids": [s["case_id"] for s in samples],
    }


def infinite_iter(loader: DataLoader) -> Iterator[dict[str, Tensor]]:
    while True:
        for batch in loader:
            yield batch


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def build_lr_lambda(args: argparse.Namespace) -> "callable[[int], float]":
    warmup_iters = max(1, int(args.warmup_frac * args.max_iters))
    floor = 0.05

    def lr_lambda(step: int) -> float:
        if step < warmup_iters:
            return float(step + 1) / float(warmup_iters)
        progress = (step - warmup_iters) / max(1, args.max_iters - warmup_iters)
        progress = min(max(progress, 0.0), 1.0)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


def load_encoder_meta(ckpt_path: Path) -> dict:
    """Extract latent_dim and hyperparameters from a frozen encoder ckpt."""
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = blob.get("args", {})
    return {
        "latent_dim": int(args.get("d", 64)),
        "projection_norm": args.get("projection_norm", "batchnorm"),
        "encoder_iter": int(blob.get("iteration", -1)),
        "encoder_run_config": blob.get("run_config", {}),
    }


@torch.no_grad()
def evaluate_test_a(
    model: ReversePredictor,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Per-frame MSE on the held-out test_a sub-trajectories.

    Used as the validation signal (Thrust 5 has no separate "Test B"
    held-out cases; the held-out signal is the contiguous Test A
    encounters from training cases).
    """
    was_training = model.training
    model.eval()
    total = 0.0
    n_elements = 0
    n_nan_skipped = 0
    accum_dim_se = None
    accum_dim_count = None
    latent_dim_local = 0
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            z_hat = model(batch["forces"], batch["c"])
        z_hat = z_hat.float()
        z = batch["z"].float()
        diff = (z_hat - z) ** 2  # (B, T, D)
        finite_mask = torch.isfinite(diff)
        n_nan_skipped += int((~finite_mask).sum().item())
        diff_clean = torch.where(finite_mask, diff, torch.zeros_like(diff))
        total += float(diff_clean.sum().item())
        n_elements += int(finite_mask.sum().item())
        latent_dim_local = z.shape[-1]
        # Per-dim accumulator (sum + count) so the running max is over the full eval, not just the last batch
        dim_sum = diff_clean.sum(dim=(0, 1)).cpu().numpy()
        dim_count = finite_mask.sum(dim=(0, 1)).cpu().numpy()
        if accum_dim_se is None:
            accum_dim_se = dim_sum
            accum_dim_count = dim_count
        else:
            accum_dim_se = accum_dim_se + dim_sum
            accum_dim_count = accum_dim_count + dim_count
    if was_training:
        model.train()
    if n_elements == 0:
        return {"test_a_mse": float("nan"), "test_a_rmse": float("nan"),
                "test_a_n_nan_skipped": n_nan_skipped}
    mse = total / n_elements
    per_dim_mse = np.where(accum_dim_count > 0, accum_dim_se / np.maximum(accum_dim_count, 1), 0.0)
    return {
        "test_a_mse": float(mse),
        "test_a_rmse": float(math.sqrt(mse)),
        "test_a_mse_per_dim_max": float(np.max(per_dim_mse)) if per_dim_mse.size else 0.0,
        "test_a_n_nan_skipped": int(n_nan_skipped),
        "test_a_n_elements_used": int(n_elements),
    }


def main() -> None:
    args = parse_args()
    device = require_rtx6000(gpu_index=args.gpu)
    set_all_seeds(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder_ckpt = Path(args.encoder_checkpoint)
    if not encoder_ckpt.is_absolute():
        encoder_ckpt = REPO / encoder_ckpt
    if not encoder_ckpt.exists():
        raise FileNotFoundError(f"encoder checkpoint missing: {encoder_ckpt}")
    encoder_meta = load_encoder_meta(encoder_ckpt)
    latent_dim = encoder_meta["latent_dim"]

    latents_dir = Path(args.latents_dir)
    if not latents_dir.is_absolute():
        latents_dir = REPO / latents_dir
    train_latents = load_latent_pool(latents_dir, "train")
    test_a_latents = load_latent_pool(latents_dir, "test_a")
    print(
        f"[train_reverse] latent_dim={latent_dim} "
        f"train_pool={len(train_latents)} test_a_pool={len(test_a_latents)}",
        flush=True,
    )

    train_encs = gather_train_encounters(args.partition)
    test_a_encs = gather_test_a_encounters(args.partition)

    if args.standardize_forces:
        force_mean, force_std = load_force_stats(train_encs, args.partition)
    else:
        force_mean, force_std = None, None
    print(
        f"[train_reverse] force_mean={force_mean} force_std={force_std} "
        f"(standardize={args.standardize_forces})",
        flush=True,
    )

    train_ds = ReversePredictorDataset(
        encs=train_encs,
        latents=train_latents,
        partition=args.partition,
        subtraj_len=args.T,
        impact_aware_fraction=args.impact_aware_fraction,
        impact_overlap_start_range=tuple(args.impact_overlap_start_range),
        force_mean=force_mean,
        force_std=force_std,
        seed=args.seed,
    )
    test_a_ds = ReversePredictorDataset(
        encs=test_a_encs,
        latents=test_a_latents,
        partition=args.partition,
        subtraj_len=args.T,
        impact_aware_fraction=0.0,
        impact_overlap_start_range=tuple(args.impact_overlap_start_range),
        force_mean=force_mean,
        force_std=force_std,
        seed=args.seed + 1,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.B,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=reverse_collate,
        drop_last=len(train_ds) >= args.B,
        persistent_workers=args.num_workers > 0,
    )
    test_a_loader = DataLoader(
        test_a_ds,
        batch_size=args.B,
        shuffle=False,
        num_workers=0,
        collate_fn=reverse_collate,
        drop_last=False,
    )

    model = ReversePredictor(
        latent_dim=latent_dim,
        input_dim=2,
        cond_dim=3,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        heads=args.heads,
        dropout=args.dropout,
        max_seq_len=args.T,
        output_norm="none",
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    scheduler = LambdaLR(optimizer, lr_lambda=build_lr_lambda(args))

    gpu_name = torch.cuda.get_device_name(device.index)
    if "RTX" not in gpu_name or "6000" not in gpu_name:
        raise RuntimeError(
            f"Hardware policy violation (CLAUDE.md): gpu_name={gpu_name!r} does not "
            "contain both 'RTX' and '6000'. Run aborted."
        )

    with open(REPO / "configs" / "preprocessing.yaml") as f:
        preprocessing_cfg = yaml.safe_load(f)

    run_config = {
        "preprocessing_version": preprocessing_cfg["preprocessing_version"],
        "partition_version": args.partition,
        "lambda_sigreg": None,
        "seed": args.seed,
        "split_sha256": file_sha256(SPLIT_MANIFEST_PATH),
        "code_sha256": git_commit_hash(),
        "auto_fallback_triggered": False,
        "gpu_name": gpu_name,
        "encoder_checkpoint": str(encoder_ckpt),
        "encoder_checkpoint_sha256": file_sha256(encoder_ckpt),
        "encoder_iter": encoder_meta["encoder_iter"],
        "latent_dim": latent_dim,
        "max_iters": args.max_iters,
        "B": args.B,
        "T": args.T,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_frac": args.warmup_frac,
        "hidden_dim": args.hidden_dim,
        "depth": args.depth,
        "heads": args.heads,
        "dropout": args.dropout,
        "standardize_forces": args.standardize_forces,
        "force_mean": force_mean.tolist() if force_mean is not None else None,
        "force_std": force_std.tolist() if force_std is not None else None,
        "impact_aware_fraction": args.impact_aware_fraction,
        "tag_suffix": args.tag_suffix,
        "n_train_samples": len(train_ds),
        "n_test_a_samples": len(test_a_ds),
    }

    import wandb

    wandb_tags = ["reverse_predictor", "thrust5", f"d_{latent_dim}"]
    if args.tag_suffix:
        wandb_tags.append(f"run:{args.tag_suffix}")
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "vortex-jepa"),
        group=f"partition_{args.partition}_thrust5",
        tags=wandb_tags,
        mode=args.wandb_mode,
        config=run_config,
        dir=str(output_dir),
    )
    wandb.run.summary["wandb_run_id"] = wandb.run.id
    run_config["wandb_run_id"] = wandb.run.id

    metrics_jsonl = output_dir / "metrics.jsonl"
    with open(metrics_jsonl, "w") as f:
        f.write(json.dumps({"event": "config", **run_config}) + "\n")

    def _log_metrics(payload: dict[str, Any], step: int) -> None:
        wandb.log(payload, step=step)
        with open(metrics_jsonl, "a") as fh:
            fh.write(json.dumps({"event": "log", "step": int(step), **payload}) + "\n")

    print(
        f"[train_reverse] device={device} gpu={gpu_name} "
        f"n_train={len(train_ds)} n_test_a={len(test_a_ds)} "
        f"hidden={args.hidden_dim} depth={args.depth} heads={args.heads}",
        flush=True,
    )

    model.train()
    train_iter = infinite_iter(train_loader)
    last_loss = float("nan")

    for iteration in range(args.max_iters):
        batch = move_batch(next(train_iter), device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            z_hat = model(batch["forces"], batch["c"])
        # MSE in fp32 for numerical stability; the autocast region is the
        # forward attention/MLP stack only.
        z_hat_f = z_hat.float()
        z = batch["z"].float()
        loss = (z_hat_f - z).pow(2).mean()
        last_loss = float(loss.item())
        if not math.isfinite(last_loss):
            raise RuntimeError(f"non-finite loss at iter {iteration}: {last_loss}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        if iteration % args.log_every == 0:
            payload = {
                "iter": iteration,
                "loss_total": last_loss,
                "loss_mse": last_loss,
                "lr": optimizer.param_groups[0]["lr"],
            }
            _log_metrics(payload, step=iteration)
            print(
                f"[iter {iteration}/{args.max_iters}] mse={last_loss:.6f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e}",
                flush=True,
            )

        if iteration % args.diagnostic_every == 0 and iteration > 0:
            diag = evaluate_test_a(model, test_a_loader, device)
            _log_metrics({f"diag/{k}": v for k, v in diag.items()}, step=iteration)
            print(
                f"[diag iter {iteration}] test_a_mse={diag['test_a_mse']:.6f} "
                f"test_a_rmse={diag['test_a_rmse']:.6f}",
                flush=True,
            )

        if iteration > 0 and iteration % args.checkpoint_every == 0:
            ckpt_path = output_dir / f"checkpoint_iter{iteration:06d}.pt"
            torch.save(
                {
                    "iteration": iteration,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "args": vars(args),
                    "run_config": run_config,
                },
                ckpt_path,
            )
            print(f"[checkpoint] wrote {ckpt_path}", flush=True)

    final_path = output_dir / f"checkpoint_iter{args.max_iters:06d}.pt"
    torch.save(
        {
            "iteration": args.max_iters,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "args": vars(args),
            "run_config": run_config,
        },
        final_path,
    )
    final_diag = evaluate_test_a(model, test_a_loader, device)
    _log_metrics({f"diag_final/{k}": v for k, v in final_diag.items()}, step=args.max_iters)
    print(
        f"[final checkpoint] wrote {final_path}; "
        f"final test_a_mse={final_diag['test_a_mse']:.6f}",
        flush=True,
    )
    wandb.run.summary["loss_total_final"] = last_loss
    wandb.run.summary["test_a_mse_final"] = final_diag["test_a_mse"]
    wandb.run.summary["test_a_rmse_final"] = final_diag["test_a_rmse"]
    wandb.run.summary["final_iter"] = args.max_iters
    wandb.finish()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[train_reverse] interrupted", flush=True)
        sys.exit(130)
