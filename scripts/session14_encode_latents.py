"""One-shot encoding of every encounter through the Session 12 E d=64 encoder.

Loads the production E d=64 JEPA encoder (W0_C_lam100 recipe at d=64,
``outputs/runs/session12/S12_E_d64/encoder/checkpoint_iter020000.pt``),
walks the four splits (train, test_a, test_b, test_c) from
``configs/splits/split_v1.json``, opens each cached encounter HDF5 directly
(bypassing ``EpisodeDataset`` to skip sub-trajectory sampling), applies the
frozen omega pipeline (mask + per-encounter clip + 3-sigma scale), and pushes
the full 120-frame stack through the encoder in bf16.

Writes one ``.npz`` per split to
``outputs/session14/latents/S12_E_d64/{split}.npz`` with arrays:

    z                (n, 64)            impact-frame latent (per CLAUDE.md,
                                        impact_frame_estimate=40 by default;
                                        actual attr per encounter is honoured)
    z_full           (n, 120, 64)       per-frame latents for the full encounter
    G, D, Y          (n,)               case-level conditioning
    case_id          (n,)               case identifier string
    encounter_index  (n,)               encounter index within its case
    split            (n,)               repeated split name
    impact_frame     (n,)               impact frame used for z slicing

Run after ``source .venv/bin/activate && export PREVENT_ROOT=$HOME/PREVENT``::

    python scripts/session14_encode_latents.py

Roughly 5-15 minutes on a single RTX 6000 Blackwell.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.omega_pipeline import OmegaPipeline  # noqa: E402
from src.models.encoder import HybridCNNViTEncoder  # noqa: E402
from src.utils.device import NoRTX6000Error, require_rtx6000  # noqa: E402


CHECKPOINT_PATH = (
    REPO_ROOT / "outputs" / "runs" / "session12" / "S12_E_d64" / "encoder"
    / "checkpoint_iter020000.pt"
)
SPLIT_MANIFEST_PATH = REPO_ROOT / "configs" / "splits" / "split_v1.json"
OMEGA_PIPELINE_MANIFEST = REPO_ROOT / "outputs" / "data_pipeline" / "v1" / "manifest.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "session14" / "latents" / "S12_E_d64"
DEFAULT_IMPACT_FRAME = 40
PARTITION = "v1"
SPLITS: tuple[str, ...] = ("train", "test_a", "test_b", "test_c")


def select_device() -> torch.device:
    """Select cuda:0 (first RTX 6000); fall back to cuda:1 if unavailable.

    CLAUDE.md authoritative: ``--gpu 0`` / ``--gpu 1`` are 0-indexed selectors
    into the RTX 6000 subset, not torch's full enumeration.

    Returns:
        The chosen ``torch.device``.

    Raises:
        NoRTX6000Error: If neither RTX 6000 card is available.
    """
    try:
        device = require_rtx6000(gpu_index=0)
        print(f"[encode] device={device} ({torch.cuda.get_device_name(device.index)})",
              flush=True)
        return device
    except NoRTX6000Error as err0:
        print(f"[encode] gpu_index=0 unavailable ({err0}); falling back to gpu_index=1",
              flush=True)
        device = require_rtx6000(gpu_index=1)
        print(f"[encode] device={device} ({torch.cuda.get_device_name(device.index)})",
              flush=True)
        return device


def resolve_cache_root() -> Path:
    """Return the partition-v1 cache directory.

    Honours ``VORTEX_JEPA_CACHE`` if set, else falls back to
    ``${PREVENT_ROOT}/data/processed/vortex-jepa`` per
    ``configs/preprocessing.yaml::cache.root_default``.
    """
    env_cache = os.environ.get("VORTEX_JEPA_CACHE")
    if env_cache:
        return Path(env_cache) / PARTITION
    prevent_root = Path(os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT")))
    return prevent_root / "data" / "processed" / "vortex-jepa" / PARTITION


def load_encoder(ckpt_path: Path, device: torch.device) -> tuple[HybridCNNViTEncoder, int]:
    """Load the frozen JEPA encoder.

    Mirrors the loading pattern used in
    ``scripts/session12_train_refiner.py::load_encoder``: instantiate the
    encoder with ``latent_dim`` and ``projection_norm`` read from the
    checkpoint's saved args, then filter ``jepa_state_dict`` by the
    ``encoder.`` prefix and load into the bare encoder.

    Args:
        ckpt_path: Path to ``checkpoint_iter*.pt``.
        device: Target device.

    Returns:
        Tuple ``(encoder, latent_dim)``. The encoder is in eval mode,
        on ``device``, with all parameters frozen.
    """
    print(f"[encode] loading checkpoint {ckpt_path}", flush=True)
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = blob["args"]
    latent_dim = int(args["d"])
    proj_norm = args.get("projection_norm", "batchnorm")
    enc = HybridCNNViTEncoder(latent_dim=latent_dim, projection_norm=proj_norm)
    state = {
        k.removeprefix("encoder."): v
        for k, v in blob["jepa_state_dict"].items()
        if k.startswith("encoder.")
    }
    missing, unexpected = enc.load_state_dict(state, strict=False)
    if missing:
        print(f"[encode] WARNING: missing encoder keys: {missing}", flush=True)
    if unexpected:
        print(f"[encode] WARNING: unexpected encoder keys: {unexpected}", flush=True)
    enc.eval().to(device)
    for p in enc.parameters():
        p.requires_grad_(False)
    print(
        f"[encode] encoder ready: latent_dim={latent_dim} "
        f"projection_norm={proj_norm}",
        flush=True,
    )
    return enc, latent_dim


def gather_encounters(split: str, cache_root: Path) -> list[dict]:
    """Enumerate every encounter belonging to ``split``.

    train / test_a are drawn from cases with ``case['split'] == 'train'``
    using ``train_encounter_indices`` / ``test_a_encounter_indices``
    respectively. test_b / test_c use every encounter in
    ``range(n_encounters_full)`` for cases with the matching
    ``case['split']``.

    Args:
        split: One of ``train``, ``test_a``, ``test_b``, ``test_c``.
        cache_root: Partition-v1 cache root.

    Returns:
        List of dicts ``{case_id, k, G, D, Y, path}``.
    """
    with open(SPLIT_MANIFEST_PATH) as f:
        manifest = json.load(f)
    out: list[dict] = []
    for cid, case in manifest["cases"].items():
        if split == "train" and case["split"] == "train":
            ks = list(case["train_encounter_indices"])
        elif split == "test_a" and case["split"] == "train":
            ks = list(case["test_a_encounter_indices"])
        elif split == "test_b" and case["split"] == "test_b":
            ks = list(range(int(case["n_encounters_full"])))
        elif split == "test_c" and case["split"] == "test_c":
            ks = list(range(int(case["n_encounters_full"])))
        else:
            continue
        for k in ks:
            path = cache_root / cid / f"encounter_{int(k):02d}.h5"
            out.append({
                "case_id": cid,
                "k": int(k),
                "G": float(case["G"]),
                "D": float(case["D"]),
                "Y": float(case["Y"]),
                "path": path,
            })
    return out


@torch.no_grad()
def encode_split(
    split: str,
    encs: list[dict],
    encoder: HybridCNNViTEncoder,
    pipeline: OmegaPipeline,
    device: torch.device,
    latent_dim: int,
) -> dict[str, np.ndarray]:
    """Encode every encounter in ``encs`` through ``encoder``.

    Each encounter contributes one row of arrays. ``z_full`` is computed in
    a single bf16 forward per encounter (T=120 frames batched along the
    time axis), then re-cast to fp32 for storage.

    Args:
        split: Split name (only used for log lines).
        encs: Per-encounter records from ``gather_encounters``.
        encoder: Loaded encoder on ``device``.
        pipeline: Frozen omega pipeline (mask + clip + scale).
        device: cuda device.
        latent_dim: Encoder latent dim (64 for E).

    Returns:
        Dict of column arrays keyed as in the module docstring.
        Skipped encounters (missing cache file) are dropped silently after
        a printed warning.
    """
    n_frames = 120
    z_imp_rows: list[np.ndarray] = []
    z_full_rows: list[np.ndarray] = []
    g_col: list[float] = []
    d_col: list[float] = []
    y_col: list[float] = []
    case_col: list[str] = []
    k_col: list[int] = []
    imp_col: list[int] = []
    skipped: list[str] = []

    n_total = len(encs)
    t_start = time.time()
    n_done = 0
    for rec in encs:
        path: Path = rec["path"]
        if not path.exists():
            msg = f"[encode] WARNING: missing cache file {path}; skipping"
            print(msg, flush=True)
            skipped.append(str(path))
            continue
        with h5py.File(path, "r") as f:
            omega_np = np.asarray(f["omega_z"], dtype=np.float32)
            impact_frame = int(f.attrs.get("impact_frame_estimate", DEFAULT_IMPACT_FRAME))
        if omega_np.shape[0] != n_frames:
            print(
                f"[encode] WARNING: unexpected n_frames={omega_np.shape[0]} "
                f"for {rec['case_id']} enc {rec['k']}; using actual length",
                flush=True,
            )
        # Stages 1 + 2 (numpy) then Stage 3 (torch) -- matches the
        # EpisodeDataset.__getitem__ pattern.
        omega_np = pipeline.preprocess_raw(omega_np, rec["case_id"], int(rec["k"]))
        omega_t = torch.from_numpy(omega_np)
        omega_t = pipeline.normalize(omega_t)
        # (T, H, W) -> (B=1, T, C=1, H, W)
        omega_t = omega_t.unsqueeze(0).unsqueeze(2).to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            z_t = encoder(omega_t)
        z_arr = z_t.squeeze(0).float().cpu().numpy()  # (T, d)
        # Clamp impact frame into [0, T-1] in case the encounter is shorter.
        imp = min(impact_frame, z_arr.shape[0] - 1)
        z_imp_rows.append(z_arr[imp])
        z_full_rows.append(z_arr)
        g_col.append(rec["G"])
        d_col.append(rec["D"])
        y_col.append(rec["Y"])
        case_col.append(rec["case_id"])
        k_col.append(int(rec["k"]))
        imp_col.append(int(imp))
        n_done += 1
        if n_done % 10 == 0 or n_done == n_total:
            elapsed = time.time() - t_start
            rate = n_done / max(elapsed, 1e-6)
            print(
                f"[encode] {split}: {n_done}/{n_total} encounters "
                f"({rate:.2f} enc/s, elapsed {elapsed:.1f}s)",
                flush=True,
            )

    if not z_imp_rows:
        raise RuntimeError(f"No encounters encoded for split={split}")

    # Pad / truncate to a uniform T axis. Cache files are always 120 frames
    # (see configs/preprocessing.yaml::encounter.frames_per_encounter) so
    # this is a defensive check; everything stacks cleanly in practice.
    T_uniform = max(arr.shape[0] for arr in z_full_rows)
    z_full = np.zeros((len(z_full_rows), T_uniform, latent_dim), dtype=np.float32)
    for i, arr in enumerate(z_full_rows):
        z_full[i, : arr.shape[0], :] = arr

    out = {
        "z": np.stack(z_imp_rows, axis=0).astype(np.float32),
        "z_full": z_full,
        "G": np.asarray(g_col, dtype=np.float32),
        "D": np.asarray(d_col, dtype=np.float32),
        "Y": np.asarray(y_col, dtype=np.float32),
        "case_id": np.asarray(case_col, dtype=object),
        "encounter_index": np.asarray(k_col, dtype=np.int32),
        "split": np.asarray([split] * len(case_col), dtype=object),
        "impact_frame": np.asarray(imp_col, dtype=np.int32),
    }
    if skipped:
        out["_skipped"] = np.asarray(skipped, dtype=object)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "One-shot encoding of every encounter through the Session 12 "
            "E d=64 JEPA encoder."
        )
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_PATH,
        help="Path to the JEPA encoder checkpoint.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for the per-split .npz files.",
    )
    p.add_argument(
        "--omega-pipeline-manifest",
        type=Path,
        default=OMEGA_PIPELINE_MANIFEST,
        help="Path to the omega preprocessing pipeline manifest.",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=list(SPLITS),
        choices=list(SPLITS),
        help="Splits to encode (default: all four).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device()
    cache_root = resolve_cache_root()
    print(f"[encode] cache_root={cache_root}", flush=True)
    if not cache_root.exists():
        raise FileNotFoundError(
            f"Partition v1 cache root not found: {cache_root}. "
            "Set VORTEX_JEPA_CACHE or PREVENT_ROOT correctly."
        )

    encoder, latent_dim = load_encoder(args.checkpoint, device)
    pipeline = OmegaPipeline.from_manifest(args.omega_pipeline_manifest)
    print(
        f"[encode] omega pipeline: mask cells={int(pipeline.mask.sum().item())} "
        f"thresholds={sum(len(v) for v in pipeline.thresholds.values())} encs "
        f"train_stats(mean={pipeline.train_stats.mean:.4f}, "
        f"std={pipeline.train_stats.std:.4f})",
        flush=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    t_global = time.time()
    summary: list[tuple[str, int, int, float, bool]] = []  # (split, n, bytes, ok_finite)
    for split in args.splits:
        encs = gather_encounters(split, cache_root)
        print(f"\n[encode] split={split}: {len(encs)} encounters planned", flush=True)
        cols = encode_split(split, encs, encoder, pipeline, device, latent_dim)
        # Sanity: finite latents
        z_finite = bool(np.isfinite(cols["z"]).all() and np.isfinite(cols["z_full"]).all())
        if not z_finite:
            print(
                f"[encode] WARNING: split={split} contains non-finite values "
                f"in z/z_full",
                flush=True,
            )
        out_path = args.output_dir / f"{split}.npz"
        np.savez_compressed(out_path, **cols)
        size_bytes = out_path.stat().st_size
        n_rows = int(cols["z"].shape[0])
        summary.append((split, n_rows, size_bytes, z_finite))
        print(
            f"[encode] wrote {out_path} "
            f"(rows={n_rows}, z.shape={cols['z'].shape}, "
            f"z_full.shape={cols['z_full'].shape}, "
            f"size={size_bytes / 1e6:.1f} MB, all finite={z_finite})",
            flush=True,
        )
        # Friendly head + uniqueness for a quick sanity check
        head_str = np.array2string(
            cols["z"][:1, :8], precision=4, suppress_small=True
        )
        unique_cases = sorted(set(cols["case_id"].tolist()))
        print(
            f"[encode] {split} z[0, :8]={head_str}  unique_cases={len(unique_cases)}",
            flush=True,
        )

    elapsed_total = time.time() - t_global
    print("\n[encode] ============ SUMMARY ============")
    print(f"[encode] total wall time: {elapsed_total:.1f}s")
    for split, n, size_bytes, ok in summary:
        flag = "OK" if ok else "NON-FINITE"
        print(
            f"[encode]   {split:<8s}  rows={n:<5d}  "
            f"size={size_bytes / 1e6:6.1f} MB  finite={flag}",
            flush=True,
        )


if __name__ == "__main__":
    main()
