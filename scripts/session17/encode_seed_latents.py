"""Session 17 helper: encode per-seed latents for the 3 Thrust-6 seed retrains.

For each seed in {production, seed0, seed1, seed2}, encode:
  - train (180 encounters): impact-frame z only
  - test_b (56 v1p5 encounters): FULL z_full (120 frames each)
  - test_c (24 encounters): impact-frame z only

Production seed is already cached at outputs/session14/latents/S12_E_d64/*.npz;
we only encode it here for test_b (we need full 120-frame trajectories per
encounter to compute the per-frame distance matrix for Exp 1(d)) and re-use
the production cache for impact-frame latents.

Outputs:
    outputs/session17/seed_latents/{seed}/{split}.npz

Reuses the production omega pipeline manifest at
outputs/data_pipeline/v1/manifest.json. Each split's npz holds:
  z_full           (n_enc, 120, 64) or (n_enc, 64) for impact-only splits
  z                (n_enc, 64)      impact-frame z (slice at t=40)
  G, D, Y          (n_enc,)
  case_id          (n_enc,) dtype=object
  encounter_index  (n_enc,)
  impact_frame     (n_enc,)

This script encodes 4 seeds * (180 + 56 + 24) = 1040 encounter forward passes
on the RTX 6000. Each pass is one HDF5 read + one 120-frame forward; with
bf16 autocast on Blackwell this is well under 30 minutes total wall time.
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


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.data.omega_pipeline import OmegaPipeline  # noqa: E402
from src.models.encoder import HybridCNNViTEncoder  # noqa: E402
from src.utils.device import require_rtx6000  # noqa: E402


OMEGA_MANIFEST = REPO / "outputs" / "data_pipeline" / "v1" / "manifest.json"
SPLIT_MANIFEST = REPO / "configs" / "splits" / "split_v1p5.json"
PARTITION = "v1"
DEFAULT_IMPACT_FRAME = 40
OUT_ROOT = REPO / "outputs" / "session17" / "seed_latents"

SEED_ENCODERS = {
    "production": REPO / "outputs" / "runs" / "session12" / "S12_E_d64" / "encoder" / "checkpoint_iter020000.pt",
    "seed0": REPO / "outputs" / "runs" / "session14" / "thrust6" / "jepa_d64_seed0" / "encoder" / "checkpoint_iter020000.pt",
    "seed1": REPO / "outputs" / "runs" / "session14" / "thrust6" / "jepa_d64_seed1" / "encoder" / "checkpoint_iter020000.pt",
    "seed2": REPO / "outputs" / "runs" / "session14" / "thrust6" / "jepa_d64_seed2" / "encoder" / "checkpoint_iter020000.pt",
}


def resolve_cache_root() -> Path:
    env_cache = os.environ.get("VORTEX_JEPA_CACHE")
    if env_cache:
        return Path(env_cache) / PARTITION
    prevent_root = Path(os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT")))
    return prevent_root / "data" / "processed" / "vortex-jepa" / PARTITION


def gather_encounters(split: str, cache_root: Path) -> list[dict]:
    with open(SPLIT_MANIFEST) as f:
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


def load_encoder(ckpt_path: Path, device: torch.device) -> tuple[HybridCNNViTEncoder, int]:
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
    enc.load_state_dict(state, strict=False)
    enc.eval().to(device)
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc, latent_dim


@torch.no_grad()
def encode_encounter(
    rec: dict,
    encoder: HybridCNNViTEncoder,
    pipeline: OmegaPipeline,
    device: torch.device,
) -> tuple[np.ndarray, int]:
    """Encode one encounter; return (z_full (T,d), impact_frame int)."""
    path: Path = rec["path"]
    with h5py.File(path, "r") as f:
        omega_np = np.asarray(f["omega_z"], dtype=np.float32)
        impact_frame = int(f.attrs.get("impact_frame_estimate", DEFAULT_IMPACT_FRAME))
    omega_np = pipeline.preprocess_raw(omega_np, rec["case_id"], int(rec["k"]))
    omega_t = torch.from_numpy(omega_np)
    omega_t = pipeline.normalize(omega_t)
    omega_t = omega_t.unsqueeze(0).unsqueeze(2).to(device, non_blocking=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        z_t = encoder(omega_t)
    return z_t.squeeze(0).float().cpu().numpy(), impact_frame


def encode_split(
    encs: list[dict],
    encoder: HybridCNNViTEncoder,
    pipeline: OmegaPipeline,
    device: torch.device,
    keep_full: bool,
) -> dict:
    z_full_rows: list[np.ndarray] = []
    z_imp_rows: list[np.ndarray] = []
    impact_list: list[int] = []
    g_col, d_col, y_col, case_col, k_col = [], [], [], [], []
    for rec in encs:
        if not rec["path"].exists():
            continue
        z_arr, impact_frame = encode_encounter(rec, encoder, pipeline, device)
        imp = min(impact_frame, z_arr.shape[0] - 1)
        if keep_full:
            z_full_rows.append(z_arr)
        z_imp_rows.append(z_arr[imp])
        impact_list.append(impact_frame)
        g_col.append(rec["G"])
        d_col.append(rec["D"])
        y_col.append(rec["Y"])
        case_col.append(rec["case_id"])
        k_col.append(int(rec["k"]))
    out = {
        "z": np.stack(z_imp_rows, axis=0).astype(np.float32),
        "G": np.asarray(g_col, dtype=np.float32),
        "D": np.asarray(d_col, dtype=np.float32),
        "Y": np.asarray(y_col, dtype=np.float32),
        "case_id": np.asarray(case_col, dtype=object),
        "encounter_index": np.asarray(k_col, dtype=np.int32),
        "impact_frame": np.asarray(impact_list, dtype=np.int32),
    }
    if keep_full:
        out["z_full"] = np.stack(z_full_rows, axis=0).astype(np.float32)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", default=list(SEED_ENCODERS))
    parser.add_argument(
        "--splits", nargs="+",
        default=["train", "test_b", "test_c"],
        help="splits to encode; test_b gets z_full, others only impact-frame z",
    )
    parser.add_argument("--full-splits", nargs="+", default=["test_b"])
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    device = require_rtx6000(gpu_index=args.gpu)
    print(f"[encode] device={device} ({torch.cuda.get_device_name(device.index)})")

    pipeline = OmegaPipeline.from_manifest(OMEGA_MANIFEST)
    cache_root = resolve_cache_root()

    encs_by_split: dict[str, list[dict]] = {
        s: gather_encounters(s, cache_root) for s in args.splits
    }
    for s, encs in encs_by_split.items():
        print(f"[encode] {s}: {len(encs)} encounters")

    for seed_name in args.seeds:
        ckpt_path = SEED_ENCODERS[seed_name]
        seed_out_dir = OUT_ROOT / seed_name
        seed_out_dir.mkdir(parents=True, exist_ok=True)
        if not ckpt_path.exists():
            print(f"[encode] WARNING: missing {ckpt_path}, skipping {seed_name}")
            continue
        t0 = time.time()
        enc, latent_dim = load_encoder(ckpt_path, device)
        print(
            f"[encode] {seed_name}: encoder loaded, d={latent_dim} "
            f"(in {time.time() - t0:.2f}s)"
        )
        for split, encs in encs_by_split.items():
            out_path = seed_out_dir / f"{split}.npz"
            if args.skip_existing and out_path.exists():
                print(f"[encode] {seed_name}/{split}: exists, skip")
                continue
            tsplit = time.time()
            keep_full = split in args.full_splits
            split_data = encode_split(encs, enc, pipeline, device, keep_full)
            np.savez_compressed(out_path, **split_data)
            print(
                f"[encode] {seed_name}/{split:8s}: wrote {out_path.name} "
                f"(z_full={'yes' if keep_full else 'no '}; "
                f"{time.time() - tsplit:.1f}s, n={split_data['z'].shape[0]})"
            )
        del enc
        torch.cuda.empty_cache()

    print(f"[encode] all done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
