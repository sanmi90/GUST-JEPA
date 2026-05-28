"""POD (Proper Orthogonal Decomposition) baseline at d=32 for Session 11.

The natural linear baseline for the matched-d ablation table:

- Build the snapshot matrix from all train encounters' pipeline-normalized
  omega frames.
- Take the truncated SVD at rank d=32 to get a fixed orthonormal basis.
- Reconstruct each test_a / test_b / test_c frame by projecting onto the
  d=32 modes and decoding ``mean + Phi @ coeffs``.
- Evaluate with the same Session 10 metric bundle (SSIM, eps_vol, mse_wake,
  enstrophy_rel_err_wake, radial_spectrum_l2_wake) used for the JEPA and
  Fukami runs.

This is the classic *linear* floor for matched-d=32: any nonlinear method
should beat POD on reconstruction quality, but the gap is the meaningful
quantity for the paper.

Usage::

    python scripts/session11_pod_baseline.py \\
        --omega-pipeline-manifest outputs/data_pipeline/v1/manifest.json \\
        --d 32 \\
        --output-dir outputs/runs/session11/POD_d32
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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.omega_pipeline import OmegaPipeline  # noqa: E402
from src.evaluation.decoder_metrics import (  # noqa: E402
    aggregate_split_metrics,
    compute_encounter_metrics,
)


PREVENT = Path(os.environ.get("PREVENT_ROOT", "/home/carlos/PREVENT"))
CACHE = Path(os.environ.get("VORTEX_JEPA_CACHE", PREVENT / "data" / "processed" / "vortex-jepa"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="POD baseline for Session 11")
    p.add_argument("--d", type=int, default=32, help="Number of POD modes.")
    p.add_argument("--partition", type=str, default="v1",
                   help="Cache partition; v1 cache stays valid for the v2 rerun.")
    p.add_argument("--split", type=str,
                   default="configs/splits/split_v2.json",
                   help="Path to split manifest. Default split_v2.json (v2 rerun).")
    p.add_argument(
        "--omega-pipeline-manifest", type=str,
        default="outputs/data_pipeline/v1/manifest.json",
    )
    p.add_argument("--output-dir", required=True, type=str)
    p.add_argument(
        "--frame-stride", type=int, default=1,
        help="Subsample train frames every N for SVD speed. 1 = use all.",
    )
    return p.parse_args()


def gather_encounters(partition: str, split: str,
                     split_manifest_path: "str | Path | None" = None) -> list[dict]:
    if split_manifest_path is None:
        manifest_path = REPO / "configs" / "splits" / "split_v2.json"
    else:
        manifest_path = Path(split_manifest_path)
        if not manifest_path.is_absolute():
            manifest_path = REPO / manifest_path
    with open(manifest_path) as f:
        m = json.load(f)
    out = []
    for cid, c in m["cases"].items():
        if split == "train" and c["split"] == "train":
            ks = c["train_encounter_indices"]
        elif split == "test_a" and c["split"] == "train":
            ks = (c.get("val_encounter_indices") or c["test_a_encounter_indices"])
        elif split in ("test_b", "test_c") and c["split"] == split:
            ks = list(range(c["n_encounters_full"]))
        else:
            continue
        for k in ks:
            path = CACHE / partition / cid / f"encounter_{k:02d}.h5"
            if path.exists():
                out.append({"case_id": cid, "k": int(k), "path": str(path)})
    return out


def load_normalized_omega(path: str, case_id: str, k: int,
                          pipe: OmegaPipeline) -> np.ndarray:
    """Load omega, preprocess (mask + clip + normalize). Returns (T, H, W) float32."""
    with h5py.File(path, "r") as f:
        omega_raw = np.asarray(f["omega_z"], dtype=np.float32)
    omega_clean = pipe.preprocess_raw(omega_raw, case_id, int(k))
    omega_t = torch.from_numpy(omega_clean)
    omega_norm = pipe.normalize(omega_t).numpy()
    return omega_norm


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "pod.log"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    log(f"[pod] d={args.d}, partition={args.partition}, stride={args.frame_stride}")

    manifest_path = Path(args.omega_pipeline_manifest)
    if not manifest_path.is_absolute():
        manifest_path = REPO / manifest_path
    pipe = OmegaPipeline.from_manifest(manifest_path)
    log(f"[pod] pipeline loaded from {manifest_path}")

    H, W = 192, 96
    train_encs = gather_encounters(args.partition, "train", split_manifest_path=args.split)
    log(f"[pod] train encounters: {len(train_encs)}")

    # Build snapshot matrix: (n_frames, H*W) of pipeline-normalized omega.
    t0 = time.time()
    snapshots = []
    for i, e in enumerate(train_encs):
        omega_norm = load_normalized_omega(e["path"], e["case_id"], int(e["k"]), pipe)
        T = omega_norm.shape[0]
        flat = omega_norm.reshape(T, H * W)
        if args.frame_stride > 1:
            flat = flat[::args.frame_stride]
        snapshots.append(flat)
        if (i + 1) % 30 == 0:
            log(f"[pod] loaded {i+1}/{len(train_encs)} train encounters "
                f"({time.time() - t0:.1f}s)")
    X = np.concatenate(snapshots, axis=0).astype(np.float32)
    log(f"[pod] snapshot matrix shape: {X.shape}; total load {time.time() - t0:.1f}s")

    # Center
    mean = X.mean(axis=0)
    Xc = X - mean[None]

    # Truncated SVD via torch (uses LAPACK; fast on CPU for our size).
    t0 = time.time()
    Xc_t = torch.from_numpy(Xc)
    log(f"[pod] computing truncated SVD on {Xc.shape} matrix...")
    U, S, Vh = torch.svd_lowrank(Xc_t, q=args.d + 10, niter=4)
    # svd_lowrank returns rank=q approximation; take top d.
    U = U[:, :args.d]
    S = S[:args.d]
    Vh_full = Vh[:, :args.d].T  # (d, H*W)
    log(f"[pod] SVD done in {time.time() - t0:.1f}s; "
        f"top {args.d} singular values: {S[:5].tolist()} ... {S[-3:].tolist()}")

    energy = (S ** 2).sum() / (Xc_t ** 2).sum()
    log(f"[pod] cumulative energy fraction at d={args.d}: {energy:.4f}")

    # POD basis Phi: (H*W, d). Reconstruct as mean + Phi @ coeffs.
    Phi = Vh_full.T.numpy()  # (H*W, d)
    mean_2d = mean.reshape(H, W)

    np.savez(out_dir / "pod_basis.npz", Phi=Phi, mean=mean, S=S.numpy(),
             d=args.d, energy_fraction=float(energy))
    log(f"[pod] saved basis to {out_dir / 'pod_basis.npz'}")

    # Evaluate on each split with the Session 10 metric bundle.
    summary: dict = {"d": args.d, "energy_fraction": float(energy)}
    for split in ("test_a", "test_b", "test_c"):
        encs = gather_encounters(args.partition, split, split_manifest_path=args.split)
        per_enc = []
        for e in encs:
            with h5py.File(e["path"], "r") as f:
                omega_raw = np.asarray(f["omega_z"], dtype=np.float32)
            omega_clean = pipe.preprocess_raw(omega_raw, e["case_id"], int(e["k"]))
            T = omega_clean.shape[0]
            # Project + reconstruct in normalized space, then unnormalize for
            # raw-scale metric computation.
            omega_norm = pipe.normalize(torch.from_numpy(omega_clean)).numpy()
            flat = omega_norm.reshape(T, H * W)
            coeffs = (flat - mean[None]) @ Phi  # (T, d)
            recon_flat_norm = mean[None] + coeffs @ Phi.T  # (T, H*W)
            recon_norm = recon_flat_norm.reshape(T, H, W)
            recon_raw = pipe.unnormalize(torch.from_numpy(recon_norm)).numpy()
            per_enc.append(compute_encounter_metrics(omega_clean, recon_raw))
        agg = aggregate_split_metrics(per_enc)
        summary[split] = agg
        log(f"[pod] {split} ({len(encs)} encs):")
        for k in ("ssim_mean", "eps_volume", "mse_wake",
                  "enstrophy_rel_err_wake", "radial_spectrum_l2_wake"):
            mk, sk = f"{k}_median", f"{k}_mean"
            if mk in agg:
                log(f"   {k}: median={agg[mk]:.4f} mean={agg[sk]:.4f}")

    with open(out_dir / "pod_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"[pod] wrote summary to {out_dir / 'pod_summary.json'}")


if __name__ == "__main__":
    main()
