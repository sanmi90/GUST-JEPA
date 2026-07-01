"""Fit the d=32 POD linear reference on the v2.2 train pool (Session 31 Track C).

POD is the linear floor: a fixed orthonormal basis fit on the pipeline-normalised
omega snapshots of the v2.2 train split. No neural training and no GPU maths are
required (a truncated randomised SVD of the snapshot matrix suffices), so this runs
CPU-only and writes ``outputs/runs/session31/pod/pod_basis.npz`` that
``src.evaluation.rom_eval.load_reference_model`` reads at eval time.

Usage:
    OMP_NUM_THREADS=8 taskset -c 0-7 python scripts/session31/fit_pod.py \\
        --partition v2p2 --d 32 \\
        --pipeline-manifest outputs/data_pipeline/v2p2/manifest.json \\
        --out outputs/runs/session31/pod
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.baselines.pod import PODBasis, save_pod_basis  # noqa: E402
from src.data.omega_pipeline import OmegaPipeline  # noqa: E402
from src.evaluation.rom_eval import enumerate_encounters  # noqa: E402


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO / p


def _cache_dir(partition: str) -> Path:
    import os

    env = os.environ.get("VORTEX_JEPA_CACHE")
    if env:
        root = Path(env)
    else:
        prevent = os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT"))
        root = Path(prevent) / "data" / "processed" / "vortex-jepa"
    return root / partition


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--partition", type=str, default="v2p2")
    p.add_argument("--split", type=str, default="configs/splits/split_v2p2.json")
    p.add_argument(
        "--pipeline-manifest", type=str, default="outputs/data_pipeline/v2p2/manifest.json"
    )
    p.add_argument(
        "--out", "--output-dir", dest="out", type=str, default="outputs/runs/session31/pod"
    )
    p.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Subsample train frames every N for SVD speed (1 = all).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    H, W = 192, 96
    out_dir = _resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "pod.log"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(log_path, "a") as fh:
            fh.write(msg + "\n")

    with open(_resolve(args.split)) as fh:
        manifest = json.load(fh)
    pipe = OmegaPipeline.from_manifest(_resolve(args.pipeline_manifest))
    cache_dir = _cache_dir(args.partition)
    log(
        f"[pod] partition={args.partition} d={args.d} stride={args.frame_stride} "
        f"train_std={pipe.train_stats.std:.4f}"
    )

    encounters = enumerate_encounters("train", manifest)
    log(f"[pod] train encounters: {len(encounters)}")

    t0 = time.time()
    snaps: list[np.ndarray] = []
    for i, (cid, k) in enumerate(encounters):
        path = cache_dir / cid / f"encounter_{int(k):02d}.h5"
        with h5py.File(path, "r") as g:
            omega_raw = np.asarray(g["omega_z"], dtype=np.float32)
        omega_clean = pipe.preprocess_raw(omega_raw, cid, int(k))
        omega_norm = np.asarray(pipe.normalize(omega_clean), dtype=np.float32)
        flat = omega_norm.reshape(omega_norm.shape[0], H * W)
        if args.frame_stride > 1:
            flat = flat[:: args.frame_stride]
        snaps.append(flat)
        if (i + 1) % 40 == 0 or i == len(encounters) - 1:
            log(f"[pod] loaded {i + 1}/{len(encounters)} encounters ({time.time() - t0:.1f}s)")
    X = np.concatenate(snaps, axis=0).astype(np.float32)
    log(f"[pod] snapshot matrix {X.shape}; load {time.time() - t0:.1f}s")

    mean = X.mean(axis=0)
    Xc = torch.from_numpy(X - mean[None, :])
    total_energy = float((Xc**2).sum().item())

    t0 = time.time()
    q = min(args.d + 10, min(Xc.shape) - 1)
    _, s, v = torch.svd_lowrank(Xc, q=q, niter=6)  # Xc ~= U diag(s) V^T; V: (P, q)
    components = v[:, : args.d].contiguous().numpy().astype(np.float32)  # (P, d)
    singular_values = s[: args.d].numpy().astype(np.float64)
    energy_fraction = float((singular_values**2).sum() / total_energy) if total_energy > 0 else 0.0
    log(
        f"[pod] low-rank SVD in {time.time() - t0:.1f}s; energy@d={energy_fraction:.4f}; "
        f"s[:3]={singular_values[:3].tolist()}"
    )

    basis = PODBasis(
        mean=mean.astype(np.float32),
        components=components,
        singular_values=singular_values,
        energy_fraction=energy_fraction,
        height=H,
        width=W,
    )
    save_pod_basis(out_dir / "pod_basis.npz", basis)
    log(f"[pod] wrote {out_dir / 'pod_basis.npz'} (d={basis.d})")

    summary = {
        "d": args.d,
        "partition": args.partition,
        "n_snapshots": int(X.shape[0]),
        "energy_fraction": energy_fraction,
        "singular_values_head": singular_values[:5].tolist(),
        "pipeline_manifest": str(args.pipeline_manifest),
        "reference": True,
        "encoder": "pod_linear",
    }
    (out_dir / "pod_summary.json").write_text(json.dumps(summary, indent=2))
    log(f"[pod] wrote {out_dir / 'pod_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
