"""POD basis + field preload for the SIGReg-JEPA-ROM no-lift arm (Session 34).

Builds the r-mode POD anchor from the TRAIN pool of pipeline-normalized
omega_z (mean-subtracted snapshot POD via randomized SVD) and caches:

    outputs/session34/rom_pod_basis.npz
        mean_field (192, 96), phi (18432, r), lam (r,), energy_fraction (r,)

The loader helpers here are shared with scripts/session34/run_rom_nolift.py:
fields are pipeline-normalized exactly like every canonical run (OmegaPipeline
preprocess_raw + normalize) and centering by the train mean is done DATA-SIDE,
so the uploaded model file needs no modification (its decoder reconstructs the
fluctuation field).

Run (CPU): taskset -c 0-15 python -m scripts.session34.rom_pod_basis
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

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.omega_pipeline import OmegaPipeline  # noqa: E402

PREVENT = Path(os.environ.get("PREVENT_ROOT", "/home/carlos/PREVENT"))
CACHE = Path(os.environ.get("VORTEX_JEPA_CACHE", PREVENT / "data" / "processed" / "vortex-jepa"))
H, W = 192, 96


def enumerate_encounters(split_manifest: dict, split: str) -> list[tuple[str, int]]:
    out = []
    for cid, case in split_manifest["cases"].items():
        if split == "train" and case["split"] == "train":
            out += [(cid, int(k)) for k in case["train_encounter_indices"]]
        elif split == "test_b" and case["split"] == "test_b":
            out += [(cid, k) for k in range(int(case["n_encounters_full"]))]
    return sorted(out)


def load_split_fields(
    split: str,
    partition: str = "v2p2",
    split_path: Path | None = None,
    pipeline_manifest: Path | None = None,
) -> dict:
    """Load a split's pipeline-normalized fields + alignment + C_L into RAM.

    Returns dict with fields (M, H, W) float32, case_id (M,) list[str],
    encounter_index (M,), frame (M,), cl (M,), enc_offsets
    {(cid, k): (row0, n_frames)}.
    """
    split_path = split_path or REPO_ROOT / "configs/splits/split_v2p2.json"
    pipeline_manifest = pipeline_manifest or REPO_ROOT / "outputs/data_pipeline/v2p2/manifest.json"
    manifest = json.loads(Path(split_path).read_text())
    pipe = OmegaPipeline.from_manifest(pipeline_manifest)
    encs = enumerate_encounters(manifest, split)

    fields, cids, eidx, frames, cls_ = [], [], [], [], []
    offsets: dict[tuple, tuple[int, int]] = {}
    row = 0
    t0 = time.time()
    for i, (cid, k) in enumerate(encs):
        path = CACHE / partition / cid / f"encounter_{k:02d}.h5"
        with h5py.File(path, "r") as g:
            omega_raw = np.asarray(g["omega_z"], dtype=np.float32)
            cl = np.asarray(g["C_L"], dtype=np.float32)
        omega = pipe.normalize(
            torch.from_numpy(pipe.preprocess_raw(omega_raw, cid, k))
        ).numpy().astype(np.float32)
        T = omega.shape[0]
        fields.append(omega)
        cids += [cid] * T
        eidx += [k] * T
        frames += list(range(T))
        cls_.append(cl)
        offsets[(cid, k)] = (row, T)
        row += T
        if (i + 1) % 50 == 0 or i == len(encs) - 1:
            print(f"[rom-loader] {split}: {i + 1}/{len(encs)} encounters "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return {
        "fields": np.concatenate(fields, axis=0),
        "case_id": np.array(cids),
        "encounter_index": np.array(eidx, dtype=np.int64),
        "frame": np.array(frames, dtype=np.int64),
        "cl": np.concatenate(cls_, axis=0),
        "enc_offsets": offsets,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ROM POD basis")
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--out", default="outputs/session34/rom_pod_basis.npz")
    args = ap.parse_args(argv)

    tr = load_split_fields("train")
    X = tr["fields"].reshape(tr["fields"].shape[0], -1)  # (M, N)
    mean_field = X.mean(axis=0)
    print(f"[rom-pod] snapshot matrix {X.shape}, centering", flush=True)
    Xc = X - mean_field

    from sklearn.utils.extmath import randomized_svd

    t0 = time.time()
    U, S, Vt = randomized_svd(Xc, n_components=args.r, n_iter=5, random_state=0)
    lam = (S ** 2) / max(Xc.shape[0] - 1, 1)          # modal variance
    total_var = float((Xc ** 2).sum() / max(Xc.shape[0] - 1, 1))
    energy = lam / total_var
    print(f"[rom-pod] SVD in {time.time() - t0:.1f}s; "
          f"r={args.r} captures {energy.sum():.3f} of variance "
          f"(mode1 {energy[0]:.3f})", flush=True)

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        mean_field=mean_field.reshape(H, W).astype(np.float32),
        phi=Vt.T.astype(np.float32),                   # (N, r)
        lam=lam.astype(np.float32),
        energy_fraction=energy.astype(np.float32),
        n_train_frames=np.int64(X.shape[0]),
    )
    print(f"[rom-pod] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
