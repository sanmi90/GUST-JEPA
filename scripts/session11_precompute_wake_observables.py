"""Precompute per-encounter wake observable targets for Session 11.

Reads pipeline-normalized omega for every encounter in the v1 cache and writes
per-encounter HDF5 files holding the four target modes:

- ``enstrophy_scalar``     (T, 1)
- ``patch_signed``        (T, 64)
- ``patch_signed_spectrum`` (T, 80)
- ``wake_coarse_pool``    (T, 288)

Output layout::

    ${VORTEX_JEPA_CACHE}/v1/wake_observables/{case_id}/encounter_{k:02d}.h5
    ${VORTEX_JEPA_CACHE}/v1/wake_observables/_train_stats.json
    ${VORTEX_JEPA_CACHE}/v1/wake_observables/_manifest.json

The standardization statistics in ``_train_stats.json`` are computed over the
train split only (``split == 'train'`` AND encounter in ``train_encounter_indices``),
so the test_a held-out encounters and test_b/test_c never enter the train mean
or std.

Targets are computed in pipeline-normalized omega space (CLAUDE.md "Omega
preprocessing pipeline" / D71). They are saved un-standardized; the training
loader applies standardization on the fly using ``_train_stats.json``.

Usage::

    python scripts/session11_precompute_wake_observables.py \\
        --omega-pipeline-manifest outputs/data_pipeline/v1/manifest.json \\
        --partition v1
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
from src.data.wake_observables import (  # noqa: E402
    compute_standardization_from_targets,
    compute_wake_observable,
    mode_output_dim,
)


PREVENT = Path(os.environ.get("PREVENT_ROOT", "/home/carlos/PREVENT"))
CACHE = Path(os.environ.get("VORTEX_JEPA_CACHE", PREVENT / "data" / "processed" / "vortex-jepa"))

_MODES = ("enstrophy_scalar", "patch_signed", "patch_signed_spectrum", "wake_coarse_pool")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute wake observables for Session 11")
    p.add_argument("--partition", default="v1")
    p.add_argument(
        "--omega-pipeline-manifest",
        default="outputs/data_pipeline/v1/manifest.json",
    )
    p.add_argument(
        "--modes",
        nargs="+",
        default=list(_MODES),
        choices=list(_MODES),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing per-encounter files.",
    )
    p.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help="Optional subset of case_ids.",
    )
    return p.parse_args()


def gather_encounters(partition: str, cases_filter: list[str] | None) -> list[dict]:
    manifest_path = REPO / "configs" / "splits" / f"split_{partition}.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    out = []
    for cid, case in manifest["cases"].items():
        if cases_filter is not None and cid not in cases_filter:
            continue
        for k in range(int(case["n_encounters_full"])):
            in_train = (
                case["split"] == "train" and k in case["train_encounter_indices"]
            )
            path = CACHE / partition / cid / f"encounter_{k:02d}.h5"
            if not path.exists():
                continue
            out.append({
                "case_id": cid,
                "encounter_index": int(k),
                "split": case["split"],
                "in_train_pool": bool(in_train),
                "src": str(path),
            })
    return out


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.omega_pipeline_manifest)
    if not manifest_path.is_absolute():
        manifest_path = REPO / manifest_path
    omega_pipeline = OmegaPipeline.from_manifest(manifest_path)
    print(f"[wake-precompute] loaded omega pipeline from {manifest_path}", flush=True)
    print(f"[wake-precompute] modes={args.modes}", flush=True)

    encs = gather_encounters(args.partition, args.cases)
    print(f"[wake-precompute] {len(encs)} encounters found in partition {args.partition}", flush=True)

    out_root = CACHE / args.partition / "wake_observables"
    out_root.mkdir(parents=True, exist_ok=True)

    # Per-mode: accumulate train targets to compute stats after the encounter loop.
    train_targets: dict[str, list[np.ndarray]] = {m: [] for m in args.modes}

    t0 = time.time()
    for i, e in enumerate(encs):
        out_dir = out_root / e["case_id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"encounter_{e['encounter_index']:02d}.h5"
        with h5py.File(e["src"], "r") as g:
            omega_raw = np.asarray(g["omega_z"], dtype=np.float32)
        omega_clean = omega_pipeline.preprocess_raw(omega_raw, e["case_id"], int(e["encounter_index"]))
        omega_norm = omega_pipeline.normalize(
            torch.from_numpy(omega_clean)
        ).numpy()  # (T, H, W) normalized
        per_mode_payload = {}
        for m in args.modes:
            tgt = compute_wake_observable(torch.from_numpy(omega_norm), m).numpy()
            per_mode_payload[m] = tgt
            if e["in_train_pool"]:
                train_targets[m].append(tgt.astype(np.float32))
        if out_path.exists() and not args.force:
            continue
        with h5py.File(out_path, "w") as out_g:
            out_g.attrs["case_id"] = e["case_id"]
            out_g.attrs["encounter_index"] = int(e["encounter_index"])
            out_g.attrs["split"] = e["split"]
            out_g.attrs["in_train_pool"] = bool(e["in_train_pool"])
            out_g.attrs["preprocessing_version"] = "wake_observables_v1"
            for m, arr in per_mode_payload.items():
                out_g.create_dataset(m, data=arr, dtype="float32")
        if (i + 1) % 20 == 0 or (i + 1) == len(encs):
            print(
                f"[wake-precompute] {i + 1}/{len(encs)} encounters in "
                f"{time.time() - t0:.1f}s",
                flush=True,
            )

    # Compute and write train-pool standardization stats per mode.
    stats_path = out_root / "_train_stats.json"
    stats_payload: dict[str, dict] = {}
    for m in args.modes:
        n_train = sum(t.shape[0] for t in train_targets[m])
        print(
            f"[wake-precompute] mode={m}: pooling {n_train} train frames "
            f"({len(train_targets[m])} encounters) for standardization",
            flush=True,
        )
        s = compute_standardization_from_targets(train_targets[m], mode=m)
        stats_payload[m] = s.to_dict()
        print(
            f"[wake-precompute] mode={m}: mean[:3]={s.mean[:3].tolist()}, "
            f"std[:3]={s.std[:3].tolist()}",
            flush=True,
        )
    with open(stats_path, "w") as f:
        json.dump(stats_payload, f, indent=2)
    print(f"[wake-precompute] wrote train stats to {stats_path}", flush=True)

    manifest_out = out_root / "_manifest.json"
    with open(manifest_out, "w") as f:
        json.dump(
            {
                "partition": args.partition,
                "modes": args.modes,
                "mode_dims": {m: mode_output_dim(m) for m in args.modes},
                "n_encounters_written": len(encs),
                "omega_pipeline_manifest": str(manifest_path),
                "train_stats_path": str(stats_path),
            },
            f,
            indent=2,
        )
    print(f"[wake-precompute] wrote manifest to {manifest_out}", flush=True)
    print(f"[wake-precompute] done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
