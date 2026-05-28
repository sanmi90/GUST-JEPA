"""Session 11 wake-probe summary for a JEPA encoder checkpoint.

Loads a JEPA checkpoint, encodes Test B encounters into latents ``z``,
and reports the Track 1/2 wake-probe summary needed to apply the
pre-decoder gate:

- ``r2(z -> G, D, Y)`` (parametric probe, per-axis + overall)
- ``r2(z -> CL_future)`` at the configured deltas (or CL_present if delta=[0])
- ``r2(z -> wake_patch_signed)`` (64 dim)
- ``r2(z -> wake_patch_signed_spectrum)`` (80 dim)
- ``r2(z -> wake_enstrophy_scalar)`` (1 dim)
- ``r2(z -> wake_coarse_pool)`` (288 dim)
- ``PR(z)`` (participation ratio)

Probes are fit on a random 75% of Test B frames and evaluated on the held-
out 25%. The split is seeded so the comparison across runs is paired.

Reference: SESSION11_WAKE_RESULTS_FIRST.md "Evaluation per run".

Usage::

    python scripts/session11_wake_probe.py \\
        --jepa-checkpoint outputs/runs/session11/W0_C_lam10/checkpoint_iter020000.pt \\
        --wake-observables-root ${VORTEX_JEPA_CACHE}/v1/wake_observables \\
        --output-dir outputs/runs/session11/W0_C_lam10/probe
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.omega_pipeline import OmegaPipeline  # noqa: E402
from src.models.encoder import HybridCNNViTEncoder  # noqa: E402
from src.training.diagnostics import (  # noqa: E402
    linear_probe_r2,
    participation_ratio,
)
from src.utils.device import require_rtx6000  # noqa: E402


PREVENT = Path(os.environ.get("PREVENT_ROOT", "/home/carlos/PREVENT"))
CACHE = Path(os.environ.get("VORTEX_JEPA_CACHE", PREVENT / "data" / "processed" / "vortex-jepa"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Session 11 wake-probe summary")
    p.add_argument("--jepa-checkpoint", required=True, type=str)
    p.add_argument("--output-dir", required=True, type=str)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument(
        "--wake-observables-root",
        type=str,
        default=None,
        help="Defaults to ${VORTEX_JEPA_CACHE}/v1/wake_observables.",
    )
    p.add_argument(
        "--omega-pipeline-manifest",
        type=str,
        default="outputs/data_pipeline/v1/manifest.json",
    )
    p.add_argument(
        "--cl-future-deltas",
        type=int,
        nargs="+",
        default=[0],
        help="CL future deltas. Default [0] (CL_present, Fukami-aligned).",
    )
    p.add_argument("--probe-seed", type=int, default=42)
    p.add_argument("--probe-fit-fraction", type=float, default=0.75)
    return p.parse_args()


def gather_test_b_encounters() -> list[dict]:
    with open(REPO / "configs" / "splits" / "split_v2.json") as f:
        manifest = json.load(f)
    out = []
    for cid, case in manifest["cases"].items():
        if case["split"] != "test_b":
            continue
        for k in range(case["n_encounters_full"]):
            path = CACHE / "v1" / cid / f"encounter_{k:02d}.h5"
            if not path.exists():
                continue
            out.append({
                "case_id": cid,
                "encounter_index": int(k),
                "G": float(case["G"]),
                "D": float(case["D"]),
                "Y": float(case["Y"]),
                "src": str(path),
            })
    return out


def load_jepa_encoder(ckpt_path: Path, device: torch.device) -> tuple[HybridCNNViTEncoder, int]:
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
    enc.load_state_dict(state, strict=False)
    enc.eval().to(device)
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc, int(args["d"])


def encode_all(
    encs: list[dict],
    enc: HybridCNNViTEncoder,
    omega_pipeline: OmegaPipeline,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Encode every frame of every Test B encounter; return ``(N, d)`` z plus
    per-frame metadata tensors (G, D, Y, case_id_idx, encounter_idx, frame_idx).
    """
    z_chunks: list[torch.Tensor] = []
    G_list: list[float] = []
    D_list: list[float] = []
    Y_list: list[float] = []
    case_ids: list[str] = []
    enc_idx_list: list[int] = []
    frame_idx_list: list[int] = []
    for e in encs:
        with h5py.File(e["src"], "r") as g:
            omega_raw = np.asarray(g["omega_z"], dtype=np.float32)
        omega_clean = omega_pipeline.preprocess_raw(omega_raw, e["case_id"], int(e["encounter_index"]))
        x = torch.from_numpy(omega_clean).unsqueeze(0).unsqueeze(2).to(device)
        x = omega_pipeline.normalize(x)
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            z = enc(x).float().squeeze(0)  # (T, d)
        T = z.shape[0]
        z_chunks.append(z.cpu())
        G_list += [e["G"]] * T
        D_list += [e["D"]] * T
        Y_list += [e["Y"]] * T
        case_ids += [e["case_id"]] * T
        enc_idx_list += [e["encounter_index"]] * T
        frame_idx_list += list(range(T))
    z_all = torch.cat(z_chunks, dim=0)
    meta = {
        "G": torch.tensor(G_list, dtype=torch.float32),
        "D": torch.tensor(D_list, dtype=torch.float32),
        "Y": torch.tensor(Y_list, dtype=torch.float32),
        "case_ids": case_ids,
        "encounter_idx": torch.tensor(enc_idx_list, dtype=torch.long),
        "frame_idx": torch.tensor(frame_idx_list, dtype=torch.long),
    }
    return z_all, meta


def load_wake_targets(
    encs: list[dict],
    wake_root: Path,
    mode: str,
) -> torch.Tensor:
    parts: list[np.ndarray] = []
    for e in encs:
        path = wake_root / e["case_id"] / f"encounter_{e['encounter_index']:02d}.h5"
        with h5py.File(path, "r") as g:
            parts.append(g[mode][...].astype(np.float32))
    return torch.from_numpy(np.concatenate(parts, axis=0))


def load_cl_at_frames(
    encs: list[dict],
    deltas: list[int],
) -> torch.Tensor:
    """Load ``CL(t + delta)`` for each (encounter, frame, delta). Clamps to last
    valid frame past the end (matching ``EpisodeDataset.cl_future_deltas``)."""
    cols: list[np.ndarray] = []
    for e in encs:
        with h5py.File(e["src"], "r") as g:
            cl = np.asarray(g["C_L"], dtype=np.float32)
        T = cl.shape[0]
        per_frame = np.empty((T, len(deltas)), dtype=np.float32)
        for j, d in enumerate(deltas):
            for i in range(T):
                src = i + d
                per_frame[i, j] = cl[src] if src < T else cl[T - 1]
        cols.append(per_frame)
    return torch.from_numpy(np.concatenate(cols, axis=0))


def main() -> None:
    args = parse_args()
    device = require_rtx6000(gpu_index=args.gpu)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "wake_probe.log"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    log(f"[wake-probe] device={device}")
    ckpt_path = Path(args.jepa_checkpoint).resolve()
    log(f"[wake-probe] checkpoint={ckpt_path}")

    enc, d = load_jepa_encoder(ckpt_path, device)
    log(f"[wake-probe] encoder loaded, d={d}")

    manifest_path = Path(args.omega_pipeline_manifest)
    if not manifest_path.is_absolute():
        manifest_path = REPO / manifest_path
    omega_pipeline = OmegaPipeline.from_manifest(manifest_path)

    wake_root = Path(args.wake_observables_root) if args.wake_observables_root else (
        CACHE / "v1" / "wake_observables"
    )
    if not wake_root.exists():
        raise SystemExit(f"wake observables root not found: {wake_root}")

    encs = gather_test_b_encounters()
    log(f"[wake-probe] test_b: {len(encs)} encounters")

    z_all, meta = encode_all(encs, enc, omega_pipeline, device)
    log(f"[wake-probe] z shape: {tuple(z_all.shape)}")
    N = z_all.shape[0]

    rng = np.random.default_rng(args.probe_seed)
    perm = torch.from_numpy(rng.permutation(N)).long()
    n_fit = int(args.probe_fit_fraction * N)
    fit_idx = perm[:n_fit]
    eval_idx = perm[n_fit:]

    summary: dict[str, dict | float] = {}

    c_gdy = torch.stack([meta["G"], meta["D"], meta["Y"]], dim=1)
    summary["r2_GDY"] = linear_probe_r2(z_all, c_gdy, fit_idx, eval_idx)
    log(f"[probe] r2_GDY: {summary['r2_GDY']}")

    cl_targets = load_cl_at_frames(encs, args.cl_future_deltas)
    summary["r2_cl"] = linear_probe_r2(z_all, cl_targets, fit_idx, eval_idx)
    log(f"[probe] r2_cl (deltas={args.cl_future_deltas}): {summary['r2_cl']}")

    for mode in ("enstrophy_scalar", "patch_signed", "patch_signed_spectrum",
                 "wake_coarse_pool"):
        try:
            targets = load_wake_targets(encs, wake_root, mode)
        except (KeyError, FileNotFoundError) as e:
            log(f"[probe] skipping {mode}: {e}")
            continue
        # NB: linear_probe_r2 emits r2_0, r2_1, ... when c_dim != 3. We only
        # keep r2_overall for these high-dim targets.
        probe = linear_probe_r2(z_all, targets, fit_idx, eval_idx)
        summary[f"r2_{mode}"] = {
            "r2_overall": probe["r2_overall"],
            "out_dim": targets.shape[1],
        }
        log(f"[probe] r2_{mode}: overall={probe['r2_overall']:.4f} "
            f"out_dim={targets.shape[1]}")

    pr = participation_ratio(z_all)
    summary["pr"] = float(pr)
    log(f"[probe] PR={pr:.3f}")

    summary["jepa_checkpoint"] = str(ckpt_path)
    summary["test_b_n_encounters"] = len(encs)
    summary["test_b_n_frames"] = N
    summary["d"] = d
    summary["probe_seed"] = args.probe_seed
    summary["probe_fit_fraction"] = args.probe_fit_fraction
    summary["cl_future_deltas"] = args.cl_future_deltas

    with open(out_dir / "wake_probe.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"[wake-probe] wrote {out_dir / 'wake_probe.json'}")


if __name__ == "__main__":
    main()
