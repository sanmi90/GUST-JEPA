"""Run the wake-probe summary on a Fukami AE checkpoint (matched-d=32 baseline).

Mirrors ``scripts/session11_wake_probe.py`` but loads ``FukamiAEWrapper.encoder``
(``FukamiCNNEncoder``) instead of ``HybridCNNViTEncoder``. The same
linear-probe-on-Test-B framework reports ``r2(z -> G, D, Y)``,
``r2(z -> CL)``, and ``r2(z -> wake_*)`` for the four wake modes plus
``PR(z)``.

Useful for the paper's matched-d JEPA-vs-Fukami comparison: does the
Fukami AE's d=32 latent encode wake info that the JEPA latent does not?

Usage::

    python scripts/session11_fukami_wake_probe.py \\
        --fukami-checkpoint outputs/runs/session11/D4_fukami_ae_d32_matched/checkpoint_iter020000.pt \\
        --output-dir outputs/runs/session11/D4_fukami_ae_d32_matched/probe \\
        --gpu 1 --cl-future-deltas 0
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

from src.baselines.fukami_ae import FukamiAEWrapper  # noqa: E402
from src.data.omega_pipeline import OmegaPipeline  # noqa: E402
from src.training.diagnostics import (  # noqa: E402
    linear_probe_r2,
    participation_ratio,
)
from src.utils.device import require_rtx6000  # noqa: E402

# Re-use helpers from the JEPA wake-probe script.
sys.path.insert(0, str(REPO / "scripts"))
from session11_wake_probe import (  # noqa: E402
    gather_test_b_encounters,
    load_cl_at_frames,
    load_wake_targets,
)


PREVENT = Path(os.environ.get("PREVENT_ROOT", "/home/carlos/PREVENT"))
CACHE = Path(os.environ.get("VORTEX_JEPA_CACHE", PREVENT / "data" / "processed" / "vortex-jepa"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Session 11 Fukami-AE wake-probe summary")
    p.add_argument("--fukami-checkpoint", required=True, type=str)
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
    p.add_argument("--cl-future-deltas", type=int, nargs="+", default=[0])
    p.add_argument("--probe-seed", type=int, default=42)
    p.add_argument("--probe-fit-fraction", type=float, default=0.75)
    return p.parse_args()


def load_fukami_encoder(
    ckpt_path: Path,
    device: torch.device,
    omega_pipeline: OmegaPipeline,
) -> tuple[FukamiAEWrapper, int]:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = blob["args"]
    d = int(args.get("d", args.get("latent_dim", 32)))
    wrapper = FukamiAEWrapper(
        latent_dim=d,
        n_deltas=len(args.get("observable_head_deltas", [8, 16, 24])),
        activation=args.get("activation", "relu"),
        use_conv_norm=not args.get("no_conv_norm", False),
        omega_pipeline=omega_pipeline,
    )
    wrapper.load_state_dict(blob["wrapper_state_dict"], strict=False)
    wrapper.eval().to(device)
    for p in wrapper.parameters():
        p.requires_grad_(False)
    return wrapper, d


def encode_all(
    encs: list[dict],
    wrapper: FukamiAEWrapper,
    omega_pipeline: OmegaPipeline,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    z_chunks: list[torch.Tensor] = []
    G_list: list[float] = []
    D_list: list[float] = []
    Y_list: list[float] = []
    enc_idx_list: list[int] = []
    frame_idx_list: list[int] = []
    case_ids: list[str] = []
    for e in encs:
        with h5py.File(e["src"], "r") as g:
            omega_raw = np.asarray(g["omega_z"], dtype=np.float32)
        omega_clean = omega_pipeline.preprocess_raw(omega_raw, e["case_id"], int(e["encounter_index"]))
        x = torch.from_numpy(omega_clean).unsqueeze(0).unsqueeze(2).to(device)
        x = omega_pipeline.normalize(x)  # (1, T, 1, H, W) normalized
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            # FukamiCNNEncoder accepts (B, T, 1, H, W) or (B, 1, H, W) and returns
            # matching latent shape (B, T, d) or (B, d).
            z = wrapper.encoder(x).float().squeeze(0)
            if z.dim() == 1:
                z = z.unsqueeze(0)
        T = z.shape[0]
        z_chunks.append(z.cpu())
        G_list += [e["G"]] * T
        D_list += [e["D"]] * T
        Y_list += [e["Y"]] * T
        case_ids += [e["case_id"]] * T
        enc_idx_list += [e["encounter_index"]] * T
        frame_idx_list += list(range(T))
    z_all = torch.cat(z_chunks, dim=0)
    return z_all, {
        "G": torch.tensor(G_list, dtype=torch.float32),
        "D": torch.tensor(D_list, dtype=torch.float32),
        "Y": torch.tensor(Y_list, dtype=torch.float32),
        "case_ids": case_ids,
        "encounter_idx": torch.tensor(enc_idx_list, dtype=torch.long),
        "frame_idx": torch.tensor(frame_idx_list, dtype=torch.long),
    }


def main() -> None:
    args = parse_args()
    device = require_rtx6000(gpu_index=args.gpu)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "fukami_wake_probe.log"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    log(f"[fukami-wake-probe] device={device}")
    ckpt_path = Path(args.fukami_checkpoint).resolve()
    log(f"[fukami-wake-probe] checkpoint={ckpt_path}")

    manifest_path = Path(args.omega_pipeline_manifest)
    if not manifest_path.is_absolute():
        manifest_path = REPO / manifest_path
    omega_pipeline = OmegaPipeline.from_manifest(manifest_path)

    wrapper, d = load_fukami_encoder(ckpt_path, device, omega_pipeline)
    log(f"[fukami-wake-probe] Fukami AE loaded, d={d}")

    wake_root = Path(args.wake_observables_root) if args.wake_observables_root else (
        CACHE / "v1" / "wake_observables"
    )

    encs = gather_test_b_encounters()
    log(f"[fukami-wake-probe] test_b: {len(encs)} encounters")

    z_all, meta = encode_all(encs, wrapper, omega_pipeline, device)
    log(f"[fukami-wake-probe] z shape: {tuple(z_all.shape)}")
    N = z_all.shape[0]

    rng = np.random.default_rng(args.probe_seed)
    perm = torch.from_numpy(rng.permutation(N)).long()
    n_fit = int(args.probe_fit_fraction * N)
    fit_idx = perm[:n_fit]
    eval_idx = perm[n_fit:]

    summary: dict = {}

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

    summary["fukami_checkpoint"] = str(ckpt_path)
    summary["test_b_n_encounters"] = len(encs)
    summary["test_b_n_frames"] = N
    summary["d"] = d
    summary["probe_seed"] = args.probe_seed
    summary["probe_fit_fraction"] = args.probe_fit_fraction
    summary["cl_future_deltas"] = args.cl_future_deltas

    out_path = out_dir / "fukami_wake_probe.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"[fukami-wake-probe] wrote {out_path}")


if __name__ == "__main__":
    main()
