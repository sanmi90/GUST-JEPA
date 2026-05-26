"""Session 18 B1 Part (c) prep: precompute per-frame latents on each
baseline encoder for all four splits.

For each (baseline, d) pair, this script emits four .npz files at
``outputs/session18/exp_b1/latents_{baseline}_d{d}/{train,test_a,test_b,
test_c}.npz`` with keys:

    z_full           (n_enc, T=120, d)  per-frame latents
    G, D, Y          (n_enc,)           static episode descriptors
    case_ids         list[str] of length n_enc
    encounter_indices (n_enc,) int
    impact_frame     (n_enc,) int (HDF5 attr ``impact_frame_estimate``,
                                   typically 40)

The precomputed latents are the input to ``train_baseline_predictor.py``
(B1 Part c) and to ``eval_physical_closure.py`` (B1 Part d). Saving them
once avoids re-encoding 6 x 70 (train) + 6 x 28 (test_b) + 6 x 24
(test_c) = 732 encounter-encoder forward passes per training run.

Baselines:
    fukami   Fukami AE checkpoint at outputs/session18/exp_b1/fukami_ae_d{d}/
             checkpoint_iter020000.pt. Uses ``wrapper.encode(omega)``
             (applies omega_pipeline normalisation internally).
    pod      POD basis at outputs/session18/exp_b1/pod_d{d}/pod_basis.npz.
             Coefficients ``z = (omega_norm.flat - mean) @ Phi``.
    jepa     Production JEPA encoder at outputs/runs/session12/S12_E_d64/
             encoder/checkpoint_iter020000.pt (d=64 only). Uses the
             frozen HybridCNNViTEncoder forward pass.

Usage:
    python scripts/session18/encode_baseline_latents.py \\
        --baseline fukami --d 64 \\
        --checkpoint outputs/session18/exp_b1/fukami_ae_d64/checkpoint_iter020000.pt

    python scripts/session18/encode_baseline_latents.py \\
        --baseline pod --d 32 \\
        --basis outputs/session18/exp_b1/pod_d32/pod_basis.npz
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.data.omega_pipeline import OmegaPipeline  # noqa: E402

PREVENT = Path(os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT")))
CACHE = Path(
    os.environ.get(
        "VORTEX_JEPA_CACHE",
        str(PREVENT / "data" / "processed" / "vortex-jepa"),
    )
)


def gather_encounters(partition: str, split: str) -> list[dict]:
    """Resolve (case_id, encounter_index, path) triples for one split."""
    manifest_path = REPO / "configs" / "splits" / f"split_{partition}.json"
    with open(manifest_path) as f:
        m = json.load(f)
    out: list[dict] = []
    for cid, case in m["cases"].items():
        if split == "train" and case["split"] == "train":
            ks = case["train_encounter_indices"]
        elif split == "test_a" and case["split"] == "train":
            ks = case["test_a_encounter_indices"]
        elif split in ("test_b", "test_c") and case["split"] == split:
            ks = list(range(case["n_encounters_full"]))
        else:
            continue
        for k in ks:
            path = CACHE / partition / cid / f"encounter_{k:02d}.h5"
            if not path.exists():
                continue
            out.append(
                {
                    "case_id": cid,
                    "k": int(k),
                    "path": str(path),
                    "G": float(case["G"]),
                    "D": float(case["D"]),
                    "Y": float(case["Y"]),
                }
            )
    return out


def _load_fukami_encoder(
    checkpoint_path: Path,
    pipeline: OmegaPipeline,
    device: torch.device,
):
    """Rebuild a FukamiAEWrapper from a checkpoint and return ``encode_fn``.

    Honours whatever preprocessing the checkpoint was trained with:
      - If ``args["omega_pipeline_manifest"]`` is set -> use the OmegaPipeline.
      - Otherwise -> use raw / ``omega_scale`` (default 1000) like the
        original Session 9 Fukami runs.
    """
    from src.baselines.fukami_ae import FukamiAEWrapper

    blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_args = blob["args"]
    used_pipeline = train_args.get("omega_pipeline_manifest") is not None

    def _opt(key, default):
        v = train_args.get(key)
        return default if v is None else v

    wrapper = FukamiAEWrapper(
        latent_dim=int(train_args["d"]),
        n_deltas=len(_opt("observable_head_deltas", [8, 16, 24])),
        lambda_recon=float(_opt("lambda_recon", 1.0)),
        lambda_lift=float(_opt("lambda_lift", 1.0)),
        omega_pipeline=pipeline if used_pipeline else None,
        omega_scale=float(_opt("omega_scale", 1000.0)),
        recon_loss_type=str(_opt("recon_loss_type", "mse")),
        charbonnier_epsilon=float(_opt("charbonnier_epsilon", 0.05)),
        activation=str(_opt("activation", "relu")),
        use_conv_norm=not bool(_opt("no_conv_norm", False)),
    ).to(device)
    wrapper.load_state_dict(blob["wrapper_state_dict"])
    wrapper.eval()

    @torch.no_grad()
    def encode_fn(omega_THW: np.ndarray, case_id: str, k: int) -> np.ndarray:
        if used_pipeline:
            omega = pipeline.preprocess_raw(omega_THW, case_id, int(k))
            omega_t = torch.from_numpy(omega).unsqueeze(1).to(device)
            omega_norm = pipeline.normalize(omega_t)
            z = wrapper.encoder(omega_norm)
        else:
            omega_t = torch.from_numpy(omega_THW).unsqueeze(1).to(device)
            z = wrapper.encoder(omega_t / wrapper.omega_scale)
        return z.float().cpu().numpy()

    return encode_fn, int(train_args["d"])


def _load_pod_encoder(basis_path: Path, pipeline: OmegaPipeline):
    """Load a POD basis and return ``encode_fn`` that projects normalised
    omega frames onto the d truncated modes."""
    blob = np.load(basis_path)
    Phi = blob["Phi"].astype(np.float32)  # (H*W, d)
    mean = blob["mean"].astype(np.float32)  # (H*W,)
    d = int(blob["d"])

    def encode_fn(omega_THW: np.ndarray, case_id: str, k: int) -> np.ndarray:
        omega = pipeline.preprocess_raw(omega_THW, case_id, int(k))
        omega_t = torch.from_numpy(omega)
        omega_norm = pipeline.normalize(omega_t).numpy()  # (T, H, W)
        T = omega_norm.shape[0]
        flat = omega_norm.reshape(T, -1).astype(np.float32)
        coeffs = (flat - mean[None]) @ Phi  # (T, d)
        return coeffs

    return encode_fn, d


def _load_jepa_encoder(
    checkpoint_path: Path,
    pipeline: OmegaPipeline,
    device: torch.device,
):
    """Reconstruct the HybridCNNViTEncoder from a JEPA checkpoint."""
    from src.models.encoder import HybridCNNViTEncoder

    blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    d = int(blob.get("d", blob.get("run_config", {}).get("d", 64)))
    proj_norm = str(blob.get("run_config", {}).get("projection_norm", "batchnorm"))
    encoder = HybridCNNViTEncoder(latent_dim=d, projection_norm=proj_norm).to(device)
    # JEPA checkpoint stores the full JEPA module; extract encoder weights
    state = blob.get("jepa_state_dict", blob.get("encoder_state_dict", blob))
    enc_state = {
        k[len("encoder.") :]: v for k, v in state.items() if k.startswith("encoder.")
    }
    if not enc_state and "encoder_state_dict" in blob:
        enc_state = blob["encoder_state_dict"]
    if not enc_state:
        raise RuntimeError(
            f"could not extract encoder weights from {checkpoint_path}"
        )
    encoder.load_state_dict(enc_state)
    encoder.eval()

    @torch.no_grad()
    def encode_fn(omega_THW: np.ndarray, case_id: str, k: int) -> np.ndarray:
        omega = pipeline.preprocess_raw(omega_THW, case_id, int(k))
        omega_t = torch.from_numpy(omega).unsqueeze(1).to(device)  # (T, 1, H, W)
        omega_norm = pipeline.normalize(omega_t)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            z = encoder(omega_norm)  # (T, d)
        return z.float().cpu().numpy()

    return encode_fn, d


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Encode baseline latents (B1)")
    p.add_argument(
        "--baseline",
        type=str,
        choices=["fukami", "pod", "jepa"],
        required=True,
    )
    p.add_argument("--d", type=int, required=True)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Fukami AE or JEPA checkpoint path.",
    )
    p.add_argument(
        "--basis",
        type=Path,
        default=None,
        help="POD basis .npz path.",
    )
    p.add_argument(
        "--pipeline-manifest",
        type=Path,
        default=REPO / "outputs/data_pipeline/v1/manifest.json",
    )
    p.add_argument("--partition", type=str, default="v1")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: outputs/session18/exp_b1/latents_{baseline}_d{d}/",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test_a", "test_b", "test_c"],
    )
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.baseline in ("fukami", "jepa") and args.checkpoint is None:
        raise SystemExit(f"--checkpoint required for baseline={args.baseline}")
    if args.baseline == "pod" and args.basis is None:
        raise SystemExit("--basis required for baseline=pod")

    if args.output_dir is None:
        args.output_dir = (
            REPO / "outputs" / "session18" / "exp_b1"
            / f"latents_{args.baseline}_d{args.d}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = OmegaPipeline.from_manifest(args.pipeline_manifest)
    print(f"[encode] pipeline loaded; train_std={pipeline.train_stats.std:.4f}")

    if args.baseline == "fukami":
        from src.utils.device import require_rtx6000

        device = require_rtx6000(gpu_index=args.gpu)
        encode_fn, d_ckpt = _load_fukami_encoder(args.checkpoint, pipeline, device)
    elif args.baseline == "pod":
        device = torch.device("cpu")
        encode_fn, d_ckpt = _load_pod_encoder(args.basis, pipeline)
    else:
        from src.utils.device import require_rtx6000

        device = require_rtx6000(gpu_index=args.gpu)
        encode_fn, d_ckpt = _load_jepa_encoder(args.checkpoint, pipeline, device)

    if d_ckpt != args.d:
        print(
            f"[encode] WARNING: --d={args.d} disagrees with checkpoint d={d_ckpt}; "
            f"using checkpoint d. Filenames still use --d. Investigate before proceeding."
        )
        d_used = d_ckpt
    else:
        d_used = args.d

    print(
        f"[encode] baseline={args.baseline}  d={d_used}  device={device}  "
        f"output={args.output_dir}"
    )

    for split in args.splits:
        encs = gather_encounters(args.partition, split)
        n_enc = len(encs)
        if n_enc == 0:
            print(f"[encode] {split}: 0 encounters; skipping")
            continue

        z_full = np.zeros((n_enc, 120, d_used), dtype=np.float32)
        G = np.zeros(n_enc, dtype=np.float32)
        D = np.zeros(n_enc, dtype=np.float32)
        Y = np.zeros(n_enc, dtype=np.float32)
        case_ids: list[str] = []
        enc_idx = np.zeros(n_enc, dtype=np.int32)
        impact_frame = np.zeros(n_enc, dtype=np.int32)

        for i, e in enumerate(encs):
            with h5py.File(e["path"], "r") as f:
                omega = np.asarray(f["omega_z"], dtype=np.float32)
                impact_frame[i] = int(f.attrs.get("impact_frame_estimate", 40))
            z = encode_fn(omega, e["case_id"], e["k"])
            assert z.shape == (120, d_used), (
                f"unexpected z shape {z.shape} for {e['case_id']} k={e['k']}"
            )
            z_full[i] = z
            G[i] = e["G"]
            D[i] = e["D"]
            Y[i] = e["Y"]
            case_ids.append(e["case_id"])
            enc_idx[i] = e["k"]

            if (i + 1) % 25 == 0 or (i + 1) == n_enc:
                print(
                    f"[encode] {split}: encoded {i + 1}/{n_enc} encounters"
                )

        out_path = args.output_dir / f"{split}.npz"
        np.savez(
            out_path,
            z_full=z_full,
            G=G,
            D=D,
            Y=Y,
            case_ids=np.array(case_ids),
            encounter_indices=enc_idx,
            impact_frame=impact_frame,
        )
        print(f"[encode] {split}: wrote {out_path} ({z_full.nbytes / 1e6:.2f} MB)")

    print("[encode] DONE")


if __name__ == "__main__":
    main()
