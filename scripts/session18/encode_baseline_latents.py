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


def gather_encounters(partition: str, split: str,
                     split_manifest_path: "str | Path | None" = None) -> list[dict]:
    """Resolve (case_id, encounter_index, path) triples for one split."""
    if split_manifest_path is None:
        manifest_path = REPO / "configs" / "splits" / "split_v2.json"
    else:
        manifest_path = Path(split_manifest_path)
        if not manifest_path.is_absolute():
            manifest_path = REPO / manifest_path
    with open(manifest_path) as f:
        m = json.load(f)
    out: list[dict] = []
    for cid, case in m["cases"].items():
        if split == "train" and case["split"] == "train":
            ks = case["train_encounter_indices"]
        elif split == "test_a" and case["split"] == "train":
            ks = (case.get("val_encounter_indices") or case["test_a_encounter_indices"])
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

    # T8 beta-VAE cells (--vae) carry a BetaVAEEncoder (FC head emits 2d for
    # mu/logvar); BetaVAEWrapper subclasses FukamiAEWrapper and its plain
    # encoder.forward returns mu, so the encode path below is unchanged.
    wrapper_cls = FukamiAEWrapper
    if bool(_opt("vae", False)):
        from src.baselines.solera_rico import BetaVAEWrapper

        wrapper_cls = BetaVAEWrapper

    wrapper = wrapper_cls(
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
        encoder_kind=str(_opt("encoder", "cnn")),
    ).to(device)
    # Trained Track A Fukami cells carry a wake_observable_head used only in the
    # training loss; the encode-time wrapper omits it, so load non-strict and
    # ignore those unexpected keys. Guard that the encoder itself fully loaded.
    incompat = wrapper.load_state_dict(blob["wrapper_state_dict"], strict=False)
    enc_missing = [k for k in incompat.missing_keys if k.startswith("encoder.")]
    if enc_missing:
        raise RuntimeError(
            f"Fukami encoder weights did not load (missing {len(enc_missing)} "
            f"encoder keys, e.g. {enc_missing[:3]}); encoder_kind mismatch?"
        )
    wrapper.eval()

    @torch.no_grad()
    def encode_fn(omega_THW: np.ndarray, case_id: str, k: int) -> np.ndarray:
        # Feed 5D (1, T, 1, H, W): the FukamiCNNEncoder and the HybridCNNViTEncoder
        # (Track A A3 cnn_vit) both accept it and return (1, T, d); squeeze the
        # leading batch back to (T, d). The earlier 4D path worked only for the
        # CNN encoder, which is frame-independent.
        if used_pipeline:
            omega = pipeline.preprocess_raw(omega_THW, case_id, int(k))
            omega_t = torch.from_numpy(omega).unsqueeze(0).unsqueeze(2).to(device)
            omega_norm = pipeline.normalize(omega_t)
            z = wrapper.encoder(omega_norm)
        else:
            omega_t = torch.from_numpy(omega_THW).unsqueeze(0).unsqueeze(2).to(device)
            z = wrapper.encoder(omega_t / wrapper.omega_scale)
        return z.float().squeeze(0).cpu().numpy()

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
    from src.models.encoder import (
        CNNOnlyEncoder,
        HybridCNNViTEncoder,
        SpatioTemporalCNNViTEncoder,
    )

    blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    targs = blob.get("args", {})
    d = int(targs.get("d", blob.get("d", blob.get("run_config", {}).get("d", 64))))
    proj_norm = str(targs.get("projection_norm",
                              blob.get("run_config", {}).get("projection_norm", "batchnorm")))
    encoder_kind = str(targs.get("encoder", "hybrid"))
    if encoder_kind == "cnn_only":
        encoder = CNNOnlyEncoder(latent_dim=d, projection_norm=proj_norm).to(device)
    elif encoder_kind == "st_hybrid":
        tk = int(targs.get("temporal_kernel", 3))
        encoder = SpatioTemporalCNNViTEncoder(
            latent_dim=d, projection_norm=proj_norm, temporal_kernel=tk
        ).to(device)
    else:
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
        # JEPA encoder requires 5D (B, T, C, H, W).
        omega_t = torch.from_numpy(omega).unsqueeze(0).unsqueeze(2).to(device)
        omega_norm = pipeline.normalize(omega_t)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            z = encoder(omega_norm)  # (1, T, d)
        return z.float().squeeze(0).cpu().numpy()  # (T, d)

    return encode_fn, d


def _load_vjepa_encoder(checkpoint_path, pipeline, device, eval_stride: int = 32,
                        eval_interp: str = "nearest"):
    """Rebuild VJEPA from checkpoint; encode_fn -> (120, hidden=384) per encounter.

    Default (``eval_stride >= clip_len and eval_interp == "nearest"``) tiles the
    120-frame encounter into NON-overlapping 32-frame clips, frame-mean-pools each
    clip's tokens, concatenates along time, and nearest-upsamples to 120 (the
    original Session 18 path, byte-for-byte preserved).

    The finer path (``eval_stride < clip_len`` or ``eval_interp == "linear"``)
    uses OVERLAPPING clips, assigns each pooled frame-token its tubelet-center
    time, and linear-interpolates the scattered features onto the 120-frame grid.
    This removes the piecewise-constant-within-clip artefact that confounds
    short-horizon forecasting."""
    import torch
    from src.models.vjepa import VJEPA
    from src.models.vjepa_pool import frame_mean_pool

    blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    a = blob.get("args", {})
    model = VJEPA(hidden=int(a.get("hidden", 384)), depth=int(a.get("depth", 8)),
                  pred_depth=int(a.get("pred_depth", 6))).to(device)
    model.load_state_dict(blob["vjepa_state_dict"])
    model.eval()
    grid = model.grid
    clip_len = 32
    hidden = int(model.tokenizer.proj.out_channels)
    t_tubelet = clip_len // grid[0]

    @torch.no_grad()
    def encode_fn(omega_THW, case_id, k):
        import numpy as np
        omega = pipeline.preprocess_raw(omega_THW, case_id, int(k))  # (T,H,W)
        t_total = omega.shape[0]
        ot = torch.from_numpy(omega).unsqueeze(0).unsqueeze(2).to(device)  # (1,T,1,H,W)
        ot = pipeline.normalize(ot)

        def _clip_pool(s0):
            chunk = ot[:, s0:s0 + clip_len]
            if chunk.shape[1] < clip_len:
                pad = clip_len - chunk.shape[1]
                chunk = torch.cat([chunk, chunk[:, -1:].repeat(1, pad, 1, 1, 1)], dim=1)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                tok = model.encode_tokens(chunk)  # (1,N,D)
            return frame_mean_pool(tok.float(), grid).squeeze(0).cpu().numpy()  # (gt,D)

        if eval_stride >= clip_len and eval_interp == "nearest":
            feats = [_clip_pool(s) for s in range(0, t_total, clip_len)]
            feat = np.concatenate(feats, axis=0)
            idx = np.clip((np.arange(120) * feat.shape[0] / 120).astype(int), 0, feat.shape[0] - 1)
            return feat[idx]

        # finer path: overlapping clips, tubelet-center times, linear interp to 120
        starts = list(range(0, max(1, t_total - clip_len + 1), eval_stride))
        if starts[-1] != t_total - clip_len:
            starts.append(t_total - clip_len)
        times, feats = [], []
        for s0 in starts:
            fp = _clip_pool(s0)  # (gt, D)
            for j in range(grid[0]):
                times.append(s0 + j * t_tubelet + (t_tubelet - 1) / 2.0)
                feats.append(fp[j])
        return _vjepa_assemble(times, feats, 120)

    return encode_fn, hidden


def _vjepa_assemble(times, feats, n_out: int = 120):
    """Scattered (times, feats[M,D]) -> dense (n_out, D): average duplicate
    times, then linear-interpolate onto arange(n_out)."""
    import numpy as np
    times = np.asarray(times, dtype=np.float64)
    feats = np.asarray(feats, dtype=np.float64)
    ut, inv = np.unique(times, return_inverse=True)
    uf = np.zeros((len(ut), feats.shape[1]))
    cnt = np.zeros(len(ut))
    np.add.at(uf, inv, feats)
    np.add.at(cnt, inv, 1.0)
    uf /= cnt[:, None]
    grid = np.arange(n_out, dtype=np.float64)
    out = np.empty((n_out, feats.shape[1]), dtype=np.float32)
    for d in range(feats.shape[1]):
        out[:, d] = np.interp(grid, ut, uf[:, d])
    return out


def _vjepa_pca_to_64(output_dir, splits, n_components: int = 64) -> None:
    """Fit PCA(n_components) on TRAIN pooled features, transform + re-save every
    split's z_full from (n,120,hidden) to (n,120,n_components)."""
    import numpy as np
    from sklearn.decomposition import PCA

    tr = np.load(output_dir / "train.npz", allow_pickle=True)
    h = tr["z_full"].shape[-1]
    pca = PCA(n_components=n_components, random_state=0)
    pca.fit(tr["z_full"].reshape(-1, h))
    for split in splits:
        p = output_dir / f"{split}.npz"
        if not p.exists():
            continue
        d = {k: v for k, v in np.load(p, allow_pickle=True).items()}
        n, t, _ = d["z_full"].shape
        d["z_full"] = pca.transform(d["z_full"].reshape(-1, h)).reshape(
            n, t, n_components).astype(np.float32)
        np.savez(p, **d)
        print(f"[vjepa-pca] {split}: z_full -> {d['z_full'].shape}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Encode baseline latents (B1)")
    p.add_argument(
        "--baseline",
        type=str,
        choices=["fukami", "pod", "jepa", "vjepa"],
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
    p.add_argument("--partition", type=str, default="v1",
                   help="Cache partition; v1 cache stays valid for the v2 rerun.")
    p.add_argument("--split", type=str,
                   default="configs/splits/split_v2.json",
                   help="Path to split manifest. Default split_v2.json (v2 rerun).")
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
    p.add_argument("--device", type=str, default=None,
                   help="Explicit torch device (e.g. cuda:0 for an L40S); bypasses require_rtx6000.")
    p.add_argument("--vjepa-eval-stride", type=int, default=32,
                   help="V-JEPA eval clip stride (frames). <32 = overlapping clips.")
    p.add_argument("--vjepa-eval-interp", default="nearest", choices=["nearest", "linear"],
                   help="V-JEPA eval time interpolation to 120 frames.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.baseline in ("fukami", "jepa", "vjepa") and args.checkpoint is None:
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
    elif args.baseline == "vjepa":
        from src.utils.device import require_rtx6000

        device = torch.device(args.device) if args.device else require_rtx6000(gpu_index=args.gpu)
        encode_fn, d_ckpt = _load_vjepa_encoder(
            args.checkpoint, pipeline, device,
            eval_stride=args.vjepa_eval_stride, eval_interp=args.vjepa_eval_interp)
    else:
        from src.utils.device import require_rtx6000

        device = torch.device(args.device) if args.device else require_rtx6000(gpu_index=args.gpu)
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
        encs = gather_encounters(args.partition, split, split_manifest_path=args.split)
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

    if args.baseline == "vjepa":
        _vjepa_pca_to_64(args.output_dir, args.splits)

    print("[encode] DONE")


if __name__ == "__main__":
    main()
