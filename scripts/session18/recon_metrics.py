"""Session 18 B1: consistent reconstruction-quality metrics for Fukami AE.

All metrics are computed on RAW omega scale (matches Fukami paper's
ε reporting). Internal consistency: data_range is set per encounter
from the encounter's own max|ω|, so SSIM constants C1, C2 scale with
the actual signal magnitude (Wang et al. 2004 K1=0.01, K2=0.03 defaults).

Metrics reported per encounter (mean across 120 frames):

    eps_volume      = sqrt(sum (ω - ω̂)² ) / max(sqrt(sum ω²), 1.0)
                      Relative L2 reconstruction error in raw scale.
                      Fukami's primary metric (PRF 2025 eqn defining ε).

    ssim_skimage    = skimage.metrics.structural_similarity on raw ω
                      with data_range = 2·max|ω| per encounter. Uses the
                      canonical Wang et al. (2004) Gaussian-windowed
                      implementation.

    pearson_r       = Pearson correlation between ω and ω̂ flattened.
                      Pure shape preservation (scale-invariant).

    amp_ratio       = std(ω̂) / std(ω). Pure amplitude preservation
                      (shape-invariant).

    mse_raw         = mean((ω - ω̂)²) in raw scale.

Reports per (split, encounter), aggregated per split.

Usage:
    python scripts/session18/recon_metrics.py \\
        --checkpoint outputs/session18/exp_b1/fukami_ae_d3/checkpoint_iter006000.pt \\
        --output outputs/session18/exp_b1/fukami_ae_d3/recon_metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.baselines.fukami_ae import FukamiAEWrapper  # noqa: E402
from src.data.omega_pipeline import OmegaPipeline  # noqa: E402

try:
    from skimage.metrics import structural_similarity as ssim_skimage
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


CACHE = Path("/home/carlos/PREVENT/data/processed/vortex-jepa/v1")


def gather_encs(split: str) -> list[dict]:
    with open(REPO / "configs/splits/split_v1.json") as f:
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
            p = CACHE / cid / f"encounter_{int(k):02d}.h5"
            if p.exists():
                out.append({"case_id": cid, "k": int(k), "path": str(p)})
    return out


def load_wrapper(ckpt: Path, pipe: OmegaPipeline, device: torch.device) -> FukamiAEWrapper:
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    args = blob["args"]
    w = FukamiAEWrapper(
        latent_dim=int(args["d"]),
        n_deltas=len(args.get("observable_head_deltas", [0])),
        lambda_recon=float(args.get("lambda_recon", 1.0)),
        lambda_lift=float(args.get("lambda_lift", 1.0)),
        omega_pipeline=pipe,
        recon_loss_type=str(args.get("recon_loss_type", "mse") or "mse"),
        activation=str(args.get("activation", "relu") or "relu"),
        use_conv_norm=not bool(args.get("no_conv_norm", False)),
    ).to(device)
    w.load_state_dict(blob["wrapper_state_dict"])
    w.eval()
    return w


def reconstruct_encounter(
    wrapper: FukamiAEWrapper,
    pipe: OmegaPipeline,
    e: dict,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (target_raw, pred_raw) of shape (120, 192, 96)."""
    with h5py.File(e["path"], "r") as f:
        omega = np.asarray(f["omega_z"], dtype=np.float32)
    target = pipe.preprocess_raw(omega, e["case_id"], int(e["k"]))  # raw, clipped
    x = torch.from_numpy(target).unsqueeze(0).unsqueeze(2).to(device)
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        xn = pipe.normalize(x)
        z = wrapper.encoder(xn)
        xn_hat = wrapper.decoder(z)
        x_hat = pipe.unnormalize(xn_hat).float()
    pred = x_hat.squeeze(0).squeeze(1).cpu().numpy()
    return target, pred


def per_encounter_metrics(target: np.ndarray, pred: np.ndarray) -> dict:
    """Compute all reconstruction metrics per encounter."""
    T = target.shape[0]
    # Per-frame metrics
    eps_per_frame = np.zeros(T)
    ssim_per_frame = np.zeros(T)
    pearson_per_frame = np.full(T, np.nan)
    amp_per_frame = np.zeros(T)
    mse_per_frame = np.zeros(T)
    for t in range(T):
        tgt, prd = target[t], pred[t]
        # eps_volume
        sig = np.linalg.norm(tgt)
        eps_per_frame[t] = np.linalg.norm(tgt - prd) / max(sig, 1.0)
        # MSE
        mse_per_frame[t] = float(((tgt - prd) ** 2).mean())
        # amp_ratio
        amp_per_frame[t] = prd.std() / max(tgt.std(), 1e-9)
        # pearson
        if tgt.std() > 1e-6 and prd.std() > 1e-6:
            pearson_per_frame[t] = float(np.corrcoef(tgt.flatten(), prd.flatten())[0, 1])
        # SSIM with data_range = 2 * max|tgt| (per-frame)
        if HAS_SKIMAGE:
            dr = 2 * max(float(np.abs(tgt).max()), 1e-3)
            ssim_per_frame[t] = float(ssim_skimage(tgt, prd, data_range=dr))
        else:
            ssim_per_frame[t] = np.nan

    return {
        "eps_volume_mean": float(eps_per_frame.mean()),
        "eps_volume_median": float(np.median(eps_per_frame)),
        "ssim_mean": float(np.nanmean(ssim_per_frame)),
        "ssim_median": float(np.nanmedian(ssim_per_frame)),
        "pearson_r_mean": float(np.nanmean(pearson_per_frame)),
        "pearson_r_median": float(np.nanmedian(pearson_per_frame)),
        "amp_ratio_mean": float(amp_per_frame.mean()),
        "amp_ratio_median": float(np.median(amp_per_frame)),
        "mse_raw_mean": float(mse_per_frame.mean()),
    }


def aggregate_split(per_enc: list[dict]) -> dict:
    if not per_enc:
        return {"n_encounters": 0}
    keys = list(per_enc[0].keys())
    out = {"n_encounters": len(per_enc)}
    for k in keys:
        vals = np.array([e[k] for e in per_enc])
        out[k + "_split_mean"] = float(np.nanmean(vals))
        out[k + "_split_median"] = float(np.nanmedian(vals))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Consistent reconstruction metrics")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--pipeline-manifest", type=Path,
                   default=REPO / "outputs/data_pipeline/v1/manifest.json")
    p.add_argument("--splits", nargs="+", default=["test_a", "test_b", "test_c"])
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not HAS_SKIMAGE:
        print("[recon_metrics] WARNING: scikit-image not available; SSIM will be NaN")
    from src.utils.device import require_rtx6000
    device = require_rtx6000(gpu_index=args.gpu)
    pipe = OmegaPipeline.from_manifest(args.pipeline_manifest)
    wrapper = load_wrapper(args.checkpoint, pipe, device)
    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    a = blob["args"]
    print(f"[recon_metrics] checkpoint d={a['d']} lambda_lift={a['lambda_lift']} "
          f"deltas={a.get('observable_head_deltas')} iter={blob.get('iteration')}")
    print(f"  using skimage SSIM with per-encounter data_range = 2*max|ω|")

    summary = {
        "checkpoint": str(args.checkpoint),
        "d": int(a["d"]),
        "lambda_lift": float(a["lambda_lift"]),
        "iteration": int(blob.get("iteration", -1)),
    }
    for split in args.splits:
        encs = gather_encs(split)
        per_enc = []
        for i, e in enumerate(encs):
            target, pred = reconstruct_encounter(wrapper, pipe, e, device)
            per_enc.append(per_encounter_metrics(target, pred))
            if (i + 1) % 10 == 0 or (i + 1) == len(encs):
                print(f"  {split}: {i+1}/{len(encs)}")
        agg = aggregate_split(per_enc)
        summary[split] = agg
        print(f"  {split}: "
              f"eps={agg.get('eps_volume_mean_split_mean', float('nan')):.4f}  "
              f"ssim={agg.get('ssim_mean_split_mean', float('nan')):.4f}  "
              f"pearson={agg.get('pearson_r_mean_split_mean', float('nan')):.4f}  "
              f"amp={agg.get('amp_ratio_mean_split_mean', float('nan')):.4f}")

    out_path = args.output or args.checkpoint.parent / "recon_metrics.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[recon_metrics] wrote {out_path}")


if __name__ == "__main__":
    main()
