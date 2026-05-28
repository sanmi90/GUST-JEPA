"""Session 18 B1: Test A reconstruction figure for the chosen Fukami d=3 β=0.01.

Mimics Fukami Fig 16 style: rows are cases (Baseline + representative gust),
columns are frames (pre-impact, impact, post-impact). For each row:
  - top:    reference omega (raw scale)
  - bottom: decoded omega from Fukami AE
ε (relative L2 error) shown per frame.

Usage:
    python scripts/session18/plot_recon_test_a.py \\
        --checkpoint outputs/session18/exp_b1/lcurve_sweep/d3_beta0.01/checkpoint_iter006000.pt \\
        --output outputs/session18/figures/exp_b1_fukami_d3_beta0.01_recon.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.baselines.fukami_ae import FukamiAEWrapper  # noqa: E402
from src.data.omega_pipeline import OmegaPipeline  # noqa: E402


def gather_test_a_encounters() -> list[dict]:
    """Get Test A encounters (same-case held-out, matches Fukami's val split)."""
    with open(REPO / "configs/splits/split_v2.json") as f:
        m = json.load(f)
    out = []
    for cid, case in m["cases"].items():
        if case["split"] == "train":
            for k in (case.get("val_encounter_indices") or case["test_a_encounter_indices"]):
                out.append({
                    "case_id": cid, "k": int(k),
                    "G": float(case["G"]), "D": float(case["D"]), "Y": float(case["Y"]),
                    "path": Path(f"/home/carlos/PREVENT/data/processed/vortex-jepa/v1/{cid}/encounter_{int(k):02d}.h5"),
                })
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test A reconstruction figure")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path,
                   default=REPO / "outputs/session18/figures/exp_b1_fukami_d3_recon.png")
    p.add_argument("--pipeline-manifest", type=Path,
                   default=REPO / "outputs/data_pipeline/v1/manifest.json")
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from src.utils.device import require_rtx6000
    device = require_rtx6000(gpu_index=args.gpu)
    pipe = OmegaPipeline.from_manifest(args.pipeline_manifest)

    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    a = blob["args"]
    print(f"[recon-fig] checkpoint: d={a['d']}  β={a['lambda_lift']}  iter={blob.get('iteration')}")
    wrapper = FukamiAEWrapper(
        latent_dim=int(a["d"]), n_deltas=len(a.get("observable_head_deltas", [0])),
        lambda_recon=1.0, lambda_lift=float(a["lambda_lift"]),
        omega_pipeline=pipe, recon_loss_type=str(a.get("recon_loss_type", "mse") or "mse"),
        activation="relu", use_conv_norm=True,
    ).to(device)
    wrapper.load_state_dict(blob["wrapper_state_dict"])
    wrapper.eval()

    # Pick representative Test A encounters covering parameter ranges
    encs = gather_test_a_encounters()
    chosen_ids = [
        "Baseline",                  # G=0 baseline
        "G+1.00_D0.50_Y+0.10",        # +G mild
        "G+2.00_D1.00_Y+0.10",        # +G strong
        "G-1.50_D0.50_Y-0.20",        # -G
        "G+0.50_D1.50_Y-0.20",        # weak gust, wide D
    ]
    rows = []
    for cid in chosen_ids:
        for e in encs:
            if e["case_id"] == cid:
                rows.append(e)
                break
    if not rows:
        # Fallback: pick the first 5
        rows = encs[:5]
    print(f"[recon-fig] using {len(rows)} representative encounters")

    # Three time slices per row: pre-impact (t=20), impact (40), post (60)
    t_slices = [20, 40, 60]

    fig, axes = plt.subplots(
        len(rows) * 2, len(t_slices) + 1,  # +1 for case label column
        figsize=(3.5 * (len(t_slices) + 1), 2.6 * len(rows) * 2),
        gridspec_kw={"width_ratios": [0.6] + [1.0] * len(t_slices)},
    )

    # Per-frame ε computation helper
    def frame_eps(tgt, prd):
        return float(np.linalg.norm(tgt - prd) / max(np.linalg.norm(tgt), 1.0))

    for r_idx, e in enumerate(rows):
        with h5py.File(e["path"], "r") as f:
            omega = np.asarray(f["omega_z"], dtype=np.float32)
        omega_clean = pipe.preprocess_raw(omega, e["case_id"], int(e["k"]))
        x = torch.from_numpy(omega_clean).unsqueeze(0).unsqueeze(2).to(device)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            xn = pipe.normalize(x); z = wrapper.encoder(xn); xn_hat = wrapper.decoder(z)
            x_hat = pipe.unnormalize(xn_hat).float()
        pred = x_hat.squeeze(0).squeeze(1).cpu().numpy()  # (120, 192, 96)

        # Per-encounter eps_volume
        eps_enc = np.linalg.norm(omega_clean - pred) / max(np.linalg.norm(omega_clean), 1.0)

        # Per-frame color limits (per row, fixed across cols for visual consistency)
        vmax = max(1.0, float(np.abs(omega_clean[t_slices]).max()))
        vlim = vmax * 0.8  # slight clip for visibility

        # Label column
        label_ax_t = axes[r_idx * 2, 0]
        label_ax_p = axes[r_idx * 2 + 1, 0]
        label_ax_t.axis("off")
        label_ax_p.axis("off")
        label_ax_t.text(0.5, 0.5,
                        f"{e['case_id']}\nk={e['k']}\n(G={e['G']:+.2f}, D={e['D']:.1f}, Y={e['Y']:+.2f})\nε_enc={eps_enc:.3f}",
                        ha="center", va="center", fontsize=9, transform=label_ax_t.transAxes)
        label_ax_t.set_title("Target → Decoded", fontsize=8) if r_idx == 0 else None
        for c_idx, t in enumerate(t_slices):
            tgt = omega_clean[t].T  # transpose for natural display (x horizontal, y vertical)
            prd = pred[t].T
            ax_t = axes[r_idx * 2, c_idx + 1]
            ax_p = axes[r_idx * 2 + 1, c_idx + 1]
            im_t = ax_t.imshow(tgt, origin="lower", cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto")
            im_p = ax_p.imshow(prd, origin="lower", cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto")
            eps_frame = frame_eps(omega_clean[t], pred[t])
            ax_t.set_title(f"t={t}  (ref)" + (f"   ε={eps_frame:.3f}" if r_idx == 0 else ""), fontsize=8)
            ax_p.set_title(f"decoded  ε={eps_frame:.3f}", fontsize=8)
            ax_t.set_xticks([]); ax_t.set_yticks([])
            ax_p.set_xticks([]); ax_p.set_yticks([])
            if c_idx == 0:
                ax_t.set_ylabel("y", fontsize=7)
                ax_p.set_ylabel("y", fontsize=7)

    fig.suptitle(
        f"Test A reconstruction: Fukami AE d=3, β={a['lambda_lift']}, "
        f"iter {blob.get('iteration')} (mid-plane Re=5000, MSE loss + bug fix)",
        fontsize=12, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[recon-fig] wrote {args.output}")


if __name__ == "__main__":
    main()
