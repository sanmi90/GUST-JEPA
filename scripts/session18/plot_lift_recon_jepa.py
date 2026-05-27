"""Session 18 B1: Fukami-style C_L predicted vs real plot for JEPA.

Mirrors scripts/session18/plot_lift_recon.py but uses the production JEPA's
encoder + observable_head (cl_future at delta=0) on Session 14 precomputed
JEPA latents (or re-encodes from raw omega if those don't have C_L).

Usage:
    python scripts/session18/plot_lift_recon_jepa.py \\
        --jepa-checkpoint outputs/runs/session12/S12_E_d64/encoder/checkpoint_iter020000.pt \\
        --output outputs/session18/figures/exp_b1_lift_recon_jepa_d64.png
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

from src.models.encoder import HybridCNNViTEncoder  # noqa: E402
from src.models.observable_head import ObservableHead  # noqa: E402
from src.data.omega_pipeline import OmegaPipeline  # noqa: E402


def gather_test_a():
    with open(REPO / "configs/splits/split_v1.json") as f:
        m = json.load(f)
    out = []
    for cid, case in m["cases"].items():
        if case["split"] == "train":
            for k in case["test_a_encounter_indices"]:
                out.append({
                    "case_id": cid, "k": int(k),
                    "G": float(case["G"]), "D": float(case["D"]), "Y": float(case["Y"]),
                    "path": Path(f"/home/carlos/PREVENT/data/processed/vortex-jepa/v1/{cid}/encounter_{int(k):02d}.h5"),
                })
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fukami-style C_L plot for JEPA")
    p.add_argument("--jepa-checkpoint", type=Path,
                   default=REPO / "outputs/runs/session12/S12_E_d64/encoder/checkpoint_iter020000.pt")
    p.add_argument("--output", type=Path,
                   default=REPO / "outputs/session18/figures/exp_b1_lift_recon_jepa_d64.png")
    p.add_argument("--pipeline-manifest", type=Path,
                   default=REPO / "outputs/data_pipeline/v1/manifest.json")
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--n-cases", type=int, default=12)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from src.utils.device import require_rtx6000
    device = require_rtx6000(gpu_index=args.gpu)
    pipe = OmegaPipeline.from_manifest(args.pipeline_manifest)
    blob = torch.load(args.jepa_checkpoint, map_location="cpu", weights_only=False)
    a = blob["args"]
    state = blob["jepa_state_dict"]
    print(f"[jepa-lift-plot] checkpoint: d={a['d']}  obs_deltas={a.get('observable_head_deltas')}")

    # Reconstruct encoder
    encoder = HybridCNNViTEncoder(
        latent_dim=int(a["d"]),
        projection_norm=a.get("projection_norm", "batchnorm"),
    ).to(device)
    enc_state = {k.removeprefix("encoder."): v for k, v in state.items() if k.startswith("encoder.")}
    encoder.load_state_dict(enc_state, strict=False)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    # Reconstruct observable head
    n_deltas = len(a.get("observable_head_deltas", [0]))
    obs_head = ObservableHead(latent_dim=int(a["d"]), n_deltas=n_deltas).to(device)
    obs_state = {k.removeprefix("observable_head."): v
                 for k, v in state.items() if k.startswith("observable_head.")}
    obs_head.load_state_dict(obs_state, strict=False)
    obs_head.eval()
    for p in obs_head.parameters():
        p.requires_grad_(False)
    print(f"[jepa-lift-plot] loaded encoder ({sum(p.numel() for p in encoder.parameters())/1e6:.2f}M params) "
          f"+ observable_head ({sum(p.numel() for p in obs_head.parameters())} params)")

    # Pick test_a encounters spread across G
    encs = gather_test_a()
    encs_sorted = sorted(encs, key=lambda e: e["G"])
    n = min(args.n_cases, len(encs_sorted))
    chosen_idx = np.linspace(0, len(encs_sorted) - 1, n).astype(int)
    chosen = [encs_sorted[i] for i in chosen_idx]

    cl_per_enc = []
    for e in chosen:
        with h5py.File(e["path"], "r") as f:
            omega = np.asarray(f["omega_z"], dtype=np.float32)
            cl_true = np.asarray(f["C_L"], dtype=np.float32)
        omega_clean = pipe.preprocess_raw(omega, e["case_id"], int(e["k"]))
        x = torch.from_numpy(omega_clean).unsqueeze(0).unsqueeze(2).to(device)  # (1, T, 1, H, W)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            xn = pipe.normalize(x)
            z = encoder(xn)  # (1, T, d)
            cl_hat = obs_head(z).float()  # (1, T, n_deltas)
        cl_pred = cl_hat.squeeze(0).squeeze(-1).cpu().numpy()  # (120,)
        impact_frame = 40
        t_arr = (np.arange(120) - impact_frame) * 0.05
        cl_per_enc.append((e["G"], t_arr, cl_true, cl_pred))

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for (G, t_arr, cl_true, cl_pred) in cl_per_enc:
        if G > 0:
            col, marker = "#d62728", "o"
        elif G < 0:
            col, marker = "#1f77b4", "^"
        else:
            col, marker = "#7f7f7f", "s"
        ax.plot(t_arr, cl_true, "-", color=col, alpha=0.6, linewidth=1.2)
        ax.plot(t_arr[::6], cl_pred[::6], marker, color=col, alpha=0.8,
                markersize=5, markerfacecolor="white", markeredgewidth=1.2)

    errs = [np.sqrt(np.mean((cl_true - cl_pred) ** 2)) for (_, _, cl_true, cl_pred) in cl_per_enc]
    rel_errs = [
        np.linalg.norm(cl_true - cl_pred) / max(np.linalg.norm(cl_true), 1e-6)
        for (_, _, cl_true, cl_pred) in cl_per_enc
    ]
    rms = np.mean(errs)
    rel = np.mean(rel_errs)

    legend_elements = [
        plt.Line2D([0], [0], color="#d62728", linewidth=1.4, label="Reference (G > 0)"),
        plt.Line2D([0], [0], marker="o", color="#d62728", linewidth=0, markerfacecolor="white",
                   markeredgewidth=1.2, label="Decoded lift (G > 0)"),
        plt.Line2D([0], [0], color="#1f77b4", linewidth=1.4, label="Reference (G < 0)"),
        plt.Line2D([0], [0], marker="^", color="#1f77b4", linewidth=0, markerfacecolor="white",
                   markeredgewidth=1.2, label="Decoded lift (G < 0)"),
        plt.Line2D([0], [0], color="#7f7f7f", linewidth=1.4, label="Reference (G = 0)"),
        plt.Line2D([0], [0], marker="s", color="#7f7f7f", linewidth=0, markerfacecolor="white",
                   markeredgewidth=1.2, label="Decoded lift (G = 0)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)
    ax.set_xlabel("t / c (impact at t = 0)")
    ax.set_ylabel(r"$C_L$")
    ax.set_title(
        f"JEPA observable-head lift reconstruction (Test A, mid-plane Re=5000)\n"
        f"d={a['d']}  obs_weight={a.get('observable_head_weight', 0.01)}  δ=0  "
        f"RMS={rms:.3f}  relative-L2={rel:.3f}  n={len(chosen)} encounters"
    )
    ax.grid(alpha=0.3)
    ax.axvline(0, color="black", linewidth=0.5, linestyle=":", alpha=0.5)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[jepa-lift-plot] wrote {args.output}")
    print(f"[jepa-lift-plot] RMS={rms:.4f}  relative-L2={rel:.4f}")


if __name__ == "__main__":
    main()
