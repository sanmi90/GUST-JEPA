"""Session 18 B1: Fukami-style C_L predicted vs real plot.

Mirrors Fukami and Taira PRF 2025 Fig 16(a): per-frame C_L from the lift head
applied to the encoder's latent, plotted against the DNS reference, with
multiple representative encounters colored by sign(G) (positive G red,
negative G blue, baseline gray).

Usage:
    python scripts/session18/plot_lift_recon.py \\
        --checkpoint outputs/session18/exp_b1/fukami_ae_d64/checkpoint_iter006000.pt \\
        --output outputs/session18/figures/exp_b1_lift_recon_d64.png
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
    p = argparse.ArgumentParser(description="Fukami-style C_L predicted vs real plot")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path,
                   default=REPO / "outputs/session18/figures/exp_b1_lift_recon.png")
    p.add_argument("--pipeline-manifest", type=Path,
                   default=REPO / "outputs/data_pipeline/v1/manifest.json")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument(
        "--n-cases", type=int, default=10,
        help="Number of test_a encounters to plot.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from src.utils.device import require_rtx6000
    device = require_rtx6000(gpu_index=args.gpu)
    pipe = OmegaPipeline.from_manifest(args.pipeline_manifest)
    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    a = blob["args"]
    wrapper = FukamiAEWrapper(
        latent_dim=int(a["d"]), n_deltas=len(a.get("observable_head_deltas", [0])),
        lambda_recon=1.0, lambda_lift=float(a["lambda_lift"]),
        omega_pipeline=pipe, recon_loss_type=str(a.get("recon_loss_type", "mse") or "mse"),
        activation="relu", use_conv_norm=True,
    ).to(device)
    wrapper.load_state_dict(blob["wrapper_state_dict"])
    wrapper.eval()
    print(f"[lift-plot] checkpoint d={a['d']} β={a['lambda_lift']} δ={a['observable_head_deltas']}")

    # Pick encounters covering G-spectrum
    encs = gather_test_a()
    # Sort by G, pick spread across G range
    encs_sorted = sorted(encs, key=lambda e: e["G"])
    n = min(args.n_cases, len(encs_sorted))
    chosen_idx = np.linspace(0, len(encs_sorted) - 1, n).astype(int)
    chosen = [encs_sorted[i] for i in chosen_idx]
    print(f"[lift-plot] plotting {len(chosen)} encounters:")
    for e in chosen:
        print(f"  {e['case_id']} k={e['k']}  G={e['G']:+.2f}")

    # Run encoder + lift head per encounter
    cl_per_enc = []  # list of (G, t_arr, cl_true_arr, cl_pred_arr)
    for e in chosen:
        with h5py.File(e["path"], "r") as f:
            omega = np.asarray(f["omega_z"], dtype=np.float32)
            cl_true = np.asarray(f["C_L"], dtype=np.float32)  # (120,)
        omega_clean = pipe.preprocess_raw(omega, e["case_id"], int(e["k"]))
        x = torch.from_numpy(omega_clean).unsqueeze(0).unsqueeze(2).to(device)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            xn = pipe.normalize(x); z = wrapper.encoder(xn)
            cl_hat = wrapper.predict_lift(z).float()  # (1, 120, n_deltas=1)
        cl_pred = cl_hat.squeeze(0).squeeze(-1).cpu().numpy()  # (120,)
        # Time axis in t/c units: dt_tc = 0.05, frames 0..119, impact at frame 40
        # Center on impact frame so t=0 is impact
        impact_frame = 40
        t_arr = (np.arange(120) - impact_frame) * 0.05
        cl_per_enc.append((e["G"], t_arr, cl_true, cl_pred))

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    G_vals = np.array([r[0] for r in cl_per_enc])
    pos_mask = G_vals > 0
    neg_mask = G_vals < 0
    zero_mask = np.isclose(G_vals, 0)

    # Plot each encounter
    for (G, t_arr, cl_true, cl_pred) in cl_per_enc:
        if G > 0:
            col_ref, col_pred = "#d62728", "#d62728"
            marker = "o"
        elif G < 0:
            col_ref, col_pred = "#1f77b4", "#1f77b4"
            marker = "^"
        else:
            col_ref, col_pred = "#7f7f7f", "#7f7f7f"
            marker = "s"
        ax.plot(t_arr, cl_true, "-", color=col_ref, alpha=0.6, linewidth=1.2)
        ax.plot(t_arr[::6], cl_pred[::6], marker, color=col_pred, alpha=0.8,
                markersize=5, markerfacecolor="white", markeredgewidth=1.2, label=None)

    # Compute per-encounter and overall RMS error
    errs = [np.sqrt(np.mean((cl_true - cl_pred) ** 2)) for (_, _, cl_true, cl_pred) in cl_per_enc]
    rel_errs = [
        np.linalg.norm(cl_true - cl_pred) / max(np.linalg.norm(cl_true), 1e-6)
        for (_, _, cl_true, cl_pred) in cl_per_enc
    ]
    rms_overall = np.mean(errs)
    rel_overall = np.mean(rel_errs)

    # Custom legend
    legend_elements = [
        plt.Line2D([0], [0], color="#d62728", linewidth=1.4, label="Reference (G > 0)"),
        plt.Line2D([0], [0], marker="o", color="#d62728", linewidth=0, markerfacecolor="white",
                   markeredgewidth=1.2, label="Decoded lift (G > 0)"),
        plt.Line2D([0], [0], color="#1f77b4", linewidth=1.4, label="Reference (G < 0)"),
        plt.Line2D([0], [0], marker="^", color="#1f77b4", linewidth=0, markerfacecolor="white",
                   markeredgewidth=1.2, label="Decoded lift (G < 0)"),
        plt.Line2D([0], [0], color="#7f7f7f", linewidth=1.4, label="Reference (G = 0, baseline)"),
        plt.Line2D([0], [0], marker="s", color="#7f7f7f", linewidth=0, markerfacecolor="white",
                   markeredgewidth=1.2, label="Decoded lift (G = 0)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    ax.set_xlabel("t / c (impact at t = 0)")
    ax.set_ylabel(r"$C_L$")
    ax.set_title(
        f"Fukami AE lift-head reconstruction (Test A, mid-plane Re=5000)\n"
        f"d={a['d']}  β={a['lambda_lift']}  per-frame δ=0  RMS={rms_overall:.3f}  "
        f"relative-L2={rel_overall:.3f}  n={len(chosen)} encounters"
    )
    ax.grid(alpha=0.3)
    ax.axvline(0, color="black", linewidth=0.5, linestyle=":", alpha=0.5)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[lift-plot] wrote {args.output}")
    print(f"[lift-plot] RMS error = {rms_overall:.4f}  relative-L2 = {rel_overall:.4f}")


if __name__ == "__main__":
    main()
