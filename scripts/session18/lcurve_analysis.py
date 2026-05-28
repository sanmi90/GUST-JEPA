"""Session 18 B1: L-curve analysis for the Fukami AE β hyperparameter.

Replicates the methodology of Fukami and Taira (arXiv:2305.08024, eqn 6
and ref [34]): plot reconstruction error vs lift error across β values,
find the elbow (maximum-curvature point in log-log space), and report
the β at the elbow as the dataset-optimal regularization weight.

Inputs:
    outputs/session18/exp_b1/lcurve_sweep/d3_beta{B}/
        final_eval.json     (test_a / test_b / test_c MSE, SSIM, etc.)
        metrics.jsonl       (training trajectory)

The L-curve point per β is:
    x = log10( ||q - q_hat||_2 / ||q||_2 )           [test_a eps_volume mean]
    y = log10( ||C_L - C_L_hat||_2 / ||C_L||_2 )     [test_a lift L2 relative, computed below]

For the lift relative L2 we re-evaluate each checkpoint on test_a by
loading the FukamiAEWrapper, running predict_lift, and comparing to
DNS C_L (frame-aligned).

Outputs:
    outputs/session18/figures/exp_b1_lcurve_fukami_d3.png
    outputs/session18/exp_b1/lcurve_sweep/lcurve_summary.json
        {beta_grid: [...], recon_err: [...], lift_err: [...],
         elbow_beta: float, elbow_idx: int}

Usage:
    python scripts/session18/lcurve_analysis.py \\
        --sweep-dir outputs/session18/exp_b1/lcurve_sweep
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

PREVENT = Path("/home/carlos/PREVENT")
CACHE = Path("/home/carlos/PREVENT/data/processed/vortex-jepa/v1")


def discover_betas(sweep_dir: Path) -> list[tuple[float, Path]]:
    rows = []
    for p in sorted(sweep_dir.glob("d3_beta*")):
        if not p.is_dir():
            continue
        try:
            beta = float(p.name.replace("d3_beta", ""))
        except ValueError:
            continue
        ck = p / "checkpoint_iter006000.pt"
        if not ck.exists():
            continue
        rows.append((beta, p))
    rows.sort(key=lambda r: r[0])
    return rows


def gather_test_a():
    with open(REPO / "configs/splits/split_v2.json") as f:
        m = json.load(f)
    out = []
    for cid, case in m["cases"].items():
        if case["split"] != "train":
            continue
        for k in (case.get("val_encounter_indices") or case["test_a_encounter_indices"]):
            path = CACHE / cid / f"encounter_{int(k):02d}.h5"
            if path.exists():
                out.append({"case_id": cid, "k": int(k), "path": path})
    return out


def compute_lift_recon_errors(
    ckpt_path: Path, pipeline: OmegaPipeline, device: torch.device, encs: list[dict]
) -> tuple[float, float]:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = blob["args"]
    wrapper = FukamiAEWrapper(
        latent_dim=int(args["d"]),
        n_deltas=len(args.get("observable_head_deltas", [0])),
        lambda_recon=float(args.get("lambda_recon", 1.0)),
        lambda_lift=float(args.get("lambda_lift", 1.0)),
        omega_pipeline=pipeline,
        recon_loss_type=str(args.get("recon_loss_type", "mse") or "mse"),
        activation=str(args.get("activation", "relu") or "relu"),
        use_conv_norm=not bool(args.get("no_conv_norm", False)),
    ).to(device)
    wrapper.load_state_dict(blob["wrapper_state_dict"])
    wrapper.eval()

    recon_num_sq = 0.0
    recon_den_sq = 0.0
    lift_num_sq = 0.0
    lift_den_sq = 0.0

    with torch.no_grad():
        for e in encs:
            with h5py.File(e["path"], "r") as f:
                omega = np.asarray(f["omega_z"], dtype=np.float32)
                cl_true = np.asarray(f["C_L"], dtype=np.float32)
            omega = pipeline.preprocess_raw(omega, e["case_id"], int(e["k"]))
            x = torch.from_numpy(omega).unsqueeze(0).unsqueeze(2).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                xn = pipeline.normalize(x)
                z = wrapper.encoder(xn)
                xn_hat = wrapper.decoder(z)
                x_hat = pipeline.unnormalize(xn_hat).float()
                cl_hat = wrapper.predict_lift(z).float().squeeze(-1)
            x_hat_np = x_hat.squeeze(0).squeeze(1).cpu().numpy()
            cl_hat_np = cl_hat.squeeze(0).cpu().numpy()
            recon_num_sq += float(((omega - x_hat_np) ** 2).sum())
            recon_den_sq += float((omega ** 2).sum())
            lift_num_sq += float(((cl_true - cl_hat_np) ** 2).sum())
            lift_den_sq += float((cl_true ** 2).sum())

    recon_rel = float(np.sqrt(recon_num_sq / max(recon_den_sq, 1e-12)))
    lift_rel = float(np.sqrt(lift_num_sq / max(lift_den_sq, 1e-12)))
    return recon_rel, lift_rel


def find_elbow_log(x: np.ndarray, y: np.ndarray) -> int:
    """Maximum-curvature point in log-log space (the Tikhonov L-curve elbow).

    Treat (log10 x, log10 y) as a 2D curve. Compute discrete curvature via
    finite differences and pick the maximum-curvature index.
    """
    lx = np.log10(np.maximum(x, 1e-12))
    ly = np.log10(np.maximum(y, 1e-12))
    if len(lx) < 3:
        return int(np.argmin(np.hypot(lx - lx.min(), ly - ly.min())))
    # Normalize so curvature is invariant to axis scaling
    lxn = (lx - lx.min()) / max(lx.ptp(), 1e-12)
    lyn = (ly - ly.min()) / max(ly.ptp(), 1e-12)
    dx = np.gradient(lxn)
    dy = np.gradient(lyn)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    kappa = np.abs(dx * ddy - dy * ddx) / np.maximum((dx ** 2 + dy ** 2) ** 1.5, 1e-12)
    # Avoid endpoints
    kappa[0] = kappa[-1] = 0.0
    return int(np.argmax(kappa))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="L-curve analysis for Fukami AE β")
    p.add_argument("--sweep-dir", type=Path,
                   default=REPO / "outputs/session18/exp_b1/lcurve_sweep")
    p.add_argument("--pipeline-manifest", type=Path,
                   default=REPO / "outputs/data_pipeline/v1/manifest.json")
    p.add_argument("--output-fig", type=Path,
                   default=REPO / "outputs/session18/figures/exp_b1_lcurve_fukami_d3.png")
    p.add_argument("--output-json", type=Path,
                   default=REPO / "outputs/session18/exp_b1/lcurve_sweep/lcurve_summary.json")
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = discover_betas(args.sweep_dir)
    if not rows:
        raise SystemExit(f"No completed sweep checkpoints under {args.sweep_dir}")
    print(f"[lcurve] {len(rows)} β values found")

    from src.utils.device import require_rtx6000
    device = require_rtx6000(gpu_index=args.gpu)
    pipeline = OmegaPipeline.from_manifest(args.pipeline_manifest)
    encs = gather_test_a()
    print(f"[lcurve] test_a encounters: {len(encs)}")

    betas, recon_errs, lift_errs = [], [], []
    for beta, run_dir in rows:
        ck = run_dir / "checkpoint_iter006000.pt"
        recon_rel, lift_rel = compute_lift_recon_errors(ck, pipeline, device, encs)
        print(f"[lcurve] β={beta:>6.3f}  recon_rel_L2={recon_rel:.4f}  lift_rel_L2={lift_rel:.4f}")
        betas.append(beta)
        recon_errs.append(recon_rel)
        lift_errs.append(lift_rel)

    betas = np.asarray(betas)
    recon_errs = np.asarray(recon_errs)
    lift_errs = np.asarray(lift_errs)

    elbow_idx = find_elbow_log(recon_errs, lift_errs)
    elbow_beta = float(betas[elbow_idx])
    print(f"[lcurve] elbow β = {elbow_beta} (idx {elbow_idx})")

    args.output_fig.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(recon_errs, lift_errs, "o-", color="#1f77b4")
    for i, b in enumerate(betas):
        ax.annotate(f"β={b:g}",
                    (recon_errs[i], lift_errs[i]),
                    textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.loglog(recon_errs[elbow_idx], lift_errs[elbow_idx], "s",
              color="#d62728", markersize=12, label=f"elbow: β = {elbow_beta:g}")
    ax.set_xlabel(r"$\|q - \hat{q}\|_2 / \|q\|_2$ (Test A, mean over encounters)")
    ax.set_ylabel(r"$\|C_L - \hat{C_L}\|_2 / \|C_L\|_2$ (Test A)")
    ax.set_title("L-curve for Fukami AE β on mid-plane Re=5000 data (d=3)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output_fig, dpi=200)
    plt.close(fig)
    print(f"[lcurve] wrote {args.output_fig}")

    out = {
        "beta_grid": betas.tolist(),
        "recon_err_relL2": recon_errs.tolist(),
        "lift_err_relL2": lift_errs.tolist(),
        "elbow_idx": int(elbow_idx),
        "elbow_beta": elbow_beta,
        "test_a_n_encounters": len(encs),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[lcurve] wrote {args.output_json}")


if __name__ == "__main__":
    main()
