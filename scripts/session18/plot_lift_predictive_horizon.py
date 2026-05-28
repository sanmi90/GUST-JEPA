"""Session 18 B1: how predictive is the present-frame lift estimate of FUTURE lift?

For both Fukami AE d=64 (β=0.01) and the production JEPA d=64:

  - Encode omega(t) at each Test A frame -> z(t)
  - Apply native lift/observable head -> C_L_pred(t)
  - Compare C_L_pred(t) against DNS C_L at lag tau in {0, 4, 8, 12, 16}.

If the encoder's latent at the present frame already implicitly encodes
future lift dynamics, the present-frame C_L estimate will stay close to
future DNS lift even without any explicit temporal model. The method
whose RMSE-vs-lag curve grows MORE SLOWLY is the more predictive of
future lift.

Output: outputs/session18/figures/exp_b1_lift_predictive_horizon.png
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
from src.models.encoder import HybridCNNViTEncoder  # noqa: E402
from src.models.observable_head import ObservableHead  # noqa: E402
from src.data.omega_pipeline import OmegaPipeline  # noqa: E402


PROD_JEPA_CKPT = REPO / "outputs/runs/session12/S12_E_d64/encoder/checkpoint_iter020000.pt"
FUKAMI_CKPT = REPO / "outputs/session18/exp_b1/fukami_ae_d64/checkpoint_iter006000.pt"


def gather_test_a():
    with open(REPO / "configs/splits/split_v2.json") as f:
        m = json.load(f)
    out = []
    for cid, case in m["cases"].items():
        if case["split"] == "train":
            for k in (case.get("val_encounter_indices") or case["test_a_encounter_indices"]):
                out.append({
                    "case_id": cid, "k": int(k),
                    "G": float(case["G"]),
                    "path": Path(f"/home/carlos/PREVENT/data/processed/vortex-jepa/v1/{cid}/encounter_{int(k):02d}.h5"),
                })
    return out


def load_fukami(device, pipe):
    blob = torch.load(FUKAMI_CKPT, map_location="cpu", weights_only=False)
    a = blob["args"]
    w = FukamiAEWrapper(
        latent_dim=int(a["d"]), n_deltas=len(a.get("observable_head_deltas", [0])),
        lambda_recon=1.0, lambda_lift=float(a["lambda_lift"]),
        omega_pipeline=pipe, recon_loss_type="mse", activation="relu", use_conv_norm=True,
    ).to(device)
    w.load_state_dict(blob["wrapper_state_dict"])
    w.eval()
    return w


def load_jepa(device):
    blob = torch.load(PROD_JEPA_CKPT, map_location="cpu", weights_only=False)
    a = blob["args"]
    state = blob["jepa_state_dict"]
    enc = HybridCNNViTEncoder(latent_dim=int(a["d"]),
                              projection_norm=a.get("projection_norm", "batchnorm")).to(device)
    enc.load_state_dict({k.removeprefix("encoder."): v for k, v in state.items()
                         if k.startswith("encoder.")}, strict=False)
    enc.eval()
    obs = ObservableHead(latent_dim=int(a["d"]),
                         n_deltas=len(a.get("observable_head_deltas", [0]))).to(device)
    obs.load_state_dict({k.removeprefix("observable_head."): v for k, v in state.items()
                         if k.startswith("observable_head.")}, strict=False)
    obs.eval()
    for p in enc.parameters(): p.requires_grad_(False)
    for p in obs.parameters(): p.requires_grad_(False)
    return enc, obs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path,
                   default=REPO / "outputs/session18/figures/exp_b1_lift_predictive_horizon.png")
    p.add_argument("--pipeline-manifest", type=Path,
                   default=REPO / "outputs/data_pipeline/v1/manifest.json")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--lags", nargs="+", type=int, default=[0, 4, 8, 12, 16, 20, 24])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from src.utils.device import require_rtx6000
    device = require_rtx6000(gpu_index=args.gpu)
    pipe = OmegaPipeline.from_manifest(args.pipeline_manifest)
    fukami = load_fukami(device, pipe)
    enc_jepa, obs_jepa = load_jepa(device)

    encs = gather_test_a()
    print(f"[lift-future] {len(encs)} Test A encounters; lags = {args.lags}")

    # For each encounter, collect:
    #   cl_true(t), cl_fukami_pred(t), cl_jepa_pred(t)  for t in [0, 119]
    all_true = []
    all_fukami = []
    all_jepa = []
    for e in encs:
        with h5py.File(e["path"], "r") as f:
            omega = np.asarray(f["omega_z"], dtype=np.float32)
            cl_true = np.asarray(f["C_L"], dtype=np.float32)
        if np.isnan(cl_true).any():
            print(f"  [skip] {e['case_id']} k={e['k']}: DNS C_L has NaNs")
            continue
        omega_clean = pipe.preprocess_raw(omega, e["case_id"], int(e["k"]))
        x = torch.from_numpy(omega_clean).unsqueeze(0).unsqueeze(2).to(device)
        with torch.no_grad():
            # fp32 throughout to avoid bf16 overflow on certain encounters
            xn = pipe.normalize(x)
            z_f = fukami.encoder(xn)
            cl_f = fukami.predict_lift(z_f).squeeze(0).squeeze(-1).cpu().numpy()
            z_j = enc_jepa(xn)
            cl_j = obs_jepa(z_j).squeeze(0).squeeze(-1).cpu().numpy()
        if np.isnan(cl_f).any() or np.isnan(cl_j).any():
            n_nan_f = int(np.isnan(cl_f).sum())
            n_nan_j = int(np.isnan(cl_j).sum())
            print(f"  [warn] {e['case_id']} k={e['k']}: NaNs Fukami={n_nan_f} JEPA={n_nan_j}; skipping encounter")
            continue
        all_true.append(cl_true)
        all_fukami.append(cl_f)
        all_jepa.append(cl_j)

    all_true = np.stack(all_true)        # (n_enc, 120)
    all_fukami = np.stack(all_fukami)
    all_jepa = np.stack(all_jepa)
    n_enc, T = all_true.shape

    # For each lag tau, compute RMSE between cl_pred(t) and cl_true(t+tau)
    # averaged over valid (t, encounter) pairs.
    fukami_rmse = []
    jepa_rmse = []
    persist_rmse = []  # Persistence baseline: cl_true(t) used to predict cl_true(t+tau)
    fukami_rmse_t40 = []  # RMSE specifically using t=40 (impact) as the anchor
    jepa_rmse_t40 = []
    persist_rmse_t40 = []
    for tau in args.lags:
        # Range over all valid anchor t in [0, T-1-tau]
        T_anchor = T - tau
        f_err = []
        j_err = []
        p_err = []
        for i in range(n_enc):
            f_err.append((all_fukami[i, :T_anchor] - all_true[i, tau:T_anchor + tau]) ** 2)
            j_err.append((all_jepa[i, :T_anchor] - all_true[i, tau:T_anchor + tau]) ** 2)
            p_err.append((all_true[i, :T_anchor] - all_true[i, tau:T_anchor + tau]) ** 2)
        fukami_rmse.append(float(np.sqrt(np.mean(np.concatenate(f_err)))))
        jepa_rmse.append(float(np.sqrt(np.mean(np.concatenate(j_err)))))
        persist_rmse.append(float(np.sqrt(np.mean(np.concatenate(p_err)))))
        # Anchor at impact frame t=40
        if 40 + tau < T:
            fukami_rmse_t40.append(float(np.sqrt(np.mean((all_fukami[:, 40] - all_true[:, 40 + tau]) ** 2))))
            jepa_rmse_t40.append(float(np.sqrt(np.mean((all_jepa[:, 40] - all_true[:, 40 + tau]) ** 2))))
            persist_rmse_t40.append(float(np.sqrt(np.mean((all_true[:, 40] - all_true[:, 40 + tau]) ** 2))))
        else:
            fukami_rmse_t40.append(np.nan); jepa_rmse_t40.append(np.nan); persist_rmse_t40.append(np.nan)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: all-t average
    ax = axes[0]
    ax.plot(args.lags, fukami_rmse, "o-", color="#d62728", label="Fukami AE d=64 lift head", linewidth=1.8, markersize=7)
    ax.plot(args.lags, jepa_rmse, "s-", color="#2ca02c", label="JEPA d=64 observable head", linewidth=1.8, markersize=7)
    ax.plot(args.lags, persist_rmse, "--", color="#7f7f7f", label="Persistence baseline (DNS C_L(t))", linewidth=1.2)
    ax.set_xlabel(r"Lag $\tau$ (frames)")
    ax.set_ylabel(r"RMSE: $|C_L^{pred}(t) - C_L^{true}(t+\tau)|$")
    ax.set_title("All anchor frames t (averaged)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_xlim(min(args.lags) - 0.5, max(args.lags) + 0.5)

    # Right: t=40 (impact) anchor
    ax = axes[1]
    ax.plot(args.lags, fukami_rmse_t40, "o-", color="#d62728", label="Fukami AE d=64 lift head", linewidth=1.8, markersize=7)
    ax.plot(args.lags, jepa_rmse_t40, "s-", color="#2ca02c", label="JEPA d=64 observable head", linewidth=1.8, markersize=7)
    ax.plot(args.lags, persist_rmse_t40, "--", color="#7f7f7f", label="Persistence baseline (DNS C_L(40))", linewidth=1.2)
    ax.set_xlabel(r"Lag $\tau$ from impact (frames)")
    ax.set_ylabel(r"RMSE")
    ax.set_title(r"Impact-frame anchor t = 40")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_xlim(min(args.lags) - 0.5, max(args.lags) + 0.5)

    fig.suptitle(
        "How predictive is the present-frame lift estimate of FUTURE lift?\n"
        "Lower curve = current C_L estimate matches future DNS C_L better (no temporal model used)",
        fontsize=11,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[lift-future] wrote {args.output}")
    print(f"\n=== ALL-t-average RMSE per lag ===")
    print(f"{'lag':>5s}  {'Fukami':>10s}  {'JEPA':>10s}  {'Persist':>10s}")
    for tau, f, j, p in zip(args.lags, fukami_rmse, jepa_rmse, persist_rmse):
        print(f"{tau:>5d}  {f:>10.4f}  {j:>10.4f}  {p:>10.4f}")
    print(f"\n=== Impact-anchor (t=40) RMSE per lag ===")
    print(f"{'lag':>5s}  {'Fukami':>10s}  {'JEPA':>10s}  {'Persist':>10s}")
    for tau, f, j, p in zip(args.lags, fukami_rmse_t40, jepa_rmse_t40, persist_rmse_t40):
        print(f"{tau:>5d}  {f:>10.4f}  {j:>10.4f}  {p:>10.4f}")


if __name__ == "__main__":
    main()
