"""Two-panel figure: PCA eigenspectrum + per-dim variance of raw latent z.

Panel 1 (left): PCA spectrum.
    sigma_i (or sigma_i^2) of the centered train-Z matrix, log-y, with
    cumulative variance fraction on the right axis. k=12 line annotated.

Panel 2 (right): per-channel variance of raw z (not PCA-rotated).
    Var(z[:, j]) for j in 0..d-1, optionally sorted descending. Shows which
    raw encoder dimensions carry energy vs. which are quasi-dead. Note this
    is NOT the PCA spectrum: raw dims can be correlated, so summing these
    variances overcounts.

Usage::

    python scripts/session11_pca_spectrum_figure.py \\
        --encoder-run outputs/runs/session11/W0_C_lam100 \\
        --pca-basis  outputs/runs/session11/W0_C_lam100/decoder_pca_k12/pca_basis.npz \\
        --output     outputs/runs/session11/W0_C_lam100/decoder_pca_k12/spectrum.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.omega_pipeline import OmegaPipeline  # noqa: E402
from src.models.encoder import HybridCNNViTEncoder  # noqa: E402
from src.utils.device import require_rtx6000  # noqa: E402


PREVENT = Path(os.environ.get("PREVENT_ROOT", "/home/carlos/PREVENT"))
CACHE = Path(os.environ.get("VORTEX_JEPA_CACHE", PREVENT / "data" / "processed" / "vortex-jepa"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--encoder-run", required=True, type=str)
    p.add_argument("--pca-basis", required=True, type=str)
    p.add_argument("--output", required=True, type=str)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--omega-pipeline-manifest", type=str,
                   default="outputs/data_pipeline/v1/manifest.json")
    p.add_argument("--partition", type=str, default="v1")
    p.add_argument("--k-mark", type=int, default=12,
                   help="Vertical k line to annotate on PCA spectrum.")
    p.add_argument("--sort-raw", action="store_true", default=True,
                   help="Sort raw-dim variances descending (recommended).")
    p.add_argument("--no-sort-raw", dest="sort_raw", action="store_false")
    return p.parse_args()


def gather_train(partition: str) -> list[dict]:
    with open(REPO / "configs" / "splits" / f"split_{partition}.json") as f:
        m = json.load(f)
    out = []
    for cid, c in m["cases"].items():
        if c["split"] != "train":
            continue
        for k in c["train_encounter_indices"]:
            p_ = CACHE / partition / cid / f"encounter_{k:02d}.h5"
            if p_.exists():
                out.append({"case_id": cid, "k": int(k), "path": str(p_)})
    return out


def load_encoder(encoder_run: Path, device: torch.device) -> tuple[HybridCNNViTEncoder, int]:
    cands = sorted(encoder_run.glob("checkpoint_iter*.pt"))
    if not cands:
        raise FileNotFoundError(f"No checkpoint under {encoder_run}")
    ckpt = torch.load(cands[-1], map_location="cpu", weights_only=False)
    a = ckpt["args"]
    enc = HybridCNNViTEncoder(
        latent_dim=int(a["d"]),
        projection_norm=a.get("projection_norm", "batchnorm"),
    )
    enc_state = {
        k.removeprefix("encoder."): v
        for k, v in ckpt["jepa_state_dict"].items()
        if k.startswith("encoder.")
    }
    enc.load_state_dict(enc_state, strict=False)
    enc.eval().to(device)
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc, int(a["d"])


def main() -> None:
    args = parse_args()
    device = require_rtx6000(gpu_index=args.gpu)

    manifest_path = Path(args.omega_pipeline_manifest)
    if not manifest_path.is_absolute():
        manifest_path = REPO / manifest_path
    pipe = OmegaPipeline.from_manifest(manifest_path)

    enc, d = load_encoder(Path(args.encoder_run), device)

    # Encode the train set -> Z (n_frames, d) for raw-dim variance.
    encs = gather_train(args.partition)
    Z_list = []
    for i, e in enumerate(encs):
        with h5py.File(e["path"], "r") as f:
            omega_raw = np.asarray(f["omega_z"], dtype=np.float32)
        omega_clean = pipe.preprocess_raw(omega_raw, e["case_id"], int(e["k"]))
        x = pipe.normalize(torch.from_numpy(omega_clean)).to(device)
        x = x.unsqueeze(0).unsqueeze(2)  # (1,T,1,H,W)
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            z = enc(x).float().squeeze(0)  # (T, d)
        Z_list.append(z.cpu())
    Z = torch.cat(Z_list, dim=0).numpy()
    raw_var = Z.var(axis=0)  # (d,)
    print(f"[spectrum] Z.shape={Z.shape}, raw-var sum={raw_var.sum():.2f}, "
          f"min={raw_var.min():.4f}, max={raw_var.max():.4f}")

    # PCA spectrum from saved basis.
    basis = np.load(args.pca_basis)
    S = np.asarray(basis["singular_values"], dtype=np.float64)  # (d,)
    if S.shape[0] < d:
        # PCA basis may have been truncated to k+10 in svd_lowrank; expand
        # by appending zeros so the x-axis spans the full d.
        S = np.concatenate([S, np.zeros(d - S.shape[0])])
    eig = S ** 2  # variance-proportional
    cumvar = np.cumsum(eig) / eig.sum()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.0))

    # ---- Panel 1: PCA spectrum ----
    idx = np.arange(1, d + 1)
    ax1.semilogy(idx, S, "o-", color="#1f77b4", label=r"$\sigma_i$ (singular value)")
    ax1.set_xlabel("PC index $i$ (1 = largest)")
    ax1.set_ylabel(r"$\sigma_i$ (log scale)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_xticks(np.arange(0, d + 1, 4))
    ax1.set_xlim(0.5, d + 0.5)
    ax1.grid(True, which="both", alpha=0.3)

    # cumulative variance on twin axis
    ax1b = ax1.twinx()
    ax1b.plot(idx, cumvar, "s-", color="#d62728", markersize=4,
              label="cumulative variance fraction")
    ax1b.set_ylabel("cumulative variance fraction", color="#d62728")
    ax1b.tick_params(axis="y", labelcolor="#d62728")
    ax1b.set_ylim(0, 1.02)
    ax1b.axhline(0.95, color="#888", linestyle=":", alpha=0.7)
    ax1b.text(d - 0.5, 0.955, "95%", ha="right", va="bottom", fontsize=8, color="#555")

    if args.k_mark is not None and 1 <= args.k_mark <= d:
        ax1.axvline(args.k_mark, color="k", linestyle="--", alpha=0.6)
        cv_at_k = float(cumvar[args.k_mark - 1])
        ax1b.plot([args.k_mark], [cv_at_k], "D", color="k", markersize=6, zorder=5)
        ax1.text(args.k_mark + 0.4, S[0], f"k={args.k_mark}\n{cv_at_k*100:.1f}%",
                 fontsize=9, va="top", ha="left")

    ax1.set_title(rf"PCA eigenspectrum of $z\in\mathbb{{R}}^{{{d}}}$  "
                  rf"(PR={ (S.sum()**2 / (S**2).sum()):.2f})")

    # ---- Panel 2: per-dim raw variance ----
    if args.sort_raw:
        order = np.argsort(raw_var)[::-1]
        plotted = raw_var[order]
        x_labels_kind = "sorted"
    else:
        plotted = raw_var
        x_labels_kind = "channel index"

    ax2.bar(np.arange(d), plotted, color="#2ca02c", alpha=0.85)
    ax2.set_xlabel(f"raw latent channel ({x_labels_kind})")
    ax2.set_ylabel(r"$\mathrm{Var}(z_j)$")
    ax2.set_title("Per-channel variance of raw z (NOT decorrelated)")
    ax2.set_xticks(np.arange(0, d, 4))
    ax2.grid(True, axis="y", alpha=0.3)
    # Annotate min/max
    ax2.text(0.98, 0.95,
             f"max={plotted.max():.3f}\nmin={plotted.min():.3f}\n"
             f"max/min={plotted.max()/max(plotted.min(),1e-9):.1f}x\n"
             f"sum={plotted.sum():.2f}",
             transform=ax2.transAxes, ha="right", va="top", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    fig.suptitle(f"W0_C_lam100 encoder latent diagnostics  "
                 f"(train: {Z.shape[0]} frames over {len(encs)} encounters)",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[spectrum] saved {out_path}")


if __name__ == "__main__":
    main()
