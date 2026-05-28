"""Disentanglement diagnostic: relate latent coords to (G, D, Y).

Inspired by Wang, Tirelli, Discetti, Ianiro (arXiv:2604.18059, 2026):
look for axis-aligned encoding of physical factors of variation in the
JEPA latent. We do NOT use their VAE loss; we just borrow the diagnostic
of regressing each latent coordinate on each physical parameter and
reporting per-axis R^2.

Two views:

1. Raw latent z in R^32 from the encoder (per-encounter, averaged in a
   small window around the impact frame).
2. PCA-projected z in R^k (k=12) using the basis saved by
   session11_pca_decoder.py.

Three outputs:

- Heatmap (top): R^2 of each raw latent axis predicting G / D / Y.
  Off-diagonal pattern reveals correlation; broad rows mean entanglement.
- Heatmap (middle): same for the PCA-projected axes (ordered by
  decreasing variance). Cleaner because PCs are decorrelated by
  construction, but the question is whether (G, D, Y) align with the
  high-variance PCs.
- Scatter (bottom): 2D projection of (PC1, PC2) colored by each of
  G, D, Y across train + test_b + test_c. A clean colour gradient = the
  parameter lives in this plane; a checkerboard = entangled across more
  PCs.

Usage::

    python scripts/session11_latent_disentanglement.py \\
        --encoder-run outputs/runs/session11/W0_C_lam100 \\
        --pca-basis  outputs/runs/session11/W0_C_lam100/decoder_pca_k12/pca_basis.npz \\
        --output     outputs/runs/session11/W0_C_lam100/decoder_pca_k12/disentanglement.png
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
    p.add_argument("--impact-frame", type=int, default=40)
    p.add_argument("--frame-window", type=int, default=5,
                   help="Average latent over [impact-w, impact+w] inclusive.")
    return p.parse_args()


def gather_all(partition: str) -> list[dict]:
    """All encounters across train / test_a / test_b / test_c with (G,D,Y)."""
    with open(REPO / "configs" / "splits" / f"split_{partition}.json") as f:
        m = json.load(f)
    out = []
    for cid, c in m["cases"].items():
        if c["split"] == "train":
            train_ks = c.get("train_encounter_indices", [])
            tA_ks = (c.get("val_encounter_indices") or c.get("test_a_encounter_indices", []))
            for k in train_ks:
                out.append(dict(case_id=cid, k=int(k), G=c["G"], D=c["D"], Y=c["Y"],
                                split="train"))
            for k in tA_ks:
                out.append(dict(case_id=cid, k=int(k), G=c["G"], D=c["D"], Y=c["Y"],
                                split="test_a"))
        elif c["split"] in ("test_b", "test_c"):
            for k in range(c["n_encounters_full"]):
                out.append(dict(case_id=cid, k=int(k), G=c["G"], D=c["D"], Y=c["Y"],
                                split=c["split"]))
    for r in out:
        r["path"] = str(CACHE / partition / r["case_id"] / f"encounter_{r['k']:02d}.h5")
    return [r for r in out if Path(r["path"]).exists()]


def load_encoder(encoder_run: Path, device: torch.device) -> tuple[HybridCNNViTEncoder, int]:
    cands = sorted(encoder_run.glob("checkpoint_iter*.pt"))
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


def per_axis_r2(Z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """R^2 of a univariate linear regression y ~ a + b * Z[:, j] per j.

    For univariate linear regression, R^2 == Pearson_r^2.
    """
    Zc = Z - Z.mean(0, keepdims=True)
    yc = y - y.mean()
    num = (Zc * yc[:, None]).mean(0)
    var_z = (Zc ** 2).mean(0) + 1e-12
    var_y = (yc ** 2).mean() + 1e-12
    r = num / np.sqrt(var_z * var_y)
    return r ** 2  # (d,)


def main() -> None:
    args = parse_args()
    device = require_rtx6000(gpu_index=args.gpu)

    manifest_path = Path(args.omega_pipeline_manifest)
    if not manifest_path.is_absolute():
        manifest_path = REPO / manifest_path
    pipe = OmegaPipeline.from_manifest(manifest_path)
    enc, d = load_encoder(Path(args.encoder_run), device)

    encs = gather_all(args.partition)
    print(f"[disent] encounters: {len(encs)}")

    w = args.frame_window
    fL, fR = args.impact_frame - w, args.impact_frame + w + 1
    Z_imp = np.zeros((len(encs), d), dtype=np.float32)
    G = np.zeros(len(encs), dtype=np.float32)
    D = np.zeros(len(encs), dtype=np.float32)
    Y = np.zeros(len(encs), dtype=np.float32)
    splits = np.empty(len(encs), dtype=object)
    for i, e in enumerate(encs):
        with h5py.File(e["path"], "r") as f:
            omega_raw = np.asarray(f["omega_z"], dtype=np.float32)
        omega_clean = pipe.preprocess_raw(omega_raw, e["case_id"], int(e["k"]))
        x = pipe.normalize(torch.from_numpy(omega_clean)).to(device)
        x = x.unsqueeze(0).unsqueeze(2)
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            z = enc(x).float().squeeze(0).cpu().numpy()  # (T, d)
        Z_imp[i] = z[fL:fR].mean(axis=0)
        G[i] = e["G"]; D[i] = e["D"]; Y[i] = e["Y"]; splits[i] = e["split"]
    print(f"[disent] Z_imp shape={Z_imp.shape}")

    # PCA projection.
    basis = np.load(args.pca_basis)
    mean = basis["mean"].astype(np.float32)
    P = basis["P"].astype(np.float32)   # (d, k)
    k = int(basis["k"])
    Z_pca = (Z_imp - mean[None]) @ P    # (n, k)

    # Per-axis R^2 against each of G, D, Y.
    r2_raw = np.stack([per_axis_r2(Z_imp, v) for v in (G, D, Y)], axis=0)  # (3, d)
    r2_pca = np.stack([per_axis_r2(Z_pca, v) for v in (G, D, Y)], axis=0)  # (3, k)

    # Multivariate R^2 (full subspace): R^2 of best linear regression
    # using ALL axes. Tells us the ceiling vs. the per-axis story.
    def multi_r2(Z: np.ndarray, y: np.ndarray) -> float:
        Zc = np.concatenate([np.ones((Z.shape[0], 1)), Z], axis=1)
        beta, *_ = np.linalg.lstsq(Zc, y, rcond=None)
        yhat = Zc @ beta
        ss_res = ((y - yhat) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum() + 1e-12
        return 1.0 - ss_res / ss_tot
    multi_raw = [multi_r2(Z_imp, v) for v in (G, D, Y)]
    multi_pca_full = [multi_r2(Z_pca, v) for v in (G, D, Y)]
    print(f"[disent] full-subspace R^2 raw d=32: G={multi_raw[0]:.3f} "
          f"D={multi_raw[1]:.3f} Y={multi_raw[2]:.3f}")
    print(f"[disent] full-subspace R^2 PCA k={k}: G={multi_pca_full[0]:.3f} "
          f"D={multi_pca_full[1]:.3f} Y={multi_pca_full[2]:.3f}")

    # ---------------- figure ----------------
    fig = plt.figure(figsize=(13.5, 9.5))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.6],
                          width_ratios=[1, 1, 1, 0.04],
                          hspace=0.55, wspace=0.25)

    # Heatmap A: raw d=32.
    axA = fig.add_subplot(gs[0, 0:3])
    imA = axA.imshow(r2_raw, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    axA.set_yticks(range(3)); axA.set_yticklabels(["G", "D", "Y"])
    axA.set_xticks(np.arange(0, d, 4))
    axA.set_xlabel(f"raw latent channel (j = 0 .. {d-1})")
    axA.set_title(
        f"Per-channel R$^2$, raw z (d={d})  |  full-subspace R$^2$: "
        f"G={multi_raw[0]:.2f}, D={multi_raw[1]:.2f}, Y={multi_raw[2]:.2f}"
    )
    cbA = fig.add_subplot(gs[0, 3])
    fig.colorbar(imA, cax=cbA, label="R$^2$")

    # Heatmap B: PCA k.
    axB = fig.add_subplot(gs[1, 0:3])
    imB = axB.imshow(r2_pca, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    axB.set_yticks(range(3)); axB.set_yticklabels(["G", "D", "Y"])
    axB.set_xticks(np.arange(0, k))
    axB.set_xticklabels([f"PC{i+1}" for i in range(k)], rotation=0, fontsize=8)
    axB.set_xlabel(f"PCA component (1 .. {k}, ordered by variance)")
    axB.set_title(
        f"Per-PC R$^2$, PCA-projected z (k={k})  |  full-subspace R$^2$: "
        f"G={multi_pca_full[0]:.2f}, D={multi_pca_full[1]:.2f}, Y={multi_pca_full[2]:.2f}"
    )
    cbB = fig.add_subplot(gs[1, 3])
    fig.colorbar(imB, cax=cbB, label="R$^2$")

    # Scatter: PC1 vs PC2 colored by each of G, D, Y.
    for ci, (name, v) in enumerate((("G", G), ("D", D), ("Y", Y))):
        ax = fig.add_subplot(gs[2, ci])
        sc = ax.scatter(Z_pca[:, 0], Z_pca[:, 1], c=v, cmap="coolwarm",
                        s=18, alpha=0.85, edgecolor="k", linewidth=0.2)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2" if ci == 0 else "")
        ax.set_title(f"colored by {name}")
        fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02, label=name)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Latent disentanglement diagnostic: W0_C_lam100 vs (G, D, Y) at "
        f"impact frame {args.impact_frame} ± {w}\n"
        f"(after the diagnostic of Wang/Tirelli/Discetti/Ianiro, arXiv:2604.18059, 2026)",
        fontsize=11, y=1.00,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)

    # Also save raw numbers.
    np.savez(out.with_suffix(".npz"),
             Z_imp=Z_imp, Z_pca=Z_pca, G=G, D=D, Y=Y, splits=splits,
             r2_raw=r2_raw, r2_pca=r2_pca,
             multi_raw=multi_raw, multi_pca_full=multi_pca_full)
    print(f"[disent] saved {out}")
    print(f"[disent] saved {out.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
