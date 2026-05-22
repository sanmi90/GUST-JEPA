"""Figure 3 for the PCA-truncated JEPA decoder (latent_dim=k, k<d).

Loads a PCA basis (mean, P) saved alongside a LapFiLM decoder that was
trained with `latent_dim=k`, encodes a Test B encounter through the
frozen JEPA encoder, projects z to k dims, decodes, and renders the
canonical 3 x 3 panel (raw / decoded / residual; pre-impact / impact /
post-impact) at +/- 3 sigma colorbar with the NACA airfoil overlaid.

Drop-in companion of:
- scripts/session9_decoder_fig3_pipeline.py (full-d=32 JEPA decoder)
- scripts/session11_pod_figure3.py (linear POD baseline)

Usage::

    python scripts/session11_pca_figure3.py \\
        --encoder-run outputs/runs/session11/W0_C_lam100 \\
        --decoder-checkpoint outputs/runs/session11/W0_C_lam100/decoder_pca_k12/decoder_iter020000.pt \\
        --pca-basis        outputs/runs/session11/W0_C_lam100/decoder_pca_k12/pca_basis.npz \\
        --output outputs/runs/session11/W0_C_lam100/decoder_pca_k12/figure3.png
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
from matplotlib.patches import Polygon

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.omega_pipeline import OmegaPipeline  # noqa: E402
from src.models.encoder import HybridCNNViTEncoder  # noqa: E402
from src.models.lap_film_decoder import LapFiLMDecoder  # noqa: E402
from src.utils.device import require_rtx6000  # noqa: E402


_PHYSICAL_X = (-1.5, 4.5)
_PHYSICAL_Y = (-1.5, 1.5)
PREVENT = Path(os.environ.get("PREVENT_ROOT", "/home/carlos/PREVENT"))
CACHE = Path(os.environ.get("VORTEX_JEPA_CACHE", PREVENT / "data" / "processed" / "vortex-jepa"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Figure 3 for PCA-truncated JEPA decoder")
    p.add_argument("--encoder-run", required=True, type=str)
    p.add_argument("--decoder-checkpoint", required=True, type=str)
    p.add_argument("--pca-basis", required=True, type=str,
                   help="pca_basis.npz produced by session11_pca_decoder.py "
                        "(mean shape (d,), P shape (d, k))")
    p.add_argument("--output", required=True, type=str)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--omega-pipeline-manifest", type=str,
                   default="outputs/data_pipeline/v1/manifest.json")
    p.add_argument("--partition", type=str, default="v1")
    p.add_argument("--encounter-idx", type=int, default=0)
    p.add_argument("--frames", type=int, nargs="+", default=[25, 40, 55])
    p.add_argument("--label", type=str, default=None)
    return p.parse_args()


def gather_test_b(partition: str) -> list[dict]:
    with open(REPO / "configs" / "splits" / f"split_{partition}.json") as f:
        m = json.load(f)
    out = []
    for cid, c in m["cases"].items():
        if c["split"] != "test_b":
            continue
        for k in range(c["n_encounters_full"]):
            path = CACHE / partition / cid / f"encounter_{k:02d}.h5"
            if path.exists():
                out.append({"case_id": cid, "k": int(k), "path": str(path)})
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

    basis = np.load(args.pca_basis)
    mean = torch.from_numpy(basis["mean"]).float().to(device)
    P = torch.from_numpy(basis["P"]).float().to(device)
    k = int(basis["k"])
    # Build decoder with latent_dim=k.
    ck = torch.load(args.decoder_checkpoint, map_location="cpu", weights_only=False)
    da = ck["args"]
    bc = int(da.get("decoder_base_ch", 64))
    channels = (bc, bc, int(bc * 0.75), int(bc * 0.5), int(bc * 0.375))
    dec = LapFiLMDecoder(
        latent_dim=k,
        channels=channels,
        upsample="pixelshuffle",
        fourier_bands=4,
        use_film=True,
    ).to(device).eval()
    dec.load_state_dict(ck["decoder_state_dict"], strict=True)
    print(f"[fig3-pca] encoder d={d}; PCA k={k}; "
          f"decoder params={sum(p.numel() for p in dec.parameters()):,}")

    encs = gather_test_b(args.partition)
    e = encs[args.encounter_idx]
    print(f"[fig3-pca] encounter: {e['case_id']} k={e['k']:02d}")

    H, W = 192, 96
    with h5py.File(e["path"], "r") as f:
        omega_raw = np.asarray(f["omega_z"], dtype=np.float32)
    omega_clean = pipe.preprocess_raw(omega_raw, e["case_id"], int(e["k"]))
    x_norm = pipe.normalize(torch.from_numpy(omega_clean)).to(device)  # (T,H,W)
    x = x_norm.unsqueeze(0).unsqueeze(2)  # (1,T,1,H,W)
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        z = enc(x).float()  # (1,T,d)
        z_proj = (z - mean) @ P  # (1,T,k)
        pred_norm = dec(z_proj)["pred"].float().squeeze(0).squeeze(1)  # (T,H,W)
        pred_raw = pipe.unnormalize(pred_norm.unsqueeze(0)).squeeze(0).cpu().numpy()

    target = omega_clean
    pred = pred_raw
    resid = target - pred

    # Airfoil overlay
    base = PREVENT / "data" / "raw" / "periodic" / "Baseline.h5"
    airfoil_px = None
    if base.exists():
        with h5py.File(base, "r") as bf:
            if "airfoil_xy" in bf:
                xy = np.asarray(bf["airfoil_xy"], dtype=np.float32)
                px = (xy[:, 0] - _PHYSICAL_X[0]) / (_PHYSICAL_X[1] - _PHYSICAL_X[0]) * 191
                py = (xy[:, 1] - _PHYSICAL_Y[0]) / (_PHYSICAL_Y[1] - _PHYSICAL_Y[0]) * 95
                airfoil_px = np.stack([px, py], axis=1)

    fig, axes = plt.subplots(3, len(args.frames) + 1, figsize=(4 * len(args.frames) + 0.5, 9),
                             gridspec_kw={"width_ratios": [1] * len(args.frames) + [0.06]})
    vlim = 3.0
    label = args.label or f"JEPA d={d} -> PCA k={k} + LapFiLM"
    row_specs = [
        (r"raw $\omega_z$", target),
        (f"{label} decoded $\\hat\\omega_z$", pred),
        (r"residual $\omega_z - \hat\omega_z$", resid),
    ]
    for ri, (row_label, data) in enumerate(row_specs):
        im = None
        for ci, fidx in enumerate(args.frames):
            ax = axes[ri, ci]
            im = ax.imshow(data[fidx].T, origin="lower", cmap="RdBu_r",
                           vmin=-vlim, vmax=vlim)
            if airfoil_px is not None:
                ax.add_patch(Polygon(airfoil_px, closed=True, facecolor="black",
                                     edgecolor="black", linewidth=0.7, zorder=10))
            ax.set_xticks([])
            ax.set_yticks([])
            if ri == 0:
                kind = ("pre-impact" if fidx < 35 else
                        "impact" if fidx < 50 else "post-impact")
                ax.set_title(f"frame {fidx} ({kind})")
        cbar = fig.colorbar(im, cax=axes[ri, -1], extend="both")
        mx = float(np.abs(data[args.frames]).max())
        cbar.set_label(f"{row_label}\nvlim=$\\pm$3.0, max={mx:.1f}", fontsize=9)
        axes[ri, 0].set_ylabel(row_label, fontsize=11)

    fig.suptitle(
        f"Figure 3 ({label}). Reconstruction on Test B case "
        f"{e['case_id']} encounter {e['k']:02d}",
        y=1.02, fontsize=11,
    )
    plt.tight_layout()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig3-pca] saved {out_path}")


if __name__ == "__main__":
    main()
