"""3D trajectories of the latent across an encounter, coloured by G and D.

Each encounter is a 120-frame sequence z_t. We project z_t to the
PCA basis saved by session11_pca_decoder.py and plot the curve in
(PC_i, PC_j, PC_k) space. One curve per encounter; line colour
encodes the physical parameter (G or D), and the impact frame is
marked with a diamond.

To avoid spaghetti, by default we plot one encounter per unique
(G, D, Y) case (~60 curves). Pass --all to show all 282.

Usage::

    python scripts/session11_latent_trajectories.py \\
        --encoder-run outputs/runs/session11/W0_C_lam100 \\
        --pca-basis  outputs/runs/session11/W0_C_lam100/decoder_pca_k12/pca_basis.npz \\
        --output     outputs/runs/session11/W0_C_lam100/decoder_pca_k12/latent3d_trajectories.png
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

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
    p.add_argument("--components", type=int, nargs=3, default=[0, 1, 2])
    p.add_argument("--elev", type=float, default=22.0)
    p.add_argument("--azim", type=float, default=-65.0)
    p.add_argument("--all", action="store_true",
                   help="Plot all 282 encounters (default: 1 per (G,D,Y) case).")
    return p.parse_args()


def gather_one_per_case(partition: str) -> list[dict]:
    with open(REPO / "configs" / "splits" / f"split_{partition}.json") as f:
        m = json.load(f)
    out = []
    for cid, c in m["cases"].items():
        if c["split"] == "train":
            ks = c.get("train_encounter_indices", [])
        elif c["split"] in ("test_b", "test_c"):
            ks = list(range(c["n_encounters_full"]))
        else:
            continue
        if not ks:
            continue
        k = int(ks[0])
        p_ = CACHE / partition / cid / f"encounter_{k:02d}.h5"
        if p_.exists():
            out.append(dict(case_id=cid, k=k, G=c["G"], D=c["D"], Y=c["Y"],
                            split=c["split"], path=str(p_)))
    return out


def gather_all(partition: str) -> list[dict]:
    with open(REPO / "configs" / "splits" / f"split_{partition}.json") as f:
        m = json.load(f)
    out = []
    for cid, c in m["cases"].items():
        if c["split"] == "train":
            ks = c.get("train_encounter_indices", [])
        elif c["split"] in ("test_b", "test_c"):
            ks = list(range(c["n_encounters_full"]))
        else:
            continue
        for k in ks:
            p_ = CACHE / partition / cid / f"encounter_{k:02d}.h5"
            if p_.exists():
                out.append(dict(case_id=cid, k=int(k), G=c["G"], D=c["D"], Y=c["Y"],
                                split=c["split"], path=str(p_)))
    return out


def load_encoder(encoder_run: Path, device: torch.device) -> tuple[HybridCNNViTEncoder, int]:
    cands = sorted(encoder_run.glob("checkpoint_iter*.pt"))
    ckpt = torch.load(cands[-1], map_location="cpu", weights_only=False)
    a = ckpt["args"]
    enc = HybridCNNViTEncoder(
        latent_dim=int(a["d"]),
        projection_norm=a.get("projection_norm", "batchnorm"),
    )
    state = {
        k.removeprefix("encoder."): v
        for k, v in ckpt["jepa_state_dict"].items()
        if k.startswith("encoder.")
    }
    enc.load_state_dict(state, strict=False)
    enc.eval().to(device)
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc, int(a["d"])


def main() -> None:
    args = parse_args()
    device = require_rtx6000(gpu_index=args.gpu)
    manifest = Path(args.omega_pipeline_manifest)
    if not manifest.is_absolute():
        manifest = REPO / manifest
    pipe = OmegaPipeline.from_manifest(manifest)
    enc, d = load_encoder(Path(args.encoder_run), device)

    encs = gather_all(args.partition) if args.all else gather_one_per_case(args.partition)
    print(f"[traj] {len(encs)} encounters")

    basis = np.load(args.pca_basis)
    mean = basis["mean"].astype(np.float32)
    P = basis["P"].astype(np.float32)
    k = int(basis["k"])

    trajs = []
    Gs = []; Ds = []; Ys = []; sps = []
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
        zp = (z - mean[None]) @ P  # (T, k)
        trajs.append(zp)
        Gs.append(e["G"]); Ds.append(e["D"]); Ys.append(e["Y"]); sps.append(e["split"])
        if (i + 1) % 20 == 0:
            print(f"[traj] encoded {i+1}/{len(encs)}")
    Gs = np.asarray(Gs); Ds = np.asarray(Ds); Ys = np.asarray(Ys); sps = np.asarray(sps)
    trajs = np.stack(trajs, axis=0)  # (N, T, k)
    print(f"[traj] traj shape={trajs.shape}")

    cx, cy, cz = args.components
    fig = plt.figure(figsize=(15, 7))
    axG = fig.add_subplot(1, 2, 1, projection="3d")
    axD = fig.add_subplot(1, 2, 2, projection="3d")
    fi = args.impact_frame

    def _plot(ax, vals, label, cmap):
        vmin, vmax = float(vals.min()), float(vals.max())
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        for i in range(trajs.shape[0]):
            c = sm.to_rgba(vals[i])
            xs = trajs[i, :, cx]; ys = trajs[i, :, cy]; zs = trajs[i, :, cz]
            # background trajectory line
            ax.plot(xs, ys, zs, color=c, linewidth=0.9, alpha=0.55)
            # impact marker
            ax.scatter([xs[fi]], [ys[fi]], [zs[fi]], color=c,
                       marker="D", s=38, edgecolor="k", linewidth=0.4, zorder=5)
            # start marker (very small)
            ax.scatter([xs[0]], [ys[0]], [zs[0]], color=c,
                       marker="o", s=10, edgecolor="k", linewidth=0.3, alpha=0.6)
        ax.set_xlabel(f"PC{cx+1}")
        ax.set_ylabel(f"PC{cy+1}")
        ax.set_zlabel(f"PC{cz+1}")
        ax.view_init(elev=args.elev, azim=args.azim)
        ax.set_title(f"trajectories coloured by {label}")
        cb = fig.colorbar(sm, ax=ax, shrink=0.65, pad=0.10)
        cb.set_label(label)

    _plot(axG, Gs, "G (gust strength)", "coolwarm")
    _plot(axD, Ds, "D (gust diameter)", "viridis")

    fig.suptitle(
        f"Latent trajectories in PC{cx+1}-PC{cy+1}-PC{cz+1} for W0_C_lam100  "
        f"(impact frame {fi} marked with diamond, start with small circle)\n"
        f"{trajs.shape[0]} encounters, T={trajs.shape[1]} frames each "
        + ("(one per unique (G,D,Y) case)" if not args.all else "(all encounters)"),
        fontsize=11, y=1.00,
    )
    plt.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[traj] saved {out}")


if __name__ == "__main__":
    main()
