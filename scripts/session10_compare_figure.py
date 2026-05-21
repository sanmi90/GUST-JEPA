"""Session 10 multi-decoder comparison figure.

Loads the frozen JEPA encoder and a list of decoder checkpoints, then
emits a single figure of the canonical Test B encounter
``G+1.00_D1.00_Y+0.10`` encounter 00 at frames 25, 40, 55. Each row is a
frame; each column is target / Session 9 baseline / E1 / E2 / E4 (and
optionally E_noFiLM). Fixed colorbar +/-3 in normalised omega units;
airfoil overlay matches the Session 9 figure style.

Usage::

    python -m scripts.session10_compare_figure \\
        --encoder-run outputs/runs/session9/run_jepa_pipeline_lam0p01_seed42 \\
        --decoder-run baseline outputs/runs/session9/decoder_pipeline_mse \\
        --decoder-run "E1 LapFiLM"     outputs/runs/session10/E1_jepa_lapfilm_pyr_noffl \\
        --decoder-run "E2 LapFiLM+FFL" outputs/runs/session10/E2_jepa_lapfilm_pyr_ffl \\
        --decoder-run "E4 CoordMLP"    outputs/runs/session10/E4_jepa_coordmlp_audit \\
        --output outputs/runs/session10/figure3_compare.png \\
        --gpu 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.utils.device import require_rtx6000  # noqa: E402
from scripts.session9_decoder_fig3_pipeline import (  # noqa: E402
    build_decoder_from_ckpt,
    load_encoder,
    resolve_encoder_ckpt,
    _extract_pred,
)


PREVENT = Path(os.environ.get("PREVENT_ROOT", "/home/carlos/PREVENT"))
CACHE = Path(os.environ.get("VORTEX_JEPA_CACHE", PREVENT / "data" / "processed" / "vortex-jepa"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Session 10 decoder comparison figure")
    enc_group = p.add_mutually_exclusive_group(required=True)
    enc_group.add_argument("--jepa-checkpoint", type=str)
    enc_group.add_argument("--encoder-run", type=str)
    p.add_argument("--decoder-run", action="append", nargs=2, required=True,
                   metavar=("LABEL", "PATH"),
                   help="Decoder run dir or .pt checkpoint with a human-readable "
                        "label. May be specified multiple times.")
    p.add_argument("--target-case", default="G+1.00_D1.00_Y+0.10",
                   help="Canonical Test B case for the figure.")
    p.add_argument("--target-encounter", type=int, default=0)
    p.add_argument("--frames", nargs=3, type=int, default=[25, 40, 55])
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output", required=True, type=str)
    p.add_argument("--show-residuals", action="store_true",
                   help="Add a residual row below the predictions row.")
    return p.parse_args()


def load_airfoil_xy() -> np.ndarray:
    raw = PREVENT / "data" / "raw" / "periodic" / "Baseline.h5"
    with h5py.File(raw, "r") as f:
        xy = np.asarray(f["airfoil_xy"])
    if not np.allclose(xy[0], xy[-1]):
        xy = np.vstack([xy, xy[0:1]])
    return xy


def _omega_to_pixel(xy: np.ndarray, H: int = 192, W: int = 96) -> np.ndarray:
    x_min, x_max = -1.5, 4.5
    y_min, y_max = -1.5, 1.5
    px_x = (xy[:, 0] - x_min) * (H - 1) / (x_max - x_min)
    px_y = (xy[:, 1] - y_min) * (W - 1) / (y_max - y_min)
    return np.stack([px_x, px_y], axis=-1)


def _resolve_decoder_ckpt(path: str) -> Path:
    p = Path(path).resolve()
    if p.is_file():
        return p
    candidates = sorted(p.glob("decoder_iter*.pt"))
    if not candidates:
        raise FileNotFoundError(f"no decoder_iter*.pt under {p}")
    return candidates[-1]


def main() -> None:
    args = parse_args()
    device = require_rtx6000(gpu_index=args.gpu)

    encoder_ckpt = resolve_encoder_ckpt(args.jepa_checkpoint, args.encoder_run)
    enc, d, pipe = load_encoder(encoder_ckpt, device)
    print(f"[fig-compare] encoder d={d}, pipeline={'yes' if pipe is not None else 'no'}", flush=True)

    # Load the canonical encounter
    path = CACHE / "v1" / args.target_case / f"encounter_{args.target_encounter:02d}.h5"
    if not path.exists():
        raise FileNotFoundError(f"target case file missing: {path}")
    with h5py.File(path, "r") as f:
        omega = np.asarray(f["omega_z"], dtype=np.float32)
    if pipe is not None:
        omega_proc = pipe.preprocess_raw(omega, args.target_case, args.target_encounter)
    else:
        omega_proc = omega

    x = torch.from_numpy(omega_proc).unsqueeze(0).unsqueeze(2).to(device)
    x_in = pipe.normalize(x) if pipe is not None else x

    # Encode once
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            z = enc(x_in)

    # Run each decoder
    predictions = []
    labels = []
    for label, run_path in args.decoder_run:
        ckpt = _resolve_decoder_ckpt(run_path)
        dec, dec_type = build_decoder_from_ckpt(ckpt, d, device)
        print(f"[fig-compare] {label}: {ckpt.name} ({dec_type})", flush=True)
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                dec_out = dec(z)
                x_hat = _extract_pred(dec_out)
                if pipe is not None:
                    x_hat = pipe.unnormalize(x_hat)
        x_hat = x_hat.float().squeeze(0).squeeze(1).cpu().numpy()
        predictions.append(x_hat)
        labels.append(label)

    airfoil_xy = load_airfoil_xy()
    airfoil_px = _omega_to_pixel(airfoil_xy)

    # Build figure: rows = frames (+ residual row optionally); cols = target + decoders.
    n_decoders = len(predictions)
    n_cols = 1 + n_decoders
    n_rows = len(args.frames) * (2 if args.show_residuals else 1)
    fig, axes = plt.subplots(n_rows, n_cols + 1, figsize=(3.0 * (n_cols + 0.5), 3.0 * n_rows),
                              gridspec_kw={"width_ratios": [1] * n_cols + [0.05]})
    if n_rows == 1:
        axes = axes[None, :]
    vlim = 3.0 if pipe is None else (3.0 * pipe.train_stats.std)

    row_idx = 0
    for t in args.frames:
        # Predictions row
        ax = axes[row_idx, 0]
        im = ax.imshow(omega[t].T, origin="lower", cmap="RdBu_r", vmin=-vlim, vmax=vlim)
        ax.add_patch(Polygon(airfoil_px, closed=True, facecolor="black",
                             edgecolor="black", linewidth=0.7, zorder=10))
        ax.set_xticks([]); ax.set_yticks([])
        if t == args.frames[0]:
            ax.set_title("target")
        ax.set_ylabel(f"frame {t}", fontsize=10)
        for j, (label, x_hat) in enumerate(zip(labels, predictions)):
            ax = axes[row_idx, j + 1]
            ax.imshow(x_hat[t].T, origin="lower", cmap="RdBu_r", vmin=-vlim, vmax=vlim)
            ax.add_patch(Polygon(airfoil_px, closed=True, facecolor="black",
                                 edgecolor="black", linewidth=0.7, zorder=10))
            ax.set_xticks([]); ax.set_yticks([])
            if t == args.frames[0]:
                ax.set_title(label)
        cbar = fig.colorbar(im, cax=axes[row_idx, -1], extend="both")
        cbar.set_label(f"$\\omega_z$ vlim=$\\pm${vlim:.1f}", fontsize=8)
        row_idx += 1

        if args.show_residuals:
            ax = axes[row_idx, 0]
            ax.text(0.5, 0.5, "residual", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            for j, (label, x_hat) in enumerate(zip(labels, predictions)):
                resid = omega[t] - x_hat[t]
                ax = axes[row_idx, j + 1]
                ax.imshow(resid.T, origin="lower", cmap="RdBu_r", vmin=-vlim, vmax=vlim)
                ax.add_patch(Polygon(airfoil_px, closed=True, facecolor="black",
                                     edgecolor="black", linewidth=0.7, zorder=10))
                ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, cax=axes[row_idx, -1], extend="both")
            row_idx += 1

    fig.suptitle(
        f"Session 10 Figure 3 -- Test B case {args.target_case} encounter "
        f"{args.target_encounter:02d}, JEPA d={d} frozen encoder",
        y=1.01, fontsize=11,
    )
    fig.tight_layout()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig-compare] saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
