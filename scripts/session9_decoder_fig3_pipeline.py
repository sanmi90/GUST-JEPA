"""Pipeline-aware Figure 3 generator for the JEPA decoder.

Loads JEPA encoder + decoder, applies OmegaPipeline (mask + per-encounter
clip + 3-sigma normalize), encodes -> decodes -> unnormalizes, and emits a
3x3 (raw / decoded / residual) figure at frames 25 / 40 / 55 for one
Test B encounter. Fixed colorbar +/-3 to match the Fukami figure style.

Usage:
    python scripts/session9_decoder_fig3_pipeline.py \\
        --jepa-checkpoint outputs/runs/session9/run_jepa_pipeline_lam0p01_seed42/checkpoint_iter020000.pt \\
        --decoder-checkpoint outputs/runs/session9/decoder_pipeline_charb/decoder_iter010000.pt \\
        --output-dir outputs/runs/session9/decoder_pipeline_charb --gpu 0
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")

import h5py
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.omega_pipeline import OmegaPipeline  # noqa: E402
from src.models.coord_mlp_decoder import CoordMLPDecoder  # noqa: E402
from src.models.decoder import HybridViTConvDecoder  # noqa: E402
from src.models.encoder import HybridCNNViTEncoder  # noqa: E402
from src.models.lap_film_decoder import LapFiLMDecoder  # noqa: E402
from src.utils.device import require_rtx6000  # noqa: E402
from scripts.session9_decoder_figures import gather_encounters  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pipeline-aware Figure 3 for JEPA decoder")
    enc_group = p.add_mutually_exclusive_group(required=True)
    enc_group.add_argument("--jepa-checkpoint", type=str,
                           help="Path to a JEPA checkpoint .pt file.")
    enc_group.add_argument("--encoder-run", type=str,
                           help="JEPA run directory; uses the largest-iter checkpoint.")
    p.add_argument("--decoder-checkpoint", required=True, type=str)
    p.add_argument("--decoder-type", type=str, default=None,
                   choices=["fukami", "lapfilm", "coord_mlp"],
                   help="Decoder architecture. If omitted, inferred from the "
                        "checkpoint's saved args.")
    p.add_argument("--output-dir", required=True, type=str)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--fig-test-b-idx", type=int, default=0)
    p.add_argument("--label", type=str, default=None,
                   help="Optional human-readable name for the title (e.g. 'E1 LapFiLM').")
    return p.parse_args()


def resolve_encoder_ckpt(jepa_ckpt: str | None, encoder_run: str | None) -> Path:
    if jepa_ckpt is not None:
        return Path(jepa_ckpt).resolve()
    run_dir = Path(encoder_run).resolve()
    candidates = sorted(run_dir.glob("checkpoint_iter*.pt"))
    if not candidates:
        raise FileNotFoundError(f"no checkpoint_iter*.pt under {run_dir}")
    return candidates[-1]


def build_decoder_from_ckpt(
    ckpt_path: Path, d: int, device: torch.device, override_type: str | None = None
) -> tuple[nn.Module, str]:
    """Build a decoder matching the checkpoint and return (model, decoder_type).

    The decoder type is inferred from the ``decoder_type`` key in the
    saved args (Session 10), with a fallback to ``fukami`` for Session 9
    checkpoints that pre-date the type flag. ``override_type`` forces
    the type explicitly when set.
    """
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_args = blob.get("args", {})
    decoder_type = override_type or saved_args.get("decoder_type", "fukami")

    if decoder_type == "fukami":
        dec = HybridViTConvDecoder(latent_dim=d)
    elif decoder_type == "lapfilm":
        bc = int(saved_args.get("decoder_base_ch", 64))
        channels = (bc, bc, int(bc * 0.75), int(bc * 0.5), int(bc * 0.375))
        dec = LapFiLMDecoder(
            latent_dim=d,
            channels=channels,
            resblocks_per_level=int(saved_args.get("decoder_resblocks_per_level", 2)),
            upsample=saved_args.get("decoder_upsample", "pixelshuffle"),
            fourier_bands=int(saved_args.get("decoder_fourier_bands", 4) or 4),
            use_film=bool(saved_args.get("decoder_use_film", True)),
        )
    elif decoder_type == "coord_mlp":
        dec = CoordMLPDecoder(
            latent_dim=d,
            hidden=int(saved_args.get("decoder_mlp_hidden", 128)),
            layers=int(saved_args.get("decoder_mlp_layers", 5)),
            fourier_bands=int(saved_args.get("decoder_fourier_bands", 8) or 8),
            activation=saved_args.get("decoder_mlp_activation", "sine"),
            chunk_pixels=int(saved_args.get("decoder_mlp_chunk", 4096)),
        )
    else:
        raise ValueError(f"unknown decoder_type {decoder_type!r}")
    dec.load_state_dict(blob["decoder_state_dict"])
    return dec.eval().to(device), decoder_type


def _extract_pred(dec_out) -> torch.Tensor:
    if isinstance(dec_out, dict):
        return dec_out["pred"]
    return dec_out


def load_encoder(ckpt_path: Path, device: torch.device) -> tuple[HybridCNNViTEncoder, int, OmegaPipeline | None]:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = blob["args"]
    enc = HybridCNNViTEncoder(
        latent_dim=int(args["d"]),
        projection_norm=args.get("projection_norm", "batchnorm"),
    )
    state = {
        k.removeprefix("encoder."): v
        for k, v in blob["jepa_state_dict"].items()
        if k.startswith("encoder.")
    }
    enc.load_state_dict(state, strict=False)
    enc = enc.eval().to(device)
    pipe = None
    manifest_rel = args.get("omega_pipeline_manifest")
    if manifest_rel:
        manifest = Path(manifest_rel)
        if not manifest.is_absolute():
            manifest = REPO / manifest
        pipe = OmegaPipeline.from_manifest(manifest)
    return enc, int(args["d"]), pipe


def load_decoder(ckpt_path: Path, d: int, device: torch.device) -> nn.Module:
    """Legacy entry point retained for Session 9 callers; dispatches to the
    type recorded in the checkpoint (defaults to fukami)."""
    dec, _ = build_decoder_from_ckpt(ckpt_path, d, device, override_type=None)
    return dec


def load_airfoil_xy() -> np.ndarray:
    raw = Path(os.environ.get("PREVENT_ROOT", "/home/carlos/PREVENT")) / "data" / "raw" / "periodic" / "Baseline.h5"
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


def main() -> None:
    args = parse_args()
    device = require_rtx6000(gpu_index=args.gpu)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    enc_ckpt = resolve_encoder_ckpt(args.jepa_checkpoint, args.encoder_run)
    enc, d, pipe = load_encoder(enc_ckpt, device)
    dec, decoder_type = build_decoder_from_ckpt(
        Path(args.decoder_checkpoint), d, device, override_type=args.decoder_type,
    )
    print(f"[decoder-fig] encoder d={d}, decoder_type={decoder_type}, "
          f"pipeline={'yes' if pipe else 'no'}", flush=True)

    airfoil_xy = load_airfoil_xy()
    airfoil_px = _omega_to_pixel(airfoil_xy)

    encs_b = gather_encounters("test_b")
    e = encs_b[args.fig_test_b_idx]
    print(f"[decoder-fig] encounter: {e['case_id']} k={e['k']:02d}", flush=True)

    with h5py.File(e["path"], "r") as f:
        omega = np.asarray(f["omega_z"], dtype=np.float32)
    if pipe is not None:
        omega = pipe.preprocess_raw(omega, e["case_id"], int(e["k"]))

    x = torch.from_numpy(omega).unsqueeze(0).unsqueeze(2).to(device)
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            x_in = pipe.normalize(x) if pipe is not None else x
            z = enc(x_in)
            dec_out = dec(z)
            x_hat = _extract_pred(dec_out)
            if pipe is not None:
                x_hat = pipe.unnormalize(x_hat)
    x_hat = x_hat.float().squeeze(0).squeeze(1).cpu().numpy()
    residual = omega - x_hat

    frames = [25, 40, 55]
    fig, axes = plt.subplots(3, 4, figsize=(14, 11),
                             gridspec_kw={"width_ratios": [1, 1, 1, 0.06]})
    vlim = 3.0
    max_raw = float(np.abs(omega[frames]).max())
    max_dec = float(np.abs(x_hat[frames]).max())
    max_res = float(np.abs(residual[frames]).max())
    row_specs = [
        ("raw $\\omega_z$", omega, vlim, max_raw),
        ("JEPA decoded $\\hat\\omega_z$", x_hat, vlim, max_dec),
        ("residual $\\omega_z - \\hat\\omega_z$", residual, vlim, max_res),
    ]
    for row_idx, (row_label, data, vmax, mx) in enumerate(row_specs):
        im = None
        for col, t in enumerate(frames):
            ax = axes[row_idx, col]
            im = ax.imshow(
                data[t].T, origin="lower", cmap="RdBu_r",
                vmin=-vmax, vmax=vmax,
            )
            ax.add_patch(Polygon(airfoil_px, closed=True, facecolor="black",
                                 edgecolor="black", linewidth=0.7, zorder=10))
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(
                    f"frame {t} ({'pre-impact' if t < 35 else 'impact' if t < 50 else 'post-impact'})"
                )
        cbar = fig.colorbar(im, cax=axes[row_idx, 3], extend="both")
        cbar.set_label(f"{row_label}\nvlim=$\\pm${vmax:.1f}, max={mx:.1f}", fontsize=9)
        axes[row_idx, 0].set_ylabel(row_label, fontsize=11)

    label = args.label or f"JEPA d={d} + {decoder_type}"
    fig.suptitle(
        f"Figure 3 ({label}). Reconstruction on Test B case {e['case_id']} "
        f"encounter {e['k']:02d}\n"
        f"G={e['G']:.2f}, D={e['D']:.2f}, Y={e['Y']:.2f}; "
        f"JEPA d={d} frozen encoder + {decoder_type} visualisation decoder",
        y=1.02, fontsize=11,
    )
    fig.tight_layout()
    fig_path = out_dir / "fig3_jepa_reconstruction.png"
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[decoder-fig] saved {fig_path}", flush=True)


if __name__ == "__main__":
    main()
