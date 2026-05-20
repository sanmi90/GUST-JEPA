"""Generate the Figure 3 equivalent for the Fukami AE baseline.

Uses the same 3x3 raw / decoded / residual layout as
``session9_decoder_figures.py`` with per-row p95 saturation, but
encodes + decodes via the Fukami CNN AE (`src/baselines/fukami_ae.py`)
instead of the frozen JEPA encoder + separate decoder. The Test B
encounter and frames match the JEPA figure for direct comparison.

Usage:
    python scripts/session9_fukami_figure.py \\
        --fukami-checkpoint outputs/runs/session9/run_a11_fukami_ae/checkpoint_iter020000.pt \\
        --output-dir outputs/runs/session9/run_a11_fukami_ae \\
        --gpu 0
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
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.session9_decoder_figures import gather_encounters  # noqa: E402
from scripts.session9_fukami_evaluation import load_fukami  # noqa: E402
from src.utils.device import require_rtx6000  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fukami AE Figure 3 equivalent")
    p.add_argument("--fukami-checkpoint", required=True, type=str)
    p.add_argument("--output-dir", required=True, type=str)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--fig-test-b-idx", type=int, default=0,
                   help="Which Test B encounter to feature (default 0; matches JEPA Figure 3).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = require_rtx6000(gpu_index=args.gpu)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wrapper = load_fukami(Path(args.fukami_checkpoint), device)
    print(f"[fukami-fig] wrapper loaded ({sum(p.numel() for p in wrapper.parameters()):,} params)",
          flush=True)

    encs_b = gather_encounters("test_b")
    e = encs_b[args.fig_test_b_idx]
    print(f"[fukami-fig] encounter: {e['case_id']} k={e['k']:02d} "
          f"G={e['G']:.2f} D={e['D']:.2f} Y={e['Y']:.2f}", flush=True)

    with h5py.File(e["path"], "r") as f:
        omega = np.asarray(f["omega_z"], dtype=np.float32)

    x = torch.from_numpy(omega).unsqueeze(0).unsqueeze(2).to(device)  # (1, T, 1, H, W)
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            z = wrapper.encode(x)
            x_hat = wrapper.decode(z)
    x_hat = x_hat.float().squeeze(0).squeeze(1).cpu().numpy()  # (T, H, W)
    residual = omega - x_hat

    frames = [25, 40, 55]
    fig, axes = plt.subplots(3, 4, figsize=(14, 11),
                             gridspec_kw={"width_ratios": [1, 1, 1, 0.06]})

    p95_raw = float(np.percentile(np.abs(omega[frames]), 95))
    p95_dec = float(np.percentile(np.abs(x_hat[frames]), 95))
    p95_res = float(np.percentile(np.abs(residual[frames]), 95))
    max_raw = float(np.abs(omega[frames]).max())
    max_dec = float(np.abs(x_hat[frames]).max())
    max_res = float(np.abs(residual[frames]).max())
    row_specs = [
        ("raw $\\omega_z$", omega, p95_raw, max_raw),
        ("Fukami decoded $\\hat\\omega_z$", x_hat, p95_dec, max_dec),
        ("residual $\\omega_z - \\hat\\omega_z$", residual, p95_res, max_res),
    ]
    for row_idx, (row_label, data, vmax, mx) in enumerate(row_specs):
        im = None
        for col, t in enumerate(frames):
            im = axes[row_idx, col].imshow(
                data[t].T, origin="lower", cmap="RdBu_r",
                vmin=-vmax, vmax=vmax,
            )
            axes[row_idx, col].set_xticks([])
            axes[row_idx, col].set_yticks([])
            if row_idx == 0:
                axes[row_idx, col].set_title(
                    f"frame {t} ({'pre-impact' if t < 35 else 'impact' if t < 50 else 'post-impact'})"
                )
        cbar = fig.colorbar(im, cax=axes[row_idx, 3], extend="both")
        cbar.set_label(f"{row_label}\np95={vmax:.1f}, max={mx:.1f}", fontsize=9)
        axes[row_idx, 0].set_ylabel(row_label, fontsize=11)

    fig.suptitle(
        f"Figure 3 (Fukami AE). Reconstruction on Test B case {e['case_id']} "
        f"encounter {e['k']:02d}\n"
        f"G={e['G']:.2f}, D={e['D']:.2f}, Y={e['Y']:.2f}; Fukami CNN AE + lift "
        f"head jointly trained 20k iters",
        y=1.02, fontsize=11,
    )
    fig.tight_layout()
    fig_path = out_dir / "fig3_fukami_reconstruction.png"
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fukami-fig] saved {fig_path}", flush=True)


if __name__ == "__main__":
    main()
