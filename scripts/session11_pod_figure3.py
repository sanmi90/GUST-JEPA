"""Generate the canonical Test B Figure 3 for the POD d=32 baseline.

Same layout as scripts/session9_decoder_fig3_pipeline.py (3 rows: raw,
decoded, residual; 3 cols: frames 25, 40, 55; +/- 3-sigma colorbar;
NACA airfoil overlay). Lets us put the POD reconstruction side by side
with the JEPA + wake-head reconstruction.
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
from matplotlib.patches import Polygon

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.omega_pipeline import OmegaPipeline  # noqa: E402


_PHYSICAL_X = (-1.5, 4.5)
_PHYSICAL_Y = (-1.5, 1.5)
PREVENT = Path(os.environ.get("PREVENT_ROOT", "/home/carlos/PREVENT"))
CACHE = Path(os.environ.get("VORTEX_JEPA_CACHE", PREVENT / "data" / "processed" / "vortex-jepa"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Figure 3 for POD baseline")
    p.add_argument("--pod-basis", required=True, type=str,
                   help="Path to pod_basis.npz produced by session11_pod_baseline.py")
    p.add_argument("--output", required=True, type=str)
    p.add_argument(
        "--omega-pipeline-manifest", type=str,
        default="outputs/data_pipeline/v1/manifest.json",
    )
    p.add_argument("--partition", type=str, default="v1")
    p.add_argument("--encounter-idx", type=int, default=0,
                   help="Index into test_b encounter list (0 = canonical G+1.00_D1.00_Y+0.10 enc 0)")
    p.add_argument("--frames", type=int, nargs="+", default=[25, 40, 55])
    p.add_argument("--label", type=str, default="POD d=32")
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


def _airfoil_polygon():
    """Load the NACA 0012 polygon vertices in pixel coordinates."""
    base = PREVENT / "data" / "raw" / "periodic" / "Baseline.h5"
    if not base.exists():
        return None
    with h5py.File(base, "r") as f:
        if "airfoil_xy" not in f:
            return None
        xy = np.asarray(f["airfoil_xy"], dtype=np.float32)
    # xy is in physical coords (alpha=14 already applied to mesh). Convert to
    # pixel: pixel_x = (phys - x_min) / (x_max - x_min) * (192 - 1)
    px = (xy[:, 0] - _PHYSICAL_X[0]) / (_PHYSICAL_X[1] - _PHYSICAL_X[0]) * 191
    py = (xy[:, 1] - _PHYSICAL_Y[0]) / (_PHYSICAL_Y[1] - _PHYSICAL_Y[0]) * 95
    # Need (col, row) for imshow with origin='lower'? The original
    # session9_decoder_fig3_pipeline draws omega.T so columns = x. Match it.
    return np.stack([px, py], axis=1)


def main() -> None:
    args = parse_args()
    bundle = np.load(args.pod_basis)
    Phi = bundle["Phi"]  # (H*W, d)
    mean = bundle["mean"]  # (H*W,)
    d = int(bundle["d"])

    manifest_path = Path(args.omega_pipeline_manifest)
    if not manifest_path.is_absolute():
        manifest_path = REPO / manifest_path
    pipe = OmegaPipeline.from_manifest(manifest_path)

    encs = gather_test_b(args.partition)
    e = encs[args.encounter_idx]
    print(f"[fig3-pod] encounter: {e['case_id']} k={e['k']:02d}; "
          f"d={d}; energy={float(bundle['energy_fraction']):.4f}")

    H, W = 192, 96
    with h5py.File(e["path"], "r") as f:
        omega_raw = np.asarray(f["omega_z"], dtype=np.float32)
    omega_clean = pipe.preprocess_raw(omega_raw, e["case_id"], int(e["k"]))
    import torch
    omega_norm = pipe.normalize(torch.from_numpy(omega_clean)).numpy()
    T = omega_norm.shape[0]
    flat = omega_norm.reshape(T, H * W)
    coeffs = (flat - mean[None]) @ Phi
    recon_norm_flat = mean[None] + coeffs @ Phi.T
    recon_norm = recon_norm_flat.reshape(T, H, W)
    recon_raw = pipe.unnormalize(torch.from_numpy(recon_norm)).numpy()
    # Match scripts/session9_decoder_fig3_pipeline.py: plot raw omega
    # directly with vmin/vmax = +/-3 (vorticity units in 1/t_c; Fukami's
    # published Figure 3 range). Most of the field sits inside [-3, +3];
    # the gust core saturates the colorbar by design.
    target = omega_clean
    pred = recon_raw
    resid = target - pred

    # Read airfoil polygon vertices in physical coords for overlay.
    base = PREVENT / "data" / "raw" / "periodic" / "Baseline.h5"
    airfoil_phys = None
    if base.exists():
        with h5py.File(base, "r") as bf:
            if "airfoil_xy" in bf:
                airfoil_phys = np.asarray(bf["airfoil_xy"], dtype=np.float32)
                # Convert to pixel coords matching omega.T with origin='lower'
                # (rows = y, cols = x); image extent uses (x_min, x_max, y_min, y_max).
                px = (airfoil_phys[:, 0] - _PHYSICAL_X[0]) \
                    / (_PHYSICAL_X[1] - _PHYSICAL_X[0]) * 191
                py = (airfoil_phys[:, 1] - _PHYSICAL_Y[0]) \
                    / (_PHYSICAL_Y[1] - _PHYSICAL_Y[0]) * 95
                airfoil_px = np.stack([px, py], axis=1)
            else:
                airfoil_px = None
    else:
        airfoil_px = None

    fig, axes = plt.subplots(3, len(args.frames) + 1, figsize=(4 * len(args.frames) + 0.5, 9),
                             gridspec_kw={"width_ratios": [1] * len(args.frames) + [0.06]})
    vlim = 3.0
    row_specs = [
        (r"raw $\omega_z$", target),
        (f"POD d={d} decoded $\\hat\\omega_z$", pred),
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
        f"Figure 3 ({args.label}). Reconstruction on Test B case "
        f"{e['case_id']} encounter {e['k']:02d}\n"
        f"POD d={d}, energy fraction={float(bundle['energy_fraction']):.3f}",
        y=1.02, fontsize=11,
    )
    plt.tight_layout()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig3-pod] saved {out_path}")


if __name__ == "__main__":
    main()
