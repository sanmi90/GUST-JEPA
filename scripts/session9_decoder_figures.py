"""Session 9 Step 2: produce decoder reconstruction figures for Section 6.

Loads the trained decoder + the frozen JEPA encoder from Step 2
(``--decoder-checkpoint`` and the corresponding ``--jepa-checkpoint``)
and produces:

1. ``fig3_decoder_reconstruction.png``: 3x3 grid of (raw, decoded,
   residual) vorticity at frames 25 (pre-impact), 40 (impact), 55
   (post-impact) for one chosen Test B encounter (default
   first encounter of the first Test B case).
2. ``fig_decoder_mse_distribution.png``: per-encounter MSE histograms
   for Test A, Test B, Test C overlaid on a common axis.
3. ``decoder_per_encounter.csv``: per-encounter MSE table for all
   three splits.

Usage:
    python scripts/session9_decoder_figures.py \\
        --jepa-checkpoint outputs/runs/session9/run_f5_lam0p01_seed123/checkpoint_iter020000.pt \\
        --decoder-checkpoint outputs/runs/session9/decoder/decoder_iter010000.pt \\
        --output-dir outputs/runs/session9/decoder \\
        --gpu 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.models.decoder import HybridViTConvDecoder  # noqa: E402
from src.models.encoder import HybridCNNViTEncoder  # noqa: E402
from src.utils.device import require_rtx6000  # noqa: E402


PREVENT = Path(os.environ.get("PREVENT_ROOT", "/home/carlos/PREVENT"))
CACHE = Path(os.environ.get("VORTEX_JEPA_CACHE", PREVENT / "data" / "processed" / "vortex-jepa"))


def load_encoder(ckpt_path: Path, device: torch.device) -> tuple[HybridCNNViTEncoder, int]:
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
    return enc.eval().to(device), int(args["d"])


def load_decoder(ckpt_path: Path, d: int, device: torch.device) -> HybridViTConvDecoder:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    dec = HybridViTConvDecoder(latent_dim=d)
    dec.load_state_dict(blob["decoder_state_dict"])
    return dec.eval().to(device)


def gather_encounters(split: str) -> list[dict]:
    with open(REPO / "configs" / "splits" / "split_v2.json") as f:
        manifest = json.load(f)
    out = []
    for cid, case in manifest["cases"].items():
        if split == "test_a" and case["split"] == "train":
            ks = (case.get("val_encounter_indices") or case["test_a_encounter_indices"])
        elif split == "test_b" and case["split"] == "test_b":
            ks = list(range(case["n_encounters_full"]))
        elif split == "test_c" and case["split"] == "test_c":
            ks = list(range(case["n_encounters_full"]))
        else:
            continue
        for k in ks:
            path = CACHE / "v1" / cid / f"encounter_{k:02d}.h5"
            if not path.exists():
                continue
            out.append({"case_id": cid, "k": int(k), "path": str(path),
                        "G": float(case.get("G", 0)),
                        "D": float(case.get("D", 0)),
                        "Y": float(case.get("Y", 0))})
    return out


def _ssim(x: np.ndarray, y: np.ndarray, c1: float = 0.16, c2: float = 1.44) -> float:
    """Fukami's SSIM (arXiv:2305.18394 Eq. 1) on a (H, W) array pair."""
    mu_x, mu_y = x.mean(), y.mean()
    var_x, var_y = x.var(), y.var()
    cov_xy = ((x - mu_x) * (y - mu_y)).mean()
    num = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
    return float(num / max(den, 1e-12))


def per_encounter_mse(enc, dec, encs, device) -> pd.DataFrame:
    rows = []
    case_arr = {}
    for e in encs:
        with h5py.File(e["path"], "r") as f:
            omega = np.asarray(f["omega_z"], dtype=np.float32)
        case_arr.setdefault(e["case_id"], []).append(omega)
    case_mean = {cid: np.stack(arrs).mean(axis=0) for cid, arrs in case_arr.items()}

    with torch.no_grad():
        for e in encs:
            with h5py.File(e["path"], "r") as f:
                omega = np.asarray(f["omega_z"], dtype=np.float32)
            x = torch.from_numpy(omega).unsqueeze(0).unsqueeze(2).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                z = enc(x)
                x_hat = dec(z)
            x_hat = x_hat.float().squeeze(0).squeeze(1).cpu().numpy()
            mse = float(((omega - x_hat) ** 2).mean())
            floor = float(((omega - case_mean[e["case_id"]]) ** 2).mean())
            ssim_t = float(np.mean([_ssim(omega[t], x_hat[t]) for t in range(omega.shape[0])]))
            rows.append({"case_id": e["case_id"], "k": e["k"],
                         "mse": mse, "floor": floor,
                         "ratio": mse / max(floor, 1e-12),
                         "ssim": ssim_t})
    return pd.DataFrame(rows)


def make_figure_3(enc, dec, encs_b, device, out_path: Path, pick_idx: int = 0):
    e = encs_b[pick_idx]
    with h5py.File(e["path"], "r") as f:
        omega = np.asarray(f["omega_z"], dtype=np.float32)
    x = torch.from_numpy(omega).unsqueeze(0).unsqueeze(2).to(device)
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            z = enc(x)
            x_hat = dec(z)
    x_hat = x_hat.float().squeeze(0).squeeze(1).cpu().numpy()  # (T, H, W)
    residual = omega - x_hat

    frames = [25, 40, 55]
    fig, axes = plt.subplots(3, 4, figsize=(14, 11),
                             gridspec_kw={"width_ratios": [1, 1, 1, 0.06]})
    # Per-row vmax based on the 95th-percentile of |.|, not the max. The
    # raw omega field is highly localized (vortex cores hit |omega| ~ 1000+
    # while the bulk of the field is near zero), so using max would saturate
    # only the extreme pixels and wash out the wake. The 95th-percentile
    # saturation lets the vortex core clip but keeps the wake structure
    # legible. The per-row labels show both p95 (the colorbar limit) and
    # max|.| (the actual extreme value, which clips).
    p95_raw = float(np.percentile(np.abs(omega[frames]), 95))
    p95_dec = float(np.percentile(np.abs(x_hat[frames]), 95))
    p95_res = float(np.percentile(np.abs(residual[frames]), 95))
    vmax_raw = p95_raw
    vmax_dec = p95_dec
    vmax_res = p95_res
    max_raw = float(np.abs(omega[frames]).max())
    max_dec = float(np.abs(x_hat[frames]).max())
    max_res = float(np.abs(residual[frames]).max())
    row_specs = [
        ("raw $\\omega_z$", omega, vmax_raw, max_raw),
        ("decoded $\\hat\\omega_z$", x_hat, vmax_dec, max_dec),
        ("residual $\\omega_z - \\hat\\omega_z$", residual, vmax_res, max_res),
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
        f"Figure 3. Decoder reconstruction on Test B case {e['case_id']} "
        f"encounter {e['k']:02d}\n"
        f"G={e['G']:.2f}, D={e['D']:.2f}, Y={e['Y']:.2f}; JEPA + frozen E4 "
        f"encoder + 10k-iter decoder",
        y=1.02, fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Session 9 Step 2: decoder figures")
    p.add_argument("--jepa-checkpoint", required=True, type=str)
    p.add_argument("--decoder-checkpoint", required=True, type=str)
    p.add_argument("--output-dir", required=True, type=str)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--fig3-test-b-idx", type=int, default=0,
                   help="Which Test B encounter to feature in Figure 3 (default 0).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = require_rtx6000(gpu_index=args.gpu)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    enc, d = load_encoder(Path(args.jepa_checkpoint), device)
    dec = load_decoder(Path(args.decoder_checkpoint), d, device)
    print(f"[decoder-figures] encoder + decoder loaded, d={d}", flush=True)

    encs_a = gather_encounters("test_a")
    encs_b = gather_encounters("test_b")
    encs_c = gather_encounters("test_c")
    print(f"[decoder-figures] test_a={len(encs_a)}, test_b={len(encs_b)}, "
          f"test_c={len(encs_c)}", flush=True)

    df_a = per_encounter_mse(enc, dec, encs_a, device)
    df_b = per_encounter_mse(enc, dec, encs_b, device)
    df_c = per_encounter_mse(enc, dec, encs_c, device)
    df = pd.concat([
        df_a.assign(split="test_a"),
        df_b.assign(split="test_b"),
        df_c.assign(split="test_c"),
    ], ignore_index=True)
    df.to_csv(out_dir / "decoder_per_encounter.csv", index=False)
    print(f"[decoder-figures] wrote {out_dir / 'decoder_per_encounter.csv'}", flush=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for split, dfx, color in (("test_a", df_a, "tab:blue"),
                              ("test_b", df_b, "tab:orange"),
                              ("test_c", df_c, "tab:green")):
        ax.hist(dfx["ratio"], bins=20, alpha=0.5, color=color, label=f"{split} (n={len(dfx)})")
    ax.set_xlabel("MSE / per-case-mean floor")
    ax.set_ylabel("count of encounters")
    ax.set_title("Decoder per-encounter MSE ratio across splits")
    ax.axvline(1.0, color="black", linestyle="--", linewidth=0.8, label="floor")
    ax.axvline(2.0, color="red", linestyle=":", linewidth=0.8, label="2x pass criterion")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig_decoder_mse_distribution.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[decoder-figures] saved {out_dir / 'fig_decoder_mse_distribution.png'}",
          flush=True)

    fig3_path = out_dir / "fig3_decoder_reconstruction.png"
    make_figure_3(enc, dec, encs_b, device, fig3_path, pick_idx=args.fig3_test_b_idx)
    print(f"[decoder-figures] saved {fig3_path}", flush=True)


if __name__ == "__main__":
    main()
