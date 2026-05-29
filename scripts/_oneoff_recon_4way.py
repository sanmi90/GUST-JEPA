"""4-way mid-plane vorticity reconstruction comparison at the impact frame.

For each held-out split (test_a, test_b, test_c), this script picks one
representative case + encounter, then renders a 4-panel side-by-side
figure:

    TRUE (DNS) | JEPA | Fukami AE | POD

Each method encodes the pipeline-normalised DNS omega field through its
own latent (d=64) and decodes back. Reconstructions are un-normalised back
to raw vorticity units; the displayed colorbar is fixed at vmin=-3,
vmax=+3 in 3-sigma normalised space (per CLAUDE.md "Figure 3-style"
convention; equivalent to roughly +/- 32 raw omega units for v1 train_std
3.55).

The encoder-decoder pairing matters. The session15/decoder_bc32 LapFiLM
decoder was trained on the production session12 d=64 encoder (sha256
0ea921c6f...). Using the session14/thrust6/jepa_d64_seed0 encoder would
not be compatible because the decoder was never trained against that
latent geometry. We therefore use the matched pair:
    encoder = outputs/runs/session12/S12_E_d64/encoder/checkpoint_iter020000.pt
    decoder = outputs/runs/session15/decoder_bc32/decoder_iter012000.pt

Usage:
    python scripts/_oneoff_recon_4way.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch
from matplotlib.patches import Polygon
from skimage.metrics import structural_similarity as ssim_skimage

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.baselines.fukami_ae import FukamiAEWrapper  # noqa: E402
from src.data.omega_pipeline import OmegaPipeline  # noqa: E402
from src.models.encoder import HybridCNNViTEncoder  # noqa: E402
from src.models.lap_film_decoder import LapFiLMDecoder  # noqa: E402

# Paths
PREVENT_ROOT = Path(os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT")))
CACHE = Path(
    os.environ.get(
        "VORTEX_JEPA_CACHE",
        str(PREVENT_ROOT / "data" / "processed" / "vortex-jepa"),
    )
)
JEPA_ENCODER_CKPT = REPO / "outputs/runs/session12/S12_E_d64/encoder/checkpoint_iter020000.pt"
JEPA_DECODER_CKPT = REPO / "outputs/runs/session15/decoder_bc32/decoder_iter012000.pt"
FUKAMI_CKPT = REPO / "outputs/session18/exp_b1/fukami_ae_d64/checkpoint_iter020000.pt"
POD_BASIS = REPO / "outputs/session18/exp_b1/pod_d64/pod_basis.npz"
PIPELINE_MANIFEST = REPO / "outputs/data_pipeline/v1/manifest.json"

OUT_DIR = REPO / "paper" / "sections" / "figures" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Figure / display conventions (CLAUDE.md "Figure 3-style"):
# vmin/vmax = +/-3 in pipeline-normalised space (3-sigma scale).
# Display is in raw omega units so the convention says +/-3 * (3 * train_std).
# train_std for v1 manifest = 3.6622, so vlim_raw = 3 * 3 * 3.6622 = 32.96.
# SSIM is reported on pipeline-normalised target; data_range = L = 8.31
# (= 2 * global p99.9(|target_norm|) for split_v2).
VLIM_NORM = 3.0
SSIM_L = 8.31

# Physical extent for the omega image (192, 96) -> (W=192, H=96 px after .T).
X_EXTENT = (-1.5, 4.5)
Y_EXTENT = (-1.5, 1.5)


def load_airfoil_xy() -> np.ndarray:
    """Load NACA 0012 surface coordinates (chord-normalised) from Baseline.h5."""
    raw = PREVENT_ROOT / "data" / "raw" / "periodic" / "Baseline.h5"
    with h5py.File(raw, "r") as f:
        xy = np.asarray(f["airfoil_xy"])
    if not np.allclose(xy[0], xy[-1]):
        xy = np.vstack([xy, xy[0:1]])
    return xy


def airfoil_to_pixels(xy: np.ndarray, H: int = 192, W: int = 96) -> np.ndarray:
    """Convert (x, y) physical coords to image-pixel coords after omega[t].T.

    After omega[t].T, the displayed array has shape (W, H) with origin
    lower-left. The horizontal axis is H (NX=192 cells, x in [-1.5, 4.5]);
    the vertical axis is W (NY=96 cells, y in [-1.5, 1.5]).
    """
    x_min, x_max = X_EXTENT
    y_min, y_max = Y_EXTENT
    px_x = (xy[:, 0] - x_min) * (H - 1) / (x_max - x_min)
    px_y = (xy[:, 1] - y_min) * (W - 1) / (y_max - y_min)
    return np.stack([px_x, px_y], axis=-1)


def encounter_path(case_id: str, k: int) -> Path:
    return CACHE / "v1" / case_id / f"encounter_{k:02d}.h5"


def load_omega_impact_frame(case_id: str, k: int, pipeline: OmegaPipeline):
    """Load the impact-frame omega slice for an encounter.

    Returns (omega_norm_impact, omega_norm_all, impact_frame, raw_omega_clean).
    omega_norm_impact: (192, 96) at impact, pipeline-applied.
    omega_norm_all: (120, 192, 96) full trajectory, pipeline-applied.
    """
    p = encounter_path(case_id, k)
    with h5py.File(p, "r") as f:
        omega_raw = np.asarray(f["omega_z"], dtype=np.float32)
        impact_frame = int(f.attrs.get("impact_frame_estimate", 40))
    omega_clean = pipeline.preprocess_raw(omega_raw, case_id, int(k))  # mask + clip
    omega_norm = pipeline.normalize(omega_clean)  # 3-sigma scale
    return omega_norm[impact_frame], omega_norm, impact_frame, omega_clean


def jepa_encode_decode(
    omega_norm_all: np.ndarray,
    impact_frame: int,
    encoder: HybridCNNViTEncoder,
    decoder: LapFiLMDecoder,
    device: torch.device,
) -> np.ndarray:
    """JEPA: encode the full trajectory, decode the impact-frame latent.

    The production encoder is unconditional and produces a per-frame z. The
    LapFiLM decoder takes a per-frame z and emits a per-frame omega map in
    pipeline-normalised space.
    """
    x = torch.from_numpy(omega_norm_all).unsqueeze(0).unsqueeze(2).to(device)  # (1, T, 1, H, W)
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        z = encoder(x)  # (1, T, d)
        z_impact = z[:, impact_frame : impact_frame + 1, :]  # (1, 1, d)
        out = decoder(z_impact)
        pred_norm = out["pred"] if isinstance(out, dict) else out
        pred_norm = pred_norm.float()
        # shape may be (1, 1, 1, H, W) or (1, 1, H, W)
        while pred_norm.dim() > 2:
            pred_norm = pred_norm.squeeze(0)
    return pred_norm.cpu().numpy()  # (H, W) at native (192, 96)


def fukami_encode_decode(
    omega_norm_impact: np.ndarray,
    wrapper: FukamiAEWrapper,
    device: torch.device,
) -> np.ndarray:
    """Fukami AE: encode-decode the impact-frame omega in pipeline-normalised space."""
    x = torch.from_numpy(omega_norm_impact).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        z = wrapper.encoder(x)
        x_hat_norm = wrapper.decoder(z).float()
    return x_hat_norm.squeeze(0).squeeze(0).cpu().numpy()


def pod_project_reconstruct(
    omega_norm_impact: np.ndarray,
    pod: dict,
) -> np.ndarray:
    """POD: alpha = (omega_norm_flat - mean) @ Phi; omega_hat = mean + Phi @ alpha."""
    flat = omega_norm_impact.reshape(-1).astype(np.float32)
    alpha = (flat - pod["mean"]) @ pod["Phi"]  # (d,)
    recon_flat = pod["mean"] + pod["Phi"] @ alpha  # (H*W,)
    return recon_flat.reshape(omega_norm_impact.shape)


def compute_panel_metrics(target_norm: np.ndarray, pred_norm: np.ndarray) -> tuple[float, float, float]:
    """SSIM (Wang K1=0.01, K2=0.03 default; data_range=L=8.31), relative L2 error,
    and mean squared error, all on pipeline-normalised fields."""
    ssim_val = float(
        ssim_skimage(
            target_norm.astype(np.float32),
            pred_norm.astype(np.float32),
            data_range=SSIM_L,
        )
    )
    sig = np.linalg.norm(target_norm)
    eps = float(np.linalg.norm(target_norm - pred_norm) / max(sig, 1e-9))
    mse = float(np.mean((target_norm - pred_norm) ** 2))
    return ssim_val, eps, mse


def load_jepa_models(device: torch.device):
    enc_blob = torch.load(JEPA_ENCODER_CKPT, map_location="cpu", weights_only=False)
    enc_args = enc_blob["args"]
    encoder = HybridCNNViTEncoder(
        latent_dim=int(enc_args["d"]),
        projection_norm=enc_args.get("projection_norm", "batchnorm"),
    )
    state = {
        k.removeprefix("encoder."): v
        for k, v in enc_blob["jepa_state_dict"].items()
        if k.startswith("encoder.")
    }
    encoder.load_state_dict(state, strict=False)
    encoder.eval().to(device)

    dec_blob = torch.load(JEPA_DECODER_CKPT, map_location="cpu", weights_only=False)
    dec_args = dec_blob["args"]
    bc = int(dec_args.get("decoder_base_ch", 64))
    channels = (bc, bc, int(bc * 0.75), int(bc * 0.5), int(bc * 0.375))
    decoder = LapFiLMDecoder(
        latent_dim=int(enc_args["d"]),
        channels=channels,
        resblocks_per_level=int(dec_args.get("decoder_resblocks_per_level", 2)),
        upsample=dec_args.get("decoder_upsample", "pixelshuffle"),
        fourier_bands=int(dec_args.get("decoder_fourier_bands") or 4),
        use_film=bool(dec_args.get("decoder_use_film", True)),
        airfoil_mask_path=dec_args.get("airfoil_mask_path"),
    )
    decoder.load_state_dict(dec_blob["decoder_state_dict"])
    decoder.eval().to(device)
    return encoder, decoder


def load_fukami(device: torch.device, pipeline: OmegaPipeline) -> FukamiAEWrapper:
    blob = torch.load(FUKAMI_CKPT, map_location="cpu", weights_only=False)
    a = blob["args"]
    wrapper = FukamiAEWrapper(
        latent_dim=int(a["d"]),
        n_deltas=len(a.get("observable_head_deltas", [8, 16, 24])),
        lambda_recon=float(a.get("lambda_recon", 1.0)),
        lambda_lift=float(a.get("lambda_lift", 1.0)),
        omega_pipeline=pipeline,
        recon_loss_type=str(a.get("recon_loss_type", "mse") or "mse"),
        charbonnier_epsilon=float(a.get("charbonnier_epsilon", 0.05)),
        activation=str(a.get("activation", "relu")),
        use_conv_norm=not bool(a.get("no_conv_norm", False)),
    ).to(device)
    wrapper.load_state_dict(blob["wrapper_state_dict"])
    wrapper.eval()
    return wrapper


def load_pod() -> dict:
    blob = np.load(POD_BASIS)
    return {
        "Phi": blob["Phi"].astype(np.float32),
        "mean": blob["mean"].astype(np.float32),
        "d": int(blob["d"]),
    }


def render_4panel(
    target_norm: np.ndarray,
    panels: dict[str, np.ndarray],
    title_meta: dict,
    out_basename: str,
    airfoil_px: np.ndarray,
) -> None:
    """Render TRUE | JEPA | Fukami AE | POD as a 4-panel figure.

    panels keys: "JEPA", "Fukami AE", "POD". target_norm is the DNS field
    in pipeline-normalised space. We display in 3-sigma normalised space at
    vmin/vmax = +/-3 (equivalent to roughly raw omega in [-32.96, +32.96]).
    """
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "axes.linewidth": 1.0,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    method_order = ["TRUE", "JEPA", "Fukami AE", "POD"]
    fields = {
        "TRUE": target_norm,
        "JEPA": panels["JEPA"],
        "Fukami AE": panels["Fukami AE"],
        "POD": panels["POD"],
    }

    fig, axes = plt.subplots(
        1, 4,
        figsize=(20, 4.4),
        gridspec_kw={"wspace": 0.05},
    )

    im = None
    for col, name in enumerate(method_order):
        ax = axes[col]
        field = fields[name]
        im = ax.imshow(
            field.T,
            origin="lower",
            cmap="RdBu_r",
            vmin=-VLIM_NORM,
            vmax=+VLIM_NORM,
            aspect="auto",
        )
        ax.add_patch(Polygon(
            airfoil_px, closed=True,
            facecolor="black", edgecolor="black",
            linewidth=0.8, zorder=10,
        ))
        ax.set_title(name, fontsize=15, fontweight="bold", pad=8)
        ax.set_xticks([])
        ax.set_yticks([])

        # Annotate metadata top-left on TRUE; metrics bottom-right on others.
        if name == "TRUE":
            txt = (
                f"{title_meta['split'].upper()}\n"
                f"{title_meta['case_id']}\n"
                f"enc {title_meta['k']:02d}, frame {title_meta['frame']}\n"
                f"G={title_meta['G']:+.2f}, D={title_meta['D']:.2f}, "
                f"Y={title_meta['Y']:+.2f}"
            )
            ax.text(
                0.02, 0.98, txt,
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=13,
                family="monospace",
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor="white",
                    alpha=0.82,
                    edgecolor="gray",
                    linewidth=0.6,
                ),
            )
        else:
            ssim_val, eps_val, mse_val = compute_panel_metrics(target_norm, field)
            txt = (f"SSIM={ssim_val:.3f}\n"
                   f"$\\varepsilon_{{L^{{2}}}}$={eps_val:.3f}\n"
                   f"MSE={mse_val:.3f}")
            ax.text(
                0.98, 0.04, txt,
                transform=ax.transAxes,
                ha="right", va="bottom",
                fontsize=13,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    alpha=0.85,
                    edgecolor="gray",
                    linewidth=0.6,
                ),
            )

    # Shared colorbar to the right.
    cbar_ax = fig.add_axes([0.91, 0.13, 0.012, 0.74])
    cbar = fig.colorbar(im, cax=cbar_ax, extend="both")
    cbar.set_label(r"$\omega_z\,/\,(3\sigma)$", fontsize=13)
    cbar.ax.tick_params(labelsize=12)

    # Tighter layout, leaving room for the colorbar.
    fig.subplots_adjust(left=0.01, right=0.90, top=0.88, bottom=0.04)

    png_path = OUT_DIR / f"{out_basename}.png"
    pdf_path = OUT_DIR / f"{out_basename}.pdf"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[recon-4way] wrote {png_path}")
    print(f"[recon-4way] wrote {pdf_path}")


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # Optional: enforce RTX 6000 per CLAUDE.md. Fail loudly if absent.
        from src.utils.device import require_rtx6000

        device = require_rtx6000(gpu_index=0)
    print(f"[recon-4way] device={device}")

    pipeline = OmegaPipeline.from_manifest(PIPELINE_MANIFEST)
    print(f"[recon-4way] pipeline loaded; train_std={pipeline.train_stats.std:.4f}")

    encoder, decoder = load_jepa_models(device)
    print("[recon-4way] JEPA encoder + decoder loaded")

    fukami = load_fukami(device, pipeline)
    print("[recon-4way] Fukami AE loaded")

    pod = load_pod()
    print(f"[recon-4way] POD basis loaded (d={pod['d']})")

    airfoil_xy = load_airfoil_xy()
    airfoil_px = airfoil_to_pixels(airfoil_xy)

    # Pick one representative encounter per split. Criteria:
    #   - test_a (held-out encounter in trained case): moderate +G to show
    #     non-trivial gust dynamics. G+2.00_D1.00_Y+0.10 encounter 04 is
    #     a val_encounter for that case in split_v2.
    #   - test_b (interior parametric value, unseen): G+1.50_D1.50_Y+0.10
    #     encounter 00. Interior of the (G, D, Y) grid, never seen in train.
    #   - test_c (G=+4 extrapolation): G+4.00_D1.00_Y+0.10 encounter 00.
    splits_to_render = [
        {
            "split": "test_a",
            "case_id": "G+2.00_D1.00_Y+0.10",
            "k": 4,
            "G": 2.00, "D": 1.00, "Y": 0.10,
            "out_basename": "figS_recon_test_a",
        },
        {
            "split": "test_b",
            "case_id": "G+1.50_D1.50_Y+0.10",
            "k": 0,
            "G": 1.50, "D": 1.50, "Y": 0.10,
            "out_basename": "figS_recon_test_b",
        },
        {
            "split": "test_c",
            "case_id": "G+4.00_D1.00_Y+0.10",
            "k": 0,
            "G": 4.00, "D": 1.00, "Y": 0.10,
            "out_basename": "figS_recon_test_c",
        },
    ]

    for spec in splits_to_render:
        target_norm, omega_norm_all, impact_frame, _ = load_omega_impact_frame(
            spec["case_id"], spec["k"], pipeline,
        )
        print(
            f"[recon-4way] {spec['split']}: {spec['case_id']} k={spec['k']}, "
            f"impact_frame={impact_frame}"
        )

        jepa_pred = jepa_encode_decode(
            omega_norm_all, impact_frame, encoder, decoder, device,
        )
        fukami_pred = fukami_encode_decode(target_norm, fukami, device)
        pod_pred = pod_project_reconstruct(target_norm, pod)

        # Shape sanity.
        assert jepa_pred.shape == target_norm.shape, (jepa_pred.shape, target_norm.shape)
        assert fukami_pred.shape == target_norm.shape, (fukami_pred.shape, target_norm.shape)
        assert pod_pred.shape == target_norm.shape, (pod_pred.shape, target_norm.shape)

        # Print sanity metrics to stdout so we can verify before writing the file.
        for name, pred in [("JEPA", jepa_pred), ("Fukami AE", fukami_pred), ("POD", pod_pred)]:
            ssim_val, eps_val, mse_val = compute_panel_metrics(target_norm, pred)
            print(
                f"  {name:>10s}: SSIM={ssim_val:.3f}, eps={eps_val:.3f}, "
                f"MSE={mse_val:.3f}, max|pred|={np.abs(pred).max():.3f}"
            )

        title_meta = {
            "split": spec["split"],
            "case_id": spec["case_id"],
            "k": spec["k"],
            "frame": impact_frame,
            "G": spec["G"], "D": spec["D"], "Y": spec["Y"],
        }
        render_4panel(
            target_norm,
            {"JEPA": jepa_pred, "Fukami AE": fukami_pred, "POD": pod_pred},
            title_meta,
            spec["out_basename"],
            airfoil_px,
        )

    print("[recon-4way] DONE")


if __name__ == "__main__":
    main()
