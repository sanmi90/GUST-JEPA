"""5-panel parametric reconstruction comparison at the impact frame.

Protocol 2 (manuscript Section 6.7): predict the impact-frame omega field at
a held-out parameter c = (G, D, Y) WITHOUT seeing any DNS field at that c.

For each split (test_b, test_c) we render a 5-panel row:

    TRUE | JEPA KRR | JEPA rollout | Fukami AE KRR | POD KRR

Methods:
- TRUE: DNS omega at impact frame, displayed for visual reference only.
  Parametric methods do NOT see this field at the test c.
- JEPA KRR: Kernel ridge regression (RBF) from training (c, z_impact) pairs
  to predict z_impact at the test c, decoded via the JEPA visualisation
  decoder (LapFiLM).
- JEPA rollout: Load a Baseline (no-gust) initial latent at t=0 from test_a
  (Baseline encounters 4-5 are held out), roll the JEPA predictor forward
  40 frames conditioned on the test c, decode the resulting z_40.
- Fukami AE KRR: Same KRR-RBF on Fukami AE training (c, z_impact) pairs,
  decode via the Fukami AE decoder.
- POD KRR: KRR-RBF on training (c, alpha_impact) pairs (alpha = POD modal
  coefficients), reconstruct omega = mean + Phi @ alpha_pred.

The figure layout, fonts, colormap, airfoil overlay, and SSIM/eps_L2
metrics follow the existing ``scripts/_oneoff_recon_4way.py`` conventions.

Usage:
    python scripts/_oneoff_recon_4way_parametric.py
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
from sklearn.kernel_ridge import KernelRidge

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.baselines.fukami_ae import FukamiAEWrapper  # noqa: E402
from src.data.omega_pipeline import OmegaPipeline  # noqa: E402
from src.models.encoder import HybridCNNViTEncoder  # noqa: E402
from src.models.lap_film_decoder import LapFiLMDecoder  # noqa: E402
from src.models.predictor import AutoregressivePredictor  # noqa: E402

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
# Pre-paired JEPA predictor for the session12/S12_E_d64 encoder. Two candidates:
#   - exp_b1/predictor_jepa_d64/checkpoint_iter006000.pt   (only iter 6000)
#   - exp_b1_test3/predictor_jepa_d64_test1_noBN/checkpoint_iter020000.pt
# The test3 noBN variant is fully trained (20k iters) and was trained on
# the SAME latents_jepa_d64/train.npz. The latent_mean/std stored in the
# checkpoint will be used to normalise. We use this as the production pick.
JEPA_PREDICTOR_CKPT = (
    REPO
    / "outputs/session18/exp_b1_test3/predictor_jepa_d64_test1_noBN/checkpoint_iter020000.pt"
)
FUKAMI_CKPT = REPO / "outputs/session18/exp_b1/fukami_ae_d64/checkpoint_iter020000.pt"
POD_BASIS = REPO / "outputs/session18/exp_b1/pod_d64/pod_basis.npz"
PIPELINE_MANIFEST = REPO / "outputs/data_pipeline/v1/manifest.json"

LATENTS_JEPA = REPO / "outputs/session18/exp_b1/latents_jepa_d64"
LATENTS_FUKAMI = REPO / "outputs/session18/exp_b1/latents_fukami_d64"
LATENTS_POD = REPO / "outputs/session18/exp_b1/latents_pod_d64"

OUT_DIR = REPO / "paper" / "sections" / "figures" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Figure / display conventions (CLAUDE.md "Figure 3-style"):
VLIM_NORM = 3.0
SSIM_L = 8.31

# Physical extent for the omega image (192, 96).
X_EXTENT = (-1.5, 4.5)
Y_EXTENT = (-1.5, 1.5)


def load_airfoil_xy() -> np.ndarray:
    raw = PREVENT_ROOT / "data" / "raw" / "periodic" / "Baseline.h5"
    with h5py.File(raw, "r") as f:
        xy = np.asarray(f["airfoil_xy"])
    if not np.allclose(xy[0], xy[-1]):
        xy = np.vstack([xy, xy[0:1]])
    return xy


def airfoil_to_pixels(xy: np.ndarray, H: int = 192, W: int = 96) -> np.ndarray:
    x_min, x_max = X_EXTENT
    y_min, y_max = Y_EXTENT
    px_x = (xy[:, 0] - x_min) * (H - 1) / (x_max - x_min)
    px_y = (xy[:, 1] - y_min) * (W - 1) / (y_max - y_min)
    return np.stack([px_x, px_y], axis=-1)


def encounter_path(case_id: str, k: int) -> Path:
    return CACHE / "v1" / case_id / f"encounter_{k:02d}.h5"


def load_omega_impact_frame(case_id: str, k: int, pipeline: OmegaPipeline):
    p = encounter_path(case_id, k)
    with h5py.File(p, "r") as f:
        omega_raw = np.asarray(f["omega_z"], dtype=np.float32)
        impact_frame = int(f.attrs.get("impact_frame_estimate", 40))
    omega_clean = pipeline.preprocess_raw(omega_raw, case_id, int(k))
    omega_norm = pipeline.normalize(omega_clean)
    return omega_norm[impact_frame], omega_norm, impact_frame


def _get(blob, *names):
    for n in names:
        if n in blob.files:
            return blob[n]
    raise KeyError(f"none of {names} present in npz")


def gather_train_c_z_impact(latents_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (C_train [N, 3], Z_impact_train [N, d]).

    Uses each encounter's stored impact_frame to index z_full[:, impact_frame, :].
    """
    blob = np.load(latents_dir / "train.npz", allow_pickle=True)
    z_full = blob["z_full"].astype(np.float32)
    G = blob["G"].astype(np.float32)
    D = blob["D"].astype(np.float32)
    Y = blob["Y"].astype(np.float32)
    imps = _get(blob, "impact_frame").astype(np.int64)
    N, T, d = z_full.shape
    Z_imp = np.zeros((N, d), dtype=np.float32)
    for i in range(N):
        Z_imp[i] = z_full[i, int(imps[i]), :]
    C = np.stack([G, D, Y], axis=1)
    return C, Z_imp


def fit_krr_rbf(C_train: np.ndarray, Y_train: np.ndarray) -> KernelRidge:
    """Per-output KRR-RBF.

    sklearn KernelRidge with kernel='rbf' supports multi-output regression
    natively. Use alpha=0.1, gamma=0.5 (default per task spec).
    """
    krr = KernelRidge(alpha=0.1, kernel="rbf", gamma=0.5)
    krr.fit(C_train.astype(np.float64), Y_train.astype(np.float64))
    return krr


def jepa_decode_latent(
    z_norm: np.ndarray,
    decoder: LapFiLMDecoder,
    device: torch.device,
) -> np.ndarray:
    """Decode a single latent (d,) through the LapFiLM decoder.

    Returns omega map in pipeline-normalised space, shape (H, W) = (192, 96).
    """
    z = torch.from_numpy(z_norm).to(device).view(1, 1, -1)  # (1, 1, d)
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        out = decoder(z)
        pred_norm = out["pred"] if isinstance(out, dict) else out
        pred_norm = pred_norm.float()
        while pred_norm.dim() > 2:
            pred_norm = pred_norm.squeeze(0)
    return pred_norm.cpu().numpy()


def fukami_decode_latent(
    z: np.ndarray,
    wrapper: FukamiAEWrapper,
    device: torch.device,
) -> np.ndarray:
    """Decode a Fukami AE latent (d,) through the convolutional decoder."""
    z_t = torch.from_numpy(z).to(device).view(1, -1)
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        x_hat = wrapper.decoder(z_t).float()
    return x_hat.squeeze(0).squeeze(0).cpu().numpy()


def pod_reconstruct_from_alpha(alpha: np.ndarray, pod: dict, shape: tuple[int, int]) -> np.ndarray:
    """Reconstruct omega = mean + Phi @ alpha in pipeline-normalised space."""
    recon_flat = pod["mean"] + pod["Phi"] @ alpha.astype(np.float32)
    return recon_flat.reshape(shape)


def compute_panel_metrics(target_norm: np.ndarray, pred_norm: np.ndarray) -> tuple[float, float, float]:
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


def load_jepa_encoder_decoder(device: torch.device):
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


def load_jepa_predictor(ckpt_path: Path, d: int, device: torch.device):
    """Load predictor, patch out_proj if needed, return (pred, mean, std)."""
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = blob.get("run_config", {})
    pcfg = cfg.get("predictor_config", {})
    pred = AutoregressivePredictor(
        latent_dim=d,
        cond_dim=int(pcfg.get("cond_dim", 3)),
        hidden_dim=int(pcfg.get("hidden_dim", 384)),
        depth=int(pcfg.get("depth", 6)),
        heads=int(pcfg.get("heads", 16)),
        mlp_ratio=float(pcfg.get("mlp_ratio", 4.0)),
        dropout=float(pcfg.get("dropout", 0.1)),
        max_seq_len=int(pcfg.get("max_seq_len", 32)),
    ).to(device)
    state = blob["predictor_state_dict"]
    if "out_proj.1.weight" not in state and "out_proj.1.running_mean" not in state:
        from torch import nn as _nn

        out_lin = pred.out_proj[0]
        pred.out_proj = _nn.Sequential(out_lin, _nn.Identity()).to(device)
        print(f"[predictor] patched out_proj to Identity (no-output-bn checkpoint)")
    pred.load_state_dict(state)
    pred.eval()
    for p in pred.parameters():
        p.requires_grad_(False)
    mean = blob.get("latent_mean")
    std = blob.get("latent_std")
    if mean is None or std is None:
        # Fall back to run_config['latent_norm']
        ln = cfg.get("latent_norm", {})
        mean = ln.get("mean")
        std = ln.get("std")
    if mean is None or std is None:
        raise RuntimeError(
            f"checkpoint {ckpt_path} missing latent_mean / latent_std"
        )
    mean_t = torch.tensor(np.asarray(mean), dtype=torch.float32, device=device)
    std_t = torch.tensor(np.asarray(std), dtype=torch.float32, device=device)
    return pred, mean_t, std_t


def get_baseline_z0_from_test_a(latents_dir: Path) -> np.ndarray:
    """Pick Baseline encounter 4 from test_a (held-out Baseline trajectory).

    Returns the per-frame latent at t=0 in raw (un-normalised) latent space.
    """
    blob = np.load(latents_dir / "test_a.npz", allow_pickle=True)
    cids = _get(blob, "case_id", "case_ids")
    eis = _get(blob, "encounter_index", "encounter_indices")
    z_full = blob["z_full"]
    for i in range(len(cids)):
        if str(cids[i]) == "Baseline" and int(eis[i]) == 4:
            return z_full[i, 0, :].astype(np.float32)
    # Fallback: any Baseline encounter
    for i in range(len(cids)):
        if str(cids[i]) == "Baseline":
            return z_full[i, 0, :].astype(np.float32)
    raise RuntimeError("No Baseline encounter found in test_a latents")


@torch.no_grad()
def jepa_rollout_to_frame(
    pred: AutoregressivePredictor,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
    z0_raw: np.ndarray,
    cond_c: tuple[float, float, float],
    target_frame: int,
    device: torch.device,
) -> np.ndarray:
    """Full-context autoregressive rollout from a single z_0 to z_target_frame.

    Returns the predicted latent z_{target_frame} in RAW latent space
    (i.e. encoder output scale, ready for the LapFiLM decoder).
    """
    # Normalise to predictor's training space.
    z0 = torch.from_numpy(z0_raw).to(device)
    z0_n = (z0 - mean_t) / std_t  # (d,)
    z_seq = z0_n.view(1, 1, -1)  # (1, T=1, d)
    cond = torch.tensor([list(cond_c)], dtype=torch.float32, device=device)  # (1, 3)
    max_seq = int(pred.max_seq_len)
    for _ in range(target_frame):
        ctx = z_seq[:, -max_seq:, :]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            z_hat = pred(ctx, cond)
        z_seq = torch.cat([z_seq, z_hat[:, -1:, :].float()], dim=1)
    # Take frame target_frame in normalised space and un-normalise.
    z_target_n = z_seq[0, target_frame, :]  # (d,)
    z_target_raw = z_target_n * std_t + mean_t
    return z_target_raw.cpu().numpy().astype(np.float32)


def render_5panel(
    target_norm: np.ndarray,
    panels: dict[str, np.ndarray],
    title_meta: dict,
    out_basename: str,
    airfoil_px: np.ndarray,
    method_order: list[str],
) -> dict[str, tuple[float, float]]:
    """Render TRUE | JEPA KRR | JEPA rollout | Fukami AE KRR | POD KRR row."""
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "axes.linewidth": 1.0,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    n_panels = len(method_order)
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(5 * n_panels, 4.4),
        gridspec_kw={"wspace": 0.05},
    )

    metrics: dict[str, tuple[float, float]] = {}

    im = None
    for col, name in enumerate(method_order):
        ax = axes[col]
        field = panels[name] if name != "TRUE" else target_norm
        is_grey = isinstance(field, str) and field == "GREY"

        if is_grey:
            ax.imshow(
                np.zeros((96, 192)),
                origin="lower",
                cmap="Greys",
                vmin=0, vmax=1,
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
            ax.text(
                0.5, 0.5,
                "JEPA rollout\n(not run; see Sec. 8.5)",
                transform=ax.transAxes,
                ha="center", va="center",
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          alpha=0.85, edgecolor="gray", linewidth=0.6),
            )
            continue

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

        if name == "TRUE":
            txt = (
                f"{title_meta['split'].upper()} (param.)\n"
                f"{title_meta['case_id']}\n"
                f"enc {title_meta['k']:02d}, frame {title_meta['frame']}\n"
                f"G={title_meta['G']:+.2f}, D={title_meta['D']:.2f}, "
                f"Y={title_meta['Y']:+.2f}"
            )
            ax.text(
                0.02, 0.98, txt,
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=12,
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
            metrics[name] = (ssim_val, eps_val, mse_val)
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
    cbar_ax = fig.add_axes([0.915, 0.13, 0.010, 0.74])
    cbar = fig.colorbar(im, cax=cbar_ax, extend="both")
    cbar.set_label(r"$\omega_z\,/\,(3\sigma)$", fontsize=13)
    cbar.ax.tick_params(labelsize=12)

    fig.subplots_adjust(left=0.01, right=0.905, top=0.88, bottom=0.04)

    png_path = OUT_DIR / f"{out_basename}.png"
    pdf_path = OUT_DIR / f"{out_basename}.pdf"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[recon-param] wrote {png_path}")
    print(f"[recon-param] wrote {pdf_path}")
    return metrics


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        from src.utils.device import require_rtx6000

        device = require_rtx6000(gpu_index=0)
    print(f"[recon-param] device={device}")

    pipeline = OmegaPipeline.from_manifest(PIPELINE_MANIFEST)
    print(f"[recon-param] pipeline loaded; train_std={pipeline.train_stats.std:.4f}")

    encoder, decoder = load_jepa_encoder_decoder(device)
    print("[recon-param] JEPA encoder + decoder loaded")

    fukami = load_fukami(device, pipeline)
    print("[recon-param] Fukami AE loaded")

    pod = load_pod()
    print(f"[recon-param] POD basis loaded (d={pod['d']})")

    # ----- KRR fits on training (c, z_impact) pairs -----
    C_jepa, Z_jepa = gather_train_c_z_impact(LATENTS_JEPA)
    C_fuk, Z_fuk = gather_train_c_z_impact(LATENTS_FUKAMI)
    C_pod, Z_pod = gather_train_c_z_impact(LATENTS_POD)
    # Sanity: C grids should match across methods (same train manifest).
    assert np.allclose(C_jepa, C_fuk) and np.allclose(C_jepa, C_pod), \
        "Training c grids differ across method latents"
    print(f"[recon-param] N_train = {C_jepa.shape[0]} (c, z_impact) pairs")

    krr_jepa = fit_krr_rbf(C_jepa, Z_jepa)
    krr_fuk = fit_krr_rbf(C_fuk, Z_fuk)
    krr_pod = fit_krr_rbf(C_pod, Z_pod)
    print("[recon-param] KRR-RBF (alpha=0.1, gamma=0.5) fitted for JEPA, Fukami, POD")

    # ----- JEPA predictor for rollout -----
    predictor_ok = True
    predictor_label = "test1_noBN_iter020000"
    try:
        pred_model, mean_t, std_t = load_jepa_predictor(
            JEPA_PREDICTOR_CKPT, d=int(Z_jepa.shape[1]), device=device
        )
        z0_baseline = get_baseline_z0_from_test_a(LATENTS_JEPA)
        print(
            f"[recon-param] JEPA predictor loaded: {JEPA_PREDICTOR_CKPT.relative_to(REPO)}\n"
            f"[recon-param] Baseline z_0 from test_a: shape={z0_baseline.shape}, "
            f"mean={z0_baseline.mean():.4f}, std={z0_baseline.std():.4f}"
        )
    except Exception as exc:
        print(f"[recon-param] WARN: JEPA predictor unavailable ({exc!r}); rollout panel -> grey")
        predictor_ok = False
        pred_model = mean_t = std_t = z0_baseline = None
        predictor_label = "not_loaded"

    airfoil_xy = load_airfoil_xy()
    airfoil_px = airfoil_to_pixels(airfoil_xy)

    # ----- Two targets: test_b interior, test_c extrapolation -----
    splits_to_render = [
        {
            "split": "test_b",
            "case_id": "G+1.50_D1.50_Y+0.10",
            "k": 0,
            "G": 1.50, "D": 1.50, "Y": 0.10,
            "out_basename": "figS_recon_test_b_parametric",
        },
        {
            "split": "test_c",
            "case_id": "G+4.00_D1.00_Y+0.10",
            "k": 0,
            "G": 4.00, "D": 1.00, "Y": 0.10,
            "out_basename": "figS_recon_test_c_parametric",
        },
    ]

    method_order = ["TRUE", "JEPA KRR", "JEPA rollout", "Fukami AE KRR", "POD KRR"]
    summary: dict[str, dict[str, tuple[float, float]]] = {}

    for spec in splits_to_render:
        target_norm, _, impact_frame = load_omega_impact_frame(
            spec["case_id"], spec["k"], pipeline,
        )
        H, W = target_norm.shape
        c_test = np.array([[spec["G"], spec["D"], spec["Y"]]], dtype=np.float32)
        print(
            f"\n[recon-param] {spec['split']}: {spec['case_id']} k={spec['k']}, "
            f"impact_frame={impact_frame}"
        )

        # 1) JEPA KRR: predict z_impact at c_test, decode via LapFiLM
        z_jepa_pred = krr_jepa.predict(c_test.astype(np.float64))[0].astype(np.float32)
        jepa_krr_field = jepa_decode_latent(z_jepa_pred, decoder, device)

        # 2) JEPA rollout from Baseline z_0
        if predictor_ok:
            z_jepa_roll = jepa_rollout_to_frame(
                pred_model, mean_t, std_t,
                z0_raw=z0_baseline,
                cond_c=(spec["G"], spec["D"], spec["Y"]),
                target_frame=impact_frame,
                device=device,
            )
            jepa_roll_field = jepa_decode_latent(z_jepa_roll, decoder, device)
        else:
            jepa_roll_field = "GREY"

        # 3) Fukami AE KRR
        z_fuk_pred = krr_fuk.predict(c_test.astype(np.float64))[0].astype(np.float32)
        fuk_krr_field = fukami_decode_latent(z_fuk_pred, fukami, device)

        # 4) POD KRR
        alpha_pred = krr_pod.predict(c_test.astype(np.float64))[0].astype(np.float32)
        pod_krr_field = pod_reconstruct_from_alpha(alpha_pred, pod, shape=(H, W))

        # Sanity print
        for name, field in [
            ("JEPA KRR", jepa_krr_field),
            ("JEPA rollout", jepa_roll_field if not isinstance(jepa_roll_field, str) else None),
            ("Fukami AE KRR", fuk_krr_field),
            ("POD KRR", pod_krr_field),
        ]:
            if field is None:
                print(f"  {name:>14s}: skipped (predictor unavailable)")
                continue
            ssim_val, eps_val, mse_val = compute_panel_metrics(target_norm, field)
            print(
                f"  {name:>14s}: SSIM={ssim_val:.3f}, eps={eps_val:.3f}, "
                f"MSE={mse_val:.3f}, max|pred|={np.abs(field).max():.3f}"
            )

        title_meta = {
            "split": spec["split"],
            "case_id": spec["case_id"],
            "k": spec["k"],
            "frame": impact_frame,
            "G": spec["G"], "D": spec["D"], "Y": spec["Y"],
        }
        panels = {
            "JEPA KRR": jepa_krr_field,
            "JEPA rollout": jepa_roll_field,
            "Fukami AE KRR": fuk_krr_field,
            "POD KRR": pod_krr_field,
        }
        metrics = render_5panel(
            target_norm, panels, title_meta, spec["out_basename"], airfoil_px,
            method_order=method_order,
        )
        summary[spec["split"]] = metrics

    # ----- Summary table -----
    print("\n[recon-param] === SUMMARY ===")
    print(f"JEPA predictor checkpoint: {JEPA_PREDICTOR_CKPT.relative_to(REPO)} ({predictor_label})")
    print(f"{'split':>8s} | {'method':>16s} | {'SSIM':>6s} | {'eps_L2':>7s} | {'MSE':>7s}")
    print("-" * 60)
    for split, m in summary.items():
        for name, vals in m.items():
            s, e, ms = vals
            print(f"{split:>8s} | {name:>16s} | {s:6.3f} | {e:7.3f} | {ms:7.3f}")

    print("\n[recon-param] DONE")


if __name__ == "__main__":
    main()
