"""Session 16, Experiment 1, Part (b): decode unit perturbations along each
candidate axis through the production SL decoder, then correlate the decoded
fields with canonical flow descriptors.

We carry both bases per pivot_decision.json:
    1. PLS-3 (recipe-locked artefact at outputs/session16/exp1/pls_base.npz)
    2. PCA-3 (alternative basis at outputs/session16/exp1/pca_base.npz)

For each basis and each direction k in {1, 2, 3}, we sweep magnitudes
    m in {-2 sigma_k, -1 sigma_k, 0, +1 sigma_k, +2 sigma_k}
where sigma_k is the std of the k-th score across the train impact-frame
latents. The baseline z0 is the mean of train impact-frame latents.

Outputs:
    outputs/session16/exp1/exp1b_decoded_axes.npz
        omega_decoded[basis][axis][magnitude] -- raw-scale (192, 96) field
        z_perturbed[basis][axis][magnitude]   -- 64-D latent
        z0, axis_sigmas
    outputs/session16/exp1/exp1b_descriptors.json
        per-axis descriptor table + Pearson r vs magnitude per descriptor

Descriptors (canonical wake measures from the decoded raw-scale omega_z field):
    peak_pos_omega, peak_neg_omega: max and min vorticity (signed)
    centroid_x, centroid_y         : |omega|-weighted centroid in chord units
    circulation_pos, circulation_neg: int omega_+ dx dy, int omega_- dx dy
    wake_length                    : streamwise extent of |omega| > threshold
    wake_thickness                 : normal extent of |omega| > threshold

The physical domain is x in [-1.5, 4.5] (chord units), y in [-1.5, 1.5].
Grid is (192, 96) so dx = 6/192 = 0.03125, dy = 3/96 = 0.03125.
Wake region for length/thickness/circulation: x in [0.5, 4.0], y in [-1.0, 1.0]
(avoids airfoil + leading-edge stagnation; downstream of the trailing edge).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.data.omega_pipeline import OmegaPipeline  # noqa: E402
from src.models.encoder import HybridCNNViTEncoder  # noqa: E402
from src.models.lap_film_decoder import LapFiLMDecoder  # noqa: E402
from src.utils.device import require_rtx6000  # noqa: E402


OUT = REPO / "outputs" / "session16" / "exp1"
ENCODER_RUN = REPO / "outputs" / "runs" / "session12" / "S12_E_d64" / "encoder"
DECODER_CKPT = ENCODER_RUN / "decoder_specloss_recipe" / "decoder_iter012000.pt"
LATENTS_TRAIN = REPO / "outputs" / "session14" / "latents" / "S12_E_d64" / "train.npz"
PLS_BASE_PATH = OUT / "pls_base.npz"
PCA_BASE_PATH = OUT / "pca_base.npz"
OMEGA_MANIFEST = REPO / "outputs" / "data_pipeline" / "v1" / "manifest.json"

# Physical extent (from CLAUDE.md "Figure 3-style reconstruction panels"):
# -1.5 to 4.5 in x (chord units), -1.5 to 1.5 in y, 192x96 grid.
# The grid orientation in cache: omega_z shape (T, 192, 96) where the first
# spatial axis (192) is x and the second (96) is y (per
# scripts/session14_make_figures.py and similar).
X_EXTENT = (-1.5, 4.5)
Y_EXTENT = (-1.5, 1.5)
NX, NY = 192, 96
DX = (X_EXTENT[1] - X_EXTENT[0]) / NX
DY = (Y_EXTENT[1] - Y_EXTENT[0]) / NY

WAKE_X = (0.5, 4.0)
WAKE_Y = (-1.0, 1.0)
WAKE_THRESHOLD = 1.0  # raw omega units; cores ~ 500-4000, freestream ~ 0


def _coord_grid():
    x = np.linspace(X_EXTENT[0] + DX / 2, X_EXTENT[1] - DX / 2, NX)
    y = np.linspace(Y_EXTENT[0] + DY / 2, Y_EXTENT[1] - DY / 2, NY)
    return x, y


def wake_mask():
    x, y = _coord_grid()
    in_x = (x >= WAKE_X[0]) & (x <= WAKE_X[1])
    in_y = (y >= WAKE_Y[0]) & (y <= WAKE_Y[1])
    mask = in_x[:, None] & in_y[None, :]
    return mask


def compute_descriptors(omega_raw: np.ndarray) -> dict:
    """Compute canonical descriptors on a (192, 96) raw-scale omega field."""
    assert omega_raw.shape == (NX, NY), omega_raw.shape
    x, y = _coord_grid()
    abs_o = np.abs(omega_raw)
    pos_o = np.clip(omega_raw, 0.0, None)
    neg_o = np.clip(omega_raw, None, 0.0)

    desc: dict = {}
    desc["peak_pos_omega"] = float(omega_raw.max())
    desc["peak_neg_omega"] = float(omega_raw.min())

    total_w = float(abs_o.sum())
    if total_w > 0:
        cx = float((x[:, None] * abs_o).sum() / total_w)
        cy = float((y[None, :] * abs_o).sum() / total_w)
    else:
        cx = float("nan")
        cy = float("nan")
    desc["centroid_x"] = cx
    desc["centroid_y"] = cy

    desc["circulation_pos"] = float(pos_o.sum() * DX * DY)
    desc["circulation_neg"] = float(neg_o.sum() * DX * DY)

    mask = wake_mask()
    wake_field = abs_o * mask
    active = wake_field > WAKE_THRESHOLD
    if active.any():
        xs_active = np.where(active.any(axis=1))[0]
        ys_active = np.where(active.any(axis=0))[0]
        desc["wake_length"] = float((xs_active.max() - xs_active.min()) * DX)
        desc["wake_thickness"] = float((ys_active.max() - ys_active.min()) * DY)
    else:
        desc["wake_length"] = 0.0
        desc["wake_thickness"] = 0.0

    desc["wake_enstrophy"] = float((wake_field ** 2).sum() * DX * DY)
    return desc


def load_decoder(device: torch.device) -> tuple[HybridCNNViTEncoder, LapFiLMDecoder, OmegaPipeline]:
    enc_ckpts = sorted(ENCODER_RUN.glob("checkpoint_iter*.pt"))
    enc_ckpt = enc_ckpts[-1]
    enc_blob = torch.load(enc_ckpt, map_location="cpu", weights_only=False)
    enc_args = enc_blob["args"]
    enc = HybridCNNViTEncoder(
        latent_dim=int(enc_args["d"]),
        projection_norm=enc_args.get("projection_norm", "batchnorm"),
    )
    state = {
        k.removeprefix("encoder."): v
        for k, v in enc_blob["jepa_state_dict"].items()
        if k.startswith("encoder.")
    }
    enc.load_state_dict(state, strict=False)
    enc.eval().to(device)
    for p in enc.parameters():
        p.requires_grad_(False)

    dec_blob = torch.load(DECODER_CKPT, map_location="cpu", weights_only=False)
    dec_args = dec_blob["args"]
    bc = int(dec_args.get("decoder_base_ch", 64))
    channels = (bc, bc, int(bc * 0.75), int(bc * 0.5), int(bc * 0.375))
    dec = LapFiLMDecoder(
        latent_dim=int(enc_args["d"]),
        channels=channels,
        resblocks_per_level=int(dec_args.get("decoder_resblocks_per_level", 2)),
        upsample=dec_args.get("decoder_upsample", "pixelshuffle"),
        fourier_bands=int(dec_args.get("decoder_fourier_bands") or 4),
        use_film=bool(dec_args.get("decoder_use_film", True)),
        airfoil_mask_path=dec_args.get("airfoil_mask_path"),
    )
    dec.load_state_dict(dec_blob["decoder_state_dict"])
    dec.eval().to(device)
    for p in dec.parameters():
        p.requires_grad_(False)

    omega_pipeline = OmegaPipeline.from_manifest(OMEGA_MANIFEST)

    return enc, dec, omega_pipeline


def decode_latent(
    z: np.ndarray,
    dec: LapFiLMDecoder,
    omega_pipeline: OmegaPipeline,
    device: torch.device,
) -> np.ndarray:
    """Decode a (d,) latent to a (192, 96) raw-scale omega field."""
    z_t = torch.from_numpy(z.astype(np.float32)).unsqueeze(0).to(device)
    # Add a singleton time axis so we treat it as a 1-frame trajectory.
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        out = dec(z_t.unsqueeze(1))
        pred = out["pred"] if isinstance(out, dict) else out
        pred = pred.float()  # (1, 1, 1, H, W) or (1, 1, H, W)
        # Unify to (H, W)
        while pred.dim() > 2:
            pred = pred.squeeze(0)
        pred_norm = pred.cpu().numpy()
    pred_raw = omega_pipeline.unnormalize(pred_norm)
    return pred_raw


def main() -> None:
    device = require_rtx6000(gpu_index=int(os.environ.get("GPU", 0)))

    enc, dec, omega_pipeline = load_decoder(device)
    print(f"[exp1b] decoder loaded; params={sum(p.numel() for p in dec.parameters()):,}")

    train_data = np.load(LATENTS_TRAIN, allow_pickle=True)
    z_train = train_data["z"].astype(np.float64)
    z0 = z_train.mean(axis=0)
    print(f"[exp1b] baseline z0 norm = {np.linalg.norm(z0):.3f}")

    bases: dict = {}

    pls_blob = np.load(PLS_BASE_PATH, allow_pickle=True)
    pls_dirs = pls_blob["x_rotations"].T  # shape (3, 64)
    pls_scores_train = z_train @ pls_dirs.T
    pls_sigmas = pls_scores_train.std(axis=0)
    bases["PLS3"] = {"dirs": pls_dirs, "sigmas": pls_sigmas}
    print(f"[exp1b] PLS3 dirs shape = {pls_dirs.shape}, sigmas = {pls_sigmas}")

    pca_blob = np.load(PCA_BASE_PATH, allow_pickle=True)
    pca_dirs = pca_blob["components"]  # shape (3, 64)
    pca_mean = pca_blob["mean"]
    z_centered = z_train - pca_mean
    pca_scores_train = z_centered @ pca_dirs.T
    pca_sigmas = pca_scores_train.std(axis=0)
    bases["PCA3"] = {"dirs": pca_dirs, "sigmas": pca_sigmas}
    print(f"[exp1b] PCA3 dirs shape = {pca_dirs.shape}, sigmas = {pca_sigmas}")

    magnitude_levels = (-2.0, -1.0, 0.0, +1.0, +2.0)

    decoded_blob: dict = {"z0": z0}
    descriptors_summary: dict = {}

    for basis_name, basis in bases.items():
        dirs = basis["dirs"]
        sigmas = basis["sigmas"]
        descriptors_summary[basis_name] = {"axis_sigmas": sigmas.tolist()}
        for k in range(3):
            d_vec = dirs[k]
            sigma_k = float(sigmas[k])
            per_magnitude: dict = {}
            per_magnitude_omega = []
            per_magnitude_z = []
            for m in magnitude_levels:
                z_p = z0 + (m * sigma_k) * d_vec
                omega_pred = decode_latent(z_p, dec, omega_pipeline, device)
                desc = compute_descriptors(omega_pred)
                desc["magnitude_sigma"] = float(m)
                desc["magnitude_raw"] = float(m * sigma_k)
                per_magnitude[f"m={m:+g}"] = desc
                per_magnitude_omega.append(omega_pred.astype(np.float32))
                per_magnitude_z.append(z_p.astype(np.float32))
            decoded_blob[f"{basis_name}_axis{k+1}_omega"] = np.stack(per_magnitude_omega)
            decoded_blob[f"{basis_name}_axis{k+1}_z"] = np.stack(per_magnitude_z)

            mags = np.array(magnitude_levels)
            corr: dict = {}
            for desc_key in (
                "peak_pos_omega", "peak_neg_omega",
                "centroid_x", "centroid_y",
                "circulation_pos", "circulation_neg",
                "wake_length", "wake_thickness", "wake_enstrophy",
            ):
                vals = np.array([per_magnitude[f"m={m:+g}"][desc_key] for m in magnitude_levels])
                if np.any(np.isnan(vals)) or vals.std() < 1e-12:
                    corr[desc_key] = float("nan")
                else:
                    corr[desc_key] = float(np.corrcoef(mags, vals)[0, 1])

            descriptors_summary[basis_name][f"axis{k+1}"] = {
                "per_magnitude": per_magnitude,
                "pearson_r_vs_magnitude": corr,
            }
            print(f"[exp1b] {basis_name} axis{k+1} (sigma_k={sigma_k:.3f}) Pearson r:")
            for desc_key, r in corr.items():
                print(f"  {desc_key:20s} r = {r:+.3f}")

    save_np = OUT / "exp1b_decoded_axes.npz"
    np.savez(save_np, **decoded_blob)
    print(f"[exp1b] wrote {save_np.relative_to(REPO)}")

    save_js = OUT / "exp1b_descriptors.json"
    save_js.write_text(json.dumps(descriptors_summary, indent=2))
    print(f"[exp1b] wrote {save_js.relative_to(REPO)}")


if __name__ == "__main__":
    main()
