"""Recompute test_a SSIM under two conventions:
  (A) Fukami: raw scale, c1=0.16, c2=1.44 (matches decoder_summary.json)
  (B) Wang standard on NORMALIZED data, K1=0.01, K2=0.03, L=empirical range
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

REPO = Path("/home/carlos/GUST-JEPA")
sys.path.insert(0, str(REPO))

from src.data.omega_pipeline import OmegaPipeline
from src.models.encoder import HybridCNNViTEncoder
from src.models.lap_film_decoder import LapFiLMDecoder
from src.utils.device import require_rtx6000


ENC_CKPT = REPO / "outputs/runs/session12/S12_E_d64/encoder/checkpoint_iter020000.pt"
DEC_CKPT = REPO / "outputs/runs/session12/S12_E_d64/encoder/decoder_specloss_recipe/decoder_iter030000.pt"
PIPE     = REPO / "outputs/data_pipeline/v1/manifest.json"
SPLIT    = REPO / "configs/splits/split_v2.json"
CACHE    = Path("/home/carlos/PREVENT/data/processed/vortex-jepa/v1")


def fukami_ssim(x, y, c1=0.16, c2=1.44):
    mu_x, mu_y = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = ((x - mu_x) * (y - mu_y)).mean()
    num = (2 * mu_x * mu_y + c1) * (2 * cov + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (vx + vy + c2)
    return float(num / max(den, 1e-12))


def wang_ssim(x, y, L, K1=0.01, K2=0.03):
    c1 = (K1 * L) ** 2
    c2 = (K2 * L) ** 2
    return fukami_ssim(x, y, c1=c1, c2=c2)


def load_encoder(ckpt_path, device):
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = blob["args"]
    enc = HybridCNNViTEncoder(latent_dim=int(args["d"]),
                              projection_norm=args.get("projection_norm", "batchnorm"))
    state = {k.removeprefix("encoder."): v for k, v in blob["jepa_state_dict"].items()
             if k.startswith("encoder.")}
    enc.load_state_dict(state, strict=False)
    enc.eval().to(device)
    for p in enc.parameters(): p.requires_grad_(False)
    return enc, int(args["d"])


def load_decoder(ckpt_path, latent_dim, device):
    """Mirror the build path in scripts/session9_train_decoder.py::build_decoder."""
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    da = blob.get("args", {})
    bc = int(da.get("decoder_base_ch", 64))
    channels = (bc, bc, int(bc * 0.75), int(bc * 0.5), int(bc * 0.375))
    fb = da.get("decoder_fourier_bands")
    if fb is None:
        fb = 4
    dec = LapFiLMDecoder(
        latent_dim=latent_dim,
        channels=channels,
        resblocks_per_level=int(da.get("decoder_resblocks_per_level", 2)),
        upsample=da.get("decoder_upsample", "pixelshuffle"),
        fourier_bands=int(fb),
        use_film=bool(da.get("decoder_use_film", True)),
        airfoil_mask_path=da.get("airfoil_mask_path"),
    )
    dec.load_state_dict(blob["decoder_state_dict"], strict=True)
    dec.eval().to(device)
    for p in dec.parameters(): p.requires_grad_(False)
    return dec


def main():
    device = require_rtx6000(gpu_index=0)
    print(f"device: {device} ({torch.cuda.get_device_name(device.index)})")

    pipe = OmegaPipeline.from_manifest(PIPE)
    print(f"omega_pipeline: train_std={pipe.train_stats.std:.4f}, divisor=3sigma={3*pipe.train_stats.std:.4f}")

    with open(SPLIT) as f:
        manifest = json.load(f)

    enc, latent_dim = load_encoder(ENC_CKPT, device)
    print(f"encoder: latent_dim={latent_dim}")
    dec = load_decoder(DEC_CKPT, latent_dim, device)
    print(f"decoder: LapFiLM loaded")

    test_a_encs = []
    for cid, case in manifest["cases"].items():
        if case["split"] != "train":
            continue
        ks = case.get("val_encounter_indices") or case.get("test_a_encounter_indices", [])
        for k in ks:
            path = CACHE / cid / f"encounter_{int(k):02d}.h5"
            if path.exists():
                test_a_encs.append({"case_id": cid, "k": int(k), "path": path})
    print(f"test_a encounters: {len(test_a_encs)}")

    # Stats accumulators
    norm_max_per_enc = []
    norm_p999_per_enc = []
    fukami_ssims_raw = []
    wang_ssims_L6 = []
    wang_ssims_Lemp = []

    # Per-encounter target/pred (both in normalized space) for global L estimation
    all_norm_targets_max = []
    encounter_pred_norm = []
    encounter_targ_norm = []

    with torch.no_grad():
        for i, e in enumerate(test_a_encs):
            with h5py.File(e["path"], "r") as f:
                omega_raw = np.asarray(f["omega_z"], dtype=np.float32)
            # Pipeline-preprocessed target (raw scale, masked + clipped)
            target_raw = pipe.preprocess_raw(omega_raw, e["case_id"], e["k"])
            # Normalized target (3-sigma scale)
            target_norm = pipe.normalize(target_raw)

            x = torch.from_numpy(target_norm).unsqueeze(0).unsqueeze(2).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                z = enc(x)
                dec_out = dec(z)
                pred_norm_t = dec_out["pred"] if isinstance(dec_out, dict) else dec_out
            pred_norm = pred_norm_t.float().squeeze(0).squeeze(1).cpu().numpy()
            pred_raw = pipe.unnormalize(pred_norm)

            # Stats on target_norm range (for L estimation)
            norm_max_per_enc.append(float(np.abs(target_norm).max()))
            norm_p999_per_enc.append(float(np.percentile(np.abs(target_norm), 99.9)))

            T = target_raw.shape[0]
            # Fukami SSIM on raw scale (matches decoder_summary convention)
            fukami_ssims_raw.append(float(np.mean([fukami_ssim(target_raw[t], pred_raw[t]) for t in range(T)])))

            # Wang SSIM on normalized space, L=6 (matches visualization [-3,+3] range)
            wang_ssims_L6.append(float(np.mean([wang_ssim(target_norm[t], pred_norm[t], L=6.0) for t in range(T)])))

            encounter_targ_norm.append(target_norm)
            encounter_pred_norm.append(pred_norm)

            if (i + 1) % 20 == 0 or (i + 1) == len(test_a_encs):
                print(f"  {i+1}/{len(test_a_encs)} processed", flush=True)

    # Empirical global L: 99.9th percentile of |target_norm| across all encounters
    all_abs = np.concatenate([t.ravel() for t in encounter_targ_norm])
    L_emp_max = float(np.abs(all_abs).max())
    L_emp_p999 = float(np.percentile(np.abs(all_abs), 99.9))
    # 2x because L is the "range" (max - min); symmetric data -> 2x max gives upper-bound range
    L_emp_full = 2.0 * L_emp_max  # min--max swing
    L_emp_p999_full = 2.0 * L_emp_p999

    # Re-compute Wang SSIM with the empirical L (use p99.9 to avoid outlier domination)
    for tgt_n, prd_n in zip(encounter_targ_norm, encounter_pred_norm):
        T = tgt_n.shape[0]
        wang_ssims_Lemp.append(float(np.mean([wang_ssim(tgt_n[t], prd_n[t], L=L_emp_p999_full) for t in range(T)])))

    print()
    print("="*80)
    print("test_a SSIM under multiple conventions  (v2 production, n=86 encounters)")
    print("="*80)
    print(f"\nNormalized data range (target_norm = raw / (3 * train_std), train_std={pipe.train_stats.std:.4f})")
    print(f"  per-encounter |target_norm|.max(): mean={np.mean(norm_max_per_enc):.3f}  median={np.median(norm_max_per_enc):.3f}  max={max(norm_max_per_enc):.3f}")
    print(f"  per-encounter |target_norm| p99.9: mean={np.mean(norm_p999_per_enc):.3f}  median={np.median(norm_p999_per_enc):.3f}  max={max(norm_p999_per_enc):.3f}")
    print(f"  global |target_norm|.max(): {L_emp_max:.3f}  → range L = 2*max = {L_emp_full:.3f}")
    print(f"  global |target_norm| p99.9: {L_emp_p999:.3f}  → range L = 2*p99.9 = {L_emp_p999_full:.3f}")
    print()
    print(f"{'convention':<55} {'mean':>8} {'median':>8}")
    print("-" * 75)
    a = np.array(fukami_ssims_raw)
    print(f"{'(A) Fukami, raw scale, c1=0.16, c2=1.44 (current)':<55} {a.mean():>8.4f} {np.median(a):>8.4f}")
    b = np.array(wang_ssims_L6)
    print(f"{'(B) Wang, normalized, K1=0.01 K2=0.03, L=6 [visualization]':<55} {b.mean():>8.4f} {np.median(b):>8.4f}")
    c_ = np.array(wang_ssims_Lemp)
    print(f"{'(C) Wang, normalized, K1=0.01 K2=0.03, L=' + f'{L_emp_p999_full:.2f}'.ljust(8) + ' [emp p99.9]':<55} {c_.mean():>8.4f} {np.median(c_).item():>8.4f}")
    print()
    print(f"D99 gate (Test A SSIM_mean ≥ 0.60):")
    print(f"  (A) Fukami:               {a.mean():.4f}  {'PASS' if a.mean() >= 0.60 else 'FAIL'}")
    print(f"  (B) Wang L=6:             {b.mean():.4f}  {'PASS' if b.mean() >= 0.60 else 'FAIL'}")
    print(f"  (C) Wang L={L_emp_p999_full:.2f}:        {c_.mean():.4f}  {'PASS' if c_.mean() >= 0.60 else 'FAIL'}")


if __name__ == "__main__":
    main()
