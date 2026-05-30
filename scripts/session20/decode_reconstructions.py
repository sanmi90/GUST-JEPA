"""Session 20 shared decode step for Tracks D-i (OT field metric) and F
(scale decomposition).

Produces, per split (val=test_a, test_b, test_c) and per method (jepa, fukami,
pod), the reconstructed mid-plane vorticity field at a frame window around impact,
alongside the pipeline-normalised DNS target. All fields are in the 3-sigma
NORMALISED space (loss/metric space, CLAUDE.md); un-normalise only at figure time.

Reconstruction paths (canonical, matching D129/D131):
  jepa   : DNS frame -> HybridCNNViTEncoder -> z -> LapFiLMDecoder -> pred_norm.
  fukami : DNS frame -> FukamiAEWrapper.encoder(omega_norm) -> z -> .decoder(z).
  pod    : omega_norm.flat -> coeffs=(flat-mean)@Phi -> recon = coeffs@Phi.T + mean.

A self-check prints JEPA Wang-SSIM on val; it must land near the D131 anchor
(test_a SSIM ~ 0.71) or the normalisation is wrong (the D129 double-normalise bug).

Output: outputs/session20/decoded/{split}.npz with keys
  target_norm (n, F, H, W), jepa_norm, fukami_norm, pod_norm (same shape),
  frames (n, F) absolute frame indices, impact_frame (n,), case_ids, G, D, Y.
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.data.omega_pipeline import OmegaPipeline  # noqa: E402
from src.models.encoder import HybridCNNViTEncoder  # noqa: E402
from src.models.lap_film_decoder import LapFiLMDecoder  # noqa: E402

PREVENT = Path(os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT")))
CACHE = Path(os.environ.get("VORTEX_JEPA_CACHE", str(PREVENT / "data/processed/vortex-jepa")))
ENC_CKPT = REPO / "outputs/runs/session12/S12_E_d64/encoder/checkpoint_iter020000.pt"
DEC_CKPT = REPO / "outputs/runs/session12/S12_E_d64/encoder/decoder_specloss_recipe/decoder_iter030000.pt"
ENC_CKPT_D32 = REPO / "outputs/runs/session12/S12_E_d32/encoder/checkpoint_iter020000.pt"
DEC_CKPT_D32 = REPO / "outputs/runs/session12/S12_E_d32/encoder/decoder_specloss_recipe/decoder_iter030000.pt"
FUKAMI_CKPT = REPO / "outputs/session18/exp_b1/fukami_ae_d64/checkpoint_iter006000.pt"
POD_BASIS = REPO / "outputs/session18/exp_b1/pod_d64/pod_basis.npz"
PIPE = REPO / "outputs/data_pipeline/v1/manifest.json"
SPLIT = REPO / "configs/splits/split_v2.json"

OFFSETS = (-8, 0, 8, 16, 24, 32, 40)  # relative to impact frame


def pick_device() -> torch.device:
    """Prefer a non-RTX CUDA card (L40S) to leave the RTX 6000s for Track A."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    for i in range(torch.cuda.device_count()):
        if "RTX" not in torch.cuda.get_device_name(i):
            return torch.device(f"cuda:{i}")
    return torch.device("cuda:0")


def wang_ssim(x, y, L, K1=0.01, K2=0.03):
    c1, c2 = (K1 * L) ** 2, (K2 * L) ** 2
    mu_x, mu_y = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = ((x - mu_x) * (y - mu_y)).mean()
    num = (2 * mu_x * mu_y + c1) * (2 * cov + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (vx + vy + c2)
    return float(num / max(den, 1e-12))


def load_jepa(device, enc_ckpt=ENC_CKPT, dec_ckpt=DEC_CKPT):
    blob = torch.load(enc_ckpt, map_location="cpu", weights_only=False)
    args = blob["args"]
    enc = HybridCNNViTEncoder(latent_dim=int(args["d"]),
                              projection_norm=args.get("projection_norm", "batchnorm"))
    state = {k.removeprefix("encoder."): v for k, v in blob["jepa_state_dict"].items()
             if k.startswith("encoder.")}
    enc.load_state_dict(state, strict=False)
    enc.eval().to(device)
    db = torch.load(dec_ckpt, map_location="cpu", weights_only=False)
    da = db.get("args", {})
    bc = int(da.get("decoder_base_ch", 64))
    channels = (bc, bc, int(bc * 0.75), int(bc * 0.5), int(bc * 0.375))
    dec = LapFiLMDecoder(
        latent_dim=int(args["d"]), channels=channels,
        resblocks_per_level=int(da.get("decoder_resblocks_per_level", 2)),
        upsample=da.get("decoder_upsample", "pixelshuffle"),
        fourier_bands=int(da.get("decoder_fourier_bands") or 4),
        use_film=bool(da.get("decoder_use_film", True)),
        airfoil_mask_path=da.get("airfoil_mask_path"),
    )
    dec.load_state_dict(db["decoder_state_dict"], strict=True)
    dec.eval().to(device)
    for p in list(enc.parameters()) + list(dec.parameters()):
        p.requires_grad_(False)

    @torch.no_grad()
    def recon(omega_norm_THW: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(omega_norm_THW).unsqueeze(0).unsqueeze(2).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = dec(enc(x))
            pred = out["pred"] if isinstance(out, dict) else out
        return pred.float().squeeze(0).squeeze(1).cpu().numpy()

    return recon


def load_fukami(device):
    from src.baselines.fukami_ae import FukamiAEWrapper
    blob = torch.load(FUKAMI_CKPT, map_location="cpu", weights_only=False)
    ta = blob["args"]
    w = FukamiAEWrapper(
        latent_dim=int(ta["d"]),
        n_deltas=len(ta.get("observable_head_deltas") or [8, 16, 24]),
        omega_pipeline=None,  # we feed already-normalised omega to encoder/decoder
        recon_loss_type=str(ta.get("recon_loss_type", "mse")),
        activation=str(ta.get("activation", "relu")),
        use_conv_norm=not bool(ta.get("no_conv_norm", False)),
    ).to(device)
    w.load_state_dict(blob["wrapper_state_dict"])
    w.eval()

    @torch.no_grad()
    def recon(omega_norm_THW: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(omega_norm_THW).unsqueeze(1).to(device)  # (T,1,H,W)
        z = w.encoder(x)
        xh = w.decoder(z)
        return xh.float().squeeze(1).cpu().numpy()

    return recon


def load_pod():
    blob = np.load(POD_BASIS)
    Phi = blob["Phi"].astype(np.float32)   # (H*W, d)
    mean = blob["mean"].astype(np.float32)  # (H*W,)

    def recon(omega_norm_THW: np.ndarray) -> np.ndarray:
        T, H, W = omega_norm_THW.shape
        flat = omega_norm_THW.reshape(T, -1).astype(np.float32)
        coeffs = (flat - mean[None]) @ Phi
        rec = coeffs @ Phi.T + mean[None]
        return rec.reshape(T, H, W)

    return recon


def gather(split):
    with open(SPLIT) as f:
        m = json.load(f)
    out = []
    for cid, case in m["cases"].items():
        if split == "test_a" and case["split"] == "train":
            ks = case.get("val_encounter_indices") or case.get("test_a_encounter_indices", [])
        elif split in ("test_b", "test_c") and case["split"] == split:
            ks = list(range(case["n_encounters_full"]))
        else:
            continue
        for k in ks:
            p = CACHE / "v1" / cid / f"encounter_{int(k):02d}.h5"
            if p.exists():
                out.append(dict(case_id=cid, k=int(k), path=str(p),
                                G=float(case["G"]), D=float(case["D"]), Y=float(case["Y"])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["test_a", "test_b", "test_c"])
    ap.add_argument("--out", type=Path, default=REPO / "outputs/session20/decoded")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    print(f"[decode] device={device} "
          f"({torch.cuda.get_device_name(device.index) if device.type=='cuda' else 'cpu'})")
    pipe = OmegaPipeline.from_manifest(PIPE)
    jepa = load_jepa(device, ENC_CKPT, DEC_CKPT)
    jepa32 = load_jepa(device, ENC_CKPT_D32, DEC_CKPT_D32)
    fukami, pod = load_fukami(device), load_pod()

    for split in args.splits:
        encs = gather(split)
        if not encs:
            print(f"[decode] {split}: 0 encounters"); continue
        n = len(encs)
        F = len(OFFSETS)
        H, W = 192, 96
        tgt = np.zeros((n, F, H, W), np.float32)
        rj = np.zeros((n, F, H, W), np.float32)
        rj32 = np.zeros((n, F, H, W), np.float32)
        rf = np.zeros((n, F, H, W), np.float32)
        rp = np.zeros((n, F, H, W), np.float32)
        frames = np.zeros((n, F), np.int32)
        impact = np.zeros(n, np.int32)
        cids, G, D, Y = [], np.zeros(n), np.zeros(n), np.zeros(n)
        ssim_acc = []
        for i, e in enumerate(encs):
            with h5py.File(e["path"], "r") as f:
                omega_raw = np.asarray(f["omega_z"], dtype=np.float32)
                imp = int(f.attrs.get("impact_frame_estimate", 40))
            target_raw = pipe.preprocess_raw(omega_raw, e["case_id"], e["k"])
            target_norm = pipe.normalize(target_raw)  # (120,H,W)
            jn = jepa(target_norm); jn32 = jepa32(target_norm)
            fn = fukami(target_norm); pn = pod(target_norm)
            fr = np.clip(np.array(OFFSETS) + imp, 0, target_norm.shape[0] - 1)
            tgt[i] = target_norm[fr]; rj[i] = jn[fr]; rj32[i] = jn32[fr]
            rf[i] = fn[fr]; rp[i] = pn[fr]
            frames[i] = fr; impact[i] = imp
            cids.append(e["case_id"]); G[i] = e["G"]; D[i] = e["D"]; Y[i] = e["Y"]
            # self-check SSIM on the impact frame (offset 0 index)
            zi = OFFSETS.index(0)
            ssim_acc.append(wang_ssim(tgt[i, zi], rj[i, zi], L=8.31))
            if (i + 1) % 20 == 0 or i + 1 == n:
                print(f"[decode] {split}: {i+1}/{n}", flush=True)
        outp = args.out / f"{split}.npz"
        np.savez(outp, target_norm=tgt, jepa_norm=rj, jepa_d32_norm=rj32,
                 fukami_norm=rf, pod_norm=rp,
                 frames=frames, impact_frame=impact, case_ids=np.array(cids),
                 G=G, D=D, Y=Y, offsets=np.array(OFFSETS))
        print(f"[decode] {split}: wrote {outp}  JEPA impact-frame Wang-SSIM(L=8.31) mean={np.mean(ssim_acc):.3f} "
              f"(anchor ~0.71 on val)")


if __name__ == "__main__":
    main()
