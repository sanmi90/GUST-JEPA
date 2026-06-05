#!/usr/bin/env python3
"""Pixel-field forecast animation for the talk (GPU decode, RTX 6000).

For the representative low-error encounter (same one as the observable forecast
animation), decode the mid-plane vorticity field every frame from impact through
horizon H and animate three columns:
  DNS truth  /  reconstruction (decode of encoded latent z_dns)  /
  prediction (decode of the rolled latent z_markov).

All three live in the S12_E_d64 latent space (the production encoder); the
on-manifold rollout is jepa_d64_test1_noBN. The visualisation decoder is the
frozen LapFiLM decoder (never part of the JEPA loss). Output figs/field_anim.mp4.
"""
import sys
from pathlib import Path

import numpy as np
import h5py
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, FFMpegWriter  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "session21"))
import figstyle as fs  # noqa: E402
from src.utils.device import require_rtx6000  # noqa: E402
from src.data.omega_pipeline import OmegaPipeline, ssim_data_range  # noqa: E402
from src.models.encoder import HybridCNNViTEncoder  # noqa: E402
from src.models.lap_film_decoder import LapFiLMDecoder  # noqa: E402
import scripts.session20.decode_reconstructions as D  # noqa: E402

OUT_MP4 = Path(__file__).resolve().parent / "figs" / "field_anim.mp4"
OUT_POS = Path(__file__).resolve().parent / "figs" / "field_poster.png"
CASE, K = "G-1.50_D0.50_Y-0.20", 0      # representative low-error encounter (see forecast anim)
H = 24; DT = 0.05; VLIM = 2.0
NAVY = "#1F3864"


def load_enc_dec(dev):
    blob = torch.load(D.ENC_CKPT, map_location="cpu", weights_only=False); a = blob["args"]
    enc = HybridCNNViTEncoder(latent_dim=int(a["d"]), projection_norm=a.get("projection_norm", "batchnorm"))
    enc.load_state_dict({k.removeprefix("encoder."): v for k, v in blob["jepa_state_dict"].items()
                         if k.startswith("encoder.")}, strict=False)
    enc.eval().to(dev)
    db = torch.load(D.DEC_CKPT, map_location="cpu", weights_only=False); da = db.get("args", {})
    bc = int(da.get("decoder_base_ch", 64)); ch = (bc, bc, int(bc * 0.75), int(bc * 0.5), int(bc * 0.375))
    dec = LapFiLMDecoder(latent_dim=int(a["d"]), channels=ch,
                         resblocks_per_level=int(da.get("decoder_resblocks_per_level", 2)),
                         upsample=da.get("decoder_upsample", "pixelshuffle"),
                         fourier_bands=int(da.get("decoder_fourier_bands") or 4),
                         use_film=bool(da.get("decoder_use_film", True)),
                         airfoil_mask_path=da.get("airfoil_mask_path"))
    dec.load_state_dict(db["decoder_state_dict"], strict=True); dec.eval().to(dev)
    for p in list(enc.parameters()) + list(dec.parameters()):
        p.requires_grad_(False)

    @torch.no_grad()
    def decode(z_Td):
        z = torch.from_numpy(np.asarray(z_Td, np.float32)).to(dev)
        with torch.autocast("cuda", torch.bfloat16):
            out = dec(z); pred = out["pred"] if isinstance(out, dict) else out
        return pred.float().squeeze(1).cpu().numpy()
    return decode


def main():
    dev = require_rtx6000(0)
    print("device", dev, torch.cuda.get_device_name(dev.index))
    pipe = OmegaPipeline.from_manifest(D.PIPE); L = ssim_data_range(D.PIPE)
    decode = load_enc_dec(dev)

    p = REPO.parent / f"PREVENT/data/processed/vortex-jepa/v2/{CASE}/encounter_{K:02d}.h5"
    with h5py.File(p, "r") as f:
        omega_raw = np.asarray(f["omega_z"], np.float32); imp = int(f.attrs.get("impact_frame_estimate", 40))
    tgt = pipe.normalize(pipe.preprocess_raw(omega_raw, CASE, K))  # (120,192,96)

    rb = np.load(REPO / "outputs/session18/exp_b1/rollouts_jepa_d64_test1_noBN/test_b.npz", allow_pickle=True)
    rc = np.array([str(c) for c in rb["case_ids"]]); re_ = np.asarray(rb["encounter_indices"], int)
    ej = int(np.where((rc == CASE) & (re_ == K))[0][0])
    G, Dd, Y = float(rb["G"][ej]), float(rb["D"][ej]), float(rb["Y"][ej])
    fr = np.clip(np.arange(0, H + 1) + imp, 0, 119)
    truth = tgt[fr]
    recon = decode(np.asarray(rb["z_dns"], np.float32)[ej][fr])
    pred = decode(np.asarray(rb["z_markov"], np.float32)[ej][fr])
    ssim_p = np.array([D.wang_ssim(truth[j], pred[j], L=L) for j in range(len(fr))])
    print(f"{CASE} enc{K} (G={G:+.1f},D={Dd:.1f},Y={Y:+.1f}) impact={imp}  "
          f"SSIM(pred) impact={ssim_p[0]:.2f} H8={ssim_p[8]:.2f} H16={ssim_p[16]:.2f} H24={ssim_p[24]:.2f}")

    plt.rcParams.update({"font.size": 12, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 2.95))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.80, bottom=0.02, wspace=0.05)
    cols = [("DNS truth", "#242933"), ("reconstruction (encoded latent)", "#2E7D4F"),
            ("prediction (JEPA rollout)", "#C0520F")]
    ims = []
    for ax, field0, (lab, col) in zip(axes, (truth[0], recon[0], pred[0]), cols):
        ims.append(fs.vort_panel(ax, field0, vlim=VLIM))
        ax.set_title(lab, fontsize=12.5, color=col, pad=4)
    ssim_txt = axes[2].text(0.97, 0.05, "", transform=axes[2].transAxes, ha="right", va="bottom",
                            fontsize=10.5, color="#C0520F",
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
    ttl = fig.text(0.5, 0.955, "", ha="center", fontsize=13, color=NAVY, fontweight="bold")
    stacks = [truth, recon, pred]

    def update(f):
        for im, st in zip(ims, stacks):
            im.set_array(st[f].T)
        ssim_txt.set_text(f"SSIM = {ssim_p[f]:.2f}")
        ttl.set_text(f"Mid-plane vorticity, gust (G={G:+.1f}, D={Dd:.1f}, Y={Y:+.1f}):   "
                     f"t/c = {f * DT:.2f} after impact")
        return ims

    anim = FuncAnimation(fig, update, frames=H + 1, interval=140, blit=False)
    anim.save(str(OUT_MP4), writer=FFMpegWriter(fps=7, bitrate=3200), dpi=130)
    update(16); fig.savefig(str(OUT_POS), dpi=130)
    print("wrote", OUT_MP4, "and", OUT_POS)


if __name__ == "__main__":
    main()
