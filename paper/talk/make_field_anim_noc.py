#!/usr/bin/env python3
"""Pixel-field forecast animation for the talk, UNCONDITIONED tf-no-c JEPA (GPU decode).

Copy of make_field_anim.py repointed at the fully unconditioned model
(outputs/session27/JEPA_d64_noc_tf encoder + outputs/session27/decoder_noc_tf
LapFiLM decoder + the model's OWN cond_dim=0 transformer predictor). For the same
representative MODERATE-G held-out encounter as the observable forecast animation,
decode the mid-plane vorticity field every frame from impact through horizon H and
animate three columns:
  DNS truth  /  reconstruction (decode of encoded latent z_dns)  /
  prediction (decode of the OWN-predictor rolled latent z_markov).

Differences vs the production script (which is NOT modified):
  - ENC_CKPT -> unconditioned encoder; DEC_CKPT -> decoder_noc_tf (local overrides;
    module D is NOT mutated on disk).
  - the rolled latent is the model's OWN unconditioned transformer predictor rolled
    LIVE (markov-only, cond = torch.zeros(1, 0), fed RAW BatchNorm-output latents)
    seeded at the impact-frame latent; reproduces rollouts_noc_tf/test_b.npz z_markov.
  - output goes to outputs/session27/anim_noc_tf/ (NEVER paper/talk/figs/).
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
sys.path.insert(0, str(REPO / "scripts" / "session17"))
import figstyle as fs  # noqa: E402
import exp2_rollout_decode_metrics as E2  # noqa: E402
from src.utils.device import require_rtx6000  # noqa: E402
from src.data.omega_pipeline import OmegaPipeline, ssim_data_range  # noqa: E402
from src.models.encoder import HybridCNNViTEncoder  # noqa: E402
from src.models.lap_film_decoder import LapFiLMDecoder  # noqa: E402
from src.models.predictor import AutoregressivePredictor  # noqa: E402
import scripts.session20.decode_reconstructions as D  # noqa: E402

# UNCONDITIONED tf-no-c artefacts (local overrides; module D is NOT modified).
ENC_CKPT = REPO / "outputs/session27/JEPA_d64_noc_tf/checkpoint_iter020000.pt"
DEC_CKPT = REPO / "outputs/session27/decoder_noc_tf/decoder_iter030000.pt"
ROLL = REPO / "outputs/session27/rollouts_noc_tf/test_b.npz"
OUT_DIR = REPO / "outputs/session27/anim_noc_tf"
OUT_MP4 = OUT_DIR / "field_anim.mp4"
OUT_POS = OUT_DIR / "field_poster.png"

# Same representative MODERATE-G held-out encounter as the forecast animation.
CASE, K = "G+1.50_D1.50_Y+0.10", 0
H = 24; DT = 0.05; VLIM = 2.0
NAVY = "#1F3864"


def load_enc_dec(dev):
    blob = torch.load(ENC_CKPT, map_location="cpu", weights_only=False); a = blob["args"]
    enc = HybridCNNViTEncoder(latent_dim=int(a["d"]), projection_norm=a.get("projection_norm", "batchnorm"))
    enc.load_state_dict({k.removeprefix("encoder."): v for k, v in blob["jepa_state_dict"].items()
                         if k.startswith("encoder.")}, strict=False)
    enc.eval().to(dev)
    db = torch.load(DEC_CKPT, map_location="cpu", weights_only=False); da = db.get("args", {})
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


def load_own_predictor(dev):
    """The model's OWN unconditioned transformer predictor (cond_dim = 0)."""
    ck = torch.load(ENC_CKPT, map_location="cpu", weights_only=False); a = ck["args"]
    assert int(a["predictor_cond_dim"]) == 0 and a["predictor_type"] == "transformer"
    psd = {k[len("predictor."):]: v for k, v in ck["jepa_state_dict"].items()
           if k.startswith("predictor.")}
    p = AutoregressivePredictor(latent_dim=int(a["d"]), cond_dim=0, max_seq_len=int(a["T"]))
    p.load_state_dict(psd, strict=True); p.eval().to(dev)
    assert p.cond_dim == 0
    return p


def main():
    dev = require_rtx6000(0)
    print("device", dev, torch.cuda.get_device_name(dev.index))
    pipe = OmegaPipeline.from_manifest(D.PIPE); L = ssim_data_range(D.PIPE)
    decode = load_enc_dec(dev)
    pred = load_own_predictor(dev)

    p = REPO.parent / f"PREVENT/data/processed/vortex-jepa/v2/{CASE}/encounter_{K:02d}.h5"
    with h5py.File(p, "r") as f:
        omega_raw = np.asarray(f["omega_z"], np.float32); imp = int(f.attrs.get("impact_frame_estimate", 40))
    tgt = pipe.normalize(pipe.preprocess_raw(omega_raw, CASE, K))  # (120,192,96)

    rb = np.load(ROLL, allow_pickle=True)
    rc = np.array([str(c) for c in rb["case_ids"]]); re_ = np.asarray(rb["encounter_indices"], int)
    ej = int(np.where((rc == CASE) & (re_ == K))[0][0])
    G, Dd, Y = float(rb["G"][ej]), float(rb["D"][ej]), float(rb["Y"][ej])
    zdns = np.asarray(rb["z_dns"], np.float32)[ej]

    # Roll the OWN unconditioned predictor LIVE from the impact-frame latent.
    cond0 = torch.zeros(1, 0, device=dev)
    steps = 119 - imp
    z_init = torch.from_numpy(zdns[imp:imp + 1]).to(dev)
    zm_roll = E2.rollout_markov_only(pred, z_init, cond0, steps, dev)[0].float().cpu().numpy()
    zmk = zdns.copy(); zmk[imp + 1:120] = zm_roll[1:steps + 1]
    chk = float(np.abs(zmk - np.asarray(rb["z_markov"], np.float32)[ej]).max())

    fr = np.clip(np.arange(0, H + 1) + imp, 0, 119)
    truth = tgt[fr]
    recon = decode(zdns[fr])
    predf = decode(zmk[fr])
    ssim_p = np.array([D.wang_ssim(truth[j], predf[j], L=L) for j in range(len(fr))])
    print(f"{CASE} enc{K} (G={G:+.1f},D={Dd:.1f},Y={Y:+.1f}) impact={imp}  "
          f"live-roll vs precomputed z_markov max|diff|={chk:.2e} (own cond_dim=0 predictor)\n"
          f"  SSIM(pred) impact={ssim_p[0]:.2f} H8={ssim_p[8]:.2f} H16={ssim_p[16]:.2f} H24={ssim_p[24]:.2f}")

    plt.rcParams.update({"font.size": 12, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 2.95))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.80, bottom=0.02, wspace=0.05)
    cols = [("DNS truth", "#242933"), ("reconstruction (encoded latent)", "#2E7D4F"),
            ("prediction (JEPA rollout)", "#C0520F")]
    ims = []
    for ax, field0, (lab, col) in zip(axes, (truth[0], recon[0], predf[0]), cols):
        ims.append(fs.vort_panel(ax, field0, vlim=VLIM))
        ax.set_title(lab, fontsize=12.5, color=col, pad=4)
    ssim_txt = axes[2].text(0.97, 0.05, "", transform=axes[2].transAxes, ha="right", va="bottom",
                            fontsize=10.5, color="#C0520F",
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
    ttl = fig.text(0.5, 0.955, "", ha="center", fontsize=13, color=NAVY, fontweight="bold")
    stacks = [truth, recon, predf]

    def update(f):
        for im, st in zip(ims, stacks):
            im.set_array(st[f].T)
        ssim_txt.set_text(f"SSIM = {ssim_p[f]:.2f}")
        ttl.set_text(f"Mid-plane vorticity (unconditioned), gust (G={G:+.1f}, D={Dd:.1f}, Y={Y:+.1f}):   "
                     f"t/c = {f * DT:.2f} after impact")
        return ims

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    anim = FuncAnimation(fig, update, frames=H + 1, interval=140, blit=False)
    anim.save(str(OUT_MP4), writer=FFMpegWriter(fps=7, bitrate=3200), dpi=130)
    update(16); fig.savefig(str(OUT_POS), dpi=130)
    print("wrote", OUT_MP4, "and", OUT_POS, f"({H + 1} frames)")


if __name__ == "__main__":
    main()
