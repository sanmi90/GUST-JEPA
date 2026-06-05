#!/usr/bin/env python3
"""Sparse wall-pressure sensing animation for the talk (GPU decode, RTX 6000).

Instantaneous state estimation, NO temporal predictor: a per-frame ridge map from
K=16 TCSI wall-pressure taps to the S12_E_d64 latent is fit on the training
encounters, then applied frame-by-frame to a held-out (test_b) encounter. Each
recovered latent is decoded to a mid-plane vorticity field and animated next to
the DNS truth, with the 16 sensor taps marked on the airfoil.

This is the deployment-relevance panel: the state is recoverable from sparse wall
pressure. It is a state estimate, not a forecast (the predictor is not involved).
Output figs/sensing_anim.mp4 (+ poster).
"""
import sys
import json
from pathlib import Path

import numpy as np
import h5py
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, FFMpegWriter  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "session21"))
import figstyle as fs  # noqa: E402
from src.utils.device import require_rtx6000  # noqa: E402
from src.data.omega_pipeline import OmegaPipeline, ssim_data_range  # noqa: E402
from src.models.lap_film_decoder import LapFiLMDecoder  # noqa: E402
import scripts.session20.decode_reconstructions as D  # noqa: E402

OUT_MP4 = Path(__file__).resolve().parent / "figs" / "sensing_anim.mp4"
OUT_POS = Path(__file__).resolve().parent / "figs" / "sensing_poster.png"
LAT = REPO / "outputs/session18/exp_b1/latents_jepa_d64_test1_noBN/train.npz"
PICKS = REPO / "outputs/session21/pressure_v2/sensor_picks_v2.json"
CACHE = REPO.parent / "PREVENT/data/processed/vortex-jepa/v2"
CASE, K = "G-1.50_D0.50_Y-0.20", 0          # held-out representative encounter
FIT_LO, FIT_HI = 20, 100                    # frame window used to fit the map
WIN = 6                                     # causal pressure-history window (frames)
VLIM = 2.0; DT = 0.05; NAVY = "#1F3864"


def feat(pw, sensors):
    """Causal pressure-history features: [p(t-WIN+1), ..., p(t)] at the K taps."""
    s = pw[:, sensors]; T = s.shape[0]
    out = np.zeros((T, WIN * len(sensors)), np.float32)
    for w in range(WIN):
        sh = np.clip(np.arange(T) - (WIN - 1 - w), 0, T - 1)
        out[:, w * len(sensors):(w + 1) * len(sensors)] = s[sh]
    return out


def load_decoder(dev):
    db = torch.load(D.DEC_CKPT, map_location="cpu", weights_only=False); da = db.get("args", {})
    bc = int(da.get("decoder_base_ch", 64)); ch = (bc, bc, int(bc * 0.75), int(bc * 0.5), int(bc * 0.375))
    dec = LapFiLMDecoder(latent_dim=64, channels=ch,
                         resblocks_per_level=int(da.get("decoder_resblocks_per_level", 2)),
                         upsample=da.get("decoder_upsample", "pixelshuffle"),
                         fourier_bands=int(da.get("decoder_fourier_bands") or 4),
                         use_film=bool(da.get("decoder_use_film", True)),
                         airfoil_mask_path=da.get("airfoil_mask_path"))
    dec.load_state_dict(db["decoder_state_dict"], strict=True); dec.eval().to(dev)
    for p in dec.parameters():
        p.requires_grad_(False)

    @torch.no_grad()
    def decode(z_Td):
        z = torch.from_numpy(np.asarray(z_Td, np.float32)).to(dev)
        with torch.autocast("cuda", torch.bfloat16):
            out = dec(z); pred = out["pred"] if isinstance(out, dict) else out
        return pred.float().squeeze(1).cpu().numpy()
    return decode


def pwall(cid, k):
    p = CACHE / cid / f"encounter_{int(k):02d}.h5"
    if not p.exists():
        return None
    with h5py.File(p, "r") as f:
        return np.asarray(f["p_wall"], np.float32)  # (120,192)


def main():
    dev = require_rtx6000(0)
    print("device", dev, torch.cuda.get_device_name(dev.index))
    pipe = OmegaPipeline.from_manifest(D.PIPE); L = ssim_data_range(D.PIPE)
    decode = load_decoder(dev)
    sensors = json.loads(PICKS.read_text())["TCSI"]["16"]
    print("K=16 TCSI taps:", sensors)

    # ---- fit per-frame pressure -> latent ridge on train ----
    lat = np.load(LAT, allow_pickle=True)
    zf = np.asarray(lat["z_full"], np.float32); cids = np.array([str(c) for c in lat["case_id"]])
    encs = np.asarray(lat["encounter_index"], int)
    Xs, Ys = [], []
    for i in range(len(cids)):
        pw = pwall(cids[i], encs[i])
        if pw is None:
            continue
        Xs.append(feat(pw, sensors)[FIT_LO:FIT_HI]); Ys.append(zf[i, FIT_LO:FIT_HI])
        if (i + 1) % 60 == 0:
            print(f"  read {i+1}/{len(cids)} train encounters", flush=True)
    rng = np.random.default_rng(0); order = rng.permutation(len(Xs)); nval = max(1, len(Xs) // 10)
    val, trn = set(order[:nval].tolist()), order[nval:].tolist()
    Xtr = np.concatenate([Xs[i] for i in trn]); Ytr = np.concatenate([Ys[i] for i in trn])
    Xva = np.concatenate([Xs[i] for i in val]); Yva = np.concatenate([Ys[i] for i in val])
    sc = StandardScaler().fit(Xtr)
    reg = MLPRegressor(hidden_layer_sizes=(256, 128), alpha=1e-3, max_iter=400,
                       early_stopping=True, n_iter_no_change=15, random_state=0)
    reg.fit(sc.transform(Xtr), Ytr)
    print(f"fit on {len(Xtr)} frames ({WIN}-frame history x {len(sensors)} taps); "
          f"held-out-ENCOUNTER R^2(latent) = {reg.score(sc.transform(Xva), Yva):.3f}")

    # ---- held-out encounter ----
    with h5py.File(CACHE / CASE / f"encounter_{K:02d}.h5", "r") as f:
        omega_raw = np.asarray(f["omega_z"], np.float32); imp = int(f.attrs.get("impact_frame_estimate", 40))
        pw_test = np.asarray(f["p_wall"], np.float32)
    tgt = pipe.normalize(pipe.preprocess_raw(omega_raw, CASE, K))
    z_hat = reg.predict(sc.transform(feat(pw_test, sensors)))  # (120,64)

    lo, hi = imp - 8, imp + 24
    fr = np.clip(np.arange(lo, hi + 1), 0, 119)
    truth = tgt[fr]; recov = decode(z_hat[fr])
    ssim = np.array([D.wang_ssim(truth[j], recov[j], L=L) for j in range(len(fr))])
    tc = (fr - imp) * DT
    j16 = int(np.where(fr == np.clip(imp + 16, 0, 119))[0][0])
    print(f"{CASE} enc{K} impact={imp}  SSIM(recovered) impact={ssim[np.where(fr==imp)[0][0]]:.2f} H16={ssim[j16]:.2f}")

    # sensor marker positions on the airfoil polygon (193 verts ~ 192 taps)
    AP = fs.airfoil_pixels()
    spos = np.array([AP[min(s, len(AP) - 1)] for s in sensors])

    plt.rcParams.update({"font.size": 12, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.15))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.79, bottom=0.02, wspace=0.06)
    im0 = fs.vort_panel(axes[0], truth[0], vlim=VLIM); axes[0].set_title("DNS truth", fontsize=13, color="#242933", pad=4)
    im1 = fs.vort_panel(axes[1], recov[0], vlim=VLIM)
    axes[1].set_title("recovered from 16 wall-pressure taps", fontsize=13, color="#3C6FB0", pad=4)
    axes[0].scatter(spos[:, 0], spos[:, 1], s=24, c="#1A9850", edgecolors="k", linewidths=0.5, zorder=8)
    ssim_txt = axes[1].text(0.97, 0.05, "", transform=axes[1].transAxes, ha="right", va="bottom",
                            fontsize=10.5, color="#3C6FB0",
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
    ttl = fig.text(0.5, 0.95, "", ha="center", fontsize=12, color=NAVY, fontweight="bold")

    def update(f):
        im0.set_array(truth[f].T); im1.set_array(recov[f].T)
        ssim_txt.set_text(f"SSIM = {ssim[f]:.2f}")
        tag = "before impact" if tc[f] < -1e-9 else "after impact"
        ttl.set_text(f"State estimate from 16 wall-pressure taps:   t/c = {tc[f]:+.2f} {tag}")
        return [im0, im1]

    anim = FuncAnimation(fig, update, frames=len(fr), interval=140, blit=False)
    anim.save(str(OUT_MP4), writer=FFMpegWriter(fps=7, bitrate=3200), dpi=130)
    update(j16); fig.savefig(str(OUT_POS), dpi=130)
    print("wrote", OUT_MP4, "and", OUT_POS)


if __name__ == "__main__":
    main()
