#!/usr/bin/env python3
"""
session25_decode_wake_mode.py
=============================

Koshikawa-Fukami-style visualisation, adapted to our setup: decode the JEPA
wake-forecast latent DIRECTION into vorticity space through the (separately
trained, frozen) visualisation decoder, to see what physical structure that
direction corresponds to.

Method. The wake-forecast direction u is the ridge direction of the future wake
enstrophy on the d=64 latent (the basis-free direction from the cross-encoder
analysis). We perturb a reference impact-frame latent z0 along u and decode:
D(z0 - e u), D(z0), D(z0 + e u). The morphing field, and the difference
D(z0+e u) - D(z0-e u), is the vorticity structure the wake-forecast direction
controls. e is set to a few standard deviations of the latent's spread along u.

CAVEAT (honest): this visualisation decoder is blurry by design (a predictive
latent is not trained to reconstruct), so expect a coarse structure, not a sharp
field. It is a diagnostic on the frozen encoder, not part of any objective.

Hardware: RTX 6000 only (require_rtx6000), per repo rule; never the L40S pick_device.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.linear_model import Ridge

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "session21"))

from src.models.encoder import HybridCNNViTEncoder
from src.models.lap_film_decoder import LapFiLMDecoder
from src.utils.device import require_rtx6000

ENC = REPO / "outputs/runs/session12/S12_E_d64/encoder/checkpoint_iter020000.pt"
DEC = REPO / "outputs/runs/session12/S12_E_d64/encoder/decoder_specloss_recipe/decoder_iter030000.pt"
PF = "outputs/session16/exp2/per_frame_targets/{}.npz"
H = 16


def load_decoder(device):
    eb = torch.load(ENC, map_location="cpu", weights_only=False)
    d = int(eb["args"]["d"])
    db = torch.load(DEC, map_location="cpu", weights_only=False)
    da = db.get("args", {})
    bc = int(da.get("decoder_base_ch", 64))
    channels = (bc, bc, int(bc * 0.75), int(bc * 0.5), int(bc * 0.375))
    dec = LapFiLMDecoder(latent_dim=d, channels=channels,
                         resblocks_per_level=int(da.get("decoder_resblocks_per_level", 2)),
                         upsample=da.get("decoder_upsample", "pixelshuffle"),
                         fourier_bands=int(da.get("decoder_fourier_bands") or 4),
                         use_film=bool(da.get("decoder_use_film", True)),
                         airfoil_mask_path=da.get("airfoil_mask_path"))
    dec.load_state_dict(db["decoder_state_dict"], strict=True)
    dec.eval().to(device)
    for p in dec.parameters():
        p.requires_grad_(False)
    return dec, d


@torch.no_grad()
def decode(dec, z_vec, device, shape):
    z = torch.from_numpy(np.asarray(z_vec, np.float32)).reshape(shape).to(device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        out = dec(z)
        pred = out["pred"] if isinstance(out, dict) else out
    return pred.float().squeeze().cpu().numpy()


def main():
    device = require_rtx6000(gpu_index=0)
    print(f"[device] {device} {torch.cuda.get_device_name(device.index)}")
    dec, d = load_decoder(device)

    # forecast direction u from train pooled post-impact latents
    tr = np.load(PF.format("train"), allow_pickle=True)
    z = np.asarray(tr["z_full"], float); imp = np.asarray(tr["impact_frame"], int)
    we = np.asarray(tr["wake_enstrophy"], float)
    n, T, _ = z.shape
    rows = [(i, t) for i in range(n) for t in range(int(imp[i]), T - H)]
    ii = np.array([r[0] for r in rows]); tt = np.array([r[1] for r in rows])
    X = z[ii, tt, :]; y = we[ii, np.clip(tt + H, 0, T - 1)]
    yr = rankdata(y); yr = (yr - yr.mean()) / yr.std()
    u = Ridge(alpha=1.0).fit(X, yr).coef_; u = u / np.linalg.norm(u)
    proj = X @ u; eps = 2.5 * float(proj.std())
    print(f"[dir] |u|=1, latent spread along u std={proj.std():.3f}, eps={eps:.3f}")

    # reference: mean test_b impact-frame latent (a typical impact state)
    te = np.load(PF.format("test_b"), allow_pickle=True)
    zte = np.asarray(te["z_full"], float); impte = np.asarray(te["impact_frame"], int)
    z0 = np.array([zte[i, impte[i], :] for i in range(len(impte))]).mean(0)

    # figure out decoder input shape by probing with the encoder's latent rank
    for shape in [(1, d), (1, 1, d), (d,)]:
        try:
            f = decode(dec, z0, device, shape)
            print(f"[shape] decode input {shape} -> field {f.shape}, "
                  f"range [{f.min():.2f},{f.max():.2f}]")
            good_shape = shape; ref = f
            break
        except Exception as e:
            print(f"[shape] {shape} failed: {str(e)[:80]}")
    else:
        print("[fatal] no decode shape worked"); return

    fm = decode(dec, z0 + eps * u, device, good_shape)
    fp = decode(dec, z0 - eps * u, device, good_shape)
    diff = fm - fp
    print(f"[fields] ref range [{ref.min():.2f},{ref.max():.2f}], "
          f"mode-diff range [{diff.min():.2f},{diff.max():.2f}]")
    np.savez("outputs_causal/jepa_modes/decoded_wake_mode.npz",
             ref=ref, plus=fm, minus=fp, diff=diff, u=u, eps=eps)

    # render
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import figstyle; figstyle.use_style(); W = figstyle.TEXTWIDTH_IN
        cmap = figstyle.VORT_CMAP
    except Exception:
        W, cmap = 4.98, "RdBu_r"
    fig, axes = plt.subplots(1, 3, figsize=(W, 0.30 * W))
    panels = [(fp, r"$z_0 - \epsilon u$"), (ref, r"$z_0$ (reference)"), (fm, r"$z_0 + \epsilon u$")]
    vlim = 3.0
    for ax, (fld, ttl) in zip(axes, panels):
        ax.imshow(fld.T, cmap=cmap, vmin=-vlim, vmax=vlim, origin="lower", aspect="auto")
        ax.set_title(ttl, fontsize=7.5); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("decoded JEPA wake-forecast direction (blurry visualisation decoder)", fontsize=8)
    fig.tight_layout(pad=0.3)
    out = REPO / "outputs_causal/jepa_modes/fig_decoded_wake_mode.pdf"
    fig.savefig(out, bbox_inches="tight"); print(f"[fig] wrote {out}")


if __name__ == "__main__":
    main()
