"""Render regAE's OWN-decoder reconstruction (qualitative) on a v2.1 test_b
encounter. Decodes the saved encoded latent z_full through the regAE wrapper's
own (co-trained) decoder -- no re-encoding, no SSIM convention, no v1-split
helper -- and plots it against the DNS truth at the impact frame.

Usage: python scripts/session29/m_regae_own_decode.py --device cuda:3
"""
from __future__ import annotations
import argparse, sys, os
from pathlib import Path
import numpy as np
import torch
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts" / "session28"))
import decode_operating_points as DOP  # noqa: E402  (has load_pipeline)
from src.baselines.fukami_ae import FukamiAEWrapper  # noqa: E402
from src.models.observable_head import WakeObservableHead  # noqa: E402
from src.data.wake_observables import mode_output_dim  # noqa: E402

CKPT = REPO / "outputs/runs/session29_8/regae/regae_cnn_vit_d64_s0/checkpoint_iter020000.pt"
LAT = REPO / "outputs/session28/latents/regae/cnn_vit_s0/test_b.npz"
CACHE = Path(os.environ.get("VORTEX_JEPA_CACHE", str(Path(os.environ["PREVENT_ROOT"]) / "data/processed/vortex-jepa")))


def build_wrapper(device, pipe):
    b = torch.load(CKPT, map_location="cpu", weights_only=False)
    a = b["args"]; a = a if isinstance(a, dict) else vars(a)
    wd = mode_output_dim(a.get("wake_observable_type", "patch_signed_spectrum"))
    wake_head = WakeObservableHead(latent_dim=int(a["d"]), out_dim=wd,
                                   hidden_dim=int(a.get("wake_head_hidden", 128) or 128))
    w = FukamiAEWrapper(
        latent_dim=int(a["d"]), n_deltas=len(a.get("observable_head_deltas", [0])),
        omega_pipeline=pipe, recon_loss_type=a.get("recon_loss_type", "mse"),
        activation=a.get("activation", "relu"),
        use_conv_norm=not a.get("no_conv_norm", False),
        wake_observable_head=wake_head, wake_observable_weight=float(a.get("lambda_wake", 1.0)),
        wake_loss_kind=a.get("wake_loss", "smooth_l1"), wake_loss_beta=float(a.get("wake_loss_beta", 0.5)),
        encoder_kind=a.get("encoder", "cnn_vit"), lambda_sigreg=float(a.get("lambda_sigreg", 0.01)),
    ).to(device)
    missing, unexpected = w.load_state_dict(b["wrapper_state_dict"], strict=False)
    print(f"[regae-decode] loaded (missing={len(missing)} unexpected={len(unexpected)})")
    if missing:
        print("  missing sample:", missing[:4])
    w.eval()
    return w


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:3"); a = ap.parse_args()
    dev = torch.device(a.device)
    name = torch.cuda.get_device_name(dev.index or 0)
    assert "RTX" in name and "6000" in name, f"refusing {name}"
    pipe, _ = DOP.load_pipeline()
    w = build_wrapper(dev, pipe)

    d = np.load(LAT, allow_pickle=True)
    z = d["z_full"].astype(np.float32)
    cid = d["case_ids"] if "case_ids" in d.files else d["case_id"]
    enc = d["encounter_indices"] if "encounter_indices" in d.files else d["encounter_index"]
    imp = d["impact_frame"].astype(int)
    i = 0  # first held-out encounter
    ti = int(imp[i])
    zt = torch.from_numpy(z[i, ti:ti + 1]).to(dev)            # (1, d)
    xhat = w.decode(zt) if hasattr(w, "decode") else pipe.unnormalize(w.decoder(zt))
    xhat = xhat.float().squeeze().cpu().numpy()               # (H, W) raw
    # truth from cache
    with h5py.File(CACHE / "v2p1" / str(cid[i]) / f"encounter_{int(enc[i]):02d}.h5", "r") as f:
        truth = f["omega_z"][ti].astype(np.float32)
    print(f"[regae-decode] case={cid[i]} enc={int(enc[i])} impact={ti}  "
          f"xhat[min,max]=[{xhat.min():.1f},{xhat.max():.1f}] truth[min,max]=[{truth.min():.1f},{truth.max():.1f}]")
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.0))
    for axi, (img, t) in zip(ax, [(truth, "DNS truth"), (xhat, "regAE own decoder")]):
        axi.imshow(img.T, origin="lower", cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
        axi.set_title(t, fontsize=9); axi.set_xticks([]); axi.set_yticks([])
    fig.suptitle(f"regAE own-decoder reconstruction (raw $\\omega_z$, $\\pm 3$), {cid[i]} impact", fontsize=8)
    out = REPO / "outputs/session29/regae_own_decode.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[regae-decode] wrote {out}")


if __name__ == "__main__":
    main()
