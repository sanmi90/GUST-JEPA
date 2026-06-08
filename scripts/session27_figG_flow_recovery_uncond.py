"""UNCONDITIONED flow recovery from sparse wall pressure (Session 27 copy of
scripts/session21/figG_flow_recovery.py).

For a few held-out encounters we estimate the impact-frame UNCONDITIONED (no-c)
latent from the TCSI wall-pressure taps (kernel ridge refit on the unconditioned
z), decode it with the UNCONDITIONED visualisation decoder, and compare against
the simulation and the oracle decode (decode of the simulation-encoded
unconditioned latent). K=8 and K=2 taps.

Repointing vs production:
  * latents       -> outputs/session27/latents_own_jepa_noc_tf/{train,test_b}.npz
  * sensor picks  -> outputs/session27/pressure_noc_tf/sensor_picks_v2.json (refit)
  * decoded npz   -> outputs/session27/decoded_noc_tf/test_b.npz (simulation target)
  * encoder/dec   -> the no-c JEPA encoder + its dedicated LapFiLM decoder
  * pressure win  -> v2 cache
Output: outputs/session27/figs_uncond/figG_flow_recovery.pdf (+ .png)

Uses pick_device() (auto-selects an L40S), which is acceptable per the task brief.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "session21"))
import figstyle as fs  # noqa: E402
from src.models.encoder import HybridCNNViTEncoder  # noqa: E402
from src.models.lap_film_decoder import LapFiLMDecoder  # noqa: E402

# --- UNCONDITIONED checkpoints (read-only) ---
ENC_CKPT = REPO / "outputs/session27/JEPA_d64_noc_tf/checkpoint_iter020000.pt"
DEC_CKPT = REPO / "outputs/session27/decoder_noc_tf/decoder_iter030000.pt"
LAT = REPO / "outputs/session27/latents_own_jepa_noc_tf"
DECNPZ = np.load(REPO / "outputs/session27/decoded_noc_tf/test_b.npz", allow_pickle=True)
PICKS = json.load(open(REPO / "outputs/session27/pressure_noc_tf/sensor_picks_v2.json"))["TCSI"]
OUT_PDF = REPO / "outputs/session27/figs_uncond/figG_flow_recovery.pdf"
OUT_PNG = REPO / "outputs/session27/figs_uncond/figG_flow_recovery.png"

PREVENT = Path(os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT")))
CACHE = Path(os.environ.get("VORTEX_JEPA_CACHE", str(PREVENT / "data/processed/vortex-jepa"))) / "v2"
WINDOW = 30
N_PRESSURE_FULL = 192
IMPACT_OFFSET_IDX = int(list(DECNPZ["offsets"]).index(0))  # offset 0 = impact frame

# representative held-out encounters (by case id): strong-, strong+, moderate
CASES = ["G-3.00_D1.50_Y-0.10", "G+2.00_D0.50_Y+0.10", "G-1.50_D0.50_Y-0.20"]


def pick_device() -> torch.device:
    """Prefer a non-RTX CUDA card (L40S) to leave the RTX 6000s free."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    for i in range(torch.cuda.device_count()):
        if "RTX" not in torch.cuda.get_device_name(i):
            return torch.device(f"cuda:{i}")
    return torch.device("cuda:0")


def load_decoder(device):
    blob = torch.load(ENC_CKPT, map_location="cpu", weights_only=False)
    d = int(blob["args"]["d"])
    db = torch.load(DEC_CKPT, map_location="cpu", weights_only=False)
    da = db.get("args", {})
    bc = int(da.get("decoder_base_ch", 64))
    dec = LapFiLMDecoder(
        latent_dim=d, channels=(bc, bc, int(bc * 0.75), int(bc * 0.5), int(bc * 0.375)),
        resblocks_per_level=int(da.get("decoder_resblocks_per_level", 2)),
        upsample=da.get("decoder_upsample", "pixelshuffle"),
        fourier_bands=int(da.get("decoder_fourier_bands") or 4),
        use_film=bool(da.get("decoder_use_film", True)),
        airfoil_mask_path=da.get("airfoil_mask_path"))
    dec.load_state_dict(db["decoder_state_dict"], strict=True)
    dec.eval().to(device)

    @torch.no_grad()
    def decode(z_np):                                  # (n, d) -> (n, 192, 96)
        z = torch.from_numpy(z_np.astype(np.float32)).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            out = dec(z)
        pred = out["pred"] if isinstance(out, dict) else out
        return pred.float().squeeze(1).cpu().numpy()
    return decode


def load_pressure_window(case_id, k, impact_frame):
    p = CACHE / case_id / f"encounter_{int(k):02d}.h5"
    with h5py.File(p, "r") as f:
        pw = np.asarray(f["p_wall"], dtype=np.float32)
    t0 = max(0, impact_frame - WINDOW)
    w = pw[t0:impact_frame]
    if w.shape[0] < WINDOW:
        w = np.concatenate([np.zeros((WINDOW - w.shape[0], N_PRESSURE_FULL), np.float32), w], 0)
    return w


def gather(npz):
    cids = npz["case_ids"].astype(str)
    eis = npz["encounter_indices"].astype(int)
    imp = npz["impact_frame"].astype(int)
    z_full = npz["z_full"]
    z = np.stack([z_full[i, int(imp[i])] for i in range(len(cids))], 0).astype(np.float64)
    X = np.zeros((len(cids), WINDOW, N_PRESSURE_FULL), np.float64)
    for i in range(len(cids)):
        X[i] = load_pressure_window(cids[i], int(eis[i]), int(imp[i]))
    return X, z, cids


def krr(Xtr, ztr):
    sx = StandardScaler().fit(Xtr); sy = StandardScaler().fit(ztr)
    m = KernelRidge(alpha=0.1, kernel="rbf", gamma=0.01)
    m.fit(sx.transform(Xtr), sy.transform(ztr))
    return lambda X: sy.inverse_transform(m.predict(sx.transform(X)))


def main():
    fs.use_style()
    device = pick_device()
    print("device:", torch.cuda.get_device_name(device.index) if device.type == "cuda" else "cpu")
    decode = load_decoder(device)

    tr = np.load(LAT / "train.npz", allow_pickle=True)
    te = np.load(LAT / "test_b.npz", allow_pickle=True)
    Xtr, ztr, _ = gather(tr)
    Xte, zte, tcid = gather(te)
    dcid = np.array([str(c) for c in DECNPZ["case_ids"]])

    est = {K: krr(Xtr[:, :, PICKS[str(K)]].reshape(len(Xtr), -1), ztr) for K in (8, 2)}

    fig, axes = plt.subplots(len(CASES), 4, figsize=fs.figure_size(1.0, aspect=0.62))
    cols = ["simulation", "best-case decode", "from 8 taps", "from 2 taps"]
    for r, case in enumerate(CASES):
        i = int(np.where(tcid == case)[0][0])
        di = int(np.where(dcid == case)[0][0])
        truth = DECNPZ["target_norm"][di, IMPACT_OFFSET_IDX]            # impact field
        z_oracle = zte[i:i + 1]
        z8 = est[8](Xte[i:i + 1, :, PICKS["8"]].reshape(1, -1))
        z2 = est[2](Xte[i:i + 1, :, PICKS["2"]].reshape(1, -1))
        fields = [truth, decode(z_oracle)[0], decode(z8)[0], decode(z2)[0]]
        for c, fld in enumerate(fields):
            ax = axes[r, c]
            im = fs.vort_panel(ax, fld)
            if r == 0:
                ax.set_title(cols[c], fontsize=7.5)
        g, d, y = float(te["G"][i]), float(te["D"][i]), float(te["Y"][i])
        axes[r, 0].text(-0.04, 0.5, f"$G{g:+.1f}\\,D{d:.1f}\\,Y{y:+.1f}$",
                        transform=axes[r, 0].transAxes, rotation=90, va="center",
                        ha="right", fontsize=6)

    fig.subplots_adjust(left=0.07, right=0.9, top=0.93, bottom=0.02, wspace=0.05, hspace=0.08)
    cax = fig.add_axes([0.915, 0.25, 0.012, 0.5])
    fig.colorbar(im, cax=cax, label=r"$\omega_z$ (norm.)")
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF); fig.savefig(OUT_PNG, dpi=200)
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
