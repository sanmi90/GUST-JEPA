"""What flow information is recovered from sparse wall pressure.

For a few held-out encounters we estimate the impact-frame predictive latent from
the TCSI wall-pressure taps (kernel ridge), decode it with the production
visualisation decoder, and compare against the simulation and the oracle decode
(decode of the simulation-encoded latent). This shows, in physical space, what the
sparse wall sensors actually recover: the leading-edge vortex and shear layer from
K=8 taps, coarsening at K=2.

Runs the decoder on the RTX 6000 (NOT the L40S, which a collaborator is using).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "session20"))
import figstyle as fs  # noqa: E402
from src.utils.device import require_rtx6000  # noqa: E402
from src.models.lap_film_decoder import LapFiLMDecoder  # noqa: E402
from decode_reconstructions import ENC_CKPT, DEC_CKPT  # noqa: E402
from _oneoff_baseline_pressure_obs import gather_pressure_and_z  # noqa: E402

LAT = REPO / "outputs/session18/exp_b1/latents_jepa_d64_test1_noBN"
DECNPZ = np.load(REPO / "outputs/session20/decoded/test_b.npz", allow_pickle=True)
PICKS = json.load(open(REPO / "outputs/session21/pressure_v2/sensor_picks_v2.json"))["TCSI"]
OUT_PDF = REPO / "paper/sections/figures/results/figG_flow_recovery.pdf"
OUT_PNG = REPO / "outputs/session21/figs/figG_flow_recovery.png"

# representative held-out encounters (by case id): strong-, strong+, moderate
CASES = ["G-3.00_D1.50_Y-0.10", "G+2.00_D0.50_Y+0.10", "G-1.50_D0.50_Y-0.20"]


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
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = dec(z)
        pred = out["pred"] if isinstance(out, dict) else out
        return pred.float().squeeze(1).cpu().numpy()
    return decode


def krr(Xtr, ztr):
    sx = StandardScaler().fit(Xtr); sy = StandardScaler().fit(ztr)
    m = KernelRidge(alpha=0.1, kernel="rbf", gamma=0.01)
    m.fit(sx.transform(Xtr), sy.transform(ztr))
    return lambda X: sy.inverse_transform(m.predict(sx.transform(X)))


def main():
    fs.use_style()
    device = require_rtx6000(gpu_index=0)
    print("device:", torch.cuda.get_device_name(device.index))
    decode = load_decoder(device)

    tr = np.load(LAT / "train.npz", allow_pickle=True)
    te = np.load(LAT / "test_b.npz", allow_pickle=True)
    Xtr, ztr, _ = gather_pressure_and_z(tr, "train")
    Xte, zte, _ = gather_pressure_and_z(te, "test_b")
    Xtr, Xte = Xtr.astype(np.float64), Xte.astype(np.float64)
    tcid = np.array([str(c) for c in te["case_id"]])
    dcid = np.array([str(c) for c in DECNPZ["case_ids"]])

    est = {K: krr(Xtr[:, :, PICKS[str(K)]].reshape(len(Xtr), -1), ztr) for K in (8, 2)}

    fig, axes = plt.subplots(len(CASES), 4, figsize=fs.figure_size(1.0, aspect=0.62))
    cols = ["simulation", "oracle decode", "from 8 taps", "from 2 taps"]
    for r, case in enumerate(CASES):
        i = int(np.where(tcid == case)[0][0])
        di = int(np.where(dcid == case)[0][0])
        truth = DECNPZ["target_norm"][di, 1]                       # impact field
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
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF); fig.savefig(OUT_PNG, dpi=200)
    print(f"wrote {OUT_PDF.name}")


if __name__ == "__main__":
    main()
