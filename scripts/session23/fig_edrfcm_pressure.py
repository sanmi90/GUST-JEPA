"""EDRFCM abstract Figure 2: flow recovered from sparse wall pressure.

Reuses the exact decode + KRR-from-TCSI-taps logic of
scripts/session21/figG_flow_recovery.py, re-sized for the EDRFCM two-column
layout. Rows: three held-out encounters labelled by (G,D,Y). Columns: the
simulation, the oracle decode (decoder on the simulation-encoded latent), and the
decode of the predictive latent estimated from K=8 and K=2 TCSI pressure taps.
Runs the visualisation decoder on the RTX 6000.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/carlos/GUST-JEPA")
for p in ("scripts/session21", "", "scripts", "scripts/session20"):
    sys.path.insert(0, str(REPO / p))

import matplotlib.pyplot as plt  # noqa: E402
import figstyle as fs  # noqa: E402
import figG_flow_recovery as fg  # noqa: E402

OUT = REPO / "paper/edrfcm2026/Figures/pressure_flow.pdf"


def main() -> None:
    fs.use_style()
    device = fg.require_rtx6000(gpu_index=0)
    decode = fg.load_decoder(device)
    tr = np.load(fg.LAT / "train.npz", allow_pickle=True)
    te = np.load(fg.LAT / "test_b.npz", allow_pickle=True)
    Xtr, ztr, _ = fg.gather_pressure_and_z(tr, "train")
    Xte, zte, _ = fg.gather_pressure_and_z(te, "test_b")
    Xtr, Xte = Xtr.astype(np.float64), Xte.astype(np.float64)
    tcid = np.array([str(c) for c in te["case_id"]])
    dcid = np.array([str(c) for c in fg.DECNPZ["case_ids"]])
    PICKS = fg.PICKS
    est = {K: fg.krr(Xtr[:, :, PICKS[str(K)]].reshape(len(Xtr), -1), ztr) for K in (8, 2)}

    cols = ["simulation", "oracle decode", r"from $K{=}8$ taps", r"from $K{=}2$ taps"]
    fig, axes = plt.subplots(len(fg.CASES), 4, figsize=(6.68, 2.5))
    im = None
    for r, case in enumerate(fg.CASES):
        i = int(np.where(tcid == case)[0][0])
        di = int(np.where(dcid == case)[0][0])
        z8 = est[8](Xte[i:i + 1, :, PICKS["8"]].reshape(1, -1))
        z2 = est[2](Xte[i:i + 1, :, PICKS["2"]].reshape(1, -1))
        fields = [fg.DECNPZ["target_norm"][di, 1], decode(zte[i:i + 1])[0],
                  decode(z8)[0], decode(z2)[0]]
        for c, fld in enumerate(fields):
            im = fs.vort_panel(axes[r, c], fld)
            if r == 0:
                axes[r, c].set_title(cols[c], fontsize=8)
        g, dd, y = float(te["G"][i]), float(te["D"][i]), float(te["Y"][i])
        axes[r, 0].text(-0.07, 0.5, f"$({g:+.1f},\\,{dd:.1f},\\,{y:+.1f})$",
                        transform=axes[r, 0].transAxes, rotation=90, va="center",
                        ha="right", fontsize=6.5)
    fig.subplots_adjust(left=0.085, right=0.91, top=0.9, bottom=0.02,
                        wspace=0.05, hspace=0.1)
    cax = fig.add_axes([0.92, 0.2, 0.011, 0.58])
    fig.colorbar(im, cax=cax, label=r"$\omega_z$ (norm.)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
