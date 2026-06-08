"""UNCONDITIONED lead-time figure (Session 27 copy of figH_leadtime.py).

(a) Impact-state recovery R2 and (b) impact-lift C_L mean absolute error, as a
function of how many frames before impact the pre-impact pressure window ends, for
the kernel-ridge and LSTM estimators, on the fully-unconditioned (no-c) JEPA d=64
latent.

Input  : outputs/session27/pressure_noc_tf/leadtime.json
Output : outputs/session27/figs_uncond/figH_leadtime.pdf (+ .png)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt  # noqa: F401  (figstyle backend side effects)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "session21"))
import figstyle as fs  # noqa: E402

J = json.load(open(REPO / "outputs/session27/pressure_noc_tf/leadtime.json"))
OUT_PDF = REPO / "outputs/session27/figs_uncond/figH_leadtime.pdf"
OUT_PNG = REPO / "outputs/session27/figs_uncond/figH_leadtime.png"

LEADS = sorted(int(k) for k in J)
KRR_C, LSTM_C = "#1b7837", "#c0392b"


def col(key):
    return [J[str(l)][key] for l in LEADS]


def main() -> None:
    fs.use_style()
    fig, (axa, axb) = plt.subplots(1, 2, figsize=fs.figure_size(1.0, aspect=0.46))

    axa.plot(LEADS, col("r2z_krr"), "-o", ms=3.5, color=KRR_C, label="kernel ridge")
    axa.plot(LEADS, col("r2z_lstm"), "--s", ms=3.5, color=LSTM_C, label="LSTM")
    axa.set_ylabel("impact state recovery $R^2$")
    axa.set_title("(a) state", fontsize=8)
    axa.set_ylim(0.6, 0.95)

    axb.plot(LEADS, col("cl_mae_krr"), "-o", ms=3.5, color=KRR_C, label="kernel ridge")
    axb.plot(LEADS, col("cl_mae_lstm"), "--s", ms=3.5, color=LSTM_C, label="LSTM")
    axb.set_ylabel(r"impact $C_L$ error (MAE)")
    axb.set_title("(b) lift", fontsize=8)

    for ax in (axa, axb):
        ax.set_xlabel("lead before impact (frames)")
        ax.set_xticks(LEADS)
        ax.legend(loc="best", handletextpad=0.4, borderpad=0.2)

    fig.tight_layout()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF); fig.savefig(OUT_PNG, dpi=200)
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
