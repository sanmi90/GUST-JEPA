"""EDRFCM abstract Figure 2 (combined observability figure).

(a) latent recoverability: held-out (test_b) state-recovery R^2 vs sensor count K
    for all families/dimensions, from the fixed-estimator TCSI pressure map
    (outputs/session21/pressure_v2/pressure_obs_v2.csv). No legend (families are
    colour-coded: JEPA green, reconstructive red, POD blue; darker = larger d).
(b) flow recovered from sparse wall pressure: simulation, oracle decode, and the
    decode of the predictive latent estimated from K=8 and K=2 TCSI taps, for
    three held-out encounters (rows labelled by (G,D,Y)). Reuses the decode logic
    of figG_flow_recovery (RTX 6000).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/carlos/GUST-JEPA")
for p in ("scripts/session21", "", "scripts", "scripts/session20"):
    sys.path.insert(0, str(REPO / p))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
import figstyle as fs  # noqa: E402
import figG_flow_recovery as fg  # noqa: E402

OUT = REPO / "paper/edrfcm2026/Figures/observability.pdf"
CSV = REPO / "outputs/session21/pressure_v2/pressure_obs_v2.csv"
FAMS = [("jepa_d64", "jepa", 64), ("jepa_d32", "jepa", 32),
        ("fukami_d3", "fukami", 3), ("fukami_d32", "fukami", 32),
        ("fukami_d64", "fukami", 64), ("pod_d16", "pod", 16),
        ("pod_d32", "pod", 32), ("pod_d64", "pod", 64)]
KS = [2, 4, 8, 16]


def shade(kind, d):
    ds = sorted({dd for _, k, dd in FAMS if k == kind})
    base = np.array(plt.matplotlib.colors.to_rgb(fs.FAMILY_COLOR[kind]))
    f = 0.45 + 0.55 * (ds.index(d) / (len(ds) - 1)) if len(ds) > 1 else 1.0
    return tuple(f * base + (1 - f) * np.ones(3))  # higher d -> darker


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

    rows = [r for r in csv.DictReader(open(CSV)) if r["split"] == "test_b"]

    def series(tag):
        d = {int(r["K"]): float(r["R2_z"]) for r in rows if r["tag"] == tag}
        return [d[k] for k in KS]

    fig = plt.figure(figsize=(6.68, 2.5))
    gs = GridSpec(3, 6, width_ratios=[1.75, 0.18, 1, 1, 1, 1], wspace=0.06,
                  hspace=0.12, left=0.075, right=0.985, top=0.83, bottom=0.16)
    fig.text(0.165, 0.93, "(a) latent recoverability", ha="center", fontsize=8)
    fig.text(0.64, 0.93, "(b) flow recovery from sparse wall pressure",
             ha="center", fontsize=8)

    # (a) recoverability bars at K in {2,4,8}, matched d=64, three families, NO legend
    axa = fig.add_subplot(gs[:, 0])
    KS_bar = [2, 4, 8]
    fams_bar = [("jepa_d64", "jepa"), ("fukami_d64", "fukami"), ("pod_d64", "pod")]
    xb = np.arange(len(KS_bar))
    w = 0.26
    for j, (tag, kind) in enumerate(fams_bar):
        dd = {int(r["K"]): float(r["R2_z"]) for r in rows if r["tag"] == tag}
        axa.bar(xb + (j - 1) * w, [dd[k] for k in KS_bar], w,
                color=fs.FAMILY_COLOR[kind])
    axa.set_xticks(xb)
    axa.set_xticklabels(KS_bar)
    axa.set_xlabel("sensors $K$", fontsize=8)
    axa.set_ylabel("state recovery $R^2$", fontsize=8)
    axa.set_ylim(0, 1.0)
    axa.axhline(0, color="0.6", lw=0.5)

    # (b) decode grid (column index offset by the spacer)
    cols = ["simulation", "oracle", r"$K{=}8$", r"$K{=}2$"]
    im = None
    for r, case in enumerate(fg.CASES):
        i = int(np.where(tcid == case)[0][0])
        di = int(np.where(dcid == case)[0][0])
        z8 = est[8](Xte[i:i + 1, :, PICKS["8"]].reshape(1, -1))
        z2 = est[2](Xte[i:i + 1, :, PICKS["2"]].reshape(1, -1))
        flds = [fg.DECNPZ["target_norm"][di, 1], decode(zte[i:i + 1])[0],
                decode(z8)[0], decode(z2)[0]]
        for c, fld in enumerate(flds):
            ax = fig.add_subplot(gs[r, c + 2])
            im = fs.vort_panel(ax, fld)
            if r == 0:
                ax.set_title(cols[c], fontsize=7)
            if c == 0:
                g, dd, y = float(te["G"][i]), float(te["D"][i]), float(te["Y"][i])
                ax.text(-0.05, 0.5, f"$({g:+.1f},{dd:.1f},{y:+.1f})$",
                        transform=ax.transAxes, rotation=90, va="center", ha="right",
                        fontsize=5.6)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
