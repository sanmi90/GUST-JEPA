"""EDRFCM abstract Figure 1: decoded impact-frame vorticity across the three
held-out splits. Rows: a test_a (in-distribution held-out), a test_b
(interpolation), and a test_c (|G|=4 extrapolation) encounter, each the strongest
gust in its split. Columns: simulation, predictive (JEPA), AE (d=64),
POD, all at matched d=64. Double-column width for the EDRFCM two-column layout.
Source: outputs/session20/decoded/{test_a,test_b,test_c}.npz (no GPU).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/carlos/GUST-JEPA")
sys.path.insert(0, str(REPO / "scripts" / "session21"))
import figstyle as fs  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

DEC = REPO / "outputs/session20/decoded"
OUT = REPO / "paper/edrfcm2026/Figures/decode_splits.pdf"

ROWS = [("test_a", "test A\n(in-distribution)"),
        ("test_b", "test B\n(interpolation)"),
        ("test_c", r"test C ($|G|{=}4$)")]
COLS = [("target_norm", "simulation"), ("jepa_norm", "predictive (JEPA)"),
        ("fukami_norm", r"AE ($d{=}64$)"), ("pod_norm", "POD")]
IMPACT_IDX = 1   # offsets [-8,0,8,16,...]; index 1 = impact (offset 0)


def main() -> None:
    fs.use_style()
    fig, axes = plt.subplots(3, 4, figsize=(6.68, 2.5))
    im = None
    cases = {}
    for r, (split, rlab) in enumerate(ROWS):
        d = np.load(DEC / f"{split}.npz", allow_pickle=True)
        G, D, Y = d["G"], d["D"], d["Y"]
        rep = int(np.argmax(np.abs(G)))
        cases[split] = (float(G[rep]), float(D[rep]), float(Y[rep]))
        for c, (key, clab) in enumerate(COLS):
            ax = axes[r, c]
            im = fs.vort_panel(ax, d[key][rep, IMPACT_IDX])
            if r == 0:
                ax.set_title(clab, fontsize=8)
            if c == 0:
                ax.text(-0.07, 0.5, rlab, transform=ax.transAxes, rotation=90,
                        ha="right", va="center", fontsize=7)
    fig.subplots_adjust(left=0.085, right=0.91, top=0.9, bottom=0.02,
                        wspace=0.05, hspace=0.1)
    cax = fig.add_axes([0.92, 0.2, 0.011, 0.58])
    fig.colorbar(im, cax=cax, label=r"$\omega_z$ (norm.)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    for s, (g, dd, y) in cases.items():
        print(f"{s}: (G,D,Y)=({g:+.1f}, {dd:.1f}, {y:+.1f})")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
