"""F8 (v4 paper): task-dependent readability matrix across the cube cells.

Linear-probe R2 per (cell, observable) on test_b: probes fit on train
z_gap -> target, the frozen represent.py convention; seed-mean over the
3 encoder seeds per cell (single seed where only one run exists). Cells
include the AE anchors. Extends fig_readability_matrix_v3 to the Track C
cells (v4 plan F8).

Run (CPU): taskset -c 16-23 python -m scripts.session35.fig_readability_v4
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "session21"))
import figstyle  # noqa: E402

from scripts.session34.trackc_cells import CELLS  # noqa: E402
from src.evaluation.represent import fit_linear_probe  # noqa: E402

CACHE = REPO / "outputs/session34/trackc_latents"
OUT = REPO / "paper/sections/figures/results"
OBS = [("target_C_L", r"$C_L$"), ("target_C_D", r"$C_D$"),
       ("target_wake_enstrophy", r"$E_w$"),
       ("target_circulation_pos", r"$\Gamma^{+}$"),
       ("target_circulation_neg", r"$\Gamma^{-}$")]
ORDER = ["c0", "cn", "cw", "cwn", "cl", "cln", "clw", "clwn",
         "ae_l", "ae_w", "ae_lw"]
COLLAPSED = {"c0", "cn", "cw", "cwn"}


def r2(y, p):
    return 1.0 - ((y - p) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)


def main() -> int:
    figstyle.use_style()
    labels = []
    M = np.full((len(ORDER), len(OBS)), np.nan)
    n_seeds = []
    for i, cell in enumerate(ORDER):
        runs = list(CELLS[cell].values())
        labels.append(cell.upper().replace("_", "-").replace("C0", "C0"))
        n_seeds.append(len(runs))
        vals = np.full((len(runs), len(OBS)), np.nan)
        for s, run in enumerate(runs):
            tr = np.load(CACHE / f"latents_{run}_train.npz", allow_pickle=True)
            tb = np.load(CACHE / f"latents_{run}_test_b.npz", allow_pickle=True)
            for j, (key, _) in enumerate(OBS):
                probe = fit_linear_probe(tr["z_gap"], tr[key].astype(np.float64))
                vals[s, j] = r2(tb[key].astype(np.float64),
                                probe.predict(tb["z_gap"]))
        M[i] = vals.mean(axis=0)
        print(f"[f8] {cell} (n={len(runs)}):",
              np.round(M[i], 3).tolist(), flush=True)

    fig, ax = plt.subplots(
        figsize=(figstyle.TEXTWIDTH_IN * 0.72, 2.9))
    im = ax.imshow(np.clip(M, 0, 1), cmap="Greens", vmin=0, vmax=1,
                   aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}" if v > -10 else "<-10",
                    ha="center", va="center", fontsize=6.0,
                    color="white" if 0 < v > 0.6 else "black")
    ax.set_xticks(range(len(OBS)))
    ax.set_xticklabels([lab for _, lab in OBS])
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([f"{lab}{'*' if ORDER[k] in COLLAPSED else ''}"
                        f" (n={n_seeds[k]})"
                        for k, lab in enumerate(labels)], fontsize=6.2)
    ax.set_xlabel("observable (linear probe, per-frame $z$)")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(r"in-distribution $R^2$ (clipped at 0)", fontsize=6.5)
    ax.set_title("task-dependent readability; * = collapsed cell",
                 fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_readability_v4.pdf")
    fig.savefig(OUT / "fig_readability_v4.png", dpi=200)
    print("wrote", OUT / "fig_readability_v4.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
