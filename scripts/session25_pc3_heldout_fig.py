#!/usr/bin/env python3
"""
session25_pc3_heldout_fig.py
============================

Held-out validation of the PC3 forecast-asymmetry result, plus the one-panel
manuscript figure. The PCA rotation is fit on the pooled TRAIN post-impact
per-frame latent and APPLIED unchanged to test_b (proper held-out protocol; the
mode is defined on train, evaluated held out). Reports, for train and test_b:

  PC3 -> future wake enstrophy   raw        and   partial | {G, C_L, C_D}
  C_L -> future wake enstrophy   raw        and   partial | PC3

If PC3 keeps future-wake correlation after the force signature is removed while
C_L's collapses once PC3 is known, the forecast-relevant content is the latent
forcing+geometry direction, not the force signature. Writes the asymmetry figure
to paper/sections/figures/results/fig_pc3_forecast.pdf and a json of the numbers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "session21"))

H = 16
SEED = 0
PER_FRAME = "outputs/session16/exp2/per_frame_targets/{split}.npz"
GEOM = ["centroid_x", "centroid_y", "wake_thickness", "circulation_pos", "circulation_neg"]


def partial_spearman(a, b, ctrl):
    ra, rb = rankdata(a), rankdata(b)
    RC = np.column_stack([np.ones(len(a))] + [rankdata(ctrl[:, j]) for j in range(ctrl.shape[1])])
    ba = np.linalg.lstsq(RC, ra, rcond=None)[0]
    bb = np.linalg.lstsq(RC, rb, rcond=None)[0]
    return abs(float(np.corrcoef(ra - RC @ ba, rb - RC @ bb)[0, 1]))


def pooled(split):
    d = np.load(PER_FRAME.format(split=split), allow_pickle=True)
    z = np.asarray(d["z_full"], float)
    imp = np.asarray(d["impact_frame"], int)
    n, T, _ = z.shape
    rows = [(i, t) for i in range(n) for t in range(int(imp[i]), T - H)]
    ii = np.array([r[0] for r in rows]); tt = np.array([r[1] for r in rows])
    g = lambda k: np.asarray(d[k], float)
    cur = {k: g(k)[ii, tt] for k in ["G", "C_L", "C_D"] + GEOM}
    fut = g("wake_enstrophy")[ii, np.clip(tt + H, 0, T - 1)]
    return z[ii, tt, :], cur, fut


def summarise(P3, cur, fut):
    forces = np.column_stack([cur["G"], cur["C_L"], cur["C_D"]])
    return {
        "pc3_raw": abs(spearmanr(P3, fut).statistic),
        "pc3_par_forces": partial_spearman(P3, fut, forces),
        "cl_raw": abs(spearmanr(cur["C_L"], fut).statistic),
        "cl_par_pc3": partial_spearman(cur["C_L"], fut, P3[:, None]),
        "geom_beyond_forces": {k: partial_spearman(P3, cur[k], forces) for k in GEOM},
    }


def main():
    from sklearn.decomposition import PCA
    Ztr, ctr, ftr = pooled("train")
    Zte, cte, fte = pooled("test_b")
    pca = PCA(n_components=3, random_state=SEED).fit(Ztr)
    s_tr = summarise(pca.transform(Ztr)[:, 2], ctr, ftr)
    s_te = summarise(pca.transform(Zte)[:, 2], cte, fte)

    print(f"[data] train n={len(Ztr)}, test_b n={len(Zte)}; PCA fit on train, applied to test_b\n")
    print(f"{'quantity':<34}{'train':>9}{'test_b':>9}")
    print("-" * 52)
    for lab, key in [("PC3 -> future wake (raw)", "pc3_raw"),
                     ("PC3 -> future wake | G,C_L,C_D", "pc3_par_forces"),
                     ("C_L -> future wake (raw)", "cl_raw"),
                     ("C_L -> future wake | PC3", "cl_par_pc3")]:
        print(f"{lab:<34}{s_tr[key]:>9.3f}{s_te[key]:>9.3f}")
    print("\nPC3 beyond forces (partial | G,C_L,C_D):")
    for k in GEOM:
        print(f"  {k:<16}train {s_tr['geom_beyond_forces'][k]:.3f}   test_b {s_te['geom_beyond_forces'][k]:.3f}")

    # pre-registered held-out gate
    hold = (s_te["pc3_par_forces"] >= 0.25) and (s_te["pc3_par_forces"] > s_te["cl_par_pc3"])
    print(f"\n[gate] test_b PC3|forces={s_te['pc3_par_forces']:.3f} (>=0.25?), "
          f"and > C_L|PC3={s_te['cl_par_pc3']:.3f}?  -> {'HOLDS' if hold else 'FAILS'}")

    out = Path("outputs_causal/jepa_modes")
    out.mkdir(parents=True, exist_ok=True)
    (out / "pc3_heldout.json").write_text(json.dumps({"train": s_tr, "test_b": s_te, "holds": hold}, indent=2))

    # ---- one-panel figure ---------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import figstyle
        figstyle.use_style()
        green = figstyle.FAMILY_COLOR["jepa"]; grey = figstyle.FAMILY_COLOR["oracle"]
        width_in = figstyle.TEXTWIDTH_IN
    except Exception:
        green, grey, width_in = "#1b7837", "#404040", 4.98

    cats = ["PC3 raw", "PC3\n| forces", "$C_L$ raw", "$C_L$\n| PC3"]
    tr = [s_tr["pc3_raw"], s_tr["pc3_par_forces"], s_tr["cl_raw"], s_tr["cl_par_pc3"]]
    te = [s_te["pc3_raw"], s_te["pc3_par_forces"], s_te["cl_raw"], s_te["cl_par_pc3"]]
    colors = [green, green, grey, grey]
    x = np.arange(4); w = 0.38
    fig, ax = plt.subplots(figsize=(0.62 * width_in, 0.42 * width_in))
    ax.bar(x - w / 2, tr, w, color=colors, alpha=0.45, label="train")
    ax.bar(x + w / 2, te, w, color=colors, alpha=1.0, label="test B (held out)")
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylabel(r"$|\rho_s|$ to future wake $\Omega_w(t{+}H)$")
    ax.set_ylim(0, 0.65)
    ax.axhline(0.25, ls=":", lw=0.8, color="0.5")
    # legend: light vs dark = train vs held out
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="0.5", alpha=0.45, label="train"),
                       Patch(facecolor="0.5", alpha=1.0, label="test B (held out)")],
              loc="upper right", frameon=False)
    fig.tight_layout(pad=0.3)
    figpath = Path("paper/sections/figures/results/fig_pc3_forecast.pdf")
    fig.savefig(figpath, bbox_inches="tight")
    print(f"[fig] wrote {figpath}")


if __name__ == "__main__":
    main()
