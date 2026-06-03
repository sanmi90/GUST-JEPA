#!/usr/bin/env python3
"""
session25_fig_pc3_panels.py
===========================

Regenerates fig_pc3_forecast.pdf as a two-panel figure:

 (a) the partialling asymmetry (PC3 vs C_L toward the future wake, raw and
     partial, train + held-out test_b) -- the original single panel;
 (b) a mode x future-observable observability heatmap O = I_deb/H (surrogate
     null) for the leading per-frame PCA modes of the JEPA latent against the six
     manuscript observables, in the Martinez-Sanchez (JFM 967 A1, 2023)
     causal-map idiom but with observability rather than transfer entropy
     (observability is the robust single-source quantity at this sample size).

Same pooled post-impact per-frame training regime as the other Session 25 scripts.
The impulse I_y is computed here from the cached vorticity (sum_x x * sum_y omega),
since it is not stored in the per-frame descriptor file; the other five observables
are read directly. Exploratory, observational, on the frozen encoder.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import h5py
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "session21"))

from infotheory.estimators import surrogate_null_mi
from infotheory.observability import target_entropy

H = 16
SEED = 0
PF = "outputs/session16/exp2/per_frame_targets/{}.npz"
OBS = ["C_L", "C_D", "I_y", "wake_enstrophy", "circulation_pos", "circulation_neg"]
OBS_LABEL = {"C_L": r"$C_L$", "C_D": r"$C_D$", "I_y": r"$I_y$",
             "wake_enstrophy": r"$\Omega_w$", "circulation_pos": r"$\Gamma^{+}$",
             "circulation_neg": r"$\Gamma^{-}$"}


def cache_root():
    c = os.environ.get("VORTEX_JEPA_CACHE")
    if c:
        return Path(c)
    return Path(os.environ["PREVENT_ROOT"]) / "data" / "processed" / "vortex-jepa"


def compute_Iy(case_id, enc, lo, hi, root):
    """I_y(t) = sum_x x * sum_y omega_z over frames [lo, hi); affine proxy fine for MI."""
    f = root / "v1" / case_id / f"encounter_{enc:02d}.h5"
    if not f.exists():
        return None
    with h5py.File(f, "r") as g:
        om = np.asarray(g["omega_z"][lo:hi], float)  # (hi-lo, 192, 96)
    nx = om.shape[1]
    x = np.linspace(-1.5, 4.5, nx)
    return (x[None, :] * om.sum(axis=2)).sum(axis=1)  # (hi-lo,)


def pooled(split, root, need_Iy):
    d = np.load(PF.format(split), allow_pickle=True)
    z = np.asarray(d["z_full"], float)
    imp = np.asarray(d["impact_frame"], int)
    n, T, _ = z.shape
    cid = np.asarray(d["case_id"]); enc = np.asarray(d["encounter_index"], int)
    series = {k: np.asarray(d[k], float) for k in OBS if k != "I_y" and k in d.files}
    if need_Iy:
        Iy = np.full((n, T), np.nan)
        for i in range(n):
            v = compute_Iy(str(cid[i]), int(enc[i]), 0, T, root)
            if v is not None:
                Iy[i, :len(v)] = v
        series["I_y"] = Iy
    rows = [(i, t) for i in range(n) for t in range(int(imp[i]), T - H)]
    ii = np.array([r[0] for r in rows]); tt = np.array([r[1] for r in rows])
    Z = z[ii, tt, :]
    fut = {k: series[k][ii, np.clip(tt + H, 0, T - 1)] for k in series}
    cur_force = {k: series[k][ii, tt] for k in ["C_L", "C_D"] if k in series}
    return Z, fut, cur_force, ii, tt, series


def partial_spearman(a, b, ctrl):
    ra, rb = rankdata(a), rankdata(b)
    RC = np.column_stack([np.ones(len(a))] + [rankdata(ctrl[:, j]) for j in range(ctrl.shape[1])])
    res = lambda r: r - RC @ np.linalg.lstsq(RC, r, rcond=None)[0]
    return abs(float(np.corrcoef(res(ra), res(rb))[0, 1]))


def O_value(source, target, nsur=60):
    null = surrogate_null_mi(target, source, n_surrogate=nsur, k=4, random_state=SEED)
    h = target_entropy(target, n_bins=16, method="hist")
    return float(np.clip(null["mi_debiased"] / h, 0, 1)) if h > 0 else 0.0


def main():
    from sklearn.decomposition import PCA
    root = cache_root()
    Ztr, futtr, _, _, _, _ = pooled("train", root, need_Iy=True)
    Zte, futte, _, _, _, _ = pooled("test_b", root, need_Iy=False)
    pca = PCA(n_components=4, random_state=SEED).fit(Ztr)
    Ptr = pca.transform(Ztr); Pte = pca.transform(Zte)

    # ---- panel (a): partialling asymmetry (PC3 vs C_L -> future wake) --------
    def bars(P, fut, force_cl):
        pc3 = P[:, 2]; w = fut["wake_enstrophy"]
        F = np.column_stack([fut.get("G", np.zeros_like(w))]) if "G" in fut else None
        # forces control = C_L, C_D at the future? we control the CURRENT forces;
        # reuse cur via fut here is not the same, so recompute from raw arrays:
        return pc3, w, force_cl
    # recompute current forces aligned to the pooled rows
    def asym(split, P):
        d = np.load(PF.format(split), allow_pickle=True)
        z = np.asarray(d["z_full"], float); imp = np.asarray(d["impact_frame"], int)
        n, T, _ = z.shape
        cl = np.asarray(d["C_L"], float); cd = np.asarray(d["C_D"], float)
        g = np.asarray(d["G"], float); we = np.asarray(d["wake_enstrophy"], float)
        rows = [(i, t) for i in range(n) for t in range(int(imp[i]), T - H)]
        ii = np.array([r[0] for r in rows]); tt = np.array([r[1] for r in rows])
        pc3 = P[:, 2]
        wfut = we[ii, np.clip(tt + H, 0, T - 1)]
        forces = np.column_stack([g[ii, tt], cl[ii, tt], cd[ii, tt]])
        clcur = cl[ii, tt]
        return {
            "pc3_raw": abs(spearmanr(pc3, wfut).statistic),
            "pc3_par": partial_spearman(pc3, wfut, forces),
            "cl_raw": abs(spearmanr(clcur, wfut).statistic),
            "cl_par": partial_spearman(clcur, wfut, pc3[:, None]),
        }
    a_tr = asym("train", Ptr); a_te = asym("test_b", Pte)

    # ---- panel (b): mode x future-observable observability heatmap ----------
    Omat = np.zeros((4, len(OBS)))
    for k in range(4):
        for j, ob in enumerate(OBS):
            if ob in futtr:
                Omat[k, j] = O_value(Ptr[:, k], futtr[ob])
    print("observability O (PC x future observable), train pooled:")
    print("        " + "".join(f"{OBS_LABEL[o].replace('$',''):>8}" for o in OBS))
    for k in range(4):
        print(f"  PC{k+1}  " + "".join(f"{Omat[k,j]:>8.3f}" for j in range(len(OBS))))

    # ---- figure -------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import figstyle
        figstyle.use_style()
        green = figstyle.FAMILY_COLOR["jepa"]; grey = figstyle.FAMILY_COLOR["oracle"]
        W = figstyle.TEXTWIDTH_IN
    except Exception:
        green, grey, W = "#1b7837", "#404040", 4.98

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(W, 0.40 * W),
                                   gridspec_kw={"width_ratios": [1.0, 1.15]})
    # (a)
    cats = ["PC3 raw", "PC3\n| forces", "$C_L$ raw", "$C_L$\n| PC3"]
    tr = [a_tr["pc3_raw"], a_tr["pc3_par"], a_tr["cl_raw"], a_tr["cl_par"]]
    te = [a_te["pc3_raw"], a_te["pc3_par"], a_te["cl_raw"], a_te["cl_par"]]
    cols = [green, green, grey, grey]; x = np.arange(4); w = 0.38
    axA.bar(x - w/2, tr, w, color=cols, alpha=0.45)
    axA.bar(x + w/2, te, w, color=cols, alpha=1.0)
    axA.set_xticks(x); axA.set_xticklabels(cats); axA.set_ylim(0, 0.65)
    axA.set_ylabel(r"$|\rho_s|$ to future wake $\Omega_w(t{+}H)$")
    axA.axhline(0.25, ls=":", lw=0.8, color="0.5")
    from matplotlib.patches import Patch
    axA.legend(handles=[Patch(facecolor="0.5", alpha=0.45, label="train"),
                        Patch(facecolor="0.5", alpha=1.0, label="test B")],
               loc="upper right", frameon=False)
    axA.set_title("(a) forecast-info asymmetry", loc="left")
    # (b)
    im = axB.imshow(Omat, cmap="Greens", vmin=0, vmax=max(0.15, Omat.max()), aspect="auto")
    axB.set_xticks(range(len(OBS))); axB.set_xticklabels([OBS_LABEL[o] for o in OBS])
    axB.set_yticks(range(4)); axB.set_yticklabels([f"PC{k+1}" for k in range(4)])
    for k in range(4):
        for j in range(len(OBS)):
            axB.text(j, k, f"{Omat[k,j]:.2f}", ha="center", va="center",
                     fontsize=6, color="white" if Omat[k, j] > 0.10 else "0.2")
    axB.set_title("(b) observability $O=I/H$ of future observables", loc="left")
    fig.colorbar(im, ax=axB, fraction=0.046, pad=0.04, label="$O$")
    fig.tight_layout(pad=0.4)
    out = Path("paper/sections/figures/results/fig_pc3_forecast.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"[fig] wrote {out}")


if __name__ == "__main__":
    main()
