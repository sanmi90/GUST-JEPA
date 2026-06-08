#!/usr/bin/env python3
"""
session25_obs_matrix.py
=======================

Families x future-observables forecast-skill matrix (the reliable, probe-based
analogue of an observability matrix). For each encoder family (JEPA, Fukami, POD)
at matched d=64 and each future observable, the held-out skill |Spearman| of a
linear probe of that observable on the full latent (train-fit ridge, evaluated on
test_b). This is "how well each family's latent forecasts each future observable",
the families x observables view, without the unreliable 64-dim mutual-information
estimate. Iy omitted (needs the vorticity field); the five direct observables show
the force-vs-wake pattern.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "session21"))

H = 16
# UNCONDITIONED adaptation: JEPA row = unconditioned tf-no-c latents; Fukami/POD
# stay at the production B1 export. PF descriptors are DNS-derived.
PF = "outputs/session27/causal_noc_tf/per_frame_targets/{}.npz"
B1 = "outputs/session18/exp_b1/latents_{}/{}.npz"
UNCOND_JEPA = "outputs/session27/causal_noc_tf/latents_jepa_noc/{}.npz"
FAMDIR = {"JEPA": "jepa_d64", "Fukami": "fukami_d64_noBN", "POD": "pod_d64_noBN"}


def fam_path(fam, sub, split):
    if fam == "JEPA":
        return UNCOND_JEPA.format(split)
    return B1.format(sub, split)
OBS = ["C_L", "C_D", "wake_enstrophy", "circulation_pos", "circulation_neg"]
LAB = {"C_L": r"$C_L$", "C_D": r"$C_D$", "wake_enstrophy": r"$\Omega_w$",
       "circulation_pos": r"$\Gamma^{+}$", "circulation_neg": r"$\Gamma^{-}$"}


def tmap(split):
    d = np.load(PF.format(split), allow_pickle=True)
    cid = np.asarray(d["case_id"]); enc = np.asarray(d["encounter_index"], int)
    g = lambda k: np.asarray(d[k], float)
    return {(str(cid[i]), int(enc[i])): {o: g(o)[i] for o in OBS} for i in range(len(cid))}


def load_family(path):
    d = np.load(path, allow_pickle=True)
    cids = d["case_ids"] if "case_ids" in d.files else d["case_id"]
    encs = d["encounter_indices"] if "encounter_indices" in d.files else d["encounter_index"]
    return (np.asarray(d["z_full"], float), np.asarray(cids), np.asarray(encs, int),
            np.asarray(d["impact_frame"], int))


def build(z, cids, encs, imps, tm, ob):
    X, y = [], []
    for i in range(len(cids)):
        key = (str(cids[i]), int(encs[i]))
        if key not in tm:
            continue
        imp = int(imps[i]); Tn = z.shape[1]
        for tt in range(imp, Tn - H):
            X.append(z[i, tt, :]); y.append(tm[key][ob][tt + H])
    return np.asarray(X), np.asarray(y)


def main():
    tmtr, tmte = tmap("train"), tmap("test_b")
    fams = list(FAMDIR)
    Mat = np.zeros((len(fams), len(OBS)))
    for fi, fam in enumerate(fams):
        ztr = load_family(fam_path(fam, FAMDIR[fam], "train"))
        zte = load_family(fam_path(fam, FAMDIR[fam], "test_b"))
        for oj, ob in enumerate(OBS):
            Xtr, ytr = build(*ztr, tmtr, ob)
            Xte, yte = build(*zte, tmte, ob)
            sc = StandardScaler().fit(Xtr)
            yr = rankdata(ytr); yr = (yr - yr.mean()) / yr.std()
            u = Ridge(alpha=1.0).fit(sc.transform(Xtr), yr).coef_
            s = sc.transform(Xte) @ (u / np.linalg.norm(u))
            Mat[fi, oj] = abs(spearmanr(s, yte).statistic)

    print(f"{'family':<8}" + "".join(f"{LAB[o].replace('$',''):>9}" for o in OBS))
    for fi, fam in enumerate(fams):
        print(f"{fam:<8}" + "".join(f"{Mat[fi,j]:>9.3f}" for j in range(len(OBS))))

    op = Path("outputs/session27/causal_noc_tf/obs_matrix.npz")
    op.parent.mkdir(parents=True, exist_ok=True)
    np.savez(op, matrix=Mat, families=np.array(fams), observables=np.array(OBS))

    # heatmap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import figstyle; figstyle.use_style(); W = figstyle.TEXTWIDTH_IN
    except Exception:
        W = 4.98
    fig, ax = plt.subplots(figsize=(0.6 * W, 0.34 * W))
    im = ax.imshow(Mat, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(OBS))); ax.set_xticklabels([LAB[o] for o in OBS])
    ax.set_yticks(range(len(fams))); ax.set_yticklabels(fams)
    for fi in range(len(fams)):
        for oj in range(len(OBS)):
            ax.text(oj, fi, f"{Mat[fi,oj]:.2f}", ha="center", va="center",
                    fontsize=6.5, color="white" if Mat[fi, oj] > 0.6 else "0.2")
    ax.set_title("held-out forecast skill of each future observable, by family", loc="left", fontsize=7.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$|\rho_s|$")
    fig.tight_layout(pad=0.3)
    out = Path("outputs/session27/figs_uncond/fig_obs_matrix.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight"); print(f"[fig] wrote {out}")


if __name__ == "__main__":
    main()
