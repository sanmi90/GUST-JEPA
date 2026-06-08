#!/usr/bin/env python3
"""
session25_coord_groups.py
=========================

Group the 64 JEPA latent coordinates by FUNCTION. Since the coordinates are
heavily entangled (mean |R| ~ 0.74), per-coordinate interpretation is not
meaningful; instead we profile each coordinate by what physical quantities it
tracks and cluster coordinates with similar profiles into functional groups.

Profile of coordinate k = |Spearman(z_k(t), descriptor(t))| over the pooled
post-impact training frames, for descriptors spanning gust forcing (G, C_L, C_D),
wake vorticity (wake enstrophy, circulation +/-, wake thickness) and wake geometry
(centroid x/y). Coordinates are L2-normalised over the profile (so clustering is by
WHICH quantities, not how strongly) and k-means clustered. Each group is then
characterised by its mean profile and by its held-out future-wake forecast skill.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "session21"))

import os
H = 16
SEED = 0
# UNCONDITIONED adaptation: cluster the 64 UNCONDITIONED latent coordinates by
# physical function. Default tf-no-c; SURD_MODEL=lstm selects the lstm-no-c
# latents. PF descriptors are DNS-derived (encoder-independent).
_MODEL = os.environ.get("SURD_MODEL", "tf")
PF = f"outputs/session27/causal_noc_{_MODEL}/per_frame_targets/{{}}.npz"
B1 = f"outputs/session27/causal_noc_{_MODEL}/latents_jepa_noc/{{}}.npz"
DESC = ["G", "C_L", "C_D", "wake_enstrophy", "circulation_pos", "circulation_neg",
        "wake_thickness", "centroid_x", "centroid_y"]
DLAB = {"G": "G", "C_L": r"$C_L$", "C_D": r"$C_D$", "wake_enstrophy": r"$\Omega_w$",
        "circulation_pos": r"$\Gamma^+$", "circulation_neg": r"$\Gamma^-$",
        "wake_thickness": "thick", "centroid_x": r"$x_c$", "centroid_y": r"$y_c$"}


def tmap(split):
    d = np.load(PF.format(split), allow_pickle=True)
    cid = np.asarray(d["case_id"]); enc = np.asarray(d["encounter_index"], int)
    g = lambda k: np.asarray(d[k], float)
    keep = DESC + ["wake_enstrophy"]
    return {(str(cid[i]), int(enc[i])): {k: g(k)[i] for k in keep} for i in range(len(cid))}


def pooled(split, tm):
    d = np.load(B1.format(split), allow_pickle=True)
    z = np.asarray(d["z_full"], float)
    cid = np.asarray(d["case_ids"] if "case_ids" in d.files else d["case_id"])
    enc = np.asarray(d["encounter_indices"] if "encounter_indices" in d.files else d["encounter_index"], int)
    imp = np.asarray(d["impact_frame"], int)
    n, T, D = z.shape
    X, cur, fut = [], {k: [] for k in DESC}, []
    for i in range(n):
        key = (str(cid[i]), int(enc[i]))
        if key not in tm:
            continue
        for tt in range(int(imp[i]), T - H):
            X.append(z[i, tt, :])
            for k in DESC:
                cur[k].append(tm[key][k][tt])
            fut.append(tm[key]["wake_enstrophy"][tt + H])
    return np.asarray(X), {k: np.asarray(v) for k, v in cur.items()}, np.asarray(fut)


def main():
    tmtr, tmte = tmap("train"), tmap("test_b")
    Xtr, ctr, ftr = pooled("train", tmtr)
    Xte, cte, fte = pooled("test_b", tmte)
    D = Xtr.shape[1]

    # profile each coordinate by |Spearman| with each descriptor (train)
    prof = np.zeros((D, len(DESC)))
    for k in range(D):
        for j, dd in enumerate(DESC):
            prof[k, j] = abs(spearmanr(Xtr[:, k], ctr[dd]).statistic)
    pnorm = prof / (np.linalg.norm(prof, axis=1, keepdims=True) + 1e-9)

    K = 4
    lab = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit_predict(pnorm)

    def wake_skill(cols):
        if len(cols) == 0:
            return 0.0
        sc = StandardScaler().fit(Xtr[:, cols])
        yr = rankdata(ftr); yr = (yr - yr.mean()) / yr.std()
        u = Ridge(alpha=1.0).fit(sc.transform(Xtr[:, cols]), yr).coef_
        s = sc.transform(Xte[:, cols]) @ (u / (np.linalg.norm(u) + 1e-9))
        return abs(spearmanr(s, fte).statistic)

    # order clusters by size; characterise each
    order = sorted(range(K), key=lambda c: -(lab == c).sum())
    print(f"{'group':<7}{'#coords':>8}   top descriptors (mean |rho|)        wake-fc skill")
    print("-" * 74)
    groups = []
    for gi, c in enumerate(order):
        cols = np.where(lab == c)[0]
        mp = prof[cols].mean(0)
        top = sorted(range(len(DESC)), key=lambda j: -mp[j])[:3]
        topstr = ", ".join(f"{DESC[j]}={mp[j]:.2f}" for j in top)
        sk = wake_skill(cols)
        groups.append((cols, mp, sk))
        print(f"G{gi+1:<6}{len(cols):>8}   {topstr:<38}{sk:>8.3f}")
    print(f"\nfor reference, all 64 coordinates -> future wake skill = {wake_skill(np.arange(D)):.3f}")

    # figure: coords (grouped) x descriptors |corr| heatmap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import figstyle; figstyle.use_style(); W = figstyle.TEXTWIDTH_IN
    except Exception:
        W = 4.98
    row_order = np.concatenate([g[0] for g in groups])
    fig, ax = plt.subplots(figsize=(0.64 * W, 0.6 * W))
    im = ax.imshow(prof[row_order], cmap="Greens", vmin=0, vmax=0.8, aspect="auto")
    ax.set_xticks(range(len(DESC))); ax.set_xticklabels([DLAB[d] for d in DESC], rotation=45, ha="right")
    # Label each group as a proper y-tick (matplotlib reserves the left margin for
    # these). The previous version drew them as free text in the same space as a
    # rotated y-axis label, so the two overlapped; dropping the axis label and using
    # tick labels removes the collision.
    centres, ylabels, b = [], [], 0
    for gi, g in enumerate(groups):
        n = len(g[0]); centres.append(b + n / 2 - 0.5)
        ylabels.append(f"G{gi+1}  ({n}c, wake {g[2]:.2f})")
        b += n
        if b < len(row_order):
            ax.axhline(b - 0.5, color="k", lw=0.8)
    ax.set_yticks(centres); ax.set_yticklabels(ylabels, fontsize=7, va="center")
    ax.tick_params(axis="y", length=0)
    ax.set_title("JEPA latent coordinates grouped by physical function", loc="left", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$|\rho_s|$ with descriptor")
    fig.tight_layout(pad=0.4)
    _suffix = "" if _MODEL == "tf" else f"_{_MODEL}"
    figp = Path(f"outputs/session27/figs_uncond/fig_coord_groups{_suffix}.pdf")
    figp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figp, bbox_inches="tight")
    print(f"[fig] wrote {figp}")
    # also persist the grouping numbers for the report
    grp_summary = {
        f"G{gi+1}": {"n_coords": int(len(g[0])),
                     "coords": g[0].tolist(),
                     "mean_profile": {DESC[j]: float(g[1][j]) for j in range(len(DESC))},
                     "wake_fc_skill": float(g[2])}
        for gi, g in enumerate(groups)
    }
    grp_summary["all64_wake_fc_skill"] = float(wake_skill(np.arange(D)))
    import json as _json
    jp = Path(f"outputs/session27/causal_noc_{_MODEL}/coord_groups.json")
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(_json.dumps(grp_summary, indent=2))
    print(f"[done] wrote {jp}")


if __name__ == "__main__":
    main()
