#!/usr/bin/env python3
"""
session25_jepa_mode_surd.py
===========================

EXPLORATORY (post-C0, interpretability only, NOT a pre-registered manuscript
claim). The Session 25 C0 gate ruled out the *physical-scalar* synergy mechanism
(future wake synergistic in (G, current wake state)); it says nothing about how the
*learned JEPA latent* lays out future-relevant information across its own modes.

This script decomposes the information the JEPA d=64 latent carries about the
future wake enstrophy (H = 16) into unique / redundant / synergistic parts, using
SURD with the latent's own coordinates as sources, two ways:

  RAW   : the three raw latent dimensions individually most informative about the
          future wake (the model's native, BatchNorm-unit axes).
  PCA   : the three leading principal components of the latent (a within-JEPA
          rotation). POD of the flow is deliberately excluded (user request).

Estimation on the pooled TRAINING trajectories (per-frame, post-impact window),
the regime the infotheory README recommends for synergy. Target = future wake
enstrophy is a state descriptor, so per-frame z is the correct regime per the
repo's probe-methodology note. Caveat: pooled frames are autocorrelated, so the
effective sample size is below the nominal one; read the raw-vs-PCA contrast
qualitatively, with the bins sweep and the surrogate-null observability as the
robustness checks.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infotheory.estimators import quantise, joint_pmf, mutual_information_knn
from infotheory.surd import surd_discrete
from infotheory.observability import observability

H = 16
SEED = 0
PER_FRAME = "outputs/session16/exp2/per_frame_targets/train.npz"


def load_pooled(window: str = "postimpact"):
    d = np.load(PER_FRAME, allow_pickle=True)
    z = np.asarray(d["z_full"], float)          # (n, 120, 64)
    we = np.asarray(d["wake_enstrophy"], float)  # (n, 120)
    imp = np.asarray(d["impact_frame"], int)     # (n,)
    n, T, dlat = z.shape
    X, y = [], []
    for i in range(n):
        t0 = int(imp[i]) if window == "postimpact" else 0
        for t in range(t0, T - H):
            X.append(z[i, t, :])
            y.append(we[i, t + H])
    return np.asarray(X), np.asarray(y), n, dlat


def surd_3(sources: np.ndarray, target: np.ndarray, nb: int):
    """SURD of target from 3 source columns at nb quantile bins. Return summary."""
    cols = [quantise(sources[:, j], nb) for j in range(sources.shape[1])]
    t = quantise(target, nb)
    p = joint_pmf(np.column_stack([t] + cols), tuple([nb] * (sources.shape[1] + 1)))
    res = surd_discrete(p)
    h = res.h_target or 1.0
    U = sum(res.unique.values()); R = sum(res.redundant.values()); S = sum(res.synergistic.values())
    # dominant single unique mode
    top_u = max(res.unique.items(), key=lambda kv: kv[1]) if res.unique else (None, 0.0)
    return {
        "MI": res.mi_total, "H": res.h_target, "leak": res.info_leak,
        "U": U / h, "R": R / h, "S": S / h,
        "top_unique": (sorted(top_u[0])[0] if top_u[0] else None, top_u[1] / h),
    }


def main():
    X, y, n_enc, dlat = load_pooled("postimpact")
    print(f"[data] {n_enc} train encounters, post-impact window, H={H}, "
          f"pooled n={len(X)} frames, latent d={dlat}\n")

    # ---- rank raw dims by univariate MI to the future wake -------------------
    mis = np.array([mutual_information_knn(X[:, k], y, k=4, random_state=SEED) for k in range(dlat)])
    order = np.argsort(mis)[::-1]
    top_raw = order[:3]
    print("=== RAW JEPA modes (model-native axes, no rotation) ===")
    print(f"top-3 raw dims by MI(z_k; future wake): {top_raw.tolist()}  "
          f"MI(nats)={np.round(mis[top_raw], 3).tolist()}")
    print(f"MI spread over all 64 dims: max {mis.max():.3f}, "
          f"#dims with MI>0.5*max = {int((mis > 0.5*mis.max()).sum())} "
          f"(many informative dims => expect redundancy)")
    for nb in (4, 6, 8):
        r = surd_3(X[:, top_raw], y, nb)
        print(f"  bins={nb}: MI={r['MI']:.3f}b leak={r['leak']:.3f} | "
              f"U={r['U']:.3f} R={r['R']:.3f} S={r['S']:.3f}  (R/MI*H share dominant?)")
    # per-mode observability with surrogate null
    print("  per-raw-dim observability O=I_deb/H (surrogate null, 80):")
    for d in top_raw:
        o = observability(X[:, d], y, source_name=f"z{d}", target_name="wake_fut",
                          n_surrogate=80, random_state=SEED)
        print(f"    z[{d}]: O={o.observability:.3f}  p={o.p_value:.3g}  spread={o.robustness_spread:.3f}")

    # ---- PCA rotation of the latent ------------------------------------------
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3, random_state=SEED).fit(X)
    P = pca.transform(X)
    print("\n=== PCA modes of the JEPA latent (within-JEPA rotation; NO POD) ===")
    print(f"PC variance fractions: {np.round(pca.explained_variance_ratio_, 3).tolist()}")
    for nb in (4, 6, 8):
        r = surd_3(P, y, nb)
        print(f"  bins={nb}: MI={r['MI']:.3f}b leak={r['leak']:.3f} | "
              f"U={r['U']:.3f} R={r['R']:.3f} S={r['S']:.3f}  "
              f"top-unique=PC{r['top_unique'][0]} ({r['top_unique'][1]:.3f})")
    print("  per-PC observability O=I_deb/H (surrogate null, 80):")
    for j in range(3):
        o = observability(P[:, j], y, source_name=f"PC{j+1}", target_name="wake_fut",
                          n_surrogate=80, random_state=SEED)
        print(f"    PC{j+1}: O={o.observability:.3f}  p={o.p_value:.3g}  spread={o.robustness_spread:.3f}")

    # ---- impact-frame-only cross-check (manuscript regime, n=226) -------------
    d = np.load(PER_FRAME, allow_pickle=True)
    z = np.asarray(d["z_full"], float); we = np.asarray(d["wake_enstrophy"], float)
    imp = np.asarray(d["impact_frame"], int)
    zi = np.array([z[i, imp[i], :] for i in range(len(imp))])
    yi = np.array([we[i, imp[i] + H] for i in range(len(imp))])
    misi = np.array([mutual_information_knn(zi[:, k], yi, k=4, random_state=SEED) for k in range(dlat)])
    topi = np.argsort(misi)[::-1][:3]
    Pi = PCA(n_components=3, random_state=SEED).fit_transform(zi)
    print(f"\n=== impact-frame only (n={len(zi)}, manuscript regime; thin, bins=4) ===")
    rr = surd_3(zi[:, topi], yi, 4); rp = surd_3(Pi, yi, 4)
    print(f"  RAW  top3 {topi.tolist()}: U={rr['U']:.3f} R={rr['R']:.3f} S={rr['S']:.3f} MI={rr['MI']:.3f}b")
    print(f"  PCA  top3            : U={rp['U']:.3f} R={rp['R']:.3f} S={rp['S']:.3f} MI={rp['MI']:.3f}b")


if __name__ == "__main__":
    main()
