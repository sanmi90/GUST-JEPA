"""Coauthor question: can an UNCONDITIONED temporal predictor match the conditioned
forecast, given a Takens-style temporal window of the latent?

Two no-training probes on the FROZEN production encoder (jepa_d64_test1_noBN):
  (1) trajectory probe: is c=(G,D,Y) recoverable from the encoded z-trajectory?
  (2) Takens window probe: regress wake-enstrophy at impact+16 from a delay window
      of encoded latents z[impact-W .. impact] (no c), vs c-only, vs z(impact),
      vs window+c, vs the encoded-future oracle. Held-out test_b R^2 (RidgeCV).

If the window does NOT recover wake@16 beyond the c-only floor, the forcing is an
observability limit of the mid-plane slice, not an architecture gap, and no
unconditioned predictor (any class) can close it. Validates the conditioned
baseline (~0.449) first so the metric is anchored.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "session20"))
import exp_closure_r2 as cr  # noqa: E402

LATDIR = REPO / "outputs/session18/exp_b1/latents_jepa_d64_test1_noBN"
dns = np.load(cr.DNS_METRICS_PATH, allow_pickle=True)
ALPHAS = np.logspace(-2, 4, 13)
H = 16


def heldout_r2(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    r = RidgeCV(alphas=ALPHAS).fit(sc.transform(Xtr), ytr)
    yp = r.predict(sc.transform(Xte))
    ss_res = float(((yp - yte) ** 2).sum()); ss_tot = float(((yte - yte.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def load_split(split):
    b = np.load(LATDIR / f"{split}.npz", allow_pickle=True)
    z = b["z_full"].astype(np.float64)                       # (n, T, d)
    cid = cr._get(b, "case_ids", "case_id"); ei = cr._get(b, "encounter_indices", "encounter_index")
    imp = b["impact_frame"].astype(int)
    c = np.c_[b["G"].astype(float), b["D"].astype(float), b["Y"].astype(float)]
    di = cr.match_index(cid, ei, dns[f"{split}_case_id"], dns[f"{split}_encounter_index"])
    keep = di >= 0
    return z[keep], imp[keep], c[keep], di[keep]


def window_feats(z, imp, W):
    n, T, d = z.shape
    out = np.zeros((n, (W + 1) * d))
    for i in range(n):
        lo = max(0, imp[i] - W); seg = z[i, lo:imp[i] + 1]
        if seg.shape[0] < W + 1:                              # left-pad with first frame
            seg = np.vstack([np.repeat(seg[:1], W + 1 - seg.shape[0], 0), seg])
        out[i] = seg.reshape(-1)
    return out


def main():
    # ---- 0. validate the conditioned baseline with the canonical metric ----
    print("=== conditioned baseline (canonical exp_closure_r2 probe) ===")
    rows = cr.evaluate("jepa_d64_test1_noBN", "jepa", 64, dns, (8, H), ["test_b", "test_c"], n_boot=200)
    for r in rows:
        if r["metric"] == "wake_enstrophy" and r["horizon"] == H and r["mode"] in ("z_markov", "z_dns"):
            print(f"  {r['split']:7s} {r['mode']:9s} H{H} wake R2 = {r['r2']:+.3f}")

    ztr, itr, ctr, dtr = load_split("train")
    zte, ite, cte, dte = load_split("test_b")
    wake_tr = np.array([dns["train_wake_enstrophy"][dtr[i], itr[i] + H] for i in range(len(dtr))])
    wake_te = np.array([dns["test_b_wake_enstrophy"][dte[i], ite[i] + H] for i in range(len(dte))])

    # ---- 1. trajectory probe: c from the encoded trajectory ----
    print("\n=== (1) trajectory probe: c=(G,D,Y) from encoded z-trajectory (test_b R2) ===")
    Xtr_traj = window_feats(ztr, itr, 16); Xte_traj = window_feats(zte, ite, 16)
    for j, name in enumerate(["G", "D", "Y"]):
        r2 = heldout_r2(Xtr_traj, ctr[:, j], Xte_traj, cte[:, j])
        print(f"  {name}: window[impact-16..impact] -> {name}  R2 = {r2:+.3f}")

    # ---- 2. Takens window probe: wake@impact+16 ----
    print("\n=== (2) wake-enstrophy @ impact+16 from various features (held-out test_b R2) ===")
    feats = {
        "c only (G,D,Y) [floor]": (ctr, cte),
        "z(impact) only": (ztr[np.arange(len(itr)), itr], zte[np.arange(len(ite)), ite]),
        "window W=4 (no c)": (window_feats(ztr, itr, 4), window_feats(zte, ite, 4)),
        "window W=8 (no c)": (window_feats(ztr, itr, 8), window_feats(zte, ite, 8)),
        "window W=16 (no c)": (Xtr_traj, Xte_traj),
        "window W=8 + c": (np.c_[window_feats(ztr, itr, 8), ctr], np.c_[window_feats(zte, ite, 8), cte]),
        "ORACLE z(impact+16) [ceiling]": (ztr[np.arange(len(itr)), itr + H], zte[np.arange(len(ite)), ite + H]),
    }
    for name, (Xtr, Xte) in feats.items():
        print(f"  {name:34s} R2 = {heldout_r2(Xtr, wake_tr, Xte, wake_te):+.3f}")


if __name__ == "__main__":
    main()
