"""Low-order matched-head AE vs JEPA: observable readability.

For each family (matched-head predictive JEPA jepa_tf_noc, matched-head
reconstructive AE ctrl_recon_cnnvit) at each low latent dim, fit ONE linear
collective readout (StandardScaler + Ridge, alpha by GroupKFold-by-case on
train) of the full d-dim per-frame latent to each per-frame flow-state
observable, and report held-out test_b R^2. Observables are encoder-independent
(from per_frame_targets) and joined to the low-d latents by (case_id, encounter).
PER-FRAME state-descriptor regime; parameters omitted (impact-frame quantity).
CPU only.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import r2_score

REPO = Path("/home/carlos/GUST-JEPA")
LAT = REPO / "outputs/session28/latents"
PF = REPO / "outputs/session28/exp2/per_frame_targets"
OBS = ["wake_enstrophy", "circulation_pos", "circulation_neg", "centroid_x",
       "centroid_y", "peak_pos_omega", "peak_neg_omega", "C_L", "C_D"]
FAMILIES = {"JEPA": "jepa_tf_noc_d{d}_s42", "AE": "ctrl_recon_cnnvit_d{d}_s42"}
DIMS = [int(a) for a in sys.argv[1:]] or [4, 8]


def load_lat(tag, split):
    d = np.load(LAT / tag / f"{split}.npz", allow_pickle=True)
    cid = d["case_ids"] if "case_ids" in d.files else d["case_id"]
    enc = d["encounter_indices"] if "encounter_indices" in d.files else d["encounter_index"]
    return d["z_full"].astype(np.float64), np.asarray(cid), np.asarray(enc)


def load_obs(split):
    d = np.load(PF / f"{split}.npz", allow_pickle=True)
    cid = np.asarray(d["case_id"]); enc = np.asarray(d["encounter_index"])
    idx = {(str(c), int(e)): i for i, (c, e) in enumerate(zip(cid, enc))}
    return {o: d[o] for o in OBS}, idx


def xy(tag, split, observable, obsmap, obsidx):
    z, cid, enc = load_lat(tag, split)
    obs = obsmap[observable]
    X, y, g = [], [], []
    for i, (c, e) in enumerate(zip(cid, enc)):
        key = (str(c), int(e))
        if key not in obsidx:
            continue
        yt = obs[obsidx[key]]
        m = np.isfinite(yt)
        X.append(z[i][m]); y.append(yt[m]); g.append(np.full(int(m.sum()), str(c)))
    return np.concatenate(X), np.concatenate(y), np.concatenate(g)


def probe(Xtr, ytr, gtr):
    pipe = Pipeline([("s", StandardScaler()), ("m", Ridge())])
    cv = GroupKFold(n_splits=min(5, len(set(gtr))))
    gs = GridSearchCV(pipe, {"m__alpha": [0.1, 1.0, 10.0, 100.0]}, cv=cv, scoring="r2")
    gs.fit(Xtr, ytr, groups=gtr)
    return gs


def main():
    otr, itr = load_obs("train")
    otb, itb = load_obs("test_b")
    print(f"observables x families, held-out test_b R^2 (full-latent linear readout)")
    for d in DIMS:
        print(f"\n=== d = {d} ===")
        print(f"{'observable':20s} {'JEPA':>8s} {'AE':>8s} {'Δ(JEPA-AE)':>12s}")
        for o in OBS:
            r = {}
            for fam, tmpl in FAMILIES.items():
                tag = tmpl.format(d=d)
                Xtr, ytr, gtr = xy(tag, "train", o, otr, itr)
                Xtb, ytb, _ = xy(tag, "test_b", o, otb, itb)
                r[fam] = float(r2_score(ytb, probe(Xtr, ytr, gtr).predict(Xtb)))
            print(f"{o:20s} {r['JEPA']:+8.2f} {r['AE']:+8.2f} {r['JEPA']-r['AE']:+12.2f}")


if __name__ == "__main__":
    main()
