"""Low-order matched-head AE vs JEPA: FORECAST readability vs horizon (h = 1..16).

For each family/dim, fit ONE ridge probe on the ENCODED train per-frame latent ->
wake observable (the same readout as the readability probe), then apply it to the
matched-predictor ROLLED test_b latent at frame (impact + h) for h = 1..16. A
family whose rollout leaves the encoded manifold loses forecast R^2 as h grows;
this tests whether the JEPA latent SPACE is more forecastable than the matched-
head AE latent space under an IDENTICAL predictor. CPU only.

Inputs:
  encoded train latents : outputs/session28/latents/<lat_tag>/train.npz
  rolled  test_b latents: outputs/session29/lowd_rollouts/<roll_key>/test_b.npz
  true observables      : outputs/session28/exp2/per_frame_targets/{train,test_b}.npz
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
ROLL = REPO / "outputs/session29/lowd_rollouts"
PF = REPO / "outputs/session28/exp2/per_frame_targets"
OBS = ["wake_enstrophy", "circulation_pos", "circulation_neg"]
DIMS = [int(a) for a in sys.argv[1:]] or [4, 8]
HMAX = 16
FAM = {  # name -> (encoded-latent tag, rollout key)
    # JEPA-own : the end-to-end model rolled with its OWN co-trained predictor.
    # JEPA-matched : a FRESH matched predictor on the JEPA latent (isolates the
    #   latent geometry from the predictor-training asymmetry).
    # AE-matched : a fresh matched predictor on the reconstructive AE latent.
    "JEPA-own": ("jepa_tf_noc_d{d}_s42", "jepa_own_d{d}"),
    "JEPA-matched": ("jepa_tf_noc_d{d}_s42", "jepa_d{d}"),
    "AE-matched": ("ctrl_recon_cnnvit_d{d}_s42", "ae_d{d}"),
    # regAE: reconstructive AE + SIGReg anti-collapse (d64 only). Tests whether
    # the d64 forecast advantage is the predictive objective or the anti-collapse.
    "regAE-matched": ("regae/cnn_vit_s0", "regae_d{d}"),
}


def _ce(d):
    cid = d["case_ids"] if "case_ids" in d.files else d["case_id"]
    enc = d["encounter_indices"] if "encounter_indices" in d.files else d["encounter_index"]
    return np.asarray(cid), np.asarray(enc)


def load_obs(split):
    d = np.load(PF / f"{split}.npz", allow_pickle=True)
    cid, enc = np.asarray(d["case_id"]), np.asarray(d["encounter_index"])
    idx = {(str(c), int(e)): i for i, (c, e) in enumerate(zip(cid, enc))}
    return {o: d[o] for o in OBS}, idx, d["impact_frame"]


def fit_probe(lat_tag, observable, obsmap, obsidx):
    d = np.load(LAT / lat_tag / "train.npz", allow_pickle=True)
    z = d["z_full"].astype(np.float64); cid, enc = _ce(d)
    obs = obsmap[observable]
    X, y, g = [], [], []
    for i, (c, e) in enumerate(zip(cid, enc)):
        key = (str(c), int(e))
        if key not in obsidx:
            continue
        yt = obs[obsidx[key]]; m = np.isfinite(yt)
        X.append(z[i][m]); y.append(yt[m]); g.append(np.full(int(m.sum()), str(c)))
    X, y, g = np.concatenate(X), np.concatenate(y), np.concatenate(g)
    pipe = Pipeline([("s", StandardScaler()), ("m", Ridge())])
    gs = GridSearchCV(pipe, {"m__alpha": [0.1, 1.0, 10.0, 100.0]},
                      cv=GroupKFold(n_splits=min(5, len(set(g)))), scoring="r2")
    gs.fit(X, y, groups=g)
    return gs


def forecast_curve(roll_key, lat_tag, observable, gs, obsmap_tb, obsidx_tb):
    rp = ROLL / roll_key / "test_b.npz"
    if not rp.exists():
        return None
    d = np.load(rp, allow_pickle=True)
    zr = d["z_full"].astype(np.float64)            # (n,120,dim) rolled
    cid, enc = _ce(d)
    imp = d["impact_frame"] if "impact_frame" in d.files else None
    obs = obsmap_tb[observable]
    curve = {}
    for h in range(1, HMAX + 1):
        xs, ys = [], []
        for i, (c, e) in enumerate(zip(cid, enc)):
            key = (str(c), int(e))
            if key not in obsidx_tb:
                continue
            ti = int(imp[i]); f = ti + h
            if f >= zr.shape[1]:
                continue
            yt = obs[obsidx_tb[key]][f]
            if not np.isfinite(yt):
                continue
            xs.append(zr[i, f]); ys.append(yt)
        if len(ys) < 5:
            curve[h] = float("nan"); continue
        curve[h] = float(r2_score(np.array(ys), gs.predict(np.array(xs))))
    return curve


def main():
    otr, itr, _ = load_obs("train")
    otb, itb, _ = load_obs("test_b")
    for o in OBS:
        print(f"\n############ observable: {o}  (forecast R^2 vs horizon) ############")
        for d in DIMS:
            res = {}
            for fam, (lat_tmpl, roll_tmpl) in FAM.items():
                gs = fit_probe(lat_tmpl.format(d=d), o, otr, itr)
                res[fam] = forecast_curve(roll_tmpl.format(d=d), lat_tmpl.format(d=d), o, gs, otb, itb)
            hs = (1, 2, 4, 8, 12, 16)
            avail = [f for f in FAM if res[f] is not None]
            missing = [f for f in FAM if res[f] is None]
            print(f"  --- d={d} ---" + (f"  (missing: {missing})" if missing else ""))
            print("    h:            " + " ".join(f"{h:>6d}" for h in hs))
            for fam in avail:
                print(f"    {fam:13s}:" + " ".join(f"{res[fam][h]:+6.2f}" for h in hs))


if __name__ == "__main__":
    main()
