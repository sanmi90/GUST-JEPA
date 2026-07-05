"""Does the AE's d64 forecast collapse come from too many near-null latent dims?

Part A: encoded-latent variance spectrum (PR, near-null count, top-5 var) for
        JEPA vs matched-head AE at d=4,8,16,64.
Part B: CAUSAL test -- re-run the AE d64 wake-enstrophy forecast (h=8) but project
        BOTH the encoded-train fit and the rolled test_b onto the top-k encoded PCs
        before the ridge probe. If forecast R^2 recovers as k drops (null dims
        removed), the near-null directions are conditioning the collapse.
CPU only.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

REPO = Path("/home/carlos/GUST-JEPA")
LAT = REPO / "outputs/session28/latents"
ROLL = REPO / "outputs/session29/lowd_rollouts"
PF = REPO / "outputs/session28/exp2/per_frame_targets"
JEPA = {4: "jepa_tf_noc_d4_s42", 8: "jepa_tf_noc_d8_s42", 16: "jepa_tf_noc_d16_s42", 64: "jepa_tf_noc_d64_s42"}
AE = {4: "ctrl_recon_cnnvit_d4_s42", 8: "ctrl_recon_cnnvit_d8_s42",
      16: "ctrl_recon_cnnvit_d16_s42", 64: "ctrl_recon_cnnvit_s0"}


def perframe_z(tag, split):
    d = np.load(LAT / tag / f"{split}.npz", allow_pickle=True)
    z = d["z_full"].astype(np.float64)
    return z.reshape(-1, z.shape[-1])


def spectrum(tag):
    Z = perframe_z(tag, "train")
    C = np.cov(Z, rowvar=False)
    ev = np.sort(np.linalg.eigvalsh(C))[::-1]
    ev = np.clip(ev, 0, None)
    pr = (ev.sum() ** 2) / (np.square(ev).sum() + 1e-30)
    nn = int((ev < 1e-3 * ev[0]).sum())
    top5 = float(ev[:5].sum() / (ev.sum() + 1e-30))
    return pr, nn, top5, len(ev)


def load_obs(split):
    d = np.load(PF / f"{split}.npz", allow_pickle=True)
    cid, enc = np.asarray(d["case_id"]), np.asarray(d["encounter_index"])
    return d["wake_enstrophy"], {(str(c), int(e)): i for i, (c, e) in enumerate(zip(cid, enc))}, d["impact_frame"]


def _ce(d):
    cid = d["case_ids"] if "case_ids" in d.files else d["case_id"]
    enc = d["encounter_indices"] if "encounter_indices" in d.files else d["encounter_index"]
    return np.asarray(cid), np.asarray(enc)


def ae_d64_truncation_test(h=8):
    obs_tr, idx_tr, _ = load_obs("train")
    obs_tb, idx_tb, _ = load_obs("test_b")
    # encoded train (fit), per-frame
    dtr = np.load(LAT / AE[64] / "train.npz", allow_pickle=True)
    ztr = dtr["z_full"].astype(np.float64); ctr, etr = _ce(dtr)
    Xtr, ytr = [], []
    for i, (c, e) in enumerate(zip(ctr, etr)):
        k = (str(c), int(e))
        if k not in idx_tr:
            continue
        yt = obs_tr[idx_tr[k]]; m = np.isfinite(yt)
        Xtr.append(ztr[i][m]); ytr.append(yt[m])
    Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr)
    # rolled test_b at impact+h
    dr = np.load(ROLL / "ae_d64" / "test_b.npz", allow_pickle=True)
    zr = dr["z_full"].astype(np.float64); cr, er = _ce(dr); imp = dr["impact_frame"]
    Xtb, ytb = [], []
    for i, (c, e) in enumerate(zip(cr, er)):
        k = (str(c), int(e))
        if k not in idx_tb:
            continue
        f = int(imp[i]) + h
        if f >= zr.shape[1]:
            continue
        yt = obs_tb[idx_tb[k]][f]
        if not np.isfinite(yt):
            continue
        Xtb.append(zr[i, f]); ytb.append(yt)
    Xtb = np.array(Xtb); ytb = np.array(ytb)
    print(f"\nPart B: AE d64 wake forecast (h={h}) projected onto top-k encoded PCs")
    print(f"  {'k':>5s} {'R^2':>8s}")
    for k in (2, 4, 8, 16, 32, 64):
        pca = PCA(n_components=k).fit(Xtr)
        pipe = Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=1.0))])
        pipe.fit(pca.transform(Xtr), ytr)
        r2 = r2_score(ytb, pipe.predict(pca.transform(Xtb)))
        print(f"  {k:>5d} {r2:>+8.2f}")


def main():
    print("Part A: encoded-latent spectrum (train, per-frame)")
    print(f"  {'family':6s} {'d':>4s} {'PR':>6s} {'near-null':>10s} {'top5_var':>9s}")
    for d in (4, 8, 16, 64):
        for name, tags in (("JEPA", JEPA), ("AE", AE)):
            pr, nn, top5, dim = spectrum(tags[d])
            print(f"  {name:6s} {d:>4d} {pr:>6.2f} {nn:>7d}/{dim:<3d} {top5:>9.3f}")
    ae_d64_truncation_test(h=8)


if __name__ == "__main__":
    main()
