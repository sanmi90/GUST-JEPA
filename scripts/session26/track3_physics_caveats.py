#!/usr/bin/env python3
"""Session 26 Track 3: physical-definition caveats (traceable numbers).

3a. The mid-plane 2D vorticity impulse I_y is NOT the impulse-theorem lift: recompute the DNS
    correlation r(dI_y/dt, C_L) from the cached observables (it should be near zero, not ~0.95).
3b. chi_3D reference: read the committed D147 output so the in-distribution out-of-plane fraction
    (~0.20 for |G|<=3, omega_z, post-impact) is cited from a committed file, not asserted.
3c. The circulation threshold omega_c is arbitrary: recompute the DNS circulation at
    omega_c in {0.5, 1, 2} from the cache, report (i) the cross-threshold collinearity of the
    observable and (ii) the representational closure R^2 (JEPA d64 vs Fukami d64, test_b, H=16)
    refit at each threshold, so the closure can be shown insensitive to the choice.

CPU only, no training. Reads the preprocessed per-encounter cache and the cached latents.
Output: outputs/session26/physics_caveats/{impulse,chi3d_ref,omega_c_sensitivity}.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import h5py

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "session26" / "physics_caveats"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO / "scripts" / "session17"))
sys.path.insert(0, str(REPO / "scripts" / "session20"))

import exp2_dns_physical_metrics as exp2  # noqa: E402
from exp_closure_r2 import (  # noqa: E402
    DNS_METRICS_PATH, LATENTS_ROOT, ROLLOUTS_ROOT,
    fit_ridge, apply_probe, match_index,
)

THRESHOLDS = [0.5, 1.0, 2.0]
H = 16
FAMILY_TAG = {"jepa_d64": "jepa_d64_test1_noBN", "fukami_d64": "fukami_d64_noBN"}


# ----------------------------------------------------------------------------- 3a
def impulse_lift_correlation():
    d = np.load(DNS_METRICS_PATH, allow_pickle=True)
    out = {"definition": "r(dI_y/dt, C_L); the 2D impulse-lift relation would give |r|~1"}
    for split in ("train", "test_b", "test_c"):
        Iy = np.asarray(d[f"{split}_I_y"], dtype=float)
        CL = np.asarray(d[f"{split}_C_L"], dtype=float)
        dIy = np.gradient(Iy, axis=1)
        r_pool = float(np.corrcoef(dIy.ravel(), CL.ravel())[0, 1])
        rs = [float(np.corrcoef(dIy[i], CL[i])[0, 1])
              for i in range(Iy.shape[0]) if dIy[i].std() > 0 and CL[i].std() > 0]
        out[split] = {"n_enc": int(Iy.shape[0]), "r_pooled": r_pool,
                      "r_per_encounter_mean": float(np.mean(rs))}
    return out


# ----------------------------------------------------------------------------- 3b
def chi3d_reference():
    g = json.load(open(REPO / "outputs/session23/chi3d/chi3d_gate.json"))
    pe = g["post_impact"]["wz"]
    return {"source": "outputs/session23/chi3d/chi3d_gate.json (post_impact, omega_z)",
            "median_le3": pe["median_le3"], "median_g4": pe["median_g4"],
            "ratio_g4_over_le3": pe["ratio_g4_over_le3"],
            "note": "in-distribution (|G|<=3) out-of-plane omega_z enstrophy fraction ~0.20; "
                    "rises to ~0.555 at |G|=4 (n_le3=74, n_g4=4 cases)."}


# ----------------------------------------------------------------------------- 3c
def circ_at_thresholds(omega_clean: np.ndarray) -> dict:
    """Per-frame signed wake circulation at each omega_c, on the same cleaned field exp2 uses."""
    xx = exp2.X_GRID[:, None]
    yy = exp2.Y_GRID[None, :]
    wake = (xx >= exp2.WAKE_X_MIN) & (xx <= exp2.WAKE_X_MAX) & (np.abs(yy) <= exp2.WAKE_Y_MAX)
    ow = omega_clean * wake[None, :, :]
    res = {}
    for thr in THRESHOLDS:
        pos = (ow * (ow > thr)).sum(axis=(1, 2)) * exp2.DX * exp2.DY
        neg = (ow * (ow < -thr)).sum(axis=(1, 2)) * exp2.DX * exp2.DY
        res[thr] = {"circulation_pos": pos.astype(np.float64), "circulation_neg": neg.astype(np.float64)}
    return res


def recompute_dns_circ(splits):
    pipe = exp2.OmegaPipeline.from_manifest(exp2.OMEGA_MANIFEST)
    store = {}  # split -> {(cid,enc): {thr: {obs:(T,)}}}, plus impact
    for split in splits:
        encs = exp2.gather_split_encounters(split)
        d = {}
        for e in encs:
            with h5py.File(e["path"], "r") as f:
                omega_raw = np.asarray(f["omega_z"], dtype=np.float32)
                impact = int(f.attrs.get("impact_frame_estimate", exp2.DEFAULT_IMPACT_FRAME))
            omega_clean = pipe.preprocess_raw(omega_raw, e["case_id"], e["k"]).astype(np.float32)
            d[(e["case_id"], int(e["k"]))] = {"impact": impact, "circ": circ_at_thresholds(omega_clean)}
        store[split] = d
        print(f"[3c] recomputed DNS circ for {split}: {len(d)} encounters")
    return store


def closure_r2_for_threshold(dns_store, thr, family, observable):
    """Repr closure R^2 at impact+H for one (threshold, family, observable). Probe = ridge on
    train z_full -> recomputed DNS circ@thr; eval on test_b z_dns at impact+H."""
    tag = FAMILY_TAG[family]
    # build train design from latents z_full and recomputed dns circ
    tr = np.load(LATENTS_ROOT / f"latents_{tag}" / "train.npz", allow_pickle=True)
    zf = tr["z_full"].astype(np.float64)  # (n, T, d)
    cid = tr["case_ids"] if "case_ids" in tr.files else tr["case_id"]
    ei = tr["encounter_indices"] if "encounter_indices" in tr.files else tr["encounter_index"]
    n, T, dlat = zf.shape
    Z_rows, y_rows = [], []
    for i in range(n):
        key = (str(cid[i]), int(ei[i]))
        if key not in dns_store["train"]:
            continue
        yv = dns_store["train"][key]["circ"][thr][observable]  # (T,)
        Tm = min(T, yv.shape[0])
        Z_rows.append(zf[i, :Tm]); y_rows.append(yv[:Tm])
    Z = np.concatenate(Z_rows); y = np.concatenate(y_rows)
    probe = fit_ridge(Z, y, alpha=1.0)
    # eval on test_b z_dns at impact+H
    blob = np.load(ROLLOUTS_ROOT / f"rollouts_{tag}" / "test_b.npz", allow_pickle=True)
    z = blob["z_dns"].astype(np.float64)
    bcid = blob["case_ids"] if "case_ids" in blob.files else blob["case_id"]
    bei = blob["encounter_indices"] if "encounter_indices" in blob.files else blob["encounter_index"]
    bimp = blob["impact_frame"].astype(int)
    yp, yt = [], []
    for i in range(len(bcid)):
        key = (str(bcid[i]), int(bei[i]))
        if key not in dns_store["test_b"]:
            continue
        te = int(bimp[i]) + H
        circ = dns_store["test_b"][key]["circ"][thr][observable]
        if te >= z.shape[1] or te >= circ.shape[0]:
            continue
        yp.append(float(apply_probe(z[i, te].reshape(1, dlat), probe)[0]))
        yt.append(float(circ[te]))
    yp = np.asarray(yp); yt = np.asarray(yt)
    ss_res = float(((yp - yt) ** 2).sum()); ss_tot = float(((yt - yt.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def omega_c_sensitivity():
    dns_store = recompute_dns_circ(["train", "test_b"])
    # cross-threshold collinearity of the DNS observable on test_b (pooled over enc x frame)
    collin = {}
    for obs in ("circulation_pos", "circulation_neg"):
        vecs = {thr: np.concatenate([v["circ"][thr][obs] for v in dns_store["test_b"].values()])
                for thr in THRESHOLDS}
        collin[obs] = {f"{a}_vs_{b}": float(np.corrcoef(vecs[a], vecs[b])[0, 1])
                       for a in THRESHOLDS for b in THRESHOLDS if a < b}
    # closure R^2 at each threshold
    r2 = {}
    for obs in ("circulation_pos", "circulation_neg"):
        r2[obs] = {}
        for thr in THRESHOLDS:
            r2[obs][str(thr)] = {fam: closure_r2_for_threshold(dns_store, thr, fam, obs)
                                 for fam in FAMILY_TAG}
    return {"thresholds": THRESHOLDS, "collinearity_test_b": collin,
            "repr_closure_R2_test_b_H16": r2,
            "note": "circulation closure (repr R^2) and the JEPA-minus-Fukami advantage are "
                    "compared across omega_c to test sensitivity to the threshold choice."}


def main():
    res = {"impulse_lift": impulse_lift_correlation(),
           "chi3d_reference": chi3d_reference(),
           "omega_c_sensitivity": omega_c_sensitivity()}
    (OUT / "impulse.json").write_text(json.dumps(res["impulse_lift"], indent=2))
    (OUT / "chi3d_ref.json").write_text(json.dumps(res["chi3d_reference"], indent=2))
    (OUT / "omega_c_sensitivity.json").write_text(json.dumps(res["omega_c_sensitivity"], indent=2))

    print("\n=== 3a impulse-lift ===")
    for s in ("train", "test_b", "test_c"):
        print(f"   {s}: r_pooled={res['impulse_lift'][s]['r_pooled']:+.4f} "
              f"r_per_enc_mean={res['impulse_lift'][s]['r_per_encounter_mean']:+.4f}")
    print("\n=== 3b chi_3D (post-impact omega_z) ===")
    c = res["chi3d_reference"]; print(f"   median |G|<=3 = {c['median_le3']:.3f}, |G|=4 = {c['median_g4']:.3f}")
    print("\n=== 3c omega_c sensitivity ===")
    s = res["omega_c_sensitivity"]
    print("   collinearity:", json.dumps(s["collinearity_test_b"]))
    for obs in s["repr_closure_R2_test_b_H16"]:
        print(f"   {obs}:")
        for thr in s["repr_closure_R2_test_b_H16"][obs]:
            v = s["repr_closure_R2_test_b_H16"][obs][thr]
            print(f"      omega_c={thr}: JEPA {v['jepa_d64']:+.3f}  Fukami {v['fukami_d64']:+.3f}  "
                  f"adv {v['jepa_d64']-v['fukami_d64']:+.3f}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
