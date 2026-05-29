"""Conditioning-only control baseline for the B1 / Section 5 narrative.

Question: how much of the per-observable forecast at impact is determined
by the gust parameters c = (G, D, Y) ALONE, with no latent input? If a
KRR(c) -> observable(impact) regression already reaches R² ≈ 0.8 on
test_b, then each baseline's reported advantage above that is the
contribution attributable to the latent representation.

Pipeline:
  1. Load DNS physical metrics at impact frame for train, test_b, test_c
     (use outputs/session17/exp2/dns_physical_metrics.npz; the impact
     frame index is in the JEPA d=64 latents NPZ).
  2. For each of the 6 B1 observables (C_L, C_D, I_y, wake_enstrophy,
     circulation_pos, circulation_neg), fit KRR-RBF on
     c_train -> obs_train(impact), evaluate on test_b and test_c.
  3. Write outputs/session18/exp_b1_test3/conditioning_only_baseline.csv
     with the floor R² for each observable on each split.
  4. Compare side-by-side to JEPA d=64's reported B1 train probe R²
     (read from probe_train_r2.csv) to quantify "JEPA - parameters-alone".

Note: this is the *floor* the latent must clear to be considered useful.
It is NOT the same comparison as B1 (which is "predict from rolled-out
latent"); it is the upper bound on what a "parameters-only" forecaster
could achieve.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
import warnings; warnings.filterwarnings("ignore")

REPO = Path("/home/carlos/GUST-JEPA")

DNS_METRICS = REPO / "outputs/session17/exp2/dns_physical_metrics.npz"
JEPA_LATENTS = REPO / "outputs/session18/exp_b1/latents_jepa_d64"  # for c=(G,D,Y) + impact_frame
PROBE_R2_CSV = REPO / "outputs/session18/exp_b1_test3/probe_train_r2.csv"

OBSERVABLES = [
    "C_L", "C_D", "I_y", "wake_enstrophy", "circulation_pos", "circulation_neg"
]


def r2(yhat, y, eps=1e-9):
    ss_res = ((yhat - y) ** 2).sum(0)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0)
    return 1.0 - ss_res / np.maximum(ss_tot, eps)


def load_split_metrics_at_impact(split: str):
    """Returns (cids, eis, impact_frames, c=(n,3), obs_at_impact=(n, 6))."""
    npz = np.load(JEPA_LATENTS / f"{split}.npz", allow_pickle=True)
    cids = npz["case_id"].astype(str)
    eis = npz["encounter_index"].astype(int)
    impact = npz["impact_frame"].astype(int)
    c = np.stack([npz["G"], npz["D"], npz["Y"]], axis=1).astype(np.float32)

    # DNS NPZ has FLAT keys: split_<metric> for each metric and per-split arrays.
    dns = np.load(DNS_METRICS, allow_pickle=True)
    dns_cids = dns[f"{split}_case_id"].astype(str)
    dns_eis = dns[f"{split}_encounter_index"].astype(int)
    # Build lookup (case_id, encounter_index) -> row index in DNS arrays
    lookup = {(c0, int(e0)): i for i, (c0, e0) in enumerate(zip(dns_cids, dns_eis))}

    obs = np.zeros((len(cids), len(OBSERVABLES)), dtype=np.float32)
    series_by_obs = {k: dns[f"{split}_{k}"] for k in OBSERVABLES}
    for i, (cid, ei, imp) in enumerate(zip(cids, eis, impact)):
        key = (cid, int(ei))
        if key not in lookup:
            obs[i] = np.nan
            continue
        dns_i = lookup[key]
        for j, k in enumerate(OBSERVABLES):
            ser = series_by_obs[k][dns_i]  # (T,) per-encounter time series
            obs[i, j] = float(ser[min(int(imp), len(ser) - 1)])
    return cids, eis, impact, c, obs


def main():
    print("Loading DNS metrics + JEPA latents (for c=(G,D,Y))...")
    parts = {}
    for split in ("train", "test_b", "test_c"):
        r = load_split_metrics_at_impact(split)
        if r is None:
            sys.exit(f"FATAL: split {split} missing")
        parts[split] = r

    cids_tr, eis_tr, _, c_tr, obs_tr = parts["train"]
    cids_tb, eis_tb, _, c_tb, obs_tb = parts["test_b"]
    cids_tc, eis_tc, _, c_tc, obs_tc = parts["test_c"]
    print(f"  train n={len(c_tr)}, test_b n={len(c_tb)}, test_c n={len(c_tc)}")

    rows = []
    print()
    print(f"{'observable':<18} {'train R²':>10} {'test_b R²':>11} {'test_c R²':>11}")
    print("-" * 55)
    for j, obs_name in enumerate(OBSERVABLES):
        y_tr = obs_tr[:, j]
        y_tb = obs_tb[:, j]
        y_tc = obs_tc[:, j]

        mask_tr = ~np.isnan(y_tr)
        mask_tb = ~np.isnan(y_tb)
        mask_tc = ~np.isnan(y_tc)

        sx = StandardScaler().fit(c_tr[mask_tr])
        m = KernelRidge(alpha=0.1, kernel="rbf", gamma=0.5)
        m.fit(sx.transform(c_tr[mask_tr]), y_tr[mask_tr])

        yhat_tr = m.predict(sx.transform(c_tr[mask_tr]))
        yhat_tb = m.predict(sx.transform(c_tb[mask_tb])) if mask_tb.any() else np.array([])
        yhat_tc = m.predict(sx.transform(c_tc[mask_tc])) if mask_tc.any() else np.array([])

        r2_tr = float(r2(yhat_tr[:, None], y_tr[mask_tr][:, None])[0]) if mask_tr.any() else np.nan
        r2_tb = float(r2(yhat_tb[:, None], y_tb[mask_tb][:, None])[0]) if mask_tb.any() else np.nan
        r2_tc = float(r2(yhat_tc[:, None], y_tc[mask_tc][:, None])[0]) if mask_tc.any() else np.nan

        rows.append({
            "observable": obs_name,
            "n_train": int(mask_tr.sum()),
            "n_test_b": int(mask_tb.sum()),
            "n_test_c": int(mask_tc.sum()),
            "train_R2_parameters_only": r2_tr,
            "test_b_R2_parameters_only": r2_tb,
            "test_c_R2_parameters_only": r2_tc,
        })
        print(f"{obs_name:<18} {r2_tr:>+10.3f} {r2_tb:>+11.3f} {r2_tc:>+11.3f}")

    out_csv = REPO / "outputs/session18/exp_b1_test3/conditioning_only_baseline.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out_csv}")

    # Side-by-side against JEPA's reported train probe R² (probe_train_r2.csv).
    if PROBE_R2_CSV.exists():
        import re
        jepa_tr = {}
        for line in PROBE_R2_CSV.read_text().splitlines()[1:]:
            parts2 = line.split(",")
            if len(parts2) >= 5 and parts2[0] == "jepa_d64_test1_noBN":
                jepa_tr[parts2[3]] = float(parts2[4])
        print()
        print("=== Side by side: (G,D,Y)-only floor vs JEPA d=64 train probe R² ===")
        print(f"{'observable':<18} {'(G,D,Y) only':>14} {'JEPA d=64':>11} {'JEPA - c':>10}")
        print("-" * 60)
        for r in rows:
            obs = r["observable"]
            j = jepa_tr.get(obs, np.nan)
            delta = j - r["train_R2_parameters_only"]
            print(f"{obs:<18} {r['train_R2_parameters_only']:>+14.3f} {j:>+11.3f} {delta:>+10.3f}")


if __name__ == "__main__":
    main()
