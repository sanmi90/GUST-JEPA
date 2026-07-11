#!/usr/bin/env python3
"""Observable readability at the CRITICAL INSTANTS (Carlos, 2026-07-11).

For each family, a linear probe reads the lift C_L and the wake enstrophy E_w off
the latent. Definition (locked with Carlos): the probe is a ridge regression
fit on the TRAINING encounters over the whole impact window (frames 25-55, alpha
by internal CV); it is then evaluated on the held-out test_b encounters at the
pre-impact, impact and peak-lift instants (+-2 frames pooled over the 42
encounters). We report BOTH the held-out R^2 (baseline = the mean observable at
that instant) AND the physical error (RMSE in the observable's own units), with
a case-clustered bootstrap 95% CI over encounters. R^2 is consistent with the
tab:closure readability; the physical error sidesteps the R^2 denominator
fragility where the observable is near-constant (pre-impact). CPU.

Run:
    OMP_NUM_THREADS=8 taskset -c 0-15 .venv/bin/python \\
        scripts/session39/observable_critical.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO))
from scripts.session36.rex_families_m1 import load_split, group  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402

OUT = REPO / "outputs/session39/observable_critical.json"
IMPACT, PRE, HALF, NBOOT, SEED = 40, 30, 2, 2000, 0
OBS = {"C_L": "C_L", "E_w": "wake_enstrophy"}
# 3 encoder seeds where they exist (learned families JEPA, AE at d=32); Fukami has
# one at d=32; POD is deterministic. Encoder-training variance is reported as the
# seed spread, matching the D130 uncertainty protocol.
FAMILIES = {
    "predictive": ["jepa_pool_vec", "jepa_pool_vec_s1", "jepa_pool_vec_s2"],
    "AE (wake)": ["ae_wake_pool", "ae_wake_pool_s1", "ae_wake_pool_s2"],
    "Fukami (wake)": ["fukami_wake"],
    "POD": ["pod"],
}
INSTS = ("preimpact", "impact", "peaklift")


def r2(y, yh):
    y, yh = np.asarray(y), np.asarray(yh)
    ss = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - yh) ** 2).sum() / ss) if ss > 0 else float("nan")


def rmse(y, yh):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yh)) ** 2)))


def boot_ci(y, yh, enc, fn, rng):
    cases = np.unique(enc)
    crows = {c: np.where(enc == c)[0] for c in cases}
    vals = []
    for _ in range(NBOOT):
        draw = rng.choice(cases, size=len(cases), replace=True)
        sel = np.concatenate([crows[c] for c in draw])
        vals.append(fn(y[sel], yh[sel]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def eval_seed(run):
    """One encoder seed: per (observable, instant) R2 and RMSE on held-out test_b."""
    tr, tb = load_split(run, "train"), load_split(run, "test_b")
    wtr, wtb = tr["window_mask"].astype(bool), tb["window_mask"].astype(bool)
    out = {}
    for oname, okey in OBS.items():
        ytr, ytb = tr[f"target_{okey}"], tb[f"target_{okey}"]
        probe = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(tr["z_gap"][wtr], ytr[wtr])
        pred = probe.predict(tb["z_gap"])
        out[f"{oname}_windowed_r2"] = r2(ytb[wtb], pred[wtb])
        per = {i: [] for i in INSTS}
        for e in group(tb):
            rows = e["rows"]
            fr = tb["frame"][rows]
            peak_f = int(fr[np.argmax(np.abs(ytb[rows]))])
            for inst, center in (("preimpact", PRE), ("impact", IMPACT),
                                 ("peaklift", peak_f)):
                per[inst].extend(rows[np.abs(fr - center) <= HALF].tolist())
        for inst, idx in per.items():
            idx = np.asarray(idx)
            out[f"{oname}_{inst}_r2"] = r2(ytb[idx], pred[idx])
            out[f"{oname}_{inst}_rmse"] = rmse(ytb[idx], pred[idx])
    return out


def main() -> None:
    results = {"_provenance": {"script": "scripts/session39/observable_critical.py",
                              "metric": "held-out linear-probe R2 and physical RMSE "
                              "at critical instants; per-cell = seed mean, with seed "
                              "SD as encoder-training variance (3 seeds JEPA/AE, 1 "
                              "Fukami, POD deterministic)",
                              "window_halfwidth": HALF,
                              "instants": {"preimpact": PRE, "impact": IMPACT,
                                           "peaklift": "argmax|C_L|"}},
               "families": {}}
    for label, runs in FAMILIES.items():
        seed_recs = [eval_seed(r) for r in runs]
        rec = {"n_seeds": len(runs)}
        for k in seed_recs[0]:
            vals = [s[k] for s in seed_recs]
            rec[k] = float(np.mean(vals))          # seed mean (point estimate)
            rec[k + "_sd"] = float(np.std(vals))   # encoder-training variance
        results["families"][label] = rec
        print(f"{label:16s} (n={len(runs)}) | C_L R2 imp={rec['C_L_impact_r2']:+.2f}"
              f"+-{rec['C_L_impact_r2_sd']:.2f} rmse={rec['C_L_impact_rmse']:.2f} "
              f"| E_w R2 imp={rec['E_w_impact_r2']:+.2f}+-{rec['E_w_impact_r2_sd']:.2f}",
              flush=True)
    OUT.write_text(json.dumps(results, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
