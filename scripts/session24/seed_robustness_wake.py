#!/usr/bin/env python3
"""Session 24: 3-seed robustness of the d=64 JEPA-vs-Fukami wake-enstrophy result.

REPRESENTATION mode (z_dns): encode the held-out DNS field per frame and probe the
latent at impact+H. No predictor / rollout is needed because z_dns == per-frame
z_full from the encoder (verified bit-identical to the production rollout z_dns).

For each seed s in {0,1,2}:
  1. JEPA seed s: fit wake ridge probe on TRAIN z_full -> DNS wake_enstrophy, predict
     test_b wake at impact+H, per-encounter abs error e_JEPA^s (n=42), held-out R^2.
  2. Fukami seed s: same -> e_Fukami^s, held-out R^2.
  3. Paired improvement per seed: Delta_e^s = e_Fukami^s - e_JEPA^s per encounter
     (canonical sorted (case_id, encounter) order so the two families are index
     aligned). Report paired mean, k-of-42 where JEPA error smaller, one-sided sign p.

Probe and DNS source are IDENTICAL to the production closure
(scripts/session20/exp_closure_r2.py fit_probes; DNS wake_enstrophy from
outputs/session17/exp2/dns_physical_metrics.npz). H=16 is the headline horizon.

Seed-to-checkpoint map:
  JEPA   seed s -> outputs/runs/session14/thrust6/jepa_d64_seed{s}/encoder/checkpoint_iter020000.pt
  Fukami seed0 -> outputs/session18/exp_b1/fukami_ae_d64/checkpoint_iter020000.pt (production headline)
  Fukami seed1 -> outputs/runs/session23/AE_d64_seed1/checkpoint_iter020000.pt
  Fukami seed2 -> outputs/runs/session23/AE_d64_seed2/checkpoint_iter020000.pt
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "session20"))
from exp_closure_r2 import (  # noqa: E402
    DNS_METRICS_PATH, fit_probes, apply_probe, match_index,
)

try:
    from scipy.stats import binomtest
    def sign_p(k: int, n: int) -> float:
        return float(binomtest(k, n, 0.5, alternative="greater").pvalue)
except Exception:  # pragma: no cover
    from scipy.stats import binom_test  # type: ignore
    def sign_p(k: int, n: int) -> float:
        return float(binom_test(k, n, 0.5, alternative="greater"))

METRIC = "wake_enstrophy"
SPLIT = "test_b"
H = 16
N_BOOT = 2000
RNG_SEED = 0

SR = REPO / "outputs" / "session24" / "seed_robustness_wake"
B1 = REPO / "outputs" / "session18" / "exp_b1"
# Seed s -> (jepa latents dir, fukami latents dir). Each dir holds train.npz +
# test_b.npz with per-frame z_full (= z_dns for representation mode).
#
# JEPA seeds are the 3 Thrust-6 retrains (the canonical seed-variance triple per
# CLAUDE.md). Fukami seed0 is the PRODUCTION headline latents (latents_fukami_d64,
# bit-identical to the headline rollouts_fukami_d64_noBN z_dns and reproducing the
# published R^2 = -0.406); seed1/seed2 are the Session 23 AE retrains, freshly
# encoded with the identical pipeline. (NOTE: the on-disk
# fukami_ae_d64/checkpoint_iter020000.pt no longer reproduces latents_fukami_d64
# bit-for-bit -- a deterministic re-encode lands ~5 latent units away -- so we use
# the stored production latents that the paper Table 2 actually reports, not a
# re-encode of the possibly-superseded checkpoint.)
SEED_DIRS = {
    0: (SR / "jepa_latents" / "seed0", B1 / "latents_fukami_d64"),
    1: (SR / "jepa_latents" / "seed1", SR / "fukami_latents_seed1"),
    2: (SR / "jepa_latents" / "seed2", SR / "fukami_latents_seed2"),
}
# Published single-seed reference: production JEPA (S12_E_d64) vs production Fukami.
# Must reproduce the headline (R^2 0.754 / -0.406, paired +43.1, 31/42).
REF_DIRS = (B1 / "latents_jepa_d64", B1 / "latents_fukami_d64")


def _cid_ei(blob):
    cid = blob["case_id"] if "case_id" in blob.files else blob["case_ids"]
    ei = blob["encounter_index"] if "encounter_index" in blob.files else blob["encounter_indices"]
    return cid, ei


def per_encounter(latents_dir: Path, dns) -> dict:
    """Return {(case_id, enc): abs_error} and aligned (yp, yt) lists for test_b at impact+H."""
    probe = fit_probes(latents_dir, dns)[METRIC]
    blob = np.load(latents_dir / f"{SPLIT}.npz", allow_pickle=True)
    z = blob["z_full"].astype(np.float64)
    cid, ei = _cid_ei(blob)
    impact = blob["impact_frame"].astype(int)
    di = match_index(cid, ei, dns[f"{SPLIT}_case_id"], dns[f"{SPLIT}_encounter_index"])
    d = z.shape[2]
    err: dict[tuple, float] = {}
    pred: dict[tuple, tuple] = {}
    for i in range(len(cid)):
        if di[i] < 0:
            continue
        te = int(impact[i]) + H
        if te >= z.shape[1]:
            continue
        yp = float(apply_probe(z[i, te].reshape(1, d), probe)[0])
        yt = float(dns[f"{SPLIT}_{METRIC}"][di[i], te])
        key = (str(cid[i]), int(ei[i]))
        err[key] = abs(yp - yt)
        pred[key] = (yp, yt)
    return err, pred


def held_out_r2(pred: dict) -> float:
    yp = np.array([v[0] for k, v in sorted(pred.items())])
    yt = np.array([v[1] for k, v in sorted(pred.items())])
    ss_res = float(((yp - yt) ** 2).sum())
    ss_tot = float(((yt - yt.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def paired(jepa_err: dict, fuk_err: dict, rng) -> dict:
    keys = sorted(set(jepa_err) & set(fuk_err))
    je = np.array([jepa_err[k] for k in keys])
    fe = np.array([fuk_err[k] for k in keys])
    delta = fe - je  # > 0 means JEPA smaller error
    n = len(keys)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    boot = delta[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    k = int(np.sum(je < fe))
    n_eff = int(np.sum(je != fe))
    return dict(
        n=n, mean_jepa=float(je.mean()), mean_fukami=float(fe.mean()),
        mean_delta=float(delta.mean()), ci_lo=float(lo), ci_hi=float(hi),
        k_jepa_wins=k, n_eff=n_eff, sign_p=sign_p(k, n_eff) if n_eff else float("nan"),
    )


def main() -> None:
    dns = np.load(DNS_METRICS_PATH, allow_pickle=True)
    rng = np.random.default_rng(RNG_SEED)
    rows = []

    # --- Published single-seed reference (production JEPA vs production Fukami) ---
    jdir, fdir = REF_DIRS
    je, jp = per_encounter(jdir, dns)
    fe, fp = per_encounter(fdir, dns)
    ref_pr = paired(je, fe, np.random.default_rng(RNG_SEED))
    print("SANITY (published single-seed: production S12_E JEPA vs production Fukami):")
    print(f"  JEPA wake R^2 = {held_out_r2(jp):.4f} (target ~0.754)   "
          f"Fukami wake R^2 = {held_out_r2(fp):.4f} (target ~-0.406)")
    print(f"  paired mean improvement = {ref_pr['mean_delta']:+.2f} (target ~+43.1)   "
          f"k/42 = {ref_pr['k_jepa_wins']}/{ref_pr['n']} (target ~31/42)   "
          f"sign p = {ref_pr['sign_p']:.2e}")
    print()
    jepa_r2s, fuk_r2s, deltas = [], [], []
    print(f"{'seed':4} {'JEPA_R2':>8} {'Fuk_R2':>8} {'err_J':>7} {'err_F':>7} "
          f"{'delta':>7} {'k/42':>6} {'sign_p':>9} {'CI':>20}")
    print("-" * 90)
    for s, (jdir, fdir) in SEED_DIRS.items():
        je, jp = per_encounter(jdir, dns)
        fe, fp = per_encounter(fdir, dns)
        jr2 = held_out_r2(jp)
        fr2 = held_out_r2(fp)
        pr = paired(je, fe, rng)
        jepa_r2s.append(jr2); fuk_r2s.append(fr2); deltas.append(pr["mean_delta"])
        ci = f"[{pr['ci_lo']:+.2f},{pr['ci_hi']:+.2f}]"
        print(f"{s:<4} {jr2:8.4f} {fr2:8.4f} {pr['mean_jepa']:7.2f} {pr['mean_fukami']:7.2f} "
              f"{pr['mean_delta']:+7.2f} {pr['k_jepa_wins']:>2}/{pr['n']:<3} "
              f"{pr['sign_p']:9.2e} {ci:>20}")
        rows.append(dict(
            seed=s, n=pr["n"], jepa_wake_r2=jr2, fukami_wake_r2=fr2,
            err_jepa=pr["mean_jepa"], err_fukami=pr["mean_fukami"],
            paired_mean_improvement=pr["mean_delta"],
            paired_ci_lo=pr["ci_lo"], paired_ci_hi=pr["ci_hi"],
            k_jepa_wins=pr["k_jepa_wins"], n_eff=pr["n_eff"], sign_p_one_sided=pr["sign_p"],
        ))

    jepa_r2s = np.array(jepa_r2s); fuk_r2s = np.array(fuk_r2s); deltas = np.array(deltas)
    print("-" * 90)
    print(f"JEPA   wake R^2 across seeds: {jepa_r2s.mean():.4f} +- {jepa_r2s.std(ddof=0):.4f} "
          f"(ddof=1 {jepa_r2s.std(ddof=1):.4f})  values={np.round(jepa_r2s,4).tolist()}")
    print(f"Fukami wake R^2 across seeds: {fuk_r2s.mean():.4f} +- {fuk_r2s.std(ddof=0):.4f} "
          f"(ddof=1 {fuk_r2s.std(ddof=1):.4f})  values={np.round(fuk_r2s,4).tolist()}")
    gate = bool(np.all(deltas > 0))
    print(f"\nGATE: paired mean improvement (Fukami - JEPA error) > 0 in all 3 seeds? "
          f"{'YES' if gate else 'NO'}  deltas={np.round(deltas,2).tolist()}")
    print(f"      median paired improvement = {np.median(deltas):+.2f}")

    out_csv = SR / "seed_robustness_wake.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
        # summary row
        w.writerow(dict(
            seed="mean_pm_std", n=42,
            jepa_wake_r2=f"{jepa_r2s.mean():.4f}+-{jepa_r2s.std(ddof=0):.4f}",
            fukami_wake_r2=f"{fuk_r2s.mean():.4f}+-{fuk_r2s.std(ddof=0):.4f}",
            err_jepa="", err_fukami="",
            paired_mean_improvement=f"median={np.median(deltas):.2f}",
            paired_ci_lo="", paired_ci_hi="",
            k_jepa_wins=f"gate_all3_positive={gate}", n_eff="", sign_p_one_sided="",
        ))
    print(f"\n[wrote] {out_csv}")


if __name__ == "__main__":
    main()
