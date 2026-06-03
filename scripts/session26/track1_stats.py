#!/usr/bin/env python3
"""Session 26 Track 1: statistics hardening (case-level dependence, multiplicity, floor).

CPU-only, no training. Re-analysis of cached latents/rollouts/DNS metrics + the cached
topology/OT/scale JSONs. Three deliverables, written to outputs/session26/stats/:

  1a Case-level dependence:
     - The wake-enstrophy paired comparison (predictive JEPA d=64 vs reconstructive
       Fukami d=64, test_b, H=16) in BOTH modes, recomputed with a case-clustered
       block bootstrap (>=10000 resamples; cases resampled with replacement, encounters
       averaged within case), plus a case-level paired statistic (Wilcoxon signed-rank on
       per-case mean improvements, and a linear mixed-effects model with case random effect).
     - The topology Mann-Whitney, the transport Spearman, and the scale-decomposition
       correlation re-checked at the case level (a case contributes multiple encounters).

  1b Multiple comparisons:
     - Holm-Bonferroni over the twelve paired sign-test p-values of Table 10. Verifies the
       forecast wake result (p=0.044) does NOT survive family-wide correction while the
       representational wake result (p=0.0014) does. Wake enstrophy is the pre-registered
       PRIMARY endpoint; the other five observables are secondary.

  1c Predictive vs the conditioning floor:
     - Per-encounter predictive-forecast wake error vs the frame-matched c-only KRR floor
       error (Table 5, H=16), paired and case-clustered. Same for representational closure.

Outputs: outputs/session26/stats/{wake_paired,holm,floor,topology,transport,scale}.json
and is summarised by the companion stats_summary.md (written by hand from these).
Test C is NOT used for any selection here; it is reported only for the case-count line.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "session26" / "stats"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO / "scripts" / "session20"))
sys.path.insert(0, str(REPO / "scripts" / "session21"))
sys.path.insert(0, str(REPO / "scripts" / "session23"))

from scipy.stats import binomtest, mannwhitneyu, wilcoxon  # noqa: E402

# Reuse the verified closure machinery (same ridge probe, rollouts, DNS metrics behind
# Tables 4/10). exp_closure_r2 defines the paths and probe; we replicate the per-encounter
# loader but keep the (case_id, encounter) key so we can cluster by case.
from exp_closure_r2 import (  # noqa: E402
    DNS_METRICS_PATH, LATENTS_ROOT, ROLLOUTS_ROOT,
    apply_probe, fit_probes, match_index,
)

OBSERVABLES = ["CL", "CD", "Iy", "wake_enstrophy", "circ_pos", "circ_neg"]
_OBS_MAP = {"CL": "C_L", "CD": "C_D", "Iy": "I_y", "wake_enstrophy": "wake_enstrophy",
            "circ_pos": "circulation_pos", "circ_neg": "circulation_neg"}
_FAMILY_TAG = {"jepa_d64": "jepa_d64_test1_noBN", "fukami_d64": "fukami_d64_noBN"}
_MODE_KEY = {"repr": "z_dns", "forecast": "z_markov"}
_SPLIT = "test_b"
_H = 16
RNG = np.random.default_rng(0)
N_BOOT_ENC = 2000
N_BOOT_CASE = 10000

_DNS = np.load(DNS_METRICS_PATH, allow_pickle=True)
_probe_cache: dict[str, dict] = {}


def load_abs_error_with_cases(observable: str, mode: str, family: str):
    """Per-encounter |probe-DNS| on test_b at H=16, plus aligned case_id and enc arrays.

    Returns (err, case_ids, encs) all length n, in canonical sorted (case_id, encounter)
    order so arrays are index-aligned across families. Same probe/data as Tables 4/10.
    """
    metric = _OBS_MAP[observable]
    tag = _FAMILY_TAG[family]
    zkey = _MODE_KEY[mode]
    if tag not in _probe_cache:
        _probe_cache[tag] = fit_probes(LATENTS_ROOT / f"latents_{tag}", _DNS)
    probe = _probe_cache[tag][metric]
    blob = np.load(ROLLOUTS_ROOT / f"rollouts_{tag}" / f"{_SPLIT}.npz", allow_pickle=True)
    z = blob[zkey].astype(np.float64)
    cid = blob["case_ids"] if "case_ids" in blob.files else blob["case_id"]
    ei = blob["encounter_indices"] if "encounter_indices" in blob.files else blob["encounter_index"]
    impact = blob["impact_frame"].astype(int)
    di = match_index(cid, ei, _DNS[f"{_SPLIT}_case_id"], _DNS[f"{_SPLIT}_encounter_index"])
    d = z.shape[2]
    err = {}
    for i in range(len(cid)):
        if di[i] < 0:
            continue
        te = int(impact[i]) + _H
        if te >= z.shape[1]:
            continue
        yp = float(apply_probe(z[i, te].reshape(1, d), probe)[0])
        yt = float(_DNS[f"{_SPLIT}_{metric}"][di[i], te])
        err[(str(cid[i]), int(ei[i]))] = abs(yp - yt)
    keys = sorted(err.keys())
    e = np.array([err[k] for k in keys], dtype=float)
    cids = np.array([k[0] for k in keys])
    encs = np.array([k[1] for k in keys])
    return e, cids, encs


# --------------------------------------------------------------------------------------
# Case-clustered statistics
# --------------------------------------------------------------------------------------
def case_means(values: np.ndarray, case_ids: np.ndarray):
    """Return (unique_cases, per_case_mean) averaging values within each case."""
    uc = np.array(sorted(set(case_ids.tolist())))
    cm = np.array([values[case_ids == c].mean() for c in uc])
    return uc, cm


def case_cluster_bootstrap(delta: np.ndarray, case_ids: np.ndarray,
                           n_boot: int = N_BOOT_CASE, rng=RNG):
    """Block bootstrap that resamples CASES with replacement.

    Reports two CIs for the paired improvement delta = err_recon - err_jepa (>0 = JEPA
    better): (i) on the encounter-mean (cluster bootstrap: pool the encounters of the
    resampled cases, take their mean); (ii) on the case-mean (equal weight per case:
    average the resampled cases' within-case means). The point estimates are the observed
    encounter-mean and the observed mean-of-case-means.
    """
    uc = np.array(sorted(set(case_ids.tolist())))
    by_case = {c: delta[case_ids == c] for c in uc}
    case_mean_arr = np.array([by_case[c].mean() for c in uc])
    enc_mean = float(delta.mean())
    casemean_point = float(case_mean_arr.mean())
    n_cases = len(uc)
    boot_enc = np.empty(n_boot)
    boot_case = np.empty(n_boot)
    for b in range(n_boot):
        pick = uc[rng.integers(0, n_cases, size=n_cases)]
        pooled = np.concatenate([by_case[c] for c in pick])
        boot_enc[b] = pooled.mean()
        boot_case[b] = np.mean([by_case[c].mean() for c in pick])
    return {
        "n_encounters": int(delta.size),
        "n_cases": int(n_cases),
        "enc_mean": enc_mean,
        "enc_mean_ci": [float(np.percentile(boot_enc, 2.5)), float(np.percentile(boot_enc, 97.5))],
        "case_mean": casemean_point,
        "case_mean_ci": [float(np.percentile(boot_case, 2.5)), float(np.percentile(boot_case, 97.5))],
        "per_case_mean_delta": {c: float(v.mean()) for c, v in by_case.items()},
    }


def encounter_bootstrap_ci(delta: np.ndarray, n_boot: int = N_BOOT_ENC, rng=RNG):
    n = delta.size
    bm = np.array([delta[rng.integers(0, n, size=n)].mean() for _ in range(n_boot)])
    return [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]


def sign_test_one_sided(jepa_err: np.ndarray, recon_err: np.ndarray):
    k = int(np.sum(jepa_err < recon_err))
    n_eff = int(np.sum(jepa_err != recon_err))
    p = float(binomtest(k, n_eff, 0.5, alternative="greater").pvalue) if n_eff > 0 else float("nan")
    return k, n_eff, p


def case_level_paired_stats(case_mean_delta: np.ndarray):
    """Wilcoxon signed-rank + sign test on per-case mean improvements (10 cases)."""
    cmd = np.asarray(case_mean_delta, dtype=float)
    n = cmd.size
    k = int(np.sum(cmd > 0))
    try:
        w_stat, w_p = wilcoxon(cmd, alternative="greater")
        w_stat, w_p = float(w_stat), float(w_p)
    except Exception as exc:  # all-zero or degenerate
        w_stat, w_p = float("nan"), f"error:{exc}"
    sign_p = float(binomtest(k, n, 0.5, alternative="greater").pvalue)
    return {"n_cases": n, "cases_jepa_better": k, "wilcoxon_stat": w_stat,
            "wilcoxon_p_one_sided": w_p, "sign_p_one_sided": sign_p,
            "median_case_mean_delta": float(np.median(cmd))}


def mixedlm_intercept(delta: np.ndarray, case_ids: np.ndarray):
    """Random-intercept model delta ~ 1 + (1|case). Two-sided p on the mean improvement."""
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
        df = pd.DataFrame({"delta": delta, "case": case_ids})
        m = smf.mixedlm("delta ~ 1", df, groups=df["case"]).fit(reml=False, method="lbfgs")
        return {"intercept": float(m.params["Intercept"]),
                "p_two_sided": float(m.pvalues["Intercept"]),
                "group_var": float(m.cov_re.iloc[0, 0])}
    except Exception as exc:
        return {"error": str(exc)}


def holm(pvals: dict[str, float], alpha: float = 0.05):
    """Holm-Bonferroni step-down. Returns per-key adjusted p and survive flag."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj = {}
    running = 0.0
    for rank, (key, p) in enumerate(items):
        a = (m - rank) * p
        running = max(running, a)  # enforce monotone non-decreasing adjusted p
        adj[key] = min(running, 1.0)
    survive = {k: bool(adj[k] <= alpha) for k in pvals}
    return adj, survive


# ======================================================================================
# 1a + 1b: wake paired, all observables, both modes
# ======================================================================================
def run_paired():
    rows = {}
    sign_pvals = {}
    for mode in ("repr", "forecast"):
        for obs in OBSERVABLES:
            j, jc, _ = load_abs_error_with_cases(obs, mode, "jepa_d64")
            r, rc, _ = load_abs_error_with_cases(obs, mode, "fukami_d64")
            assert np.array_equal(jc, rc), f"case order mismatch {obs} {mode}"
            delta = r - j
            k, n_eff, sp = sign_test_one_sided(j, r)
            cc = case_cluster_bootstrap(delta, jc)
            _, cmd = case_means(delta, jc)
            clp = case_level_paired_stats(cmd)
            key = f"{mode}/{obs}"
            sign_pvals[key] = sp
            entry = {
                "mode": mode, "observable": obs,
                "mean_jepa_err": float(j.mean()), "mean_recon_err": float(r.mean()),
                "enc_mean_delta": float(delta.mean()),
                "enc_bootstrap_ci": encounter_bootstrap_ci(delta),
                "sign_k": k, "sign_n_eff": n_eff, "sign_p_one_sided": sp,
                "case_clustered": cc,
                "case_level_paired": clp,
            }
            if obs == "wake_enstrophy":
                entry["mixedlm"] = mixedlm_intercept(delta, jc)
            rows[key] = entry
    return rows, sign_pvals


# ======================================================================================
# 1c: predictive forecast vs conditioning floor (and representational vs floor)
# ======================================================================================
def run_floor():
    from exp_conditioning_floor_plus import load_impact_table, select_krr
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score

    tab = load_impact_table(horizon=_H)
    j_obs = OBSERVABLES.index("wake_enstrophy")
    metric = "wake_enstrophy"
    obs_col = list(("C_L", "C_D", "I_y", "wake_enstrophy", "circulation_pos", "circulation_neg")).index(metric)

    c_tr = tab["train"]["c"]
    y_tr = tab["train"]["obs"][:, obs_col]
    groups = tab["train"]["case_id"]
    alpha, gamma = select_krr(c_tr, y_tr, groups)
    sx = StandardScaler().fit(c_tr)
    m = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma)
    m.fit(sx.transform(c_tr), y_tr)

    c_tb = tab["test_b"]["c"]
    y_tb = tab["test_b"]["obs"][:, obs_col]
    cid_tb = tab["test_b"]["case_id"]
    enc_tb = tab["test_b"]["enc"]
    floor_pred = m.predict(sx.transform(c_tb))
    floor_r2 = float(r2_score(y_tb, floor_pred))
    # floor per-encounter error keyed by (case_id, enc)
    floor_err = {(str(cid_tb[i]), int(enc_tb[i])): abs(float(floor_pred[i] - y_tb[i]))
                 for i in range(len(cid_tb))}

    out = {"floor_recipe": {"alpha": alpha, "gamma": gamma},
           "floor_wake_test_b_R2": floor_r2}
    for mode in ("forecast", "repr"):
        j, jc, je = load_abs_error_with_cases("wake_enstrophy", mode, "jepa_d64")
        # align floor errors to the JEPA canonical (case_id, enc) order
        fe = np.array([floor_err[(str(c), int(e))] for c, e in zip(jc, je)])
        delta = fe - j  # >0 means JEPA beats the floor
        k, n_eff, sp = sign_test_one_sided(j, fe)
        cc = case_cluster_bootstrap(delta, jc)
        _, cmd = case_means(delta, jc)
        clp = case_level_paired_stats(cmd)
        out[mode] = {
            "mean_jepa_err": float(j.mean()), "mean_floor_err": float(fe.mean()),
            "enc_mean_delta_floor_minus_jepa": float(delta.mean()),
            "enc_bootstrap_ci": encounter_bootstrap_ci(delta),
            "sign_k_jepa_better": k, "sign_n_eff": n_eff, "sign_p_one_sided": sp,
            "case_clustered": cc, "case_level_paired": clp,
        }
    return out


# ======================================================================================
# 1a third bullet: topology, transport, scale at the case level
# ======================================================================================
def run_topology():
    d = json.load(open(REPO / "outputs/session20/persistent_homology/persistent_homology.json"))
    pe = d["test_b"]["per_encounter"]
    cids = np.array([r["case_id"] for r in pe])
    jep = np.array([r["jepa_dns_nsig"] for r in pe], dtype=float)
    fuk = np.array([r["fukami_dns_nsig"] for r in pe], dtype=float)
    # encounter-level Mann-Whitney (reproduce the headline)
    u_enc, p_enc = mannwhitneyu(jep, fuk, alternative="less")  # JEPA fewer generators
    # case-level: per-case mean nsig, paired (same cases) Wilcoxon on the difference
    uc, jep_cm = case_means(jep, cids)
    _, fuk_cm = case_means(fuk, cids)
    diff = fuk_cm - jep_cm  # >0 means JEPA fewer generators (cleaner loop)
    try:
        w_stat, w_p = wilcoxon(diff, alternative="greater")
        w_stat, w_p = float(w_stat), float(w_p)
    except Exception as exc:
        w_stat, w_p = float("nan"), f"error:{exc}"
    u_case, p_case = mannwhitneyu(jep_cm, fuk_cm, alternative="less")
    return {
        "metric": "significant H1 generator count, simulation-encoded latent (z_dns)",
        "encounter_level": {"n": int(jep.size), "jepa_median": float(np.median(jep)),
                            "fukami_median": float(np.median(fuk)),
                            "mannwhitney_U": float(u_enc), "mannwhitney_p_one_sided": float(p_enc)},
        "case_level": {"n_cases": int(uc.size),
                       "jepa_median_case_mean": float(np.median(jep_cm)),
                       "fukami_median_case_mean": float(np.median(fuk_cm)),
                       "cases_jepa_fewer": int(np.sum(diff > 0)),
                       "wilcoxon_stat": w_stat, "wilcoxon_p_one_sided": w_p,
                       "mannwhitney_U": float(u_case), "mannwhitney_p_one_sided": float(p_case)},
    }


def run_transport():
    d = json.load(open(REPO / "outputs/session20/ot/ot_results.json"))
    pm = d["d_ii"]["per_method"]
    jep = np.array(pm["jepa_d64"]["spearman_per_encounter"], dtype=float)
    fuk = np.array(pm["fukami"]["spearman_per_encounter"], dtype=float)
    # canonical case order from the same reference latents file the OT script used
    ref = np.load(REPO / "outputs/session14/latents/S12_E_d64/test_b.npz", allow_pickle=True)
    cids = np.array([str(c) for c in ref["case_id"]])
    assert cids.size == jep.size, f"OT order mismatch {cids.size} vs {jep.size}"
    margin = jep - fuk
    uc, jep_cm = case_means(jep, cids)
    _, fuk_cm = case_means(fuk, cids)
    _, marg_cm = case_means(margin, cids)
    try:
        w_stat, w_p = wilcoxon(marg_cm, alternative="greater")
        w_stat, w_p = float(w_stat), float(w_p)
    except Exception as exc:
        w_stat, w_p = float("nan"), f"error:{exc}"
    return {
        "metric": "per-encounter Spearman(OT frame-frame, latent frame-frame); JEPA d64 vs Fukami",
        "encounter_level": {"n": int(jep.size), "jepa_mean": float(jep.mean()),
                            "fukami_mean": float(fuk.mean()), "margin_mean": float(margin.mean())},
        "case_level": {"n_cases": int(uc.size), "jepa_case_mean": float(jep_cm.mean()),
                       "fukami_case_mean": float(fuk_cm.mean()), "margin_case_mean": float(marg_cm.mean()),
                       "cases_jepa_higher": int(np.sum(marg_cm > 0)),
                       "wilcoxon_stat": w_stat, "wilcoxon_p_one_sided": w_p},
    }


def run_scale():
    import importlib
    sd = importlib.import_module("exp_scale_decomposition")
    wake = sd.load_wake_mask()  # same mask the script's main() uses
    rng = np.random.default_rng(0)
    res = sd.process_split("test_b", wake, rng)
    ens_L = res["_ens_L"]
    cids = np.array([str(c) for c in res["_case_ids"]])
    idx16 = sd.PLUS16_IDX
    dns = ens_L["dns"][:, idx16]
    out = {"metric": "large-scale (sigma/c=0.05) wake-enstrophy corr(pred, DNS) at impact+16",
           "stage": "impact+16", "encounter_level": {}, "case_level": {}}
    uc = np.array(sorted(set(cids.tolist())))
    dns_cm = np.array([dns[cids == c].mean() for c in uc])
    for key, label in sd.METHODS:
        pv = ens_L[label][:, idx16]
        r_enc = float(np.corrcoef(pv, dns)[0, 1])
        pv_cm = np.array([pv[cids == c].mean() for c in uc])
        r_case = float(np.corrcoef(pv_cm, dns_cm)[0, 1])
        out["encounter_level"][label] = {"n": int(pv.size), "corr": r_enc}
        out["case_level"][label] = {"n_cases": int(uc.size), "corr": r_case}
    return out


def main():
    print("Track 1 statistics hardening...")
    paired, sign_pvals = run_paired()
    adj, survive = holm(sign_pvals)
    holm_out = {"n_tests": len(sign_pvals),
                "tests": {k: {"raw_p": sign_pvals[k], "holm_p": adj[k], "survives_0.05": survive[k]}
                          for k in sign_pvals},
                "primary_endpoint": "wake_enstrophy",
                "note": ("Wake enstrophy is the pre-registered PRIMARY endpoint (reported "
                         "uncorrected). Holm over all 12 is reported so the family-wide "
                         "statement is explicit: representational wake survives, forecast wake does not.")}
    floor = run_floor()
    topo = run_topology()
    transport = run_transport()
    try:
        scale = run_scale()
    except Exception as exc:
        scale = {"error": f"{type(exc).__name__}: {exc}"}

    (OUT / "wake_paired.json").write_text(json.dumps(paired, indent=2))
    (OUT / "holm.json").write_text(json.dumps(holm_out, indent=2))
    (OUT / "floor.json").write_text(json.dumps(floor, indent=2))
    (OUT / "topology.json").write_text(json.dumps(topo, indent=2))
    (OUT / "transport.json").write_text(json.dumps(transport, indent=2))
    (OUT / "scale.json").write_text(json.dumps(scale, indent=2))

    # console summary
    print("\n=== 1a WAKE ENSTROPHY paired, case-clustered ===")
    for mode in ("repr", "forecast"):
        e = paired[f"{mode}/wake_enstrophy"]
        cc = e["case_clustered"]; clp = e["case_level_paired"]
        print(f"[{mode}] enc-mean dErr {e['enc_mean_delta']:+.1f} enc-CI {e['enc_bootstrap_ci']}")
        print(f"   case-clustered enc-mean CI {cc['enc_mean_ci']}  case-mean {cc['case_mean']:+.1f} CI {cc['case_mean_ci']}")
        print(f"   case-level Wilcoxon p={clp['wilcoxon_p_one_sided']}, {clp['cases_jepa_better']}/{clp['n_cases']} cases, sign p={clp['sign_p_one_sided']:.3g}")
        print(f"   mixedlm {e.get('mixedlm')}")
    print("\n=== 1b HOLM over 12 ===")
    for k in sorted(sign_pvals, key=lambda x: sign_pvals[x]):
        print(f"   {k:28s} raw={sign_pvals[k]:.2e} holm={adj[k]:.3f} survive={survive[k]}")
    print("\n=== 1c FLOOR (wake) ===")
    print(f"   floor R2={floor['floor_wake_test_b_R2']:.3f} recipe={floor['floor_recipe']}")
    for mode in ("forecast", "repr"):
        f = floor[mode]; cc = f["case_clustered"]; clp = f["case_level_paired"]
        print(f"   [{mode} vs floor] dErr(floor-jepa) {f['enc_mean_delta_floor_minus_jepa']:+.1f} "
              f"case-clustered enc-CI {cc['enc_mean_ci']} Wilcoxon p={clp['wilcoxon_p_one_sided']} "
              f"{clp['cases_jepa_better']}/{clp['n_cases']}")
    print("\n=== 1a TOPOLOGY ===")
    print(f"   enc MW p={topo['encounter_level']['mannwhitney_p_one_sided']:.2e} "
          f"(median {topo['encounter_level']['jepa_median']} vs {topo['encounter_level']['fukami_median']})")
    print(f"   case Wilcoxon p={topo['case_level']['wilcoxon_p_one_sided']} MW p={topo['case_level']['mannwhitney_p_one_sided']:.2e} "
          f"({topo['case_level']['cases_jepa_fewer']}/{topo['case_level']['n_cases']} cases)")
    print("\n=== 1a TRANSPORT ===")
    print(f"   enc margin {transport['encounter_level']['margin_mean']:+.3f}; "
          f"case margin {transport['case_level']['margin_case_mean']:+.3f} "
          f"Wilcoxon p={transport['case_level']['wilcoxon_p_one_sided']} "
          f"({transport['case_level']['cases_jepa_higher']}/{transport['case_level']['n_cases']})")
    print("\n=== 1a SCALE ===")
    print(f"   {json.dumps(scale)[:300]}")
    print(f"\nWrote JSONs to {OUT}")


if __name__ == "__main__":
    main()
