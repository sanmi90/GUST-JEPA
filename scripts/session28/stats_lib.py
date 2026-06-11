"""Session 28 statistics library (master plan A7; ports session26/track1_stats.py).

Importable, side-effect-free statistical functions for the v2.1 closure matrix
and all Phase B/C analyses: paired per-encounter differences, case-clustered
block bootstrap, case-level paired tests, mixed-effects intercept, Holm
step-down, and case-permutation p-values.

FAITHFULNESS CONTRACT: the functions below are byte-identical in behaviour to
scripts/session26/track1_stats.py (D165). tests/test_stats_lib.py replays the
exact global-RNG call sequence of the v2 run and asserts every cell of the
committed outputs/session26/stats/wake_paired.json reproduces exactly. Do not
change defaults or draw order without re-anchoring that regression test.

Conventions (frozen in configs/eval_protocol_v2p1.yaml):
    bootstrap unit = encounter (n = 2000), cluster unit = case (n = 10000),
    Holm over the 12-test family, one-sided sign tests where direction is
    pre-registered (delta > 0 means the predictive family is better).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest, wilcoxon

N_BOOT_ENC = 2000
N_BOOT_CASE = 10000


def case_means(values: np.ndarray, case_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (unique_cases, per_case_mean) averaging values within each case.

    Unique cases are sorted lexicographically, matching track1_stats.py.
    """
    uc = np.array(sorted(set(case_ids.tolist())))
    cm = np.array([values[case_ids == c].mean() for c in uc])
    return uc, cm


def case_cluster_bootstrap(
    delta: np.ndarray,
    case_ids: np.ndarray,
    n_boot: int = N_BOOT_CASE,
    rng: np.random.Generator | None = None,
) -> dict:
    """Block bootstrap that resamples CASES with replacement (track1, verbatim).

    Two CIs for the paired improvement delta (>0 = predictive family better):
    (i) on the encounter-mean (pool encounters of the resampled cases), and
    (ii) on the case-mean (equal weight per case). Point estimates are the
    observed encounter-mean and the observed mean-of-case-means.
    """
    if rng is None:
        rng = np.random.default_rng(0)
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
        "enc_mean_ci": [
            float(np.percentile(boot_enc, 2.5)),
            float(np.percentile(boot_enc, 97.5)),
        ],
        "case_mean": casemean_point,
        "case_mean_ci": [
            float(np.percentile(boot_case, 2.5)),
            float(np.percentile(boot_case, 97.5)),
        ],
        "per_case_mean_delta": {c: float(v.mean()) for c, v in by_case.items()},
    }


def encounter_bootstrap_ci(
    delta: np.ndarray,
    n_boot: int = N_BOOT_ENC,
    rng: np.random.Generator | None = None,
) -> list[float]:
    """Percentile bootstrap CI of the mean, resampling encounters (track1, verbatim)."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = delta.size
    bm = np.array([delta[rng.integers(0, n, size=n)].mean() for _ in range(n_boot)])
    return [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]


def sign_test_one_sided(err_a: np.ndarray, err_b: np.ndarray) -> tuple[int, int, float]:
    """One-sided sign test that err_a < err_b (a better). Returns (k, n_eff, p)."""
    k = int(np.sum(err_a < err_b))
    n_eff = int(np.sum(err_a != err_b))
    p = (
        float(binomtest(k, n_eff, 0.5, alternative="greater").pvalue)
        if n_eff > 0
        else float("nan")
    )
    return k, n_eff, p


def case_level_paired_stats(case_mean_delta: np.ndarray) -> dict:
    """Wilcoxon signed-rank + sign test on per-case mean improvements (track1, verbatim)."""
    cmd = np.asarray(case_mean_delta, dtype=float)
    n = cmd.size
    k = int(np.sum(cmd > 0))
    try:
        w_stat, w_p = wilcoxon(cmd, alternative="greater")
        w_stat, w_p = float(w_stat), float(w_p)
    except Exception as exc:  # all-zero or degenerate
        w_stat, w_p = float("nan"), f"error:{exc}"
    sign_p = float(binomtest(k, n, 0.5, alternative="greater").pvalue)
    return {
        "n_cases": n,
        "cases_jepa_better": k,
        "wilcoxon_stat": w_stat,
        "wilcoxon_p_one_sided": w_p,
        "sign_p_one_sided": sign_p,
        "median_case_mean_delta": float(np.median(cmd)),
    }


def mixedlm_intercept(delta: np.ndarray, case_ids: np.ndarray) -> dict:
    """Random-intercept model delta ~ 1 + (1|case); two-sided p on the mean improvement."""
    try:
        import pandas as pd
        import statsmodels.formula.api as smf

        df = pd.DataFrame({"delta": delta, "case": case_ids})
        m = smf.mixedlm("delta ~ 1", df, groups=df["case"]).fit(reml=False, method="lbfgs")
        return {
            "intercept": float(m.params["Intercept"]),
            "p_two_sided": float(m.pvalues["Intercept"]),
            "group_var": float(m.cov_re.iloc[0, 0]),
        }
    except Exception as exc:
        return {"error": str(exc)}


def holm(pvals: dict[str, float], alpha: float = 0.05) -> tuple[dict, dict]:
    """Holm-Bonferroni step-down (track1, verbatim): adjusted p + survive flags."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj: dict[str, float] = {}
    running = 0.0
    for rank, (key, p) in enumerate(items):
        a = (m - rank) * p
        running = max(running, a)  # enforce monotone non-decreasing adjusted p
        adj[key] = min(running, 1.0)
    survive = {k: bool(adj[k] <= alpha) for k in pvals}
    return adj, survive


def case_permutation_p(
    x: np.ndarray,
    y: np.ndarray,
    case_ids: np.ndarray,
    stat_fn,
    n_perm: int = 10000,
    rng: np.random.Generator | None = None,
    alternative: str = "greater",
) -> dict:
    """Case-level permutation p for a dependence statistic stat_fn(x, y).

    Permutes y between CASES (within-case structure preserved): the case-mean
    blocks of y are reassigned to the cases of x, so the null is "no
    case-level association" while encounter-level autocorrelation is kept.
    Used for the Fig-8-style Spearman trends (plan B6).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    uc = np.array(sorted(set(case_ids.tolist())))
    idx_by_case = {c: np.where(case_ids == c)[0] for c in uc}
    sizes = {c: idx_by_case[c].size for c in uc}
    # permutation only valid when case blocks are exchangeable in size
    obs = float(stat_fn(x, y))
    count = 0
    valid = 0
    for _ in range(n_perm):
        perm = rng.permutation(uc)
        if any(sizes[a] != sizes[b] for a, b in zip(uc, perm)):
            # unequal block sizes: permute case-mean residuals instead
            perm_y = y.copy()
            means = {c: y[idx_by_case[c]].mean() for c in uc}
            shuffled = {a: means[b] for a, b in zip(uc, perm)}
            for c in uc:
                perm_y[idx_by_case[c]] = (
                    y[idx_by_case[c]] - means[c] + shuffled[c]
                )
        else:
            perm_y = np.empty_like(y)
            for a, b in zip(uc, perm):
                perm_y[idx_by_case[a]] = y[idx_by_case[b]]
        s = float(stat_fn(x, perm_y))
        valid += 1
        if alternative == "greater" and s >= obs:
            count += 1
        elif alternative == "less" and s <= obs:
            count += 1
        elif alternative == "two-sided" and abs(s) >= abs(obs):
            count += 1
    p = (count + 1) / (valid + 1)
    return {"stat": obs, "p": float(p), "n_perm": int(valid), "alternative": alternative}
