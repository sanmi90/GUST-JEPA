#!/usr/bin/env python3
"""
session21_paired_closure_stats.py  --  Tier-1 statistic for the JFM-bar revision.

WHY THIS EXISTS
---------------
The current manuscript reports the wake closure as two MARGINAL estimates whose
bootstrap CIs are wide (forecast wake R2 = 0.449 with CI [-0.96, 0.79]), then
concedes the per-observable separation "is not individually significant at this
sample size." The review (Tier 1) is correct that this undersells the data: the
wide marginal CI is dominated by encounter-to-encounter difficulty that is SHARED
by every model. Pairing the comparison per encounter cancels that shared
difficulty and should collapse the interval.

This script reframes the comparison as a PAIRED predictive-minus-reconstructive
difference. It is post-processing only; no retraining.

For each observable and each mode it reports:
  * mean per-encounter absolute error for JEPA and for the reconstructive baseline
  * mean paired difference  delta_i = err_recon_i - err_jepa_i  (>0 means JEPA better)
  * 2000-resample bootstrap 95% CI of the mean paired difference (paired resampling)
  * one-sided sign test: k = #{ err_jepa_i < err_recon_i }, exact binomial p
  * a 'significant' flag (CI excludes 0) for the abstract sentence

It then prints (i) a human-readable table, (ii) a LaTeX-ready block to drop into
Table 3 / Section 4.1, and (iii) the one-line headline for the reframed abstract.

DEPENDENCIES: numpy, scipy (>= 1.7 for stats.binomtest; a fallback is included).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

try:
    from scipy.stats import binomtest  # scipy >= 1.7
    def _sign_p(k: int, n: int) -> float:
        return float(binomtest(k, n, 0.5, alternative="greater").pvalue)
except Exception:  # pragma: no cover - old scipy fallback
    from scipy.stats import binom_test  # type: ignore
    def _sign_p(k: int, n: int) -> float:
        return float(binom_test(k, n, 0.5, alternative="greater"))

# Reproducible, single seed. Bump only if you report seed sensitivity.
RNG = np.random.default_rng(0)
N_BOOT = 2000

# Order and display names of the six observables (matches Tables 2 and 3).
OBSERVABLES = ["CL", "CD", "Iy", "wake_enstrophy", "circ_pos", "circ_neg"]
PRETTY = {
    "CL": r"$C_L$", "CD": r"$C_D$", "Iy": r"$I_y$",
    "wake_enstrophy": "wake enstrophy", "circ_pos": "circ. pos.",
    "circ_neg": "circ. neg.",
}
# The headline observable for the abstract.
HEADLINE_OBS = "wake_enstrophy"

# ============================================================================
# DATA INTERFACE  --  THE ONLY PART YOU WIRE TO THE REPO.
# ----------------------------------------------------------------------------
# Return, for a given (observable, mode, family), a 1-D array of per-encounter
# ABSOLUTE errors on the n=42 test_b encounters, aligned by encounter index so
# that index i is the SAME held-out encounter for jepa and for fukami.
#
#   mode = "repr"     : probe applied to the simulation-encoded latent  (Table 2)
#   mode = "forecast" : probe applied to the Markov rollout from impact  (Table 3)
#   family in {"jepa_d64", "fukami_d64"}
#
# These arrays already exist wherever Tables 2 and 3 were aggregated; the means
# you currently report ARE means of exactly these vectors. This loader only has
# to expose them unaggregated and index-aligned. Wire it to your cache, e.g.
#   data = np.load(f".../closure_per_encounter/{mode}_{family}.npz")
#   return np.abs(data[observable])          # shape (42,)
# ============================================================================
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts" / "session20"))
# Reuse the verified Session 20 closure machinery (same ridge probe, same
# rollouts and DNS metrics that Tables 2/3 were aggregated from).
from exp_closure_r2 import (  # noqa: E402
    DNS_METRICS_PATH, LATENTS_ROOT, ROLLOUTS_ROOT,
    apply_probe, fit_probes, match_index,
)

_SPLIT = "test_b"
_H = 16  # impact + H frames, the horizon Tables 2/3 report
_OBS_MAP = {"CL": "C_L", "CD": "C_D", "Iy": "I_y",
            "wake_enstrophy": "wake_enstrophy",
            "circ_pos": "circulation_pos", "circ_neg": "circulation_neg"}
_FAMILY_TAG = {"jepa_d64": "jepa_d64_test1_noBN", "fukami_d64": "fukami_d64_noBN"}
_MODE_KEY = {"repr": "z_dns", "forecast": "z_markov"}
_DNS = np.load(DNS_METRICS_PATH, allow_pickle=True)
_probe_cache: dict[str, dict] = {}


def load_per_encounter_abs_error(observable: str, mode: str, family: str) -> np.ndarray:
    """Per-encounter |probe - DNS| on test_b, returned in a canonical (case_id,
    encounter) order so the array is index-aligned across families for pairing.

    mode='repr' uses the simulation-encoded latent (Table 2); mode='forecast'
    uses the Markov rollout from impact (Table 3). Probe and data are exactly
    those behind the reported closure means.
    """
    metric = _OBS_MAP[observable]
    tag = _FAMILY_TAG[family]
    zkey = _MODE_KEY[mode]
    if tag not in _probe_cache:  # fit the ridge probe once per family
        _probe_cache[tag] = fit_probes(LATENTS_ROOT / f"latents_{tag}", _DNS)
    probe = _probe_cache[tag][metric]

    blob = np.load(ROLLOUTS_ROOT / f"rollouts_{tag}" / f"{_SPLIT}.npz", allow_pickle=True)
    z = blob[zkey].astype(np.float64)
    cid = blob["case_ids"] if "case_ids" in blob.files else blob["case_id"]
    ei = blob["encounter_indices"] if "encounter_indices" in blob.files else blob["encounter_index"]
    impact = blob["impact_frame"].astype(int)
    di = match_index(cid, ei, _DNS[f"{_SPLIT}_case_id"], _DNS[f"{_SPLIT}_encounter_index"])
    d = z.shape[2]

    err: dict[tuple, float] = {}
    for i in range(len(cid)):
        if di[i] < 0:
            continue
        te = int(impact[i]) + _H
        if te >= z.shape[1]:
            continue
        yp = float(apply_probe(z[i, te].reshape(1, d), probe)[0])
        yt = float(_DNS[f"{_SPLIT}_{metric}"][di[i], te])
        err[(str(cid[i]), int(ei[i]))] = abs(yp - yt)
    # canonical order: sorted (case_id, encounter) -> identical across families
    return np.array([err[k] for k in sorted(err.keys())], dtype=float)


@dataclass
class PairedResult:
    observable: str
    mode: str
    n: int
    mean_jepa: float
    mean_recon: float
    mean_delta: float       # recon - jepa, > 0 means JEPA has the smaller error
    ci_lo: float
    ci_hi: float
    k_jepa_wins: int
    n_eff: int              # encounters with a strict winner (ties dropped)
    sign_p_one_sided: float

    @property
    def significant(self) -> bool:
        # Paired bootstrap CI of the mean improvement excludes zero on the
        # 'JEPA better' side.
        return self.ci_lo > 0.0


def paired_bootstrap(jepa_err: np.ndarray, recon_err: np.ndarray,
                     n_boot: int = N_BOOT, rng: np.random.Generator = RNG) -> PairedResult:
    """Paired bootstrap of (recon_err - jepa_err) plus a one-sided sign test."""
    jepa_err = np.asarray(jepa_err, dtype=float).ravel()
    recon_err = np.asarray(recon_err, dtype=float).ravel()
    if jepa_err.shape != recon_err.shape:
        raise ValueError(f"shape mismatch: {jepa_err.shape} vs {recon_err.shape}")
    n = jepa_err.size
    delta = recon_err - jepa_err                      # paired, per encounter

    # Resample ENCOUNTERS (the pairing is preserved because we index delta).
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = delta[idx].mean(axis=1)
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

    # Distribution-free companion: one-sided sign test, ties dropped.
    k = int(np.sum(jepa_err < recon_err))
    n_eff = int(np.sum(jepa_err != recon_err))
    p = _sign_p(k, n_eff) if n_eff > 0 else float("nan")

    return PairedResult(
        observable="", mode="", n=n,
        mean_jepa=float(jepa_err.mean()), mean_recon=float(recon_err.mean()),
        mean_delta=float(delta.mean()), ci_lo=float(ci_lo), ci_hi=float(ci_hi),
        k_jepa_wins=k, n_eff=n_eff, sign_p_one_sided=p,
    )


def run_all() -> list[PairedResult]:
    rows: list[PairedResult] = []
    for mode in ("repr", "forecast"):
        for obs in OBSERVABLES:
            j = load_per_encounter_abs_error(obs, mode, "jepa_d64")
            r = load_per_encounter_abs_error(obs, mode, "fukami_d64")
            res = paired_bootstrap(j, r)
            res.observable, res.mode = obs, mode
            rows.append(res)
    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_human(rows: list[PairedResult]) -> None:
    hdr = (f"{'mode':9} {'observable':16} {'err_JEPA':>9} {'err_recon':>10} "
           f"{'delta':>8} {'95% CI':>20} {'sign':>10} {'sig':>4}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        ci = f"[{r.ci_lo:+.3g}, {r.ci_hi:+.3g}]"
        sign = f"{r.k_jepa_wins}/{r.n_eff} p={r.sign_p_one_sided:.1e}"
        print(f"{r.mode:9} {r.observable:16} {r.mean_jepa:9.3g} {r.mean_recon:10.3g} "
              f"{r.mean_delta:+8.3g} {ci:>20} {sign:>10} {'Y' if r.significant else 'n':>4}")


def print_latex(rows: list[PairedResult]) -> None:
    """A drop-in LaTeX block: paired improvement + sign test, by mode/observable."""
    print("\n% ---- paste into Section 4.1 / Table 3 (paired closure) ----")
    print(r"\begin{tabular}{llrrl}")
    print(r"\toprule")
    print(r"mode & observable & $\Delta$err (recon$-$JEPA) & 95\% CI & sign test \\")
    print(r"\midrule")
    for r in rows:
        ci = f"[{r.ci_lo:+.3g}, {r.ci_hi:+.3g}]"
        sign = f"{r.k_jepa_wins}/{r.n_eff}, $p={r.sign_p_one_sided:.1e}$"
        star = r"\,$^{*}$" if r.significant else ""
        print(f"{r.mode} & {PRETTY[r.observable]} & ${r.mean_delta:+.3g}${star} & {ci} & {sign} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"% $^{*}$ paired bootstrap CI excludes zero (JEPA better at 95\%).")


def print_abstract_headline(rows: list[PairedResult]) -> None:
    fc = next(r for r in rows if r.observable == HEADLINE_OBS and r.mode == "forecast")
    rp = next(r for r in rows if r.observable == HEADLINE_OBS and r.mode == "repr")
    print("\n% ---- one-line headline for the reframed abstract ----")
    for tag, r in (("forecast", fc), ("representational", rp)):
        verdict = "significant" if r.significant else "not individually significant"
        print(f"% [{tag}] JEPA has the smaller wake-enstrophy error on "
              f"{r.k_jepa_wins} of {r.n} held-out encounters "
              f"(paired mean improvement {r.mean_delta:+.3g}, 95% CI "
              f"[{r.ci_lo:+.3g}, {r.ci_hi:+.3g}], one-sided sign p="
              f"{r.sign_p_one_sided:.1e}); {verdict}.")


if __name__ == "__main__":
    rows = run_all()           # raises until you wire load_per_encounter_abs_error
    print_human(rows)
    print_latex(rows)
    print_abstract_headline(rows)
