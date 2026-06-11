"""Regression test: scripts/session28/stats_lib.py reproduces D165 exactly (plan A7).

Replays the exact global-RNG call sequence of scripts/session26/track1_stats.py
(one default_rng(0); for mode in (repr, forecast) for obs in the 6 observables:
case_cluster_bootstrap with 10000 draws, then encounter_bootstrap_ci with 2000
draws) through the PORTED functions, and asserts every cell of the committed
outputs/session26/stats/wake_paired.json reproduces to float precision.

Needs the v2 artifacts on disk (session17 DNS metrics + session18 latents and
rollouts); skipped wherever they are absent. Runtime ~2-4 min (24000 bootstrap
draws x 12 cells); run it targeted:

    timeout 600 pytest tests/test_stats_lib.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "session28"))
sys.path.insert(0, str(REPO / "scripts" / "session26"))
sys.path.insert(0, str(REPO / "scripts" / "session20"))

import stats_lib  # noqa: E402

WAKE_PAIRED = REPO / "outputs" / "session26" / "stats" / "wake_paired.json"
DNS_METRICS = REPO / "outputs" / "session17" / "exp2" / "dns_physical_metrics.npz"
LATENTS = REPO / "outputs" / "session18" / "exp_b1" / "latents_jepa_d64_test1_noBN" / "train.npz"

artifacts_present = WAKE_PAIRED.exists() and DNS_METRICS.exists() and LATENTS.exists()


def test_holm_pure():
    """Holm step-down on the published D165 raw p-values reproduces the committed flags."""
    holm_json = REPO / "outputs" / "session26" / "stats" / "holm.json"
    if not holm_json.exists():
        pytest.skip("holm.json not on disk")
    blob = json.loads(holm_json.read_text())
    raw = {k: v["raw_p"] for k, v in blob["tests"].items()}
    adj, survive = stats_lib.holm(raw)
    for k, v in blob["tests"].items():
        assert adj[k] == pytest.approx(v["holm_p"], rel=0, abs=1e-12), k
        assert survive[k] == v["survives_0.05"], k


@pytest.mark.skipif(not artifacts_present, reason="v2 artifacts not on disk")
def test_v2_wake_paired_reproduces_exactly():
    import track1_stats as t1  # heavy import: loads DNS metrics at module level

    expected = json.loads(WAKE_PAIRED.read_text())
    rng = np.random.default_rng(0)  # the single global RNG of the v2 run

    for mode in ("repr", "forecast"):
        for obs in t1.OBSERVABLES:
            j, jc, _ = t1.load_abs_error_with_cases(obs, mode, "jepa_d64")
            r, rc, _ = t1.load_abs_error_with_cases(obs, mode, "fukami_d64")
            assert np.array_equal(jc, rc)
            delta = r - j
            key = f"{mode}/{obs}"
            e = expected[key]

            k, n_eff, sp = stats_lib.sign_test_one_sided(j, r)
            assert k == e["sign_k"], key
            assert n_eff == e["sign_n_eff"], key
            assert sp == pytest.approx(e["sign_p_one_sided"], abs=1e-12), key

            cc = stats_lib.case_cluster_bootstrap(delta, jc, rng=rng)
            assert cc["enc_mean"] == pytest.approx(e["case_clustered"]["enc_mean"], abs=1e-9)
            assert cc["case_mean"] == pytest.approx(e["case_clustered"]["case_mean"], abs=1e-9)
            assert cc["enc_mean_ci"] == pytest.approx(e["case_clustered"]["enc_mean_ci"], abs=1e-9), key
            assert cc["case_mean_ci"] == pytest.approx(e["case_clustered"]["case_mean_ci"], abs=1e-9), key

            enc_ci = stats_lib.encounter_bootstrap_ci(delta, rng=rng)
            assert enc_ci == pytest.approx(e["enc_bootstrap_ci"], abs=1e-9), key

            _, cmd = stats_lib.case_means(delta, jc)
            clp = stats_lib.case_level_paired_stats(cmd)
            assert clp["cases_jepa_better"] == e["case_level_paired"]["cases_jepa_better"], key
            assert clp["sign_p_one_sided"] == pytest.approx(
                e["case_level_paired"]["sign_p_one_sided"], abs=1e-12
            ), key


def test_case_permutation_p_smoke():
    """Sanity: strong case-level association gives small p, none gives large p."""
    rng = np.random.default_rng(7)
    cases = np.repeat([f"c{i}" for i in range(10)], 4)
    x = np.repeat(rng.normal(size=10), 4) + 0.05 * rng.normal(size=40)
    y_dep = x + 0.05 * rng.normal(size=40)
    y_indep = rng.normal(size=40)

    def spear(a, b):
        from scipy.stats import spearmanr

        return spearmanr(a, b).statistic

    p_dep = stats_lib.case_permutation_p(x, y_dep, cases, spear, n_perm=500)
    p_indep = stats_lib.case_permutation_p(x, y_indep, cases, spear, n_perm=500)
    assert p_dep["p"] < 0.05
    assert p_indep["p"] > 0.05
