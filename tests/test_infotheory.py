#!/usr/bin/env python3
"""
tests/test_infotheory.py
========================

Eight self-contained gates for the infotheory package (estimators + SURD),
runnable with no project data:

  1. analytic Gaussian MI        -- KSG reproduces -0.5 ln(1 - rho^2)
  2. independence null           -- surrogate p-value is non-significant for X _||_ Y
  3. CMI chain rule              -- conditioning on a common cause collapses I(X;Y|Z)
  4. KL differential entropy     -- Kozachenko-Leonenko reproduces 0.5 ln(2 pi e) for N(0,1)
  5. SURD XOR  -> pure synergy   -- T = X1 xor X2
  6. SURD COPY -> pure redundancy-- X1 = X2 = T
  7. SURD UNIQUE -> pure unique  -- T = X1, X2 independent
  8. 3-source smoke + conservation- T = X1 with X2,X3 noise; sum(R+U+S) == I(T;all)

Run:  python tests/test_infotheory.py   ->  "All 8 tests passed."
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infotheory.estimators import (  # noqa: E402
    mutual_information_knn,
    conditional_mutual_information_knn,
    entropy_knn,
    surrogate_null_mi,
)
from infotheory.surd import surd_discrete  # noqa: E402


def _components_sum(res) -> float:
    return (
        sum(res.redundant.values())
        + sum(res.unique.values())
        + sum(res.synergistic.values())
    )


def test_analytic_gaussian_mi():
    """KSG MI of a correlated bivariate Gaussian matches the closed form."""
    rng = np.random.default_rng(0)
    n, rho = 4000, 0.6
    z = rng.standard_normal((n, 2))
    x = z[:, 0]
    y = rho * z[:, 0] + np.sqrt(1 - rho**2) * z[:, 1]
    mi = mutual_information_knn(x, y, k=4, random_state=0)
    true = -0.5 * np.log(1 - rho**2)
    assert abs(mi - true) < 0.05, f"KSG MI {mi:.4f} vs analytic {true:.4f}"
    return f"MI={mi:.4f} (analytic {true:.4f})"


def test_independence_null():
    """Independent X, Y give a non-significant surrogate p-value and ~0 debiased MI."""
    rng = np.random.default_rng(1)
    n = 1500
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    out = surrogate_null_mi(x, y, n_surrogate=120, k=4, random_state=1)
    assert out["p_value"] > 0.05, f"independent vars flagged significant: p={out['p_value']:.3f}"
    assert abs(out["mi_debiased"]) < 0.05, f"debiased MI not ~0: {out['mi_debiased']:.4f}"
    return f"p={out['p_value']:.3f}, mi_debiased={out['mi_debiased']:.4f}"


def test_cmi_chain_rule():
    """For X <- Z -> Y, conditioning on the common cause Z collapses the MI."""
    rng = np.random.default_rng(2)
    n = 3000
    z = rng.standard_normal(n)
    x = z + 0.5 * rng.standard_normal(n)
    y = z + 0.5 * rng.standard_normal(n)
    i_xy = mutual_information_knn(x, y, k=4, random_state=2)
    cmi = conditional_mutual_information_knn(x, y, z, k=4, random_state=2)
    assert i_xy > 0.15, f"unconditioned MI too small to be a meaningful test: {i_xy:.4f}"
    assert cmi < 0.5 * i_xy, f"conditioning did not reduce MI: I(X;Y)={i_xy:.4f}, CMI={cmi:.4f}"
    assert cmi < 0.12, f"conditional MI not ~0: {cmi:.4f}"
    return f"I(X;Y)={i_xy:.4f}, I(X;Y|Z)={cmi:.4f}"


def test_kl_entropy():
    """Kozachenko-Leonenko entropy of N(0,1) matches 0.5 ln(2 pi e)."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal(4000)
    h = entropy_knn(x, k=4, standardise=False, random_state=3)
    true = 0.5 * np.log(2 * np.pi * np.e)
    assert abs(h - true) < 0.1, f"KL entropy {h:.4f} vs analytic {true:.4f}"
    return f"H={h:.4f} (analytic {true:.4f})"


def test_surd_xor_synergy():
    """T = X1 xor X2 -> all information is synergistic."""
    p = np.zeros((2, 2, 2))
    for x1 in (0, 1):
        for x2 in (0, 1):
            p[x1 ^ x2, x1, x2] = 0.25
    res = surd_discrete(p)
    syn = sum(res.synergistic.values())
    assert syn > 0.8 * res.mi_total, f"synergy {syn:.3f} not dominant (MI {res.mi_total:.3f})"
    assert sum(res.unique.values()) < 0.05, "unexpected unique information in XOR"
    assert sum(res.redundant.values()) < 0.05, "unexpected redundancy in XOR"
    return f"S={syn:.3f}, MI={res.mi_total:.3f}, leak={res.info_leak:.3f}"


def test_surd_copy_redundant():
    """X1 = X2 = T -> all information is redundant."""
    p = np.zeros((2, 2, 2))
    for t in (0, 1):
        p[t, t, t] = 0.5
    res = surd_discrete(p)
    red = sum(res.redundant.values())
    assert red > 0.8 * res.mi_total, f"redundancy {red:.3f} not dominant (MI {res.mi_total:.3f})"
    assert sum(res.synergistic.values()) < 0.05, "unexpected synergy in COPY"
    return f"R={red:.3f}, MI={res.mi_total:.3f}"


def test_surd_unique():
    """T = X1, with X2 independent -> all information is unique to source 1."""
    p = np.zeros((2, 2, 2))
    for t in (0, 1):
        for x2 in (0, 1):
            p[t, t, x2] = 0.25  # t == x1, x2 free
    res = surd_discrete(p)
    u1 = res.unique.get(frozenset({1}), 0.0)
    u2 = res.unique.get(frozenset({2}), 0.0)
    assert u1 > 0.8 * res.mi_total, f"U[1] {u1:.3f} not dominant (MI {res.mi_total:.3f})"
    assert u2 < 0.05, f"U[2] should be ~0, got {u2:.3f}"
    return f"U[1]={u1:.3f}, U[2]={u2:.3f}, MI={res.mi_total:.3f}"


def test_three_source_conservation():
    """3-source lattice runs and conserves information: sum(R+U+S) == I(T; all)."""
    p = np.zeros((2, 2, 2, 2))
    for x1 in (0, 1):
        for x2 in (0, 1):
            for x3 in (0, 1):
                p[x1, x1, x2, x3] = 0.125  # T == X1; X2, X3 are independent noise
    res = surd_discrete(p)
    assert res.n_sources == 3
    assert 0.0 <= res.info_leak < 1.0, f"info_leak out of range: {res.info_leak:.3f}"
    total = _components_sum(res)
    assert abs(total - res.mi_total) < 1e-6, f"conservation broken: {total:.6f} vs MI {res.mi_total:.6f}"
    u1 = res.unique.get(frozenset({1}), 0.0)
    assert u1 > 0.8 * res.mi_total, f"U[1] {u1:.3f} not dominant (MI {res.mi_total:.3f})"
    return f"sum(R+U+S)={total:.4f}, MI={res.mi_total:.4f}, U[1]={u1:.4f}"


TESTS = [
    test_analytic_gaussian_mi,
    test_independence_null,
    test_cmi_chain_rule,
    test_kl_entropy,
    test_surd_xor_synergy,
    test_surd_copy_redundant,
    test_surd_unique,
    test_three_source_conservation,
]


def main() -> int:
    passed = 0
    for t in TESTS:
        try:
            detail = t()
            print(f"  PASS  {t.__name__:<34} {detail}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__:<34} {e}")
    print()
    if passed == len(TESTS):
        print(f"All {len(TESTS)} tests passed.")
        return 0
    print(f"{passed}/{len(TESTS)} tests passed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
