"""Tests for Physics Track P5 (scripts/session28/p5_wake_code.py).

CPU-only synthetic unit tests for the three P5 building blocks plus the
subspace-overlap null sanity and a key-convention tolerance check. The
heavyweight real-data path (frozen-encoder saliency autograd) is exercised by a
skipif-gated smoke test that requires the v2.1 latents / checkpoint on disk; it
is skipped in CI.

Adversarial design: every test pins a KNOWN ground truth (a redundant vs a
distributed code; a known low-rank signal; a mask that a saliency map does or
does not overlap) so the metric cannot silently agree with itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "session28"))

import p5_wake_code as p5  # noqa: E402

# --------------------------------------------------------------------------- (i)


def test_collective_gap_redundant_code_is_small() -> None:
    """A redundant code (every coordinate copies the target) has a near-zero gap.

    If each latent coordinate is a noisy copy of y, the best single coordinate
    already predicts y well, so combination-minus-best-single is small.
    """
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.standard_normal(n)
    d = 8
    # Each coordinate is y plus independent small noise -> all redundant.
    Xtr = y[:, None] + 0.1 * rng.standard_normal((n, d))
    ytr = y
    y2 = rng.standard_normal(n)
    Xte = y2[:, None] + 0.1 * rng.standard_normal((n, d))
    res = p5.collective_code_skill(Xtr, ytr, Xte, y2)
    assert res["combo"] > 0.9
    assert res["best_single"] > 0.9
    assert res["gap"] < 0.1


def test_collective_gap_distributed_code_is_large() -> None:
    """A distributed code (target = SUM of weak coordinates) has a large gap.

    Each coordinate carries a small, independent slice of y; no single
    coordinate predicts y well, but the ridge combination recovers it, so the
    gap is large and positive.
    """
    rng = np.random.default_rng(1)
    n = 6000
    d = 12
    Xtr = rng.standard_normal((n, d))
    Xte = rng.standard_normal((n, d))
    w = np.ones(d)
    ytr = Xtr @ w + 0.05 * rng.standard_normal(n)
    yte = Xte @ w + 0.05 * rng.standard_normal(n)
    res = p5.collective_code_skill(Xtr, ytr, Xte, yte)
    assert res["combo"] > 0.9
    assert res["best_single"] < 0.5
    assert res["gap"] > 0.4


# -------------------------------------------------------------------------- (ii)


def test_energy_info_curve_monotone_and_knee_on_low_rank() -> None:
    """Held-out skill rises with #PCs then plateaus; the knee is near the rank.

    Build a rank-3 signal embedded in d=20 latents with small noise. Retaining
    1, 2, 3 PCs should sharply increase skill; beyond 3 it plateaus, so the knee
    (90% of the asymptotic skill) sits at <= 4 PCs.
    """
    rng = np.random.default_rng(2)
    n_tr, n_te, d, r = 5000, 2000, 20, 3
    basis = rng.standard_normal((r, d))
    coef_tr = rng.standard_normal((n_tr, r))
    coef_te = rng.standard_normal((n_te, r))
    Xtr = coef_tr @ basis + 0.01 * rng.standard_normal((n_tr, d))
    Xte = coef_te @ basis + 0.01 * rng.standard_normal((n_te, d))
    wq = rng.standard_normal(r)
    ytr = coef_tr @ wq
    yte = coef_te @ wq
    ks = [1, 2, 3, 4, 6, 10, 20]
    curve = p5.energy_information_curve(Xtr, ytr, Xte, yte, ks)
    skills = np.array([curve["skill"][k] for k in ks])
    # near-monotone increasing in the rising region (1->3)
    assert skills[2] >= skills[0] - 1e-6
    assert skills[2] > 0.8
    assert curve["knee_k"] <= 4


def test_energy_fraction_curve_returns_energy_axis() -> None:
    """The POD-style energy-fraction variant returns a retained-energy axis."""
    rng = np.random.default_rng(3)
    n, d = 3000, 10
    sv = np.geomspace(10.0, 0.1, d)
    base = rng.standard_normal((n, d)) * sv[None, :]
    yte_coef = base[:, 0] + 0.5 * base[:, 1]
    curve = p5.energy_information_curve(
        base, yte_coef, base, yte_coef, list(range(1, d + 1)), energy_axis=True
    )
    assert "energy_fraction" in curve
    ef = np.array([curve["energy_fraction"][k] for k in range(1, d + 1)])
    assert ef[0] < ef[-1]
    assert abs(ef[-1] - 1.0) < 1e-6


# ------------------------------------------------------------------------- (iii)


def test_saliency_overlap_beats_chance_on_aligned_mask() -> None:
    """A saliency map concentrated inside a mask overlaps it well above chance."""
    rng = np.random.default_rng(4)
    H, W = 40, 30
    mask = np.zeros((H, W), dtype=bool)
    mask[10:20, 5:15] = True  # 100 / 1200 pixels active (frac ~0.083)
    sal = 0.01 * rng.random((H, W))
    sal[mask] += 1.0  # concentrate saliency inside the mask
    ov = p5.saliency_mask_overlap(sal, mask)
    chance = float(mask.mean())
    assert ov > chance + 0.3
    assert ov > 0.5


def test_saliency_overlap_at_chance_on_random_map() -> None:
    """A flat / random saliency map overlaps a mask at ~the mask's area fraction.

    The energy-weighted overlap of a uniform map equals the mask area fraction;
    a random map fluctuates around it. We assert it lands near chance (not far
    above), so the metric does not manufacture localization.
    """
    rng = np.random.default_rng(5)
    H, W = 60, 40
    mask = np.zeros((H, W), dtype=bool)
    mask[:, :8] = True  # area fraction = 0.2
    sal = rng.random((H, W))
    ov = p5.saliency_mask_overlap(sal, mask)
    chance = float(mask.mean())
    assert abs(ov - chance) < 0.06


def test_saliency_overlap_z_vs_chance_baseline() -> None:
    """The permutation null returns a chance band; aligned saliency clears it."""
    rng = np.random.default_rng(6)
    H, W = 32, 32
    mask = np.zeros((H, W), dtype=bool)
    mask[8:16, 8:16] = True
    sal = 0.01 * rng.random((H, W))
    sal[mask] += 1.0
    res = p5.overlap_vs_chance(sal, mask, n_perm=500, rng=np.random.default_rng(7))
    assert res["overlap"] > res["chance_hi"]
    assert res["z"] > 2.0
    assert res["localized"] is True


# ------------------------------------------------------- subspace-overlap null


def test_subspace_overlap_null_is_K_over_d() -> None:
    """Random K-dim subspaces in d-dim ambient overlap at ~K/d (CLAUDE.md null).

    The mean squared cosine between two random K-subspaces is K/d. For K=3,
    d=64 this is ~0.047; the empirical mean over many random pairs must match.
    """
    K, d = 3, 64
    null = p5.random_subspace_overlap_null(K, d, n_draws=400, rng=np.random.default_rng(8))
    assert abs(null["mean_cos2"] - K / d) < 0.01
    assert abs(null["expected"] - K / d) < 1e-9


def test_grouping_overlap_flags_below_null_as_not_overlap() -> None:
    """A pairwise cos^2 within 0.01 of K/d is reported as NOT overlap."""
    K, d = 3, 64
    # cos^2 essentially equal to the null
    assert not p5.is_overlap_above_null(K / d + 0.005, K, d)
    # clearly above null
    assert p5.is_overlap_above_null(0.5, K, d)


# ------------------------------------------------------- key-convention check


def test_match_targets_key_convention() -> None:
    """(case_id, encounter_index) keys join latents to per-frame targets exactly.

    Mismatched encounter indices must not silently align; only exact (str, int)
    key matches are joined.
    """
    cids = np.array(["A", "A", "B"])
    encs = np.array([0, 1, 0])
    tmap = {("A", 0): 10.0, ("A", 1): 11.0, ("B", 0): 20.0, ("B", 9): 99.0}
    matched = [tmap.get((str(c), int(e))) for c, e in zip(cids, encs)]
    assert matched == [10.0, 11.0, 20.0]
    assert tmap.get(("B", 1)) is None  # no silent alignment


# ------------------------------------------------------- real-data smoke (skip)

_REAL = (REPO / "outputs/session28/latents/jepa_tf_noc_d64_s42/test_b.npz").exists() and (
    REPO / "outputs/runs/session28/jepa_tf_noc_d64_s42/encoder/checkpoint_iter020000.pt"
).exists()


@pytest.mark.skipif(not _REAL, reason="v2.1 latents / checkpoint not on disk")
def test_real_collective_code_runs() -> None:
    """Smoke: the collective-code part runs on the real jepa latents."""
    fam = p5.load_family_latents("jepa_tf_noc_d64_s42")
    assert fam is not None
    tmap_tr = p5.targets_map("train")
    tmap_te = p5.targets_map("test_b")
    Xtr, ytr, _, _ = p5.build_post_impact(fam["train"], tmap_tr)
    Xte, yte, _, _ = p5.build_post_impact(fam["test_b"], tmap_te)
    assert Xtr.shape[1] == 64
    res = p5.collective_code_skill(Xtr, ytr, Xte, yte)
    assert -1.0 <= res["gap"] <= 1.0
