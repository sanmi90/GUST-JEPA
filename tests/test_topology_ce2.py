"""Tests for the C-E2 topology-fairness mechanism (referee M4).

CPU-only, synthetic point clouds, no real artifacts. The central test
demonstrates the M4 mechanism itself: anisotropic per-coordinate scaling
destroys the single clean H1 generator of a circle under the RAW metric, and
per-dimension standardisation restores it. The remaining tests fix the noise
floor semantics (clean circle -> one generator; floor monotonic in lifetime;
two separated blobs -> two persistent components) and the NPZ key-spelling
tolerance.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "scripts" / "session28" / "topology_ce2.py"

_spec = importlib.util.spec_from_file_location("topology_ce2", MODULE_PATH)
topo = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(topo)


def _circle(n: int = 80, seed: int = 0) -> np.ndarray:
    """A clean closed loop in 2D (one persistent H1 generator)."""
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    xy = np.c_[np.cos(t), np.sin(t)]
    # Tiny jitter so the cloud is generic; far below any tested floor.
    rng = np.random.default_rng(seed)
    return xy + rng.normal(0.0, 1e-3, xy.shape)


# --------------------------------------------------------------------------- #
# Clean circle -> exactly one persistent H1 generator at the 5 percent floor
# --------------------------------------------------------------------------- #
def test_clean_circle_one_h1_generator():
    counts = topo.persistence_counts(_circle(), floors=[topo.CANONICAL_FLOOR])
    c = counts[topo.CANONICAL_FLOOR]
    assert c["H1"] == 1, f"clean circle should give one H1 loop, got {c['H1']}"
    assert c["H0"] == 1, f"clean circle is connected (one H0 component), got {c['H0']}"


# --------------------------------------------------------------------------- #
# THE M4 MECHANISM: anisotropic rescaling fragments under raw, std heals it
# --------------------------------------------------------------------------- #
def test_anisotropic_scale_breaks_loop_standardisation_restores():
    circle = _circle()

    # Baseline: the clean loop is a single H1 generator under the raw metric.
    base = topo.persistence_counts(circle, floors=[topo.CANONICAL_FLOOR])
    assert base[topo.CANONICAL_FLOOR]["H1"] == 1

    # Anisotropic per-coordinate scaling: stretch axis 0 by 50x. Under the raw
    # metric the diameter is dominated by axis 0, the 5 percent floor swamps the
    # loop's perpendicular extent, and the single clean generator no longer
    # survives the floor (count != 1: the M4 failure mode).
    aniso = circle.copy()
    aniso[:, 0] *= 50.0
    raw = topo.persistence_counts(aniso, floors=[topo.CANONICAL_FLOOR])
    assert raw[topo.CANONICAL_FLOOR]["H1"] != 1, (
        "anisotropic 50x rescaling must break the single clean generator under "
        "the raw metric (this IS the referee-M4 defect)"
    )

    # Per-dim standardisation removes the anisotropic scale and restores the
    # single clean H1 generator. apply_metric("stdz") needs a whitener fit on
    # the same cloud's per-dim std; build one explicitly.
    whit = {
        "stdz": {"mean": aniso.mean(axis=0), "scale": aniso.std(axis=0)},
        "maha": {"mean": aniso.mean(axis=0), "matrix": np.eye(aniso.shape[1])},
    }
    healed_cloud = topo.apply_metric(aniso, "stdz", whit)
    healed = topo.persistence_counts(healed_cloud, floors=[topo.CANONICAL_FLOOR])
    assert (
        healed[topo.CANONICAL_FLOOR]["H1"] == 1
    ), "per-dimension standardisation must restore the single clean generator"


# --------------------------------------------------------------------------- #
# Mahalanobis whitening also restores the loop (full-covariance metric option)
# --------------------------------------------------------------------------- #
def test_mahalanobis_metric_restores_loop():
    circle = _circle()
    aniso = circle.copy()
    aniso[:, 0] *= 50.0
    cov = np.cov(aniso, rowvar=False) + 1e-8 * np.eye(2)
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, 1e-12, None)
    w_maha = evecs @ np.diag(evals**-0.5) @ evecs.T
    whit = {
        "stdz": {"mean": aniso.mean(axis=0), "scale": aniso.std(axis=0)},
        "maha": {"mean": aniso.mean(axis=0), "matrix": w_maha},
    }
    healed = topo.persistence_counts(
        topo.apply_metric(aniso, "maha", whit), floors=[topo.CANONICAL_FLOOR]
    )
    assert healed[topo.CANONICAL_FLOOR]["H1"] == 1


# --------------------------------------------------------------------------- #
# Noise floor as fraction of diameter behaves monotonically
# --------------------------------------------------------------------------- #
def test_floor_monotonic_in_lifetime():
    # A circle plus a handful of short-lived spurious loops (small inner ring).
    big = _circle(n=80, seed=1)
    t = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    small = 0.06 * np.c_[np.cos(t), np.sin(t)] + np.array([0.0, 0.0])
    cloud = np.vstack([big, small])
    counts = topo.persistence_counts(cloud, floors=topo.FLOOR_GRID)
    h1 = [counts[f]["H1"] for f in topo.FLOOR_GRID]
    h0 = [counts[f]["H0"] for f in topo.FLOOR_GRID]
    # Raising the floor can only remove generators, never add them.
    assert all(h1[i] >= h1[i + 1] for i in range(len(h1) - 1)), h1
    assert all(h0[i] >= h0[i + 1] for i in range(len(h0) - 1)), h0


# --------------------------------------------------------------------------- #
# Two well-separated blobs -> two persistent H0 components
# --------------------------------------------------------------------------- #
def test_two_blobs_two_components():
    rng = np.random.default_rng(2)
    a = rng.normal(0.0, 0.05, (40, 2))
    b = a + np.array([10.0, 0.0])
    cloud = np.vstack([a, b])
    counts = topo.persistence_counts(cloud, floors=[topo.CANONICAL_FLOOR])
    assert (
        counts[topo.CANONICAL_FLOOR]["H0"] == 2
    ), "two well-separated blobs are two persistent connected components"


# --------------------------------------------------------------------------- #
# Degenerate (coincident) cloud is handled (one component, no loop)
# --------------------------------------------------------------------------- #
def test_degenerate_cloud():
    cloud = np.zeros((30, 4))
    counts = topo.persistence_counts(cloud, floors=topo.FLOOR_GRID)
    for f in topo.FLOOR_GRID:
        assert counts[f] == {"H0": 1, "H1": 0}


# --------------------------------------------------------------------------- #
# NPZ key-convention tolerance: singular and plural spellings both load
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "cid_key,enc_key",
    [
        ("case_ids", "encounter_indices"),
        ("case_id", "encounter_index"),
    ],
)
def test_key_convention_tolerance(tmp_path, monkeypatch, cid_key, enc_key):
    fam = "synthetic_fam"
    fam_dir = tmp_path / fam
    fam_dir.mkdir()
    z = np.random.default_rng(3).normal(size=(3, 120, 8)).astype(np.float32)
    cids = np.array(["Baseline", "G+1.00_D0.50_Y+0.10", "G-1.00_D0.50_Y-0.10"])
    encs = np.array([0, 1, 2], dtype=np.int32)
    payload = {"z_full": z, cid_key: cids, enc_key: encs}
    np.savez(fam_dir / "train.npz", **payload)

    monkeypatch.setattr(topo, "LATENTS_DIR", tmp_path)
    out = topo.load_split(fam, "train")
    assert out["z_full"].shape == (3, 120, 8)
    assert list(out["case_ids"]) == list(cids)
    assert list(out["encounter_indices"]) == [0, 1, 2]


# --------------------------------------------------------------------------- #
# Gate logic on tiny synthetic per_family dicts (comparative fukami-vs-jepa)
# --------------------------------------------------------------------------- #
def _rows(metric, h1, n):
    return [{"metric": metric, "floor": topo.CANONICAL_FLOOR, "H1": h1, "H0": 1} for _ in range(n)]


def _family(gusted_raw_h1, gusted_wht_h1, period_wht_h1, n=10, nb=8):
    return {
        "gusted": _rows("raw", gusted_raw_h1, n) + _rows("stdz", gusted_wht_h1, n),
        "baseline": _rows("raw", 3, nb) + _rows("stdz", 3, nb),
        "baseline_period": _rows("raw", period_wht_h1, nb) + _rows("stdz", period_wht_h1, nb),
    }


def test_gate_weak_when_whitening_closes_gap():
    # Under whitening fukami reaches JEPA's single-cycle fraction (both H1 == 1):
    # the gap closes, so the fragmentation was a metric artefact -> WEAK.
    per_family = {
        "jepa_tf_noc_d64_s42": _family(1, 1, 1),
        "fukami_d64_s42": _family(0, 1, 1),
    }
    gate = topo.decide_gate(per_family, "stdz")
    assert gate["branch"] == "WEAK"
    assert gate["gusted_gap_jepa_minus_fukami"] <= 0.1


def test_gate_strong_when_gap_survives_whitening():
    # After whitening JEPA keeps the clean cycle (H1 == 1) while fukami stays
    # fragmented (H1 == 3) on BOTH gusted encounters and the single-period
    # no-gust control -> STRONG.
    per_family = {
        "jepa_tf_noc_d64_s42": _family(1, 1, 1),
        "fukami_d64_s42": _family(3, 3, 3),
    }
    gate = topo.decide_gate(per_family, "stdz")
    assert gate["branch"] == "STRONG"
    assert gate["gusted_gap_jepa_minus_fukami"] > 0.1
    assert gate["baseline_period_gap_jepa_minus_fukami"] > 0.0


def test_gate_weak_mixed_split_verdict():
    # The split verdict that the real v2.1 data hits: on the WHOLE gusted
    # encounters whitening closes/reverses the gap (fukami single-cycle as often
    # as JEPA), but the SINGLE-PERIOD no-gust control separates the families in
    # JEPA's favour (fukami fragments the clean cycle, JEPA does not).
    jepa = _family(gusted_raw_h1=1, gusted_wht_h1=1, period_wht_h1=1)
    fukami = _family(gusted_raw_h1=1, gusted_wht_h1=1, period_wht_h1=3)
    per_family = {"jepa_tf_noc_d64_s42": jepa, "fukami_d64_s42": fukami}
    gate = topo.decide_gate(per_family, "stdz")
    assert gate["branch"] == "WEAK-MIXED"
    assert gate["gusted_gap_jepa_minus_fukami"] <= 0.1
    assert gate["baseline_period_gap_jepa_minus_fukami"] > 0.2
