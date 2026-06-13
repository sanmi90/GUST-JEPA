"""Tests for Physics Track P2 latent-clock tie-in (Gate GP2).

CPU synthetic tests for the four load-bearing pieces:
    1. the tube-return rule on a synthetic excursion-and-return trace (known return frame
       + censoring + short-window),
    2. the baseline-orbit Mahalanobis-to-distribution on a known anisotropic cloud,
    3. the latent tube radius calibration (q95 of the orbit's own spread),
    4. the Spearman + case-permutation wiring (perfect-rank -> rho = 1, p small; shuffled ->
       rho ~ 0),
plus a skipif-real smoke test that the pipeline runs end-to-end on the committed artefacts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "session28"))

import p2_latent_clock as p2  # noqa: E402
import s46_regen as s46  # noqa: E402

PHYSICS_NPZ = REPO / "outputs/session28/physics/per_encounter_physics.npz"
LATENTS = REPO / "outputs/session28/latents" / p2.PRED_FAMILY / "train.npz"
SPLIT_MANIFEST = REPO / "configs/splits/split_v2p1.json"


# ----------------------------------------------------------- 1. tube-return rule
def test_tube_return_known_frame():
    """Excursion then sustained return at a known frame -> recovered at that frame."""
    n = 120
    inside = np.zeros(n, dtype=bool)
    inside[60:] = True  # re-enters at 60 and stays inside for 60 frames
    status, t = p2.tube_return_clock(inside, start=40, dwell=10)
    assert status == "recovered"
    assert t == 60


def test_tube_return_starts_inside():
    """If already inside at `start` and it dwells, t* = start."""
    inside = np.ones(80, dtype=bool)
    status, t = p2.tube_return_clock(inside, start=40, dwell=20)
    assert status == "recovered"
    assert t == 40


def test_tube_return_short_run_then_excursion_is_skipped():
    """A short inside-run that breaks before dwell is not the recovery instant."""
    n = 120
    inside = np.zeros(n, dtype=bool)
    inside[50:55] = True  # 5-frame run, < dwell
    inside[70:] = True  # sustained run from 70
    status, t = p2.tube_return_clock(inside, start=40, dwell=10)
    assert status == "recovered"
    assert t == 70  # the short run at 50 is skipped


def test_tube_return_censored():
    """Never re-enters for a full dwell and ends on an excursion -> censored."""
    n = 80
    inside = np.zeros(n, dtype=bool)
    inside[50:55] = True  # only ever a 5-frame run, then excursion to the end
    status, t = p2.tube_return_clock(inside, start=40, dwell=20)
    assert status == "censored"
    assert t is None


def test_tube_return_short_window_counts_separately():
    """Re-enters and stays to the end but < dwell frames remain -> recovered_short_window."""
    n = 100
    inside = np.zeros(n, dtype=bool)
    inside[90:] = True  # 10 frames to the end, < dwell=20, no excursion after
    status, t = p2.tube_return_clock(inside, start=40, dwell=20)
    assert status == "recovered_short_window"
    assert t == 90
    # And it is treated as recovered by the predicate used everywhere downstream.
    assert p2.phys_is_recovered(status) is True


def test_tube_return_full_window_preferred_over_short():
    """A full dwell earlier wins over a short window at the end."""
    n = 120
    inside = np.zeros(n, dtype=bool)
    inside[50:90] = True  # full dwell run
    inside[115:] = True  # short window at the end
    status, t = p2.tube_return_clock(inside, start=40, dwell=20)
    assert status == "recovered"
    assert t == 50


def test_strict_dwell_equals_occupancy_at_theta_one():
    """tube_return_clock equals physics_prep.occupancy_recovery at theta = 1.0 (the rule the
    production latent clock reuses, so the two are the same in the strict limit)."""
    import physics_prep as pp_mod

    rng = np.random.default_rng(11)
    for _ in range(20):
        inside = rng.random(120) < 0.6
        dwell = int(rng.integers(10, 40))
        s_strict, t_strict = p2.tube_return_clock(inside, start=40, dwell=dwell)
        # occupancy on a 0/1 distance trace with band [-inf, 0.5]: inside iff value 0.
        dist = np.where(inside, 0.0, 1.0)
        s_occ, t_occ = pp_mod.occupancy_recovery(
            dist, 40, -np.inf, 0.5, theta=1.0, window=dwell, min_window=dwell
        )
        # With min_window == dwell, occupancy has no short-window slack below dwell, while
        # tube_return_clock flags short windows; compare only the full-recovery branch.
        if s_strict == "recovered":
            assert s_occ == "recovered" and t_strict == t_occ
        elif s_occ == "recovered":
            assert s_strict == "recovered" and t_strict == t_occ


def test_latent_recovery_uses_occupancy_semantics():
    """latent_recovery_for_encounter tolerates isolated excursions (occupancy theta<1), so a
    trajectory with a single off-tube frame inside the window still recovers."""
    rng = np.random.default_rng(12)
    d = 5
    # Tiny tube at the orbit mean; a trajectory that returns to the mean with one blip.
    orbit = rng.standard_normal((100, d)) * 0.1
    train = {
        "z_full": orbit[None, :, :].copy(),
        "case_ids": np.array(["Baseline"]),
        "encounter_indices": np.array([4], dtype=int),
    }
    tube = p2.build_latent_tube(train, "Baseline", radius_pctl=95.0)
    z = np.zeros((120, d))
    z[:40] = 10.0  # far outside the tube pre-impact
    z[40:] = orbit.mean(axis=0)  # snaps back to the tube centre at impact
    z[70] = 10.0  # a single off-tube blip well inside the dwell window
    status, t, _ = p2.latent_recovery_for_encounter(
        z, impact=40, tube=tube, metric="maha", window=56, theta=0.8, min_window=28
    )
    assert status in ("recovered", "recovered_short_window")
    assert t == 40  # the single blip is tolerated at theta = 0.8


# ----------------------------------------------------- 2. Mahalanobis on a known cloud
def test_mahalanobis_to_distribution_known_cloud():
    """Anisotropic Gaussian cloud: whitened distance is isotropic, so equal Maha for
    points scaled by the cloud's own std in each axis."""
    rng = np.random.default_rng(0)
    d = 4
    scales = np.array([5.0, 1.0, 0.5, 2.0])
    cloud = rng.standard_normal((4000, d)) * scales  # zero-mean, diagonal cov ~ scales^2
    geom = s46._orbit_geometry(cloud)
    # A point at +3 std along EACH axis should have Mahalanobis ~ sqrt(d) * 3 (isotropic
    # after whitening), independent of the axis scale.
    pts = np.diag(3.0 * scales)  # row i = 3*scale_i along axis i
    md = p2.mahalanobis_to_distribution(pts, geom)
    # Each single-axis 3-std point has Maha ~ 3 (one whitened unit per axis).
    assert np.allclose(md, 3.0, atol=0.25), md
    # The cloud mean maps to ~0.
    assert p2.mahalanobis_to_distribution(geom.mean[None, :], geom)[0] < 0.2


def test_mahalanobis_invariant_to_axis_scaling():
    """Mahalanobis distance is unchanged if we rescale one axis of cloud AND query."""
    rng = np.random.default_rng(1)
    cloud = rng.standard_normal((2000, 3))
    geom = s46._orbit_geometry(cloud)
    q = np.array([[1.0, 2.0, -1.5]])
    md0 = p2.mahalanobis_to_distribution(q, geom)[0]
    S = np.diag([10.0, 1.0, 1.0])
    geom2 = s46._orbit_geometry(cloud @ S)
    md1 = p2.mahalanobis_to_distribution(q @ S, geom2)[0]
    assert abs(md0 - md1) < 0.05, (md0, md1)


# --------------------------------------------------- 3. tube radius calibration
def test_build_latent_tube_radius_is_q95():
    """The Mahalanobis tube radius is the q95 of the orbit's own Maha spread, so ~95% of
    the settled-Baseline frames lie inside the tube by construction. Also checks that ONLY
    settled frames (global raw frame >= 400) enter the orbit."""
    rng = np.random.default_rng(2)
    d = 8
    # One Baseline encounter at k=4: global frame 480..599, every frame settled (>= 400),
    # so all 120 frames enter the orbit. settled offset for k=4 = max(0, 400-480) = 0.
    orbit = rng.standard_normal((120, d)) * np.array([4, 3, 2, 1, 1, 0.5, 0.5, 0.3])
    train = {
        "z_full": orbit[None, :, :].copy(),  # (1, 120, d)
        "case_ids": np.array(["Baseline"]),
        "encounter_indices": np.array([4], dtype=int),
    }
    tube = p2.build_latent_tube(train, "Baseline", radius_pctl=95.0)
    assert tube.geom.n_pts == 120  # all 120 frames of the k=4 encounter are settled
    md = p2.mahalanobis_to_distribution(tube.orbit, tube.geom)
    frac_inside = float(np.mean(md <= tube.maha_radius))
    assert 0.93 <= frac_inside <= 0.97, frac_inside  # q95 by construction
    assert tube.maha_radius > 0 and tube.euclid_radius > 0


def test_build_latent_tube_excludes_transient_frames():
    """A k=0 encounter (global 0..119, all < 400) contributes NO settled orbit frames."""
    rng = np.random.default_rng(7)
    d = 6
    z = np.zeros((2, 120, d))
    # k=0: global 0..119 (transient, none settled). k=4: global 480..599 (all settled).
    z[1] = rng.standard_normal((120, d))
    train = {
        "z_full": z,
        "case_ids": np.array(["Baseline", "Baseline"]),
        "encounter_indices": np.array([0, 4], dtype=int),
    }
    tube = p2.build_latent_tube(train, "Baseline", radius_pctl=95.0)
    assert tube.geom.n_pts == 120  # only the k=4 encounter's frames


# --------------------------------------------------- 4. Spearman + case-permutation wiring
def test_gate_block_perfect_rank():
    """Monotone physical->latent relation gives Spearman ~1 and a small permutation p."""
    rng = np.random.default_rng(3)
    # 6 cases, ~4 encounters each; latent tau a monotone (noisy) function of physical tau.
    case_ids, phys, lat = [], [], []
    for ci in range(6):
        base = ci * 1.0
        for _ in range(4):
            x = base + rng.uniform(0, 0.3)
            case_ids.append(f"case{ci}")
            phys.append(x)
            lat.append(2.0 * x + rng.normal(0, 0.05))
    cfg = p2.Config(n_boot_case=300, n_perm=300)
    out = p2.gate_block(np.array(phys), np.array(lat), np.array(case_ids), cfg, "test")
    assert out["spearman_rho"] > 0.9
    assert out["passes_gp2"] is True
    assert out["case_permutation_p_one_sided_greater"] < 0.1
    assert out["ci_lo"] <= out["spearman_rho"] <= out["ci_hi"] + 1e-9


def test_gate_block_no_relation():
    """Independent physical and latent taus give a Spearman near zero and fail GP2."""
    rng = np.random.default_rng(4)
    case_ids, phys, lat = [], [], []
    for ci in range(8):
        for _ in range(4):
            case_ids.append(f"case{ci}")
            phys.append(rng.normal())
            lat.append(rng.normal())
    cfg = p2.Config(n_boot_case=300, n_perm=300)
    out = p2.gate_block(np.array(phys), np.array(lat), np.array(case_ids), cfg, "test")
    assert abs(out["spearman_rho"]) < 0.5
    assert out["passes_gp2"] is False


def test_spearman_stat_nan_safe():
    """A constant input yields a defined (0.0) statistic, never NaN."""
    assert p2.spearman_stat(np.ones(5), np.arange(5)) == 0.0


# --------------------------------------------------- skipif-real end-to-end smoke
@pytest.mark.skipif(
    not (PHYSICS_NPZ.exists() and LATENTS.exists() and SPLIT_MANIFEST.exists()),
    reason="real artefacts (physics NPZ + tf_noc latents + split manifest) not present",
)
def test_run_p2_real_smoke(tmp_path):
    """End-to-end on the committed artefacts: gate computed, fractions add up, taus sane."""
    cfg = p2.Config(n_boot_case=200, n_perm=200)
    results = p2.run_p2(cfg, PHYSICS_NPZ, SPLIT_MANIFEST)

    # Gate present with a finite headline rho and a valid pass flag.
    h = results["gate_gp2"]["headline"]
    assert "spearman_rho" in h
    assert isinstance(results["gate_gp2"]["passes_gp2"], bool)

    # Fractions: recovered + censored == n for every clock; fractions in [0, 1].
    for clock in ("physical_v2", "latent_maha", "latent_euclid"):
        b = results["fractions_test_b"][clock]
        assert b["n_recovered"] + b["n_censored"] == b["n"]
        assert 0.0 <= b["frac_recovered"] <= 1.0
        assert 0.0 <= b["frac_censored"] <= 1.0

    # The paired n entering the gate cannot exceed the test_b encounter count.
    n_tb = results["fractions_test_b"]["physical_v2"]["n"]
    assert 0 <= h["n_encounters"] <= n_tb

    # Latent taus, where finite, are non-negative (post-impact) and <= window in t/c.
    for r in results["per_encounter_test_b"]:
        for key in ("latent_tau_tc_maha", "latent_tau_tc_euclid", "phys_tau_tc"):
            v = r[key]
            if v is not None:
                assert -1e-9 <= v <= 120 * p2.pp.DT_TC

    # numbers part: alphabetic macros, every record has a value.
    part = p2.build_numbers_part(results)
    for name, rec in part["numbers"].items():
        assert "value" in rec
        assert rec["macro"].isalpha(), rec["macro"]
