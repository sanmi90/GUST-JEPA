"""CPU-fast synthetic-data unit tests for scripts/session28/physics_prep.py.

No cache, no torch, no GPU: every test builds its own signal or field. Covers the
Hilbert impact-phase extraction (known sinusoid + drift), the phase-matched Delta C_L
reference, the tau_rec re-entry / dwell / censoring logic, the mean +- 2 sd envelope
rule, the Gaussian-filter scale separation, the numeric Taylor-circulation and
Martinez-Muriel & Flores velocity-ratio candidates against their analytic values, the
v2 occupancy-dwell recovery rule and its null-case theta calibration, the circular
arithmetic helpers, and the within-case phase ladder construction.

Run targeted:  timeout 120 pytest tests/test_physics_prep.py -q
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "session28"))

import physics_prep as pp  # noqa: E402

TWO_PI = 2.0 * math.pi


def circ_diff(a: float, b: float) -> float:
    d = (a - b) % TWO_PI
    return min(d, TWO_PI - d)


def make_lift(n: int, omega: float, phi0: float, mean=0.76, amp=0.18, drift=0.0, noise=0.0, seed=0):
    t = np.arange(n, dtype=float)
    rng = np.random.default_rng(seed)
    return mean + drift * t + amp * np.cos(omega * t + phi0) + noise * rng.standard_normal(n)


# ----------------------------------------------------------------------------------
# Hilbert impact phase
# ----------------------------------------------------------------------------------
class TestImpactPhase:
    OMEGA = TWO_PI / 30.0  # the measured dominant lift line: period ~30 frames
    PHI0 = 1.0

    def test_known_sinusoid_recovers_impact_phase(self):
        n_pre = 40
        cl = make_lift(n_pre, self.OMEGA, self.PHI0)
        fit = pp.fit_pre_impact_cycle(cl)
        expected = (self.OMEGA * n_pre + self.PHI0) % TWO_PI
        assert circ_diff(fit.phi_imp, expected) < 0.15
        assert fit.phase_fit_r2 > 0.99
        assert fit.period_frames == pytest.approx(30.0, abs=1.0)
        assert fit.amplitude == pytest.approx(0.18, rel=0.15)

    def test_linear_drift_does_not_corrupt_phase(self):
        """The slow modulation (locally linear) must be removed by the detrend."""
        n_pre = 40
        cl = make_lift(n_pre, self.OMEGA, self.PHI0, drift=0.004, noise=0.002, seed=3)
        fit = pp.fit_pre_impact_cycle(cl)
        expected = (self.OMEGA * n_pre + self.PHI0) % TWO_PI
        assert circ_diff(fit.phi_imp, expected) < 0.2
        assert fit.phase_fit_r2 > 0.98

    def test_interior_fit_beats_endpoint_sample(self):
        """The end-sample Hilbert phase is edge-biased (FFT periodicity assumption on a
        non-periodic 1.33-cycle window); the interior linear fit must be strictly more
        accurate, which is why phi_imp (fit) is canonical and phi_imp_endpoint is only a
        robustness diagnostic."""
        cl = make_lift(40, self.OMEGA, self.PHI0)
        fit = pp.fit_pre_impact_cycle(cl)
        expected = (self.OMEGA * 40 + self.PHI0) % TWO_PI
        err_fit = circ_diff(fit.phi_imp, expected)
        err_endpoint = circ_diff(fit.phi_imp_endpoint, expected)
        assert err_fit < 0.15
        assert err_fit < err_endpoint

    def test_too_short_window_returns_nan(self):
        fit = pp.fit_pre_impact_cycle(np.ones(5))
        assert math.isnan(fit.phi_imp)


# ----------------------------------------------------------------------------------
# Phase-matched Delta C_L reference
# ----------------------------------------------------------------------------------
class TestDeltaCL:
    def test_undisturbed_continuation_gives_small_phase_matched_peak(self):
        """A pure continuing cycle: phase-matched peak << simple peak (~ the amplitude)."""
        omega, phi0, amp = TWO_PI / 30.0, 0.7, 0.18
        cl = make_lift(120, omega, phi0, amp=amp, drift=0.0005, noise=0.001, seed=1)
        impact = 40
        fit = pp.fit_pre_impact_cycle(cl[:impact])
        pm, simple = pp.delta_cl_peaks(cl, impact, fit, window=40)
        # The phase-matched residual carries the 40-frame phase/amplitude extrapolation
        # error (~0.3 amp here); the simple variant floors at the full carrier amplitude.
        assert pm < 0.35 * amp
        assert simple > 0.8 * amp
        assert pm < 0.45 * simple

    def test_gust_bump_detected_by_both(self):
        omega, amp = TWO_PI / 30.0, 0.18
        cl = make_lift(120, omega, 0.7, amp=amp)
        t = np.arange(120, dtype=float)
        cl = cl + 3.0 * np.exp(-0.5 * ((t - 55.0) / 4.0) ** 2)  # known response peak
        fit = pp.fit_pre_impact_cycle(cl[:40])
        pm, simple = pp.delta_cl_peaks(cl, 40, fit, window=40)
        assert pm == pytest.approx(3.0, abs=0.4)
        assert simple == pytest.approx(3.0, abs=0.4)


# ----------------------------------------------------------------------------------
# Envelope rule
# ----------------------------------------------------------------------------------
class TestEnvelope:
    def test_mean_pm_two_sd(self):
        pre = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lo, hi = pp.envelope_bounds(pre, nsd=2.0)
        m, s = pre.mean(), pre.std(ddof=1)
        assert lo == pytest.approx(m - 2 * s)
        assert hi == pytest.approx(m + 2 * s)

    def test_nsd_parameter(self):
        pre = np.linspace(0, 1, 40)
        lo3, hi3 = pp.envelope_bounds(pre, nsd=3.0)
        lo2, hi2 = pp.envelope_bounds(pre, nsd=2.0)
        assert lo3 < lo2 < hi2 < hi3


# ----------------------------------------------------------------------------------
# tau_rec: re-entry, dwell, censoring
# ----------------------------------------------------------------------------------
class TestTauRec:
    LO, HI = 8.0, 12.0

    def make_trace(self, reentry: int, n: int = 120, impact: int = 40) -> np.ndarray:
        e = np.full(n, 10.0)
        e[impact:reentry] = 30.0  # excursion strictly outside [LO, HI]
        return e

    def test_known_reentry_time(self):
        e = self.make_trace(reentry=70)
        status, tau = pp.first_sustained_reentry(e, 40, self.LO, self.HI, dwell=30)
        assert status == "recovered"
        assert tau == 70

    def test_censored_when_never_returns(self):
        e = np.full(120, 10.0)
        e[40:] = 30.0
        status, tau = pp.first_sustained_reentry(e, 40, self.LO, self.HI, dwell=30)
        assert status == "censored"
        assert tau is None

    def test_short_bounce_does_not_count(self):
        """A 5-frame dip back inside must not be called recovery; the sustained run is."""
        e = self.make_trace(reentry=80)
        e[55:60] = 10.0  # bounce inside, then exits again
        status, tau = pp.first_sustained_reentry(e, 40, self.LO, self.HI, dwell=30)
        assert status == "recovered"
        assert tau == 80

    def test_reentry_too_late_to_certify_dwell_is_unconfirmed(self):
        e = self.make_trace(reentry=100)
        status, tau = pp.first_sustained_reentry(e, 40, self.LO, self.HI, dwell=56)
        assert status == "reentered_unconfirmed"
        assert tau == 100

    def test_already_inside_at_impact_recovers_immediately(self):
        e = np.full(120, 10.0)
        status, tau = pp.first_sustained_reentry(e, 40, self.LO, self.HI, dwell=56)
        assert status == "recovered"
        assert tau == 40


# ----------------------------------------------------------------------------------
# v2 occupancy recovery rule
# ----------------------------------------------------------------------------------
class TestOccupancyRecovery:
    LO, HI = 8.0, 12.0
    IMPACT = 40

    def trace(self, n: int = 120, outside_frames=()) -> np.ndarray:
        e = np.full(n, 10.0)
        for f in outside_frames:
            e[f] = 30.0
        return e

    def test_clean_reentry_recovers_at_reentry_frame(self):
        e = self.trace(outside_frames=range(40, 60))
        status, tau = pp.occupancy_recovery(e, 40, self.LO, self.HI, theta=0.95, window=56)
        assert status == "recovered"
        assert tau == 60

    def test_single_frame_excursion_inside_window_is_tolerated(self):
        """theta = 0.95 over W = 56 tolerates up to 3 outside frames: a single later
        excursion must not delay tau."""
        e = self.trace(outside_frames=[80])
        status, tau = pp.occupancy_recovery(e, 40, self.LO, self.HI, theta=0.95, window=56)
        assert status == "recovered"
        assert tau == 40

    def test_candidate_frame_itself_must_be_inside(self):
        """An excursion AT the candidate frame moves tau to the next inside frame even
        when the occupancy over the window would already pass."""
        e = self.trace(outside_frames=[40, 41])
        status, tau = pp.occupancy_recovery(e, 40, self.LO, self.HI, theta=0.95, window=56)
        assert status == "recovered"
        assert tau == 42

    def test_short_window_flagged(self):
        """Re-entry with >= min_window but < window frames left: recovered_short_window."""
        e = self.trace(outside_frames=range(40, 70))  # 50 frames remain at t* = 70
        status, tau = pp.occupancy_recovery(
            e, 40, self.LO, self.HI, theta=0.95, window=56, min_window=28
        )
        assert status == "recovered_short_window"
        assert tau == 70

    def test_reentry_too_late_to_judge_is_censored(self):
        e = self.trace(outside_frames=range(40, 95))  # 25 frames remain < min_window
        status, tau = pp.occupancy_recovery(
            e, 40, self.LO, self.HI, theta=0.95, window=56, min_window=28
        )
        assert status == "censored"
        assert tau is None

    def test_never_reenters_is_censored(self):
        e = self.trace(outside_frames=range(40, 120))
        status, tau = pp.occupancy_recovery(e, 40, self.LO, self.HI, theta=0.95, window=56)
        assert status == "censored"
        assert tau is None

    def test_theta_controls_the_verdict(self):
        """8 outside frames inside the first window: occ = 48/56 = 0.857. theta = 0.85
        recovers immediately; theta = 0.9 must wait until the window clears them."""
        e = self.trace(outside_frames=range(45, 53))
        status85, tau85 = pp.occupancy_recovery(e, 40, self.LO, self.HI, theta=0.85, window=56)
        status90, tau90 = pp.occupancy_recovery(e, 40, self.LO, self.HI, theta=0.90, window=56)
        assert (status85, tau85) == ("recovered", 40)
        assert (status90, tau90) == ("recovered", 53)


# ----------------------------------------------------------------------------------
# v2 theta calibration on a synthetic null set
# ----------------------------------------------------------------------------------
class TestCalibrationV2:
    """Six synthetic null encounters at frame_start = 120 k; the settled band comes
    from global frames >= 400 (encounters 3-5), so excursions injected into encounter
    0 (fully transient) perturb the calibration verdict without moving the band."""

    def records(self, enc0_outside=()):
        rng = np.random.default_rng(5)
        recs = []
        for k in range(6):
            ens = 10.0 + 0.5 * np.sin(2 * np.pi * np.arange(120) / 56.0)
            ens += 0.05 * rng.standard_normal(120)
            if k == 0:
                for f in enc0_outside:
                    ens[f] = 100.0
            recs.append(
                {"encounter_index": k, "impact_frame": 40, "frame_start": 120 * k, "ens": ens}
            )
        return recs

    def test_clean_null_calibrates_at_largest_theta_primary_band(self):
        rule = pp.calibrate_recovery_v2(self.records())
        assert rule.theta == 0.95
        assert rule.band_quantiles == pp.BAND_QUANTILES_PRIMARY
        steps = rule.calibration["steps"]
        assert len(steps) == 1 and steps[0]["pass"]
        assert all(e["tau_frames"] == 0 for e in steps[0]["per_encounter"])
        assert rule.calibration["null_6of6"] is True

    def test_dirty_transient_encounter_lowers_theta_with_provenance(self):
        """8 excursions in encounter 0's first window: occ 0.857 fails 0.95 and 0.90,
        passes 0.85; the two failed steps must be recorded."""
        rule = pp.calibrate_recovery_v2(self.records(enc0_outside=range(45, 53)))
        assert rule.theta == 0.85
        assert rule.band_quantiles == pp.BAND_QUANTILES_PRIMARY
        steps = rule.calibration["steps"]
        assert [s["pass"] for s in steps] == [False, False, True]
        assert [s["theta"] for s in steps] == [0.95, 0.90, 0.85]

    def test_unrecoverable_null_raises_with_step_record(self):
        """Alternating excursions from impact on: occupancy ~0.5 at every t*, below
        every theta at both bands. The hard 6/6 requirement must raise."""
        with pytest.raises(RuntimeError) as exc:
            pp.calibrate_recovery_v2(self.records(enc0_outside=range(40, 120, 2)))
        steps = exc.value.args[1]
        assert len(steps) == 8  # 2 bands x 4 thetas, all recorded

    def test_tau_cap_is_enforced(self):
        """Encounter 0 outside for [40, 49]: recovery at tau = 10 > 8 fails calibration
        even at theta = 0.8 and the wide band."""
        with pytest.raises(RuntimeError):
            pp.calibrate_recovery_v2(self.records(enc0_outside=range(40, 50)))


# ----------------------------------------------------------------------------------
# Circular arithmetic helpers
# ----------------------------------------------------------------------------------
class TestCircularHelpers:
    def test_wrap_pm_pi_known_values(self):
        assert pp.wrap_pm_pi(0.0) == pytest.approx(0.0)
        assert pp.wrap_pm_pi(math.pi) == pytest.approx(-math.pi)
        assert pp.wrap_pm_pi(-math.pi) == pytest.approx(-math.pi)
        assert pp.wrap_pm_pi(1.5 * math.pi) == pytest.approx(-0.5 * math.pi)
        assert pp.wrap_pm_pi(-1.5 * math.pi) == pytest.approx(0.5 * math.pi)
        out = pp.wrap_pm_pi(np.array([0.1, TWO_PI + 0.1, -TWO_PI + 0.1]))
        assert np.allclose(out, 0.1)

    def test_circ_residual_across_the_wrap(self):
        assert pp.circ_residual(0.1, TWO_PI - 0.1) == pytest.approx(0.2)
        assert pp.circ_residual(TWO_PI - 0.1, 0.1) == pytest.approx(-0.2)

    def test_residual_stats_cluster_straddling_the_boundary(self):
        """Residuals tightly clustered around +-pi must yield a small circular sd, not
        the ~pi a linear sd would report."""
        res = np.array([math.pi - 0.05, -math.pi + 0.05, math.pi - 0.02, -math.pi + 0.03])
        stats = pp.circ_residual_stats(res)
        assert stats["circ_std"] < 0.1
        assert abs(abs(stats["mean"]) - math.pi) < 0.1

    def test_fit_ladder_cadence_recovers_known_step(self):
        k = np.arange(6)
        phi = (0.4 + 0.7 * k) % TWO_PI
        step, resultant = pp.fit_ladder_cadence(k, phi)
        assert step == pytest.approx(0.7, abs=1e-3)
        assert resultant > 0.999

    def test_fit_ladder_cadence_with_gaps_and_nan(self):
        k = np.array([0, 2, 3, 5])
        phi = (1.0 - 0.3 * k) % TWO_PI
        step, _ = pp.fit_ladder_cadence(k, phi)
        assert step == pytest.approx(-0.3, abs=1e-3)
        step2, _ = pp.fit_ladder_cadence(np.arange(3), np.array([0.1, np.nan, 0.5]))
        assert math.isnan(step2)  # < 3 finite points


# ----------------------------------------------------------------------------------
# Phase ladder construction on synthetic rows
# ----------------------------------------------------------------------------------
class TestBuildPhaseLadder:
    STEP_TRUE = 0.21

    def row(self, case_id, k, phi, r2=0.99, amp=0.15, period=33.0, G=0.25, split="train"):
        return {
            "case_id": case_id,
            "encounter_index": k,
            "split": split,
            "G": G,
            "D": 0.5 if G else 0.0,
            "Y": 0.1 if G else 0.0,
            "phi_imp": phi % TWO_PI,
            "phase_fit_r2": r2,
            "pre_cl_amp": amp,
            "phase_period_frames": period,
        }

    def rows(self):
        rows = []
        # Synthetic Baseline: perfect ladder at STEP_TRUE (drives the fit clock).
        for k in range(6):
            rows.append(self.row("Baseline", k, 0.6 + self.STEP_TRUE * k, amp=0.12, G=0.0))
        # Case A: ladder at STEP_TRUE from 1.0; anchor must be k = 1 (lowest amp);
        # k = 3 is contaminated (amp 2.0) -> predicted but excluded from the pool.
        amps = [0.20, 0.10, 0.30, 2.00]
        for k in range(4):
            rows.append(self.row("G+0.25_D0.50_Y+0.10", k, 1.0 + self.STEP_TRUE * k, amp=amps[k]))
        # Case B: all phase fits bad -> unanchored.
        for k in range(4):
            rows.append(self.row("G+3.00_D0.50_Y+0.10", k, 2.0 + 0.5 * k, r2=0.5, G=3.0))
        # Case C: clean but wrong-line lock (period 16) -> period gate must exclude,
        # case unanchored.
        for k in range(4):
            rows.append(self.row("G+0.50_D0.50_Y+0.10", k, 1.0 + 0.4 * k, period=16.0, G=0.5))
        return rows

    def test_ladder_flags_predictions_and_residuals(self):
        clocks = pp.CadenceClocks()
        arrays, summary = pp.build_phase_ladder(self.rows(), pp.PrepParams(), clocks)
        case = arrays["case_id"]
        # Fit clock measured from the synthetic Baseline ladder (k >= 3 subset).
        assert summary["clocks"]["step_fit_rad_per_enc"] == pytest.approx(self.STEP_TRUE, abs=2e-3)
        # Anchors: Baseline k = 0 (amp 0.12 uniform, first minimum) and case A k = 1.
        a = case == "G+0.25_D0.50_Y+0.10"
        assert arrays["anchor"][a].tolist() == [False, True, False, False]
        assert arrays["case_anchored"][a].all()
        # Contaminated k = 3 still gets a prediction but is not clean.
        assert np.isfinite(arrays["phi_pred_fit"][a]).all()
        assert not arrays["clean"][a][3]
        # Residuals at the fit clock are ~0 for the clean ladder encounters.
        clean_resid = arrays["resid_fit"][a & arrays["clean"]]
        assert np.max(np.abs(clean_resid)) < 0.02
        # Unanchored cases: NaN predictions, flag off.
        b = case == "G+3.00_D0.50_Y+0.10"
        assert not arrays["case_anchored"][b].any()
        assert np.isnan(arrays["phi_pred_dom"][b]).all()
        # Period gate kills the wrong-line case entirely.
        c = case == "G+0.50_D0.50_Y+0.10"
        assert not arrays["period_gate_ok"][c].any()
        assert not arrays["case_anchored"][c].any()
        assert summary["anchoring"]["n_anchored"] == 2
        # Pooled verdict: the fit clock matches the generative step -> small sd, usable.
        assert summary["verdict"]["winning_clock"] == "fit"
        assert summary["verdict"]["winning_pooled_sd_rad"] < 0.05
        assert summary["verdict"]["usable_for_phase_assignment"] is True
        # The dom clock (0.322 rad/enc) misfits the 0.21 ladder by ~0.11 rad/enc offset.
        assert (
            summary["pooled_residuals"]["dom"]["circ_std"]
            > summary["pooled_residuals"]["fit"]["circ_std"]
        )

    def test_anchor_residual_excluded_from_pool(self):
        clocks = pp.CadenceClocks()
        arrays, summary = pp.build_phase_ladder(self.rows(), pp.PrepParams(), clocks)
        n_clean_nonanchor = int(
            (arrays["clean"] & arrays["case_anchored"] & ~arrays["anchor"]).sum()
        )
        assert summary["anchoring"]["n_pooled_residuals"] == n_clean_nonanchor
        assert arrays["resid_fit"][arrays["anchor"]] == pytest.approx(0.0, abs=1e-12)


# ----------------------------------------------------------------------------------
# Gaussian-filter scale separation
# ----------------------------------------------------------------------------------
class TestGaussianScale:
    def test_fine_scale_suppressed_coarse_retained(self):
        """sigma = 1.6 px kills a 2-px checkerboard but keeps a 32-px sine.

        Gaussian transfer function exp(-sigma^2 |k|^2 / 2) is ~e^-25 for the
        checkerboard and ~0.95 at lambda = 32 px. Assertions are made on the interior
        (8 px trimmed) because the reflect-padding at the array corners turns the
        checkerboard locally constant; in production the wake mask sits >= 16 px from
        every domain edge, so the boundary mode is irrelevant there.
        """
        nx, ny = pp.NX, pp.NY
        x = np.arange(nx)[:, None]
        y = np.arange(ny)[None, :]
        coarse = np.sin(TWO_PI * x / 32.0) * np.ones((1, ny))
        fine = (x + y) % 2 * 2.0 - 1.0  # +-1 checkerboard, sqrt(2)-px wavelength
        from scipy.ndimage import gaussian_filter

        sigma = 0.05 / pp.DX  # the production scale: 1.6 px
        assert sigma == pytest.approx(1.6)
        interior = (slice(8, -8), slice(8, -8))
        filt_coarse = gaussian_filter(coarse, sigma)[interior]
        filt_fine = gaussian_filter(fine, sigma)[interior]
        assert np.abs(filt_fine).max() < 0.01 * np.abs(fine).max()
        assert np.abs(filt_coarse).max() > 0.85 * np.abs(coarse).max()

    def test_wake_enstrophy_localises_to_wake_region(self):
        """Vorticity outside the wake region must not contribute to E(t)."""
        omega = np.zeros((3, pp.NX, pp.NY))
        xc = pp.X_MIN + (np.arange(pp.NX) + 0.5) * pp.DX
        yc = pp.Y_MIN + (np.arange(pp.NY) + 0.5) * pp.DY
        i_wake = int(np.argmin(np.abs(xc - 2.0)))
        j_mid = int(np.argmin(np.abs(yc)))
        i_out = int(np.argmin(np.abs(xc - (-1.0))))  # upstream of the wake box
        omega[1, i_wake - 3 : i_wake + 3, j_mid - 3 : j_mid + 3] = 100.0
        omega[2, i_out - 3 : i_out + 3, j_mid - 3 : j_mid + 3] = 100.0
        ens = pp.large_scale_wake_enstrophy(omega, sigma_px=1.6)
        assert ens[0] == 0.0
        assert ens[1] > 0.0
        assert ens[2] < 1e-9 * ens[1]


# ----------------------------------------------------------------------------------
# Scaling candidates: numeric vs analytic
# ----------------------------------------------------------------------------------
class TestScalingCandidates:
    def test_taylor_profile_peaks_at_R_with_value_G(self):
        G, D = 2.0, 1.0
        r = np.linspace(1e-6, 4.0, 100001)
        u = pp.taylor_u_theta(r, G, D)
        i = np.argmax(np.abs(u))
        assert r[i] == pytest.approx(0.5 * D, abs=1e-3)
        assert u[i] == pytest.approx(G, rel=1e-6)

    def test_core_circulation_matches_analytic(self):
        """Numeric integral must reproduce Gamma_g = 2 pi e^{-1/2} G D."""
        for G, D in [(1.0, 0.5), (-3.0, 1.5), (4.0, 1.0), (0.25, 0.5)]:
            expected = TWO_PI * math.exp(-0.5) * G * D
            assert pp.taylor_circulation(G, D) == pytest.approx(expected, rel=1e-3), (G, D)

    def test_circulation_zero_for_baseline(self):
        assert pp.taylor_circulation(0.0, 0.0) == 0.0

    def test_mmf_ratio_reduces_to_G_at_Y_zero(self):
        for G, D in [(1.0, 0.5), (-2.0, 1.5)]:
            assert pp.mmf_velocity_ratio(G, D, 0.0) == pytest.approx(G, rel=1e-4)

    def test_mmf_ratio_decays_with_miss_distance_and_keeps_sign(self):
        G, D = -2.0, 1.0
        v0 = pp.mmf_velocity_ratio(G, D, 0.0)
        v1 = pp.mmf_velocity_ratio(G, D, 0.2)
        v2 = pp.mmf_velocity_ratio(G, D, 0.4)
        assert v0 < v1 < v2 < 0  # negative G: all negative, magnitude decreasing
        assert abs(v2) < abs(v1) < abs(v0)

    def test_mmf_zero_for_baseline(self):
        assert pp.mmf_velocity_ratio(0.0, 0.0, 0.0) == 0.0


# ----------------------------------------------------------------------------------
# Coverage statistics
# ----------------------------------------------------------------------------------
class TestCoverage:
    def test_clustered_phases_have_low_coverage(self):
        phi = np.array([1.0, 1.05, 1.1, 0.95, 1.02])
        cov = pp.coverage_stats(phi, n_bins=12)
        assert cov["occupied_bin_fraction"] <= 2 / 12
        assert cov["arc_coverage_fraction"] < 0.1
        assert cov["R"] > 0.99

    def test_uniform_phases_have_high_coverage(self):
        phi = np.linspace(0, TWO_PI, 25, endpoint=False)
        cov = pp.coverage_stats(phi, n_bins=12)
        assert cov["occupied_bin_fraction"] == 1.0
        assert cov["arc_coverage_fraction"] > 0.9
        assert cov["R"] < 0.05


# ----------------------------------------------------------------------------------
# End-to-end on a synthetic encounter file
# ----------------------------------------------------------------------------------
class TestProcessEncounterSynthetic:
    def test_full_row_on_synthetic_h5(self, tmp_path):
        h5py = pytest.importorskip("h5py")
        n, impact = 120, 40
        omega_t = TWO_PI / 30.0
        cl = make_lift(n, omega_t, 0.5, noise=0.002, seed=7)
        t = np.arange(n, dtype=float)
        cl = cl + 2.0 * np.exp(-0.5 * ((t - 52.0) / 4.0) ** 2)

        # Frozen base wake field modulated by a deterministic shedding-like sinusoid:
        # the pre-impact mean +- 2 sd envelope then covers the post-recovery oscillation
        # (iid per-frame noise would break the dwell run ~5 percent of frames by design
        # of the 2-sd rule, which is a property of noise, not of recovery).
        rng = np.random.default_rng(11)
        base = rng.standard_normal((pp.NX, pp.NY)) * 0.5
        modulation = 1.0 + 0.01 * np.sin(omega_t * np.arange(n))
        omega = base[None, :, :] * modulation[:, None, None]
        xc = pp.X_MIN + (np.arange(pp.NX) + 0.5) * pp.DX
        i0 = int(np.argmin(np.abs(xc - 2.0)))
        omega[impact:80, i0 - 8 : i0 + 8, 40:56] += 50.0  # wake excursion, recovers at 80

        path = tmp_path / "encounter_00.h5"
        with h5py.File(path, "w") as f:
            f["C_L"] = cl.astype(np.float32)
            f["C_D"] = np.zeros(n, dtype=np.float32)
            f["omega_z"] = omega.astype(np.float32)
            f.attrs.update(
                {
                    "case_id": "G+1.00_D0.50_Y+0.10",
                    "encounter_index": 0,
                    "G": 1.0,
                    "D": 0.5,
                    "Y": 0.1,
                    "impact_frame_estimate": impact,
                    "source_group": "periodic",
                }
            )

        params = pp.PrepParams(dwell_frames=30)
        row = pp.process_encounter(path, "train", params)
        assert row["split"] == "train"
        assert row["impact_frame"] == impact
        assert 0.0 <= row["phi_imp"] < TWO_PI
        assert row["dcl_peak_phase_matched"] == pytest.approx(2.0, abs=0.4)
        assert row["recovery_status"] == "recovered"
        assert row["tau_rec_frame"] == pytest.approx(80.0, abs=2.0)
        assert row["tau_rec_tc"] == pytest.approx((80 - impact) * pp.DT_TC, abs=0.1)
        assert row["s1"] == 1.0
        assert row["s2"] == 0.5
        assert row["s3"] == pytest.approx(TWO_PI * math.exp(-0.5) * 0.5, rel=1e-3)
        assert 0 < row["s4"] < 1.0  # |Y| = 0.1 with R = 0.25: attenuated below G
        assert math.isfinite(row["denstrophy_peak_post"])
        assert row["denstrophy_peak_post_signed"] > 0
