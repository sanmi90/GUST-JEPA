"""CPU-fast synthetic-data unit tests for scripts/session28/physics_prep.py.

No cache, no torch, no GPU: every test builds its own signal or field. Covers the
Hilbert impact-phase extraction (known sinusoid + drift), the phase-matched Delta C_L
reference, the tau_rec re-entry / dwell / censoring logic, the mean +- 2 sd envelope
rule, the Gaussian-filter scale separation, and the numeric Taylor-circulation and
Martinez-Muriel & Flores velocity-ratio candidates against their analytic values.

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
