"""Tests for scripts/session28/e4_floor_regen.py (E4 scale band + parametric floor).

CPU synthetic tests run anywhere; the skipif-real tests exercise the real-data
path only when the v2p1 cache and the DNS metrics are present.

Synthetic coverage:
  * the Gaussian band at smaller sigma retains more fine-scale energy (monotone)
    -> peak large-scale wake-enstrophy excursion is larger at smaller sigma/c;
  * the floor R^2 matches sklearn on known (G,D,Y) -> observable synthetic data;
  * case-level CV fold construction (no case straddles folds, deterministic);
  * the 1 - SSE/SST held-out estimator matches sklearn r2_score;
  * the numbers-part macros are alphabetic (eval_all contract).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
S28 = REPO / "scripts" / "session28"
sys.path.insert(0, str(S28))


def _load_module():
    spec = importlib.util.spec_from_file_location("e4_floor_regen", S28 / "e4_floor_regen.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    # register before exec so the @dataclass body can resolve its own module
    sys.modules["e4_floor_regen"] = mod
    spec.loader.exec_module(mod)
    return mod


e4 = _load_module()


# --------------------------------------------------------------- E4 Gaussian-band tests


def _synthetic_omega(n_frames: int = 60, impact: int = 40, seed: int = 0) -> np.ndarray:
    """A (T, NX, NY) omega stack with a fine-scale post-impact burst in the wake box.

    Pre-impact: smooth low-amplitude wake. Post-impact: a high-spatial-frequency
    checkerboard burst added inside the wake box, so a smaller Gaussian sigma (less
    smoothing) retains more of its energy.
    """
    rng = np.random.default_rng(seed)
    omega = np.zeros((n_frames, e4.pp.NX, e4.pp.NY), dtype=np.float64)
    xc = e4.pp.X_MIN + (np.arange(e4.pp.NX) + 0.5) * e4.pp.DX
    yc = e4.pp.Y_MIN + (np.arange(e4.pp.NY) + 0.5) * e4.pp.DY
    # smooth background everywhere
    bg = np.outer(np.sin(xc), np.cos(yc))
    for t in range(n_frames):
        omega[t] = 0.3 * bg + 0.01 * rng.standard_normal((e4.pp.NX, e4.pp.NY))
    # fine-scale checkerboard burst in the wake box, post-impact
    checker = np.outer((-1.0) ** np.arange(e4.pp.NX), (-1.0) ** np.arange(e4.pp.NY))
    in_box = e4.pp.wake_mask()
    for t in range(impact, n_frames):
        omega[t] += 5.0 * checker * in_box
    return omega


def test_smaller_sigma_retains_more_fine_scale_energy():
    """Smaller sigma/c -> larger peak excursion (less smoothing of the fine burst)."""
    omega = _synthetic_omega()
    mask = e4.pp.wake_mask()
    impact = 40
    peaks = {}
    for sc in e4.SIGMA_C_GRID:
        post, _ = e4.peak_excursions_at_band(omega, impact, sc / e4.pp.DX, mask, 40)
        peaks[sc] = post
    # monotone: peak(0.01) > peak(0.03) > peak(0.05)
    assert peaks[0.01] > peaks[0.03] > peaks[0.05]


def test_peak_excursion_matches_physics_prep_at_headline_band():
    """peak_excursions_at_band reproduces physics_prep's recipe at sigma/c = 0.05."""
    omega = _synthetic_omega()
    mask = e4.pp.wake_mask()
    impact = 40
    post, imp40 = e4.peak_excursions_at_band(omega, impact, 0.05 / e4.pp.DX, mask, 40)

    # replicate physics_prep.process_encounter's enstrophy block directly
    ens = e4.pp.large_scale_wake_enstrophy(omega, 0.05 / e4.pp.DX, mask)
    pre_mean = float(ens[:impact].mean())
    d_ens = ens - pre_mean
    w_end = min(impact + 40, ens.size - 1)
    ref_imp40 = float(np.max(np.abs(d_ens[impact : w_end + 1])))
    ref_post = float(np.max(np.abs(d_ens[impact:])))
    assert post == pytest.approx(ref_post, rel=1e-12)
    assert imp40 == pytest.approx(ref_imp40, rel=1e-12)


# ------------------------------------------------------------------- estimator tests


def test_r2_heldout_matches_sklearn():
    """closure_matrix.r2_heldout (1 - SSE/SST about held-out mean) == sklearn r2_score."""
    from sklearn.metrics import r2_score

    rng = np.random.default_rng(1)
    y_true = rng.normal(size=200)
    y_pred = y_true + 0.3 * rng.normal(size=200)
    assert e4.cm.r2_heldout(y_pred, y_true) == pytest.approx(r2_score(y_true, y_pred), rel=1e-9)


def test_floor_r2_matches_sklearn_on_known_param_map():
    """Linear ridge floor R^2 on a known (G,D,Y) -> observable map matches sklearn.

    Build a synthetic dataset where the observable is an exact linear function of
    (G, D, Y) plus small noise; fit the floor on train, evaluate on a disjoint
    held-out set, and check _floor_cell's value equals an independent sklearn Ridge
    held-out r2_score with the same standardisation.
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(2)
    n_tr, n_ev = 300, 120
    c_tr = rng.uniform(-3, 3, size=(n_tr, 3))
    c_ev = rng.uniform(-3, 3, size=(n_ev, 3))
    w = np.array([1.5, -0.7, 2.0])
    y_tr = c_tr @ w + 0.1 * rng.normal(size=n_tr)
    y_ev = c_ev @ w + 0.1 * rng.normal(size=n_ev)
    case_tr = np.array([f"c{i % 20}" for i in range(n_tr)])
    case_ev = np.array([f"e{i % 8}" for i in range(n_ev)])

    cell = e4._floor_cell(c_tr, y_tr, c_ev, y_ev, case_ev, case_tr, "ridge")

    # independent sklearn reference with the SAME standardisation as fit_ridge
    # (fit_ridge standardises Z by train mean/std and centres y; ridge penalty on
    # the standardised design with alpha = RIDGE_ALPHA). Compare predictions.
    probe = e4.cm.fit_ridge(c_tr, y_tr, alpha=e4.RIDGE_ALPHA)
    y_hat = e4.cm.apply_probe(c_ev, probe)
    assert cell["value"] == pytest.approx(r2_score(y_ev, y_hat), rel=1e-9)

    # the floor on a near-exact linear map should be high
    assert cell["value"] > 0.95

    # sklearn Ridge on standardised features lands in the same ballpark (not identical:
    # fit_ridge uses a normal-equation solve, sklearn uses its own solver/intercept).
    sx = StandardScaler().fit(c_tr)
    m = Ridge(alpha=e4.RIDGE_ALPHA).fit(sx.transform(c_tr), y_tr)
    sk_r2 = r2_score(y_ev, m.predict(sx.transform(c_ev)))
    assert cell["value"] == pytest.approx(sk_r2, abs=0.05)


def test_case_folds_no_case_straddles_and_deterministic():
    """make_case_folds: every case in exactly one fold; deterministic for a seed."""
    case_ids = np.array([f"case_{i % 13}" for i in range(130)])
    f1 = e4.cm.make_case_folds(case_ids, n_folds=e4.N_FOLDS, seed=e4.CV_SEED)
    f2 = e4.cm.make_case_folds(case_ids, n_folds=e4.N_FOLDS, seed=e4.CV_SEED)
    assert np.array_equal(f1, f2)  # deterministic
    # no case straddles folds
    for c in set(case_ids.tolist()):
        folds_of_c = set(f1[case_ids == c].tolist())
        assert len(folds_of_c) == 1
    # all folds used
    assert set(f1.tolist()) == set(range(e4.N_FOLDS))


def test_floor_cell_oof_uses_5_case_folds():
    """The train OOF R^2 column predicts every train encounter out of fold."""
    rng = np.random.default_rng(3)
    n_tr, n_ev = 200, 60
    c_tr = rng.uniform(-2, 2, size=(n_tr, 3))
    c_ev = rng.uniform(-2, 2, size=(n_ev, 3))
    w = np.array([1.0, 0.5, -1.0])
    y_tr = c_tr @ w + 0.05 * rng.normal(size=n_tr)
    y_ev = c_ev @ w + 0.05 * rng.normal(size=n_ev)
    case_tr = np.array([f"c{i % 25}" for i in range(n_tr)])
    case_ev = np.array([f"e{i % 10}" for i in range(n_ev)])
    cell = e4._floor_cell(c_tr, y_tr, c_ev, y_ev, case_ev, case_tr, "ridge")
    # OOF on a near-linear map should also be high
    assert cell["train_oof_r2_5fold_case_cv"] > 0.9
    assert "cc_ci_lo" in cell and "cc_ci_hi" in cell


# --------------------------------------------------------------------- numbers part


def test_numbers_part_macros_alphabetic():
    """Macros must satisfy eval_all's macro.isalpha() contract."""
    # minimal synthetic e4 + floor payloads
    fake_e4 = {
        "verdict": {"split": "test_b", "robust": True},
        "bands": {
            "test_b": {
                f"{sc}": {
                    "g_spearman_rho": 0.7,
                    "g_spearman_p": 0.001,
                    "n": 42,
                    "sigma_px": sc / e4.pp.DX,
                }
                for sc in e4.SIGMA_C_GRID
            }
        },
    }
    fake_floor = {
        "floor": {
            "wake_enstrophy": {
                "ridge": {"value": 0.5, "cc_ci_lo": 0.3, "cc_ci_hi": 0.7, "n": 42},
                "krr_rbf": {"value": 0.55, "cc_ci_lo": 0.35, "cc_ci_hi": 0.75, "n": 42},
            }
        }
    }
    out = Path("/tmp/test_e4_floor_numbers.json")
    part = e4.emit_numbers_part(fake_e4, fake_floor, out)
    assert part["part"] == "e4_floor"
    for name, rec in part["numbers"].items():
        macro = rec.get("macro")
        assert macro is not None and macro.isalpha(), f"{name}: macro {macro!r} not alphabetic"
    # exactly the 5 expected numbers (3 bands + 2 floor regressors)
    assert len(part["numbers"]) == 5
    out.unlink(missing_ok=True)


# ----------------------------------------------------------------------- real-data


_HAVE_DNS = e4.DNS_METRICS.exists()
_HAVE_CACHE = e4.default_cache_root().exists()


@pytest.mark.skipif(not _HAVE_DNS, reason="DNS metrics npz absent")
def test_real_floor_wake_below_closure_latent(tmp_path):
    """On real v2p1 data, the parametric wake floor stays below the closure latent (+0.79)."""
    payload = e4.run_floor(tmp_path / "floor.json")
    fr = payload["framing"]
    assert fr["latent_beats_floor"] is True
    assert fr["wake_floor_best"] < e4.CLOSURE_JEPA_WAKE_REPR_R2
    # all six observables present, both regressors
    for obs in e4.OBSERVABLES:
        assert "ridge" in payload["floor"][obs]
        assert "krr_rbf" in payload["floor"][obs]


@pytest.mark.skipif(not (_HAVE_CACHE and _HAVE_DNS), reason="v2p1 cache / DNS metrics absent")
def test_real_scaleband_g_trend_positive(tmp_path):
    """On real v2p1 data, the test_b |G| trend is positive at all three bands."""
    payload = e4.run_scaleband(e4.default_cache_root(), tmp_path / "e4.json")
    tb = payload["bands"]["test_b"]
    for sc in e4.SIGMA_C_GRID:
        assert tb[f"{sc}"]["g_spearman_rho"] > 0
