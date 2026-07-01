"""TDD for the pure pieces of Session 31 Track D Q3 (pressure -> latent -> readout).

These exercise, with no GPU / no encoder / no cache:
- the pressure -> latent-row alignment (:func:`align_pressure_to_rows`), which
  joins per-encounter wall pressure onto the flat Q1 latent rows by (case, enc,
  frame);
- the aggregated latent-estimation R^2 (:func:`latent_estimation_r2`);
- the window-restricted readout compose: an *identity* pressure -> latent map fed
  through the frozen observable probe recovers the *direct* probe R^2 exactly, and
  through the field readout recovers the decode floor exactly (the estimator +
  readout plumbing is correct and window selection stays aligned).

The heavy GPU decoder fit + kernel-ridge estimator fit are exercised by the
runner, not here.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.pressure_infer import (
    IdentityEstimator,
    align_pressure_to_rows,
    latent_estimation_r2,
    readout_field_from_pressure,
    readout_observable_from_pressure,
)
from src.evaluation.represent import fit_linear_probe, r2_score_np


# --------------------------------------------------------------------------- alignment
def test_align_pressure_to_rows_matches_keys_and_frames() -> None:
    """Each flat row gets the p_wall of its (case, enc) at its own frame index."""
    # Two encounters, 3 and 2 frames, 4 taps each; flat rows are interleaved.
    pw = {
        "A/00": np.arange(3 * 4, dtype=np.float32).reshape(3, 4),  # frames 0,1,2
        "B/01": 100.0 + np.arange(2 * 4, dtype=np.float32).reshape(2, 4),  # frames 0,1
    }
    case_id = np.array(["A", "B", "A", "A", "B"])
    enc = np.array([0, 1, 0, 0, 1])
    frame = np.array([2, 0, 0, 1, 1])
    out = align_pressure_to_rows(pw, case_id, enc, frame, n_surface=4)
    assert out.shape == (5, 4)
    assert np.array_equal(out[0], pw["A/00"][2])
    assert np.array_equal(out[1], pw["B/01"][0])
    assert np.array_equal(out[2], pw["A/00"][0])
    assert np.array_equal(out[3], pw["A/00"][1])
    assert np.array_equal(out[4], pw["B/01"][1])


def test_align_pressure_to_rows_missing_key_raises() -> None:
    pw = {"A/00": np.zeros((2, 4), dtype=np.float32)}
    with pytest.raises(KeyError):
        align_pressure_to_rows(
            pw, np.array(["A", "C"]), np.array([0, 0]), np.array([0, 1]), n_surface=4
        )


# --------------------------------------------------------------------------- latent R^2
def test_latent_estimation_r2_perfect_is_one() -> None:
    z = np.random.RandomState(0).randn(20, 8)
    assert latent_estimation_r2(z.copy(), z, mask=None) == pytest.approx(1.0)


def test_latent_estimation_r2_mean_prediction_is_zero() -> None:
    z = np.random.RandomState(1).randn(30, 5)
    pred = np.broadcast_to(z.mean(axis=0, keepdims=True), z.shape).copy()
    assert latent_estimation_r2(pred, z, mask=None) == pytest.approx(0.0, abs=1e-9)


def test_latent_estimation_r2_respects_window_mask() -> None:
    """Masked-out rows must not affect the score (aligned selection)."""
    z = np.random.RandomState(2).randn(10, 4)
    pred = z.copy()
    pred[5:] = 999.0  # corrupt the out-of-window rows
    mask = np.zeros(10, dtype=bool)
    mask[:5] = True
    assert latent_estimation_r2(pred, z, mask=mask) == pytest.approx(1.0)


# --------------------------------------------------------------------------- compose
def test_identity_pressure_recovers_direct_probe_r2() -> None:
    """Identity pressure->latent map + frozen probe == the direct probe R^2.

    If the pressure "estimate" *is* the true latent, reading the observable
    through the frozen probe must reproduce the direct-probe number exactly. This
    pins the estimator+readout+window plumbing.
    """
    rng = np.random.RandomState(3)
    d = 6
    z_tr = rng.randn(200, d)
    w = rng.randn(d)
    y_tr = z_tr @ w + 0.05 * rng.randn(200)
    z_ev = rng.randn(60, d)
    y_ev = z_ev @ w + 0.05 * rng.randn(60)
    mask = np.zeros(60, dtype=bool)
    mask[:40] = True

    probe = fit_linear_probe(z_tr, y_tr)
    # direct probe R^2 on the windowed eval rows (true latent).
    direct = r2_score_np(y_ev[mask], probe.predict(z_ev[mask]))
    # pressure "=" the true latent, via an identity estimator.
    got = readout_observable_from_pressure(IdentityEstimator(), probe, z_ev, y_ev, mask)
    assert got["pressure_r2"] == pytest.approx(direct, rel=1e-9, abs=1e-9)
    assert got["n"] == int(mask.sum())


def test_identity_pressure_recovers_field_floor() -> None:
    """Identity pressure->spatial-latent + a decoder callable == the decode floor.

    The field readout decodes the estimated latent; with an identity estimator the
    decoded field is the same as decoding the true latent (the floor), so VRMSE and
    SSIM match to numerical precision on the windowed rows.
    """
    rng = np.random.RandomState(4)
    n, d, h, w = 12, 3, 4, 4
    z_true = rng.randn(n, d, h, w).astype(np.float32)
    field_true = rng.randn(n, 8, 8).astype(np.float32)  # DNS-like target
    mask = np.zeros(n, dtype=bool)
    mask[[0, 2, 4, 6, 8]] = True

    # A deterministic "decoder": flatten+linear map latent -> field, as a stand-in
    # for the SpatialLatentFieldDecoder (any fixed map exercises the plumbing).
    proj = rng.randn(d * h * w, 8 * 8).astype(np.float32)

    def decode(z_flat_batch: np.ndarray) -> np.ndarray:
        flat = z_flat_batch.reshape(z_flat_batch.shape[0], -1)
        return (flat @ proj).reshape(-1, 8, 8)

    z_flat = z_true.reshape(n, -1)
    out = readout_field_from_pressure(
        IdentityEstimator(),
        decode,
        z_flat,
        z_true_flat=z_flat,
        field_true=field_true,
        latent_grid=(h, w),
        latent_dim=d,
        mask=mask,
        ssim_L=2.0,
    )
    # identity estimate == true latent -> model curve equals the floor curve.
    assert out["pressure_vrmse"] == pytest.approx(out["floor_vrmse"], rel=1e-6, abs=1e-6)
    assert out["pressure_ssim"] == pytest.approx(out["floor_ssim"], rel=1e-6, abs=1e-6)
    assert out["n"] == int(mask.sum())
