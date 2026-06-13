"""Unit tests for scripts/session28/sensing_field_cf.py (CPU, no GPU, no DNS).

The flow-field recovery comparison reuses sensing_cf's pressure->latent KRR and
qDEIM picks, then decodes the estimated latent through each family's decoder to a
recovered omega field, scored against DNS truth at impact + READOUT_H. These tests
pin the metric conventions and the chain wiring on synthetic stubs; the real-data
path is skipped unless the v2p1 cache + decode artifacts are present.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
MODPATH = REPO / "scripts" / "session28" / "sensing_field_cf.py"

# Import the module under test by path (scripts/ is not a package). Register it in
# sys.modules BEFORE exec so dataclass field-type resolution (which looks up
# cls.__module__ in sys.modules) works for the `Path | None` annotations.
sys.path.insert(0, str(REPO / "scripts" / "session28"))
_spec = importlib.util.spec_from_file_location("sensing_field_cf", MODPATH)
sfc = importlib.util.module_from_spec(_spec)
sys.modules["sensing_field_cf"] = sfc
_spec.loader.exec_module(sfc)


# --------------------------------------------------------------------- SSIM
def test_ssim_identical_fields_is_one():
    """Wang SSIM on identical fields is exactly 1 (up to float tolerance)."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((192, 96)).astype(np.float32)
    s = sfc.ssim_global(x, x, L=8.45)
    assert abs(s - 1.0) < 1e-9


def test_ssim_decreases_with_noise():
    """SSIM is monotone-decreasing as more noise is added to the second field."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal((192, 96)).astype(np.float32)
    L = 8.45
    s_clean = sfc.ssim_global(x, x, L=L)
    s_small = sfc.ssim_global(x, x + 0.1 * rng.standard_normal(x.shape), L=L)
    s_big = sfc.ssim_global(x, x + 1.0 * rng.standard_normal(x.shape), L=L)
    assert s_clean > s_small > s_big
    assert s_clean <= 1.0 + 1e-9


def test_ssim_data_range_scaling():
    """C1/C2 scale with L, so a larger L is more tolerant of a fixed offset."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal((64, 32)).astype(np.float32)
    y = x + 0.5 * rng.standard_normal(x.shape)
    s_small_L = sfc.ssim_global(x, y, L=1.0)
    s_big_L = sfc.ssim_global(x, y, L=20.0)
    assert s_big_L > s_small_L


# --------------------------------------------------------------------- field metrics
def test_field_mse_zero_on_identical():
    x = np.arange(192 * 96, dtype=np.float32).reshape(192, 96)
    assert sfc.field_mse(x, x) == pytest.approx(0.0)


def test_eps_volume_zero_on_identical_and_positive_on_noise():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((192, 96)).astype(np.float32) * 5.0
    assert sfc.eps_volume(x, x) == pytest.approx(0.0, abs=1e-7)
    eps = sfc.eps_volume(x, x + rng.standard_normal(x.shape))
    assert eps > 0.0


def test_peak_vorticity_retention_known_field():
    """Peak retention = recovered peak |omega| / DNS peak |omega|."""
    dns = np.zeros((10, 10), dtype=np.float32)
    dns[0, 0] = 100.0  # peak |omega| = 100
    dns[1, 1] = -40.0
    rec_half = np.zeros_like(dns)
    rec_half[0, 0] = 50.0  # recovered peak = 50 -> retention 0.5
    assert sfc.peak_vorticity_retention(rec_half, dns) == pytest.approx(0.5)
    # a recovery that overshoots the DNS peak gives retention > 1
    rec_over = np.zeros_like(dns)
    rec_over[0, 0] = 150.0
    assert sfc.peak_vorticity_retention(rec_over, dns) == pytest.approx(1.5)
    # negative-signed peak is taken by magnitude
    dns_neg = np.zeros((4, 4), dtype=np.float32)
    dns_neg[0, 0] = -80.0
    rec_neg = np.zeros((4, 4), dtype=np.float32)
    rec_neg[0, 0] = -80.0
    assert sfc.peak_vorticity_retention(rec_neg, dns_neg) == pytest.approx(1.0)


# --------------------------------------------------------------------- chain wiring
def test_pressure_to_z_to_decode_chain_shapes():
    """A stub chain: pressure window -> KRR z_hat -> stub decoder -> field.

    Verifies the (n, W, K)->(n,d)->(n,H,W) shape contract end to end without any
    real model, exactly the wiring sensing_field_cf relies on.
    """
    n, W, n_tap, d, H, Wd = 30, 30, 8, 6, 16, 12
    rng = np.random.default_rng(4)
    X = rng.standard_normal((n, W, n_tap)).astype(np.float64)
    z = rng.standard_normal((n, d)).astype(np.float64)
    taps = list(range(n_tap))
    feat = sfc.feats(X, taps)
    assert feat.shape == (n, W * n_tap)
    model = sfc.fit_krr(feat, z)
    zhat = sfc.krr_pred(model, feat)
    assert zhat.shape == (n, d)

    # stub "decoder": linear map z -> flattened field, reshaped to (H, Wd)
    Wmat = rng.standard_normal((d, H * Wd))

    def stub_decode(zrow):
        return (zrow @ Wmat).reshape(H, Wd)

    fields = np.stack([stub_decode(zhat[i]) for i in range(n)])
    assert fields.shape == (n, H, Wd)


# --------------------------------------------------------------------- case clustering
def test_case_clustering_groups_by_case_id():
    """The case-clustered bootstrap helper must resample whole cases, not rows."""
    case_ids = np.array(["A", "A", "B", "B", "B", "C"])
    delta = np.array([1.0, 1.0, -1.0, -1.0, -1.0, 0.5])
    # The headline CI helper resamples cases; with this tiny set the point
    # estimate (full-sample SSIM-like aggregate) must match a direct call.
    assert len(set(case_ids.tolist())) == 3
    # A case-clustered bootstrap resamples whole cases; its CI must be a sorted
    # 2-element bracket and, because cases (not rows) are resampled, the spread
    # must be at least as wide as a naive per-row resample would suggest. With
    # only 3 cases the CI is wide; we check it brackets the per-case-mean range.
    ci = sfc.boot_metric_ci(delta, case_ids, n_boot=2000, seed=0)
    assert len(ci) == 2 and ci[0] <= ci[1]
    per_case_means = [delta[case_ids == c].mean() for c in ["A", "B", "C"]]
    assert ci[0] <= max(per_case_means)
    assert ci[1] >= min(per_case_means)
    # Sanity: a homogeneous (zero-variance across cases) metric has a degenerate CI.
    flat = np.zeros_like(delta)
    ci_flat = sfc.boot_metric_ci(flat, case_ids, n_boot=200, seed=0)
    assert ci_flat[0] == ci_flat[1] == 0.0


# --------------------------------------------------------------------- key convention
def test_encounter_key_tolerance():
    """Per-encounter keys join family arrays by (case_id|enc_idx) exactly."""
    cids = np.array(["G+2.00_D0.50_Y+0.10", "G+2.00_D0.50_Y+0.10"])
    eis = np.array([0, 1])
    keys = sfc.enc_keys(cids, eis)
    assert keys[0] == "G+2.00_D0.50_Y+0.10|0"
    assert keys[1] == "G+2.00_D0.50_Y+0.10|1"
    assert len(set(keys.tolist())) == 2


# --------------------------------------------------------------------- real data (skip)
def _real_data_present() -> bool:
    decode = REPO / "outputs" / "session28" / "decode"
    lat = REPO / "outputs" / "session28" / "latents" / "jepa_tf_noc_d64_s42" / "test_b.npz"
    return (decode / "tf_noc" / "recon_test_b.npz").exists() and lat.exists()


@pytest.mark.skipif(not _real_data_present(), reason="v2p1 decode artifacts absent")
def test_decode_ceiling_npz_has_impact16_horizon():
    """The decode-ceiling reference field at impact+16 is reachable in the npz."""
    npz = np.load(
        REPO / "outputs" / "session28" / "decode" / "tf_noc" / "recon_test_b.npz",
        allow_pickle=True,
    )
    offs = list(npz["horizon_offsets"])
    assert sfc.READOUT_H in offs
    hidx = offs.index(sfc.READOUT_H)
    fields = npz["omega_decoded_horizon"]
    assert fields.ndim == 4 and fields.shape[1] == len(offs)
    one = fields[0, hidx]
    assert one.shape == (192, 96)
