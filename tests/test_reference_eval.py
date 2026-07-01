"""TDD for the Session 31 reference-baseline (fukami / POD) pure logic.

No GPU, no cache: the POD linear algebra (fit / project / reconstruct round-trip
and orthonormality), the POD frozen-encoder module shape + linear correctness, and
the reference report plumbing (REFERENCE_MODELS registry, alphabetic macros).
The heavy fukami train + GPU encode are exercised by the runner, not here.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.baselines.pod import (
    fit_pod_basis,
    load_pod_basis,
    pod_project,
    pod_reconstruct,
    save_pod_basis,
)


# --------------------------------------------------------------------------- POD math
def _toy_snapshots(n: int = 200, p: int = 48, seed: int = 0) -> np.ndarray:
    """A rank-3 signal plus small noise so a d=3 POD basis captures ~all energy."""
    rng = np.random.default_rng(seed)
    basis = rng.standard_normal((3, p))
    coeffs = rng.standard_normal((n, 3)) * np.array([5.0, 3.0, 1.0])
    return coeffs @ basis + 0.01 * rng.standard_normal((n, p))


def test_pod_basis_shapes_and_orthonormal() -> None:
    X = _toy_snapshots()
    basis = fit_pod_basis(X, d=3, height=6, width=8)
    assert basis.components.shape == (48, 3)
    assert basis.mean.shape == (48,)
    assert basis.d == 3
    # modes are orthonormal columns
    gram = basis.components.T @ basis.components
    assert np.allclose(gram, np.eye(3), atol=1e-4)


def test_pod_energy_fraction_high_for_low_rank_signal() -> None:
    X = _toy_snapshots()
    basis = fit_pod_basis(X, d=3, height=6, width=8)
    assert basis.energy_fraction > 0.99


def test_pod_project_reconstruct_roundtrip() -> None:
    X = _toy_snapshots()
    basis = fit_pod_basis(X, d=3, height=6, width=8)
    coeffs = pod_project(X, basis)
    assert coeffs.shape == (X.shape[0], 3)
    recon = pod_reconstruct(coeffs, basis)
    # rank-3 signal is reconstructed to within the injected noise floor
    rel = np.linalg.norm(recon - X) / np.linalg.norm(X)
    assert rel < 0.05


def test_pod_project_accepts_thw() -> None:
    X = _toy_snapshots(n=10, p=48)
    basis = fit_pod_basis(X, d=3, height=6, width=8)
    frames = X.reshape(10, 6, 8)
    c_thw = pod_project(frames, basis)
    c_flat = pod_project(X, basis)
    assert np.allclose(c_thw, c_flat)


def test_pod_save_load_roundtrip(tmp_path) -> None:
    X = _toy_snapshots()
    basis = fit_pod_basis(X, d=3, height=6, width=8)
    path = tmp_path / "pod_basis.npz"
    save_pod_basis(path, basis)
    loaded = load_pod_basis(path)
    assert np.allclose(loaded.components, basis.components)
    assert np.allclose(loaded.mean, basis.mean)
    assert loaded.energy_fraction == pytest.approx(basis.energy_fraction)
    assert (loaded.height, loaded.width) == (6, 8)


def test_pod_fit_rejects_bad_d() -> None:
    X = _toy_snapshots(n=10, p=48)
    with pytest.raises(ValueError):
        fit_pod_basis(X, d=0)
    with pytest.raises(ValueError):
        fit_pod_basis(X, d=11)  # > min(n, p) is fine here (n=10) -> d>10 invalid


# --------------------------------------------------------------------------- POD encoder module
def test_pod_encoder_module_matches_projection() -> None:
    import torch

    from src.evaluation.rom_eval import _PODEncoder

    # Build a small basis at the true field size but tiny d.
    rng = np.random.default_rng(1)
    P = 192 * 96
    snaps = rng.standard_normal((30, P)).astype(np.float32)
    basis = fit_pod_basis(snaps, d=4, height=192, width=96)
    enc = _PODEncoder(basis)
    assert enc.latent_mode == "pooled"
    # (B, T, 1, H, W) normalised omega
    x = rng.standard_normal((1, 5, 1, 192, 96)).astype(np.float32)
    z = enc(torch.from_numpy(x))
    assert tuple(z.shape) == (1, 5, 4)
    # matches the numpy projection frame-for-frame
    ref = pod_project(x[0, :, 0], basis)
    assert np.allclose(z[0].numpy(), ref, atol=1e-3)


# --------------------------------------------------------------------------- report plumbing
def test_reference_models_registered() -> None:
    from src.evaluation.report_session31 import (
        MODEL_LABEL,
        MODEL_SHORT,
        REFERENCE_MODELS,
    )

    assert REFERENCE_MODELS == ("fukami", "fukami_wake", "pod")
    for m in REFERENCE_MODELS:
        assert m in MODEL_SHORT
        assert m in MODEL_LABEL
        assert MODEL_SHORT[m].isalpha()  # LaTeX command names must be alphabetic


def test_reference_name_and_macro_alphabetic() -> None:
    from src.evaluation.report_session31 import REFERENCE_MODELS, name_and_macro

    for m in REFERENCE_MODELS:
        _n, macro = name_and_macro("repr_floor_vrmse", m)
        assert macro.isalpha()
        _n, macro = name_and_macro("fore_merit", m, h=8)
        assert macro.isalpha()
