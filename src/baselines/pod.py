"""POD (Proper Orthogonal Decomposition) linear reference for Session 31.

The classic linear floor at matched latent dimension ``d``. The basis is fit on
the pipeline-normalised omega snapshots of the v2.2 train pool; a frame is encoded
by mean-subtraction and projection onto the top-``d`` left-singular vectors, giving
a ``d``-vector "pooled" latent that flows through the exact same Session 31
frozen-probe harness as the neural encoders (the pooled -> spatial broadcast used
for the ``jepa_pool`` ablation).

This module holds only the PURE linear algebra (unit-tested in
``tests/test_reference_eval.py``); the training script
``scripts/session31/fit_pod.py`` gathers the snapshots and persists the basis, and
``src.evaluation.rom_eval.load_reference_model`` wraps it as a frozen encoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PODBasis:
    """A fitted POD basis: snapshot mean + the top-``d`` spatial modes.

    Attributes:
        mean: ``(P,)`` snapshot mean (``P = H * W``), pipeline-normalised units.
        components: ``(P, d)`` orthonormal spatial modes (``Phi``); frame
            coefficients are ``(x - mean) @ components``.
        singular_values: ``(d,)`` singular values of the centred snapshot matrix.
        energy_fraction: cumulative energy captured at rank ``d``.
        height: field height ``H`` (192).
        width: field width ``W`` (96).
    """

    mean: np.ndarray
    components: np.ndarray
    singular_values: np.ndarray
    energy_fraction: float
    height: int
    width: int

    @property
    def d(self) -> int:
        return int(self.components.shape[1])


def fit_pod_basis(snapshots: np.ndarray, d: int, *, height: int = 192, width: int = 96) -> PODBasis:
    """Fit a rank-``d`` POD basis from a centred snapshot matrix.

    Args:
        snapshots: ``(N, P)`` matrix of ``N`` flattened frames (``P = H * W``),
            pipeline-normalised.
        d: Number of modes to keep.
        height: Field height for the stored metadata.
        width: Field width for the stored metadata.

    Returns:
        A :class:`PODBasis`. Uses the economy SVD of the centred matrix so the
        modes are the leading right-singular vectors (spatial patterns).
    """
    X = np.asarray(snapshots, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"snapshots must be 2-D (N, P); got {X.shape}")
    n, p = X.shape
    if d < 1 or d > min(n, p):
        raise ValueError(f"d={d} out of range for a ({n}, {p}) snapshot matrix")
    mean = X.mean(axis=0)
    Xc = X - mean[None, :]
    # Economy SVD: Xc = U S Vt; the rows of Vt are the spatial modes.
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    components = vt[:d].T.astype(np.float32)  # (P, d)
    singular_values = s[:d].astype(np.float64)
    total = float((s**2).sum())
    energy_fraction = float((singular_values**2).sum() / total) if total > 0 else 0.0
    return PODBasis(
        mean=mean.astype(np.float32),
        components=components,
        singular_values=singular_values,
        energy_fraction=energy_fraction,
        height=int(height),
        width=int(width),
    )


def pod_project(frames: np.ndarray, basis: PODBasis) -> np.ndarray:
    """Project ``(T, H, W)`` (or ``(T, P)``) normalised frames onto the POD modes.

    Returns:
        ``(T, d)`` float32 POD coefficients (the "pooled" latent).
    """
    X = np.asarray(frames, dtype=np.float32)
    if X.ndim == 3:
        X = X.reshape(X.shape[0], -1)
    elif X.ndim != 2:
        raise ValueError(f"frames must be (T, H, W) or (T, P); got {frames.shape}")
    if X.shape[1] != basis.components.shape[0]:
        raise ValueError(f"frame feature dim {X.shape[1]} != basis P {basis.components.shape[0]}")
    return ((X - basis.mean[None, :]) @ basis.components).astype(np.float32)


def pod_reconstruct(coeffs: np.ndarray, basis: PODBasis) -> np.ndarray:
    """Inverse map ``(T, d) -> (T, P)`` normalised field: ``mean + coeffs @ Phi^T``."""
    C = np.asarray(coeffs, dtype=np.float32)
    return (basis.mean[None, :] + C @ basis.components.T).astype(np.float32)


def save_pod_basis(path, basis: PODBasis) -> None:
    """Persist a :class:`PODBasis` to an ``.npz`` file."""
    np.savez(
        path,
        mean=basis.mean.astype(np.float32),
        components=basis.components.astype(np.float32),
        singular_values=basis.singular_values.astype(np.float64),
        energy_fraction=np.float64(basis.energy_fraction),
        height=np.int64(basis.height),
        width=np.int64(basis.width),
    )


def load_pod_basis(path) -> PODBasis:
    """Load a :class:`PODBasis` from an ``.npz`` written by :func:`save_pod_basis`."""
    d = np.load(path, allow_pickle=False)
    return PODBasis(
        mean=np.asarray(d["mean"], dtype=np.float32),
        components=np.asarray(d["components"], dtype=np.float32),
        singular_values=np.asarray(d["singular_values"], dtype=np.float64),
        energy_fraction=float(d["energy_fraction"]),
        height=int(d["height"]),
        width=int(d["width"]),
    )
