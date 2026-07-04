"""Chang force-element machinery for the near-body observable (Session 34, Track C).

Implements the lift auxiliary potential phi_L of Chang (Proc. R. Soc. Lond. A 437,
1992): the Laplace problem

    lap(phi_L) = 0        outside the body,
    n . grad(phi_L) = -n . e_L   on the body surface,
    phi_L -> 0            far field,

where ``e_L`` is the unit lift direction and ``n`` the solid-outward normal.
The per-frame lift-element density on the mid-plane is

    e(x, t) = omega_z * (-v * dphi/dx + u * dphi/dy)
            = (omega x u) . grad(phi_L)   restricted to 2D,

whose volume integral tracks the circulatory lift. The near-body observable
target (``src.data.nearbody_observables``) weights this field with a
sign-symmetric wall-distance band so the supervision follows the LEV across
gust sign.

Grid conventions (matching the encounter cache): arrays are ``(H=192, W=96)``
with axis 0 = x in ``[-1.5, 4.5]`` and axis 1 = y in ``[-1.5, 1.5]`` (node
coordinates, so ``dx = 6/191``, ``dy = 3/95``). The airfoil is chord-aligned
with the x axis (footprint angle 0 deg); the freestream is inclined at
``alpha = 14 deg``, so the lift direction is ``e_L = (-sin(alpha), cos(alpha))``.
The empirical direction check lives in the precompute QC
(``scripts/session34/precompute_nearbody_observables.py``), which correlates
the band-integrated element against the stored C_L(t).

The solver uses a staircase immersed Neumann boundary on the raw mid-span
``inside_solid`` mask: for a face between a fluid node and a solid node the
prescribed face-normal gradient component is the corresponding component of
``e_L`` (this is the discrete form of ``n . grad(phi_L) = -n . e_L`` for both
face orientations). Outer domain boundary is Dirichlet ``phi_L = 0``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import spsolve

# Physical grid (see src/evaluation/decoder_metrics.py _PHYSICAL_X / _PHYSICAL_Y).
PHYSICAL_X = (-1.5, 4.5)
PHYSICAL_Y = (-1.5, 1.5)
GRID_H = 192  # x axis
GRID_W = 96   # y axis
DX = (PHYSICAL_X[1] - PHYSICAL_X[0]) / (GRID_H - 1)
DY = (PHYSICAL_Y[1] - PHYSICAL_Y[0]) / (GRID_W - 1)

ALPHA_DEG = 14.0
# Unit lift direction: perpendicular to the (inclined) freestream.
LIFT_DIR = (-float(np.sin(np.deg2rad(ALPHA_DEG))), float(np.cos(np.deg2rad(ALPHA_DEG))))

# Global scale applied to the raw lift-element field before log1p compression in
# the observable target. Session-34 QC over 4 train encounters: |band * e|
# p90 = 1.7, p99 = 26, p99.9 = 122, so E_SCALE = 25 puts the squared-log1p
# target in the same dynamic range as the pipeline-normalized wake targets.
# Recorded in the precompute manifest; standardization absorbs the remainder.
E_SCALE = 25.0

# The raw /curlU z-component is stored with the du/dy - dv/dx convention
# (CLAUDE.md schema notes; verified empirically in the Session-34 QC: stored
# omega is POSITIVE in the upper-surface boundary layer where standard
# omega_z = dv/dx - du/dy is negative). Multiply stored values by this sign
# to obtain standard omega_z before forming the lift element.
OMEGA_STORED_SIGN = -1.0

DEFAULT_PHI_L_PATH = Path("outputs/data_pipeline/v2p2/phi_L.npz")
DEFAULT_ADJACENT_MASK_PATH = Path("outputs/data_pipeline/v2p2/airfoil_adjacent_mask.npy")
MID_SPAN_INDEX = 16  # configs/preprocessing.yaml mid_span_index


def load_midspan_solid(baseline_h5: Path, mid_span_index: int = MID_SPAN_INDEX) -> np.ndarray:
    """Read the raw mid-span ``inside_solid`` mask, shape ``(192, 96)`` bool."""
    import h5py

    with h5py.File(baseline_h5, "r") as f:
        sol = f["inside_solid"][..., 0]  # (192, 96, 32)
    return sol[:, :, mid_span_index].astype(bool)


def solve_phi_L(
    solid: np.ndarray,
    lift_dir: tuple[float, float] = LIFT_DIR,
    dx: float = DX,
    dy: float = DY,
) -> dict:
    """Solve the Chang lift auxiliary potential on the cache grid.

    Args:
        solid: ``(H, W)`` bool mask, True inside the body (raw mid-span
            ``inside_solid``; NOT the dilated adjacent mask).
        lift_dir: unit lift direction ``e_L``.
        dx, dy: node spacings along axes 0 (x) and 1 (y).

    Returns:
        dict with ``phi (H, W) float64`` (0 inside the solid), ``grad_x``,
        ``grad_y`` (central differences, 0 inside the solid), and
        ``residual_linf`` (max |lap(phi)| over interior fluid nodes away from
        the immersed boundary, as a discretization sanity check).
    """
    solid = np.asarray(solid, dtype=bool)
    H, W = solid.shape
    elx, ely = float(lift_dir[0]), float(lift_dir[1])

    fluid = ~solid
    idx = -np.ones((H, W), dtype=np.int64)
    # Outer boundary nodes are Dirichlet phi = 0 knowns; interior fluid nodes unknown.
    interior = fluid.copy()
    interior[0, :] = interior[-1, :] = False
    interior[:, 0] = interior[:, -1] = False
    n_unknown = int(interior.sum())
    idx[interior] = np.arange(n_unknown)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    rhs = np.zeros(n_unknown)

    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)
    # Neighbor stencil: (di, dj, inv_h2, face_gradient_component, face_h).
    # BC n.grad(phi) = -n.e_L prescribes grad(phi) = -e_L along the face
    # normal at every wall face, independent of orientation, hence -elx/-ely.
    stencil = (
        (+1, 0, inv_dx2, -elx, dx),
        (-1, 0, inv_dx2, -elx, dx),
        (0, +1, inv_dy2, -ely, dy),
        (0, -1, inv_dy2, -ely, dy),
    )

    ii, jj = np.nonzero(interior)
    for i, j in zip(ii.tolist(), jj.tolist()):
        k = idx[i, j]
        diag = 0.0
        for di, dj, inv_h2, gcomp, h in stencil:
            ni, nj = i + di, j + dj
            if solid[ni, nj]:
                # Immersed Neumann face: prescribed face gradient component
                # (gcomp = -elx on x faces, -ely on y faces, both orientations).
                # Signed flux-divergence contribution +sign(d) * gcomp / h
                # moves to the RHS.
                sign = float(di + dj)  # +1 for east/north face, -1 for west/south
                rhs[k] -= sign * gcomp / h
                continue
            diag -= inv_h2
            if interior[ni, nj]:
                rows.append(k)
                cols.append(idx[ni, nj])
                vals.append(inv_h2)
            # else: outer Dirichlet neighbor phi = 0 -> contributes nothing.
        rows.append(k)
        cols.append(k)
        vals.append(diag)

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(n_unknown, n_unknown))
    phi_vec = spsolve(A, rhs)

    phi = np.zeros((H, W), dtype=np.float64)
    phi[interior] = phi_vec

    # Residual check away from the immersed boundary (pure 5-point Laplacian).
    lap = np.zeros_like(phi)
    lap[1:-1, 1:-1] = (
        (phi[2:, 1:-1] - 2 * phi[1:-1, 1:-1] + phi[:-2, 1:-1]) * inv_dx2
        + (phi[1:-1, 2:] - 2 * phi[1:-1, 1:-1] + phi[1:-1, :-2]) * inv_dy2
    )
    near_solid = ndimage.binary_dilation(solid, iterations=1)
    check = interior & ~near_solid
    check[0, :] = check[-1, :] = False
    check[:, 0] = check[:, -1] = False
    residual_linf = float(np.abs(lap[check]).max()) if check.any() else float("nan")

    grad_x = np.zeros_like(phi)
    grad_y = np.zeros_like(phi)
    grad_x[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2 * dx)
    grad_y[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2 * dy)
    grad_x[solid] = 0.0
    grad_y[solid] = 0.0

    return {
        "phi": phi,
        "grad_x": grad_x,
        "grad_y": grad_y,
        "residual_linf": residual_linf,
    }


def save_phi_L(
    result: dict,
    solid: np.ndarray,
    path: Path = DEFAULT_PHI_L_PATH,
    lift_dir: tuple[float, float] = LIFT_DIR,
) -> None:
    """Persist the phi_L artifact with provenance."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    solid_sha = hashlib.sha256(np.ascontiguousarray(solid.astype(np.uint8))).hexdigest()
    np.savez_compressed(
        path,
        phi=result["phi"].astype(np.float32),
        grad_x=result["grad_x"].astype(np.float32),
        grad_y=result["grad_y"].astype(np.float32),
        residual_linf=np.float64(result["residual_linf"]),
        lift_dir=np.asarray(lift_dir, dtype=np.float64),
        alpha_deg=np.float64(ALPHA_DEG),
        solid_sha256=np.bytes_(solid_sha.encode()),
        e_scale=np.float64(E_SCALE),
    )


def load_phi_L(path: Path = DEFAULT_PHI_L_PATH) -> dict:
    """Load the persisted phi_L artifact as a dict of float32 arrays."""
    with np.load(Path(path)) as z:
        return {
            "phi": z["phi"],
            "grad_x": z["grad_x"],
            "grad_y": z["grad_y"],
            "residual_linf": float(z["residual_linf"]),
            "lift_dir": tuple(z["lift_dir"].tolist()),
        }


def lift_element_field(
    omega_z: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    grad_x: np.ndarray,
    grad_y: np.ndarray,
) -> np.ndarray:
    """Per-frame lift-element density ``e = omega_z * (-v * gx + u * gy)``.

    All arrays broadcast over leading axes; NaNs (inside-solid fill in the raw
    fields) are replaced with 0 before the product.

    Args:
        omega_z: ``(..., H, W)`` raw mid-plane spanwise vorticity.
        u, v: ``(..., H, W)`` raw mid-plane velocity components.
        grad_x, grad_y: ``(H, W)`` phi_L gradient from :func:`solve_phi_L`.

    Returns:
        ``(..., H, W)`` float32 signed lift-element field.
    """
    omega_z = np.nan_to_num(np.asarray(omega_z, dtype=np.float32), nan=0.0)
    u = np.nan_to_num(np.asarray(u, dtype=np.float32), nan=0.0)
    v = np.nan_to_num(np.asarray(v, dtype=np.float32), nan=0.0)
    return omega_z * (-v * grad_x + u * grad_y)


def build_nearbody_band(
    adjacent_mask: np.ndarray,
    delta_n: float = 0.3,
    dx: float = DX,
    dy: float = DY,
) -> np.ndarray:
    """Sign-symmetric near-body band weight in ``[0, 1]``.

    ``w = clip(1 - dist / delta_n, 0, 1)`` where ``dist`` is the Euclidean
    distance (physical units) to the solid+adjacent set; ``w = 0`` on the
    solid+adjacent cells themselves (they carry the LE finite-difference
    artifact and are masked by the omega pipeline). The band is symmetric
    about the surface, so it follows the LEV across gust sign.

    Args:
        adjacent_mask: ``(H, W)`` bool, True on solid + 1-cell-adjacent pixels
            (``airfoil_adjacent_mask.npy``).
        delta_n: band half-width in chords.

    Returns:
        ``(H, W)`` float32 weight.
    """
    adjacent_mask = np.asarray(adjacent_mask, dtype=bool)
    dist = ndimage.distance_transform_edt(~adjacent_mask, sampling=(dx, dy))
    band = np.clip(1.0 - dist / float(delta_n), 0.0, 1.0).astype(np.float32)
    band[adjacent_mask] = 0.0
    return band
