"""Observation operator H for the Session 32 predict-correct filter (Track B).

The observation operator maps the estimable **pooled coefficient state**
``z_pool in R^d`` to the sparse wall-pressure measurement vector
``p_K in R^K`` (the ``K=8`` qDEIM wall-pressure taps, span-averaged). It is a
frozen map fit on TRAIN, with an obs-noise covariance ``R`` (diagonal) estimated
from validation residuals.

CRITICAL LEAKAGE RULE (enforced by construction here):
    H maps ``z_pool -> p_K`` ONLY. The only measurement the filter ever sees is
    ``p_K`` (K wall taps). The frozen Q1 observable probes (``z -> E_w``,
    ``z -> C_L``) are truth quantities that are NOT available at deployment; they
    are eval-time readouts scored on prior/analysis states, and they must NEVER
    enter the filter innovation. This module has no knowledge of E_w / C_L and
    cannot produce them: ``ObservationOperator.apply`` returns a ``K``-vector, and
    ``output_dim == K`` is asserted. Any code that wires an observable probe into
    the innovation is a leakage bug by definition, because the innovation is
    computed against ``H.apply(z)`` which is pressure-only.

qDEIM tap placement is target-blind and representation-independent (Drmac &
Gugercin, SIAM J. Sci. Comput. 38(2), A631, 2016). Track O owns the canonical
``outputs/session32/qdeim_taps_v2p2.json``; this module READS it if present and
otherwise computes a self-contained fallback on the TRAIN wall-pressure SVD modes
(recorded in ``provenance`` so the fallback is never mistaken for the frozen set).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
N_SURFACE = 192
DEFAULT_TAPS_PATH = "outputs/session32/qdeim_taps_v2p2.json"


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


# =========================================================================== qDEIM
def qdeim_indices(modes: np.ndarray, k: int) -> np.ndarray:
    """Q-DEIM point selection: ``k`` rows of the ``(n_points, r)`` mode matrix.

    Implements the pivoted-QR variant of DEIM (Drmac & Gugercin 2016): the
    interpolation points are the first ``k`` pivots of a column-pivoted QR of the
    transposed leading-``k`` mode block ``U_k^T`` (shape ``(k, n_points)``). This
    is the standard target-blind sensor placement on the pressure POD modes.

    Args:
        modes: ``(n_points, r)`` orthonormal spatial modes (right singular vectors
            of the snapshot matrix), most-energetic first.
        k: Number of taps to select (``k <= r`` and ``k <= n_points``).

    Returns:
        ``(k,)`` sorted int array of selected point indices in ``[0, n_points)``.
    """
    from scipy.linalg import qr

    modes = np.asarray(modes, dtype=np.float64)
    n_points, r = modes.shape
    k = int(k)
    if k > r:
        raise ValueError(f"k={k} exceeds available modes r={r}")
    if k > n_points:
        raise ValueError(f"k={k} exceeds n_points={n_points}")
    u_k = modes[:, :k]  # (n_points, k)
    # Column-pivoted QR of U_k^T; the first k pivots are the interpolation points.
    _, _, piv = qr(u_k.T, pivoting=True, mode="economic")
    taps = np.sort(np.asarray(piv[:k], dtype=int))
    return taps


def compute_qdeim_taps_from_pressure(
    p_train: np.ndarray, k: int = 8, n_modes: int | None = None
) -> np.ndarray:
    """Compute qDEIM taps from a TRAIN wall-pressure snapshot matrix (fallback).

    The snapshot matrix ``p_train`` is ``(n_snapshots, n_points)`` span-averaged
    wall pressure. The spatial modes are the right singular vectors of the
    mean-subtracted matrix; qDEIM then picks ``k`` taps from the leading ``k``
    modes. Target-blind (no latent, no observable enters), so the placement is
    representation-independent, exactly as the Track O qDEIM recompute intends.
    """
    p = np.asarray(p_train, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError(f"expected (n_snapshots, n_points) pressure; got {p.shape}")
    n_modes = int(n_modes) if n_modes is not None else int(k)
    pc = p - p.mean(axis=0, keepdims=True)
    # right singular vectors V: columns are spatial modes over the n_points axis.
    _, _, vt = np.linalg.svd(pc, full_matrices=False)
    modes = vt[:n_modes].T  # (n_points, n_modes)
    return qdeim_indices(modes, k)


DEFAULT_OSP_PATH = "outputs/session32/osp_taps_v2p2.json"


def load_osp_taps(
    model: str,
    k: int = 8,
    *,
    osp_path: str | Path = DEFAULT_OSP_PATH,
) -> tuple[np.ndarray, dict]:
    """Load a model's OSP (optimal-sensor-placement) taps from the Track O file.

    Track O (rework, HANDOFF D232) publishes MODEL-CONDITIONED taps in
    ``outputs/session32/osp_taps_v2p2.json``: ``osp[model]["K{k}"]`` is a per-model
    TCSI placement (jepa_pool / jepa_wake / pod / fukami), and
    ``osp["qdeim_shared"]["K{k}"]`` is the old target-blind array. Pass
    ``model="qdeim_shared"`` for the shared baseline. The observation operator H is
    then fit to sense at exactly these taps, so a model's filter measures where its
    OWN sensing objective says the wall is most informative.

    Returns:
        ``(taps, provenance)`` with ``taps`` a sorted ``(k,)`` int array.
    """
    path = _resolve(osp_path)
    blob = json.loads(path.read_text())
    key = f"K{int(k)}"
    if model not in blob:
        raise KeyError(f"{path} has no entry for model {model!r}; keys={list(blob)}")
    if key not in blob[model]:
        raise KeyError(f"{path}[{model!r}] has no {key}")
    taps = np.sort(np.asarray(blob[model][key], dtype=int))
    if taps.shape[0] != int(k):
        raise ValueError(f"{path}[{model!r}][{key}] has {taps.shape[0]} taps, expected {k}")
    prov = {
        "source": "qdeim_shared" if model == "qdeim_shared" else "osp_per_model",
        "path": str(path),
        "model": model,
        "method": blob[model].get("method"),
        "target": blob[model].get("target"),
        "key": key,
    }
    return taps, prov


def load_or_compute_taps(
    k: int = 8,
    *,
    taps_path: str | Path = DEFAULT_TAPS_PATH,
    p_train: np.ndarray | None = None,
    n_surface: int = N_SURFACE,
) -> tuple[np.ndarray, dict]:
    """Read Track O's frozen qDEIM taps if present, else compute a fallback.

    Track O owns ``outputs/session32/qdeim_taps_v2p2.json`` (schema
    ``{"K8": [ints], ...}``). If the file exists and has a ``"K{k}"`` key, those
    taps are used verbatim. Otherwise a self-contained fallback is computed from
    ``p_train`` (required in that case) and the ``provenance`` records that it is a
    Track-B fallback, not the frozen canonical set.

    Returns:
        ``(taps, provenance)`` where ``taps`` is a sorted ``(k,)`` int array and
        ``provenance`` documents the source.
    """
    path = _resolve(taps_path)
    key = f"K{int(k)}"
    if path.exists():
        blob = json.loads(path.read_text())
        if key in blob:
            taps = np.sort(np.asarray(blob[key], dtype=int))
            if taps.shape[0] != int(k):
                raise ValueError(f"{path} {key} has {taps.shape[0]} taps, expected {k}")
            return taps, {"source": "track_o_frozen", "path": str(path), "key": key}
    if p_train is None:
        raise FileNotFoundError(
            f"qDEIM taps file {path} missing key {key} and no p_train given for fallback"
        )
    taps = compute_qdeim_taps_from_pressure(p_train, k=int(k))
    return taps, {
        "source": "track_b_fallback_svd",
        "note": (
            "computed by Track B from TRAIN wall-pressure SVD modes; "
            "Track O canonical taps file was absent or lacked the key"
        ),
        "n_surface": int(n_surface),
        "k": int(k),
    }


# =========================================================================== operator
@dataclass
class ObservationOperator:
    """Frozen observation operator ``H: z_pool (d,) -> p_K (K,)``.

    Attributes:
        taps: ``(K,)`` selected wall-tap indices into the ``n_surface`` span-
            averaged pressure vector.
        weight: ``(K, d)`` linear map (used when ``kind == 'linear'``).
        bias: ``(K,)`` linear intercept.
        R: ``(K, K)`` diagonal obs-noise covariance from validation residuals.
        mlp: optional fitted sklearn regressor (used when ``kind == 'mlp'``).
        kind: ``'linear'`` or ``'mlp'``.
        state_dim: ``d`` (pooled state dimension).
        n_surface: number of wall taps in the full pressure vector.
        provenance: tap-source metadata from :func:`load_or_compute_taps`.
    """

    taps: np.ndarray
    weight: np.ndarray
    bias: np.ndarray
    R: np.ndarray
    kind: str = "linear"
    mlp: object | None = None
    state_dim: int = 0
    n_surface: int = N_SURFACE
    provenance: dict = field(default_factory=dict)

    @property
    def output_dim(self) -> int:
        """Measurement dimension K. This is pressure-only; never an observable."""
        return int(self.taps.shape[0])

    def select_taps(self, p_full: np.ndarray) -> np.ndarray:
        """Extract the ``K`` tap channels from a full ``(..., n_surface)`` pressure."""
        p_full = np.asarray(p_full)
        return p_full[..., self.taps]

    def apply(self, z: np.ndarray) -> np.ndarray:
        """Map pooled state to predicted wall pressure at the taps.

        Args:
            z: ``(d,)`` or ``(N, d)`` pooled state(s).

        Returns:
            ``(K,)`` or ``(N, K)`` predicted tap pressure. LEAKAGE GUARANTEE: the
            output dimension is exactly ``K`` (the pressure taps); no observable
            (E_w, C_L) can be produced by this operator.
        """
        z = np.asarray(z, dtype=np.float64)
        single = z.ndim == 1
        zz = z[None, :] if single else z
        if self.kind == "mlp":
            if self.mlp is None:
                raise ValueError("kind='mlp' but no fitted mlp is attached")
            out = np.asarray(self.mlp.predict(zz), dtype=np.float64)
        else:
            out = zz @ self.weight.T + self.bias[None, :]
        assert out.shape[1] == self.output_dim, "H must emit exactly K pressure taps"
        return out[0] if single else out

    # convenience alias
    def __call__(self, z: np.ndarray) -> np.ndarray:
        return self.apply(z)


def _fit_linear(z: np.ndarray, y: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    """Ridge least squares ``y ~ z W^T + b``. Returns ``(W (K,d), b (K,))``."""
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, d = z.shape
    zt = np.hstack([z, np.ones((n, 1))])  # (n, d+1)
    a = zt.T @ zt
    reg = ridge * np.eye(d + 1)
    reg[-1, -1] = 0.0  # do not penalise the intercept
    beta = np.linalg.solve(a + reg, zt.T @ y)  # (d+1, K)
    weight = beta[:-1].T  # (K, d)
    bias = beta[-1]  # (K,)
    return weight, bias


def fit_observation_operator(
    z_train: np.ndarray,
    p_train: np.ndarray,
    taps: np.ndarray,
    *,
    kind: str = "linear",
    z_val: np.ndarray | None = None,
    p_val: np.ndarray | None = None,
    ridge: float = 1.0,
    r_floor: float = 1e-6,
    seed: int = 0,
    provenance: dict | None = None,
) -> ObservationOperator:
    """Fit the frozen H (``z_pool -> p_K``) on TRAIN and estimate diagonal R.

    Args:
        z_train: ``(N, d)`` TRAIN pooled states.
        p_train: ``(N, n_surface)`` TRAIN span-averaged wall pressure.
        taps: ``(K,)`` selected tap indices.
        kind: ``'linear'`` (default) or ``'mlp'`` (small MLPRegressor variant).
        z_val, p_val: optional validation pair for the R estimate. When omitted, R
            is computed from TRAIN residuals (recorded in ``provenance``).
        ridge: ridge penalty for the linear fit.
        r_floor: minimum per-tap obs-noise variance (numerical floor).

    Returns:
        A fitted :class:`ObservationOperator`. R is the diagonal covariance of the
        per-tap residuals ``p_K - H(z)`` on the validation (or train) set.
    """
    z_train = np.asarray(z_train, dtype=np.float64)
    taps = np.sort(np.asarray(taps, dtype=int))
    pk_train = np.asarray(p_train, dtype=np.float64)[:, taps]
    d = int(z_train.shape[1])
    k = int(taps.shape[0])

    mlp = None
    if kind == "linear":
        weight, bias = _fit_linear(z_train, pk_train, ridge)

        def _predict(zz: np.ndarray) -> np.ndarray:
            return zz @ weight.T + bias[None, :]

    elif kind == "mlp":
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        mlp = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64, 64),
                activation="relu",
                alpha=1e-3,
                max_iter=800,
                early_stopping=True,
                n_iter_no_change=20,
                random_state=seed,
            ),
        )
        mlp.fit(z_train, pk_train)
        weight = np.zeros((k, d))
        bias = np.zeros(k)

        def _predict(zz: np.ndarray) -> np.ndarray:
            return np.asarray(mlp.predict(zz), dtype=np.float64)

    else:
        raise ValueError(f"kind must be 'linear' or 'mlp'; got {kind!r}")

    if z_val is not None and p_val is not None:
        z_r = np.asarray(z_val, dtype=np.float64)
        pk_r = np.asarray(p_val, dtype=np.float64)[:, taps]
        r_source = "validation_residuals"
    else:
        z_r, pk_r = z_train, pk_train
        r_source = "train_residuals"

    resid = pk_r - _predict(z_r)  # (M, K)
    r_diag = np.maximum(resid.var(axis=0), r_floor)
    R = np.diag(r_diag)

    prov = dict(provenance or {})
    prov.update({"R_source": r_source, "ridge": ridge, "kind": kind})
    return ObservationOperator(
        taps=taps,
        weight=np.asarray(weight, dtype=np.float64),
        bias=np.asarray(bias, dtype=np.float64),
        R=R,
        kind=kind,
        mlp=mlp,
        state_dim=d,
        n_surface=int(p_train.shape[1]),
        provenance=prov,
    )


def h_matrix(op: ObservationOperator) -> np.ndarray:
    """Return the linear H matrix ``(K, d)`` (only meaningful for ``kind='linear'``)."""
    if op.kind != "linear":
        raise ValueError("h_matrix is only defined for a linear observation operator")
    return op.weight


__all__ = [
    "N_SURFACE",
    "ObservationOperator",
    "qdeim_indices",
    "compute_qdeim_taps_from_pressure",
    "load_or_compute_taps",
    "load_osp_taps",
    "fit_observation_operator",
    "h_matrix",
]
