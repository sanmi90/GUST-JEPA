"""Track O -- qDEIM sensor placement on the v2.2 training wall-pressure snapshots.

Target-blind, representation-independent sensor placement: POD (SVD) of the
v2.2 TRAINING span-averaged wall-pressure snapshot matrix, then qDEIM (QR with
column pivoting on the leading right singular vectors; Manohar, Brunton, Kutz,
Brunton 2018) to pick the interpolation taps. There is no "pooled pressure
mode" -- this is purely the SVD of the training ``p_wall`` matrix.

Output: ``outputs/session32/qdeim_taps_v2p2.json`` with the K in {2,4,8,16} tap
indices (physical taps 0..191). CONSUMED BY TRACK B -- the schema is stable:

    {"K8": [...ints...], "K4": [...], "K2": [...], "K16": [...],
     "n_taps": 192, "method": "qdeim", "source": "v2p2 train p_wall SVD", ...}

The K sets are NESTED (K2 subset K4 subset K8 subset K16), computed from a
single QR pivot on the leading r=16 POD modes and stored in qDEIM pivot
importance order (most informative tap first). This mirrors the Session 21
convention (``qdeim16[:K]``) so the picks are comparable across K.

Reuses the pre-aligned training pressure matrix at
``outputs/session31/q1_latents/pressure_train.npz`` (all train frames x 192
span-averaged taps), which is byte-identical to reading ``p_wall`` from the
per-encounter v2.2 cache.

Run (CPU only, no GPU):
    taskset -c 16-23 python -m scripts.session32.qdeim_recompute
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
K_LIST = (2, 4, 8, 16)
N_TAPS = 192


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def qr_pivot_order(modes: np.ndarray) -> np.ndarray:
    """qDEIM pivots (sensor indices) in decreasing importance for a mode basis.

    Args:
        modes: ``(n_sensors, r)`` leading POD modes over the sensor axis (the
            right singular vectors of the snapshot matrix).

    Returns:
        ``(n_sensors,)`` int array: the QR-with-column-pivoting permutation of
        the sensor axis. The first ``r`` entries are the qDEIM interpolation
        points; ``perm[:K]`` gives the nested K-sensor pick.
    """
    from scipy.linalg import qr

    # qDEIM pivots the transpose of the mode matrix (r x n_sensors); the pivot
    # order of the columns (= sensors) is the sensor ranking.
    _, _, perm = qr(modes.T, pivoting=True, mode="economic")
    return np.asarray(perm, dtype=np.int64)


def compute_qdeim_taps(
    snapshots: np.ndarray,
    ks: tuple[int, ...] = K_LIST,
    *,
    center: bool = True,
    r: int | None = None,
) -> dict:
    """POD (SVD) of the snapshot matrix then qDEIM pivoting to K nested taps.

    Args:
        snapshots: ``(n_snapshots, n_sensors)`` training wall-pressure matrix.
        ks: sensor counts to emit (nested).
        center: subtract the per-sensor mean over snapshots (POD of the
            fluctuation field; the standard sensor-placement convention).
        r: number of leading modes for the single QR pivot (default max(ks)).

    Returns:
        dict with ``K{k}`` lists (pivot-importance order), plus provenance.
    """
    A = np.asarray(snapshots, dtype=np.float64)
    n_snap, n_sens = A.shape
    mean = A.mean(axis=0)
    Ac = A - mean if center else A
    r_eff = int(min(max(ks) if r is None else r, n_snap, n_sens))
    # Right singular vectors span the sensor axis: Vt (r, n_sensors).
    _, svals, Vt = np.linalg.svd(Ac, full_matrices=False)
    modes = Vt[:r_eff, :].T  # (n_sensors, r_eff)
    perm = qr_pivot_order(modes)  # (n_sensors,) importance-ordered sensor ids
    out: dict = {}
    for k in ks:
        out[f"K{int(k)}"] = [int(x) for x in perm[: int(k)]]
    # variance captured by the r leading modes (diagnostic)
    tot = float((svals**2).sum())
    out["_diag"] = {
        "n_snapshots": int(n_snap),
        "n_modes_r": r_eff,
        "singular_values_top": [float(s) for s in svals[:r_eff]],
        "energy_frac_top_r": float((svals[:r_eff] ** 2).sum() / max(tot, 1e-30)),
        "centered": bool(center),
        "pivot_order_top16": [int(x) for x in perm[:16]],
    }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Track O qDEIM sensor recompute on v2.2 train p_wall")
    ap.add_argument(
        "--pressure",
        default="outputs/session31/q1_latents/pressure_train.npz",
        help="Aligned training p_wall npz (all train frames x 192 taps).",
    )
    ap.add_argument("--out", default="outputs/session32/qdeim_taps_v2p2.json")
    ap.add_argument("--no-center", action="store_true", help="Skip POD mean centering.")
    ap.add_argument(
        "--seed", type=int, default=0, help="Logged for provenance (SVD is deterministic)."
    )
    args = ap.parse_args(argv)

    blob = np.load(_resolve(args.pressure), allow_pickle=True)
    snap = np.asarray(blob["p_wall"], dtype=np.float64)
    assert snap.ndim == 2 and snap.shape[1] == N_TAPS, f"unexpected p_wall shape {snap.shape}"
    assert np.isfinite(snap).all(), "non-finite training pressure snapshots"

    res = compute_qdeim_taps(snap, K_LIST, center=not args.no_center)

    # Cross-check against the repo's uncentered selector_qdeim (set overlap).
    overlap = {}
    try:
        import sys

        sys.path.insert(0, str(_resolve("scripts")))
        from session14_tcsi_pilot import selector_qdeim  # type: ignore

        for k in K_LIST:
            ref = set(selector_qdeim(snap, int(k)))
            here = set(res[f"K{k}"])
            overlap[f"K{k}"] = {
                "n_shared": len(ref & here),
                "repo_uncentered_sorted": sorted(ref),
            }
    except Exception as exc:  # pragma: no cover
        overlap["error"] = repr(exc)

    payload = {
        "K2": res["K2"],
        "K4": res["K4"],
        "K8": res["K8"],
        "K16": res["K16"],
        "n_taps": N_TAPS,
        "method": "qdeim",
        "source": "v2p2 train p_wall SVD",
        "schema": {
            "K{k}": "physical span-averaged wall-tap indices 0..191, qDEIM pivot-"
            "importance order (most informative first); nested K2<K4<K8<K16",
            "n_taps": "number of physical wall taps (span-averaged)",
            "method": "POD/SVD of training p_wall then QR-column-pivot (qDEIM)",
            "source": "training wall-pressure snapshots, split_v2p2",
        },
        "provenance": {
            "pressure_snapshots": str(args.pressure),
            "n_snapshots": res["_diag"]["n_snapshots"],
            "centered": res["_diag"]["centered"],
            "n_modes_r": res["_diag"]["n_modes_r"],
            "energy_frac_top_r": res["_diag"]["energy_frac_top_r"],
            "singular_values_top": res["_diag"]["singular_values_top"],
            "seed": int(args.seed),
            "repo_selector_qdeim_overlap": overlap,
        },
    }
    out_path = _resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[qdeim] wrote {out_path}")
    for k in K_LIST:
        print(f"[qdeim] K{k}: {payload[f'K{k}']}")
    print(
        f"[qdeim] repo-selector overlap: "
        f"{ {k: overlap[k]['n_shared'] for k in overlap if k.startswith('K')} }"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
