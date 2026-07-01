"""Impact-window definition for SESSION 31 Track 0.B.

Every temporal (Q2 forecast, Q3 pressure) metric in SESSION 31 is restricted to
physically motivated windows around the gust strike: a lead-in, the impact, and
the relaxation tail. This module defines those windows from the held-out lift
signal and persists them per ``(case_id, encounter)``.

Impact instant
    ``t_impact = argmax_t |dC_L/dt|`` -- the steepest lift transient, the gust
    strike. ``dC_L/dt`` is the central-difference gradient of the per-encounter
    lift trace (``numpy.gradient``).

Windows (half-open ``[start, end)``, clamped to ``[0, n_frames]``), following the
plan's definition where ``W_relax`` is measured from ``t_impact``:
    lead_in     = [t_impact - W_in,  t_impact)
    impact      = [t_impact,         t_impact + W_imp)
    relaxation  = [t_impact + W_imp, t_impact + W_relax)

Acceptance gate 0 (the data-integrity gate) needs two scalar diagnostics per
encounter:
    peak_clarity      -- how dominant the single |dC_L/dt| peak is (unimodality)
    is_well_separated -- the full window fits inside the trajectory

The pure functions below carry the math; the ``main`` CLI applies them across a
partition's cache and writes ``windows_v2p2.json``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

# Plan defaults (Track 0.B): one set of window widths for the whole study.
W_IN_DEFAULT = 8
W_IMP_DEFAULT = 16
W_RELAX_DEFAULT = 48
_CLARITY_SEP = 5  # frames masked either side of the primary peak for peak_clarity
_EPS = 1e-9


def abs_dcl_dt(cl: np.ndarray) -> np.ndarray:
    """``|dC_L/dt|`` as the magnitude of the central-difference gradient."""
    cl = np.asarray(cl, dtype=np.float64)
    if cl.ndim != 1:
        raise ValueError(f"cl must be 1-D, got shape {cl.shape}")
    return np.abs(np.gradient(cl))


def impact_frame(cl: np.ndarray) -> int:
    """Frame index of the steepest lift transient, ``argmax_t |dC_L/dt|``."""
    return int(np.argmax(abs_dcl_dt(cl)))


def impact_frame_anchored(cl: np.ndarray, lo: int, hi: int) -> int:
    """``argmax_t |dC_L/dt|`` restricted to the physics window ``[lo, hi]``.

    D-N1 trigger ``anchored_local``: the impact is the lift-response peak, but the
    search is bounded to the kinematically determined impact window so vortex
    shedding elsewhere cannot hijack it. ``lo``/``hi`` are inclusive frame indices.
    """
    g = abs_dcl_dt(cl)
    lo = max(0, int(lo))
    hi = min(g.shape[0] - 1, int(hi))
    if hi < lo:
        raise ValueError(f"empty search window [{lo}, {hi}]")
    hi_excl = hi + 1
    return lo + int(np.argmax(g[lo:hi_excl]))


def peak_clarity(cl: np.ndarray, sep: int = _CLARITY_SEP) -> float:
    """Dominance of the primary ``|dC_L/dt|`` peak over the next one.

    Ratio of the global ``|dC_L/dt|`` maximum to the largest value at least
    ``sep`` frames away from it. A clean single strike gives a large ratio; two
    comparable peaks give a ratio near one. Returned as a finite float (a lone
    peak yields ``primary / eps``, a large but JSON-serialisable number).
    """
    g = abs_dcl_dt(cl)
    imp = int(np.argmax(g))
    primary = float(g[imp])
    lo = max(0, imp - sep)
    hi = min(g.shape[0], imp + sep + 1)
    masked = g.copy()
    masked[lo:hi] = 0.0
    second = float(masked.max()) if masked.size else 0.0
    return primary / (second + _EPS)


def build_windows(
    t_impact: int,
    n_frames: int,
    w_in: int = W_IN_DEFAULT,
    w_imp: int = W_IMP_DEFAULT,
    w_relax: int = W_RELAX_DEFAULT,
) -> dict:
    """Half-open, clamped window bounds keyed by name plus ``t_impact``."""

    def clamp(x: int) -> int:
        return int(max(0, min(int(x), n_frames)))

    lead_in = (clamp(t_impact - w_in), clamp(t_impact))
    impact = (clamp(t_impact), clamp(t_impact + w_imp))
    relaxation = (clamp(t_impact + w_imp), clamp(t_impact + w_relax))
    return {
        "t_impact": int(t_impact),
        "n_frames": int(n_frames),
        "lead_in": lead_in,
        "impact": impact,
        "relaxation": relaxation,
    }


def is_well_separated(
    t_impact: int,
    n_frames: int,
    w_in: int = W_IN_DEFAULT,
    w_relax: int = W_RELAX_DEFAULT,
) -> bool:
    """True when the full ``[t_impact - W_in, t_impact + W_relax]`` span fits.

    Uses the last valid frame index ``n_frames - 1`` as the upper bound, matching
    the plan's "impact within W_in of a trajectory boundary" WEAK condition.
    """
    return bool((t_impact - w_in >= 0) and (t_impact + w_relax <= n_frames - 1))


def window_masks(windows: dict, n_frames: int) -> dict:
    """Expand the bounds in ``windows`` into per-frame boolean masks."""
    masks = {}
    for name in ("lead_in", "impact", "relaxation"):
        start, end = windows[name]
        m = np.zeros(n_frames, dtype=bool)
        m[start:end] = True
        masks[name] = m
    return masks


# --------------------------------------------------------------------------- #
# CLI: apply the pure functions across a partition cache.
# --------------------------------------------------------------------------- #
def _resolve_cache_dir(partition: str) -> Path:
    cache_root = os.environ.get("VORTEX_JEPA_CACHE")
    if cache_root is None:
        prevent = Path(os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT")))
        cache_root = prevent / "data" / "processed" / "vortex-jepa"
    return Path(cache_root) / partition


def _iter_split_encounters(split_manifest: dict):
    """Yield ``(case_id, encounter_index, split)`` for every cached encounter."""
    for case_id, case in split_manifest["cases"].items():
        for k in range(int(case["n_encounters_full"])):
            yield case_id, k, case["split"]


def compute_partition_windows(
    split_manifest: dict,
    cache_dir: Path,
    signal: str = "C_L",
    trigger: str = "anchored_local",
    search_lo: int = 25,
    search_hi: int = 55,
    w_in: int = W_IN_DEFAULT,
    w_imp: int = W_IMP_DEFAULT,
    w_relax: int = W_RELAX_DEFAULT,
) -> dict:
    """Compute windows for every ``(case_id, encounter)`` in the manifest.

    ``trigger`` selects the D-N1 impact definition:
        ``anchored_local`` -- argmax|dC_L/dt| within [search_lo, search_hi]
        ``kinematic``      -- the constant ``impact_frame_estimate`` attr
        ``naive``          -- global argmax|dC_L/dt| (the rejected default)
    """
    import h5py

    if trigger not in ("anchored_local", "kinematic", "naive"):
        raise ValueError(f"unknown trigger {trigger!r}")

    entries: dict[str, dict] = {}
    for case_id, k, split in _iter_split_encounters(split_manifest):
        enc_path = cache_dir / case_id / f"encounter_{k:02d}.h5"
        with h5py.File(enc_path, "r") as g:
            cl = g[signal][:].astype(np.float64)
            est = (
                int(g.attrs["impact_frame_estimate"])
                if "impact_frame_estimate" in g.attrs
                else None
            )
        n_frames = int(cl.shape[0])
        if trigger == "kinematic":
            if est is None:
                raise KeyError(f"{case_id}/{k:02d}: missing impact_frame_estimate attr")
            t_imp = est
        elif trigger == "anchored_local":
            t_imp = impact_frame_anchored(cl, search_lo, search_hi)
        else:
            t_imp = impact_frame(cl)
        w = build_windows(t_imp, n_frames, w_in, w_imp, w_relax)
        w["split"] = split
        w["trigger"] = trigger
        w["peak_clarity"] = peak_clarity(cl)
        w["is_well_separated"] = is_well_separated(t_imp, n_frames, w_in, w_relax)
        if est is not None:
            w["kinematic_estimate"] = est
        entries[f"{case_id}/{k:02d}"] = w
    return entries


def main() -> None:
    ap = argparse.ArgumentParser(description="SESSION 31 Track 0.B impact-window definition")
    ap.add_argument("--partition", default="v2p2")
    ap.add_argument("--split-manifest", default="configs/splits/split_v2p2.json")
    ap.add_argument("--signal", default="C_L")
    ap.add_argument("--metric", default="abs_dCl_dt", choices=["abs_dCl_dt"])
    ap.add_argument(
        "--trigger",
        default="anchored_local",
        choices=["anchored_local", "kinematic", "naive"],
        help="D-N1 impact-frame definition",
    )
    ap.add_argument(
        "--search-lo", type=int, default=25, help="anchored_local search window lower frame"
    )
    ap.add_argument(
        "--search-hi", type=int, default=55, help="anchored_local search window upper frame"
    )
    ap.add_argument("--w_in", type=int, default=W_IN_DEFAULT)
    ap.add_argument("--w_imp", type=int, default=W_IMP_DEFAULT)
    ap.add_argument("--w_relax", type=int, default=W_RELAX_DEFAULT)
    ap.add_argument(
        "--clarity-threshold",
        type=float,
        default=1.5,
        help="peak_clarity below this counts as ambiguous (diagnostic only "
        "for the anchored triggers; the gate is well-separation)",
    )
    ap.add_argument("--out", default="outputs/session31/windows_v2p2.json")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[2]
    split_path = Path(args.split_manifest)
    if not split_path.is_absolute():
        split_path = repo / split_path
    with open(split_path) as f:
        split_manifest = json.load(f)
    cache_dir = _resolve_cache_dir(args.partition)

    entries = compute_partition_windows(
        split_manifest,
        cache_dir,
        args.signal,
        args.trigger,
        args.search_lo,
        args.search_hi,
        args.w_in,
        args.w_imp,
        args.w_relax,
    )

    n = len(entries)
    # The acceptance gate is well-separation (the window fits the trajectory).
    # peak_clarity is reported as a diagnostic; for the anchored triggers it is
    # not a gate (the kinematic anchoring, not unimodality, places the window).
    n_well_sep = sum(1 for e in entries.values() if e["is_well_separated"])
    not_sep = [k for k, e in entries.items() if not e["is_well_separated"]]
    n_unimodal = sum(1 for e in entries.values() if e["peak_clarity"] >= args.clarity_threshold)
    t_impacts = sorted(e["t_impact"] for e in entries.values())

    payload = {
        "schema": {
            "key": "case_id/encounter (zero-padded)",
            "windows": "half-open [start, end), clamped to [0, n_frames); "
            "boolean per-frame masks via window_masks()",
            "metric": args.metric,
            "gate": "is_well_separated (window fits trajectory); peak_clarity is diagnostic",
        },
        "params": {
            "trigger": args.trigger,
            "search_lo": args.search_lo,
            "search_hi": args.search_hi,
            "w_in": args.w_in,
            "w_imp": args.w_imp,
            "w_relax": args.w_relax,
            "signal": args.signal,
            "clarity_threshold": args.clarity_threshold,
        },
        "partition": args.partition,
        "summary": {
            "n_encounters": n,
            "n_well_separated": n_well_sep,
            "frac_well_separated": n_well_sep / n if n else 0.0,
            "not_well_separated_keys": not_sep,
            "n_unimodal_diag": n_unimodal,
            "frac_unimodal_diag": n_unimodal / n if n else 0.0,
            "t_impact_min": t_impacts[0] if t_impacts else None,
            "t_impact_median": int(np.median(t_impacts)) if t_impacts else None,
            "t_impact_max": t_impacts[-1] if t_impacts else None,
        },
        "windows": entries,
    }

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = repo / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(
        f"[trigger={args.trigger}] {n} encounters, "
        f"{n_well_sep} well-separated ({100*n_well_sep/max(n,1):.1f}%), "
        f"{n_unimodal} unimodal-diag; t_impact median "
        f"{payload['summary']['t_impact_median']} "
        f"[{payload['summary']['t_impact_min']}, {payload['summary']['t_impact_max']}]"
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
