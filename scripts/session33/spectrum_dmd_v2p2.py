"""Session 33 latent DMD spectrum on the pooled v2.2 latents (F9 inset).

Re-run of Part 1 (exact DMD on latent trajectories) of
scripts/session28/spectrum_dmd.py for the v2.2 manuscript, per
SESSION_33_MANUSCRIPT_V3.md Section 11 item 3 ("The DMD/Strouhal spectrum per
family, Section 4.6, F9 inset"). The Floquet / native-predictor analysis
(session28 Part 2) is intentionally NOT ported: the paper only needs the DMD
spectrum inset answering "does each family's latent recover the shedding
Strouhal number on a marginally stable orbit?".

WHAT: for each latent family (jepa_pool, supervised_only_pool, regAE_pool,
fukami, fukami_wake, pod; all pooled d = 32 per-frame latents, key z_gap), fit
the best-fit linear operator A (z_{t+1} ~ A z_t) by exact DMD (A = Z' Z^+,
eigendecompose A) on (a) the Baseline no-gust limit-cycle encounters and (b)
the pooled train trajectories (the session28 convention; session28 did NOT use
pre-impact segments of periodic gust cases, and neither does this script).
Discrete eigenvalue lambda -> rate = log|lambda| / dt_tc and
St = |angle(lambda)| / (2 pi dt_tc) with dt_tc = 0.05.

WHY: the v2.1 manuscript found the predictive latent recovers the shedding
line (St ~ 0.66 vs the measured 0.68) on a marginally stable orbit
(|lambda| ~ 1) while the reconstructive latent is damped and off-frequency
(St ~ 0.50). This script recomputes that comparison on the v2.2 pooled
latents. Ground-truth reference lines are kept at the v2.1 values,
St_dominant = 0.675 and St_subharmonic = 0.338, and are stored in the JSON so
the inset can draw them.

Rows in each latents NPZ are per (encounter, frame); trajectories are
reassembled by grouping on (case_id, encounter_index) and sorting by frame.
CPU-only (numpy); no GPU is touched.

Output: outputs/session33/spectrum_dmd_v2p2.json with a params/provenance
block (inputs, dt_tc, model -> cache map, ground-truth constants) and
per-model results (dominant St, |lambda|, subharmonic match if present, full
spectrum tables of (Re, Im, |lambda|, rate, St) for the F9 inset).

Usage:
    export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
    taskset -c 16-23 nice -n 10 .venv/bin/python \\
        scripts/session33/spectrum_dmd_v2p2.py
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

# DNS shedding reference (v2.1 reference values; see module docstring).
DT_TC = 0.05
ST_DOMINANT = 0.675  # dominant lift line (period ~30 frames)
ST_SUBHARMONIC = 0.338  # subharmonic / full-cycle clock (period ~59 frames)

# Oscillatory floor (exclude near-DC real modes) and subharmonic match window,
# both in St units. Floor as in session28. The subharmonic window is tight so
# "present" means genuinely near the 0.338 line: with a loose window (0.15) the
# heavily damped fukami_wake spectrum matched a spurious St = 0.475 mode at
# |lambda| = 0.06, which is not a subharmonic in any physical sense.
ST_OSC_FLOOR = 0.02
ST_SUBHARMONIC_TOL = 0.05

# Model tag -> per-frame pooled-latent cache (train split), repo-relative.
MODEL_CACHES = {
    # D250 flagship: the predictive family is the native-vector jepa_pool_vec.
    "jepa_pool": "outputs/session33/q1_vec_latents/latents_jepa_pool_vec_train.npz",
    "supervised_only_pool": (
        "outputs/session32/q1_pool_latents/latents_supervised_only_pool_train.npz"
    ),
    "regAE_pool": "outputs/session32/q1_pool_latents/latents_regAE_pool_train.npz",
    "fukami": "outputs/session31/q1_latents/latents_fukami_train.npz",
    "fukami_wake": "outputs/session31/q1_latents/latents_fukami_wake_train.npz",
    "pod": "outputs/session31/q1_latents/latents_pod_train.npz",
}

BASELINE_CASE = "Baseline"


def load_trajectories(path: Path) -> dict[tuple[str, int], np.ndarray]:
    """Reassemble per-encounter latent trajectories from a per-frame NPZ.

    Rows of the session31/32 latent caches are per (encounter, frame). We group
    on (case_id, encounter_index), sort by frame, and require contiguous frame
    indices so DMD snapshot pairs never straddle a gap.

    Args:
        path: latents NPZ with keys z_gap (N, d), case_id, encounter_index,
            frame.

    Returns:
        {(case_id, encounter_index): (T, d) float64 trajectory}.
    """
    blob = np.load(path, allow_pickle=True)
    z = blob["z_gap"].astype(np.float64)
    cids = np.array([str(c) for c in blob["case_id"]])
    encs = blob["encounter_index"].astype(np.int64)
    frames = blob["frame"].astype(np.int64)
    trajs: dict[tuple[str, int], np.ndarray] = {}
    for cid in np.unique(cids):
        cmask = cids == cid
        for enc in np.unique(encs[cmask]):
            sel = np.where(cmask & (encs == enc))[0]
            order = np.argsort(frames[sel])
            f = frames[sel][order]
            if not np.all(np.diff(f) == 1):
                raise ValueError(f"non-contiguous frames for ({cid}, {enc}) in {path}")
            trajs[(cid, int(enc))] = z[sel[order]]
    return trajs


def exact_dmd(snapshots: list[np.ndarray]) -> dict:
    """Exact DMD: fit z_{t+1} ~ A z_t by least squares over snapshot pairs.

    A = Z' Z^+ where Z = [z_0, ..., z_{m-1}], Z' = [z_1, ..., z_m] are columns
    stacked over every contiguous trajectory (pairs never cross a trajectory
    boundary). d = 32 is small, so the full A is eigendecomposed directly with
    no rank truncation (same choice as session28).

    Args:
        snapshots: list of (T_i, d) contiguous latent trajectories.

    Returns:
        {"eigvals" (complex, d), "n_pairs", "d",
         "fit_residual" = ||AZ - Z'||_F / ||Z'||_F}.
    """
    x_cols = []
    y_cols = []
    for traj in snapshots:
        traj = np.asarray(traj, dtype=np.float64)
        if traj.shape[0] < 2:
            continue
        x_cols.append(traj[:-1].T)  # (d, T_i - 1)
        y_cols.append(traj[1:].T)
    if not x_cols:
        raise ValueError("no usable snapshot pairs")
    x = np.concatenate(x_cols, axis=1)  # (d, m)
    y = np.concatenate(y_cols, axis=1)  # (d, m)
    a_t, *_ = np.linalg.lstsq(x.T, y.T, rcond=None)
    a = a_t.T  # (d, d)
    eigvals = np.linalg.eigvals(a)
    resid = float(np.linalg.norm(a @ x - y) / max(np.linalg.norm(y), 1e-12))
    return {"eigvals": eigvals, "n_pairs": int(x.shape[1]), "d": int(x.shape[0]),
            "fit_residual": resid}


def lambda_to_physical(lam: complex, dt_tc: float = DT_TC) -> tuple[float, float, float]:
    """Discrete eigenvalue lambda -> (modulus, growth rate, Strouhal number).

    rate = log|lambda| / dt_tc (per t/c; < 0 decaying, ~0 marginally stable),
    St = |angle(lambda)| / (2 pi dt_tc) (0 for a purely real positive mode).
    """
    modulus = float(abs(lam))
    rate = float(np.log(max(modulus, 1e-300)) / dt_tc)
    st = float(abs(np.angle(lam)) / (2.0 * np.pi * dt_tc))
    return modulus, rate, st


def spectrum_table(eigvals: np.ndarray, dt_tc: float = DT_TC) -> list[dict]:
    """Full eigenvalue table sorted by modulus (descending), for the F9 inset."""
    rows = []
    for lam in eigvals:
        modulus, rate, st = lambda_to_physical(lam, dt_tc)
        rows.append({"re": float(lam.real), "im": float(lam.imag),
                     "modulus": modulus, "rate": rate, "St": st})
    return sorted(rows, key=lambda c: c["modulus"], reverse=True)


def pick_dominant_and_subharmonic(spec: list[dict]) -> tuple[dict, dict | None, int]:
    """Dominant nontrivial pair + the subharmonic match if present.

    Dominant = largest-modulus eigenvalue with St > ST_OSC_FLOOR (the session28
    "leading oscillatory pair" rule; excludes near-DC real modes). Subharmonic
    = largest-modulus oscillatory eigenvalue with |St - 0.338| <=
    ST_SUBHARMONIC_TOL at a frequency distinct from the dominant pair
    (conjugate pairs share the same St, so a same-St test excludes both the
    dominant mode and its conjugate); None if absent.

    Args:
        spec: modulus-sorted table from spectrum_table().

    Returns:
        (dominant row, subharmonic row or None, number of oscillatory modes).
    """
    osc = [c for c in spec if c["St"] > ST_OSC_FLOOR]
    pool = osc if osc else spec
    dominant = pool[0]  # spec is modulus-sorted
    sub = None
    for c in osc:
        if abs(c["St"] - dominant["St"]) < 1e-9:
            continue  # the dominant eigenvalue or its complex conjugate
        if abs(c["St"] - ST_SUBHARMONIC) <= ST_SUBHARMONIC_TOL:
            sub = c
            break
    return dominant, sub, len(osc)


def run_family(name: str, path: Path) -> dict:
    """Exact DMD for one latent family on Baseline + pooled-train trajectories."""
    trajs = load_trajectories(path)
    base = [t for (cid, _), t in sorted(trajs.items()) if cid == BASELINE_CASE]
    if not base:
        raise ValueError(f"{name}: no {BASELINE_CASE} trajectories in {path}")
    pooled = [t for _, t in sorted(trajs.items())]
    rec: dict = {"cache": str(path.relative_to(REPO)), "d": int(base[0].shape[1])}
    for regime, snap in (("baseline", base), ("pooled_train", pooled)):
        dmd = exact_dmd(snap)
        spec = spectrum_table(dmd["eigvals"])
        dominant, sub, n_osc = pick_dominant_and_subharmonic(spec)
        rec[regime] = {
            "n_trajectories": len(snap),
            "n_pairs": dmd["n_pairs"],
            "fit_residual": dmd["fit_residual"],
            "dominant": {**dominant, "delta_St_vs_dominant_ref": dominant["St"] - ST_DOMINANT},
            "subharmonic": (
                None if sub is None
                else {**sub, "delta_St_vs_subharmonic_ref": sub["St"] - ST_SUBHARMONIC}
            ),
            "n_oscillatory": n_osc,
            "spectrum": spec,
        }
    return rec


def best_st_recovery(models: dict, regime: str) -> dict:
    """Which family's dominant St is closest to a DNS reference line."""
    best: dict = {}
    for name, rec in models.items():
        if regime not in rec:
            continue
        st = rec[regime]["dominant"]["St"]
        err_dom = abs(st - ST_DOMINANT)
        err_sub = abs(st - ST_SUBHARMONIC)
        err = min(err_dom, err_sub)
        line = "dominant" if err_dom <= err_sub else "subharmonic"
        if not best or err < best["abs_err"]:
            best = {"family": name, "dominant_St": st, "abs_err": err, "matched_line": line}
    return best


def main(argv: list[str] | None = None) -> None:
    """Run exact DMD per family and write the F9-inset spectrum JSON."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=REPO / "outputs/session33/spectrum_dmd_v2p2.json",
    )
    args = p.parse_args(argv)

    t0 = time.time()
    models: dict = {}
    print("[dmd] exact DMD on pooled v2.2 train latents (z_gap, d=32), dt_tc =", DT_TC)
    for name, rel in MODEL_CACHES.items():
        path = REPO / rel
        if not path.exists():
            print(f"[dmd] SKIP {name}: {path} missing")
            continue
        rec = run_family(name, path)
        models[name] = rec
        b = rec["baseline"]["dominant"]
        print(
            f"[dmd] {name:22s} baseline: St={b['St']:.3f} |lambda|={b['modulus']:.4f} "
            f"dSt={b['St'] - ST_DOMINANT:+.3f} vs {ST_DOMINANT} "
            f"(n_traj={rec['baseline']['n_trajectories']}, "
            f"resid={rec['baseline']['fit_residual']:.3e})"
        )

    out = {
        "generated_iso": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/session33/spectrum_dmd_v2p2.py",
        "params": {
            "split": "train",
            "partition": "v2p2",
            "latent_key": "z_gap",
            "dt_tc": DT_TC,
            "baseline_case": BASELINE_CASE,
            "oscillatory_st_floor": ST_OSC_FLOOR,
            "subharmonic_st_tol": ST_SUBHARMONIC_TOL,
            "model_caches": MODEL_CACHES,
            "method": (
                "exact DMD, A = Z' Z^+ by lstsq over per-encounter snapshot pairs; "
                "full d=32 eigendecomposition, no rank truncation (session28 Part 1)"
            ),
        },
        "ground_truth": {
            "St_dominant": ST_DOMINANT,
            "St_subharmonic": ST_SUBHARMONIC,
            "note": "v2.1 reference values (outputs/session28/undisturbed_stats.json)",
        },
        "models": models,
        "best_family_baseline": best_st_recovery(models, "baseline"),
        "best_family_pooled_train": best_st_recovery(models, "pooled_train"),
        "wall_time_s": time.time() - t0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[dmd] wrote {args.out} ({time.time() - t0:.1f}s)")

    bf = out["best_family_baseline"]
    if bf:
        print(
            f"[dmd] best baseline St recovery: {bf['family']} "
            f"(St={bf['dominant_St']:.3f}, matched {bf['matched_line']} line, "
            f"|err|={bf['abs_err']:.3f})"
        )


if __name__ == "__main__":
    main()
