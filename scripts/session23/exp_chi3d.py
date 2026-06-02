"""Session 23 -- the three-dimensional observability boundary at |G|=4.

The GUST-JEPA encoder reads only the mid-plane omega_z slice. The source physics
(Fukami, Smith & Taira 2025) says that at |G| >= 4 the gust-airfoil interaction
becomes genuinely three-dimensional, so a mid-plane-only encoder loses
information and the out-of-distribution test_c (|G|=4) degrades. This script
measures that 3D content DIRECTLY from the raw 3D vorticity field /curlU, to
test whether the spanwise-fluctuating enstrophy fraction jumps at |G|=4.

Definition (project revision plan Part V.2). Per frame t, with
  omega(x, y, z, t)  the 3-component vorticity at spanwise station z (32 stations),
  bar_omega(x, y, t) = mean_z omega   the spanwise (z) average,
the spanwise-fluctuating enstrophy fraction is

  chi_3D(t) = sum_xyz || omega - bar_omega ||^2  /  sum_xyz || omega ||^2 .

chi_3D = 0 means a perfectly 2D (spanwise-uniform) field; chi_3D -> 1 means the
field is dominated by spanwise fluctuations the mid-plane slice cannot see.

We report two variants:
  * "full": all three vorticity components in the norm.
  * "wz":   omega_z only (component index 2) -- the single channel the encoder
            actually ingests, so its fluctuating fraction bounds what a
            mid-plane-only encoder can be blind to.

inside-solid cells (where /inside_solid > 0, equivalently where /curlU is NaN)
are masked out of BOTH numerator and denominator, consistently.

We summarise each case by max_t chi_3D over the analysed frame window, and plot
max_t chi_3D vs |G|.

Frame subsampling. Raw HDF5 are ~4-6.5 GB each and reading every frame x 32 z
x 3 comps is heavy. We analyse ONE representative encounter window per case:
the first full encounter, frames [0, 120), subsampled every SUBSAMPLE-th frame.
Both periodic (800 frames, 6 encounters) and run3 (480 frames, 4 encounters)
have at least one full encounter, so the choice is identical across source
groups and captures the gust-impact event at frame ~40. This is stated in the
output JSON and CSV (n_frames_used) and is consistent across cases.

Window decomposition (a measured confound, not cosmetic). The spun-up periodic
shedding wake at Re=5000 already carries substantial three-dimensionality:
the |G|=0 Baseline sits at max chi_full ~ 0.52, wz ~ 0.27, roughly flat across
the encounter, and frame 0 of EVERY case (gust not yet released) is ~0.52
regardless of G. A naive max over the whole encounter is therefore dominated by
this ambient shedding floor and is blind to the gust-induced 3D content. We
report the max chi_3D over three windows so the gust signal is separable:
  * whole encounter [0,120)        -- the literal spec metric (ambient-floored),
  * impact window [25,55]          -- when the gust physically hits the airfoil,
  * post-impact wake [56,120)      -- where gust-induced wake breakdown shows.
Physically, a moderate gust transiently ORGANISES the near field into a coherent
2D vortex (chi drops at impact); the gust-induced three-dimensionalisation
re-emerges downstream. The GATE is evaluated on the post-impact-wake metric
(and its excess over the |G|=0 Baseline), which isolates the gust contribution;
the whole-encounter metric is reported alongside for completeness.

CPU only. Run from repo root:
  source .venv/bin/activate
  export PREVENT_ROOT=$HOME/PREVENT WANDB_PROJECT=vortex-jepa
  python scripts/session23/exp_chi3d.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
SPLIT_PATH = REPO / "configs/splits/split_v2.json"
OUT_DIR = REPO / "outputs/session23/chi3d"

# Analysis window: first full encounter, every SUBSAMPLE-th frame.
ENCOUNTER_START = 0
ENCOUNTER_END = 120  # one full encounter (encounter_frames in split_v2)
SUBSAMPLE = 2  # every 2nd frame -> 60 frames per case

# Sub-windows within the encounter (frame indices, inclusive/exclusive as noted).
# impact window from split_v2 impact_metadata: [25, 55]; post-impact wake after.
IMPACT_WIN = (25, 55)  # inclusive both ends (gust-airfoil interaction)
POSTIMPACT_START = 56  # post-impact wake [56, 120)

# Expected raw geometry. We verify shape per file and adapt / report if it differs.
EXPECTED_NDIM = 5
EXPECTED_SPATIAL = (192, 96, 32)  # (x, y, z)
EXPECTED_NCOMP = 3
WZ_INDEX = 2  # omega_z is component index 2 (du/dy - dv/dx)


def prevent_root() -> Path:
    root = os.environ.get("PREVENT_ROOT")
    if not root:
        sys.exit("PREVENT_ROOT is not set. `export PREVENT_ROOT=$HOME/PREVENT` first.")
    p = Path(root).expanduser()
    if not p.exists():
        sys.exit(f"PREVENT_ROOT does not exist: {p}")
    return p


# Read-integrity guard. A collaborator is concurrently regenerating run3 raw and
# running SOD2D on the same storage; under that I/O contention an HDF5 chunk read
# can silently return inconsistent bytes (observed: a frame's chi value flipped
# between invocations with no exception and a stable file afterwards). We defend
# by reading each frame twice with a fresh dataset handle and requiring the two
# reads to be byte-identical (NaNs compared as equal); retry up to N times.
READ_VERIFY_RETRIES = 4
_read_repair_events: list[dict] = []


def read_frame_verified(h: "h5py.File", t: int, case_id: str) -> np.ndarray:
    """Return /curlU[t] as float64, verified stable across two independent reads.

    Reads via a re-fetched dataset object each attempt (so any per-handle chunk
    cache is bypassed). Compares raw bytes with NaN==NaN. Raises after the retry
    budget so a genuinely unstable file is loud, not silently averaged.
    """
    prev: np.ndarray | None = None
    for attempt in range(READ_VERIFY_RETRIES + 1):
        a = np.asarray(h["/curlU"][t])  # fresh dataset fetch each attempt
        if prev is not None:
            same = np.array_equal(a, prev, equal_nan=True)
            if same:
                if attempt > 1:
                    _read_repair_events.append(
                        {"case_id": case_id, "frame": int(t), "attempts": attempt}
                    )
                return prev.astype(np.float64)
        prev = a
    raise RuntimeError(
        f"unstable read for {case_id} frame {t}: /curlU bytes differed across "
        f"{READ_VERIFY_RETRIES + 1} reads (storage under concurrent write?)"
    )


def chi3d_for_file(path: Path) -> dict:
    """Compute max_t chi_3D (full and wz) over the analysis window for one case.

    Returns a dict with max_chi3d_full, max_chi3d_wz, n_frames_used, and the
    per-frame series (for optional inspection). Masks inside-solid (NaN) cells
    out of both numerator and denominator. Accumulates frame-by-frame so memory
    stays at a single (192, 96, 32, 3) frame.
    """
    with h5py.File(path, "r") as h:
        cu = h["/curlU"]
        shape = tuple(cu.shape)
        ndim = cu.ndim
        # Verify geometry; adapt the analysis window to the available frames.
        if ndim != EXPECTED_NDIM:
            return {"error": f"curlU ndim {ndim} != {EXPECTED_NDIM}, shape {shape}"}
        n_frames_total = shape[0]
        spatial = shape[1:4]
        ncomp = shape[4]
        if spatial != EXPECTED_SPATIAL or ncomp != EXPECTED_NCOMP:
            return {
                "error": f"unexpected curlU shape {shape} "
                f"(expected (T,{EXPECTED_SPATIAL[0]},{EXPECTED_SPATIAL[1]},"
                f"{EXPECTED_SPATIAL[2]},{EXPECTED_NCOMP}))"
            }

        end = min(ENCOUNTER_END, n_frames_total)
        frame_idx = list(range(ENCOUNTER_START, end, SUBSAMPLE))

        # Spatial validity mask from inside_solid (true == fluid). curlU NaN cells
        # coincide with inside_solid>0 (verified), but we drive masking off
        # inside_solid so a stray NaN elsewhere is also handled below.
        ins = np.asarray(h["/inside_solid"])[..., 0] > 0  # (192,96,32) True==solid
        valid = ~ins  # (x,y,z) True==fluid

        chi_full = np.empty(len(frame_idx), dtype=np.float64)
        chi_wz = np.empty(len(frame_idx), dtype=np.float64)

        case_id = path.stem
        for i, t in enumerate(frame_idx):
            om = read_frame_verified(h, t, case_id)  # (192,96,32,3) float64, verified
            # Robustness: treat any NaN (should be solid cells only) as invalid.
            nan_any = np.isnan(om).any(axis=-1)  # (x,y,z)
            vmask = valid & ~nan_any  # (x,y,z) fluid AND finite

            # Zero out invalid cells so they contribute nothing to either sum.
            vmask4 = vmask[..., None]  # (x,y,z,1)
            om = np.where(vmask4, om, 0.0)

            # Spanwise (z) mean over FLUID stations only, per (x, y, comp).
            # n_valid_z(x,y) = number of fluid stations in the spanwise column.
            n_valid_z = vmask.sum(axis=2)  # (x,y), int
            has_z = n_valid_z > 0
            # sum over z of omega (invalid already zeroed) / count of valid z
            sum_z = om.sum(axis=2)  # (x,y,comp)
            mean_z = np.zeros_like(sum_z)
            mean_z[has_z] = sum_z[has_z] / n_valid_z[has_z][:, None]
            # broadcast mean back over z and re-mask (so solid cells stay zero in fluct).
            # mean_z is (x,y,comp); insert the z-axis at position 2 -> (x,y,1,comp).
            fluct = (om - mean_z[:, :, None, :]) * vmask4  # (x,y,z,comp)

            # ||.||^2 summed over comp, then over valid xyz cells.
            sq_full_total = float((om**2).sum())  # denom (full)
            sq_fluct_full = float((fluct**2).sum())  # numer (full)
            # omega_z-only variant
            om_wz = om[..., WZ_INDEX]
            fl_wz = fluct[..., WZ_INDEX]
            sq_full_wz = float((om_wz**2).sum())
            sq_fluct_wz = float((fl_wz**2).sum())

            chi_full[i] = sq_fluct_full / sq_full_total if sq_full_total > 0 else np.nan
            chi_wz[i] = sq_fluct_wz / sq_full_wz if sq_full_wz > 0 else np.nan

        fi = np.asarray(frame_idx)
        imp_mask = (fi >= IMPACT_WIN[0]) & (fi <= IMPACT_WIN[1])
        post_mask = fi >= POSTIMPACT_START

        def safe_max(arr, mask):
            sel = arr[mask]
            return float(np.nanmax(sel)) if sel.size and np.isfinite(sel).any() else np.nan

        return {
            # whole-encounter (literal spec metric; ambient-shedding-floored)
            "max_chi3d_full": float(np.nanmax(chi_full)) if chi_full.size else np.nan,
            "max_chi3d_wz": float(np.nanmax(chi_wz)) if chi_wz.size else np.nan,
            # impact window [25,55]
            "max_chi3d_full_impact": safe_max(chi_full, imp_mask),
            "max_chi3d_wz_impact": safe_max(chi_wz, imp_mask),
            # post-impact wake [56,120) -- isolates gust-induced 3D content
            "max_chi3d_full_post": safe_max(chi_full, post_mask),
            "max_chi3d_wz_post": safe_max(chi_wz, post_mask),
            "mean_chi3d_full": float(np.nanmean(chi_full)) if chi_full.size else np.nan,
            "mean_chi3d_wz": float(np.nanmean(chi_wz)) if chi_wz.size else np.nan,
            "argmax_frame_full": (
                int(frame_idx[int(np.nanargmax(chi_full))]) if chi_full.size else -1
            ),
            "n_frames_used": len(frame_idx),
            "frame_window": [ENCOUNTER_START, end, SUBSAMPLE],
            "n_frames_total": int(n_frames_total),
            "series_full": chi_full.tolist(),
            "series_wz": chi_wz.tolist(),
            "frame_idx": frame_idx,
        }


def main() -> None:
    PR = prevent_root()
    with open(SPLIT_PATH) as f:
        split = json.load(f)
    cases = split["cases"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    missing = []
    print(
        f"Analysing {len(cases)} cases | window [{ENCOUNTER_START},{ENCOUNTER_END}) "
        f"every {SUBSAMPLE} frames"
    )
    t_start = time.time()
    for n, (cid, c) in enumerate(sorted(cases.items(), key=lambda kv: abs(kv[1]["G"]))):
        path = PR / c["relative_path"]
        G = float(c["G"])
        rec = {
            "case_id": cid,
            "G": G,
            "absG": abs(G),
            "D": float(c["D"]),
            "Y": float(c["Y"]),
            "split": c.get("split", "?"),
            "source_group": c.get("source_group", "?"),
            "relative_path": c["relative_path"],
        }
        if not path.exists():
            rec.update(
                {
                    "missing_flag": True,
                    "max_chi3d_full": None,
                    "max_chi3d_wz": None,
                    "n_frames_used": 0,
                }
            )
            missing.append((cid, abs(G), c["relative_path"]))
            records.append(rec)
            print(f"[{n+1:2d}/{len(cases)}] MISSING {cid}  ({c['relative_path']})")
            continue
        t0 = time.time()
        res = chi3d_for_file(path)
        dt = time.time() - t0
        if "error" in res:
            rec.update(
                {
                    "missing_flag": True,
                    "error": res["error"],
                    "max_chi3d_full": None,
                    "max_chi3d_wz": None,
                    "n_frames_used": 0,
                }
            )
            missing.append((cid, abs(G), f"{c['relative_path']} ({res['error']})"))
            records.append(rec)
            print(f"[{n+1:2d}/{len(cases)}] SHAPE-ERR {cid}: {res['error']}")
            continue
        rec.update({"missing_flag": False, **res})
        records.append(rec)
        print(
            f"[{n+1:2d}/{len(cases)}] {cid:24s} |G|={abs(G):.2f} "
            f"whole={res['max_chi3d_full']:.3f} post={res['max_chi3d_full_post']:.3f} "
            f"(wz post={res['max_chi3d_wz_post']:.3f}) "
            f"({res['n_frames_used']} fr, {dt:.1f}s)"
        )

    elapsed = time.time() - t_start
    print(
        f"\nDone in {elapsed:.0f}s. present={sum(1 for r in records if not r['missing_flag'])} "
        f"missing={len(missing)}"
    )

    # --- write JSON (full records incl. per-frame series) --------------------
    meta = {
        "definition": "chi_3D(t) = sum_xyz ||omega - mean_z(omega)||^2 / sum_xyz ||omega||^2",
        "variants": {
            "full": "all 3 vorticity components in the norm",
            "wz": "omega_z only (component index 2; the channel the encoder sees)",
        },
        "masking": "inside_solid>0 (== curlU NaN) cells removed from BOTH sums; "
        "spanwise mean taken over fluid stations only",
        "summary_stat": "max over the analysed frame window per case",
        "frame_window": [ENCOUNTER_START, ENCOUNTER_END, SUBSAMPLE],
        "frame_window_note": "first full encounter [0,120), every 2nd frame; "
        "identical across periodic (800fr) and run3 (480fr); "
        "captures gust impact at frame ~40",
        "sub_windows": {
            "whole": [ENCOUNTER_START, ENCOUNTER_END],
            "impact": [IMPACT_WIN[0], IMPACT_WIN[1] + 1],
            "post_impact": [POSTIMPACT_START, ENCOUNTER_END],
        },
        "ambient_floor_note": "the spun-up periodic shedding wake already carries "
        "max chi_full ~ 0.52 / wz ~ 0.27 at |G|=0 (Baseline) and at frame 0 of "
        "every case; the whole-encounter max is dominated by this floor. The GATE "
        "uses the post-impact-wake metric, which isolates the gust contribution.",
        "split_path": str(SPLIT_PATH),
        "split_sha256_source_inventory": split.get("source_inventory", {}).get("sha256"),
        "n_cases": len(cases),
        "n_present": sum(1 for r in records if not r["missing_flag"]),
        "n_missing": len(missing),
        "missing_cases": [{"case_id": m[0], "absG": m[1], "path": m[2]} for m in missing],
        "elapsed_seconds": round(elapsed, 1),
        "read_verification": {
            "retries_budget": READ_VERIFY_RETRIES,
            "n_repair_events": len(_read_repair_events),
            "events": _read_repair_events,
            "note": "each frame read twice with a fresh handle and required "
            "byte-identical; guards against silent chunk corruption under "
            "concurrent storage writes (collaborator regenerating run3).",
        },
    }
    json_path = OUT_DIR / "chi3d.json"
    with open(json_path, "w") as f:
        json.dump({"meta": meta, "records": records}, f, indent=2)
    print(f"wrote {json_path}")

    # --- write CSV (one row per case) ----------------------------------------
    csv_path = OUT_DIR / "chi3d.csv"
    header = [
        "case_id",
        "G",
        "|G|",
        "D",
        "Y",
        "split",
        "source_group",
        "max_chi3d_full",
        "max_chi3d_wz",
        "max_chi3d_full_impact",
        "max_chi3d_wz_impact",
        "max_chi3d_full_post",
        "max_chi3d_wz_post",
        "n_frames_used",
        "missing_flag",
    ]
    keys = [
        "max_chi3d_full",
        "max_chi3d_wz",
        "max_chi3d_full_impact",
        "max_chi3d_wz_impact",
        "max_chi3d_full_post",
        "max_chi3d_wz_post",
    ]
    with open(csv_path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in sorted(records, key=lambda x: (x["absG"], x["case_id"])):

            def fmt(v):
                if v is None or (isinstance(v, float) and not np.isfinite(v)):
                    return ""
                if isinstance(v, float):
                    return f"{v:.6f}"
                return str(v)

            row = [
                r["case_id"],
                fmt(r["G"]),
                fmt(r["absG"]),
                fmt(r["D"]),
                fmt(r["Y"]),
                r["split"],
                r["source_group"],
            ]
            row += [fmt(r.get(k)) for k in keys]
            row += [str(r.get("n_frames_used", 0)), str(r["missing_flag"])]
            f.write(",".join(row) + "\n")
    print(f"wrote {csv_path}")

    # --- gate quantification --------------------------------------------------
    quant = quantify_jump(records)
    with open(OUT_DIR / "chi3d_gate.json", "w") as f:
        json.dump(quant, f, indent=2)
    print("\n=== GATE: jump in max_t chi_3D at |G|=4 vs |G|<=3 ===")
    for window in ("whole", "impact", "post_impact"):
        tag = {
            "whole": "WHOLE encounter (ambient-floored)",
            "impact": "IMPACT window [25,55]",
            "post_impact": "POST-IMPACT wake [56,120) <-- gust-isolating GATE",
        }[window]
        print(f"\n  -- {tag} --")
        for variant in ("full", "wz"):
            q = quant[window][variant]
            print(
                f"    [{variant}] median |G|<=3 = {q['median_le3']:.4f} (n={q['n_le3']}), "
                f"median |G|=4 = {q['median_g4']:.4f} (n={q['n_g4']}), "
                f"ratio = {q['ratio_g4_over_le3']:.2f}x"
            )
            print(
                f"           within |G|<=3 trend: Spearman rho = {q['spearman_le3']:.3f}, "
                f"OLS slope = {q['ols_slope_le3']:.5f}/unit-|G|; "
                f"|G|=4 min = {q['min_g4']:.4f} vs |G|<=3 max = {q['max_le3']:.4f}; "
                f"baseline(|G|=0) = {q['baseline_value']}"
            )

    # --- figure ---------------------------------------------------------------
    make_figure(records, quant)
    print(f"wrote {OUT_DIR / 'chi3d.png'}")


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (robust to nonlinearity)."""
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def quantify_jump(records: list[dict]) -> dict:
    """Quantify the |G|=4 jump and within-|G|<=3 trend per window and variant.

    Windows: whole / impact / post_impact. Variants: full / wz. For each, we
    report the |G|=4-vs-|G|<=3 median ratio, the within-|G|<=3 Spearman rho and
    OLS slope, and the |G|=0 Baseline value (so the ambient-shedding floor is
    explicit and the gust-induced excess can be read off).
    """
    window_keys = {
        "whole": ("max_chi3d_full", "max_chi3d_wz"),
        "impact": ("max_chi3d_full_impact", "max_chi3d_wz_impact"),
        "post_impact": ("max_chi3d_full_post", "max_chi3d_wz_post"),
    }
    out: dict = {}
    for window, (kf, kw) in window_keys.items():
        out[window] = {}
        for variant, key in (("full", kf), ("wz", kw)):
            rows = [
                (r["absG"], r[key])
                for r in records
                if not r["missing_flag"]
                and r.get(key) is not None
                and np.isfinite(r.get(key, np.nan))
            ]
            absg = np.array([a for a, _ in rows])
            val = np.array([v for _, v in rows])
            le3_m = absg <= 3.0 + 1e-9
            le3 = val[le3_m]
            absg_le3 = absg[le3_m]
            g4 = val[np.isclose(absg, 4.0)]

            if len(absg_le3) >= 2:
                A = np.vstack([absg_le3, np.ones_like(absg_le3)]).T
                slope, intercept = np.linalg.lstsq(A, le3, rcond=None)[0]
            else:
                slope = intercept = float("nan")

            med_le3 = float(np.median(le3)) if le3.size else float("nan")
            med_g4 = float(np.median(g4)) if g4.size else float("nan")
            # |G|=0 Baseline value (ambient shedding floor)
            base = [
                r[key]
                for r in records
                if np.isclose(r["absG"], 0.0) and not r["missing_flag"] and r.get(key) is not None
            ]
            base_val = float(base[0]) if base else float("nan")

            out[window][variant] = {
                "n_le3": int(le3.size),
                "n_g4": int(g4.size),
                "median_le3": med_le3,
                "median_g4": med_g4,
                "ratio_g4_over_le3": float(med_g4 / med_le3) if med_le3 > 0 else float("nan"),
                "max_le3": float(le3.max()) if le3.size else float("nan"),
                "min_g4": float(g4.min()) if g4.size else float("nan"),
                "mean_le3": float(le3.mean()) if le3.size else float("nan"),
                "mean_g4": float(g4.mean()) if g4.size else float("nan"),
                "spearman_le3": _spearman(absg_le3, le3),
                "ols_slope_le3": float(slope),
                "ols_intercept_le3": float(intercept),
                "baseline_value": round(base_val, 6) if np.isfinite(base_val) else None,
                "excess_g4_over_baseline": (
                    float(med_g4 - base_val) if np.isfinite(base_val) else float("nan")
                ),
                "per_level_median": {
                    f"{lvl:.2f}": float(np.median(val[np.isclose(absg, lvl)]))
                    for lvl in sorted(set(np.round(absg, 2)))
                },
                "per_level_n": {
                    f"{lvl:.2f}": int(np.isclose(absg, lvl).sum())
                    for lvl in sorted(set(np.round(absg, 2)))
                },
            }
    return out


def make_figure(records: list[dict], quant: dict) -> None:
    """Scatter max_t chi_3D vs |G|: post-impact-wake (gust-isolating, top row)
    and whole-encounter (ambient-floored, bottom row), each in full and wz.
    """
    sys.path.insert(0, str(REPO / "scripts/session21"))
    import figstyle  # noqa: E402

    import matplotlib.pyplot as plt

    figstyle.use_style()

    present = [
        r for r in records if not r["missing_flag"] and r.get("max_chi3d_full_post") is not None
    ]
    absg = np.array([r["absG"] for r in present])
    is_g4 = np.isclose(absg, 4.0)
    col_train = "#2166ac"  # blue
    col_g4 = "#b2182b"  # red (test C, |G|=4)

    panels = [
        (
            "post_impact",
            "full",
            "max_chi3d_full_post",
            r"post-impact wake, full $\|\boldsymbol{\omega}\|$",
        ),
        (
            "post_impact",
            "wz",
            "max_chi3d_wz_post",
            r"post-impact wake, $\omega_z$ (encoder channel)",
        ),
        (
            "whole",
            "full",
            "max_chi3d_full",
            r"whole encounter, full $\|\boldsymbol{\omega}\|$ (ambient-floored)",
        ),
        ("whole", "wz", "max_chi3d_wz", r"whole encounter, $\omega_z$ (ambient-floored)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=figstyle.figure_size(1.0, aspect=0.85))
    axflat = axes.ravel()
    for ax, (window, variant, key, title) in zip(axflat, panels):
        val = np.array([r[key] for r in present])
        ax.scatter(
            absg[~is_g4],
            val[~is_g4],
            s=14,
            c=col_train,
            marker="o",
            edgecolors="white",
            linewidths=0.3,
            zorder=3,
            label=r"training range $|G|\leq3$",
        )
        ax.scatter(
            absg[is_g4],
            val[is_g4],
            s=24,
            c=col_g4,
            marker="^",
            edgecolors="white",
            linewidths=0.4,
            zorder=4,
            label=r"test C $|G|=4$",
        )
        q = quant[window][variant]
        lvls = sorted(float(k) for k in q["per_level_median"] if float(k) <= 3.0)
        meds = [q["per_level_median"][f"{lv:.2f}"] for lv in lvls]
        ax.plot(lvls, meds, "-", color="#404040", lw=0.9, zorder=2, label="per-level median")
        # baseline floor as a dashed grey reference
        if q["baseline_value"] is not None:
            ax.axhline(q["baseline_value"], color="#888888", lw=0.7, ls=(0, (4, 3)), zorder=1)
        ax.set_title(title, fontsize=7.2)
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_xlim(-0.2, 4.3)
        ax.set_ylim(bottom=0)
    axes[1, 0].set_xlabel(r"$|G|$")
    axes[1, 1].set_xlabel(r"$|G|$")
    axes[0, 0].set_ylabel(r"$\max_t\,\chi_{3\mathrm{D}}$")
    axes[1, 0].set_ylabel(r"$\max_t\,\chi_{3\mathrm{D}}$")
    axes[0, 1].legend(loc="upper center", fontsize=6.0)
    fig.suptitle(
        r"Spanwise-fluctuating enstrophy fraction "
        r"$\chi_{3\mathrm{D}}=\sum\|\boldsymbol{\omega}-\bar{\boldsymbol{\omega}}_z\|^2"
        r"/\sum\|\boldsymbol{\omega}\|^2$  vs gust ratio $|G|$"
        "\n(dashed grey: $|G|=0$ ambient shedding floor)",
        fontsize=8,
        y=1.00,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "chi3d.png", dpi=300)
    fig.savefig(OUT_DIR / "chi3d.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
