"""Internal undisturbed-flow validation (master plan A2; closes the computable half of B1).

Computes mean C_L, rms C_L', mean C_D, rms C_D', and the lift-spectrum Strouhal
peak(s) of the undisturbed Baseline DNS, and tabulates them against the PRF 2025
lineage values (Fukami, Smith and Taira, PRF 10, 084703, Fig. 2(b)) and the two
external references PRF validates against. Cross-checks the manuscript's
St = 0.36 (referee M13).

Data: raw ${PREVENT_ROOT}/data/raw/periodic/Baseline.h5 /forces/{CL,CD}
(800 frames at dt_tc = 0.05, already non-dimensionalised). The cached Baseline
encounters are cross-checked against the raw series (cache integrity).

Outputs:
    outputs/session28/undisturbed_stats.json          full record
    outputs/session28/undisturbed_spectrum.npz        f, P(C_L) for an appendix figure
    outputs/session28/numbers_parts/undisturbed.json  macro-bound part for eval_all.py

Usage:
    python scripts/session28/undisturbed_validation.py \
        [--split configs/splits/split_v2p1.json] [--transient-cut-tc 5.0]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DT_TC = 0.05

# PRF 2025 Fig. 2(b): undisturbed-flow validation table (Lz/c = 1 unless noted).
REFERENCES = {
    "prf2025_regular_8M": {
        "CL_mean": 0.730,
        "CL_rms": 0.103,
        "CD_mean": 0.245,
        "CD_rms": 0.0224,
        "note": "Fukami et al. PRF 2025, regular grid (~8M)",
    },
    "prf2025_fine_20M": {
        "CL_mean": 0.737,
        "CL_rms": 0.116,
        "CD_mean": 0.249,
        "CD_rms": 0.0190,
        "note": "Fukami et al. PRF 2025, fine grid (~20M, production)",
    },
    "prf2025_finer_40M": {
        "CL_mean": 0.738,
        "CL_rms": 0.098,
        "CD_mean": 0.249,
        "CD_rms": 0.0183,
        "note": "Fukami et al. PRF 2025, finer grid (~40M)",
    },
    "rolandi2025_les": {
        "CL_mean": 0.734,
        "CD_mean": 0.246,
        "note": "Rolandi et al. (2025), LES at M_inf = 0.2, Lz/c = 1 (PRF Fig. 2(b))",
    },
    "gupta2023_exp": {
        "CL_mean": 0.763,
        "CD_mean": 0.223,
        "note": "Gupta et al. (2023), wind-tunnel experiment, Lz/c = 10 (PRF Fig. 2(b))",
    },
}
MANUSCRIPT_ST = 0.36  # the value currently printed in Sec. 2; M13 cross-check


def spectral_peaks(x: np.ndarray, dt: float, n_peaks: int = 3) -> list[dict]:
    """Dominant local-maximum peaks of the one-sided fluctuation spectrum.

    Only interior local maxima of the spectrum are eligible (the original
    picker took the argmax of a window-suppressed copy, which could land on
    the slope at a suppressed-window edge and emit a negative frequency via
    unclamped interpolation). The log-parabolic shift is applied only where
    the log-domain curvature is negative and is clamped to half a bin, so
    every returned St is >= 0 and within half a bin of a spectral local max.
    """
    xf = x - x.mean()
    n = xf.size
    win = np.hanning(n)
    X = np.fft.rfft(xf * win)
    P = (np.abs(X) ** 2) / (np.sum(win**2) / 2.0)
    f = np.fft.rfftfreq(n, d=dt)
    df = float(f[1] - f[0])
    interior = np.arange(1, P.size - 1)
    locmax = interior[(P[interior] >= P[interior - 1]) & (P[interior] >= P[interior + 1])]
    peaks: list[dict] = []
    suppressed = np.zeros(P.size, dtype=bool)
    for i in locmax[np.argsort(P[locmax])[::-1]]:
        if suppressed[i] or P[i] <= 0:
            continue
        shift = 0.0
        if P[i - 1] > 0 and P[i + 1] > 0:
            la, lb, lc = np.log(P[i - 1]), np.log(P[i]), np.log(P[i + 1])
            denom = la - 2 * lb + lc
            if denom < 0:
                shift = float(np.clip(0.5 * (la - lc) / denom, -0.5, 0.5))
        peaks.append({"St": float(f[i] + shift * df), "power": float(P[i])})
        suppressed[max(i - 2, 0) : i + 3] = True
        if len(peaks) >= n_peaks:
            break
    return peaks


def lift_spectrum_summary(x: np.ndarray, dt: float, st_floor: float = 0.15) -> dict:
    """Shedding line vs low-frequency modulation vs subharmonic, separated.

    The Baseline lift spectrum (measured 2026-06-11) carries three distinct
    features: the dominant shedding line (St ~ 0.69), a subharmonic at half
    that frequency (St ~ 0.34, the full vortex-shedding cycle; this is the
    clock the latent phase portrait traces and the manuscript's "St near
    0.36"), and a low-frequency modulation of the separated flow (St ~ 0.04).
    A single argmax conflates them; this reports each under its own key.
    """
    peaks = spectral_peaks(x, dt, n_peaks=12)
    above = [p for p in peaks if p["St"] >= st_floor]
    below = [p for p in peaks if p["St"] < st_floor]
    shedding = above[0] if above else None
    modulation = below[0] if below else None
    subharmonic = None
    if shedding is not None:
        tol = 1.5 / (x.size * dt)
        cands = [p for p in above[1:] if abs(p["St"] - shedding["St"] / 2.0) <= tol]
        subharmonic = cands[0] if cands else None
    return {
        "St_shedding": shedding,
        "St_modulation": modulation,
        "St_subharmonic": subharmonic,
        "st_floor": st_floor,
    }


def window_stats(cl: np.ndarray, cd: np.ndarray, dt: float, lo_tc: float, hi_tc: float) -> dict:
    """Moment statistics of C_L / C_D over a stated convective-time window."""
    lo, hi = int(round(lo_tc / dt)), int(round(hi_tc / dt))
    c, d = cl[lo:hi], cd[lo:hi]
    return {
        "CL_mean": float(c.mean()),
        "CL_rms": float(c.std(ddof=0)),
        "CD_mean": float(d.mean()),
        "CD_rms": float(d.std(ddof=0)),
        "n_frames": int(c.size),
        "window_tc": [float(lo_tc), float(hi_tc)],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", default="configs/splits/split_v2p1.json")
    p.add_argument(
        "--transient-cut-tc",
        type=float,
        default=5.0,
        help="Convective times discarded at the head of the raw series "
        "(spectrum window; kept for transparency).",
    )
    p.add_argument(
        "--stationary-window-tc",
        type=float,
        nargs=2,
        default=(20.0, 40.0),
        metavar=("LO", "HI"),
        help="Stationary window for the validation-row moment statistics. "
        "Block means over t/c [5, 15] sit at 0.92-0.99 vs 0.64-0.83 "
        "after; the first half of the record is startup transient.",
    )
    p.add_argument("--out", default="outputs/session28/undisturbed_stats.json")
    args = p.parse_args()

    prevent_root = Path(os.environ.get("PREVENT_ROOT", str(Path.home() / "PREVENT")))
    split = json.loads((REPO / args.split).read_text())
    base = split["cases"].get("Baseline")
    if base is None:
        raise SystemExit("Baseline case missing from the split manifest")
    raw_path = prevent_root / base["relative_path"]
    if not raw_path.exists():
        raise SystemExit(f"raw Baseline missing: {raw_path}")

    with h5py.File(raw_path, "r") as g:
        cl = np.asarray(g["forces/CL"], dtype=np.float64).squeeze()
        cd = np.asarray(g["forces/CD"], dtype=np.float64).squeeze()

    cut = int(round(args.transient_cut_tc / DT_TC))
    cl_s = cl[cut:]
    lo_tc, hi_tc = args.stationary_window_tc

    # Validation-row moments from the stationary window; the post-cut full
    # record is kept alongside for transparency (it includes the startup
    # segment and overstates the rms).
    stationary = window_stats(cl, cd, DT_TC, lo_tc, hi_tc)
    full_after_cut = window_stats(cl, cd, DT_TC, args.transient_cut_tc, cl.size * DT_TC)
    stats = dict(stationary)
    stats.update(
        {
            "n_frames_total": int(cl.size),
            "stationary_window_tc": [float(lo_tc), float(hi_tc)],
            "full_record_after_cut": full_after_cut,
            "transient_cut_tc": args.transient_cut_tc,
        }
    )

    # Spectrum on the longest post-transient record (best frequency
    # resolution, df = 1/35 in St units for the 800-frame Baseline).
    summary = lift_spectrum_summary(cl_s, DT_TC)
    stats["St_peaks"] = spectral_peaks(cl_s, DT_TC)
    stats["St_lift_dominant"] = (
        summary["St_shedding"]["St"] if summary["St_shedding"] else float("nan")
    )
    stats["St_lift_subharmonic"] = (
        summary["St_subharmonic"]["St"] if summary["St_subharmonic"] else None
    )
    stats["St_low_freq_modulation"] = (
        summary["St_modulation"]["St"] if summary["St_modulation"] else None
    )
    stats["St_floor"] = summary["st_floor"]
    stats["St_manuscript"] = MANUSCRIPT_ST
    # M13 resolution: the manuscript's 0.36 is the full-cycle clock (the
    # latent orbit period, 56 frames = 2.8 t/c), i.e. the SUBHARMONIC of the
    # dominant lift line at ~0.69, not the dominant line itself.
    sub = stats["St_lift_subharmonic"]
    # Comparison tolerance cannot be tighter than the spectral resolution
    # (df = 1/(n * dt) in St units; 0.029 for the 700-frame post-cut record).
    df_st = 1.0 / (cl_s.size * DT_TC)
    tol = max(0.02, df_st)
    stats["St_comparison_tolerance"] = tol
    stats["manuscript_0p36_matches_subharmonic"] = bool(
        sub is not None and abs(sub - MANUSCRIPT_ST) <= tol
    )
    stats["manuscript_0p36_matches_dominant"] = bool(
        abs(stats["St_lift_dominant"] - MANUSCRIPT_ST) <= tol
    )

    # Cache cross-check: concatenated cached Baseline encounters == raw segments.
    cache = Path(
        os.environ.get("VORTEX_JEPA_CACHE", str(prevent_root / "data/processed/vortex-jepa"))
    )
    part_dir = cache / "v2p1" / "Baseline"
    cache_check = {"checked": False}
    if part_dir.exists():
        cls = []
        for k in range(int(base["n_encounters_full"])):
            f = part_dir / f"encounter_{k:02d}.h5"
            if f.exists():
                with h5py.File(f, "r") as g:
                    cls.append(np.asarray(g["C_L"], dtype=np.float64))
        if cls:
            cat = np.concatenate(cls)
            n = cat.size
            cache_check = {
                "checked": True,
                "n_cached_frames": int(n),
                "max_abs_diff_vs_raw": float(np.max(np.abs(cat - cl[:n]))),
            }
    stats["cache_cross_check"] = cache_check

    deltas = {}
    fine = REFERENCES["prf2025_fine_20M"]
    for k in ("CL_mean", "CL_rms", "CD_mean", "CD_rms"):
        deltas[f"{k}_vs_prf_fine_pct"] = float(100.0 * (stats[k] - fine[k]) / fine[k])
    payload = {
        "_doc": "Undisturbed-flow validation (plan A2). DNS via SOD2D; configuration from "
        "Fukami et al. (2025). References from PRF 2025 Fig. 2(b).",
        "ours": stats,
        "references": REFERENCES,
        "deltas_vs_prf_fine": deltas,
        "raw_path": str(raw_path),
    }
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"[undisturbed] wrote {out}")
    print(json.dumps({"ours": stats, "deltas_vs_prf_fine": deltas}, indent=2)[:1200])

    np.savez(
        REPO / "outputs/session28/undisturbed_spectrum.npz",
        f=np.fft.rfftfreq(cl_s.size, d=DT_TC),
        # spectrum recomputed here so the NPZ matches the JSON peaks
        P=(lambda xf, w: (np.abs(np.fft.rfft(xf * w)) ** 2) / (np.sum(w**2) / 2.0))(
            cl_s - cl_s.mean(), np.hanning(cl_s.size)
        ),
        cl=cl,
        cd=cd,
        transient_cut=cut,
    )

    src = "undisturbed_validation.py"
    note = f"stationary window t/c [{lo_tc:g}, {hi_tc:g}]"
    part = {
        "part": "undisturbed",
        "numbers": {
            "undisturbed_CL_mean": {
                "macro": "NumUndistCLmean",
                "value": stats["CL_mean"],
                "fmt": "%.3f",
                "split": "baseline",
                "source": src,
                "note": note,
            },
            "undisturbed_CL_rms": {
                "macro": "NumUndistCLrms",
                "value": stats["CL_rms"],
                "fmt": "%.3f",
                "source": src,
                "note": note,
            },
            "undisturbed_CD_mean": {
                "macro": "NumUndistCDmean",
                "value": stats["CD_mean"],
                "fmt": "%.3f",
                "source": src,
                "note": note,
            },
            "undisturbed_CD_rms": {
                "macro": "NumUndistCDrms",
                "value": stats["CD_rms"],
                "fmt": "%.4f",
                "source": src,
                "note": note,
            },
            "undisturbed_St": {
                "macro": "NumUndistSt",
                "value": stats["St_lift_dominant"],
                "fmt": "%.2f",
                "source": src,
                "note": "dominant lift-spectrum line",
            },
            "undisturbed_St_subharmonic": {
                "macro": "NumUndistStSub",
                "value": stats["St_lift_subharmonic"],
                "fmt": "%.2f",
                "source": src,
                "note": "full-cycle clock; the manuscript's " "'St near 0.36' (M13)",
            },
            "undisturbed_St_modulation": {
                "macro": "NumUndistStMod",
                "value": stats["St_low_freq_modulation"],
                "fmt": "%.2f",
                "source": src,
                "note": "low-frequency modulation of the " "separated flow",
            },
        },
    }
    parts_dir = REPO / "outputs/session28/numbers_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    (parts_dir / "undisturbed.json").write_text(json.dumps(part, indent=2))
    print(f"[undisturbed] wrote numbers part -> {parts_dir / 'undisturbed.json'}")


if __name__ == "__main__":
    main()
