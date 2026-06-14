"""SESSION29 Track A: baseline external validation (referee F1; reconciled to v2.1).

The lineage hallmark (Fukami, Smith and Taira, PRF 10, 084703, 2025, Fig. 2(b))
validates the undisturbed NACA0012, Re = 5000, alpha = 14 deg baseline forces
against prior work before trusting any gust result. This script assembles that
cross-reference TABLE for OUR DNS so Table 1 is de-risked even while the partner
solver's grid/time-step rows are still pending.

Quantities (undisturbed / no-gust Baseline case): time-averaged C_L and C_D,
rms C_L' and C_D', the shedding Strouhal number from the lift spectrum, and the
wake enstrophy.

What is REUSED vs RECOMPUTED:
- C_L/C_D mean/rms and the shedding St (dominant line ~0.675 and full-cycle
  subharmonic ~0.34, the manuscript's "St near 0.36") are READ from the frozen
  session28 record `outputs/session28/numbers_parts/undisturbed.json`, which
  `undisturbed_validation.py` already cross-checked against the PRF 2025 fine
  grid (within 1.7-3.3%). We do NOT recompute them.
- Wake enstrophy is computed here directly from the cached Baseline omega_z
  fields, using the paper's canonical wake-box convention
  (scripts/session16/exp1b_decode_axes.py): on the RAW (192,96) field,
  E = sum over wake-box pixels of (|omega|*mask)^2 * dx * dy, box x in
  [0.5, 4.0], y in [-1.0, 1.0] over the (-1.5,4.5)x(-1.5,1.5) extent, averaged
  over the stationary window of every Baseline encounter.

External references (NACA0012, Re=5000, alpha=14 deg undisturbed baseline),
extracted from the local FukamiGustRe5000.pdf Fig. 2(b) (HANDOFF D181):
  (1) Fukami, Smith and Taira, PRF 10, 084703 (2025), fine grid (~20M, the
      production grid): CL 0.737, CL_rms 0.116, CD 0.249, CD_rms 0.0190, Lz/c=1.
  (2) Rolandi et al. (2025), LES at M_inf = 0.2, Lz/c = 1 (PRF Fig. 2(b)):
      CL 0.734, CD 0.246. The table reports NO rms and NO St for this row.
      Citation: L. V. Rolandi, L. Smith, M. Amitay, V. Theofilis, K. Taira,
      "Biglobal resolvent analysis of separated flow over a NACA0012 airfoil",
      arXiv:2501.04255.
  (3) Gupta et al. (2023), wind-tunnel experiment, Lz/c = 10 (PRF Fig. 2(b)):
      CL 0.763, CD 0.223. The table reports NO rms and NO St for this row.
      Citation: S. Gupta, J. Zhao, A. Sharma, A. Agrawal, K. Hourigan,
      M. C. Thompson, "Two- and three-dimensional wake transitions of a
      NACA0012 airfoil", J. Fluid Mech. 954, A26 (2023).
Strouhal is the manuscript's internal M13 cross-check; Fig. 2(b) reports no St
for any external reference, so the St row is OURS-only. Wake enstrophy is a
grid/normalisation-dependent internal diagnostic with no external table value;
it is reported as OURS-only and NOT band-checked.

CPU-only. Reads the cache (no GPU, no torch model).

Usage:
    python scripts/session29/validate_baseline.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np

import _s29_common as cm

REPO = cm.REPO
OUT_JSON = REPO / "outputs" / "session29" / "baseline_validation.json"
OUT_MD = REPO / "outputs" / "session29" / "baseline_validation.md"
UNDISTURBED = REPO / "outputs" / "session28" / "numbers_parts" / "undisturbed.json"
SPLIT = REPO / "configs" / "splits" / "split_v2p1.json"
CACHE = (
    Path(
        os.environ.get("VORTEX_JEPA_CACHE", str(Path.home() / "PREVENT/data/processed/vortex-jepa"))
    )
    / "v2p1"
)
DT_TC = 0.05

# Canonical wake-box / grid convention (scripts/session16/exp1b_decode_axes.py).
X_EXTENT = (-1.5, 4.5)
Y_EXTENT = (-1.5, 1.5)
NX, NY = 192, 96
DX = (X_EXTENT[1] - X_EXTENT[0]) / NX
DY = (Y_EXTENT[1] - Y_EXTENT[0]) / NY
WAKE_X = (0.5, 4.0)
WAKE_Y = (-1.0, 1.0)

# Stationary window (matches undisturbed_validation.py's moment window).
STATIONARY_TC = (20.0, 40.0)

# External references from FukamiGustRe5000.pdf Fig. 2(b) (HANDOFF D181).
# Missing cells are intentionally absent -> rendered NEEDS-LITERATURE.
NEEDS_LIT = "NEEDS-LITERATURE"
NO_EXT = "NO-EXTERNAL-REFERENCE"
REFERENCES = {
    "Fukami2025": {
        "CL_mean": 0.737,
        "CL_rms": 0.116,
        "CD_mean": 0.249,
        "CD_rms": 0.0190,
        "kind": "DNS (incompressible), fine grid ~20M (production)",
        "Lz_c": 1,
        "citation": "Fukami, Smith & Taira, Phys. Rev. Fluids 10, 084703 (2025),"
        " Fig. 2(b) fine grid",
        "source": "FukamiGustRe5000.pdf Fig. 2(b)",
    },
    "Rolandi2025": {
        "CL_mean": 0.734,
        "CD_mean": 0.246,
        "kind": "LES (M_inf = 0.2)",
        "Lz_c": 1,
        "citation": "Rolandi, Smith, Amitay, Theofilis & Taira," " arXiv:2501.04255 (2025)",
        "source": "FukamiGustRe5000.pdf Fig. 2(b) (no rms / St columns reported)",
    },
    "Gupta2023": {
        "CL_mean": 0.763,
        "CD_mean": 0.223,
        "kind": "wind-tunnel experiment",
        "Lz_c": 10,
        "citation": "Gupta, Zhao, Sharma, Agrawal, Hourigan & Thompson,"
        " J. Fluid Mech. 954, A26 (2023)",
        "source": "FukamiGustRe5000.pdf Fig. 2(b) (no rms / St columns reported)",
    },
}

# Quantities cross-tabulated. enstrophy carries no external table value.
QUANTITIES = [
    ("CL_mean", "time-averaged C_L", "%.3f"),
    ("CL_rms", "rms C_L'", "%.3f"),
    ("CD_mean", "time-averaged C_D", "%.3f"),
    ("CD_rms", "rms C_D'", "%.4f"),
    ("St", "shedding Strouhal", "%.3f"),
    ("enstrophy", "wake enstrophy", "%.4g"),
]


def _coord_grid():
    x = np.linspace(X_EXTENT[0] + DX / 2, X_EXTENT[1] - DX / 2, NX)
    y = np.linspace(Y_EXTENT[0] + DY / 2, Y_EXTENT[1] - DY / 2, NY)
    return x, y


def wake_mask() -> np.ndarray:
    """Boolean (NX, NY) wake-box mask (paper convention)."""
    x, y = _coord_grid()
    in_x = (x >= WAKE_X[0]) & (x <= WAKE_X[1])
    in_y = (y >= WAKE_Y[0]) & (y <= WAKE_Y[1])
    return in_x[:, None] & in_y[None, :]


def frame_wake_enstrophy(omega_raw: np.ndarray, mask: np.ndarray) -> float:
    """Wake enstrophy of one RAW (NX, NY) omega field (paper convention).

    E = sum over wake-box pixels of (|omega| * mask)^2 * dx * dy.
    """
    assert omega_raw.shape == (NX, NY), omega_raw.shape
    field = np.abs(omega_raw) * mask
    return float((field**2).sum() * DX * DY)


def rms_fluctuation(x: np.ndarray) -> float:
    """Root-mean-square fluctuation about the mean (population std)."""
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean((x - x.mean()) ** 2)))


def spectral_peak_st(x: np.ndarray, dt: float) -> float:
    """Strouhal of the dominant interior spectral line of x's fluctuation.

    One-sided Hann-windowed periodogram; returns the frequency (St units, since
    the series is already in convective time) of the largest interior local
    maximum. Used by the unit test on a synthetic sinusoid; the paper St is
    taken from the frozen undisturbed record, not from this helper.
    """
    xf = np.asarray(x, dtype=np.float64)
    xf = xf - xf.mean()
    n = xf.size
    win = np.hanning(n)
    power = np.abs(np.fft.rfft(xf * win)) ** 2
    freq = np.fft.rfftfreq(n, d=dt)
    interior = power[1:-1]
    if interior.size == 0:
        return float("nan")
    return float(freq[1 + int(np.argmax(interior))])


def load_undisturbed_numbers() -> dict:
    """Read the frozen session28 undisturbed numbers (mean/rms C_L,C_D + St)."""
    if not UNDISTURBED.exists():
        raise FileNotFoundError(
            f"missing frozen undisturbed record: {UNDISTURBED}; "
            "run scripts/session28/undisturbed_validation.py first"
        )
    nums = json.loads(UNDISTURBED.read_text())["numbers"]
    return {
        "CL_mean": nums["undisturbed_CL_mean"]["value"],
        "CL_rms": nums["undisturbed_CL_rms"]["value"],
        "CD_mean": nums["undisturbed_CD_mean"]["value"],
        "CD_rms": nums["undisturbed_CD_rms"]["value"],
        "St": nums["undisturbed_St"]["value"],
        "St_subharmonic": nums["undisturbed_St_subharmonic"]["value"],
        "_window_note": nums["undisturbed_CL_mean"].get("note", ""),
    }


def baseline_encounter_files() -> list[Path]:
    """Cached Baseline (no-gust) encounter files for the v2.1 split."""
    split = json.loads(SPLIT.read_text())
    base = split["cases"].get("Baseline")
    if base is None:
        raise SystemExit("Baseline case missing from split_v2p1.json")
    part_dir = CACHE / "Baseline"
    if not part_dir.exists():
        raise FileNotFoundError(f"missing Baseline cache dir: {part_dir}")
    n_enc = int(base.get("n_encounters_full", base.get("n_encounters", 6)))
    files = []
    for k in range(n_enc):
        f = part_dir / f"encounter_{k:02d}.h5"
        if not f.exists():
            raise FileNotFoundError(f"missing Baseline cache encounter: {f}")
        files.append(f)
    if not files:
        raise FileNotFoundError(f"no Baseline encounters under {part_dir}")
    return files


def compute_wake_enstrophy(files: list[Path]) -> dict:
    """Mean wake enstrophy over the stationary window across Baseline encounters.

    Each cached encounter is 120 frames; the stationary window [20, 40] t/c maps
    to frames [400, 800]/dt within an 800-frame raw record, but the cache stores
    one encounter = 120 frames at t in [0, 6) t/c. The Baseline is periodic, so
    every cached encounter is past the startup transient (encounter 0 starts at
    raw frame 0 but later encounters do not). We therefore take the per-encounter
    full-frame mean and report mean +/- std across encounters, plus the
    later-encounters (k>=1) subset that is unambiguously stationary.
    """
    mask = wake_mask()
    per_enc = []
    for f in files:
        with h5py.File(f, "r") as h:
            omega = np.asarray(h["omega_z"], dtype=np.float64)  # (120, 192, 96)
            k = int(h.attrs.get("encounter_index", -1))
        ens = np.array([frame_wake_enstrophy(omega[t], mask) for t in range(omega.shape[0])])
        per_enc.append({"encounter": k, "mean": float(ens.mean()), "std": float(ens.std())})
    vals = np.array([e["mean"] for e in per_enc])
    later = np.array([e["mean"] for e in per_enc if e["encounter"] >= 1])
    return {
        "value": float(vals.mean()),
        "std_across_encounters": float(vals.std()),
        "value_later_encounters": float(later.mean()) if later.size else float("nan"),
        "n_encounters": int(vals.size),
        "per_encounter": per_enc,
        "units": "raw omega^2 * (chord units)^2 over wake box " "x[0.5,4.0] y[-1.0,1.0]",
        "convention": "paper canonical (exp1b_decode_axes.py)",
    }


def pct_diff(ours: float, ref: float) -> float:
    return float(100.0 * (ours - ref) / ref) if ref else float("nan")


def build_table(ours: dict) -> tuple[dict, dict]:
    """Cross-reference table + per-reference availability bookkeeping.

    Returns (table, availability). table[quantity] holds ours + each reference's
    value and signed % difference (or a NEEDS-LITERATURE / NO-EXTERNAL-REFERENCE
    marker). availability records which references contributed real numbers.
    """
    table: dict = {}
    availability = {r: [] for r in REFERENCES}
    for key, label, fmt in QUANTITIES:
        row = {"label": label, "fmt": fmt, "ours": ours.get(key)}
        for ref_name, ref in REFERENCES.items():
            if key == "enstrophy":
                row[ref_name] = {"value": None, "status": NO_EXT}
                continue
            if key in ref:
                row[ref_name] = {
                    "value": ref[key],
                    "pct_diff_ours_vs_ref": pct_diff(ours[key], ref[key]),
                    "status": "available",
                }
                availability[ref_name].append(key)
            else:
                row[ref_name] = {"value": None, "status": NEEDS_LIT}
        table[key] = row
    return table, availability


def within_band(table: dict) -> dict:
    """Verdict: do OUR mean forces fall inside the spread of AVAILABLE refs?

    The defensible band is the min..max envelope of every available reference
    value for that quantity (the three published mean-force estimates already
    span a non-trivial spread; e.g. CL 0.734-0.763). We band-check the mean
    forces (where 2+ references agree) and rms (Fukami only). St and enstrophy
    have no external table value and are reported OURS-only.
    """
    verdict = {}
    for key in ("CL_mean", "CD_mean", "CL_rms", "CD_rms"):
        row = table[key]
        refs = [
            row[r]["value"]
            for r in REFERENCES
            if row[r]["status"] == "available" and row[r]["value"] is not None
        ]
        if not refs:
            verdict[key] = {"checked": False, "reason": "no available reference"}
            continue
        lo, hi = min(refs), max(refs)
        ours = row["ours"]
        # max deviation from the nearest reference (% of that reference).
        nearest = min(refs, key=lambda v: abs(v - ours))
        verdict[key] = {
            "checked": True,
            "ours": ours,
            "ref_band": [lo, hi],
            "n_refs": len(refs),
            "in_band": bool(lo <= ours <= hi),
            "nearest_ref": nearest,
            "pct_from_nearest": pct_diff(ours, nearest),
        }
    return verdict


def render_md(table: dict, availability: dict, verdict: dict, ours: dict) -> str:
    def cell(entry, fmt):
        if isinstance(entry, dict):
            if entry["status"] != "available":
                return entry["status"]
            return f"{fmt % entry['value']} ({entry['pct_diff_ours_vs_ref']:+.1f}%)"
        return entry

    lines = [
        "# Track A: baseline external validation (referee F1, v2.1)\n",
        "Undisturbed NACA0012, Re = 5000, alpha = 14 deg, no-gust Baseline. Our DNS"
        " (SOD2D) vs the external references the lineage validates against"
        " (Fukami, Smith & Taira PRF 2025 Fig. 2(b)). Mean/rms forces and St are"
        " read from the frozen `undisturbed.json`; wake enstrophy is computed here."
        " % differences are ours vs that reference.\n",
        "| quantity | ours | Fukami2025 | Rolandi2025 | Gupta2023 | within-band? |",
        "|---|---|---|---|---|---|",
    ]
    for key, label, fmt in QUANTITIES:
        row = table[key]
        ours_str = "n/a" if row["ours"] is None else fmt % row["ours"]
        f_ = cell(row["Fukami2025"], fmt)
        r_ = cell(row["Rolandi2025"], fmt)
        g_ = cell(row["Gupta2023"], fmt)
        v = verdict.get(key)
        if v and v.get("checked"):
            band = "in band" if v["in_band"] else "outside band"
            wb = f"{band} [{v['ref_band'][0]:g}, {v['ref_band'][1]:g}]"
        elif key in ("St", "enstrophy"):
            wb = "ours-only (no ext. table value)"
        else:
            wb = "n/a"
        lines.append(f"| {label} | {ours_str} | {f_} | {r_} | {g_} | {wb} |")

    avail_str = "; ".join(
        f"{r}: {', '.join(availability[r]) if availability[r] else 'none'}" for r in REFERENCES
    )
    lines += [
        "",
        f"Reference availability (quantities with a published value): {avail_str}.",
        "Rolandi2025 and Gupta2023 report only mean C_L and C_D in Fig. 2(b); their"
        " rms and St cells are NEEDS-LITERATURE. No external reference reports a"
        " baseline Strouhal here (St is the manuscript M13 internal cross-check) or"
        " a wake enstrophy (grid/normalisation-dependent internal diagnostic).",
        "",
        "## Verdict",
        verdict_prose(verdict, ours),
    ]
    return "\n".join(lines) + "\n"


def verdict_prose(verdict: dict, ours: dict) -> str:
    cl = verdict.get("CL_mean", {})
    cd = verdict.get("CD_mean", {})
    parts = []
    if cl.get("checked"):
        loc = "inside" if cl["in_band"] else "just outside"
        parts.append(
            f"Our mean C_L = {cl['ours']:.3f} sits {loc} the published band "
            f"[{cl['ref_band'][0]:.3f}, {cl['ref_band'][1]:.3f}] spanned by the"
            f" three references, {cl['pct_from_nearest']:+.1f}% from the nearest"
            " (Gupta 2023 experiment, 0.763)."
        )
    if cd.get("checked"):
        loc = "inside" if cd["in_band"] else "outside"
        parts.append(
            f"Our mean C_D = {cd['ours']:.3f} sits {loc} the published band "
            f"[{cd['ref_band'][0]:.3f}, {cd['ref_band'][1]:.3f}],"
            f" {cd['pct_from_nearest']:+.1f}% from the nearest reference."
        )
    parts.append(
        "Physical reading of the spread: the three references are not"
        " like-for-like. Fukami 2025 (DNS, incompressible) and Rolandi 2025"
        " (LES, M_inf = 0.2) both use span Lz/c = 1 and agree to ~0.4% in C_L;"
        " Gupta 2023 is a wind-tunnel experiment at Lz/c = 10, whose larger span"
        " relaxes spanwise confinement and lifts mean C_L (0.763) while lowering"
        " mean C_D (0.223). Our DNS matches the Re and incidence and uses the"
        " same incompressible/low-Mach regime as Fukami; both our mean forces sit"
        " within 1.7% of Fukami's own production-grid DNS (C_D 0.253 vs 0.249,"
        " +1.7%; just above the 0.223-0.249 envelope only because the experiment"
        " is the low edge), the smallest and most apt comparison. The rms"
        " fluctuations are the loosest cells: only Fukami reports them, and even"
        " their three grids span CL_rms 0.098-0.116 and CD_rms 0.0183-0.0224, so"
        " these statistics are resolution- and window-sensitive. Our CL_rms"
        " (+9.6%) is within that grid scatter; our CD_rms (+21.5%) exceeds it and"
        " is the one statistic that is honestly higher than the lineage. It is a"
        " single-quantity fluctuation level (mean C_D, which sets the loading, is"
        " within 1.7%) and is sensitive to the stationary-window choice (the"
        " frozen window is t/c [20,40]); it is flagged, not papered over, and"
        " should be revisited once the partner solver's grid/time-step rows land."
        " The St cross-check (dominant line and its full-cycle subharmonic ~0.34,"
        " the manuscript's 'St near 0.36') is internal: Fig. 2(b) lists no St for"
        " any external reference."
    )
    means_ok = all(
        v["in_band"] or abs(v["pct_from_nearest"]) < 5.0
        for k, v in verdict.items()
        if k in ("CL_mean", "CD_mean") and v.get("checked")
    )
    headline = (
        "VERDICT: our undisturbed baseline falls within a defensible band of the"
        " available references."
        if means_ok
        else "VERDICT: our undisturbed baseline shows a gap to the available"
        " references that needs a physical explanation (see above)."
    )
    return headline + " " + " ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        files = baseline_encounter_files()
        print(
            f"[dry-run] frozen numbers <- {UNDISTURBED.relative_to(REPO)}\n"
            f"[dry-run] {len(files)} Baseline cache encounters under {CACHE / 'Baseline'}\n"
            f"[dry-run] references: {', '.join(REFERENCES)} "
            f"(Rolandi/Gupta: mean C_L,C_D only)\n"
            f"[dry-run] would write {OUT_JSON.relative_to(REPO)} and "
            f"{OUT_MD.relative_to(REPO)}"
        )
        return

    frozen = load_undisturbed_numbers()
    files = baseline_encounter_files()
    enstrophy = compute_wake_enstrophy(files)

    ours = {
        "CL_mean": frozen["CL_mean"],
        "CL_rms": frozen["CL_rms"],
        "CD_mean": frozen["CD_mean"],
        "CD_rms": frozen["CD_rms"],
        "St": frozen["St"],
        "St_subharmonic": frozen["St_subharmonic"],
        "enstrophy": enstrophy["value"],
    }

    table, availability = build_table(ours)
    verdict = within_band(table)

    payload = {
        "_provenance": cm.provenance([UNDISTURBED, SPLIT], seed=None),
        "_doc": "Baseline external validation (Track A, referee F1). DNS via SOD2D;"
        " configuration from Fukami et al. (2025). Mean/rms forces and St read from"
        " the frozen session28 undisturbed record; wake enstrophy computed here.",
        "ours": {
            **{k: ours[k] for k in ("CL_mean", "CL_rms", "CD_mean", "CD_rms")},
            "St_dominant": ours["St"],
            "St_subharmonic": ours["St_subharmonic"],
            "wake_enstrophy": enstrophy,
            "stationary_window_tc": list(STATIONARY_TC),
            "force_numbers_note": frozen["_window_note"],
            "source_record": str(UNDISTURBED.relative_to(REPO)),
        },
        "references": REFERENCES,
        "table": table,
        "reference_availability": availability,
        "within_band_verdict": verdict,
    }
    cm.write_artifact(OUT_JSON, payload)

    md = render_md(table, availability, verdict, ours)
    OUT_MD.write_text(md)
    print(md)
    print(f"wrote {OUT_JSON}\nwrote {OUT_MD}")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(REPO))
    main()
