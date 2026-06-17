"""Emit the per-observable closure + parametric-floor numbers part for Phase D.

The closure_headline part bound only the wake-enstrophy headline per family;
the S4 prose also quotes per-observable closure (C_L, C_D, I_y, both
circulations) for jepa_tf_noc and the baselines, the forecast endpoint, and
the per-observable parametric floor. This pulls those from the committed
matrix.csv (seed-mean at H=16, test_b pooled, d=64, ridge) and
parametric_floor/results.json, and writes one eval_all part so every S4 number
is macro-bound (the "no hand-typed numbers" mandate). CPU, idempotent.

    python scripts/session28/emit_closure_floor_part.py
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MATRIX = REPO / "outputs/session28/closure_matrix/matrix.csv"
FLOOR = REPO / "outputs/session28/parametric_floor/results.json"
OUT = REPO / "outputs/session28/numbers_parts/closure_floor_full.json"

# observable -> macro stem
OBS = {
    "C_L": "CL",
    "C_D": "CD",
    "wake_enstrophy": "Wake",
    "circulation_pos": "CircPos",
    "circulation_neg": "CircNeg",
}
# family -> macro stem (the three reported comparators)
FAM = {"jepa_tf_noc": "JepaTf", "fukami": "Fukami", "pod": "Pod"}


def seed_mean(rows, family, observable, endpoint):
    vals = [
        float(r["value"])
        for r in rows
        if r["family"] == family
        and r["observable"] == observable
        and r["endpoint"] == endpoint
        and r["split"] == "test_b"
        and r["tier"] == "pooled"
        and r["H"] == "16"
        and r["probe"] == "ridge"
        and r["d"] == "64"
    ]
    return statistics.mean(vals) if vals else None


def main() -> None:
    rows = list(csv.DictReader(open(MATRIX)))
    numbers = {}

    for obs, ostem in OBS.items():
        for fam, fstem in FAM.items():
            for endpoint, estem in (("representational", "Repr"), ("forecast", "Fcst")):
                v = seed_mean(rows, fam, obs, endpoint)
                if v is None:
                    continue
                name = f"clo_{estem.lower()}_{ostem.lower()}_{fstem.lower()}"
                numbers[name] = {
                    "macro": f"NumClo{estem}{ostem}{fstem}",
                    "value": round(v, 2),
                    "fmt": "%.2f",
                    "split": "test_b",
                    "source": "emit_closure_floor_part.py (matrix.csv seed-mean)",
                }
    # capacity ladder: jepa_tf_noc representational wake at each d (seed mean).
    # Macro names must be alphabetic (LaTeX), so spell the dimension.
    dword = {"4": "Four", "8": "Eight", "16": "Sixteen", "32": "Thirtytwo", "64": "Sixtyfour"}
    for dval in ("4", "8", "16", "32", "64"):
        vals = [
            float(r["value"])
            for r in rows
            if r["family"] == "jepa_tf_noc"
            and r["observable"] == "wake_enstrophy"
            and r["endpoint"] == "representational"
            and r["split"] == "test_b"
            and r["tier"] == "pooled"
            and r["H"] == "16"
            and r["probe"] == "ridge"
            and r["d"] == dval
        ]
        if vals:
            numbers[f"clo_repr_wake_jepatf_d{dval}"] = {
                "macro": f"NumCloReprWakeJepaTfD{dword[dval]}",
                "value": round(statistics.mean(vals), 2),
                "fmt": "%.2f",
                "split": "test_b",
                "source": "emit_closure_floor_part.py (capacity ladder)",
            }

    # (the mean-over-observables aggregate was dropped with I_y: a mean R^2 over
    # heterogeneous observables was dominated by the one weak outlier, so it is
    # not reported. See the I_y removal.)

    # parametric floor per observable (ridge), test_b
    floor = json.load(open(FLOOR))["floor"]
    for obs, ostem in OBS.items():
        if obs in floor and "ridge" in floor[obs]:
            numbers[f"floor_{ostem.lower()}"] = {
                "macro": f"NumFloor{ostem}",
                "value": round(floor[obs]["ridge"]["value"], 2),
                "fmt": "%.2f",
                "split": "test_b",
                "source": "emit_closure_floor_part.py (parametric_floor ridge)",
            }

    part = {"part": "closure_floor_full", "numbers": numbers}
    OUT.write_text(json.dumps(part, indent=2))
    print(f"[closure-floor] wrote {len(numbers)} numbers -> {OUT}")


if __name__ == "__main__":
    main()
