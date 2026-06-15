"""Emit the grouped-CV (Track C-full) numbers part (SESSION29.8 Track D).

Reads outputs/session29/cv_full.json (5-fold case-disjoint held-out wake R^2 for
the predictive latent and the matched reconstructive control) and emits the
median/min/max macros so the manuscript can report the grouped-CV distribution.

Usage:
    python scripts/session28/emit_cvfull_part.py
    python scripts/session28/eval_all.py
    python scripts/session28/emit_macros.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "outputs" / "session29" / "cv_full.json"
OUT = REPO / "outputs" / "session28" / "numbers_parts" / "cvfull.json"

FAM = {"jepa": "Jepa", "ctrl_recon": "Recon"}
STAT = {"median": "Median", "min": "Min", "max": "Max"}


def main() -> None:
    summary = json.loads(SRC.read_text())["summary"]
    numbers = {}
    for fk, fs in FAM.items():
        s = summary[fk]
        for stat, ss in STAT.items():
            macro = f"NumCvFull{fs}{ss}"
            numbers[macro] = {
                "macro": macro,
                "value": float(s[stat]),
                "fmt": "%.2f",
                "source": "group_case_cv_probe.py (Track C-full / D)",
                "note": f"5-fold grouped-CV held-out wake R2 {stat}, {fk}",
            }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"part": "cvfull", "numbers": numbers}, indent=2, sort_keys=True))
    print(f"[cvfull] {len(numbers)} numbers -> {OUT}")


if __name__ == "__main__":
    main()
