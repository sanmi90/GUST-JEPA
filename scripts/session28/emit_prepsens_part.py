"""Emit the preprocessing-sensitivity numbers part (SESSION29.4 sec 7 / Track B0.5).

Reads outputs/session29/preprocess_sensitivity_frozen.json (frozen-encoder
wake-enstrophy R^2 under three training-only clip treatments) and emits the
per-family-per-treatment R^2 macros plus the maximum advantage shift, so the
robustness appendix prints macro-bound numbers.

Usage:
    python scripts/session28/emit_prepsens_part.py
    python scripts/session28/eval_all.py
    python scripts/session28/emit_macros.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "outputs" / "session29" / "preprocess_sensitivity_frozen.json"
OUT = REPO / "outputs" / "session28" / "numbers_parts" / "prepsens.json"

TREAT = {"none": "None", "per_encounter": "PerEnc", "training_global": "Global"}
FAM = {"jepa_tf_noc": "Jepa", "fukami": "Fukami", "pod": "Pod"}


def main() -> None:
    d = json.loads(SRC.read_text())
    r2 = d["per_treatment_family_r2"]
    numbers = {}
    for tk, ts in TREAT.items():
        for fk, fs in FAM.items():
            macro = f"PrepSens{fs}{ts}"
            val = float(r2[tk][fk])
            if abs(val) < 0.005:  # avoid a "-0.00" rendering
                val = 0.0
            numbers[macro] = {
                "macro": macro,
                "value": val,
                "fmt": "%.2f",
                "source": "preprocess_sensitivity_frozen.py (Track B0.5)",
                "note": f"test_b wake-enstrophy R2, {fk}, {tk} clip",
            }
    # advantage = jepa - best baseline per treatment; max |shift| vs per_encounter
    adv = {tk: r2[tk]["jepa_tf_noc"] - max(r2[tk]["fukami"], r2[tk]["pod"]) for tk in TREAT}
    ref = adv["per_encounter"]
    max_shift = max(abs(adv[tk] - ref) for tk in TREAT)
    numbers["PrepSensMaxShift"] = {
        "macro": "PrepSensMaxShift",
        "value": float(max_shift),
        "fmt": "%.2f",
        "source": "preprocess_sensitivity_frozen.py (Track B0.5)",
        "note": "max |advantage shift| vs the per-encounter clip across treatments",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"part": "prepsens", "numbers": numbers}, indent=2, sort_keys=True))
    print(f"[prepsens] {len(numbers)} numbers -> {OUT}")


if __name__ == "__main__":
    main()
