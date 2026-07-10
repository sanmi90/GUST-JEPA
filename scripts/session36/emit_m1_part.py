"""Emit the Track M1 shared-operator merit part for the numbers pipeline.

Reads outputs/session36/m1_shared_merit.json (rex_families_m1.py) and the
native co-trained-predictor closure (outputs/session33/q2_vec_native.json)
and writes outputs/session33/numbers_parts/m1_shared_merit.json in the
part schema consumed by scripts/session33/eval_all_v3.py.

Macros: XmeritSh<Name> (+lo/hi) = shared-operator merit at H=16 (the
tab:closure column, pre-registered horizon); XmeritShEight<Name> (+lo/hi)
= the same at H=8 (the horizon of the superseded suited-operator column);
XmeritJepaWakeNativeSixteen = the co-trained vector predictor's merit at
H=16 (same source and convention as the existing h8 value
XmeritJepaWakeNative in part table_x).

Run: python -m scripts.session36.emit_m1_part
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
M1 = REPO / "outputs/session36/m1_shared_merit.json"
NATIVE = REPO / "outputs/session33/q2_vec_native.json"
OUT = REPO / "outputs/session33/numbers_parts/m1_shared_merit.json"

OBS = ("C_L", "C_D", "wake_enstrophy", "circulation_pos", "circulation_neg")


def main() -> int:
    m1 = json.loads(M1.read_text())
    numbers: dict = {}
    note = ("shared direct multi-horizon forecaster (s3.2.1 operator, LSTM h512 "
            "9q), 3 operator seeds, case-clustered bootstrap CI; Track M1 D310")
    for fam, rec in m1["families"].items():
        for h, prefix in ((16, "XmeritSh"), (8, "XmeritShEight")):
            numbers[f"m1_merit_shared_h{h}_{fam}"] = {
                "value": rec[f"merit_h{h}_mean"],
                "ci_lo": rec[f"merit_h{h}_ci_lo"],
                "ci_hi": rec[f"merit_h{h}_ci_hi"],
                "macro": f"{prefix}{fam}",
                "fmt": "%.3f",
                "split": "test_b",
                "horizon": h,
                "seed_sd": rec[f"merit_h{h}_sd"],
                "note": note,
            }
    q = json.loads(NATIVE.read_text())
    oc = q["models"]["jepa_pool_vec"]["predictors"]["native"]["observable_closure"]
    i = list(oc["C_L"]["horizons"]).index(16)
    native16 = float(np.mean([oc[t]["model_mlp_r2"][i] for t in OBS]))
    numbers["m1_merit_native_h16"] = {
        "value": native16,
        "macro": "XmeritJepaWakeNativeSixteen",
        "fmt": "%.3f",
        "split": "test_b",
        "horizon": 16,
        "note": "co-trained vector predictor, H=16 (same source as the h8 "
                "XmeritJepaWakeNative in part table_x)",
    }
    OUT.write_text(json.dumps({"part": "m1_shared_merit", "numbers": numbers}, indent=1))
    print(f"[emit_m1_part] {len(numbers)} numbers -> {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
