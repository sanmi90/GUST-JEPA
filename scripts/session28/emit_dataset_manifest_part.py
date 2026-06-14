"""Emit the dataset-manifest counts part for the numbers chain (B1, SESSION29.4).

The case and encounter counts printed in the manuscript (Sec. 2 and the
parameter-space figure caption) must be ONE canonical set, generated from the
split manifest rather than hand-typed, so the v2 and v2.1 numbers cannot drift
apart in the prose. This reads the frozen `summary` block of
`configs/splits/split_v2p1.json` and writes a numbers_parts record with one
integer-valued macro per count.

Usage:
    python scripts/session28/emit_dataset_manifest_part.py
    python -m scripts.session28.eval_all      # then merge
    python scripts/session28/emit_macros.py    # then render macros.tex
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SPLIT = REPO / "configs" / "splits" / "split_v2p1.json"
OUT = REPO / "outputs" / "session28" / "numbers_parts" / "dataset_manifest.json"

# summary-key -> (macro name [alphabetic only], printf fmt)
MACROS = {
    "n_cases_total": ("NumCasesTotal", "%d"),
    "n_cases_train": ("NumCasesTrain", "%d"),
    "n_cases_test_b": ("NumCasesTestB", "%d"),
    "n_cases_test_c": ("NumCasesTestC", "%d"),
    "n_encounters_total_in_splits": ("NumEncTotal", "%d"),
    "n_encounters_train": ("NumEncTrain", "%d"),
    "n_encounters_val": ("NumEncVal", "%d"),
    "n_encounters_test_b": ("NumEncTestB", "%d"),
    "n_encounters_test_c": ("NumEncTestC", "%d"),
}


def main() -> None:
    summary = json.loads(SPLIT.read_text())["summary"]
    numbers = {}
    for key, (macro, fmt) in MACROS.items():
        numbers[macro] = {
            "macro": macro,
            "value": int(summary[key]),
            "fmt": fmt,
            "source": "split_v2p1.json::summary",
            "note": f"canonical v2.1 manifest count ({key})",
        }
    blob = {"part": "dataset_manifest", "numbers": numbers}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(blob, indent=2, sort_keys=True))
    print(f"[dataset_manifest] {len(numbers)} counts -> {OUT}")


if __name__ == "__main__":
    main()
