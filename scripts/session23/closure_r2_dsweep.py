"""Closure R^2 + MAE for the session23 d-sweep encoders (d=8, d=4, no-lift d=64).

Reuses scripts/session20/exp_closure_r2.py (the canonical probe + bootstrap that
produced outputs/session23_closure/closure_r2_dimsweep_d16.csv) UNCHANGED, so the
new rows are directly comparable to the d16/d32/d64 rows already in that file.
Reads latents_{tag} from exp_b1/ and rollouts_{tag} from exp_b1_test3/.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "session20"))
import exp_closure_r2 as cr  # noqa: E402

NEW = [
    ("jepa_d8_noBN", "jepa", 8),
    ("jepa_d4_noBN", "jepa", 4),
    ("jepa_d64_nolift_noBN", "jepa", 64),
]
OUT = REPO / "outputs" / "session23_closure" / "closure_r2_dsweep.csv"


def main() -> None:
    dns = np.load(cr.DNS_METRICS_PATH, allow_pickle=True)
    horizons = (1, 4, 8, 16, 32, 64)
    splits = ["test_b", "test_c"]
    rows: list[dict] = []
    for tag, kind, d in NEW:
        rows += cr.evaluate(tag, kind, d, dns, horizons, splits, n_boot=2000)
    if not rows:
        raise SystemExit("no rows produced; check latents/rollouts for the new tags")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[closure_r2_dsweep] wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
