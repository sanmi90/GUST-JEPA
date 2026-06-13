"""Shared helpers for the Session 28 v2.1 manuscript figure regeneration.

Every figure under ``scripts/session28/figs/`` reuses the locked JFM figure style
(``scripts/session21/figstyle.py``: fixed four-colour family key, ``vort_panel``
Fig-3 vorticity convention, the measured 360 pt textwidth) and reads its
load-bearing numbers from ``outputs/session28/numbers.json`` so the figures stay
byte-faithful to the macro-wired text. Output PDFs go to
``paper/sections/figures/results/<basename>_v2p1.pdf``.

CPU only; numpy / scipy / matplotlib. No GPU, no W&B, no training.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))  # so `from src.data... import ...` resolves
sys.path.insert(0, str(REPO / "scripts" / "session21"))
sys.path.insert(0, str(REPO / "scripts" / "session28"))

import figstyle as fs  # noqa: E402,F401  (re-exported as cm.fs for every figure)

OUT_DIR = REPO / "paper" / "sections" / "figures" / "results"
NUMBERS_JSON = REPO / "outputs" / "session28" / "numbers.json"
SESSION28 = REPO / "outputs" / "session28"


def load_numbers() -> dict[str, dict]:
    """Return the v2.1 numbers keyed by their LaTeX macro name.

    Each value is the full record (value, fmt, ci_lo/ci_hi, source, ...). This is
    the same source ``emit_macros.py`` renders into ``paper/macros.tex``, so any
    number a figure prints matches the manuscript text exactly.
    """
    blob = json.loads(NUMBERS_JSON.read_text())["numbers"]
    return {v["macro"]: v for v in blob.values() if isinstance(v, dict) and "macro" in v}


def macro_value(numbers: dict[str, dict], macro: str) -> float:
    """The float value bound to ``macro`` (raises KeyError if absent)."""
    return float(numbers[macro]["value"])


def out_path(basename: str) -> Path:
    """``paper/sections/figures/results/<basename>_v2p1.pdf`` (parent ensured)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR / f"{basename}_v2p1.pdf"
