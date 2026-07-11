#!/usr/bin/env python3
"""Emit the two critical-instant numbers-pipeline parts from their source JSONs
(Carlos, 2026-07-12): makes the session39_critical (decode SSIM) and
session39_obscrit (observable readability) parts reproducible instead of ad-hoc,
and adds the lift-supervised predictive state (CLN) so both critical tables carry
the same five-family set as figure~\\ref{fig:decode}.

Sources:
  outputs/session39/multi_d_critical_ssim.json  (d=32 decode-floor SSIM per family)
  outputs/session39/observable_critical.json    (held-out C_L / E_w readability)
Targets:
  outputs/session33/numbers_parts/session39_critical.json
  outputs/session33/numbers_parts/session39_obscrit.json

Run (after the two compute scripts):
    .venv/bin/python scripts/session39/emit_critical_parts.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "outputs/session39"
PARTS = REPO / "outputs/session33/numbers_parts"

# family label (as written by the compute scripts) -> macro word; order fixes the
# row/table order and matches fig:decode (CLW, CLN, AE-LW, Fukami, POD).
LABEL2WORD = {
    "predictive": "Jepa",         # CLW, wake-supervised predictive
    "predictive (lift)": "Cln",   # CLN, lift-supervised predictive
    "AE (wake)": "Ae",
    "Fukami (wake)": "Fuk",
    "POD": "Pod",
}
INST2WORD = {"preimpact": "Pre", "impact": "Imp", "peaklift": "Peak"}


def emit_critical() -> None:
    """Decode-floor SSIM at d=32, full field + near-body band."""
    data = json.loads((SRC / "multi_d_critical_ssim.json").read_text())["families"]
    nums = {}
    for label, word in LABEL2WORD.items():
        rec = data.get(label, {}).get("32")
        if rec is None:
            continue
        for inst, iw in INST2WORD.items():
            nums[f"crit_{word}_{iw}"] = {
                "value": rec[f"{inst}_full"], "macro": f"Crit{word}{iw}", "fmt": "%.2f"}
            nums[f"critnb_{word}_{iw}"] = {
                "value": rec[f"{inst}_nearbody"], "macro": f"CritNb{word}{iw}", "fmt": "%.2f"}
    out = {"part": "session39_critical", "numbers": nums}
    (PARTS / "session39_critical.json").write_text(json.dumps(out, indent=1))
    print(f"[emit] session39_critical: {len(nums)} numbers, "
          f"{sum(1 for l in LABEL2WORD if l in data)} families")


def emit_obscrit() -> None:
    """Held-out linear-probe readability of C_L and E_w at the critical instants."""
    data = json.loads((SRC / "observable_critical.json").read_text())["families"]
    nums = {}
    read_sds = []  # seed SD over every cell that reads a real signal
    for label, word in LABEL2WORD.items():
        rec = data.get(label)
        if rec is None:
            continue
        for inst, iw in INST2WORD.items():
            nums[f"ocl_{word}_{iw}"] = {
                "value": rec[f"C_L_{inst}_r2"], "macro": f"CritCl{word}{iw}", "fmt": "%.2f"}
            nums[f"oclr_{word}_{iw}"] = {
                "value": rec[f"C_L_{inst}_rmse"], "macro": f"CritClRmse{word}{iw}", "fmt": "%.2f"}
            nums[f"oew_{word}_{iw}"] = {
                "value": rec[f"E_w_{inst}_r2"], "macro": f"CritEw{word}{iw}", "fmt": "%.2f"}
            nums[f"oewr_{word}_{iw}"] = {
                "value": rec[f"E_w_{inst}_rmse"], "macro": f"CritEwRmse{word}{iw}", "fmt": "%.0f"}
            read_sds.append(rec.get(f"C_L_{inst}_r2_sd", 0.0))
            if label != "predictive (lift)":  # CLN carries no wake; its E_w read is
                read_sds.append(rec.get(f"E_w_{inst}_r2_sd", 0.0))  # near zero + noisy
    # Blanket seed-spread bound over every cell that reads a real signal. The one
    # excluded cell, the lift-only CLN wake readability, is near zero with a large
    # seed spread by construction and is discussed in prose.
    nums["ocl_seedsd"] = {"value": max(read_sds), "macro": "CritSeedSd", "fmt": "%.2f"}
    out = {"part": "session39_obscrit", "numbers": nums}
    (PARTS / "session39_obscrit.json").write_text(json.dumps(out, indent=1))
    print(f"[emit] session39_obscrit: {len(nums)} numbers; CritSeedSd={max(read_sds):.2f}")


if __name__ == "__main__":
    emit_critical()
    emit_obscrit()
