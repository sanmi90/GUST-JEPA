"""Emit the v2.1 paired-closure table body (tab:paired_closure) from the B6 harvest.

The 12-test family (predictive jepa_tf_noc minus reconstructive Fukami, per
observable x endpoint, test_b H=16 d=64) lives in
outputs/session28/stats_harvest_full.json. Rather than hand-transcribe 12 rows,
this renders them straight to a LaTeX \\input so the table traces to the harvest.
Columns: mode, observable, case-clustered Delta err, case-clustered CI,
encounter-level sign test (k/n, p), encounter-level Holm p (the pre-registered
D165 family correction). Survivors (encounter-level Holm) are bolded.

    python scripts/session28/emit_paired_table.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "outputs/session28/stats_harvest_full.json"
OUT = REPO / "paper/sections/tables/paired_closure_body.tex"

OBS_LABEL = {
    "CL": "$C_L$",
    "CD": "$C_D$",
    "wake_enstrophy": "wake-enst.\\",
    "circ_pos": "circ.\\ pos.\\",
    "circ_neg": "circ.\\ neg.\\",
}
ORDER = ["CL", "CD", "wake_enstrophy", "circ_pos", "circ_neg"]


def fmt(x: float, nd: int = 3) -> str:
    return f"{x:+.{nd}g}" if abs(x) < 100 else f"{x:+.0f}"


def fmt_p(p: float) -> str:
    """LaTeX-formatted p value (math mode contents, no dollar signs)."""
    if p >= 1e-3:
        return f"{p:.2g}"
    mant, exp = f"{p:.1e}".split("e")
    return f"{mant}{{\\times}}10^{{{int(exp)}}}"


def prow(mode_label: str, cell: dict, bold: bool) -> str:
    d = cell["enc_mean_delta"]
    lo, hi = cell["case_clustered_ci"]
    k, n = cell["sign_k"], cell["sign_n_eff"]
    p = cell["sign_p_one_sided"]
    hp = cell["holm_enc_p"]
    dcell = f"${fmt(d)}$"
    hpcell = f"${fmt_p(hp)}$"
    if bold:
        dcell = f"$\\mathbf{{{fmt(d)}}}$"
        hpcell = f"$\\mathbf{{{fmt_p(hp)}}}$"
    return (
        f"    {mode_label} & {{obs}} & {dcell} & "
        f"$[{lo:+.3g}, {hi:+.3g}]$ & ${k}/{n}$, $p={fmt_p(p)}$ & {hpcell} \\\\"
    )


def main() -> None:
    h = json.loads(SRC.read_text())["holm"]
    tests = h["tests"]
    survivors = set(h["survivors_enc"])
    lines = []
    for mode, label in (("repr", "repr.\\"), ("forecast", "forecast")):
        for obs in ORDER:
            key = f"{mode}/{obs}"
            if key not in tests:
                continue
            cell = tests[key]
            bold = key in survivors
            row = prow(label, cell, bold).replace("{obs}", OBS_LABEL[obs])
            lines.append(row)
        if mode == "repr":
            lines.append("    \\midrule")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"[paired-table] wrote {len(tests)} rows -> {OUT}")
    print(f"[paired-table] encounter-level Holm survivors: {sorted(survivors)}")


if __name__ == "__main__":
    main()
