"""Session 31 references: assemble the published-baseline comparison table.

The references (fukami, fukami_wake, pod) use their OWN architectures and are
evaluated through the SAME frozen probes as the shared-backbone spine, so their
Q1/Q2/Q3 payloads have the identical schema. This script reads the reference
payloads (``q*_reference.json``) plus the canonical spine payloads (for the
one-line read against jepa_wake / ae_nowake) and emits, per reference:

    outputs/session31/numbers_reference.json            -- value-only ledger
    paper/macros_session31_reference.tex                -- \\providecommand cells
    outputs/session31/tables/reference_comparison.tex   -- booktabs table (macros)
    outputs/session31/tables/reference_comparison.md    -- readable markdown

CIs are point-value only (references need no CI per the plan); the canonical spine
anchors are shown in the markdown for context, not emitted as reference macros.

    python scripts/session31/make_reference_table.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.report_session31 import (  # noqa: E402
    MODEL_LABEL,
    REFERENCE_MODELS,
    REFERENCE_NOTE,
    emit_macros,
    name_and_macro,
)

REPO = Path(__file__).resolve().parents[2]
S31 = REPO / "outputs" / "session31"

# Canonical anchors shown alongside the references for the one-line read.
ANCHORS = ("jepa_wake", "ae_nowake", "regAE")
TARGETS = ("C_L", "C_D", "wake_enstrophy", "circulation_pos", "circulation_neg")


def _load(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _at(curve: dict, key: str, h: int) -> float:
    hs = list(curve.get("horizons", []))
    return float(curve[key][hs.index(h)]) if h in hs else float("nan")


def _q1_cells(q1: dict, model: str, window: str = "windowed") -> dict[str, float]:
    m = q1["models"][model]
    floor = m["decode_floor"][window]
    pr = m["probes"][window]
    return {
        "floor_vrmse": float(floor["vrmse"]),
        "floor_ssim": float(floor["ssim"]),
        "cl_mlp_r2": float(pr["C_L"]["mlp_r2"]),
        "wakeens_mlp_r2": float(pr["wake_enstrophy"]["mlp_r2"]),
    }


def _q2_cells(q2: dict, model: str, h: int = 8) -> dict[str, float]:
    pred = q2["models"][model]["predictors"]["resunet_matched"]
    fv = pred["field_vrmse"]
    oc = pred["observable_closure"]
    merit = float(sum(_at(oc[t], "model_mlp_r2", h) for t in TARGETS) / len(TARGETS))
    return {
        "field_vrmse_model_h8": _at(fv, "model", h),
        "field_vrmse_floor_h8": _at(fv, "floor", h),
        "cl_closure_mlp_h8": _at(oc["C_L"], "model_mlp_r2", h),
        "merit_mean_obs_h8": merit,
    }


def _q3_cells(q3: dict, model: str) -> dict[str, float]:
    m = q3["models"].get(model, {})
    obs = m.get("pressure_to_obs")
    out = {"press_obs_mean_r2": float("nan"), "press_field_vrmse": float("nan")}
    if obs:
        per = obs["per_observable"]
        out["press_obs_mean_r2"] = float(
            sum(per[t]["pressure_linear_r2"] for t in TARGETS) / len(TARGETS)
        )
    fld = m.get("pressure_to_field")
    if fld:
        out["press_field_vrmse"] = float(fld["pressure_vrmse"])
    return out


def _cells(q1, q2, q3, model) -> dict[str, float]:
    return {**_q1_cells(q1, model), **_q2_cells(q2, model), **_q3_cells(q3, model)}


def build(q1r, q2r, q3r) -> tuple[dict, dict]:
    """Return (numbers-records for the reference macros, per-model cell dict)."""
    numbers: dict[str, dict] = {}
    cells: dict[str, dict] = {}

    def put(name, macro, value, fmt, **meta):
        numbers[name] = {"value": float(value), "macro": macro, "fmt": fmt, **meta}

    for model in REFERENCE_MODELS:
        c = _cells(q1r, q2r, q3r, model)
        cells[model] = c
        n, mc = name_and_macro("repr_floor_vrmse", model)
        put(n, mc, c["floor_vrmse"], "%.3f", model=model, metric="decode_floor_vrmse")
        n, mc = name_and_macro("repr_floor_ssim", model)
        put(n, mc, c["floor_ssim"], "%.3f", model=model, metric="decode_floor_ssim")
        n, mc = name_and_macro("repr_mlp", model, target="C_L")
        put(n, mc, c["cl_mlp_r2"], "%.2f", model=model, metric="probe_mlp_r2")
        n, mc = name_and_macro("repr_mlp", model, target="wake_enstrophy")
        put(n, mc, c["wakeens_mlp_r2"], "%.2f", model=model, metric="probe_mlp_r2")
        n, mc = name_and_macro("fore_clos", model, target="C_L", h=8)
        put(n, mc, c["cl_closure_mlp_h8"], "%.2f", model=model, metric="closure_mlp_r2")
        n, mc = name_and_macro("fore_merit", model, h=8)
        put(n, mc, c["merit_mean_obs_h8"], "%.2f", model=model, metric="merit")
        n, mc = name_and_macro("fore_field", model, which="model", h=8)
        put(n, mc, c["field_vrmse_model_h8"], "%.3f", model=model, metric="field_vrmse_model")
        n, mc = name_and_macro("pres_obs_mean", model)
        put(n, mc, c["press_obs_mean_r2"], "%.2f", model=model, metric="pressure_obs_mean_r2")
    return numbers, cells


_COLS = [
    ("decode VRMSE", "floor_vrmse", "%.3f"),
    ("decode SSIM", "floor_ssim", "%.3f"),
    ("C_L probe R2", "cl_mlp_r2", "%.2f"),
    ("C_L closure R2 h8", "cl_closure_mlp_h8", "%.2f"),
    ("merit R2 h8", "merit_mean_obs_h8", "%.2f"),
    ("field VRMSE h8", "field_vrmse_model_h8", "%.3f"),
    ("press->obs R2", "press_obs_mean_r2", "%.2f"),
]


def markdown(ref_cells: dict, anchor_cells: dict) -> str:
    out = ["# Session 31 references -- published-baseline comparison (Test B, in-window)\n"]
    out.append(
        "References use their OWN architectures (fukami: native lift-augmented CNN AE; "
        "pod: linear basis) read through the SAME frozen probes as the spine. Canonical "
        "anchors (jepa_wake, ae_nowake, regAE) shown for the one-line read.\n"
    )
    header = ["model", "note"] + [c[0] for c in _COLS]
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "|".join(["---"] * len(header)) + "|")
    for model in REFERENCE_MODELS:
        c = ref_cells[model]
        row = [MODEL_LABEL[model], REFERENCE_NOTE[model]]
        row += [fmt % c[key] for (_lab, key, fmt) in _COLS]
        out.append("| " + " | ".join(row) + " |")
    out.append("| | | | | | | | |")
    for model in anchor_cells:
        c = anchor_cells[model]
        row = [MODEL_LABEL[model], "canonical spine (anchor)"]
        row += [fmt % c[key] for (_lab, key, fmt) in _COLS]
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out) + "\n"


def latex(numbers: dict) -> str:
    cols_kinds = [
        ("repr_floor_vrmse", {}),
        ("repr_floor_ssim", {}),
        ("repr_mlp", {"target": "C_L"}),
        ("fore_clos", {"target": "C_L", "h": 8}),
        ("fore_merit", {"h": 8}),
        ("fore_field", {"which": "model", "h": 8}),
        ("pres_obs_mean", {}),
    ]
    lines = [
        "% AUTO-GENERATED by scripts/session31/make_reference_table.py. DO NOT EDIT.",
        "% Requires \\input{macros_session31_reference.tex}.",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"reference & decode VRMSE & SSIM & $R^2(C_L)$ & $R^2(C_L)_{h8}$ & "
        r"merit$_{h8}$ & field VRMSE$_{h8}$ & press$\to$obs $\bar{R}^2$ \\",
        r"\midrule",
    ]
    for model in REFERENCE_MODELS:
        macros = []
        for kind, kw in cols_kinds:
            _n, mc = name_and_macro(kind, model, **kw)
            macros.append(f"$\\{mc}$")
        lines.append(f"{MODEL_LABEL[model]} & " + " & ".join(macros) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--q1-reference", default=str(S31 / "q1_reference.json"))
    p.add_argument("--q2-reference", default=str(S31 / "q2_reference.json"))
    p.add_argument("--q3-reference", default=str(S31 / "q3_reference.json"))
    p.add_argument("--q1-canonical", default=str(S31 / "q1_representation.json"))
    p.add_argument("--q2-canonical", default=str(S31 / "q2_temporal.json"))
    p.add_argument("--q3-canonical", default=str(S31 / "q3_pressure.json"))
    args = p.parse_args()

    q1r, q2r, q3r = (
        _load(Path(args.q1_reference)),
        _load(Path(args.q2_reference)),
        _load(Path(args.q3_reference)),
    )
    q1c, q2c, q3c = (
        _load(Path(args.q1_canonical)),
        _load(Path(args.q2_canonical)),
        _load(Path(args.q3_canonical)),
    )

    numbers, ref_cells = build(q1r, q2r, q3r)
    anchor_cells = {m: _cells(q1c, q2c, q3c, m) for m in ANCHORS if m in q1c["models"]}

    (S31 / "numbers_reference.json").write_text(json.dumps({"numbers": numbers}, indent=2))
    macros = emit_macros(
        numbers, header=["% Session 31 reference-baseline cells (point values, no CI)."]
    )
    (REPO / "paper" / "macros_session31_reference.tex").write_text(macros)
    tables = S31 / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "reference_comparison.tex").write_text(latex(numbers))
    md = markdown(ref_cells, anchor_cells)
    (tables / "reference_comparison.md").write_text(md)
    print(md)
    print(f"[reference-table] wrote {S31 / 'numbers_reference.json'}")
    print(f"[reference-table] wrote {REPO / 'paper' / 'macros_session31_reference.tex'}")
    print(f"[reference-table] wrote {tables / 'reference_comparison.tex'} and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
