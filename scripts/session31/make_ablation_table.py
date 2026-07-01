"""Session 31 Track E: assemble the one-axis ablation table (point values).

Each ablation moves exactly ONE axis versus a canonical reference; this script
reads the ablation Q1/Q2 eval payloads plus the reference canonical payloads and
emits, per ablation, the lever metric and its delta versus the reference so the
"one axis moved" reading is explicit. CIs are NOT computed here (multi-seed /
bootstrap is a later pass); point values suffice for the ablation table.

Outputs (under outputs/session31/):
    numbers_ablation.json            -- value-only ledger (report_session31 macros)
    paper/macros_session31_ablation.tex  -- \\providecommand cells (no literals)
    tables/ablation_comparison.tex   -- booktabs table referencing the macros
    tables/ablation_comparison.md    -- readable markdown with ref + delta

    python scripts/session31/make_ablation_table.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.report_session31 import (  # noqa: E402
    ABLATION_AXIS,
    ABLATION_MODELS,
    ABLATION_REFERENCE,
    MODEL_LABEL,
    emit_macros,
    name_and_macro,
)

REPO = Path(__file__).resolve().parents[2]
S31 = REPO / "outputs" / "session31"


def _load(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _at(curve: dict, key: str, h: int) -> float:
    hs = list(curve.get("horizons", []))
    return float(curve[key][hs.index(h)]) if h in hs else float("nan")


def _q1_cells(q1: dict, model: str, window: str = "windowed") -> dict[str, float]:
    m = q1["models"][model]
    floor = m["decode_floor"][window]
    return {
        "floor_vrmse": float(floor["vrmse"]),
        "floor_ssim": float(floor["ssim"]),
        "cl_mlp_r2": float(m["probes"][window]["C_L"]["mlp_r2"]),
    }


def _q2_cells(q2: dict, model: str, h: int = 8) -> dict[str, float]:
    pred = q2["models"][model]["predictors"]["resunet_matched"]
    fv = pred["field_vrmse"]
    oc = pred["observable_closure"]
    targets = ("C_L", "C_D", "wake_enstrophy", "circulation_pos", "circulation_neg")
    merit = float(sum(_at(oc[t], "model_mlp_r2", h) for t in targets) / len(targets))
    return {
        "field_vrmse_model_h8": _at(fv, "model", h),
        "field_vrmse_floor_h8": _at(fv, "floor", h),
        "cl_closure_mlp_h8": _at(oc["C_L"], "model_mlp_r2", h),
        "merit_mean_obs_h8": merit,
    }


def _train_stats(run_dir: Path) -> dict[str, float | bool]:
    mj = run_dir / "metrics.jsonl"
    out: dict[str, float | bool] = {
        "final_loss": float("nan"),
        "final_pr": float("nan"),
        "final_r2": float("nan"),
        "fallback": False,
    }
    if not mj.exists():
        return out
    recs = [json.loads(line) for line in mj.read_text().splitlines() if line.strip()]
    logs = [r for r in recs if r.get("event") == "log"]
    diags = [r for r in logs if "diag/pr" in r]
    losses = [r for r in logs if "loss_total" in r]
    if losses:
        out["final_loss"] = float(losses[-1]["loss_total"])
    if diags:
        out["final_pr"] = float(diags[-1]["diag/pr"])
        out["final_r2"] = float(diags[-1]["diag/r2_overall"])
    log = run_dir / "train.log"
    if log.exists() and "AUTO-FALLBACK" in log.read_text():
        out["fallback"] = True
    return out


# Ablation table columns: (label, kind, name_and_macro kwargs, fmt, arrow).
_COLUMNS = [
    ("decode VRMSE", "repr_floor_vrmse", {}, "%.3f", "down"),
    ("decode SSIM", "repr_floor_ssim", {}, "%.3f", "up"),
    ("C_L closure R2 h8", "fore_clos", {"target": "C_L", "h": 8}, "%.2f", "up"),
    ("merit R2 h8", "fore_merit", {"h": 8}, "%.2f", "up"),
    ("field VRMSE h8", "fore_field", {"which": "model", "h": 8}, "%.3f", "down"),
]


def build(q1a: dict, q2a: dict, q1c: dict, q2c: dict, runs_base: Path) -> tuple[dict, list[dict]]:
    """Return (numbers-records, per-ablation rows with ref + delta)."""
    numbers: dict[str, dict] = {}
    rows: list[dict] = []

    def put(name, macro, value, fmt, **meta):
        numbers[name] = {"value": float(value), "macro": macro, "fmt": fmt, **meta}

    for model in ABLATION_MODELS:
        ref = ABLATION_REFERENCE[model]
        q1m = _q1_cells(q1a, model)
        q2m = _q2_cells(q2a, model)
        q1r = _q1_cells(q1c, ref)
        q2r = _q2_cells(q2c, ref)
        stats = _train_stats(runs_base / model)

        # emit the ablation's own point cells as macros (value-only ledger).
        n, mc = name_and_macro("repr_floor_vrmse", model)
        put(n, mc, q1m["floor_vrmse"], "%.3f", model=model, metric="decode_floor_vrmse")
        n, mc = name_and_macro("repr_floor_ssim", model)
        put(n, mc, q1m["floor_ssim"], "%.3f", model=model, metric="decode_floor_ssim")
        n, mc = name_and_macro("repr_mlp", model, target="C_L")
        put(n, mc, q1m["cl_mlp_r2"], "%.2f", model=model, metric="probe_mlp_r2")
        n, mc = name_and_macro("fore_clos", model, target="C_L", h=8)
        put(n, mc, q2m["cl_closure_mlp_h8"], "%.2f", model=model, metric="closure_mlp_r2")
        n, mc = name_and_macro("fore_merit", model, h=8)
        put(n, mc, q2m["merit_mean_obs_h8"], "%.2f", model=model, metric="merit")
        n, mc = name_and_macro("fore_field", model, which="model", h=8)
        put(n, mc, q2m["field_vrmse_model_h8"], "%.3f", model=model, metric="field_vrmse_model")

        vals = {**q1m, **q2m}
        refvals = {**q1r, **q2r}
        rows.append(
            {
                "model": model,
                "ref": ref,
                "axis": ABLATION_AXIS[model],
                "vals": vals,
                "refvals": refvals,
                "stats": stats,
            }
        )
    return numbers, rows


_KEY_BY_COL = {
    "decode VRMSE": "floor_vrmse",
    "decode SSIM": "floor_ssim",
    "C_L closure R2 h8": "cl_closure_mlp_h8",
    "merit R2 h8": "merit_mean_obs_h8",
    "field VRMSE h8": "field_vrmse_model_h8",
}


def markdown(rows: list[dict]) -> str:
    cols = list(_KEY_BY_COL)
    out = ["# Session 31 Track E -- one-axis ablation table\n"]
    out.append(
        "Each ablation moves ONE axis vs its reference. Cell = ablation (ref) "
        "[delta]. PR/loss/fallback from the 10k-iter training diagnostic.\n"
    )
    header = ["ablation", "axis", "ref"] + cols + ["PR", "loss", "fallback"]
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        cells = [MODEL_LABEL[r["model"]], r["axis"], MODEL_LABEL[r["ref"]]]
        for c in cols:
            k = _KEY_BY_COL[c]
            v = r["vals"][k]
            rv = r["refvals"][k]
            fmt = "%.3f" if "VRMSE" in c or "SSIM" in c else "%.2f"
            cells.append(f"{fmt % v} ({fmt % rv}) [{'%+.3f' % (v - rv)}]")
        s = r["stats"]
        cells += [
            f"{s['final_pr']:.1f}",
            f"{s['final_loss']:.3f}",
            "YES" if s["fallback"] else "no",
        ]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def latex(numbers: dict) -> str:
    lines = [
        "% AUTO-GENERATED by scripts/session31/make_ablation_table.py. DO NOT EDIT.",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"ablation & decode VRMSE & SSIM & $R^2(C_L)_{h8}$ & "
        r"merit$_{h8}$ & field VRMSE$_{h8}$ \\",
        r"\midrule",
    ]
    for model in ABLATION_MODELS:
        macros = []
        for _lab, kind, kw, _fmt, _arrow in _COLUMNS:
            _n, mc = name_and_macro(kind, model, **kw)
            macros.append(f"$\\{mc}$")
        lines.append(f"{MODEL_LABEL[model]} & " + " & ".join(macros) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--q1-ablation", default=str(S31 / "q1_ablation.json"))
    p.add_argument("--q2-ablation", default=str(S31 / "q2_ablation.json"))
    p.add_argument("--q1-canonical", default=str(S31 / "q1_representation.json"))
    p.add_argument("--q2-canonical", default=str(S31 / "q2_temporal.json"))
    p.add_argument("--runs-base", default=str(REPO / "outputs" / "runs" / "session31"))
    args = p.parse_args()

    q1a, q2a = _load(Path(args.q1_ablation)), _load(Path(args.q2_ablation))
    q1c, q2c = _load(Path(args.q1_canonical)), _load(Path(args.q2_canonical))
    numbers, rows = build(q1a, q2a, q1c, q2c, Path(args.runs_base))

    (S31 / "numbers_ablation.json").write_text(json.dumps({"numbers": numbers}, indent=2))
    macros = emit_macros(
        numbers, header=["% Session 31 Track E ablation cells (point values, no CI)."]
    )
    (REPO / "paper" / "macros_session31_ablation.tex").write_text(macros)
    tables = S31 / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "ablation_comparison.tex").write_text(latex(numbers))
    md = markdown(rows)
    (tables / "ablation_comparison.md").write_text(md)
    print(md)
    print(f"[ablation-table] wrote {S31/'numbers_ablation.json'}")
    print(f"[ablation-table] wrote {REPO/'paper'/'macros_session31_ablation.tex'}")
    print(f"[ablation-table] wrote {tables/'ablation_comparison.tex'} and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
