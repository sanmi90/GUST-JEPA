"""Emit the Track C conditioning-ablation numbers part (Session 34, D2).

Harvests the Track C eval outputs (gates, lift, region SSIM, closure, QC)
into ``outputs/session33/numbers_parts/trackc.json`` following the session28
accretion schema, so ``eval_all_v3.py`` merges + validates it (including the
v2.1 macro-collision check) and ``emit_macros_v3.py`` renders the macros.

Skips gracefully (warning, nonzero part count unchanged) while eval outputs
are still missing; re-run after the eval chain completes.

Run (CPU):
    python -m scripts.session34.emit_trackc_parts
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

PARTS = REPO / "outputs" / "session33" / "numbers_parts"
S34 = REPO / "outputs" / "session34"

CELL_WORD = {
    "c0": "CZero", "cl": "CL", "cw": "CW", "cn": "CN", "clw": "CLW",
    "cln": "CLN", "cwn": "CWN", "clwn": "CLWN",
    "ae_l": "AeL", "ae_w": "AeW", "ae_lw": "AeLW",
}


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def write_part(name: str, numbers: dict) -> None:
    PARTS.mkdir(parents=True, exist_ok=True)
    (PARTS / f"{name}.json").write_text(
        json.dumps({"part": name, "numbers": numbers}, indent=2, default=float)
    )
    n_mac = sum(1 for r in numbers.values() if r.get("macro"))
    print(f"[parts] {name}: {len(numbers)} numbers ({n_mac} macro-bound)")


def rec(value, macro=None, fmt="%.2f", **kw):
    r = {"value": value}
    if macro:
        r["macro"] = macro
    r["fmt"] = fmt
    r.update(kw)
    return r


def main() -> int:
    numbers: dict = {}

    # ---- precompute QC / D254 -------------------------------------------------
    prevent = Path(os.environ.get("PREVENT_ROOT", "/home/carlos/PREVENT"))
    cache = Path(os.environ.get(
        "VORTEX_JEPA_CACHE", prevent / "data" / "processed" / "vortex-jepa"))
    man = _load(cache / "v2p2" / "nearbody_observables" / "_manifest.json")
    if man:
        numbers["nb_qc_median_corr"] = rec(
            man["qc"]["median_abs_lagged_corr_gust_train"], "NbQcMedianCorr", "%.2f",
            source="nearbody _manifest.json")
        numbers["nb_delta_n"] = rec(man["delta_n"], "NbDeltaN", "%.1f")
        numbers["nb_phi_residual"] = rec(
            man["phi_L_residual_linf"], "NbPhiResidual", "%.1e")
        if man.get("d254_proxy_vs_chang"):
            cmean = float(np.mean(
                [r["cosine_mean"] for r in man["d254_proxy_vs_chang"]]))
            numbers["nb_d254_proxy_cosine"] = rec(
                cmean, "NbProxyCosine", "%.2f", source="D254 comparison")

    # ---- lift metrics per cell --------------------------------------------------
    lift = _load(S34 / "trackc_lift.json")
    if lift:
        for cell, seeds in lift["results"].items():
            word = CELL_WORD.get(cell, cell)
            r2s = [seeds[s]["linear"]["pooled_peak_r2"] for s in seeds]
            lags = [seeds[s]["linear"]["median_phase_lag_tc"] for s in seeds]
            numbers[f"tc_peak_r2_{cell}"] = rec(
                float(np.mean(r2s)), f"TcPeakRTwo{word}", "%.2f",
                seed_mean=float(np.mean(r2s)), seed_sd=float(np.std(r2s)),
                n=len(r2s))
            # Session 39: bind the 3-seed SD so the peak-lift band is quotable
            # per cell (Carlos: the predictive/JEPA lift deviation was not shown).
            numbers[f"tc_peak_r2_sd_{cell}"] = rec(
                float(np.std(r2s)), f"TcPeakRTwo{word}Sd", "%.3f")
            numbers[f"tc_lag_{cell}"] = rec(
                float(np.median(lags)), f"TcLag{word}", "%.3f")
    else:
        print("[parts] WARNING: trackc_lift.json missing; lift numbers skipped")

    # ---- region SSIM per cell (s0) ----------------------------------------------
    ssim = _load(S34 / "trackc_region_ssim.json")
    if ssim:
        for cell, blob in ssim["results"].items():
            word = CELL_WORD.get(cell, cell)
            numbers[f"tc_ssim_nb_{cell}"] = rec(
                blob["ssim"]["nearbody"], f"TcSsimNb{word}", "%.3f")
    else:
        print("[parts] WARNING: trackc_region_ssim.json missing; SSIM numbers skipped")

    # ---- filter per cell (s0) ------------------------------------------------------
    tuning = _load(S34 / "filter_tuning_trackc.json")
    for cell, word in CELL_WORD.items():
        env = _load(S34 / f"envelope_trackc_{cell}.json")
        if not env:
            continue
        from scripts.session34.trackc_cells import CELLS
        model = env["models"].get(CELLS[cell][0])
        if not model:
            continue
        recs = [r for r in model["records"] if r.get("split") == "test_b"]
        r2s = [r["filter"]["CL_analysis_r2_impact"] for r in recs]
        numbers[f"tc_filter_cl_r2_{cell}"] = rec(
            float(np.median(r2s)), f"TcFilterClRTwo{word}", "%.2f",
            n=len(recs), note="median over test_b encounters")
    # ---- participation ratio per cell (final diag/pr, seed mean) ----------------
    # T6.1 fix (Session 39, paper_redesign.md 2.1): the TcRho* macros previously
    # bound the EnKF covariance-inflation factor
    # (filter_tuning_trackc.json[run]["rho"] ~ 1.0), NOT the participation ratio
    # PR(z) the prose calls them. The real PR is the final diag/pr from each
    # cell's training run, exactly as fig_cube_health_v4.py reads it, so text and
    # figure now agree. Mean over the three seeds; collapse floor is 0.3 d = 9.6.
    from scripts.session34.trackc_cells import CELLS, RUNS_BASE

    def _final_pr(run: str):
        p = RUNS_BASE / run / "metrics.jsonl"
        if not p.exists():
            return None
        last = None
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if "diag/pr" in r:
                last = r["diag/pr"]
        return last

    for cell, word in CELL_WORD.items():
        seedmap = CELLS.get(cell)
        if not seedmap:
            continue
        prs = [v for v in (_final_pr(rn) for rn in seedmap.values())
               if v is not None]
        if prs:
            numbers[f"tc_rho_{cell}"] = rec(
                float(np.mean(prs)), f"TcRho{word}", "%.2f",
                seed_mean=float(np.mean(prs)), seed_sd=float(np.std(prs)),
                n=len(prs),
                source="final diag/pr from run metrics (PR floor 0.3 d = 9.6)")

    # ---- gate verdicts ---------------------------------------------------------------
    gates = _load(S34 / "trackc_gates.json")
    if gates:
        numbers["tc_q2_verdict"] = rec(
            gates["Q2_D255"]["verdict"], "TcQTwoVerdict", "%s", note="text verdict")
        numbers["tc_q2alt_verdict"] = rec(
            gates["Q2alt_D256"]["verdict"], "TcQTwoAltVerdict", "%s", note="text verdict")
        numbers["tc_q1_verdict"] = rec(
            gates["Q1_D257"]["verdict"], "TcQOneVerdict", "%s", note="text verdict")
        numbers["tc_pr_violations"] = rec(
            len(gates["pr_violations"]), "TcPrViolations", "%d")
    else:
        print("[parts] WARNING: trackc_gates.json missing; verdicts skipped")

    write_part("trackc", numbers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
