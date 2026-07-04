"""Track C acceptance gates: Q2, Q2-alt, Q1 (Session 34) -- PRE-REGISTERED.

This docstring is the pre-registration. It was written BEFORE any Track C
result was read (only file schemas were inspected); the thresholds and branch
definitions below may not be changed after results exist.

Tolerated lift phase lag: tau_thresh = 0.1 convective time (2 frames at
dt = 0.05). Justification: the trailing-edge-to-wake-box convection time is
O(0.5) t/c at U_inf = 1 (wake box starts at the TE), so a wake-conditioned
latent that only carries lagged shed-circulation history would show lags of
that order; 0.1 t/c is 5x stricter and still coarser than the sub-frame
resolution of the estimator.

Primary readout: the LINEAR (RidgeCV) frozen probe (represent.py convention);
the MLP probe is reported as a sensitivity check and does not enter the gate
branches. Seeds: lift metrics use the per-encounter MEAN over seeds 0-2;
region SSIM and the filter use s0 only (decoder / filter compute policy).

Paired statistics: per-encounter deltas on test_b aligned on the canonical
(case_id, encounter_index) key; case-clustered bootstrap
(scripts/session28/stats_lib.case_cluster_bootstrap, case-mean CI) throughout.
"A >= B within the paired CI" means the case-mean 95% CI of (A - B) does NOT
lie entirely below 0 (ci_high >= 0); "A beats B" means ci_low > 0.

Filter metric: per-encounter ``filter.CL_analysis_r2_impact`` from the
test_b-restricted D220 envelope with per-cell tuned rho. The filter-specific
phase lag is NOT part of the automated gates (the envelope records do not
carry it); the decoded-C_L phase lag is the lag criterion. This simplification
is pre-registered here, before results.

Gate definitions (spec C.7, D255-D257):

Q2 (can W replace L) -- CW vs CL:
  STRONG: CW >= CL on pooled-peak R2 (paired CI), median |decoded lag|(CW) <
    tau_thresh, SSIM_nearbody(CW) >= (CL) (paired CI), filter CL R2(CW) >= (CL)
    (paired CI).
  WEAK: not STRONG, and CLW beats both CW and CL on peak R2 (ci_low > 0 both).
  NULL: every pairwise delta among {CW, CL, CLW} on peak R2 straddles 0.
  MIXED: anything else (reported as such, honesty over preservation).

Q2-alt (is N the better lift conditioning) -- CN vs CL:
  STRONG: CN >= CL on peak R2 (CI), median |lag|(CN) < tau_thresh,
    SSIM_nearbody(CN) BEATS (CL) (ci_low > 0), filter CL R2(CN) >= (CL).
  WEAK / NULL / MIXED: mirror Q2 with CLN in the role of CLW.

Q1 (does N add on top of W) -- CWN vs CW:
  STRONG: CWN beats CW on SSIM_nearbody (ci_low > 0), peak R2 no worse
    (ci_high >= 0), filter CL R2 no worse (ci_high >= 0).
  NULL: all three deltas straddle 0 (clean reportable null: the shed-LEV
    signature the wake carries suffices).
  MIXED: anything else.

Collapse guard (D258): every run's last 3 ``diag/pr`` values in metrics.jsonl
must exceed 0.3 * d = 9.6; violations flag the cell and are reported in the
output (the auto-fallback cannot fire in a 10k-iter run, threshold_iter=20k).

Holm correction: within each question, the bootstrap-CI conditions are the
gate; p-values from stats_lib.case_level_paired_stats are reported per metric
and Holm-corrected within the question family for the paper's tables.

Output: outputs/session34/trackc_gates.json.
Run (CPU): taskset -c 0-15 python -m scripts.session34.trackc_gates
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.session28.stats_lib import (  # noqa: E402
    case_cluster_bootstrap,
    case_level_paired_stats,
    holm,
)
from scripts.session34.trackc_cells import CELLS, RUNS_BASE  # noqa: E402

TAU_THRESH_TC = 0.1
PR_FLOOR = 9.6  # 0.3 * d, d = 32
PROBE = "linear"
SEEDS = (0, 1, 2)


# ------------------------------------------------------------- data loading
def load_lift(out_dir: Path) -> dict:
    return json.loads((out_dir / "trackc_lift.json").read_text())["results"]


def lift_per_encounter(lift: dict, cell: str, field: str) -> dict[tuple, float]:
    """Per-encounter value, averaged over seeds, keyed by (case_id, k)."""
    acc: dict[tuple, list[float]] = {}
    for seed in SEEDS:
        blob = lift[cell][f"s{seed}"][PROBE]
        for r in blob["encounters"]:
            key = (r["case_id"], r["encounter_index"])
            acc.setdefault(key, []).append(float(r[field]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def lift_pooled_r2(lift: dict, cell: str) -> dict[str, float]:
    return {
        f"s{s}": float(lift[cell][f"s{s}"][PROBE]["pooled_peak_r2"]) for s in SEEDS
    }


def median_abs_lag(lift: dict, cell: str) -> float:
    lags = []
    for seed in SEEDS:
        for r in lift[cell][f"s{seed}"][PROBE]["encounters"]:
            lags.append(abs(float(r["phase_lag_tc"])))
    return float(np.median(lags))


def ssim_per_encounter(out_dir: Path, cell: str) -> dict[tuple, float] | None:
    path = out_dir / "trackc_region_ssim.json"
    if not path.exists():
        return None
    blob = json.loads(path.read_text())["results"].get(cell)
    if blob is None:
        return None
    per_enc = blob.get("per_encounter")
    if per_enc is None:
        # scalar-only fallback (older C3 output): no pairing possible
        return {("__scalar__", -1): float(blob["ssim"]["nearbody"])}
    return {
        (r["case_id"], r["encounter_index"]): float(r["nearbody"]) for r in per_enc
    }


def filter_per_encounter(out_dir: Path, cell: str) -> dict[tuple, float] | None:
    path = out_dir / f"envelope_trackc_{cell}.json"
    if not path.exists():
        return None
    blob = json.loads(path.read_text())
    run_name = CELLS[cell][0]
    model = blob["models"].get(run_name)
    if model is None:
        return None
    return {
        (r["case_id"], r["encounter_index"]): float(r["filter"]["CL_analysis_r2_impact"])
        for r in model["records"]
        if r.get("split") == "test_b"
    }


# ------------------------------------------------------------- paired delta
def paired_delta(a: dict[tuple, float], b: dict[tuple, float]) -> dict:
    keys = sorted(set(a) & set(b))
    if not keys or keys == [("__scalar__", -1)]:
        if ("__scalar__", -1) in a and ("__scalar__", -1) in b:
            d = a[("__scalar__", -1)] - b[("__scalar__", -1)]
            return {"scalar_delta": d, "paired": False}
        return {"paired": False, "error": "no common encounters"}
    delta = np.array([a[k] - b[k] for k in keys])
    case_ids = np.array([k[0] for k in keys])
    boot = case_cluster_bootstrap(delta, case_ids)
    stats = case_level_paired_stats(
        np.array([delta[case_ids == c].mean() for c in sorted(set(case_ids.tolist()))])
    )
    return {
        "paired": True,
        "n_encounters": len(keys),
        "delta_mean": float(delta.mean()),
        "boot": {k: v for k, v in boot.items()},
        "case_stats": stats,
    }


def _ci_case(d: dict) -> tuple[float, float]:
    ci = d["boot"]["case_mean_ci"]
    return float(ci[0]), float(ci[1])


def geq_within_ci(d: dict) -> bool:
    """A >= B within the paired case-mean CI (ci_high >= 0)."""
    if not d.get("paired"):
        return bool(d.get("scalar_delta", -1) >= 0)
    return _ci_case(d)[1] >= 0


def beats(d: dict) -> bool:
    if not d.get("paired"):
        return bool(d.get("scalar_delta", -1) > 0)
    return _ci_case(d)[0] > 0


def straddles(d: dict) -> bool:
    if not d.get("paired"):
        return False
    lo, hi = _ci_case(d)
    return lo <= 0 <= hi


# ------------------------------------------------------------- PR guard
def pr_violations() -> list[dict]:
    out = []
    for cell, seeds in CELLS.items():
        for seed, run_name in seeds.items():
            path = RUNS_BASE / run_name / "metrics.jsonl"
            if not path.exists():
                out.append({"cell": cell, "seed": seed, "run": run_name,
                            "error": "metrics.jsonl missing"})
                continue
            prs = [json.loads(l)["diag/pr"] for l in path.read_text().splitlines()
                   if '"diag/pr"' in l]
            tail = prs[-3:]
            if not tail or min(tail) < PR_FLOOR:
                out.append({"cell": cell, "seed": seed, "run": run_name,
                            "pr_tail": tail})
    return out


# ------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Track C gates (pre-registered)")
    ap.add_argument("--out-dir", default="outputs/session34")
    args = ap.parse_args(argv)
    out_dir = REPO_ROOT / args.out_dir

    lift = load_lift(out_dir)

    def metric_pack(cell_a: str, cell_b: str) -> dict:
        """All paired deltas cell_a - cell_b."""
        pack = {
            "peak_r2": paired_delta(
                lift_per_encounter(lift, cell_a, "peak_r2"),
                lift_per_encounter(lift, cell_b, "peak_r2"),
            ),
            "abs_lag": paired_delta(
                {k: -abs(v) for k, v in lift_per_encounter(lift, cell_a, "phase_lag_tc").items()},
                {k: -abs(v) for k, v in lift_per_encounter(lift, cell_b, "phase_lag_tc").items()},
            ),  # sign-flipped so positive delta = cell_a has SMALLER |lag|
        }
        sa, sb = ssim_per_encounter(out_dir, cell_a), ssim_per_encounter(out_dir, cell_b)
        pack["ssim_nearbody"] = (
            paired_delta(sa, sb) if sa and sb else {"paired": False, "error": "ssim missing"}
        )
        fa, fb = filter_per_encounter(out_dir, cell_a), filter_per_encounter(out_dir, cell_b)
        pack["filter_cl_r2"] = (
            paired_delta(fa, fb) if fa and fb else {"paired": False, "error": "envelope missing"}
        )
        return pack

    result: dict = {
        "pre_registered": {
            "tau_thresh_tc": TAU_THRESH_TC,
            "probe": PROBE,
            "pr_floor": PR_FLOOR,
            "docstring": __doc__,
        },
        "pr_violations": pr_violations(),
        "pooled_peak_r2": {c: lift_pooled_r2(lift, c) for c in CELLS if c in lift},
        "median_abs_lag_tc": {c: median_abs_lag(lift, c) for c in CELLS if c in lift},
    }

    # ---- Q2: CW vs CL (with CLW) -------------------------------------------
    q2 = {"cw_vs_cl": metric_pack("cw", "cl"),
          "clw_vs_cw": metric_pack("clw", "cw"),
          "clw_vs_cl": metric_pack("clw", "cl")}
    lag_ok = result["median_abs_lag_tc"].get("cw", np.inf) < TAU_THRESH_TC
    strong = (geq_within_ci(q2["cw_vs_cl"]["peak_r2"]) and lag_ok
              and geq_within_ci(q2["cw_vs_cl"]["ssim_nearbody"])
              and geq_within_ci(q2["cw_vs_cl"]["filter_cl_r2"]))
    weak = (not strong
            and beats(q2["clw_vs_cw"]["peak_r2"]) and beats(q2["clw_vs_cl"]["peak_r2"]))
    null = all(straddles(q2[k]["peak_r2"]) for k in q2)
    q2["verdict"] = ("STRONG" if strong else "WEAK" if weak
                     else "NULL" if null else "MIXED")
    q2["lag_ok_cw"] = bool(lag_ok)
    result["Q2_D255"] = q2

    # ---- Q2-alt: CN vs CL (with CLN) ----------------------------------------
    q2a = {"cn_vs_cl": metric_pack("cn", "cl"),
           "cln_vs_cn": metric_pack("cln", "cn"),
           "cln_vs_cl": metric_pack("cln", "cl")}
    lag_ok = result["median_abs_lag_tc"].get("cn", np.inf) < TAU_THRESH_TC
    strong = (geq_within_ci(q2a["cn_vs_cl"]["peak_r2"]) and lag_ok
              and beats(q2a["cn_vs_cl"]["ssim_nearbody"])
              and geq_within_ci(q2a["cn_vs_cl"]["filter_cl_r2"]))
    weak = (not strong
            and beats(q2a["cln_vs_cn"]["peak_r2"]) and beats(q2a["cln_vs_cl"]["peak_r2"]))
    null = all(straddles(q2a[k]["peak_r2"]) for k in q2a)
    q2a["verdict"] = ("STRONG" if strong else "WEAK" if weak
                      else "NULL" if null else "MIXED")
    q2a["lag_ok_cn"] = bool(lag_ok)
    result["Q2alt_D256"] = q2a

    # ---- Q1: CWN vs CW -------------------------------------------------------
    q1 = {"cwn_vs_cw": metric_pack("cwn", "cw")}
    strong = (beats(q1["cwn_vs_cw"]["ssim_nearbody"])
              and geq_within_ci(q1["cwn_vs_cw"]["peak_r2"])
              and geq_within_ci(q1["cwn_vs_cw"]["filter_cl_r2"]))
    null = all(straddles(q1["cwn_vs_cw"][m]) for m in
               ("ssim_nearbody", "peak_r2", "filter_cl_r2")
               if q1["cwn_vs_cw"][m].get("paired"))
    q1["verdict"] = "STRONG" if strong else "NULL" if null else "MIXED"
    result["Q1_D257"] = q1

    # ---- Holm p-values per question (reported, not the gate) -----------------
    for qname, pack_key in (("Q2_D255", "cw_vs_cl"), ("Q2alt_D256", "cn_vs_cl"),
                            ("Q1_D257", "cwn_vs_cw")):
        pv = {}
        for metric, d in result[qname][pack_key].items():
            if not d.get("paired"):
                continue
            p = d.get("case_stats", {}).get("wilcoxon_p_one_sided")
            if isinstance(p, (int, float)) and np.isfinite(p):
                pv[metric] = float(p)
        if pv:
            adjusted, rejected = holm(pv)
            result[qname]["holm"] = {"p": pv, "adjusted": adjusted, "rejected": rejected}

    out_path = out_dir / "trackc_gates.json"
    out_path.write_text(json.dumps(result, indent=1, default=float))
    print(f"[gates] Q2={result['Q2_D255']['verdict']}  "
          f"Q2alt={result['Q2alt_D256']['verdict']}  "
          f"Q1={result['Q1_D257']['verdict']}  "
          f"PR violations={len(result['pr_violations'])}", flush=True)
    print(f"[gates] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
