"""Session 32 Track B: freeze inflation rho + D220 warning thresholds on test_a.

Sweeps multiplicative inflation ``rho in {1.00, 1.02, 1.05}`` on the 3 test_a
cases for ``jepa_pool`` (field-free init, per-model OSP taps) and SELECTS rho by
the pre-registered criterion (D220):

    minimize |mean NIS - K|  (K=8, target mean NIS = 8)
    subject to no divergence,
    with innovation whiteness (min mean |lag-1 autocorr|) as the second key,
    tie-broken by analysis E_w closure (impact + relaxation, higher R^2 better).

The selected rho and the D220 warning thresholds (eps_A=0.25, eps_t=0.25 t/c) are
written to ``outputs/session32/filter_tuning_frozen.json`` and LOCKED for the S33
campaign. This is the only place these are tuned; do not re-tune later, and do NOT
touch Test B/C.

Run (RTX 6000 + CPU cap):
    taskset -c 0-7 python -m scripts.session32.track_b_freeze_tuning --gpu 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

from src.data.omega_pipeline import OmegaPipeline  # noqa: E402
from scripts.session32.track_b_pilot import (  # noqa: E402
    DEFAULT_CASES,
    _cache_dir,
    build_context,
    encode_encounters,
    run_encounter,
)
from src.evaluation.rom_eval import load_frozen_model, load_reference_model  # noqa: E402

RHO_GRID = (1.00, 1.02, 1.05)
K = 8
# D220 warning thresholds (frozen with the selected rho).
EPS_A = 0.25
EPS_T = 0.25  # t/c


def _resolve(p):
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def _ew_closure(row):
    """Analysis E_w closure R^2 averaged over impact + relaxation (post_impact)."""
    ew = row["closure"]["wake_enstrophy"]["analysis"]
    vals = [ew.get("impact", {}).get("r2"), ew.get("post_impact", {}).get("r2")]
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def aggregate(rows):
    """Aggregate the 3-case rows for one rho into the selection statistics."""
    mean_nis = float(np.mean([r["nis_coverage"]["mean_nis"] for r in rows]))
    any_div = bool(any(r["divergence"]["diverged"] for r in rows))
    mean_lag1 = float(np.mean([r["innovation_whiteness"]["mean_abs_lag1"] for r in rows]))
    ew_clo = float(np.mean([_ew_closure(r) for r in rows]))
    frac_band = float(np.mean([r["nis_coverage"]["frac_in_band_95"] for r in rows]))
    return {
        "mean_nis": mean_nis,
        "nis_abs_dev_from_K": abs(mean_nis - K),
        "any_diverged": any_div,
        "mean_abs_lag1": mean_lag1,
        "analysis_Ew_closure_r2": ew_clo,
        "mean_frac_in_band_95": frac_band,
        "n_cases": len(rows),
    }


def select_rho(sweep):
    """Apply the D220 selection criterion; return (rho, reason, relaxed_flag)."""
    non_div = {r: s for r, s in sweep.items() if not s["any_diverged"]}
    relaxed = False
    pool = non_div
    if not pool:
        pool = sweep
        relaxed = True  # no rho avoided divergence on all 3 cases; report as measured
    ranked = sorted(
        pool.items(),
        key=lambda kv: (
            kv[1]["nis_abs_dev_from_K"],  # primary: |mean NIS - K|
            kv[1]["mean_abs_lag1"],  # secondary: whiteness
            -kv[1]["analysis_Ew_closure_r2"],  # tertiary: E_w closure
        ),
    )
    best_rho = ranked[0][0]
    s = ranked[0][1]
    reason = (
        f"min |meanNIS-{K}|={s['nis_abs_dev_from_K']:.2f} "
        f"(meanNIS={s['mean_nis']:.2f}); whiteness |lag1|={s['mean_abs_lag1']:.2f}; "
        f"E_w closure R2={s['analysis_Ew_closure_r2']:.2f}; "
        f"diverged={s['any_diverged']}"
    )
    return best_rho, reason, relaxed


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Track B: freeze rho + D220 thresholds on test_a")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    p.add_argument("--encounter", type=int, default=4)
    p.add_argument("--rho-grid", nargs="+", type=float, default=list(RHO_GRID))
    p.add_argument("--mode", default="stochastic", choices=["stochastic", "sqrt"])
    p.add_argument("--n-members", type=int, default=64)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--taps-mode", default="osp_per_model")
    p.add_argument("--osp-taps", default="outputs/session32/osp_taps_v2p2.json")
    p.add_argument("--taps", default="outputs/session32/qdeim_taps_v2p2.json")
    p.add_argument("--partition", default="v2p2")
    p.add_argument("--pipeline-manifest", default="outputs/data_pipeline/v2p2/manifest.json")
    p.add_argument("--windows", default="outputs/session31/windows_v2p2.json")
    p.add_argument("--checkpoint", default="checkpoint_iter010000.pt")
    p.add_argument(
        "--train-cache",
        default="outputs/session31/q1_latents_ablation/latents_%s_train.npz",
    )
    p.add_argument("--jepa-run", default="outputs/runs/session31/jepa_pool")
    p.add_argument("--model", default="jepa_pool",
                   help="Model to tune rho for (D252 per-method: jepa_pool_vec, fukami, pod).")
    p.add_argument("--run-dir", default=None,
                   help="Run dir for --model; defaults to --jepa-run.")
    p.add_argument("--is-reference", action="store_true",
                   help="Load --model as a reference (fukami/pod) not a CanonicalModel.")
    p.add_argument("--predictor-steps", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rho", type=float, default=1.02)  # default used by run_encounter fallback
    p.add_argument("--out", default="outputs/session32/filter_tuning_frozen.json")
    p.add_argument("--sweep-out", default="outputs/session32/track_b_rho_sweep.json")
    return p.parse_args(argv)


def main(argv=None):
    import torch

    args = parse_args(argv)
    if args.device is not None:
        device = torch.device(args.device)
        gpu_name = args.device
    else:
        from src.utils.device import require_rtx6000

        device = require_rtx6000(gpu_index=args.gpu)
        gpu_name = torch.cuda.get_device_name(device.index)
        if "RTX" not in gpu_name or "6000" not in gpu_name:
            raise RuntimeError(f"hardware policy: gpu_name={gpu_name!r} is not an RTX 6000")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    windows = json.loads(_resolve(args.windows).read_text())["windows"]
    pipe = OmegaPipeline.from_manifest(_resolve(args.pipeline_manifest))
    encounters = [(c, args.encounter) for c in args.cases]

    run_dir = args.run_dir or args.jepa_run
    print(
        f"[freeze] device={device} ({gpu_name}) model={args.model} "
        f"(reference={args.is_reference}) taps={args.taps_mode} rho_grid={args.rho_grid}",
        flush=True,
    )

    # Build the (rho-independent) context and encode the 3 test_a encounters ONCE.
    ctx = build_context(
        args.model,
        run_dir,
        args.is_reference,
        device,
        args,
        train_cache_tmpl=args.train_cache,
    )
    loader = load_reference_model if args.is_reference else load_frozen_model
    frozen = loader(_resolve(run_dir), args.checkpoint, device)
    encs = encode_encounters(frozen, encounters, pipe, _cache_dir(args.partition), windows, device)

    sweep: dict = {}
    sweep_detail: dict = {}
    for rho in args.rho_grid:
        rows = [run_encounter(ctx, e, args, rho=rho) for e in encs]
        agg = aggregate(rows)
        sweep[rho] = agg
        sweep_detail[f"{rho:.2f}"] = {
            "aggregate": agg,
            "per_case": [
                {
                    "case_id": r["case_id"],
                    "mean_nis": r["nis_coverage"]["mean_nis"],
                    "frac_in_band_95": r["nis_coverage"]["frac_in_band_95"],
                    "mean_abs_lag1": r["innovation_whiteness"]["mean_abs_lag1"],
                    "diverged": r["divergence"]["diverged"],
                    "Ew_analysis_r2_impact": r["closure"]["wake_enstrophy"]["analysis"]
                    .get("impact", {})
                    .get("r2"),
                    "Ew_analysis_r2_relax": r["closure"]["wake_enstrophy"]["analysis"]
                    .get("post_impact", {})
                    .get("r2"),
                    "CL_analysis_r2_impact": r["closure"]["C_L"]["analysis"]
                    .get("impact", {})
                    .get("r2"),
                }
                for r in rows
            ],
        }
        print(
            f"[freeze] rho={rho:.2f} meanNIS={agg['mean_nis']:.2f} "
            f"|NIS-K|={agg['nis_abs_dev_from_K']:.2f} |lag1|={agg['mean_abs_lag1']:.2f} "
            f"E_w R2={agg['analysis_Ew_closure_r2']:.2f} "
            f"frac_band95={agg['mean_frac_in_band_95']:.2f} diverged={agg['any_diverged']}",
            flush=True,
        )

    best_rho, reason, relaxed = select_rho(sweep)
    print(
        f"\n[freeze] SELECTED rho={best_rho:.2f} :: {reason}"
        + ("  [RELAXED: all rho diverged]" if relaxed else ""),
        flush=True,
    )

    frozen_payload = {
        "task": "SESSION 32 Track B -- frozen filter tuning (D220)",
        "decision": "D220",
        "frozen": {
            "inflation_rho": float(best_rho),
            "eps_A": EPS_A,
            "eps_t_over_c": EPS_T,
            "K_taps": args.k,
            "taps_mode": args.taps_mode,
            "n_members": args.n_members,
            "mode": args.mode,
            "init": "field_free_O1_windowed_pressure",
        },
        "selection": {
            "criterion": (
                "minimize |mean NIS - K| subject to no divergence; second key = min "
                "mean |lag-1 innovation autocorr|; tie-break = analysis E_w closure "
                "R^2 (impact+relaxation)"
            ),
            "K": K,
            "reason": reason,
            "relaxed_no_nondiverged_rho": relaxed,
            "selected_rho": float(best_rho),
        },
        "sweep": sweep_detail,
        "provenance": {
            "cases": args.cases,
            "encounter": args.encounter,
            "split": "test_a",
            "model": "jepa_pool",
            "osp_taps_file": args.osp_taps,
            "gpu_name": gpu_name,
            "seed": args.seed,
            "note": (
                "Frozen on test_a ONLY (D220). Locked for S33; do not re-tune. "
                "Test B/C untouched. eps_A / eps_t are the D220 warning thresholds."
            ),
        },
    }
    out_path = _resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(frozen_payload, indent=2, default=float))
    _resolve(args.sweep_out).write_text(json.dumps(sweep_detail, indent=2, default=float))
    print(f"[freeze] wrote {out_path} and {_resolve(args.sweep_out)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
