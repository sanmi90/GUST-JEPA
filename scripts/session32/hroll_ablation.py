"""Session 32 H_roll ablation: single-step (H_roll=1) vs multi-step (H_roll=8) rollout.

Question (Carlos): does the multi-step-rollout TRAINING objective (H_roll=8, the frozen
kit default, configs/_kit.yaml pred.horizon=8) actually help on the v2.2 pooled JEPA,
versus single-step (H_roll=1)? Compared model: ``jepa_nowake_pool``.
  H_roll=8  -> outputs/runs/session32/jepa_nowake_pool         (Track P, already trained)
  H_roll=1  -> outputs/runs/session32/jepa_nowake_pool_hroll1  (this session, --horizon-override 1)

Three axes, all on Test B / test_a (frozen protocol; Test C untouched):
(a) FORECAST  : matched-predictor observable merit + C_L/E_w closure (resunet_matched,
                Track P Q2 protocol). Read from the two Q2 rollout JSONs.
(b) DRIFT     : on-manifold rolled-latent rel-L2 (drift_rel_l2), same Q2 JSONs.
(c) FILTER    : the FROZEN Track B EnKF (filter_tuning_frozen.json: rho=1.0, field-free
                init, K=8) on 3 test_a cases for BOTH models; analysis vs open-loop
                C_L/E_w closure + divergence. OSP taps: per-model OSP is FIT on
                jepa_nowake_pool (H_roll=8) with the frozen OSP protocol (osp_select,
                TCSI greedy on latent-PC1, W=30, seed 0) and REUSED for both models so
                the two filters sense at IDENTICAL wall taps -- isolating H_roll, not
                tap placement. A qdeim_shared (target-blind, also identical) row is run
                as a tap-choice robustness check.

Only READS src/estimation and the Track B/O harnesses (build_context / run_encounter /
osp_select); trains nothing here. Writes outputs/session32/hroll_ablation.json.

Run (RTX 6000 + CPU cap), AFTER the H_roll=1 train + its Q1/Q2 eval:
    taskset -c 0-15 python -m scripts.session32.hroll_ablation --gpu 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

TARGETS = ("C_L", "C_D", "wake_enstrophy", "circulation_pos", "circulation_neg")
OSP_W = 30  # frozen OSP window width (outputs/session32/osp_taps_v2p2.json params)


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def _load(p: str | Path) -> dict:
    return json.loads(_resolve(p).read_text())


# ===================================================================== (a)/(b) forecast + drift
def _at(curve: dict, key: str, h: int) -> float:
    hs = list(curve.get("horizons", []))
    return float(curve[key][hs.index(h)]) if h in hs else float("nan")


def read_forecast_cells(q2_json: str | Path, model: str, h: int = 8) -> dict:
    """Matched-predictor forecast + drift cells at horizon ``h`` from a Q2 JSON."""
    pred = _load(q2_json)["models"][model]["predictors"]["resunet_matched"]
    oc = pred["observable_closure"]
    fv = pred["field_vrmse"]
    drift = pred["drift"]
    merit = float(np.mean([_at(oc[t], "model_mlp_r2", h) for t in TARGETS]))
    return {
        "merit_mean_obs_mlp_h8": merit,
        "cl_closure_mlp_h8": _at(oc["C_L"], "model_mlp_r2", h),
        "wake_enstrophy_closure_mlp_h8": _at(oc["wake_enstrophy"], "model_mlp_r2", h),
        "cl_closure_linear_h8": _at(oc["C_L"], "model_linear_r2", h),
        "field_vrmse_model_h8": _at(fv, "model", h),
        "field_vrmse_floor_h8": _at(fv, "floor", h),
        "drift_rel_l2_h8": _at(drift, "rel_l2", h),
        "drift_rel_l2_h16": _at(drift, "rel_l2", 16),
        "merit_curve": {
            str(hh): float(np.mean([_at(oc[t], "model_mlp_r2", hh) for t in TARGETS]))
            for hh in (1, 2, 4, 8, 16)
        },
        "drift_curve": {str(hh): _at(drift, "rel_l2", hh) for hh in (1, 2, 4, 8, 16)},
    }


# =========================================================================== (c) frozen filter
def fit_osp_taps(cache_dir: Path, model: str, windows: dict, partition: str, seed: int) -> tuple:
    """Fit per-model OSP taps (frozen protocol) from a model's TRAIN latents cache."""
    from sklearn.decomposition import PCA

    from scripts.session32.osp_select import encounter_impact_windows, tcsi_staircase
    from src.evaluation.pressure_infer import load_pressure_for_alignment

    d = np.load(cache_dir / f"latents_{model}_train.npz", allow_pickle=True)
    z_gap = np.asarray(d["z_gap"], dtype=np.float32)
    cid = np.asarray(d["case_id"]).astype(str)
    enc = np.asarray(d["encounter_index"])
    frame = np.asarray(d["frame"])
    pres = load_pressure_for_alignment(cid, enc, frame, partition=partition)
    xw, zi, _ = encounter_impact_windows(z_gap, cid, enc, frame, pres, windows, OSP_W)
    target = PCA(n_components=1, random_state=seed).fit_transform(zi).ravel()
    taps = tcsi_staircase(xw, target, ks=(2, 4, 8, 16))
    return taps, int(xw.shape[0])


def build_osp_file(
    cache_dir: Path, windows: dict, partition: str, seed: int, out_path: Path
) -> dict:
    """Fit OSP on jepa_nowake_pool (H_roll=8) and reuse it for the H_roll=1 model."""
    taps, n_enc = fit_osp_taps(cache_dir, "jepa_nowake_pool", windows, partition, seed)
    frozen_osp = _load("outputs/session32/osp_taps_v2p2.json")
    payload = {
        "schema": {
            "note": (
                "H_roll ablation OSP taps. Per-model OSP FIT on jepa_nowake_pool "
                "(H_roll=8) with the frozen protocol (osp_select TCSI greedy on "
                "latent-PC1, W=30) and REUSED for jepa_nowake_pool_hroll1 so both "
                "filters sense at identical taps (isolates H_roll, not tap placement)."
            ),
            "n_taps": 192,
        },
        "params": {"W_window": OSP_W, "target": "latent_pc1", "seed": seed},
        "jepa_nowake_pool": {
            "method": "tcsi_greedy",
            "target": "latent_pc1",
            "n_encounters": n_enc,
            **taps,
        },
        "jepa_nowake_pool_hroll1": {
            "method": "tcsi_greedy_reused_from_jepa_nowake_pool",
            "target": "latent_pc1",
            "n_encounters": n_enc,
            **taps,
        },
        "qdeim_shared": frozen_osp["qdeim_shared"],
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[hroll] wrote OSP taps -> {out_path}  K8={taps['K8']}", flush=True)
    return payload


def _phase(row: dict, obs: str, which: str, phase: str, key: str = "r2"):
    return row["closure"][obs][which].get(phase, {}).get(key)


def _fmean(vals) -> float:
    v = [float(x) for x in vals if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def _fmedian(vals) -> float:
    v = [float(x) for x in vals if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else float("nan")


def run_filter(
    model: str, run_dir: str, taps_mode: str, osp_file: Path, device, windows: dict, pipe, args_ns
) -> dict:
    """Run the frozen EnKF on the 3 test_a cases for one model/taps_mode."""
    from types import SimpleNamespace

    from scripts.session32.track_b_pilot import (
        DEFAULT_CASES,
        _cache_dir,
        build_context,
        encode_encounters,
        run_encounter,
    )
    from src.evaluation.rom_eval import load_frozen_model

    fa = SimpleNamespace(
        partition="v2p2",
        predictor_steps=args_ns.predictor_steps,
        seed=args_ns.seed,
        k=8,
        taps_mode=taps_mode,
        osp_taps=str(osp_file),
        taps="outputs/session32/qdeim_taps_v2p2.json",
        rho=1.0,  # FROZEN inflation (filter_tuning_frozen.json)
        n_members=args_ns.n_members,
        mode="stochastic",
        train_cache="outputs/session32/q1_pool_latents/latents_%s_train.npz",
    )
    encounters = [(c, args_ns.encounter) for c in DEFAULT_CASES]
    print(f"\n[hroll] === filter {model} taps={taps_mode} ===", flush=True)
    ctx = build_context(model, run_dir, False, device, fa, train_cache_tmpl=fa.train_cache)
    frozen = load_frozen_model(_resolve(run_dir), args_ns.checkpoint, device)
    encs = encode_encounters(frozen, encounters, pipe, _cache_dir("v2p2"), windows, device)
    rows = [run_encounter(ctx, e, fa, rho=1.0) for e in encs]

    per_case = []
    for r in rows:
        per_case.append(
            {
                "case_id": r["case_id"],
                "diverged": bool(r["divergence"]["diverged"]),
                "mean_nis": float(r["nis_coverage"]["mean_nis"]),
                "CL_analysis_r2_impact": _phase(r, "C_L", "analysis", "impact"),
                "CL_openloop_r2_impact": _phase(r, "C_L", "open_loop", "impact"),
                "CL_analysis_r2_relax": _phase(r, "C_L", "analysis", "post_impact"),
                "CL_analysis_rmse_impact": _phase(r, "C_L", "analysis", "impact", "rmse"),
                "CL_openloop_rmse_impact": _phase(r, "C_L", "open_loop", "impact", "rmse"),
                "Ew_analysis_r2_impact": _phase(r, "wake_enstrophy", "analysis", "impact"),
                "Ew_analysis_r2_relax": _phase(r, "wake_enstrophy", "analysis", "post_impact"),
                "Ew_analysis_rmse_impact": _phase(
                    r, "wake_enstrophy", "analysis", "impact", "rmse"
                ),
                "Ew_openloop_rmse_impact": _phase(
                    r, "wake_enstrophy", "open_loop", "impact", "rmse"
                ),
            }
        )
    taps_used = [int(x) for x in ctx["taps"][0]]
    agg = {
        "taps": taps_used,
        "divergence_rate": float(np.mean([pc["diverged"] for pc in per_case])),
        "mean_nis": _fmean([pc["mean_nis"] for pc in per_case]),
        "CL_analysis_r2_impact_median": _fmedian([pc["CL_analysis_r2_impact"] for pc in per_case]),
        "CL_openloop_r2_impact_median": _fmedian([pc["CL_openloop_r2_impact"] for pc in per_case]),
        "CL_analysis_rmse_impact_mean": _fmean([pc["CL_analysis_rmse_impact"] for pc in per_case]),
        "CL_openloop_rmse_impact_mean": _fmean([pc["CL_openloop_rmse_impact"] for pc in per_case]),
        "Ew_analysis_rmse_impact_mean": _fmean([pc["Ew_analysis_rmse_impact"] for pc in per_case]),
        "Ew_openloop_rmse_impact_mean": _fmean([pc["Ew_openloop_rmse_impact"] for pc in per_case]),
        "Ew_analysis_r2_relax_median": _fmedian([pc["Ew_analysis_r2_relax"] for pc in per_case]),
    }
    print(
        f"[hroll] {model}/{taps_mode}: div_rate={agg['divergence_rate']:.2f} "
        f"CL_a_R2_imp(med)={agg['CL_analysis_r2_impact_median']:.3f} "
        f"CL_a_RMSE_imp={agg['CL_analysis_rmse_impact_mean']:.3f} "
        f"(ol {agg['CL_openloop_rmse_impact_mean']:.3f}) meanNIS={agg['mean_nis']:.1f}",
        flush=True,
    )
    return {"aggregate": agg, "per_case": per_case}


# =========================================================================== assembly
def _delta_verdict(name: str, h1: float, h8: float, higher_better: bool, tol: float) -> dict:
    """H_roll=8 minus H_roll=1 on a metric; verdict of whether H_roll=8 helps."""
    if not (np.isfinite(h1) and np.isfinite(h8)):
        return {"metric": name, "h_roll_1": h1, "h_roll_8": h8, "verdict": "n/a"}
    delta = h8 - h1  # positive = H_roll=8 larger
    signed = delta if higher_better else -delta  # positive = H_roll=8 better
    if abs(delta) < tol:
        verdict = "neutral"
    elif signed > 0:
        verdict = "H_roll=8 helps"
    else:
        verdict = "H_roll=8 hurts"
    return {
        "metric": name,
        "h_roll_1": float(h1),
        "h_roll_8": float(h8),
        "delta_8_minus_1": float(delta),
        "higher_better": higher_better,
        "verdict": verdict,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Session 32 H_roll=1 vs H_roll=8 ablation")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--encounter", type=int, default=4)
    p.add_argument("--n-members", type=int, default=64)
    p.add_argument("--predictor-steps", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", default="checkpoint_iter010000.pt")
    p.add_argument("--windows", default="outputs/session31/windows_v2p2.json")
    p.add_argument("--pipeline-manifest", default="outputs/data_pipeline/v2p2/manifest.json")
    p.add_argument("--q2-h8", default="outputs/session32/q2_pool.json")
    p.add_argument("--q2-h1", default="outputs/session32/q2_hroll1.json")
    p.add_argument("--taps-modes", nargs="+", default=["osp_per_model", "qdeim_shared"])
    p.add_argument("--skip-filter", action="store_true")
    p.add_argument("--out", default="outputs/session32/hroll_ablation.json")
    return p.parse_args(argv)


def main(argv=None) -> int:
    import torch

    from src.data.omega_pipeline import OmegaPipeline

    args = parse_args(argv)

    runs = {
        "H_roll_8": "outputs/runs/session32/jepa_nowake_pool",
        "H_roll_1": "outputs/runs/session32/jepa_nowake_pool_hroll1",
    }

    # ---- (a)/(b) forecast + drift from the two Q2 JSONs -------------------
    fc8 = read_forecast_cells(args.q2_h8, "jepa_nowake_pool")
    fc1 = read_forecast_cells(args.q2_h1, "jepa_nowake_pool_hroll1")

    result: dict = {
        "task": "SESSION 32 H_roll=1 vs H_roll=8 ablation (jepa_nowake_pool, v2.2 pooled)",
        "question": (
            "Does the multi-step rollout TRAINING objective (H_roll=8, frozen kit "
            "default) help vs single-step (H_roll=1) on the v2.2 pooled JEPA?"
        ),
        "models": {
            "H_roll_1": runs["H_roll_1"] + " (--horizon-override 1)",
            "H_roll_8": runs["H_roll_8"] + " (frozen kit default)",
        },
        "axis_a_forecast": {
            "protocol": "matched ResUNet predictor (Track P Q2, resunet_matched, h8)",
            "H_roll_1": fc1,
            "H_roll_8": fc8,
            "deltas": [
                _delta_verdict(
                    "merit_mean_obs_mlp_h8",
                    fc1["merit_mean_obs_mlp_h8"],
                    fc8["merit_mean_obs_mlp_h8"],
                    True,
                    0.02,
                ),
                _delta_verdict(
                    "cl_closure_mlp_h8",
                    fc1["cl_closure_mlp_h8"],
                    fc8["cl_closure_mlp_h8"],
                    True,
                    0.02,
                ),
                _delta_verdict(
                    "wake_enstrophy_closure_mlp_h8",
                    fc1["wake_enstrophy_closure_mlp_h8"],
                    fc8["wake_enstrophy_closure_mlp_h8"],
                    True,
                    0.02,
                ),
                _delta_verdict(
                    "field_vrmse_model_h8",
                    fc1["field_vrmse_model_h8"],
                    fc8["field_vrmse_model_h8"],
                    False,
                    0.02,
                ),
            ],
        },
        "axis_b_drift": {
            "protocol": "aggregated rolled-latent rel-L2 drift (Track P Q2, resunet_matched)",
            "H_roll_1": {"drift_rel_l2_h8": fc1["drift_rel_l2_h8"], "curve": fc1["drift_curve"]},
            "H_roll_8": {"drift_rel_l2_h8": fc8["drift_rel_l2_h8"], "curve": fc8["drift_curve"]},
            "deltas": [
                _delta_verdict(
                    "drift_rel_l2_h8",
                    fc1["drift_rel_l2_h8"],
                    fc8["drift_rel_l2_h8"],
                    False,
                    0.02,
                ),
                _delta_verdict(
                    "drift_rel_l2_h16",
                    fc1["drift_rel_l2_h16"],
                    fc8["drift_rel_l2_h16"],
                    False,
                    0.02,
                ),
            ],
        },
    }

    # ---- (c) frozen filter ------------------------------------------------
    if not args.skip_filter:
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
        cache_dir = _resolve("outputs/session32/q1_pool_latents")
        osp_file = _resolve("outputs/session32/osp_taps_hroll_ablation.json")
        build_osp_file(cache_dir, windows, "v2p2", args.seed, osp_file)

        filt: dict = {
            "protocol": (
                "FROZEN Track B EnKF (rho=1.0, field-free init, K=8, N=64) on 3 test_a "
                "encounters; per-model OSP fit on jepa_nowake_pool and reused for both "
                "(identical taps); rho frozen from filter_tuning_frozen.json"
            ),
            "gpu_name": gpu_name,
            "taps_modes": {},
        }
        for tmode in args.taps_modes:
            per_mode: dict = {}
            for label, run_dir in runs.items():
                model = "jepa_nowake_pool" if label == "H_roll_8" else "jepa_nowake_pool_hroll1"
                per_mode[label] = run_filter(
                    model, run_dir, tmode, osp_file, device, windows, pipe, args
                )
            a1 = per_mode["H_roll_1"]["aggregate"]
            a8 = per_mode["H_roll_8"]["aggregate"]
            per_mode["deltas"] = [
                _delta_verdict(
                    "filter_CL_analysis_rmse_impact_mean (lower better)",
                    a1["CL_analysis_rmse_impact_mean"],
                    a8["CL_analysis_rmse_impact_mean"],
                    False,
                    0.05,
                ),
                _delta_verdict(
                    "filter_CL_analysis_r2_impact_median",
                    a1["CL_analysis_r2_impact_median"],
                    a8["CL_analysis_r2_impact_median"],
                    True,
                    0.05,
                ),
                _delta_verdict(
                    "filter_Ew_analysis_rmse_impact_mean (lower better)",
                    a1["Ew_analysis_rmse_impact_mean"],
                    a8["Ew_analysis_rmse_impact_mean"],
                    False,
                    0.05,
                ),
                _delta_verdict(
                    "filter_divergence_rate (lower better)",
                    a1["divergence_rate"],
                    a8["divergence_rate"],
                    False,
                    0.01,
                ),
            ]
            filt["taps_modes"][tmode] = per_mode
        result["axis_c_frozen_filter"] = filt

    out = _resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=float))
    print(f"\n[hroll] wrote {out}", flush=True)
    _print_summary(result)
    return 0


def _print_summary(result: dict) -> None:
    print("\n[hroll] ===== H_roll=1 vs H_roll=8 SUMMARY (jepa_nowake_pool, v2.2) =====")
    print("  (a) FORECAST (matched predictor, h8):")
    for d in result["axis_a_forecast"]["deltas"]:
        print(
            f"    {d['metric']:<34} H1={d['h_roll_1']:+.3f} H8={d['h_roll_8']:+.3f} "
            f"d8-1={d.get('delta_8_minus_1', float('nan')):+.3f} -> {d['verdict']}"
        )
    print("  (b) ON-MANIFOLD DRIFT (rel-L2, lower better):")
    for d in result["axis_b_drift"]["deltas"]:
        print(
            f"    {d['metric']:<34} H1={d['h_roll_1']:.3f} H8={d['h_roll_8']:.3f} "
            f"d8-1={d.get('delta_8_minus_1', float('nan')):+.3f} -> {d['verdict']}"
        )
    if "axis_c_frozen_filter" in result:
        print("  (c) FROZEN FILTER (analysis closure + divergence):")
        for tmode, pm in result["axis_c_frozen_filter"]["taps_modes"].items():
            print(f"    taps={tmode}:")
            for d in pm["deltas"]:
                print(
                    f"      {d['metric']:<44} H1={d['h_roll_1']:.3f} H8={d['h_roll_8']:.3f} "
                    f"d8-1={d.get('delta_8_minus_1', float('nan')):+.3f} -> {d['verdict']}"
                )
    print("[hroll] ================================================================\n")


if __name__ == "__main__":
    raise SystemExit(main())
