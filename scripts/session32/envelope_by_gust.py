"""Session 32 -- Gust-intensity OPERATING-ENVELOPE analysis.

Report results stratified by gust intensity (|G| magnitude and D size) to find the
envelope: "up to what gust strength / diameter can the method actually do
something?". Three method lenses, per-encounter, on ALL splits (train, val,
test_b, test_c), labelled by split, for the headline pooled model ``jepa_pool``
(plus ``POD`` as a reference lens):

1. FILTER  (frozen Track B EnKF; filter_tuning_frozen.json verbatim): analysis
   C_L / E_w closure R^2 (impact + relaxation phases), analysis-vs-open-loop
   improvement (RMSE reduction), the NIS-tail divergence flag (D231), mean NIS.
2. FORECAST (open-loop rollout through the matched AR-transformer, no measurement
   updates -- the same open-loop free run the filter is compared against):
   C_L / E_w closure R^2 per phase.
3. STATIC RECOVERY (O1): per-encounter windowed p_wall -> pooled-state recovery
   at the model's own OSP K8 taps (W=30, KRR Nystroem-RBF), read through the
   frozen probes; state R^2 + observable closure R^2 per phase.

The filter tuning is FROZEN (outputs/session32/filter_tuning_frozen.json):
inflation_rho=1.0, taps_mode=osp_per_model, field-free O1 init, K=8, N=64,
stochastic, seed 0. NOTHING is re-tuned. This is a CHARACTERISATION with the frozen
filter (reporting, not model selection), which is why touching test_c is legitimate.

CONVENTION: case_id / split ``G`` is the INVENTORY convention (G_inv = -G_phys, per
HANDOFF D224). |G| buckets are sign-agnostic. Sign asymmetry is reported in the
inventory sign; physical sign is the negation.

Run (RTX 6000 + CPU cap):
    taskset -c 0-15 python -m scripts.session32.envelope_by_gust --gpu 0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

from src.data.omega_pipeline import OmegaPipeline  # noqa: E402
from src.estimation import metrics as mt  # noqa: E402
from src.evaluation.pressure_infer import (  # noqa: E402
    fit_pressure_estimator,
    load_pressure_for_alignment,
)
from src.evaluation.rom_eval import load_frozen_model, load_reference_model  # noqa: E402
from scripts.session32.track_b_pilot import (  # noqa: E402
    PRE,
    POST,
    _cache_dir,
    _resolve,
    build_context,
    encode_encounters,
    run_encounter,
    windowed_features,
)

O1_WINDOW = 30  # O1 canonical windowed-recovery length (Track O)


# ---------------------------------------------------------------- enumeration
def enumerate_encounters(split_path: Path) -> list[dict]:
    """All (case, encounter) with split label + (G, D, Y) from split_v2p2."""
    blob = json.loads(split_path.read_text())
    out: list[dict] = []
    for cid, c in blob["cases"].items():
        base = {"case_id": cid, "G": float(c["G"]), "D": float(c["D"]), "Y": float(c["Y"])}
        if c["split"] == "train":
            for k in c.get("train_encounter_indices", []):
                out.append({**base, "encounter_index": int(k), "split": "train"})
            for k in c.get("val_encounter_indices", []):
                out.append({**base, "encounter_index": int(k), "split": "val"})
        else:  # test_b / test_c: every encounter is held out
            for k in c.get("val_encounter_indices", []):
                out.append({**base, "encounter_index": int(k), "split": c["split"]})
    return out


# ---------------------------------------------------------------- O1 recovery
def build_o1_recovery(name: str, args, taps: np.ndarray) -> object:
    """Fit a windowed(W=30) OSP-K8-tap p_wall -> pooled-state estimator on TRAIN."""
    cache = np.load(_resolve(args.train_cache % name), allow_pickle=True)
    z_gap_tr = np.asarray(cache["z_gap"], dtype=np.float32)
    cid_tr = np.asarray(cache["case_id"]).astype(str)
    enc_tr = np.asarray(cache["encounter_index"])
    frame_tr = np.asarray(cache["frame"])
    p_tr = load_pressure_for_alignment(cid_tr, enc_tr, frame_tr, partition=args.partition)
    # restrict to the model's OSP K8 taps, then window (causal, edge-padded).
    p_osp = p_tr[:, np.sort(taps)]
    win_tr = windowed_features(p_osp, O1_WINDOW)  # (N, W*K)
    groups = np.array([f"{c}/{int(e)}" for c, e in zip(cid_tr, enc_tr)])
    rng = np.random.default_rng(args.seed)
    sub = rng.choice(len(z_gap_tr), size=min(9000, len(z_gap_tr)), replace=False)
    est = fit_pressure_estimator(
        win_tr[sub], z_gap_tr[sub], n_components=300, seed=args.seed, groups=groups[sub]
    )
    return est


def forecast_truthinit(enc, ctx) -> dict:
    """Pure matched-predictor forecast skill: truth-init open-loop rollout.

    Seed the matched AR-transformer with the REAL encoded context up to the anchor
    ``f0 = t_imp - PRE``, then roll deterministically (single trajectory, no
    measurement update) to ``f1 = t_imp + POST``. This isolates the predictor's
    forecast skill from the field-free (pressure) init error that the Track-B
    open-loop baseline carries. Observables read through the frozen probes; closure
    per phase. The impact + relaxation phases are entirely forecast (never seen).
    """
    fmodel = ctx["fmodel"]
    z = np.asarray(enc["z_gap"], dtype=np.float64)
    t_imp = enc["t_impact"]
    n_frames = enc["n_frames"]
    f0 = max(0, t_imp - PRE)
    f1 = min(n_frames - 1, t_imp + POST)
    frames = np.arange(f0, f1 + 1)
    mc = int(getattr(fmodel, "max_context", 32))
    lo = max(0, f0 - mc + 1)
    buf = z[lo : f0 + 1][None, :, :].astype(np.float32)  # (1, ctx, d)
    states = np.empty((len(frames), z.shape[1]), dtype=np.float64)
    states[0] = z[f0]  # anchor frame is the real encoded state
    for i in range(1, len(frames)):
        prior = np.asarray(fmodel.step(buf), dtype=np.float64)  # (1, d)
        states[i] = prior[0]
        buf = np.concatenate([buf, prior[:, None, :].astype(np.float32)], axis=1)
        if buf.shape[1] > mc:
            buf = buf[:, -mc:, :]
    r_cl = np.asarray(ctx["probe_cl"].predict(states), dtype=np.float64).reshape(-1)
    r_ew = np.asarray(ctx["probe_ew"].predict(states), dtype=np.float64).reshape(-1)
    phase_ids = enc["phase_ids"][frames]
    return {
        "closure_C_L": mt.closure_by_phase(r_cl, enc["C_L"][frames], phase_ids),
        "closure_wake_enstrophy": mt.closure_by_phase(
            r_ew, enc["wake_enstrophy"][frames], phase_ids
        ),
    }


def recovery_for_encounter(est, taps, enc, ctx, args) -> dict:
    """Per-encounter windowed p_wall -> state recovery; closure per phase + state R^2."""
    t_imp = enc["t_impact"]
    n_frames = enc["n_frames"]
    f0 = max(0, t_imp - PRE)
    f1 = min(n_frames - 1, t_imp + POST)
    frames = np.arange(f0, f1 + 1)
    p_osp = enc["p_wall"][:, np.sort(taps)]
    win = windowed_features(p_osp, O1_WINDOW)[frames]  # (T, W*K)
    z_hat = np.asarray(est.predict(win), dtype=np.float64)  # (T, d)
    z_true = np.asarray(enc["z_gap"][frames], dtype=np.float64)
    # state R^2 over the encounter (column mean over the assimilation window).
    mean_col = z_true.mean(axis=0, keepdims=True)
    num = float(((z_hat - z_true) ** 2).sum())
    den = float(((z_true - mean_col) ** 2).sum())
    state_r2 = float(1.0 - num / den) if den > 0 else float("nan")
    # observable closure through the frozen probes.
    r_cl = np.asarray(ctx["probe_cl"].predict(z_hat), dtype=np.float64).reshape(-1)
    r_ew = np.asarray(ctx["probe_ew"].predict(z_hat), dtype=np.float64).reshape(-1)
    phase_ids = enc["phase_ids"][frames]
    clo_cl = mt.closure_by_phase(r_cl, enc["C_L"][frames], phase_ids)
    clo_ew = mt.closure_by_phase(r_ew, enc["wake_enstrophy"][frames], phase_ids)
    return {"state_r2": state_r2, "closure_C_L": clo_cl, "closure_wake_enstrophy": clo_ew}


# ---------------------------------------------------------------- record flatten
def _r2(clo: dict, phase: str) -> float:
    return float(clo.get(phase, {}).get("r2", float("nan")))


def _rmse(clo: dict, phase: str) -> float:
    return float(clo.get(phase, {}).get("rmse", float("nan")))


def flatten_record(meta: dict, filt: dict, rec: dict, fct: dict) -> dict:
    """Collapse the filter + forecast + recovery outputs into one flat record."""
    fcl = filt["closure"]["C_L"]
    few = filt["closure"]["wake_enstrophy"]
    tcl = fct["closure_C_L"]
    tew = fct["closure_wake_enstrophy"]
    out = {
        "case_id": meta["case_id"],
        "encounter_index": meta["encounter_index"],
        "split": meta["split"],
        "G_inv": meta["G"],
        "G_phys": -meta["G"],
        "abs_G": abs(meta["G"]),
        "D": meta["D"],
        "Y": meta["Y"],
        "t_impact": filt["t_impact"],
        "n_assim": filt["n_assim"],
        # -------- FILTER lens --------
        "filter": {
            "diverged": bool(filt["divergence"]["diverged"]),
            "nis_tail_run": int(filt["divergence"]["nis_tail_run"]),
            "mean_nis": float(filt["nis_coverage"]["mean_nis"]),
            "mean_abs_lag1": float(filt["innovation_whiteness"]["mean_abs_lag1"]),
            "CL_analysis_r2_impact": _r2(fcl["analysis"], "impact"),
            "CL_analysis_r2_relax": _r2(fcl["analysis"], "post_impact"),
            "Ew_analysis_r2_impact": _r2(few["analysis"], "impact"),
            "Ew_analysis_r2_relax": _r2(few["analysis"], "post_impact"),
            "CL_gain_rmse_impact": float(
                fcl["gain_vs_open_loop"].get("impact", {}).get("rmse_reduction", float("nan"))
            ),
            "CL_gain_rmse_relax": float(
                fcl["gain_vs_open_loop"].get("post_impact", {}).get("rmse_reduction", float("nan"))
            ),
            "Ew_gain_rmse_impact": float(
                few["gain_vs_open_loop"].get("impact", {}).get("rmse_reduction", float("nan"))
            ),
            "Ew_gain_rmse_relax": float(
                few["gain_vs_open_loop"].get("post_impact", {}).get("rmse_reduction", float("nan"))
            ),
        },
        # -------- FORECAST lens --------
        # truthinit_* = pure matched-predictor forecast skill (truth-init open-loop,
        #   PRIMARY forecast metric). fieldfree_* = the Track-B open-loop baseline
        #   (field-free pressure init, what the filter is scored against).
        "forecast": {
            "truthinit_CL_r2_impact": _r2(tcl, "impact"),
            "truthinit_CL_r2_relax": _r2(tcl, "post_impact"),
            "truthinit_Ew_r2_impact": _r2(tew, "impact"),
            "truthinit_Ew_r2_relax": _r2(tew, "post_impact"),
            "fieldfree_CL_r2_impact": _r2(fcl["open_loop"], "impact"),
            "fieldfree_CL_r2_relax": _r2(fcl["open_loop"], "post_impact"),
            "fieldfree_Ew_r2_impact": _r2(few["open_loop"], "impact"),
        },
        # -------- RECOVERY lens (O1 static) --------
        "recovery": {
            "state_r2": float(rec["state_r2"]),
            "CL_r2_impact": _r2(rec["closure_C_L"], "impact"),
            "CL_r2_relax": _r2(rec["closure_C_L"], "post_impact"),
            "Ew_r2_impact": _r2(rec["closure_wake_enstrophy"], "impact"),
            "Ew_r2_relax": _r2(rec["closure_wake_enstrophy"], "post_impact"),
        },
    }
    return out


# ---------------------------------------------------------------- aggregation
GBUCKETS = [
    ("0", lambda g: g == 0.0),
    ("0.25-0.5", lambda g: 0.24 < g <= 0.5),
    ("1", lambda g: 0.9 < g <= 1.0),
    ("1.5", lambda g: 1.4 < g <= 1.5),
    ("2", lambda g: 1.9 < g <= 2.0),
    ("3", lambda g: 2.9 < g <= 3.0),
    ("4", lambda g: 3.9 < g <= 4.0),
]
DBUCKETS = ["0.0", "0.5", "1.0", "1.5"]


def _stat(vals: list[float]) -> dict:
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=np.float64)
    if a.size == 0:
        return {"n": 0, "mean": None, "median": None, "std": None, "frac_pos": None}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "std": float(a.std()),
        "frac_pos": float(np.mean(a > 0.0)),
    }


def _bucket_of(g: float) -> str:
    for name, fn in GBUCKETS:
        if fn(g):
            return name
    return f"other({g})"


def aggregate(records: list[dict]) -> dict:
    """Per (|G| bucket x D) aggregates and overall-|G| aggregates."""
    metric_paths = {
        "filter_CL_analysis_r2_impact": ("filter", "CL_analysis_r2_impact"),
        "filter_CL_analysis_r2_relax": ("filter", "CL_analysis_r2_relax"),
        "filter_Ew_analysis_r2_impact": ("filter", "Ew_analysis_r2_impact"),
        "filter_Ew_analysis_r2_relax": ("filter", "Ew_analysis_r2_relax"),
        "filter_CL_gain_rmse_impact": ("filter", "CL_gain_rmse_impact"),
        "filter_Ew_gain_rmse_impact": ("filter", "Ew_gain_rmse_impact"),
        "filter_mean_nis": ("filter", "mean_nis"),
        "forecast_CL_r2_impact": ("forecast", "truthinit_CL_r2_impact"),
        "forecast_CL_r2_relax": ("forecast", "truthinit_CL_r2_relax"),
        "forecast_Ew_r2_impact": ("forecast", "truthinit_Ew_r2_impact"),
        "forecast_fieldfree_CL_r2_impact": ("forecast", "fieldfree_CL_r2_impact"),
        "recovery_state_r2": ("recovery", "state_r2"),
        "recovery_CL_r2_impact": ("recovery", "CL_r2_impact"),
        "recovery_CL_r2_relax": ("recovery", "CL_r2_relax"),
        "recovery_Ew_r2_impact": ("recovery", "Ew_r2_impact"),
    }

    def agg_over(subset: list[dict]) -> dict:
        if not subset:
            return {"n": 0}
        d = {"n": len(subset)}
        d["splits"] = {
            s: sum(1 for r in subset if r["split"] == s)
            for s in ("train", "val", "test_b", "test_c")
        }
        d["div_rate"] = float(np.mean([1.0 if r["filter"]["diverged"] else 0.0 for r in subset]))
        for mname, (lens, key) in metric_paths.items():
            d[mname] = _stat([r[lens][key] for r in subset])
        return d

    out: dict = {"by_G_and_D": {}, "by_G": {}, "by_sign": {}, "by_D": {}}
    # by |G| x D
    for gname, gfn in GBUCKETS:
        out["by_G_and_D"][gname] = {}
        gsub = [r for r in records if gfn(r["abs_G"])]
        out["by_G"][gname] = agg_over(gsub)
        for dname in DBUCKETS:
            dv = float(dname)
            sub = [r for r in gsub if abs(r["D"] - dv) < 1e-6]
            if sub:
                out["by_G_and_D"][gname][dname] = agg_over(sub)
    # by D (all |G|>0)
    for dname in DBUCKETS:
        dv = float(dname)
        sub = [r for r in records if abs(r["D"] - dv) < 1e-6]
        out["by_D"][dname] = agg_over(sub)
    # sign asymmetry (inventory sign), excluding |G|=0
    for sname, sfn in (("pos_Ginv", lambda g: g > 0), ("neg_Ginv", lambda g: g < 0)):
        sub = [r for r in records if sfn(r["G_inv"])]
        out["by_sign"][sname] = agg_over(sub)
    # sign x |G| (paired magnitudes)
    out["by_sign_and_G"] = {}
    for gname, gfn in GBUCKETS:
        if gname == "0":
            continue
        out["by_sign_and_G"][gname] = {}
        for sname, sfn in (("pos_Ginv", lambda g: g > 0), ("neg_Ginv", lambda g: g < 0)):
            sub = [r for r in records if gfn(r["abs_G"]) and sfn(r["G_inv"])]
            if sub:
                out["by_sign_and_G"][gname][sname] = agg_over(sub)
    return out


def identify_thresholds(records: list[dict]) -> dict:
    """Breakdown thresholds per D: |G| beyond which (a) div rate > 50%, (b) filter
    analysis / forecast C_L closure median R^2 drops below 0."""
    gorder = ["0", "0.25-0.5", "1", "1.5", "2", "3", "4"]
    gnum = {"0": 0.0, "0.25-0.5": 0.5, "1": 1.0, "1.5": 1.5, "2": 2.0, "3": 3.0, "4": 4.0}
    thresholds: dict = {}
    for dname in ["0.5", "1.0", "1.5"]:
        dv = float(dname)
        dsub = [r for r in records if abs(r["D"] - dv) < 1e-6]
        rows = []
        for gname in gorder:
            gfn = dict(GBUCKETS)[gname]
            sub = [r for r in dsub if gfn(r["abs_G"])]
            if not sub:
                rows.append((gname, None))
                continue
            div = float(np.mean([1.0 if r["filter"]["diverged"] else 0.0 for r in sub]))
            filt_cl = _stat([r["filter"]["CL_analysis_r2_impact"] for r in sub])
            fc_cl = _stat([r["forecast"]["truthinit_CL_r2_impact"] for r in sub])
            rec_cl = _stat([r["recovery"]["CL_r2_impact"] for r in sub])
            rows.append(
                (
                    gname,
                    {
                        "abs_G": gnum[gname],
                        "n": len(sub),
                        "div_rate": div,
                        "filter_CL_analysis_r2_impact_median": filt_cl["median"],
                        "filter_CL_analysis_r2_impact_frac_pos": filt_cl["frac_pos"],
                        "forecast_CL_r2_impact_median": fc_cl["median"],
                        "forecast_CL_r2_impact_frac_pos": fc_cl["frac_pos"],
                        "recovery_CL_r2_impact_median": rec_cl["median"],
                        "recovery_CL_r2_impact_frac_pos": rec_cl["frac_pos"],
                    },
                )
            )
        # First |G| ABOVE |G|=0.5 where a criterion breaks (|G|<=0.5 is the
        # weak-signal regime: near-constant lift makes absolute R^2 unreliable, so
        # it is excluded from the breakdown search).
        div_break = None
        clo_break_filter = None
        clo_break_forecast = None
        clo_break_recovery = None
        for gname, row in rows:
            if row is None or row["abs_G"] <= 0.5:
                continue
            if div_break is None and row["div_rate"] > 0.5:
                div_break = row["abs_G"]
            if clo_break_filter is None and (
                row["filter_CL_analysis_r2_impact_median"] is not None
                and row["filter_CL_analysis_r2_impact_median"] < 0.0
            ):
                clo_break_filter = row["abs_G"]
            if clo_break_forecast is None and (
                row["forecast_CL_r2_impact_median"] is not None
                and row["forecast_CL_r2_impact_median"] < 0.0
            ):
                clo_break_forecast = row["abs_G"]
            if clo_break_recovery is None and (
                row["recovery_CL_r2_impact_median"] is not None
                and row["recovery_CL_r2_impact_median"] < 0.0
            ):
                clo_break_recovery = row["abs_G"]
        thresholds[dname] = {
            "ladder": [r for _, r in rows if r is not None],
            "div_rate_exceeds_50pct_at_absG": div_break,
            "filter_CL_closure_below0_at_absG": clo_break_filter,
            "forecast_CL_closure_below0_at_absG": clo_break_forecast,
            "recovery_CL_closure_below0_at_absG": clo_break_recovery,
        }
    return thresholds


# ---------------------------------------------------------------- main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Session 32 gust-intensity envelope analysis")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--models", nargs="+", default=["jepa_pool", "pod"])
    p.add_argument("--limit", type=int, default=0, help="cap encounters for a smoke (0=all)")
    p.add_argument("--partition", default="v2p2")
    p.add_argument("--pipeline-manifest", default="outputs/data_pipeline/v2p2/manifest.json")
    p.add_argument("--split", default="configs/splits/split_v2p2.json")
    p.add_argument("--windows", default="outputs/session31/windows_v2p2.json")
    p.add_argument("--checkpoint", default="checkpoint_iter010000.pt")
    p.add_argument("--osp-taps", default="outputs/session32/osp_taps_v2p2.json")
    p.add_argument("--taps", default="outputs/session32/qdeim_taps_v2p2.json")
    p.add_argument("--out", default="outputs/session32/envelope_by_gust.json")
    p.add_argument("--seed", type=int, default=0)
    # frozen filter params (filter_tuning_frozen.json) -- do NOT change.
    p.add_argument("--rho", type=float, default=1.0)
    p.add_argument("--taps-mode", default="osp_per_model")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--n-members", type=int, default=64)
    p.add_argument("--mode", default="stochastic")
    p.add_argument("--predictor-steps", type=int, default=4000)
    return p.parse_args(argv)


MODEL_RUN = {
    "jepa_pool": (
        "outputs/runs/session31/jepa_pool",
        False,
        "outputs/session31/q1_latents_ablation/latents_%s_train.npz",
    ),
    "pod": (
        "outputs/runs/session31/pod",
        True,
        "outputs/session31/q1_latents/latents_%s_train.npz",
    ),
}


def main(argv=None):
    import torch

    a = parse_args(argv)
    from src.utils.device import require_rtx6000

    device = require_rtx6000(gpu_index=a.gpu)
    gpu_name = torch.cuda.get_device_name(device.index)
    if "RTX" not in gpu_name or "6000" not in gpu_name:
        raise RuntimeError(f"hardware policy: gpu_name={gpu_name!r} is not an RTX 6000")
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    windows = json.loads(_resolve(a.windows).read_text())["windows"]
    pipe = OmegaPipeline.from_manifest(_resolve(a.pipeline_manifest))
    cache_dir = _cache_dir(a.partition)
    encs_meta = enumerate_encounters(_resolve(a.split))
    if a.limit:
        encs_meta = encs_meta[: a.limit]
    print(
        f"[env] device={device} ({gpu_name}) | {len(encs_meta)} encounters | models={a.models}",
        flush=True,
    )

    payload = {
        "task": "SESSION 32 -- gust-intensity operating-envelope analysis",
        "frozen_filter": json.loads(
            _resolve("outputs/session32/filter_tuning_frozen.json").read_text()
        )["frozen"],
        "convention": (
            "G_inv = -G_phys (HANDOFF D224); |G| buckets sign-agnostic; "
            "sign reported in inventory G"
        ),
        "params": {
            "gpu_name": gpu_name,
            "partition": a.partition,
            "seed": a.seed,
            "rho": a.rho,
            "taps_mode": a.taps_mode,
            "K": a.k,
            "n_members": a.n_members,
            "mode": a.mode,
            "predictor_steps": a.predictor_steps,
            "assim_window": {"pre": PRE, "post": POST},
            "o1_window": O1_WINDOW,
            "phases": {
                "impact": "[t_imp, t_imp+16)",
                "relax(post_impact)": "[t_imp+16, t_imp+48)",
            },
            "forecast_lens": (
                "truthinit = truth-init open-loop matched AR-transformer (primary); "
                "fieldfree = Track-B field-free-init open-loop ensemble baseline"
            ),
        },
        "models": {},
    }

    # args namespace consumed by build_context / run_encounter.
    ctx_args = SimpleNamespace(
        train_cache=None,
        partition=a.partition,
        predictor_steps=a.predictor_steps,
        seed=a.seed,
        taps_mode=a.taps_mode,
        k=a.k,
        osp_taps=a.osp_taps,
        taps=a.taps,
        rho=a.rho,
        n_members=a.n_members,
        mode=a.mode,
    )

    out_path = _resolve(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for name in a.models:
        run_dir, is_ref, cache_tmpl = MODEL_RUN[name]
        ctx_args.train_cache = cache_tmpl
        print(f"\n[env] ===== building context for {name} =====", flush=True)
        t0 = time.time()
        ctx = build_context(name, run_dir, is_ref, device, ctx_args, train_cache_tmpl=cache_tmpl)
        taps = ctx["taps"][0]
        o1_est = build_o1_recovery(name, ctx_args, taps)
        frozen = (load_reference_model if is_ref else load_frozen_model)(
            _resolve(run_dir), a.checkpoint, device
        )
        print(f"[env] context built in {time.time()-t0:.0f}s; taps={list(taps)}", flush=True)

        records: list[dict] = []
        t1 = time.time()
        for i, m in enumerate(encs_meta):
            try:
                enc = encode_encounters(
                    frozen, [(m["case_id"], m["encounter_index"])], pipe, cache_dir, windows, device
                )[0]
                filt = run_encounter(ctx, enc, ctx_args, rho=a.rho)
                fct = forecast_truthinit(enc, ctx)
                rec = recovery_for_encounter(o1_est, taps, enc, ctx, ctx_args)
                records.append(flatten_record(m, filt, rec, fct))
            except Exception as e:  # noqa: BLE001
                print(
                    f"[env] FAIL {m['case_id']}/{m['encounter_index']}: {type(e).__name__}: {e}",
                    flush=True,
                )
            if (i + 1) % 20 == 0 or (i + 1) == len(encs_meta):
                rate = (i + 1) / (time.time() - t1)
                print(f"[env] {name}: {i+1}/{len(encs_meta)} ({rate:.2f}/s)", flush=True)
                payload["models"][name] = {
                    "run_dir": run_dir,
                    "taps": [int(x) for x in taps],
                    "n_records": len(records),
                    "records": records,
                    "aggregates": aggregate(records),
                    "thresholds": identify_thresholds(records),
                }
                out_path.write_text(json.dumps(payload, indent=2, default=float))
        print(f"[env] {name} done: {len(records)} records in {time.time()-t1:.0f}s", flush=True)

    print(f"\n[env] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
