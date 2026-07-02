"""Track T2b -- the reduced-budget filter: frozen EnKF at K < 8 taps.

SESSION_33_MANUSCRIPT_V3.md Section 11 item 11 (T2b); HANDOFF D238/D239.

Runs the FROZEN Track B filter (filter_tuning_frozen.json: rho=1.0, N=64,
stochastic, field-free O1 init, osp_per_model taps) on all encounters with ONLY
the tap count changed: K in (by default) {1, 2, 4}, taps = the model's own nested
OSP staircase prefixes (K1 from the derived taps_v2p2_ext.json; the frozen
session32 taps files are untouched). Nothing else is re-tuned. Like the envelope
run this is a CHARACTERISATION with a frozen filter, which is why touching test_c
is legitimate (D236 precedent).

The filter has no W knob: sequentiality carries the delays (the assimilation
window is the delay window). W enters only through the static-recovery grid
(track_t_recovery_grid) and the theory bound (track_t3).

Gate T2b (filter clause of Gate T2): at the grid-selected K_min, the reduced-K
filter's per-encounter analysis C_L closure (impact) matches the frozen K=8
filter within a case-clustered CI (delta CI includes 0) on each stratum |G| in
{1, 1.5, 2}. NIS / divergence at reduced K are reported descriptively only.

Run (RTX 6000 + CPU cap; ~10 min per K):
    taskset -c 0-15 python -m scripts.session33.track_t2b_reduced_filter --gpu 0
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
from src.estimation.obs_operator import fit_observation_operator, load_osp_taps  # noqa: E402
from src.evaluation.pressure_infer import load_pressure_for_alignment  # noqa: E402
from src.evaluation.rom_eval import load_frozen_model  # noqa: E402
from scripts.session32.envelope_by_gust import (  # noqa: E402
    GBUCKETS,
    _resolve,
    aggregate,
    enumerate_encounters,
    identify_thresholds,
)
from scripts.session32.track_b_pilot import (  # noqa: E402
    _cache_dir,
    build_context,
    encode_encounters,
    run_encounter,
)

GATE_STRATA = ("1", "1.5", "2")
PAIR_METRICS = (
    "CL_analysis_r2_impact",
    "CL_analysis_r2_relax",
    "Ew_analysis_r2_impact",
    "Ew_analysis_r2_relax",
)


def refit_H(name: str, taps, tap_prov, train_cache_tmpl: str, partition: str, seed: int):
    """Refit ONLY the observation operator at the reduced tap set (build_context's
    H block verbatim: same train cache, same seeded encounter-grouped val split)."""
    cache = np.load(_resolve(train_cache_tmpl % name), allow_pickle=True)
    z_gap_tr = np.asarray(cache["z_gap"], dtype=np.float32)
    cid_tr = np.asarray(cache["case_id"]).astype(str)
    enc_tr = np.asarray(cache["encounter_index"])
    frame_tr = np.asarray(cache["frame"])
    p_tr = load_pressure_for_alignment(cid_tr, enc_tr, frame_tr, partition=partition)
    groups = np.array([f"{c}/{int(e)}" for c, e in zip(cid_tr, enc_tr)])
    ug = np.unique(groups)
    rng = np.random.default_rng(seed)
    val_g = set(rng.choice(ug, size=max(1, len(ug) // 5), replace=False).tolist())
    vmask = np.array([g in val_g for g in groups])
    return fit_observation_operator(
        z_gap_tr[~vmask],
        p_tr[~vmask],
        taps,
        kind="linear",
        z_val=z_gap_tr[vmask],
        p_val=p_tr[vmask],
        provenance=tap_prov,
    )


def flatten_filter_record(meta: dict, filt: dict) -> dict:
    fcl = filt["closure"]["C_L"]
    few = filt["closure"]["wake_enstrophy"]

    def _r2(clo, phase):
        return float(clo.get(phase, {}).get("r2", float("nan")))

    return {
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
            "Ew_gain_rmse_impact": float(
                few["gain_vs_open_loop"].get("impact", {}).get("rmse_reduction", float("nan"))
            ),
        },
        # forecast / recovery lenses unchanged vs K=8 (no tap dependence); omitted.
        "forecast": {
            "truthinit_CL_r2_impact": float("nan"),
            "truthinit_CL_r2_relax": float("nan"),
            "truthinit_Ew_r2_impact": float("nan"),
            "fieldfree_CL_r2_impact": _r2(fcl["open_loop"], "impact"),
        },
        "recovery": {
            "state_r2": float("nan"),
            "CL_r2_impact": float("nan"),
            "CL_r2_relax": float("nan"),
            "Ew_r2_impact": float("nan"),
        },
    }


def paired_delta_by_case(reduced, frozen, metric, mask_fn, *, n_boot, seed):
    """Case-clustered bootstrap CI of the mean per-encounter delta
    (reduced - frozen) for one filter metric over encounters passing mask_fn."""
    froz = {(r["case_id"], r["encounter_index"]): r for r in frozen}
    deltas, cases = [], []
    for r in reduced:
        if not mask_fn(r):
            continue
        f = froz.get((r["case_id"], r["encounter_index"]))
        if f is None:
            continue
        a, b = r["filter"][metric], f["filter"][metric]
        if np.isfinite(a) and np.isfinite(b):
            deltas.append(a - b)
            cases.append(r["case_id"])
    if not deltas:
        return {"n_pairs": 0}
    deltas = np.asarray(deltas)
    cases = np.asarray(cases)
    uc = np.array(sorted(set(cases.tolist())))
    by = {c: np.where(cases == c)[0] for c in uc}
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        pick = uc[rng.integers(0, len(uc), size=len(uc))]
        idx = np.concatenate([by[c] for c in pick])
        boots[i] = deltas[idx].mean()
    return {
        "n_pairs": int(len(deltas)),
        "n_cases": int(len(uc)),
        "mean_delta": float(deltas.mean()),
        "median_delta": float(np.median(deltas)),
        "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Track T2b reduced-budget frozen filter")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--model", default="jepa_pool")
    p.add_argument("--ks", nargs="+", type=int, default=None,
                   help="reduced tap counts; default = {K_min from the grid} U {2, 4}")
    p.add_argument("--grid-json", default="outputs/session33/track_t_recovery_grid.json")
    p.add_argument("--frozen-envelope", default="outputs/session32/envelope_by_gust.json")
    p.add_argument("--osp-taps", default="outputs/session33/taps_v2p2_ext.json")
    p.add_argument("--limit", type=int, default=0, help="cap encounters for a smoke (0=all)")
    p.add_argument("--partition", default="v2p2")
    p.add_argument("--pipeline-manifest", default="outputs/data_pipeline/v2p2/manifest.json")
    p.add_argument("--split", default="configs/splits/split_v2p2.json")
    p.add_argument("--windows", default="outputs/session31/windows_v2p2.json")
    p.add_argument("--checkpoint", default="checkpoint_iter010000.pt")
    p.add_argument("--run-dir", default="outputs/runs/session31/jepa_pool")
    p.add_argument(
        "--train-cache",
        default="outputs/session31/q1_latents_ablation/latents_%s_train.npz",
    )
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="outputs/session33/t2b_reduced_filter.json")
    # frozen filter params (filter_tuning_frozen.json) -- do NOT change.
    p.add_argument("--rho", type=float, default=1.0)
    p.add_argument("--n-members", type=int, default=64)
    p.add_argument("--mode", default="stochastic")
    p.add_argument("--predictor-steps", type=int, default=4000)
    return p.parse_args(argv)


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

    # K list: grid selection (train-rows-only) plus the F8 overlay columns.
    grid = json.loads(_resolve(a.grid_json).read_text())
    sel = grid.get("t2b_selection") or {}
    k_min = sel.get("K_min")
    if a.ks is not None:
        ks = sorted(set(int(k) for k in a.ks))
    else:
        ks = sorted({k for k in ([k_min] if k_min else []) + [2, 4] if k and k < 8})
    print(f"[t2b] grid K_min={k_min} W_min={sel.get('W_min')} -> filter Ks={ks}", flush=True)

    frozen_env = json.loads(_resolve(a.frozen_envelope).read_text())
    frozen_records = frozen_env["models"][a.model]["records"]

    windows = json.loads(_resolve(a.windows).read_text())["windows"]
    pipe = OmegaPipeline.from_manifest(_resolve(a.pipeline_manifest))
    cache_dir = _cache_dir(a.partition)
    encs_meta = enumerate_encounters(_resolve(a.split))
    if a.limit:
        encs_meta = encs_meta[: a.limit]

    ctx_args = SimpleNamespace(
        train_cache=a.train_cache,
        partition=a.partition,
        predictor_steps=a.predictor_steps,
        seed=a.seed,
        taps_mode="osp_per_model",
        k=8,
        osp_taps=a.osp_taps,
        taps=a.osp_taps,
        rho=a.rho,
        n_members=a.n_members,
        mode=a.mode,
    )

    print(f"[t2b] device={device} ({gpu_name}) | {len(encs_meta)} encounters", flush=True)
    t0 = time.time()
    # ONE context (predictor, Q, init, probes are K-independent); H swapped per K.
    ctx = build_context(a.model, a.run_dir, False, device, ctx_args,
                        train_cache_tmpl=a.train_cache)
    frozen_model = load_frozen_model(_resolve(a.run_dir), a.checkpoint, device)
    print(f"[t2b] context built in {time.time()-t0:.0f}s", flush=True)

    payload = {
        "task": "SESSION 33 Track T2b -- reduced-budget frozen filter",
        "frozen_filter": json.loads(
            _resolve("outputs/session32/filter_tuning_frozen.json").read_text()
        )["frozen"],
        "params": {
            "gpu_name": gpu_name,
            "model": a.model,
            "reduced_Ks": ks,
            "taps": "osp_per_model nested prefixes (K1 from taps_v2p2_ext.json, D239)",
            "grid_selection": sel,
            "n_boot_case": a.n_boot,
            "seed": a.seed,
            "note": (
                "only K/taps change vs the frozen filter; no re-tuning. The filter "
                "has no W knob: sequentiality carries the delays. NIS/divergence at "
                "reduced K are descriptive only. Characterisation run (test_c "
                "legitimate, D236 precedent)."
            ),
        },
        "by_K": {},
        "gate_T2b": None,
    }
    out_path = _resolve(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # encode once, filter per K.
    for kk in ks:
        taps, tap_prov = load_osp_taps(a.model, k=kk, osp_path=a.osp_taps)
        ctx["H"] = refit_H(a.model, taps, tap_prov, a.train_cache, a.partition, a.seed)
        records = []
        t1 = time.time()
        for i, m in enumerate(encs_meta):
            try:
                enc = encode_encounters(
                    frozen_model, [(m["case_id"], m["encounter_index"])],
                    pipe, cache_dir, windows, device,
                )[0]
                filt = run_encounter(ctx, enc, ctx_args, rho=a.rho)
                records.append(flatten_filter_record(m, filt))
            except Exception as e:  # noqa: BLE001
                print(
                    f"[t2b] FAIL K{kk} {m['case_id']}/{m['encounter_index']}: "
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )
            if (i + 1) % 50 == 0 or (i + 1) == len(encs_meta):
                print(
                    f"[t2b] K{kk}: {i+1}/{len(encs_meta)} "
                    f"({(i+1)/(time.time()-t1):.2f}/s)",
                    flush=True,
                )
        paired = {}
        for gname, gfn in GBUCKETS:
            block = {}
            for metric in PAIR_METRICS:
                block[metric] = paired_delta_by_case(
                    records, frozen_records, metric,
                    lambda r, fn=gfn: fn(r["abs_G"]),
                    n_boot=a.n_boot, seed=a.seed,
                )
            paired[gname] = block
        payload["by_K"][str(kk)] = {
            "taps": [int(t) for t in taps],
            "n_records": len(records),
            "records": records,
            "aggregates": aggregate(records),
            "thresholds": identify_thresholds(records),
            "paired_vs_frozen_K8": paired,
        }
        out_path.write_text(json.dumps(payload, indent=2, default=float))
        print(f"[t2b] K{kk} done: {len(records)} records in {time.time()-t1:.0f}s", flush=True)

    # ---- Gate T2b at the grid-selected K_min (fallback: smallest K run)
    k_gate = k_min if (k_min and str(k_min) in payload["by_K"]) else ks[0]
    gate_rows = {}
    ok = True
    for g in GATE_STRATA:
        d = payload["by_K"][str(k_gate)]["paired_vs_frozen_K8"][g]["CL_analysis_r2_impact"]
        within = bool(d.get("n_pairs", 0) > 0 and d["ci95"][0] <= 0.0 <= d["ci95"][1] or
                      (d.get("n_pairs", 0) > 0 and d["ci95"][0] > 0.0))
        gate_rows[g] = {**d, "within_ci_or_better": within}
        ok = ok and within
    payload["gate_T2b"] = {
        "K_gate": int(k_gate),
        "strata": gate_rows,
        "pass": bool(ok),
        "criterion": (
            "per-stratum paired delta CI (reduced - frozen K8) of analysis C_L R2 "
            "at impact includes 0 or is positive, for |G| in {1, 1.5, 2}"
        ),
    }
    print(
        f"[t2b] GATE T2b @K{k_gate}: {'PASS' if ok else 'FAIL'} "
        + " ".join(
            f"|G|={g}: d={gate_rows[g].get('mean_delta', float('nan')):+.3f} "
            f"CI{[round(x, 3) for x in gate_rows[g].get('ci95', [float('nan')]*2)]}"
            for g in GATE_STRATA
        ),
        flush=True,
    )
    out_path.write_text(json.dumps(payload, indent=2, default=float))
    print(f"[t2b] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
