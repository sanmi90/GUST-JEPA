"""Dump per-frame assimilation traces for the F7 hero figure.

SESSION_33_MANUSCRIPT_V3.md Section 9 (F7): truth vs open-loop vs pressure-only
vs filter analysis, per frame, for representative held-out encounters. The
frozen envelope run stores per-encounter aggregates only, so this re-runs the
FROZEN filter (D220 verbatim: rho=1.0, K=8, osp_per_model, N=64, stochastic,
field-free init) on a handful of test_b encounters and dumps the series.

Case selection follows the representative-case rule (memory: pick a typical
low-error case, not the best or the hardest): per |G| stratum in {1, 1.5, 2},
the test_b encounter whose filter C_L analysis R2 (impact) is CLOSEST TO THE
STRATUM MEDIAN in the frozen envelope records. Override with --encounters.

Run (RTX 6000):
    taskset -c 0-15 python -m scripts.session33.dump_hero_traces --gpu 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

from src.data.omega_pipeline import OmegaPipeline  # noqa: E402
from src.estimation import enkf as ek  # noqa: E402
from src.evaluation.rom_eval import load_frozen_model  # noqa: E402
from scripts.session32.envelope_by_gust import (  # noqa: E402
    O1_WINDOW,
    _resolve,
    build_o1_recovery,
)
from scripts.session32.track_b_pilot import (  # noqa: E402
    INIT_WINDOW,
    POST,
    PRE,
    _cache_dir,
    build_context,
    encode_encounters,
    windowed_features,
)

GATE_STRATA = (1.0, 1.5, 2.0)


def pick_representative(env_records: list[dict]) -> list[tuple[str, int]]:
    picks = []
    for g in GATE_STRATA:
        sub = [
            r for r in env_records
            if r["split"] == "test_b" and abs(r["abs_G"] - g) < 0.05
            and np.isfinite(r["filter"]["CL_analysis_r2_impact"])
        ]
        if not sub:
            continue
        vals = np.array([r["filter"]["CL_analysis_r2_impact"] for r in sub])
        med = np.median(vals)
        r = sub[int(np.argmin(np.abs(vals - med)))]
        picks.append((r["case_id"], int(r["encounter_index"])))
    return picks


def trace_encounter(ctx, o1_est, taps, enc, args) -> dict:
    """Frozen-filter run keeping the per-frame readout SERIES (F7 needs them)."""
    H = ctx["H"]
    d = int(enc["z_gap"].shape[1])
    t_imp = enc["t_impact"]
    f0 = max(0, t_imp - PRE)
    f1 = min(enc["n_frames"] - 1, t_imp + POST)
    frames = np.arange(f0, f1 + 1)
    T = len(frames)

    y_series = H.select_taps(enc["p_wall"][frames])
    rng = np.random.default_rng(args.seed + 1)
    win0 = windowed_features(enc["p_wall"], INIT_WINDOW)[f0]
    init_ens = ctx["init"].sample(win0, args.n_members, rng)

    filt = ek.EnsembleKalmanFilter(
        ctx["fmodel"], H, ctx["Q"], n_members=args.n_members,
        inflation=args.rho, mode=args.mode, seed=args.seed + 2,
    )
    res = filt.run(y_series, init_ens, frames, meta={"case": enc["case_id"]})

    # open-loop free run (same init, no correction), as in track_b_pilot.
    ol = ek.EnsembleKalmanFilter(
        ctx["fmodel"], H, ctx["Q"], n_members=args.n_members,
        inflation=args.rho, mode=args.mode, seed=args.seed + 3,
    )
    ol_states = np.empty((T, args.n_members, d))
    ol_states[0] = init_ens
    buf = init_ens[:, None, :]
    for t in range(1, T):
        prior = ol.forecast(buf)
        ol_states[t] = prior
        buf = np.concatenate([buf, prior[:, None, :]], axis=1)
        if buf.shape[1] > filt.max_context:
            buf = buf[:, -filt.max_context:, :]

    # pressure-only (static O1) recovery series at the same taps.
    p_osp = enc["p_wall"][:, np.sort(taps)]
    z_rec = np.asarray(o1_est.predict(windowed_features(p_osp, O1_WINDOW)[frames]))

    def read(states, probe):
        return np.asarray(probe.predict(states), dtype=np.float64).reshape(-1)

    ens_cl = np.stack([ctx["probe_cl"].predict(res.analysis_ens[t]) for t in range(T)])
    ens_ew = np.stack([ctx["probe_ew"].predict(res.analysis_ens[t]) for t in range(T)])

    return {
        "case_id": enc["case_id"],
        "encounter_index": int(enc["encounter_index"]),
        "t_impact": int(t_imp),
        "frames": [int(f) for f in frames],
        "dt_tc": 0.05,
        "truth": {
            "C_L": [float(v) for v in enc["C_L"][frames]],
            "wake_enstrophy": [float(v) for v in enc["wake_enstrophy"][frames]],
        },
        "filter_analysis": {
            "C_L": [float(v) for v in read(res.analysis_mean, ctx["probe_cl"])],
            "wake_enstrophy": [float(v) for v in read(res.analysis_mean, ctx["probe_ew"])],
            "C_L_ens_std": [float(v) for v in ens_cl.std(axis=1)],
            "wake_enstrophy_ens_std": [float(v) for v in ens_ew.std(axis=1)],
        },
        "open_loop": {
            "C_L": [float(v) for v in read(ol_states.mean(axis=1), ctx["probe_cl"])],
            "wake_enstrophy": [float(v) for v in read(ol_states.mean(axis=1), ctx["probe_ew"])],
        },
        "pressure_only": {
            "C_L": [float(v) for v in read(z_rec, ctx["probe_cl"])],
            "wake_enstrophy": [float(v) for v in read(z_rec, ctx["probe_ew"])],
        },
    }


def main(argv=None):
    import torch

    ap = argparse.ArgumentParser(description="F7 hero assimilation traces")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--model", default="jepa_pool")
    ap.add_argument("--run-dir", default="outputs/runs/session31/jepa_pool")
    ap.add_argument("--checkpoint", default="checkpoint_iter010000.pt")
    ap.add_argument("--envelope-json", default="outputs/session32/envelope_by_gust.json")
    ap.add_argument(
        "--encounters", nargs="+", default=None,
        help="override case/enc picks, e.g. G+1.00_D0.50_Y+0.10:4",
    )
    ap.add_argument("--partition", default="v2p2")
    ap.add_argument("--pipeline-manifest", default="outputs/data_pipeline/v2p2/manifest.json")
    ap.add_argument("--windows", default="outputs/session31/windows_v2p2.json")
    ap.add_argument("--osp-taps", default="outputs/session32/osp_taps_v2p2.json")
    ap.add_argument(
        "--train-cache",
        default="outputs/session31/q1_latents_ablation/latents_%s_train.npz",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/session33/hero_traces.json")
    # frozen filter params (D220) -- do NOT change.
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--n-members", type=int, default=64)
    ap.add_argument("--mode", default="stochastic")
    ap.add_argument("--predictor-steps", type=int, default=4000)
    args = ap.parse_args(argv)

    from src.utils.device import require_rtx6000

    device = require_rtx6000(gpu_index=args.gpu)
    gpu_name = torch.cuda.get_device_name(device.index)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = json.loads(_resolve(args.envelope_json).read_text())
    if args.encounters:
        picks = []
        for tok in args.encounters:
            cid, k = tok.rsplit(":", 1)
            picks.append((cid, int(k)))
    else:
        picks = pick_representative(env["models"][args.model]["records"])
    print(f"[hero] device={device} ({gpu_name}) picks={picks}", flush=True)

    windows = json.loads(_resolve(args.windows).read_text())["windows"]
    pipe = OmegaPipeline.from_manifest(_resolve(args.pipeline_manifest))
    ctx_args = SimpleNamespace(
        train_cache=args.train_cache, partition=args.partition,
        predictor_steps=args.predictor_steps, seed=args.seed,
        taps_mode="osp_per_model", k=8, osp_taps=args.osp_taps,
        taps=args.osp_taps, rho=args.rho, n_members=args.n_members, mode=args.mode,
    )
    ctx = build_context(args.model, args.run_dir, False, device, ctx_args,
                        train_cache_tmpl=args.train_cache)
    taps = ctx["taps"][0]
    o1_est = build_o1_recovery(args.model, ctx_args, taps)
    frozen = load_frozen_model(_resolve(args.run_dir), args.checkpoint, device)

    traces = []
    for cid, k in picks:
        enc = encode_encounters(frozen, [(cid, k)], pipe, _cache_dir(args.partition),
                                windows, device)[0]
        tr = trace_encounter(ctx, o1_est, taps, enc, args)
        env_rec = next(
            (r for r in env["models"][args.model]["records"]
             if r["case_id"] == cid and r["encounter_index"] == k), None,
        )
        tr["envelope_record"] = env_rec
        traces.append(tr)
        print(f"[hero] dumped {cid}/{k} (t_imp={tr['t_impact']}, T={len(tr['frames'])})",
              flush=True)

    payload = {
        "task": "SESSION 33 -- F7 hero assimilation traces (frozen filter, D220)",
        "params": {
            "gpu_name": gpu_name,
            "model": args.model,
            "selection": (
                "per |G| in {1, 1.5, 2}: test_b encounter with filter C_L analysis "
                "R2 (impact) closest to the stratum median (representative-case rule)"
            ),
            "frozen_filter": {"rho": args.rho, "K": 8, "taps_mode": "osp_per_model",
                              "n_members": args.n_members, "mode": args.mode},
            "seed": args.seed,
        },
        "traces": traces,
    }
    out = _resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=float))
    print(f"[hero] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
