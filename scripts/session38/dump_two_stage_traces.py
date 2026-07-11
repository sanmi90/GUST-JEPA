"""Dump per-frame traces of the \\TwoStageKF (headline filter) for the C_L
envelope figure.

Reuses the frozen two-stage machinery of scripts/session35/two_stage_envelope
verbatim (build_context, rex_step, load_split; band c = 1.77, N = 64,
DELAY = 10, the cache window_mask as the schedule boundary) and replays
run_encounter's loop for the requested encounters while RECORDING the
per-frame analysis mean, the C_L ensemble spread, and an open-loop companion
(same stage-scheduled forecast models, no correction). Standalone rng (seed
0): values match the frozen aggregates to sampling noise, not bit-exactly
(the frozen runs thread one generator across the whole split).

Run (RTX 6000):
    taskset -c 0-15 python -m scripts.session38.dump_two_stage_traces --gpu 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.session34.lae_enkf_pilot import delay_embed, encounters  # noqa: E402
from scripts.session35.two_stage_envelope import (  # noqa: E402
    DELAY,
    MAX_CTX,
    N_MEMBERS,
    build_context,
    load_split,
    rex_step,
)

BAND = 1.77
IMPACT_FRAME = 40
DT_TC = 0.05

PICKS = [  # (case_id, encounter_index, split)
    ("G+1.00_D0.50_Y+0.40", 3, "test_b"),
    ("G+2.00_D0.50_Y+0.10", 1, "test_b"),
    ("G+3.00_D1.00_Y+0.10", 0, "test_b"),
    ("G+4.00_D0.50_Y-0.10", 4, "test_c"),
]


def run_traced(ctx, data, e, rng):
    """two_stage_envelope.run_encounter's loop, recording series + open loop."""
    n = ctx["n"]
    rows = e["rows"]
    cl_true = data["cl"][rows]
    wmask = data["wmask"][rows]
    pt = (data["p"][rows][:, ctx["taps"]] - ctx["p_mu"]) / ctx["p_sd"]
    z_obs = delay_embed(pt, DELAY) @ ctx["W"]
    T = rows.size
    t_init = DELAY - 1
    init = np.repeat(z_obs[None, : t_init + 1], N_MEMBERS, axis=0)
    init[:, -1] += rng.standard_normal((N_MEMBERS, n)) @ ctx["chol_g"].T

    def propagate(update: bool, rng_arm):
        mctx = init.copy()
        zA = np.empty((T, n))
        zA[: t_init + 1] = z_obs[: t_init + 1]
        cl_std = np.zeros(T)
        for t in range(t_init + 1, T):
            if bool(wmask[t]):
                med, sig = rex_step(ctx, mctx, BAND)
                zf = med + (rng_arm.standard_normal((N_MEMBERS, n)) * sig
                            if update else 0.0)
            else:
                prior = np.asarray(ctx["fmodel"].step(mctx.astype(np.float32)),
                                   dtype=np.float64)
                zf = prior + (rng_arm.standard_normal((N_MEMBERS, n)) @ ctx["chol_q"].T
                              if update else 0.0)
            if update and bool(wmask[t]):
                dZ = zf - zf.mean(0, keepdims=True)
                P = dZ.T @ dZ / (N_MEMBERS - 1)
                S = P + ctx["Gam"]
                K_g = P @ np.linalg.inv(S)
                innov = (z_obs[t][None]
                         + rng_arm.standard_normal((N_MEMBERS, n)) @ ctx["chol_g"].T
                         - zf)
                za = zf + innov @ K_g.T
            elif update:
                yf = zf @ ctx["C_map"]
                dZ = zf - zf.mean(0, keepdims=True)
                dY = yf - yf.mean(0, keepdims=True)
                S_y = dY.T @ dY / (N_MEMBERS - 1) + ctx["R_taps"]
                K_g = (dZ.T @ dY / (N_MEMBERS - 1)) @ np.linalg.inv(S_y)
                innov = (pt[t][None]
                         + rng_arm.standard_normal(
                             (N_MEMBERS, len(ctx["taps"]))) @ ctx["chol_R"].T
                         - yf)
                za = zf + innov @ K_g.T
            else:
                za = zf
            zA[t] = za.mean(0)
            cl_members = ctx["probe"].predict(za)
            cl_std[t] = float(np.std(cl_members))
            mctx = np.concatenate([mctx, za[:, None]], axis=1)[:, -MAX_CTX:]
        return ctx["probe"].predict(zA), cl_std

    cl_filt, cl_filt_std = propagate(True, np.random.default_rng(rng.integers(2**32)))
    cl_open, _ = propagate(False, np.random.default_rng(rng.integers(2**32)))
    return {
        "case_id": e["case_id"], "encounter_index": e["k"],
        "t_impact": IMPACT_FRAME, "frames": list(range(T)), "dt_tc": DT_TC,
        "truth": {"C_L": cl_true.tolist()},
        "filter_analysis": {"C_L": cl_filt.tolist(),
                            "C_L_ens_std": cl_filt_std.tolist()},
        "open_loop": {"C_L": cl_open.tolist()},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/session38/two_stage_traces.json")
    args = ap.parse_args()

    from src.utils.device import require_rtx6000

    device = require_rtx6000(gpu_index=args.gpu)
    import torch

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    ctx = build_context(device)

    traces = []
    for case_id, k, split in PICKS:
        data = load_split(split)
        e = next(e for e in encounters(data)
                 if e["case_id"] == case_id and e["k"] == k)
        tr = run_traced(ctx, data, e, rng)
        g = abs(float(case_id.split("_")[0][1:]))
        tr["envelope_record"] = {"abs_G": g, "split": split,
                                 "case_id": case_id}
        traces.append(tr)
        print(f"[two-stage traces] {case_id}/{k} ({split}) done", flush=True)

    out = REPO_ROOT / args.out
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "task": "SESSION 38 -- two-stage filter C_L traces (headline filter)",
        "params": {"model": "jepa_pool_vec", "filter": "two_stage",
                   "band": BAND, "n_members": N_MEMBERS, "delay": DELAY,
                   "seed": args.seed,
                   "selection": "median CL_analysis_r2_impact per |G| stratum "
                                "on the frozen two_stage_addendum records "
                                "(test_b b177 for |G|<=3, test_c b177 at 4)",
                   "note": "standalone rng; matches frozen aggregates to "
                           "sampling noise, not bit-exactly; the open-loop "
                           "arm is the DETERMINISTIC stage-scheduled rollout "
                           "(no process noise, no correction)"},
        "traces": traces,
    }, indent=1))
    print(f"[two-stage traces] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
