"""3-seed variance pass on the spine pair + P1/P2 re-check (item 8).

SESSION_33_MANUSCRIPT_V3.md Section 11 item 8 (appendix; the rest of the matrix
stays 1-seed with the community-standard justification).

Seeds: s0 = the frozen Session 31/32 runs (jepa_pool, supervised_only_pool);
s1/s2 = the Session 33 retrains. Reports per-seed cells (wake readability,
merit_mean_obs_h8, C_L closure, decode SSIM), seed mean +- sd, and re-evaluates
the P1 tie (|delta| < 0.05 on seed means) and the P2 ordering clauses that the
seed band can reach (jepa_wake >= supervised_only within tolerance).

Run (CPU):
    taskset -c 16-23 python -m scripts.session33.seed_band_v3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

from scripts.session32.track_p_gates import q1_cells, q2_cells  # noqa: E402

FAMS = {
    "jepa_wake_pool": {
        0: ("jepa_pool", "anchor_abl"),
        1: ("jepa_pool_s1", "s33"),
        2: ("jepa_pool_s2", "s33"),
    },
    "supervised_only_pool": {
        0: ("supervised_only_pool", "anchor_pool"),
        1: ("supervised_only_pool_s1", "s33"),
        2: ("supervised_only_pool_s2", "s33"),
    },
    # D247 audit: the merit-ordering claim (P2) rested on one seed of the
    # matched reconstructive control; its band closes that gap.
    "ae_wake_pool": {
        0: ("ae_wake_pool", "anchor_pool"),
        1: ("ae_wake_pool_s1", "s33aw"),
        2: ("ae_wake_pool_s2", "s33aw"),
    },
    # D250 native pooled pipeline: the vec flagship's own 3-seed band.
    "jepa_wake_pool_vec": {
        0: ("jepa_pool_vec", "vec"),
        1: ("jepa_pool_vec_s1", "vec"),
        2: ("jepa_pool_vec_s2", "vec"),
    },
}


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def _load(p):
    return json.loads(_resolve(p).read_text())


def cells_for(model: str, src: str, ctx: dict) -> dict:
    q1, q2 = ctx[src]
    out = q1_cells(q1, model)
    try:
        out.update(q2_cells(q2, model))
    except KeyError:
        out["note_q2"] = "missing"
    wake = q1["models"][model]["probes"]["windowed"]["wake_enstrophy"]
    out["wake_linear_r2"] = float(wake["linear_r2"])
    return out


def band(vals):
    a = np.array([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
    if a.size == 0:
        return {"mean": None, "sd": None, "n": 0, "values": list(vals)}
    return {
        "mean": float(a.mean()),
        "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "n": int(a.size),
        "values": [float(v) for v in vals],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="3-seed spine variance pass")
    ap.add_argument("--q1-seeds", default="outputs/session33/q1_seeds.json")
    ap.add_argument("--q2-seeds", default="outputs/session33/q2_seeds.json")
    ap.add_argument("--q1-aw", default="outputs/session33/q1_aewake_seeds.json")
    ap.add_argument("--q2-aw", default="outputs/session33/q2_aewake_seeds.json")
    ap.add_argument("--q1-abl", default="outputs/session31/q1_ablation.json")
    ap.add_argument("--q2-abl", default="outputs/session31/q2_ablation.json")
    ap.add_argument("--q1-pool", default="outputs/session32/q1_pool.json")
    ap.add_argument("--q2-pool", default="outputs/session32/q2_pool.json")
    ap.add_argument("--q1-vec", default="outputs/session33/q1_vec.json")
    ap.add_argument("--q2-vec", default="outputs/session33/q2_vec.json")
    ap.add_argument("--out", default="outputs/session33/seed_band_v3.json")
    args = ap.parse_args(argv)

    ctx = {
        "s33": (_load(args.q1_seeds), _load(args.q2_seeds)),
        "anchor_abl": (_load(args.q1_abl), _load(args.q2_abl)),
        "anchor_pool": (_load(args.q1_pool), _load(args.q2_pool)),
    }
    try:
        ctx["s33aw"] = (_load(args.q1_aw), _load(args.q2_aw))
    except FileNotFoundError:
        FAMS.pop("ae_wake_pool", None)
        print("[seeds] ae_wake seed evals not found; band limited to the spine pair")
    try:
        ctx["vec"] = (_load(args.q1_vec), _load(args.q2_vec))
    except FileNotFoundError:
        FAMS.pop("jepa_wake_pool_vec", None)
        print("[seeds] vec seed evals not found; band excludes the D250 flagship")

    metrics = ("wake_linear_r2", "merit_mean_obs_h8", "cl_closure_mlp_h8", "floor_ssim")
    fams = {}
    for fam, seeds in FAMS.items():
        per_seed = {}
        for s, (model, src) in seeds.items():
            per_seed[str(s)] = {"model": model, **cells_for(model, src, ctx)}
        fams[fam] = {
            "per_seed": per_seed,
            "bands": {
                m: band([per_seed[str(s)].get(m) for s in (0, 1, 2)]) for m in metrics
            },
        }

    jw = fams["jepa_wake_pool"]["bands"]
    so = fams["supervised_only_pool"]["bands"]
    tol = 0.05

    def _delta(key):
        a, b = so[key]["mean"], jw[key]["mean"]
        return None if (a is None or b is None) else a - b

    d_wake = _delta("wake_linear_r2")
    d_merit = _delta("merit_mean_obs_h8")
    recheck = {
        "P1_tie_on_seed_means": {
            "wake_delta_so_minus_jw": d_wake,
            "merit_delta_so_minus_jw": d_merit,
            "tol": tol,
            "pass": bool(
                d_wake is not None and abs(d_wake) < tol
                and d_merit is not None and abs(d_merit) < tol
            ),
        },
        "P2_jw_ge_so_on_seed_means": {
            "delta_jw_minus_so_merit": None if d_merit is None else -d_merit,
            "pass": None if d_merit is None else bool(-d_merit >= -tol),
        },
        "seed_sd_summary": {
            fam: {m: fams[fam]["bands"][m]["sd"] for m in metrics} for fam in fams
        },
    }

    payload = {
        "task": "SESSION 33 -- 3-seed spine variance pass (item 8)",
        "params": {
            "seeds": "s0 = frozen S31/S32 runs; s1/s2 = S33 retrains",
            "families": list(FAMS.keys()),
            "tol": tol,
        },
        "families": fams,
        "gate_recheck": recheck,
    }
    out = _resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=float))
    for fam in fams:
        b = fams[fam]["bands"]
        print(
            f"[seeds] {fam:22s} wake={b['wake_linear_r2']['mean']:.3f}"
            f"+-{b['wake_linear_r2']['sd']:.3f} "
            f"merit={b['merit_mean_obs_h8']['mean']}"
            f"+-{b['merit_mean_obs_h8']['sd']} "
            f"ssim={b['floor_ssim']['mean']:.3f}+-{b['floor_ssim']['sd']:.3f}",
            flush=True,
        )
    print(f"[seeds] P1 tie on seed means: {recheck['P1_tie_on_seed_means']['pass']} "
          f"(wake d={d_wake}, merit d={d_merit})", flush=True)
    print(f"[seeds] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
