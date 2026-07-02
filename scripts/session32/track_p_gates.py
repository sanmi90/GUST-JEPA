"""Session 32 Track P gates P1/P2/P4 (+P3 hook).

Reuses the Session 31 metric definitions (make_ablation_table) and the case-clustered
bootstrap primitives (report_session31 / stats_lib). Reads:
  pooled kit  : outputs/session32/q1_pool.json + q2_pool.json
  jepa_wake_pool = existing jepa_pool : outputs/session31/q1_ablation.json + q2_ablation.json
  spatial canonical : outputs/session31/q1_representation.json + q2_temporal.json
  P1 GAP latents:
    jepa_wake_pool -> outputs/session31/q1_latents/latents_jepa_pool_{split}.npz
    supervised_only_pool -> outputs/session32/q1_pool_latents/latents_supervised_only_pool_*.npz

Nothing is retrained; point values match the Session 31 ablation-table math exactly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.evaluation.report_session31 import agg_r2, r2_contribs  # noqa: E402

S31 = REPO / "outputs" / "session31"
S32 = REPO / "outputs" / "session32"
TARGETS = ("C_L", "C_D", "wake_enstrophy", "circulation_pos", "circulation_neg")

# pooled model -> its spatial canonical counterpart (for P4)
POOL_TO_SPATIAL = {
    "jepa_wake_pool": "jepa_wake",
    "jepa_nowake_pool": "jepa_nowake",
    "ae_wake_pool": "ae_wake",
    "ae_nowake_pool": "ae_nowake",
    "supervised_only_pool": "supervised_only",
    "regAE_pool": "regAE",
}


def _load(p: Path) -> dict:
    return json.loads(Path(p).read_text())


def _at(curve: dict, key: str, h: int) -> float:
    hs = list(curve.get("horizons", []))
    return float(curve[key][hs.index(h)]) if h in hs else float("nan")


def q1_cells(q1: dict, model: str, window: str = "windowed") -> dict:
    m = q1["models"][model]
    floor = m["decode_floor"][window]
    return {"floor_vrmse": float(floor["vrmse"]), "floor_ssim": float(floor["ssim"])}


def q2_cells(q2: dict, model: str, h: int = 8) -> dict:
    pred = q2["models"][model]["predictors"]["resunet_matched"]
    fv = pred["field_vrmse"]
    oc = pred["observable_closure"]
    merit = float(sum(_at(oc[t], "model_mlp_r2", h) for t in TARGETS) / len(TARGETS))
    cl = _at(oc["C_L"], "model_mlp_r2", h)
    return {
        "field_vrmse_model_h8": _at(fv, "model", h),
        "merit_mean_obs_h8": merit,
        "cl_closure_mlp_h8": cl,
    }


# ------------------------------------------------------------------ P1 (paired probe R2 delta)
def paired_probe_delta(
    latents_a: dict[str, Path],  # {'train':..,'test_b':..} for model A (supervised_only_pool)
    latents_b: dict[str, Path],  # for model B (jepa_wake_pool)
    *,
    n_boot: int = 10000,
    seed: int = 0,
) -> dict:
    from src.evaluation.pressure_infer import load_gap_split
    from src.evaluation.represent import fit_linear_probe

    a_tr = load_gap_split(latents_a["train"])
    a_tb = load_gap_split(latents_a["test_b"])
    b_tr = load_gap_split(latents_b["train"])
    b_tb = load_gap_split(latents_b["test_b"])

    ma, mb = a_tb["window_mask"], b_tb["window_mask"]
    # alignment check on windowed test_b rows
    ca = np.asarray(a_tb["case_id"])[ma]
    cb = np.asarray(b_tb["case_id"])[mb]
    ea = np.asarray(a_tb["encounter_index"])[ma]
    eb = np.asarray(b_tb["encounter_index"])[mb]
    fa = np.asarray(a_tb["frame"])[ma]
    fb = np.asarray(b_tb["frame"])[mb]
    aligned = (
        (ca.shape == cb.shape)
        and np.array_equal(ca, cb)
        and np.array_equal(ea, eb)
        and np.array_equal(fa, fb)
    )
    case_w = ca

    za_tr, za_tb = a_tr["z_gap"], a_tb["z_gap"][ma]
    zb_tr, zb_tb = b_tr["z_gap"], b_tb["z_gap"][mb]

    # per-target contribs for both models (aligned rows)
    contribs_a: dict[str, dict] = {}
    contribs_b: dict[str, dict] = {}
    point = {}
    for t in TARGETS:
        ya_tr = a_tr["targets"][t]
        yb_tr = b_tr["targets"][t]
        ya_w = a_tb["targets"][t][ma]
        yb_w = b_tb["targets"][t][mb]
        pa = fit_linear_probe(za_tr, ya_tr).predict(za_tb)
        pb = fit_linear_probe(zb_tr, yb_tr).predict(zb_tb)
        ca_t = r2_contribs(ya_w, pa)
        cb_t = r2_contribs(yb_w, pb)
        contribs_a[t] = ca_t
        contribs_b[t] = cb_t
        point[t] = {
            "r2_a_supervised_only_pool": agg_r2(ca_t),
            "r2_b_jepa_wake_pool": agg_r2(cb_t),
            "delta_a_minus_b": agg_r2(ca_t) - agg_r2(cb_t),
        }

    # paired case-clustered bootstrap: resample cases ONCE, apply to both models
    uniq = np.array(sorted(set(case_w.tolist())))
    rows = {c: np.where(case_w == c)[0] for c in uniq}
    rng = np.random.default_rng(seed)
    nC = len(uniq)

    def mean_over5_r2(cd: dict[str, dict], idx: np.ndarray) -> float:
        vals = []
        for t in TARGETS:
            sst = cd[t]["ss_tot"][idx].sum()
            sse = cd[t]["ss_res"][idx].sum()
            vals.append(1.0 - sse / sst if sst > 0 else np.nan)
        return float(np.nanmean(vals))

    def target_r2(cd: dict[str, dict], t: str, idx: np.ndarray) -> float:
        sst = cd[t]["ss_tot"][idx].sum()
        sse = cd[t]["ss_res"][idx].sum()
        return float(1.0 - sse / sst) if sst > 0 else np.nan

    boot = {"wake_enstrophy": np.empty(n_boot), "mean_obs": np.empty(n_boot)}
    for bi in range(n_boot):
        pick = uniq[rng.integers(0, nC, size=nC)]
        idx = np.concatenate([rows[c] for c in pick])
        boot["wake_enstrophy"][bi] = target_r2(contribs_a, "wake_enstrophy", idx) - target_r2(
            contribs_b, "wake_enstrophy", idx
        )
        boot["mean_obs"][bi] = mean_over5_r2(contribs_a, idx) - mean_over5_r2(contribs_b, idx)

    mean_a = float(np.nanmean([point[t]["r2_a_supervised_only_pool"] for t in TARGETS]))
    mean_b = float(np.nanmean([point[t]["r2_b_jepa_wake_pool"] for t in TARGETS]))
    out = {
        "aligned_rows": bool(aligned),
        "n_windowed_rows": int(case_w.shape[0]),
        "n_cases": int(nC),
        "per_target": point,
        "primary_wake_enstrophy": {
            "delta_a_minus_b": point["wake_enstrophy"]["delta_a_minus_b"],
            "ci95": [
                float(np.percentile(boot["wake_enstrophy"], 2.5)),
                float(np.percentile(boot["wake_enstrophy"], 97.5)),
            ],
        },
        "secondary_mean_obs": {
            "r2_a_supervised_only_pool": mean_a,
            "r2_b_jepa_wake_pool": mean_b,
            "delta_a_minus_b": mean_a - mean_b,
            "ci95": [
                float(np.percentile(boot["mean_obs"], 2.5)),
                float(np.percentile(boot["mean_obs"], 97.5)),
            ],
        },
    }

    def verdict(d: dict) -> str:
        lo, hi = d["ci95"]
        includes0 = lo <= 0.0 <= hi
        small = abs(d["delta_a_minus_b"]) < 0.05
        return "PASS" if (includes0 or small) else "FAIL"

    out["primary_wake_enstrophy"]["verdict"] = verdict(out["primary_wake_enstrophy"])
    out["secondary_mean_obs"]["verdict"] = verdict(out["secondary_mean_obs"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--q1-pool", default=str(S32 / "q1_pool.json"))
    ap.add_argument("--q2-pool", default=str(S32 / "q2_pool.json"))
    ap.add_argument("--pool-cache", default=str(S32 / "q1_pool_latents"))
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-p1", action="store_true")
    ap.add_argument("--out", default=str(S32 / "track_p_gates.json"))
    args = ap.parse_args()

    q1p = _load(Path(args.q1_pool))
    q2p = _load(Path(args.q2_pool))
    q1_abl = _load(S31 / "q1_ablation.json")
    q2_abl = _load(S31 / "q2_ablation.json")
    q1_rep = _load(S31 / "q1_representation.json")
    q2_tmp = _load(S31 / "q2_temporal.json")

    pool_models = list(q1p["models"].keys())

    # ---- assemble per-model pooled cells (kit + jepa_wake_pool alias) ----
    cells: dict[str, dict] = {}
    for m in pool_models:
        cells[m] = {**q1_cells(q1p, m), **q2_cells(q2p, m)}
    # jepa_wake_pool == jepa_pool ablation
    cells["jepa_wake_pool"] = {
        **q1_cells(q1_abl, "jepa_pool"),
        **q2_cells(q2_abl, "jepa_pool"),
    }

    result: dict = {"pooled_cells": cells}

    # ---- P2: merit ordering {jepa_wake >= supervised_only > ae_wake > regAE}, tol 0.05 ----
    def merit(m: str) -> float:
        return cells[m]["merit_mean_obs_h8"]

    tol = 0.05
    order = ["jepa_wake_pool", "supervised_only_pool", "ae_wake_pool", "regAE_pool"]
    have = all(m in cells for m in order)
    p2 = {"merit": {m: (merit(m) if m in cells else None) for m in order}, "tol": tol}
    if have:
        jw, so, aw, ra = (merit(m) for m in order)
        c1 = jw >= so - tol  # jepa_wake >= supervised_only (tie tol)
        c2 = so > aw - tol  # supervised_only > ae_wake (strict, tie tol)
        c3 = aw > ra - tol  # ae_wake > regAE
        p2["checks"] = {
            "jepa_wake>=supervised_only": {"delta": jw - so, "ok": bool(c1)},
            "supervised_only>ae_wake": {"delta": so - aw, "ok": bool(c2)},
            "ae_wake>regAE": {"delta": aw - ra, "ok": bool(c3)},
        }
        p2["verdict"] = "PASS" if (c1 and c2 and c3) else "FAIL"
    else:
        p2["verdict"] = "INCOMPLETE (missing models)"
    result["P2_merit_ordering"] = p2

    # ---- P4: pooling-cost per family (pooled - spatial) ----
    p4 = {}
    for pm, sm in POOL_TO_SPATIAL.items():
        if pm not in cells or sm not in q1_rep["models"]:
            p4[pm] = {"note": "missing"}
            continue
        sp_q1 = q1_cells(q1_rep, sm)
        sp_q2 = q2_cells(q2_tmp, sm)
        p4[pm] = {
            "spatial_ref": sm,
            "decode_ssim": {
                "pooled": cells[pm]["floor_ssim"],
                "spatial": sp_q1["floor_ssim"],
                "delta": cells[pm]["floor_ssim"] - sp_q1["floor_ssim"],
            },
            "merit_h8": {
                "pooled": cells[pm]["merit_mean_obs_h8"],
                "spatial": sp_q2["merit_mean_obs_h8"],
                "delta": cells[pm]["merit_mean_obs_h8"] - sp_q2["merit_mean_obs_h8"],
            },
            "field_vrmse_h8": {
                "pooled": cells[pm]["field_vrmse_model_h8"],
                "spatial": sp_q2["field_vrmse_model_h8"],
                "delta": cells[pm]["field_vrmse_model_h8"] - sp_q2["field_vrmse_model_h8"],
            },
        }
    p4["bvae"] = {"note": "n/a (no spatial counterpart)"}
    result["P4_pooling_cost"] = p4

    # ---- P1: paired probe-R2 delta (involved) ----
    if not args.skip_p1 and "supervised_only_pool" in cells:
        latents_a = {
            "train": Path(args.pool_cache) / "latents_supervised_only_pool_train.npz",
            "test_b": Path(args.pool_cache) / "latents_supervised_only_pool_test_b.npz",
        }
        latents_b = {
            "train": S31 / "q1_latents" / "latents_jepa_pool_train.npz",
            "test_b": S31 / "q1_latents" / "latents_jepa_pool_test_b.npz",
        }
        if all(p.exists() for p in list(latents_a.values()) + list(latents_b.values())):
            result["P1_readability"] = paired_probe_delta(
                latents_a, latents_b, n_boot=args.n_boot, seed=args.seed
            )
        else:
            result["P1_readability"] = {"note": "missing cached GAP latents"}

    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
