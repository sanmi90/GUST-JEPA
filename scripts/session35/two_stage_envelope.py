"""T5 (Session 35 P1): two-stage filter + test_a NIS band tuning (F20 decider).

Implements the SESSION_35.md T5 engineering gate on the flagship jepa_pool_vec
stack, in the rex_filter full-trajectory frame (so the 0.749 protocol-clean
anchor is directly comparable):

STAGE data   : encode test_a (val) and test_c latents with the frozen
               session33 jepa_pool_vec encoder + build aligned pressure caches
               (outputs/session35/t5_latents). train/test_b reuse the byte-
               identical session33/session31 caches rex_filter consumed.
STAGE tune   : REX-EnKF (tuned RexBackbone, E_obs H=I, global Gamma) on
               test_a ONLY, over the PRE-REGISTERED band grid
               {1.0, 1.4, 1.77, 2.5, 3.5, 4.5, 6.0} (p1_gates.md T5, committed
               before any test_a NIS value was seen). Selection: c* minimizes
               |pooled impact-phase mean NIS - 1| where
               NIS_t = nu_t^T S_t^{-1} nu_t / d on the PRIOR ensemble.
STAGE frozen : ONE run over test_b and test_c at c*, no second look:
               (a) REX-EnKF(c*)  [ladder headline candidate vs 0.749]
               (b) TWO-STAGE(c*) [D259(2) integration: REX + E_obs update
                   inside the impact window; matched AR-transformer forecast
                   (envelope predictor, 4000 steps, seed 0) + tap-space
                   observation (C_map / R_taps, rex_filter phase_switch
                   convention) outside it]
               plus, when c* != 1.77, REX-EnKF(1.77) on test_b as the
               reproduction anchor of the Session 34 protocol-clean 0.749.

All estimator fits (E_obs W, Gamma, C_map, R_taps, probe, transformer, Q_tf)
use TRAIN only. The pre-registered F20 gate (p1_gates.md T5) is applied
mechanically: F20-A iff frozen test_b REX-EnKF(c*) impact median R2 >=
0.749 + 0.03 with <= 2 catastrophic encounters; else F20-B.

Run (RTX 6000):
    taskset -c 0-15 python -m scripts.session35.two_stage_envelope --gpu 0 --stage all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from scripts.session34.lae_enkf_pilot import (  # noqa: E402
    delay_embed,
    encounters,
    load_aligned,
    r2,
)

BAND_TO_SIGMA = 2.5631  # q90 - q10 of a Gaussian = 2 * 1.28155 sigma
BAND_GRID = (1.0, 1.4, 1.77, 2.5, 3.5, 4.5, 6.0)  # pre-registered, p1_gates.md T5
MODEL = "jepa_pool_vec"
RUN_DIR = "outputs/runs/session33/jepa_pool_vec"
CHECKPOINT = "checkpoint_iter010000.pt"
TAPS_JSON = "outputs/session33/osp_taps_vec.json"
TRAIN_CACHE = "outputs/session33/q1_vec_latents"
TRAIN_PRESS = "outputs/session31/q1_latents"
T5_CACHE = "outputs/session35/t5_latents"
K, DELAY, N_MEMBERS, MAX_CTX, ALPHA = 8, 10, 64, 30, 1.0
ANCHOR_BAND, ANCHOR_VALUE = 1.77, 0.749
GATE_DELTA, GATE_CAT = 0.03, 2


# ---------------------------------------------------------------- data stage
def build_t5_caches(gpu: int, limit: int | None) -> None:
    """Encode test_a + test_c with the frozen flagship; write aligned pressure."""
    from src.evaluation import rom_eval as re
    from src.evaluation.pressure_infer import load_pressure_for_alignment
    from src.utils.device import require_rtx6000

    device = require_rtx6000(gpu_index=gpu)
    cache_dir = REPO_ROOT / T5_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)
    windows = re.load_windows(REPO_ROOT / "outputs/session31/windows_v2p2.json")
    frozen = None
    for split in ("test_a", "test_c"):
        lat_path = cache_dir / f"latents_{MODEL}_{split}.npz"
        if not lat_path.exists():
            if frozen is None:
                frozen = re.load_frozen_model(REPO_ROOT / RUN_DIR, CHECKPOINT, device)
            enc = re.encode_split(frozen, split, partition="v2p2",
                                  windows=windows, device=device, limit=limit)
            re.save_latents(lat_path, enc)
            print(f"[t5-data] wrote {lat_path}", flush=True)
        p_path = cache_dir / f"pressure_{split}.npz"
        if not p_path.exists():
            z = np.load(lat_path, allow_pickle=True)
            cid = z["case_id"].astype(str)
            p = load_pressure_for_alignment(cid, z["encounter_index"], z["frame"],
                                            partition="v2p2")
            np.savez(p_path, p_wall=p.astype(np.float32), case_id=cid,
                     encounter_index=z["encounter_index"], frame=z["frame"])
            print(f"[t5-data] wrote {p_path}", flush=True)
    if frozen is not None:
        del frozen
        torch.cuda.empty_cache()


def load_split(split: str) -> dict:
    if split in ("train", "test_b"):
        return load_aligned(REPO_ROOT / TRAIN_CACHE, REPO_ROOT / TRAIN_PRESS,
                            MODEL, split)
    return load_aligned(REPO_ROOT / T5_CACHE, REPO_ROOT / T5_CACHE, MODEL, split)


# ---------------------------------------------------------------- context
def build_context(device) -> dict:
    """Train-only fits: E_obs, Gamma, C_map/R_taps, probe, REX, transformer, Q."""
    from scripts.session32.track_b_pilot import group_gap_trajs, q_from_teacher_forcing
    from scripts.session34.rex_tune import RexBackbone
    from src.estimation import enkf as ek
    from src.evaluation.represent import fit_linear_probe
    from src.evaluation.rollout import fit_matched_transformer

    tr = load_split("train")
    encs_tr = encounters(tr)
    n = tr["z"].shape[1]
    taps = np.asarray(json.loads((REPO_ROOT / TAPS_JSON).read_text())
                      [MODEL][f"K{K}"], dtype=int)
    p_mu = tr["p"][:, taps].mean(axis=0)
    p_sd = tr["p"][:, taps].std(axis=0) + 1e-9

    X_tr, Zt_tr = [], []
    for e in encs_tr:
        pt = (tr["p"][e["rows"]][:, taps] - p_mu) / p_sd
        X_tr.append(delay_embed(pt, DELAY))
        Zt_tr.append(tr["z"][e["rows"]])
    X_tr = np.concatenate(X_tr)
    Zt_tr = np.concatenate(Zt_tr)
    W = np.linalg.solve(X_tr.T @ X_tr + ALPHA * np.eye(X_tr.shape[1]), X_tr.T @ Zt_tr)
    resid = Zt_tr - X_tr @ W
    Gam = np.cov(resid.T) + 1e-8 * np.eye(n)
    chol_g = np.linalg.cholesky(Gam)
    Y_tr = np.concatenate([(tr["p"][e["rows"]][:, taps] - p_mu) / p_sd
                           for e in encs_tr])
    C_map = np.linalg.solve(Zt_tr.T @ Zt_tr + ALPHA * np.eye(n), Zt_tr.T @ Y_tr)
    R_taps = np.cov((Y_tr - Zt_tr @ C_map).T) + 1e-8 * np.eye(len(taps))
    chol_R = np.linalg.cholesky(R_taps)

    probe = fit_linear_probe(tr["z"], tr["cl"])

    meta = json.loads((REPO_ROOT / "outputs/session34/rex_tune.json").read_text())
    wbest = meta["winner"]
    rex = RexBackbone(wbest["kind"], n, wbest["hidden"], 40, wbest["nq"])
    rex.load_state_dict(torch.load(
        REPO_ROOT / "outputs/session34/latent_rex_model_tuned.pt", map_location="cpu"))
    rex.to(device).eval()

    # matched AR-transformer (envelope predictor: 4000 steps, seed 0) + Q.
    gap_trajs = group_gap_trajs(tr["z"].astype(np.float32), tr["case_id"],
                                tr["enc"], tr["frame"])
    fake_spatial = [t[:, :, None, None].astype(np.float32) for t in gap_trajs]
    predictor = fit_matched_transformer(fake_spatial, device=device, latent_dim=n,
                                        steps=4000, seed=0, verbose=False)
    fmodel = ek.TransformerForecast(predictor, device)
    Q_tf = q_from_teacher_forcing(predictor, gap_trajs, device, seed=0)
    chol_q = np.linalg.cholesky(Q_tf + 1e-10 * np.eye(n))

    split_cases = json.loads(
        (REPO_ROOT / "configs/splits/split_v2p2.json").read_text())["cases"]

    return {"n": n, "taps": taps, "p_mu": p_mu, "p_sd": p_sd, "W": W,
            "Gam": Gam, "chol_g": chol_g, "C_map": C_map, "R_taps": R_taps,
            "chol_R": chol_R, "probe": probe, "rex": rex, "fmodel": fmodel,
            "chol_q": chol_q, "device": device, "cases": split_cases}


# ---------------------------------------------------------------- filter loops
@torch.no_grad()
def rex_step(ctx_obj: dict, ctx_np: np.ndarray, band: float) -> tuple[np.ndarray, np.ndarray]:
    out = ctx_obj["rex"](torch.from_numpy(ctx_np[:, -MAX_CTX:]).float()
                         .to(ctx_obj["device"]))
    step0 = out[:, 0].cpu().numpy()
    med = step0[..., step0.shape[-1] // 2]
    sig = np.clip((step0[..., -1] - step0[..., 0]) / BAND_TO_SIGMA * band, 1e-4, None)
    return med, sig


def run_encounter(ctx: dict, data: dict, e: dict, band: float, arm: str,
                  rng: np.random.Generator) -> dict:
    """One encounter through REX-EnKF ('rex') or the two-stage filter ('two_stage').

    rex_filter conventions: full trajectory, t_init = DELAY - 1, member context
    init from z_obs with a global-Gamma perturbation on the last frame, N = 64,
    perturbed-observation updates, impact window = the cache window_mask.
    ``rng`` is threaded across encounters by the caller in sorted encounter
    order, matching rex_filter's single-generator draw sequence so the 'rex'
    arm at band 1.77 on test_b reproduces the Session 34 0.749 run bit-exactly.
    NIS uses the PRIOR ensemble: nu = obs - mean(zf), S = P_f + R (no rng draw).
    """
    n = ctx["n"]
    rows = e["rows"]
    z_true, cl_true = data["z"][rows], data["cl"][rows]
    wmask = data["wmask"][rows]
    pt = (data["p"][rows][:, ctx["taps"]] - ctx["p_mu"]) / ctx["p_sd"]
    z_obs = delay_embed(pt, DELAY) @ ctx["W"]
    T = rows.size
    t_init = DELAY - 1
    mctx = np.repeat(z_obs[None, : t_init + 1], N_MEMBERS, axis=0)
    mctx[:, -1] += rng.standard_normal((N_MEMBERS, n)) @ ctx["chol_g"].T
    zA = np.empty((T, n))
    zA[: t_init + 1] = z_obs[: t_init + 1]
    nis_frames = np.full(T, np.nan)
    for t in range(t_init + 1, T):
        use_rex = (arm == "rex") or bool(wmask[t])
        if use_rex:
            med, sig = rex_step(ctx, mctx, band)
            zf = med + rng.standard_normal((N_MEMBERS, n)) * sig
        else:
            prior = np.asarray(ctx["fmodel"].step(mctx.astype(np.float32)),
                               dtype=np.float64)
            zf = prior + rng.standard_normal((N_MEMBERS, n)) @ ctx["chol_q"].T
        if (arm == "rex") or bool(wmask[t]):
            # E_obs latent observation, H = I, R = global Gamma.
            dZ = zf - zf.mean(0, keepdims=True)
            P = dZ.T @ dZ / (N_MEMBERS - 1)
            S = P + ctx["Gam"]
            nu = z_obs[t] - zf.mean(0)
            nis_frames[t] = float(nu @ np.linalg.solve(S, nu)) / n
            K_g = P @ np.linalg.inv(S)
            innov = (z_obs[t][None]
                     + rng.standard_normal((N_MEMBERS, n)) @ ctx["chol_g"].T - zf)
            za = zf + innov @ K_g.T
        else:
            # tap-space observation through C_map (rex_filter phase_switch branch).
            yf = zf @ ctx["C_map"]
            dZ = zf - zf.mean(0, keepdims=True)
            dY = yf - yf.mean(0, keepdims=True)
            S_y = dY.T @ dY / (N_MEMBERS - 1) + ctx["R_taps"]
            nu = pt[t] - yf.mean(0)
            nis_frames[t] = float(nu @ np.linalg.solve(S_y, nu)) / len(ctx["taps"])
            K_g = (dZ.T @ dY / (N_MEMBERS - 1)) @ np.linalg.inv(S_y)
            innov = (pt[t][None]
                     + rng.standard_normal((N_MEMBERS, len(ctx["taps"]))) @ ctx["chol_R"].T
                     - yf)
            za = zf + innov @ K_g.T
        zA[t] = za.mean(0)
        mctx = np.concatenate([mctx, za[:, None]], axis=1)[:, -MAX_CTX:]
    cl_hat = ctx["probe"].predict(zA)
    imp = wmask
    rel = (~wmask) & (np.arange(T) > (np.nonzero(wmask)[0].max() if wmask.any() else 55))
    cid = e["case_id"]
    case = ctx["cases"].get(cid, {})
    return {
        "case_id": cid, "encounter_index": e["k"],
        "G": float(case.get("G", np.nan)), "D": float(case.get("D", np.nan)),
        "Y": float(case.get("Y", np.nan)),
        "CL_analysis_r2_impact": r2(cl_true[imp], cl_hat[imp]) if imp.any() else None,
        "CL_analysis_r2_relax": r2(cl_true[rel], cl_hat[rel]) if rel.any() else None,
        "CL_analysis_rmse_impact": float(np.sqrt(np.mean(
            (cl_true[imp] - cl_hat[imp]) ** 2))) if imp.any() else None,
        "CL_analysis_rmse_relax": float(np.sqrt(np.mean(
            (cl_true[rel] - cl_hat[rel]) ** 2))) if rel.any() else None,
        "latent_track_r2": r2(z_true[t_init + 1:], zA[t_init + 1:]),
        "mean_nis_impact": float(np.nanmean(nis_frames[imp])) if imp.any() else None,
        "mean_nis_relax": float(np.nanmean(nis_frames[rel])) if rel.any() else None,
        "n_impact_frames": int(imp.sum()),
    }


def run_split(ctx: dict, split: str, band: float, arm: str, seed: int) -> dict:
    data = load_split(split)
    encs = encounters(data)
    rng = np.random.default_rng(seed)
    records = [run_encounter(ctx, data, e, band, arm, rng) for e in encs]
    imp = np.array([rec["CL_analysis_r2_impact"] for rec in records], float)
    rel = np.array([rec["CL_analysis_r2_relax"] for rec in records], float)
    nis_w = np.array([rec["mean_nis_impact"] for rec in records], float)
    n_w = np.array([rec["n_impact_frames"] for rec in records], float)
    pooled_nis = float(np.nansum(nis_w * n_w) / np.nansum(n_w))
    agg = {
        "median_CL_r2_impact": float(np.nanmedian(imp)),
        "median_CL_r2_relax": float(np.nanmedian(rel)),
        "median_CL_rmse_impact": float(np.nanmedian(np.array(
            [rec["CL_analysis_rmse_impact"] for rec in records], float))),
        "median_CL_rmse_relax": float(np.nanmedian(np.array(
            [rec["CL_analysis_rmse_relax"] for rec in records], float))),
        "catastrophic_impact_lt_-1": int((imp < -1).sum()),
        "catastrophic_relax_lt_-1": int((rel < -1).sum()),
        "pooled_mean_nis_impact": pooled_nis,
        "median_mean_nis_impact": float(np.nanmedian(nis_w)),
        "mean_nis_relax": float(np.nanmean(np.array(
            [rec["mean_nis_relax"] for rec in records], float))),
        "n_encounters": len(records),
    }
    # |G| buckets (envelope convention: sign-agnostic).
    by_g: dict = {}
    for rec in records:
        key = f"|G|={abs(rec['G']):g}"
        by_g.setdefault(key, []).append(rec["CL_analysis_r2_impact"])
    agg["impact_r2_by_absG"] = {
        k: {"median": float(np.nanmedian(np.array(v, float))), "n": len(v)}
        for k, v in sorted(by_g.items())
    }
    return {"aggregates": agg, "records": records}


# ---------------------------------------------------------------- stages
def stage_tune(ctx: dict, bands, seed: int) -> dict:
    out = {"split": "test_a", "protocol": "REX-EnKF eobs global Gamma; "
           "selection = argmin |pooled impact-phase mean NIS - 1| (pre-registered)",
           "bands": {}}
    for b in bands:
        t0 = time.time()
        res = run_split(ctx, "test_a", b, "rex", seed)
        a = res["aggregates"]
        out["bands"][f"{b:g}"] = a
        print(f"[t5-tune] band={b:g}: pooled impact NIS={a['pooled_mean_nis_impact']:.3f} "
              f"(median enc NIS {a['median_mean_nis_impact']:.3f}) "
              f"val impact R2={a['median_CL_r2_impact']:+.3f} "
              f"cat={a['catastrophic_impact_lt_-1']} ({time.time()-t0:.0f}s)",
              flush=True)
    sel = min(out["bands"].items(),
              key=lambda kv: abs(kv[1]["pooled_mean_nis_impact"] - 1.0))
    out["c_star"] = float(sel[0])
    out["c_star_nis"] = sel[1]["pooled_mean_nis_impact"]
    print(f"[t5-tune] SELECTED c* = {out['c_star']:g} "
          f"(pooled impact NIS {out['c_star_nis']:.3f})", flush=True)
    dest = REPO_ROOT / "outputs/session35/nis_band_tuning.json"
    dest.write_text(json.dumps(out, indent=1, default=float))
    print(f"[t5-tune] wrote {dest}", flush=True)
    return out


def stage_frozen(ctx: dict, c_star: float, seed: int) -> dict:
    payload = {
        "c_star": c_star,
        "freeze_rule": "band selected on test_a only (nis_band_tuning.json); "
                       "single frozen run below; no second look",
        "arms": {},
    }
    for split in ("test_b", "test_c"):
        for arm in ("rex", "two_stage"):
            t0 = time.time()
            res = run_split(ctx, split, c_star, arm, seed)
            payload["arms"][f"{arm}_{split}"] = res
            a = res["aggregates"]
            print(f"[t5-frozen] {arm} {split} @ c*={c_star:g}: "
                  f"impact={a['median_CL_r2_impact']:+.3f} "
                  f"relax={a['median_CL_r2_relax']:+.3f} "
                  f"rmse_imp={a['median_CL_rmse_impact']:.3f} "
                  f"cat={a['catastrophic_impact_lt_-1']}/{a['catastrophic_relax_lt_-1']} "
                  f"NIS={a['pooled_mean_nis_impact']:.2f} ({time.time()-t0:.0f}s)",
                  flush=True)
    if abs(c_star - ANCHOR_BAND) > 1e-9:
        res = run_split(ctx, "test_b", ANCHOR_BAND, "rex", seed)
        payload["arms"]["rex_test_b_anchor_1p77"] = res
        a = res["aggregates"]
        print(f"[t5-frozen] anchor rex test_b @ 1.77: "
              f"impact={a['median_CL_r2_impact']:+.3f} "
              f"(Session 34: {ANCHOR_VALUE})", flush=True)
        anchor_val = a["median_CL_r2_impact"]
    else:
        anchor_val = payload["arms"]["rex_test_b"]["aggregates"]["median_CL_r2_impact"]
    gate_arm = payload["arms"]["rex_test_b"]["aggregates"]
    payload["anchor_check"] = {
        "rex_test_b_at_1p77": anchor_val,
        "session34_value": ANCHOR_VALUE,
        "reproduces": bool(abs(anchor_val - ANCHOR_VALUE) < 0.02),
    }
    payload["f20_gate"] = {
        "rule": f"F20-A iff rex test_b impact median >= {ANCHOR_VALUE + GATE_DELTA} "
                f"and catastrophic_impact <= {GATE_CAT}",
        "impact_median": gate_arm["median_CL_r2_impact"],
        "catastrophic_impact": gate_arm["catastrophic_impact_lt_-1"],
        "verdict": "F20-A" if (
            gate_arm["median_CL_r2_impact"] >= ANCHOR_VALUE + GATE_DELTA
            and gate_arm["catastrophic_impact_lt_-1"] <= GATE_CAT) else "F20-B",
    }
    print(f"[t5-frozen] F20 GATE: {payload['f20_gate']['verdict']} "
          f"(impact {payload['f20_gate']['impact_median']:+.3f}, "
          f"cat {payload['f20_gate']['catastrophic_impact']})", flush=True)
    dest = REPO_ROOT / "outputs/session35/two_stage_envelope.json"
    dest.write_text(json.dumps(payload, indent=1, default=float))
    print(f"[t5-frozen] wrote {dest}", flush=True)
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--stage", choices=["data", "tune", "frozen", "all"], default="all")
    ap.add_argument("--limit", type=int, default=None,
                    help="encounter cap for the data stage (smoke only)")
    ap.add_argument("--bands", type=float, nargs="+", default=list(BAND_GRID))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from src.utils.device import require_rtx6000

    if args.stage in ("data", "all"):
        build_t5_caches(args.gpu, args.limit)
    if args.stage == "data":
        return 0

    device = require_rtx6000(gpu_index=args.gpu)
    torch.manual_seed(args.seed)
    t0 = time.time()
    ctx = build_context(device)
    print(f"[t5] context built in {time.time()-t0:.0f}s", flush=True)

    if args.stage in ("tune", "all"):
        tune = stage_tune(ctx, args.bands, args.seed)
        c_star = tune["c_star"]
    else:
        c_star = json.loads((REPO_ROOT / "outputs/session35/nis_band_tuning.json")
                            .read_text())["c_star"]
    if args.stage in ("frozen", "all"):
        stage_frozen(ctx, c_star, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
