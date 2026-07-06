"""T4 (Session 35 P1): Fukami d=16 seed band through the identical DA grid protocol.

For fukami_wake_d16_s1 and fukami_wake_d16_s2 (trained this session), runs the
byte-identical own-stack pipeline of scripts/session34/da_dims2.py:
  1. own OSP K-staircase (W=30 TCSI, seed 0) merged into a taps JSON,
  2. own latent-REX forecast operator (scripts.session34.latent_rex),
  3. own decode-floor decoder (identical capacity, 6000 steps),
  4. phase-resolved DA eval (scripts.session34.da_phase_eval; test_b, K=8,
     every-frame pressure, no noise),
then assembles the 3-seed band (s0 = Session 34 da_phase_dim_fukami_wake_d16.json)
with the assemble_da_grid best-recipe convention (min impact-phase C_L RMSE over
rex_enkf / linear_lae / eobs) and applies the pre-registered FK16-A/B gate
(outputs/session35/p1_gates.md T4).

Run (RTX 6000, after trackc_encode of the two new runs):
    taskset -c 0-7 python -m scripts.session35.da_fk16_seeds --gpu 0
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

CACHE = REPO_ROOT / "outputs/session34/trackc_latents"
S35 = REPO_ROOT / "outputs/session35"
MODELS = ("fukami_wake_d16_s1", "fukami_wake_d16_s2")
RECIPES = ("rex_enkf", "linear_lae", "eobs")
S0_JSON = REPO_ROOT / "outputs/session34/da_phase_dim_fukami_wake_d16.json"


def best_recipe(summary: dict) -> tuple[str, dict]:
    key = min(RECIPES, key=lambda r: summary[r]["impact"]["cl_rmse"])
    return key, summary[key]


def cell_row(summary: dict, label: str) -> dict:
    rec_name, rec = best_recipe(summary)
    return {
        "model": label,
        "best_recipe": rec_name,
        "impact_cl_rmse": rec["impact"]["cl_rmse"],
        "relax_cl_rmse": rec["relax"]["cl_rmse"],
        "impact_cl_r2": rec["impact"]["cl_r2"],
        "peak_rel_error_pct": rec["peak_rel_error_pct_median"],
        "impact_ssim_nearbody": rec["impact"]["ssim_nearbody"],
        "impact_ssim_full": rec["impact"]["ssim_full"],
        "per_recipe": {
            r: {
                "impact_cl_rmse": summary[r]["impact"]["cl_rmse"],
                "peak_rel_error_pct": summary[r]["peak_rel_error_pct_median"],
            }
            for r in RECIPES
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    import torch

    # ---- 1. OSP staircases (da_dims2 step 1, byte-identical settings) ----------
    import scripts.session32.track_o1_recovery as o1
    from scripts.session32.osp_select import build_osp_taps
    from src.evaluation.rom_eval import load_windows

    taps_path = S35 / "osp_taps_fk16.json"
    osp: dict = {}
    for src in ("outputs/session34/osp_taps_trackc.json",
                "outputs/session33/osp_taps_vec.json",
                "outputs/session34/osp_taps_dims2.json"):
        p = REPO_ROOT / src
        if p.exists():
            osp.update(json.loads(p.read_text()))
    if taps_path.exists():
        osp.update(json.loads(taps_path.read_text()))
    windows = load_windows(REPO_ROOT / "outputs/session31/windows_v2p2.json")
    qdeim = json.loads((REPO_ROOT / "outputs/session32/qdeim_taps_v2p2.json").read_text())
    p_train = o1.load_pressure(CACHE, "train")["p_wall"]
    for m in MODELS:
        if m in osp:
            print(f"[fk16] taps exist: {m}", flush=True)
            continue
        print(f"[fk16] staircase: {m}", flush=True)
        caches = {m: o1.load_cache(CACHE, m, "train")}
        osp[m] = build_osp_taps(caches, windows, p_train, w=30, qdeim_taps=qdeim, seed=0)[m]
        taps_path.write_text(json.dumps(osp, indent=2))
    taps_path.write_text(json.dumps(osp, indent=2))

    # ---- 2. REX operators (da_dims2 step 2) ------------------------------------
    for m in MODELS:
        if (REPO_ROOT / f"outputs/session34/latent_rex_model_{m}.pt").exists():
            continue
        print(f"[fk16] rex: {m}", flush=True)
        subprocess.run([sys.executable, "-m", "scripts.session34.latent_rex",
                        "--gpu", str(args.gpu), "--run", m,
                        "--out", f"outputs/session35/latent_rex_{m}.json"],
                       cwd=REPO_ROOT, check=True, capture_output=True)

    # ---- 3. decoders (da_dims2 step 3) ------------------------------------------
    from src.evaluation.represent import fit_decode_floor_decoder
    from src.utils.device import require_rtx6000

    device = require_rtx6000(gpu_index=args.gpu)
    ftr = np.load(CACHE / "fields_train.npz")["omega_norm"].astype(np.float32)
    tile = lambda z: np.repeat(  # noqa: E731
        np.repeat(z[:, :, None, None], 24, 2), 12, 3).astype(np.float32)
    for m in MODELS:
        dpath = REPO_ROOT / f"outputs/session34/trackc_decoders/decoder_{m}.pt"
        if dpath.exists():
            continue
        print(f"[fk16] decoder: {m}", flush=True)
        ztr = np.load(CACHE / f"latents_{m}_train.npz")["z_gap"]
        dec = fit_decode_floor_decoder(tile(ztr), ftr, (24, 12), device=device,
                                       steps=6000, verbose=False)
        torch.save(dec.state_dict(), dpath)
        del dec
        torch.cuda.empty_cache()

    # ---- 4. phase-resolved DA eval (da_dims2 step 4) -----------------------------
    for m in MODELS:
        out = S35 / f"da_phase_dim_{m}.json"
        if out.exists():
            continue
        print(f"[fk16] da eval: {m}", flush=True)
        subprocess.run([sys.executable, "-m", "scripts.session34.da_phase_eval",
                        "--gpu", str(args.gpu), "--model", m,
                        "--cache-dir", "outputs/session34/trackc_latents",
                        "--pressure-dir", "outputs/session34/trackc_latents",
                        "--taps", "outputs/session35/osp_taps_fk16.json",
                        "--rex-ckpt", f"outputs/session34/latent_rex_model_{m}.pt",
                        "--out", str(out)], cwd=REPO_ROOT, check=True,
                       capture_output=True)

    # ---- 5. 3-seed band + pre-registered gate -------------------------------------
    rows = [cell_row(json.loads(S0_JSON.read_text())["summary"], "fukami_wake_d16_s0")]
    for m in MODELS:
        rows.append(cell_row(json.loads((S35 / f"da_phase_dim_{m}.json").read_text())
                             ["summary"], m))
    imp = np.array([r["impact_cl_rmse"] for r in rows])
    pk = np.array([r["peak_rel_error_pct"] for r in rows])

    fk16_a = bool((imp <= 0.36).all() and (pk <= 15.0).all())
    fk16_b = bool((imp > 0.60).any() or (pk > 25.0).any()
                  or (imp.max() / max(imp.min(), 1e-9) > 2.0))
    verdict = "FK16-A" if fk16_a else ("FK16-B" if fk16_b else "INTERMEDIATE")

    band = {
        "protocol": "da_dims2 own-stack grid protocol, test_b, K=8, every-frame, "
                    "no noise; best recipe = min impact C_L RMSE over "
                    + "/".join(RECIPES),
        "rows": rows,
        "impact_cl_rmse_band": {
            "values": imp.tolist(),
            "seed_mean": float(imp.mean()),
            "seed_sd": float(imp.std(ddof=1)),
            "n": len(rows),
        },
        "peak_rel_error_pct_band": {
            "values": pk.tolist(),
            "seed_mean": float(pk.mean()),
            "seed_sd": float(pk.std(ddof=1)),
            "n": len(rows),
        },
        "gate": {
            "fk16_a_all_le_0p36_and_peak_le_15": fk16_a,
            "fk16_b_any_gt_0p60_or_peak_gt_25_or_bimodal": fk16_b,
            "verdict": verdict,
        },
    }
    dest = S35 / "fk16_seed_band.json"
    dest.write_text(json.dumps(band, indent=1))
    print(f"[fk16] impact RMSE per seed: {np.round(imp, 3).tolist()}  "
          f"peak err %: {np.round(pk, 1).tolist()}", flush=True)
    print(f"[fk16] PRE-REGISTERED VERDICT: {verdict}", flush=True)
    print(f"[fk16] wrote {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
