#!/usr/bin/env python3
"""T5 (paper_redesign.md section 6): phase-split shared-operator forecast merit.

Re-runs the Track M1 shared direct forecaster (scripts/session36/rex_families_m1)
UNCHANGED in operator/probe/merit definition, but tags every forecast sample by
the phase its horizon lands in, relative to the impact frame (~40):
  pre     : target frame < impact            (forecasting the periodic shedding)
  through : anchor < impact <= target        (horizon crosses the impact transient)
  post    : anchor frame >= impact           (relaxation)
Reports merit at h = 8 and h = 16 per family per phase (seed mean + case-clustered
CI). This is the pre-impact / through-impact split the redesign wants in s4.4,
and it makes the error-accumulation thesis phase-explicit.

GPU (RTX-6000), no retraining of encoders; trains only the shared LSTM operator
(identical recipe to M1). Run:
    OMP_NUM_THREADS=8 taskset -c 0-15 .venv/bin/python \\
        scripts/session39/t5_phase_split.py --gpu 1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO))

from scripts.session36.rex_families_m1 import (  # noqa: E402
    FAMILIES, load_split, group, train_operator, merit_from_predictions,
    CTX, MERIT_HORIZONS, OBSERVABLES, MEDIAN_IDX,
)

IMPACT = 40                     # impact frame (QC mean ~40, CLAUDE.md)
PHASES = ("pre", "through", "post")
# headline families for the phase split (predictive, matched AE, supervised-only,
# published-recipe, linear); keeps the GPU budget bounded.
HEADLINE = ("JepaWake", "AeWake", "SupOnly", "Fukami", "Pod")


def eval_phase(model, tb, encs, device):
    """Like rex_families_m1.eval_samples but records the target frame so samples
    can be split by phase. Returns per horizon: z, row, case, ftgt, fanchor."""
    out = {h: {"z": [], "row": [], "case": [], "ftgt": [], "fanc": []}
           for h in MERIT_HORIZONS}
    for e in encs:
        rows = e["rows"]
        Z = tb["z_gap"][rows]
        wm = tb["window_mask"][rows]
        frames = tb["frame"][rows]
        T = Z.shape[0]
        anchors = np.arange(CTX - 1, T - 1)
        ctx = np.stack([Z[a - CTX + 1: a + 1] for a in anchors])
        with torch.no_grad():
            pred = model(torch.from_numpy(ctx).float().to(device)).cpu().numpy()
        roll = pred[..., MEDIAN_IDX]
        for h in MERIT_HORIZONS:
            tgt = anchors + h
            ok = (tgt <= T - 1) & wm[np.clip(tgt, 0, T - 1)]
            if not ok.any():
                continue
            out[h]["z"].append(roll[ok, h - 1])
            out[h]["row"].append(rows[tgt[ok]])
            out[h]["case"].append(np.array([e["case_id"]] * int(ok.sum())))
            out[h]["ftgt"].append(frames[tgt[ok]])
            out[h]["fanc"].append(frames[anchors[ok]])
    for h in MERIT_HORIZONS:
        out[h] = {k: np.concatenate(v) if v else np.array([])
                  for k, v in out[h].items()}
    return out


def phase_of(ftgt, fanc):
    ph = np.full(ftgt.shape, "post", dtype=object)
    ph[ftgt < IMPACT] = "pre"
    ph[(fanc < IMPACT) & (ftgt >= IMPACT)] = "through"
    return ph


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--families", nargs="+", default=list(HEADLINE))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="outputs/session39/t5_phase_split.json")
    args = ap.parse_args()

    from src.evaluation.rollout import fit_observable_probes
    from src.utils.device import require_rtx6000

    device = require_rtx6000(gpu_index=args.gpu)
    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {"_provenance": {"script": "scripts/session39/t5_phase_split.py",
                               "impact_frame": IMPACT, "phases": list(PHASES),
                               "operator": "identical to rex_families_m1 (M1)",
                               "gpu_name": torch.cuda.get_device_name(device.index),
                               "iters": args.iters, "seeds": args.seeds},
               "families": {}}
    t0 = time.time()
    for fam in args.families:
        run = FAMILIES[fam]
        tr, tb = load_split(run, "train"), load_split(run, "test_b")
        encs_tr, encs_tb = group(tr), group(tb)
        Zt = torch.from_numpy(
            np.stack([tr["z_gap"][e["rows"]] for e in encs_tr])).float().to(device)
        probes = fit_observable_probes(
            tr["z_gap"], {o: tr[f"target_{o}"] for o in OBSERVABLES})
        # per (h, phase): stash true + per-seed preds + cases
        acc = {h: {p: {"true": None, "preds": [], "case": None}
                   for p in PHASES} for h in MERIT_HORIZONS}
        for seed in args.seeds:
            model = train_operator(Zt, seed, args.iters, device)
            s = eval_phase(model, tb, encs_tb, device)
            for h in MERIT_HORIZONS:
                sh = s[h]
                if sh["z"].size == 0:
                    continue
                ph = phase_of(sh["ftgt"], sh["fanc"])
                yhat = {o: probes[o]["mlp"].predict(sh["z"]) for o in OBSERVABLES}
                ytru = {o: tb[f"target_{o}"][sh["row"]] for o in OBSERVABLES}
                for p in PHASES:
                    m = ph == p
                    if not m.any():
                        continue
                    a = acc[h][p]
                    if a["true"] is None:
                        a["true"] = {o: ytru[o][m] for o in OBSERVABLES}
                        a["case"] = sh["case"][m]
                    a["preds"].append({o: yhat[o][m] for o in OBSERVABLES})
            del model
            torch.cuda.empty_cache()
            print(f"[t5] {fam} seed {seed} done ({time.time()-t0:.0f}s)", flush=True)
        # aggregate
        fam_rec = {"run": run}
        boot = np.random.default_rng(0)
        for h in MERIT_HORIZONS:
            for p in PHASES:
                a = acc[h][p]
                if a["true"] is None or not a["preds"]:
                    continue
                n = len(a["case"])
                seed_merits = [merit_from_predictions(a["true"], yp, np.arange(n))
                               for yp in a["preds"]]
                cases = np.unique(a["case"])
                crows = {c: np.where(a["case"] == c)[0] for c in cases}
                boots = []
                for _ in range(args.n_boot):
                    draw = boot.choice(cases, size=len(cases), replace=True)
                    sel = np.concatenate([crows[c] for c in draw])
                    boots.append(float(np.mean(
                        [merit_from_predictions(a["true"], yp, sel)
                         for yp in a["preds"]])))
                fam_rec[f"h{h}_{p}"] = {
                    "merit_mean": float(np.mean(seed_merits)),
                    "merit_sd": float(np.std(seed_merits)),
                    "ci_lo": float(np.percentile(boots, 2.5)),
                    "ci_hi": float(np.percentile(boots, 97.5)),
                    "n_samples": int(n)}
        results["families"][fam] = fam_rec
        out_path.write_text(json.dumps(results, indent=1))
        line = " ".join(
            f"h{h}/{p}={fam_rec.get(f'h{h}_{p}',{}).get('merit_mean',float('nan')):+.2f}"
            for h in MERIT_HORIZONS for p in PHASES)
        print(f"[t5] {fam} DONE: {line}", flush=True)
    print(f"[t5] all done ({time.time()-t0:.0f}s) -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
