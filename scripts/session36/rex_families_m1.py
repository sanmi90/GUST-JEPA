"""Track M1: shared-operator forecast merit for the ten tab:closure families.

The manuscript's families table (tab:closure, the editorial memo's table 6)
scores forecast merit under SUITED operators (matched transformer on the
pooled controlled-matrix states, residual U-Net on the reference latents),
which confounds the primary cross-family comparison. This script fits ONE
shared operator, the manuscript-selected direct multi-horizon quantile
forecaster of s3.2.1 (LSTM width 512, nine quantiles 0.1..0.9, pinball loss,
AdamW 1e-3 / weight decay 1e-4, cosine decay, batch 64, gradient clip 1.0,
6000 iterations, per-window instance norm + arcsinh, training context
sampled 16..30, horizon 40; selection provenance scripts/session34/rex_tune.py
winner), identically on every family's frozen cached latents, three operator
seeds each.

Merit definition mirrors the existing suited-operator column
(scripts/session33/emit_numbers_parts.py part_table_x): mean over the five
observables (C_L, C_D, wake_enstrophy, circulation_pos, circulation_neg) of
the pooled MLP-probe R^2 at a fixed horizon, probes fit per family on its
own train latents (src/evaluation/rollout.py fit_observable_probes, frozen
protocol), targets restricted to rows whose frame lies in the impact +
relaxation window (window_mask), anchors sliding with a 25-frame context.
The existing column is computed at horizon EIGHT despite the caption's
"horizon sixteen" (REVIEW-NUMBER, MANUSCRIPT_AUDIT.md catch on tab:closure);
this script therefore reports BOTH h=8 and h=16.

Uncertainty: case-clustered bootstrap (2000 resamples of the test_b cases,
merit recomputed per resample on the pooled samples of the drawn cases,
averaged over the three operator seeds), percentile 95% CI.

Run (RTX 6000, CPU-capped):
    taskset -c 0-15 python -m scripts.session36.rex_families_m1 --gpu 1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.session34.latent_rex import LatentRex  # noqa: E402  (architecture reused)

CACHE = REPO_ROOT / "outputs/session34/trackc_latents"
OBSERVABLES = ("C_L", "C_D", "wake_enstrophy", "circulation_pos", "circulation_neg")
Q9 = tuple(np.round(np.arange(0.1, 0.91, 0.1), 1))  # rex_tune.py winning grid cell
MEDIAN_IDX = 4  # q = 0.5 in Q9

# tab:closure paper row -> latent cache run name (mapping mirrors
# emit_numbers_parts.part_table_x rows; see editorial/PROVENANCE.md)
FAMILIES = {
    "JepaWake": "jepa_pool_vec",
    "SupOnly": "supervised_only_pool",
    "AeWake": "ae_wake_pool",
    "JepaNowake": "jepa_nowake_pool",
    "AeNowake": "ae_nowake_pool",
    "RegAE": "regAE_pool",
    "Bvae": "bvae",
    "Fukami": "fukami",
    "FukamiWake": "fukami_wake",
    "Pod": "pod",
}

CTX = 25
HORIZON = 40
MERIT_HORIZONS = (8, 16)


def load_split(run: str, split: str) -> dict:
    path = CACHE / f"latents_{run}_{split}.npz"
    z = np.load(path, allow_pickle=True)
    out = {
        "z_gap": z["z_gap"].astype(np.float32),
        "case_id": z["case_id"],
        "encounter_index": z["encounter_index"],
        "frame": z["frame"],
        "window_mask": z["window_mask"],
    }
    for obs in OBSERVABLES:
        out[f"target_{obs}"] = z[f"target_{obs}"]
    return out


def group(cache: dict) -> list[dict]:
    keys = list(zip(cache["case_id"].tolist(), cache["encounter_index"].tolist()))
    rows: dict[tuple, list[int]] = {}
    for i, key in enumerate(keys):
        rows.setdefault(key, []).append(i)
    encs = []
    for (cid, k), idx in rows.items():
        idx = np.asarray(idx)
        idx = idx[np.argsort(cache["frame"][idx])]
        encs.append({"case_id": cid, "rows": idx})
    return encs


def pinball(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    loss = 0.0
    for i, q in enumerate(Q9):
        e = target - pred[..., i]
        loss = loss + torch.maximum(q * e, (q - 1) * e).mean()
    return loss / len(Q9)


def train_operator(Zt: torch.Tensor, seed: int, iters: int, device) -> LatentRex:
    n_ep, T, d = Zt.shape
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = LatentRex(d=d, hidden=512, horizon=HORIZON, nq=len(Q9)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    for it in range(1, iters + 1):
        ep = rng.integers(0, n_ep, size=64)
        ctx_len = int(rng.integers(16, CTX + 6))  # 16..30, as latent_rex/rex_tune
        s = rng.integers(0, T - ctx_len - HORIZON + 1, size=64)
        idx = s[:, None] + np.arange(ctx_len)[None]
        ctx = Zt[torch.from_numpy(ep).long()[:, None], torch.from_numpy(idx).long()]
        tgt_idx = (s + ctx_len)[:, None] + np.arange(HORIZON)[None]
        tgt = Zt[torch.from_numpy(ep).long()[:, None], torch.from_numpy(tgt_idx).long()]
        loss = pinball(model(ctx), tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    model.eval()
    return model


def eval_samples(model: LatentRex, tb: dict, encs, device) -> dict[int, dict]:
    """Forecast every valid sliding anchor; keep (anchor, h) samples whose
    target frame is inside the impact+relaxation window. Returns per merit
    horizon: rolled latents, target rows, case labels."""
    d = tb["z_gap"].shape[1]
    out = {h: {"z": [], "row": [], "case": []} for h in MERIT_HORIZONS}
    for e in encs:
        rows = e["rows"]
        Z = tb["z_gap"][rows]  # (T, d)
        wm = tb["window_mask"][rows]
        T = Z.shape[0]
        anchors = np.arange(CTX - 1, T - 1)  # last context frame index
        ctx = np.stack([Z[a - CTX + 1 : a + 1] for a in anchors])
        with torch.no_grad():
            pred = model(torch.from_numpy(ctx).float().to(device)).cpu().numpy()
        roll = pred[..., MEDIAN_IDX]  # (A, HORIZON, d)
        for h in MERIT_HORIZONS:
            tgt = anchors + h
            ok = (tgt <= T - 1) & wm[np.clip(tgt, 0, T - 1)]
            if not ok.any():
                continue
            out[h]["z"].append(roll[ok, h - 1])
            out[h]["row"].append(rows[tgt[ok]])
            out[h]["case"].append(np.array([e["case_id"]] * int(ok.sum())))
    for h in MERIT_HORIZONS:
        out[h] = {k: np.concatenate(v) for k, v in out[h].items()}
    return out


def merit_from_predictions(y_true: dict, y_pred: dict, sel: np.ndarray) -> float:
    from src.evaluation.represent import r2_score_np

    vals = []
    for obs in OBSERVABLES:
        vals.append(r2_score_np(y_true[obs][sel], y_pred[obs][sel]))
    return float(np.mean(vals))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--families", nargs="+", default=list(FAMILIES))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="outputs/session36/m1_shared_merit.json")
    args = ap.parse_args()

    from src.evaluation.rollout import fit_observable_probes
    from src.utils.device import require_rtx6000

    device = require_rtx6000(gpu_index=args.gpu)
    git = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
    ).stdout.strip()
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "_provenance": {
            "script": "scripts/session36/rex_families_m1.py",
            "git_commit": git,
            "operator": "LatentRex LSTM h512 x 2 layers, 9 quantiles 0.1..0.9, "
                        "pinball, AdamW 1e-3 wd 1e-4, cosine, batch 64, clip 1.0, "
                        f"iters {args.iters}, train ctx 16..30, horizon {HORIZON}, "
                        f"eval ctx {CTX} sliding anchors, targets in window_mask",
            "merit": "mean over five observables of pooled MLP-probe R2 "
                     "(probes fit per family on its own train latents, "
                     "fit_observable_probes seed 0); horizons "
                     + str(list(MERIT_HORIZONS)),
            "split": "v2p2 test_b (42 encounters, 10 cases)",
            "cache": str(CACHE.relative_to(REPO_ROOT)),
            "seeds": args.seeds,
            "n_boot": args.n_boot,
            "gpu_name": torch.cuda.get_device_name(device.index),
        },
        "families": {},
    }

    t0 = time.time()
    for fam in args.families:
        run = FAMILIES[fam]
        tr = load_split(run, "train")
        tb = load_split(run, "test_b")
        encs_tr = group(tr)
        encs_tb = group(tb)
        Ztr = np.stack([tr["z_gap"][e["rows"]] for e in encs_tr])
        Zt = torch.from_numpy(Ztr).float().to(device)
        targets_tr = {obs: tr[f"target_{obs}"] for obs in OBSERVABLES}
        print(f"[m1] {fam} ({run}): probes...", flush=True)
        probes = fit_observable_probes(tr["z_gap"], targets_tr)

        fam_rec = {"run": run, "seeds": {}, "n_train_episodes": len(encs_tr)}
        # per horizon: per seed y_pred; shared y_true and case labels
        per_h: dict[int, dict] = {}
        for seed in args.seeds:
            model = train_operator(Zt, seed, args.iters, device)
            samples = eval_samples(model, tb, encs_tb, device)
            seed_rec = {}
            for h in MERIT_HORIZONS:
                s = samples[h]
                y_true = {o: tb[f"target_{o}"][s["row"]] for o in OBSERVABLES}
                y_pred = {o: probes[o]["mlp"].predict(s["z"]) for o in OBSERVABLES}
                ph = per_h.setdefault(h, {"case": s["case"], "true": y_true, "preds": []})
                ph["preds"].append(y_pred)
                all_sel = np.arange(s["z"].shape[0])
                seed_rec[f"merit_h{h}"] = merit_from_predictions(y_true, y_pred, all_sel)
                seed_rec[f"n_samples_h{h}"] = int(s["z"].shape[0])
            fam_rec["seeds"][str(seed)] = seed_rec
            del model
            torch.cuda.empty_cache()
            print(f"[m1] {fam} seed {seed}: "
                  + " ".join(f"h{h}={seed_rec[f'merit_h{h}']:+.3f}" for h in MERIT_HORIZONS)
                  + f" ({time.time() - t0:.0f}s)", flush=True)

        # case-clustered bootstrap on the seed-mean merit
        boot_rng = np.random.default_rng(0)
        for h in MERIT_HORIZONS:
            ph = per_h[h]
            cases = np.unique(ph["case"])
            case_rows = {c: np.where(ph["case"] == c)[0] for c in cases}
            seed_merits = [
                merit_from_predictions(ph["true"], yp, np.arange(len(ph["case"])))
                for yp in ph["preds"]
            ]
            boots = []
            for _ in range(args.n_boot):
                draw = boot_rng.choice(cases, size=len(cases), replace=True)
                sel = np.concatenate([case_rows[c] for c in draw])
                boots.append(
                    float(np.mean([
                        merit_from_predictions(ph["true"], yp, sel) for yp in ph["preds"]
                    ]))
                )
            fam_rec[f"merit_h{h}_mean"] = float(np.mean(seed_merits))
            fam_rec[f"merit_h{h}_sd"] = float(np.std(seed_merits))
            fam_rec[f"merit_h{h}_ci_lo"] = float(np.percentile(boots, 2.5))
            fam_rec[f"merit_h{h}_ci_hi"] = float(np.percentile(boots, 97.5))
        results["families"][fam] = fam_rec
        out_path.write_text(json.dumps(results, indent=1))
        print(f"[m1] {fam} DONE: "
              + " ".join(
                  f"h{h}={fam_rec[f'merit_h{h}_mean']:+.3f}"
                  f"[{fam_rec[f'merit_h{h}_ci_lo']:+.3f},{fam_rec[f'merit_h{h}_ci_hi']:+.3f}]"
                  for h in MERIT_HORIZONS)
              + f" ({time.time() - t0:.0f}s)", flush=True)

    results["_provenance"]["wall_s"] = time.time() - t0
    out_path.write_text(json.dumps(results, indent=1))
    print(f"[m1] all done in {time.time() - t0:.0f}s -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
