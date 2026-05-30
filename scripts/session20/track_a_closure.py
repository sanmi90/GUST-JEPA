"""Session 20 Track A gate: 2x2 objective x architecture closure + claim strength.

Reads the Track A per-cell/per-seed latents and rollouts produced by
eval_track_a.sh, fits the SAME per-metric ridge probe used in B1/Track B
(imported from exp_closure_r2), and computes held-out test_b closure R^2 and MAE
at H=16 (z_markov) for each of the 5 cells, averaged over >=3 seeds.

Cells:
  A1 predictive    CNN+ViT  lift+wake   (thrust6 jepa_d64_seed{0,1,2})
  A2 predictive    CNN only lift+wake
  A3 reconstructive CNN+ViT lift+wake
  A4 reconstructive CNN only lift+wake
  A5 predictive    CNN+ViT  lift only (no wake)

Gate (decides the central-claim wording, per SESSION20_PLAN.md):
  - A1 > A3 AND A2 > A4 on wake closure  -> "the predictive objective improves
    closure" (STRONG); abstract + Section 4.5 take that wording.
  - A1 > A3 but A2 ~ A4                   -> "the predictive CNN+ViT family
    improves closure" (the win needs the ViT).
  - A5 << A1 on wake closure             -> "wake supervision drives the
    wake-closure advantage" (WEAKEST/most honest); abstract foregrounds it.
The script prints the verdict and writes tab:controls_2x2 numbers.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "session20"))
from exp_closure_r2 import METRICS, apply_probe, fit_probes, match_index  # noqa: E402

EV = REPO / "outputs" / "session20" / "track_a"
DNS = np.load(REPO / "outputs/session17/exp2/dns_physical_metrics.npz", allow_pickle=True)
CELLS = {
    "A1_pred_cnnvit": ("predictive", "CNN+ViT", "lift+wake"),
    "A2_pred_cnn": ("predictive", "CNN", "lift+wake"),
    "A3_recon_cnnvit": ("reconstructive", "CNN+ViT", "lift+wake"),
    "A4_recon_cnn": ("reconstructive", "CNN", "lift+wake"),
    "A5_pred_nowake": ("predictive", "CNN+ViT", "lift only"),
}
SEEDS = (0, 1, 2)
H = 16
SPLIT = "test_b"


def closure_one(latents_dir: Path, rollout_npz: Path) -> dict:
    """Held-out R^2 + MAE at H, z_markov, for each observable (one cell/seed)."""
    probes = fit_probes(latents_dir, DNS)
    blob = np.load(rollout_npz, allow_pickle=True)
    z = blob["z_markov"].astype(np.float32)
    cid = blob["case_ids"] if "case_ids" in blob.files else blob["case_id"]
    ei = blob["encounter_indices"] if "encounter_indices" in blob.files else blob["encounter_index"]
    impact = blob["impact_frame"].astype(np.int64)
    di = match_index(cid, ei, DNS[f"{SPLIT}_case_id"], DNS[f"{SPLIT}_encounter_index"])
    keep = np.where(di >= 0)[0]
    d = z.shape[2]
    out = {}
    for m in METRICS:
        probe = probes[m]
        yp, yt = [], []
        for i in keep:
            te = int(impact[i]) + H
            if te >= z.shape[1]:
                continue
            yp.append(float(apply_probe(z[i, te].reshape(1, d), probe)[0]))
            yt.append(float(DNS[f"{SPLIT}_{m}"][di[i], te]))
        yp, yt = np.asarray(yp), np.asarray(yt)
        ss_res = float(((yp - yt) ** 2).sum())
        ss_tot = float(((yt - yt.mean()) ** 2).sum())
        out[m] = {"r2": 1.0 - ss_res / max(ss_tot, 1e-12), "mae": float(np.abs(yp - yt).mean())}
    return out


def main() -> None:
    per_cell = {}
    rows = []
    for cell in CELLS:
        seed_vals = {m: {"r2": [], "mae": []} for m in METRICS}
        n_ok = 0
        for s in SEEDS:
            lat = EV / f"latents_{cell}_seed{s}"
            roll = EV / f"rollouts_{cell}_seed{s}_noBN" / f"{SPLIT}.npz"
            if not (lat / "train.npz").exists() or not roll.exists():
                print(f"[track-a-closure] MISSING {cell} seed{s} (lat or rollout); skipping seed")
                continue
            c = closure_one(lat, roll)
            for m in METRICS:
                seed_vals[m]["r2"].append(c[m]["r2"])
                seed_vals[m]["mae"].append(c[m]["mae"])
            n_ok += 1
        agg = {}
        for m in METRICS:
            r2a = np.asarray(seed_vals[m]["r2"]); maea = np.asarray(seed_vals[m]["mae"])
            agg[m] = {
                "r2_mean": float(r2a.mean()) if r2a.size else float("nan"),
                "r2_std": float(r2a.std()) if r2a.size else float("nan"),
                "mae_mean": float(maea.mean()) if maea.size else float("nan"),
                "mae_std": float(maea.std()) if maea.size else float("nan"),
                "n_seeds": int(r2a.size),
            }
            rows.append({"cell": cell, "objective": CELLS[cell][0], "encoder": CELLS[cell][1],
                         "aux": CELLS[cell][2], "metric": m, **agg[m]})
        agg["mean_over_observables_r2"] = float(np.mean([agg[m]["r2_mean"] for m in METRICS]))
        per_cell[cell] = agg
        print(f"{cell:18s} ({CELLS[cell][0]:14s} {CELLS[cell][1]:8s} {CELLS[cell][2]:9s}) "
              f"n_seeds={n_ok}  wake_R2={agg['wake_enstrophy']['r2_mean']:+.3f}"
              f"+-{agg['wake_enstrophy']['r2_std']:.3f}  meanR2={agg['mean_over_observables_r2']:+.3f}")

    # ---- gate ----
    def wake(c):
        return per_cell.get(c, {}).get("wake_enstrophy", {}).get("r2_mean", float("nan"))
    a1, a2, a3, a4, a5 = (wake(c) for c in
                          ["A1_pred_cnnvit", "A2_pred_cnn", "A3_recon_cnnvit", "A4_recon_cnn", "A5_pred_nowake"])
    margin = 0.05  # R^2 margin to call a difference meaningful
    a1_gt_a3 = (a1 - a3) > margin
    a2_gt_a4 = (a2 - a4) > margin
    a5_much_less = (a1 - a5) > margin  # removing wake head hurts wake closure a lot
    if a5_much_less and not (a1_gt_a3 and a2_gt_a4):
        verdict = "WAKE_SUPERVISION_DRIVES"
        claim = ("wake supervision drives the wake-closure advantage; the abstract "
                 "must foreground the auxiliary head, not the objective")
    elif a1_gt_a3 and a2_gt_a4:
        verdict = "PREDICTIVE_OBJECTIVE_WINS"
        claim = "the predictive objective improves closure at BOTH architectures (strong form)"
    elif a1_gt_a3 and not a2_gt_a4:
        verdict = "PREDICTIVE_CNNVIT_FAMILY_WINS"
        claim = "the predictive CNN+ViT family improves closure; the win needs the ViT"
    else:
        verdict = "NO_CLEAN_SEPARATION"
        claim = ("the objective does not cleanly separate from architecture/aux on wake "
                 "closure at this scale; report descriptively")
    gate = {
        "wake_r2": {"A1": a1, "A2": a2, "A3": a3, "A4": a4, "A5": a5},
        "A1_minus_A3": a1 - a3, "A2_minus_A4": a2 - a4, "A1_minus_A5": a1 - a5,
        "margin": margin, "verdict": verdict, "claim_strength": claim,
    }
    print("\n=== TRACK A GATE ===")
    print(f"  wake R^2: A1={a1:+.3f} A2={a2:+.3f} A3={a3:+.3f} A4={a4:+.3f} A5={a5:+.3f}")
    print(f"  A1-A3={a1-a3:+.3f}  A2-A4={a2-a4:+.3f}  A1-A5={a1-a5:+.3f}  (margin {margin})")
    print(f"  VERDICT: {verdict}\n  CLAIM: {claim}")

    EV.mkdir(parents=True, exist_ok=True)
    with open(EV / "controls_2x2.json", "w") as f:
        json.dump({"per_cell": per_cell, "gate": gate, "horizon": H, "split": SPLIT}, f, indent=2)
    with open(EV / "controls_2x2.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n  wrote {EV/'controls_2x2.json'} and .csv")


if __name__ == "__main__":
    main()
