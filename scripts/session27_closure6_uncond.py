"""Session 27 (UNCONDITIONED): mean-over-6 forward physical closure.

ADAPTED COPY of scripts/session20/exp_closure_r2.py (Tracks B + G). The probe
protocol, fit_ridge / apply_probe / fit_probes / match_index helpers and the
held-out R^2 definition are IDENTICAL to the production script (D129/D131). The
only changes here are:

  * the latent / rollout roots point at the UNCONDITIONED tf-no-c (and lstm-no-c)
    artifacts under outputs/session27/ instead of the conditioned exp_b1 export;
  * the headline mean-over-6 R^2 (mean of the six per-metric held-out R^2 at a
    fixed split/horizon/mode) is computed and written to
    outputs/session27/closure6_noc.{json,txt} alongside the per-metric CSV.

The conditioned production HELD-OUT headline is mean-over-6 = 0.445 (test_b, H16,
z_markov), as stated verbatim in section_4_results.tex ("It also leads the mean
over the six observables (0.445 versus 0.147 for both the d=64 reconstructive and
linear bases)"). The "0.84" sometimes quoted is the TRAINING-SET fit (mean 0.835 at
d=64, tab:b1_closure_train_r2), which the manuscript flags as in-distribution, NOT
held-out generalisation. This script reports the unconditioned HELD-OUT mean-over-6
and per-metric for the apples-to-apples comparison vs 0.445, on both z_markov
("markov") and z_full ("fullctx") rollout modes.

Probe protocol (unchanged): a per-metric linear ridge probe fit on the model's
TRAIN per-frame latents (z_full) -> DNS observable, then applied to the rolled-out
latents (z_markov / z_full / z_dns) at frame impact+H. Held-out R^2 uses the
held-out split's own mean as the variance baseline.

Outputs (all under outputs/session27/, nothing else touched):
  closure6_noc.csv   per-(baseline,split,horizon,mode,metric) rows + CIs
  closure6_noc.json  headline mean-over-6 + per-metric, by model/split/horizon/mode
  closure6_noc.txt   human-readable summary
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DNS_METRICS_PATH = REPO / "outputs" / "session17" / "exp2" / "dns_physical_metrics.npz"
SESSION27 = REPO / "outputs" / "session27"

METRICS = ("C_L", "C_D", "I_y", "wake_enstrophy", "circulation_pos", "circulation_neg")

# Unconditioned models. Each entry maps a tag to its (latents_dir, rollouts_dir)
# under outputs/session27/. The encoder is unconditional in both; tf/lstm differ
# only in the predictor that produced the rollouts.
UNCOND_MODELS = {
    "tf-no-c": dict(
        latents=SESSION27 / "latents_own_jepa_noc_tf",
        rollouts=SESSION27 / "rollouts_noc_tf",
        kind="jepa", d=64,
    ),
    "lstm-no-c": dict(
        latents=SESSION27 / "latents_own_jepa_noc_lstm",
        rollouts=SESSION27 / "rollouts_noc_lstm",
        kind="lstm", d=64,
    ),
}


def fit_ridge(Z: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> dict:
    """Standardise Z, fit linear ridge. Identical to physical_metrics_from_rollouts."""
    Z = Z.astype(np.float64)
    y = y.astype(np.float64)
    mu_z = Z.mean(axis=0)
    sigma_z = Z.std(axis=0).clip(min=1e-9)
    Zn = (Z - mu_z) / sigma_z
    yc = y - y.mean()
    A = Zn.T @ Zn + alpha * np.eye(Zn.shape[1])
    W = np.linalg.solve(A, Zn.T @ yc)
    return {"W": W, "mu_z": mu_z, "sigma_z": sigma_z, "b": float(y.mean())}


def apply_probe(z: np.ndarray, probe: dict) -> np.ndarray:
    Zn = (z.astype(np.float64) - probe["mu_z"]) / probe["sigma_z"]
    return Zn @ probe["W"] + probe["b"]


def _get(blob, *names):
    for n in names:
        if n in blob.files:
            return blob[n]
    raise KeyError(f"none of {names} present in npz {blob}")


def match_index(cid_a, ei_a, cid_b, ei_b) -> np.ndarray:
    idx = {(str(c), int(e)): i for i, (c, e) in enumerate(zip(cid_b, ei_b))}
    return np.array([idx.get((str(c), int(e)), -1) for c, e in zip(cid_a, ei_a)], dtype=np.int64)


def fit_probes(latents_dir: Path, dns) -> dict:
    train = np.load(latents_dir / "train.npz", allow_pickle=True)
    z_full = train["z_full"].astype(np.float32)
    cid = _get(train, "case_ids", "case_id")
    ei = _get(train, "encounter_indices", "encounter_index")
    di = match_index(cid, ei, dns["train_case_id"], dns["train_encounter_index"])
    keep = di >= 0
    z_used = z_full[keep]
    di_used = di[keep]
    n, T, d = z_used.shape
    Zf = z_used.reshape(n * T, d)
    probes = {}
    for m in METRICS:
        y = dns[f"train_{m}"][di_used].reshape(n * T).astype(np.float64)
        probes[m] = fit_ridge(Zf, y)
    return probes


def boot_ci(fn, n, rng, n_resamples=2000, ci=0.95):
    """Bootstrap CI of a statistic fn(idx) over n samples."""
    if n == 0:
        return float("nan"), float("nan")
    stats = np.array([fn(rng.integers(0, n, size=n)) for _ in range(n_resamples)])
    a = (1 - ci) / 2
    return float(np.nanquantile(stats, a)), float(np.nanquantile(stats, 1 - a))


def evaluate(tag, kind, d, latents_dir, rollouts_dir, dns, horizons, splits,
             seed=0, n_boot=2000) -> list[dict]:
    if not (latents_dir / "train.npz").exists():
        print(f"[closure_r2] SKIP {tag}: {latents_dir/'train.npz'} missing")
        return []
    probes = fit_probes(latents_dir, dns)
    rows = []
    rng = np.random.default_rng(seed)
    for split in splits:
        rp = rollouts_dir / f"{split}.npz"
        if not rp.exists():
            print(f"[closure_r2] SKIP {tag} {split}: {rp} missing")
            continue
        blob = np.load(rp, allow_pickle=True)
        cid = _get(blob, "case_ids", "case_id")
        ei = _get(blob, "encounter_indices", "encounter_index")
        impact = _get(blob, "impact_frame").astype(np.int64)
        di = match_index(cid, ei, dns[f"{split}_case_id"], dns[f"{split}_encounter_index"])
        keep = np.where(di >= 0)[0]
        modes = {"z_dns": blob["z_dns"], "z_markov": blob["z_markov"], "z_full": blob["z_full"]}
        T = blob["z_dns"].shape[1]
        for mode, zarr in modes.items():
            zarr = zarr.astype(np.float32)
            for m in METRICS:
                probe = probes[m]
                for H in horizons:
                    yp, yt = [], []
                    for i in keep:
                        te = int(impact[i]) + H
                        if te >= T:
                            continue
                        yp.append(float(apply_probe(zarr[i, te].reshape(1, d), probe)[0]))
                        yt.append(float(dns[f"{split}_{m}"][di[i], te]))
                    if len(yt) < 3:
                        continue
                    yp = np.asarray(yp); yt = np.asarray(yt)
                    def r2_of(idx):
                        a, b = yp[idx], yt[idx]
                        ss_res = float(((a - b) ** 2).sum())
                        ss_tot = float(((b - b.mean()) ** 2).sum())
                        return 1.0 - ss_res / max(ss_tot, 1e-12)
                    def mae_of(idx):
                        return float(np.abs(yp[idx] - yt[idx]).mean())
                    full = np.arange(len(yt))
                    r2 = r2_of(full); mae = mae_of(full)
                    r2lo, r2hi = boot_ci(r2_of, len(yt), rng, n_boot)
                    maelo, maehi = boot_ci(mae_of, len(yt), rng, n_boot)
                    rows.append(dict(
                        baseline=tag, kind=kind, d=d, split=split, horizon=int(H),
                        mode=mode, metric=m, n_enc=len(yt),
                        r2=r2, r2_ci_lo=r2lo, r2_ci_hi=r2hi,
                        mae=mae, mae_ci_lo=maelo, mae_ci_hi=maehi,
                    ))
        print(f"[closure_r2] {tag} {split}: {sum(1 for r in rows if r['split']==split and r['baseline']==tag)} rows")
    return rows


def mean_over_6(rows, baseline, split, horizon, mode) -> dict:
    """Mean of the six per-metric held-out R^2 at a fixed slice. Returns the
    per-metric R^2 dict and the mean (the headline number)."""
    sel = {r["metric"]: r for r in rows
           if r["baseline"] == baseline and r["split"] == split
           and r["horizon"] == horizon and r["mode"] == mode}
    per = {m: (sel[m]["r2"] if m in sel else None) for m in METRICS}
    vals = [per[m] for m in METRICS if per[m] is not None]
    return {
        "per_metric": per,
        "mean_over_6": float(np.mean(vals)) if vals else None,
        "n_metrics": len(vals),
        "n_enc": int(sel[METRICS[0]]["n_enc"]) if METRICS[0] in sel else None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--horizons", nargs="+", type=int, default=[8, 16, 32])
    p.add_argument("--splits", nargs="+", default=["test_b", "test_c"])
    p.add_argument("--models", nargs="+", default=["tf-no-c", "lstm-no-c"])
    p.add_argument("--out-csv", type=Path, default=SESSION27 / "closure6_noc.csv")
    p.add_argument("--out-json", type=Path, default=SESSION27 / "closure6_noc.json")
    p.add_argument("--out-txt", type=Path, default=SESSION27 / "closure6_noc.txt")
    p.add_argument("--n-bootstrap", type=int, default=2000)
    args = p.parse_args()

    dns = np.load(DNS_METRICS_PATH, allow_pickle=True)
    rows = []
    for tag in args.models:
        spec = UNCOND_MODELS[tag]
        rows += evaluate(tag, spec["kind"], spec["d"], spec["latents"], spec["rollouts"],
                         dns, tuple(args.horizons), args.splits, n_boot=args.n_bootstrap)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[closure6] wrote {len(rows)} rows to {args.out_csv}")

    # headline mean-over-6 by model/split/horizon/mode (markov = z_markov, fullctx = z_full)
    MODE_LABEL = {"z_markov": "markov", "z_full": "fullctx", "z_dns": "dns"}
    headline = {}
    for tag in args.models:
        headline[tag] = {}
        for split in args.splits:
            headline[tag][split] = {}
            for H in args.horizons:
                headline[tag][split][str(H)] = {}
                for raw_mode, label in MODE_LABEL.items():
                    headline[tag][split][str(H)][label] = mean_over_6(rows, tag, split, H, raw_mode)
    bundle = {
        "description": "Unconditioned mean-over-6 forward physical closure (held-out R^2). "
                       "Conditioned HELD-OUT production headline = 0.445 (test_b, H16, z_markov); "
                       "the 0.84 sometimes quoted is the TRAINING-set fit (0.835), not held-out.",
        "metrics": list(METRICS),
        "conditioned_reference_heldout": {"jepa_d64_markov_H16_test_b_mean_over_6": 0.445,
                                          "fukami_d64": 0.147, "pod_d64": 0.147},
        "conditioned_reference_trainfit": {"jepa_d64_mean_over_6": 0.835},
        "headline": headline,
    }
    args.out_json.write_text(json.dumps(bundle, indent=2))
    print(f"[closure6] wrote {args.out_json}")

    # human-readable summary, focused on the headline H16/test_b slice
    lines = ["UNCONDITIONED mean-over-6 forward physical closure (held-out R^2)",
             "Metrics: " + ", ".join(METRICS),
             "Conditioned HELD-OUT reference (test_b, H16, z_markov): mean-over-6 = 0.445",
             "  (d=64 reconstructive 0.147, linear/POD 0.147; per section_4_results.tex)",
             "  NOTE: the 0.84 sometimes quoted is the TRAINING-set fit (0.835), not held-out.", ""]
    for tag in args.models:
        for split in args.splits:
            for H in args.horizons:
                for label in ("markov", "fullctx"):
                    r = headline[tag][split][str(H)][label]
                    if r["mean_over_6"] is None:
                        continue
                    pm = "  ".join(f"{m}={r['per_metric'][m]:+.3f}" if r['per_metric'][m] is not None
                                   else f"{m}=NA" for m in METRICS)
                    lines.append(f"[{tag} | {split} | H={H} | {label}] "
                                 f"mean-over-6 = {r['mean_over_6']:+.4f}  (n_enc={r['n_enc']})")
                    lines.append("    " + pm)
        lines.append("")
    args.out_txt.write_text("\n".join(lines))
    print(f"[closure6] wrote {args.out_txt}")
    # echo the headline
    for tag in args.models:
        r = headline[tag]["test_b"]["16"]["markov"]
        if r["mean_over_6"] is not None:
            print(f"  HEADLINE {tag} test_b H16 markov: mean-over-6 = {r['mean_over_6']:.4f}")


if __name__ == "__main__":
    main()
