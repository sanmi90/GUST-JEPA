"""Session 20 Tracks B + G: held-out R^2 and the horizon sweep.

Extends scripts/session18/physical_metrics_from_rollouts.py (which reports only
MAE at H in {8,16,32}) to also report held-out coefficient of determination R^2,
across an arbitrary horizon list, for every B1 baseline on test_b and test_c.

Probe protocol is IDENTICAL to D129/D131: a per-metric linear ridge probe fit on
the baseline's TRAIN per-frame latents (z_full) -> DNS observable, then applied to
the rolled-out latents (z_markov / z_full / z_dns) at frame impact+H.

Held-out R^2 for a fixed (split, horizon, mode, metric):
    collect over held-out encounters the pairs (y_pred_i, y_true_i) at impact_i+H,
    R^2 = 1 - sum_i (y_pred_i - y_true_i)^2 / sum_i (y_true_i - mean_i y_true)^2.
This is a genuine held-out R^2 (the variance baseline is the held-out split's own
mean), unlike the training-set probe R^2 the original draft reported as headline.

Track B  = the H=16, mode=z_markov slice of this CSV (the conditioning-floor and
            tab:b1_r2_heldout numbers).
Track G  = the full horizon sweep, R^2 and MAE vs H.

Output: outputs/session20/closure_r2/closure_r2_heldout.csv with columns
    baseline, kind, d, split, horizon, mode, metric, n_enc,
    r2, r2_ci_lo, r2_ci_hi, mae, mae_ci_lo, mae_ci_hi.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DNS_METRICS_PATH = REPO / "outputs" / "session17" / "exp2" / "dns_physical_metrics.npz"
LATENTS_ROOT = REPO / "outputs" / "session18" / "exp_b1"
ROLLOUTS_ROOT = REPO / "outputs" / "session18" / "exp_b1_test3"

METRICS = ("C_L", "C_D", "I_y", "wake_enstrophy", "circulation_pos", "circulation_neg")

# (tag, kind, d): the 8 B1 v2 baselines (matches physical_closure_noBN_unified.csv).
DEFAULT_BASELINES = [
    ("jepa_d64_test1_noBN", "jepa", 64),
    ("jepa_d32_noBN", "jepa", 32),
    ("fukami_d3_noBN", "fukami", 3),
    ("fukami_d32_noBN", "fukami", 32),
    ("fukami_d64_noBN", "fukami", 64),
    ("pod_d16_noBN", "pod", 16),
    ("pod_d32_noBN", "pod", 32),
    ("pod_d64_noBN", "pod", 64),
]


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


def evaluate(tag, kind, d, dns, horizons, splits, seed=0, n_boot=2000) -> list[dict]:
    latents_dir = LATENTS_ROOT / f"latents_{tag}"
    rollouts_dir = ROLLOUTS_ROOT / f"rollouts_{tag}"
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--horizons", nargs="+", type=int, default=[1, 4, 8, 16, 32, 64])
    p.add_argument("--splits", nargs="+", default=["test_b", "test_c"])
    p.add_argument("--out", type=Path, default=REPO / "outputs/session20/closure_r2/closure_r2_heldout.csv")
    p.add_argument("--n-bootstrap", type=int, default=2000)
    args = p.parse_args()

    dns = np.load(DNS_METRICS_PATH, allow_pickle=True)
    rows = []
    for tag, kind, d in DEFAULT_BASELINES:
        rows += evaluate(tag, kind, d, dns, tuple(args.horizons), args.splits, n_boot=args.n_bootstrap)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[closure_r2] wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
