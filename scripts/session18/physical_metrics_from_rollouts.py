"""Session 18 B1 Part (d) step 2: physical-metric comparison table.

For each (baseline, d) pair with precomputed latents + rollouts:

  1. Fit per-metric linear ridge probes on per-frame TRAIN samples:
        z_train -> {C_L, I_y, wake_enstrophy, circulation_pos, circulation_neg, C_D}
     using the DNS physical metrics at outputs/session17/exp2/dns_physical_metrics.npz
     and the precomputed latents at outputs/session18/exp_b1/latents_{tag}/train.npz.

  2. Apply probes to:
        z_dns   (ground-truth latents, sanity check)
        z_markov (Markov-only rollout from z_impact)
        z_full   (Full-context rollout from z[:impact+1])
     in outputs/session18/exp_b1/rollouts_{tag}/{test_b,test_c}.npz.

  3. Compute per-encounter absolute error at H = 8, 16, 32 frames past impact:
        abs_err[i, mode, metric, H] =
            | metric_pred(z_mode[i, impact[i] + H]) - DNS_metric[i, impact[i] + H] |

  4. Bootstrap 2000-resample 95% CI per (baseline, d, split, horizon, mode, metric).

  5. Write outputs/session18/exp_b1/physical_closure_comparison.csv with columns:
        baseline, d, split, horizon, mode, metric, n_enc,
        abs_err_mean, abs_err_median, ci_lo, ci_hi

This produces the headline 7 x 4 comparison table for paper Figure 5.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


METRICS = (
    "C_L",
    "C_D",
    "I_y",
    "wake_enstrophy",
    "circulation_pos",
    "circulation_neg",
)
DNS_METRICS_PATH = REPO / "outputs" / "session17" / "exp2" / "dns_physical_metrics.npz"


def fit_ridge(Z: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> dict:
    """Standardize Z, fit ridge regression. Returns dict for inference."""
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


def match_dns_to_latents(
    cid_lat: np.ndarray, ei_lat: np.ndarray,
    cid_dns: np.ndarray, ei_dns: np.ndarray,
) -> np.ndarray:
    """Return index_in_dns for each (cid_lat[i], ei_lat[i]); -1 if missing."""
    dns_index = {(str(c), int(e)): i for i, (c, e) in enumerate(zip(cid_dns, ei_dns))}
    out = np.full(len(cid_lat), -1, dtype=np.int64)
    for i, (c, e) in enumerate(zip(cid_lat, ei_lat)):
        out[i] = dns_index.get((str(c), int(e)), -1)
    return out


def bootstrap_ci(values: np.ndarray, n_resamples: int, ci_level: float, rng: np.random.Generator) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    n = len(values)
    means = np.empty(n_resamples, dtype=np.float64)
    for k in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[k] = float(values[idx].mean())
    alpha = (1.0 - ci_level) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def _get(blob, *names):
    for n in names:
        if n in blob.files:
            return blob[n]
    raise KeyError(f"none of {names} present in npz")


def fit_probes_for_baseline(
    latents_dir: Path,
    dns: np.lib.npyio.NpzFile,
    metrics: tuple[str, ...] = METRICS,
    ridge_alpha: float = 1.0,
) -> dict:
    """Fit ridge probes on the baseline's TRAIN per-frame latents."""
    train_lat = np.load(latents_dir / "train.npz", allow_pickle=True)
    z_full = train_lat["z_full"].astype(np.float32)  # (n_enc, 120, d)
    cid_lat = _get(train_lat, "case_ids", "case_id")
    ei_lat = _get(train_lat, "encounter_indices", "encounter_index")

    dns_idx = match_dns_to_latents(
        cid_lat, ei_lat, dns["train_case_id"], dns["train_encounter_index"]
    )
    keep_mask = dns_idx >= 0
    if not keep_mask.all():
        n_missing = int((~keep_mask).sum())
        print(
            f"   [fit] WARNING: {n_missing} train latents had no matching DNS row; "
            "dropped from probe fit."
        )

    z_used = z_full[keep_mask]  # (n_keep, 120, d)
    dns_idx_used = dns_idx[keep_mask]
    n_keep, T, d = z_used.shape
    Z_flat = z_used.reshape(n_keep * T, d)

    probes: dict[str, dict] = {}
    train_r2: dict[str, float] = {}
    for metric in metrics:
        y_per_enc = dns[f"train_{metric}"][dns_idx_used]  # (n_keep, 120)
        y_flat = y_per_enc.reshape(n_keep * T).astype(np.float64)
        probe = fit_ridge(Z_flat, y_flat, alpha=ridge_alpha)
        # Train R^2 for sanity
        y_pred = apply_probe(Z_flat, probe)
        ss_res = float(((y_flat - y_pred) ** 2).sum())
        ss_tot = float(((y_flat - y_flat.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        probes[metric] = probe
        train_r2[metric] = r2
    return {"probes": probes, "train_r2": train_r2, "n_train_enc": int(n_keep)}


def evaluate_split(
    latents_dir: Path,
    rollouts_dir: Path,
    split: str,
    probes: dict,
    dns: np.lib.npyio.NpzFile,
    horizons: tuple[int, ...] = (8, 16, 32),
    metrics: tuple[str, ...] = METRICS,
    n_resamples: int = 2000,
    ci_level: float = 0.95,
    seed: int = 0,
) -> list[dict]:
    """Apply probes to rollouts_{tag}/{split}.npz and compute per-horizon
    absolute error vs DNS, with bootstrap CI."""
    rollout_npz = rollouts_dir / f"{split}.npz"
    if not rollout_npz.exists():
        print(f"   [eval] {split}: missing {rollout_npz}; skipping")
        return []

    blob = np.load(rollout_npz, allow_pickle=True)
    z_dns = blob["z_dns"].astype(np.float32)
    z_markov = blob["z_markov"].astype(np.float32)
    z_full = blob["z_full"].astype(np.float32)
    cid_lat = _get(blob, "case_ids", "case_id")
    ei_lat = _get(blob, "encounter_indices", "encounter_index")
    impact = _get(blob, "impact_frame").astype(np.int64)
    n_enc, T, d = z_dns.shape

    dns_idx = match_dns_to_latents(
        cid_lat, ei_lat, dns[f"{split}_case_id"], dns[f"{split}_encounter_index"]
    )
    keep_mask = dns_idx >= 0
    if not keep_mask.all():
        print(f"   [eval] {split}: {int((~keep_mask).sum())} encounters dropped (no DNS match)")

    keep_idx = np.where(keep_mask)[0]
    rng = np.random.default_rng(seed)
    out: list[dict] = []

    modes = {"z_dns": z_dns, "z_markov": z_markov, "z_full": z_full}
    for mode_name, z_arr in modes.items():
        for metric in metrics:
            probe = probes[metric]
            for H in horizons:
                errs: list[float] = []
                for i in keep_idx:
                    ti = int(impact[i])
                    t_eval = ti + H
                    if t_eval >= T:
                        continue
                    z_pred_t = z_arr[i, t_eval].reshape(1, d)
                    y_pred = float(apply_probe(z_pred_t, probe)[0])
                    y_dns = float(
                        dns[f"{split}_{metric}"][dns_idx[i], t_eval]
                    )
                    errs.append(abs(y_pred - y_dns))
                if not errs:
                    continue
                errs_arr = np.asarray(errs)
                ci_lo, ci_hi = bootstrap_ci(errs_arr, n_resamples, ci_level, rng)
                out.append(
                    {
                        "split": split,
                        "horizon": int(H),
                        "mode": mode_name,
                        "metric": metric,
                        "n_enc": int(errs_arr.size),
                        "abs_err_mean": float(errs_arr.mean()),
                        "abs_err_median": float(np.median(errs_arr)),
                        "ci_lo": ci_lo,
                        "ci_hi": ci_hi,
                    }
                )
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Physical metrics from rollouts (B1 Part d step 2)")
    p.add_argument(
        "--baselines",
        nargs="+",
        required=True,
        help=(
            "List of baseline tags. For each tag, expects "
            "outputs/session18/exp_b1/latents_{tag}/train.npz and "
            "outputs/session18/exp_b1/rollouts_{tag}/{test_b,test_c}.npz."
        ),
    )
    p.add_argument(
        "--d-per-baseline",
        nargs="+",
        type=int,
        required=True,
        help="Latent dim per baseline (same order as --baselines).",
    )
    p.add_argument(
        "--baseline-kind",
        nargs="+",
        required=True,
        choices=["fukami", "pod", "jepa"],
        help="Baseline family per tag (same order).",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=REPO / "outputs" / "session18" / "exp_b1" / "physical_closure_comparison.csv",
    )
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--ci-level", type=float, default=0.95)
    p.add_argument("--horizons", nargs="+", type=int, default=[8, 16, 32])
    p.add_argument("--splits", nargs="+", default=["test_b", "test_c"])
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.baselines) != len(args.d_per_baseline) or len(args.baselines) != len(args.baseline_kind):
        raise SystemExit("--baselines, --d-per-baseline, --baseline-kind must match length")

    dns = np.load(DNS_METRICS_PATH, allow_pickle=True)
    print(f"[physical] loaded DNS metrics: {DNS_METRICS_PATH}")

    rows: list[dict] = []
    train_r2_summary: list[dict] = []

    for tag, d, kind in zip(args.baselines, args.d_per_baseline, args.baseline_kind):
        latents_dir = REPO / "outputs" / "session18" / "exp_b1" / f"latents_{tag}"
        rollouts_dir = REPO / "outputs" / "session18" / "exp_b1" / f"rollouts_{tag}"
        if not (latents_dir / "train.npz").exists():
            print(f"[physical] skipping {tag}: {latents_dir / 'train.npz'} missing")
            continue

        print(f"[physical] === {tag} (kind={kind}, d={d}) ===")

        probe_blob = fit_probes_for_baseline(
            latents_dir, dns,
            metrics=METRICS, ridge_alpha=args.ridge_alpha,
        )
        probes = probe_blob["probes"]
        for m, r2 in probe_blob["train_r2"].items():
            print(f"   [fit] train R^2 {m}: {r2:.3f}")
            train_r2_summary.append(
                {"baseline": tag, "kind": kind, "d": d, "metric": m, "train_r2": r2}
            )

        if not rollouts_dir.exists():
            print(f"   [eval] rollouts_{tag} missing; probe fit only")
            continue

        for split in args.splits:
            split_rows = evaluate_split(
                latents_dir, rollouts_dir, split, probes, dns,
                horizons=tuple(args.horizons),
                metrics=METRICS,
                n_resamples=args.n_bootstrap,
                ci_level=args.ci_level,
                seed=args.seed,
            )
            for r in split_rows:
                r.update({"baseline": tag, "kind": kind, "d": d})
                rows.append(r)
            print(f"   [eval] {split}: produced {len(split_rows)} rows")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "baseline", "kind", "d", "split", "horizon", "mode", "metric",
                "n_enc", "abs_err_mean", "abs_err_median", "ci_lo", "ci_hi",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[physical] wrote {len(rows)} rows to {args.output_csv}")

    r2_path = args.output_csv.with_name("probe_train_r2.csv")
    with open(r2_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["baseline", "kind", "d", "metric", "train_r2"]
        )
        writer.writeheader()
        writer.writerows(train_r2_summary)
    print(f"[physical] wrote probe train R^2 to {r2_path}")


if __name__ == "__main__":
    main()
