"""Pick the T8 beta-VAE beta from the L-curve sweep and pin it (author decision 2026-06-11).

Convention: CANONICAL beta-VAE objective (KL summed over latent dims, averaged
over batch x time; see src/baselines/solera_rico.py). The released KTH-FlowAI
code's mean-over-dims KL is a known typo (author-confirmed), so the published
production beta = 0.05 (mean convention, d = 20) maps to 2.5e-3 canonical; the
sweep brackets that value on OUR data at d = 64 instead of trusting the
transfer (referee M2 hardening, mirrors the T7 Fukami elbow verification).

Rate-distortion knee rule: for each sweep run, rate R = median of the last 5
train L_kl values, distortion D = median of the last 3 held-out
diag/L_recon_test_b values (both from metrics.jsonl). Normalise (R, D) to
[0, 1] over the sweep, draw the chord from the smallest-beta point to the
largest-beta point, pick the point of maximum perpendicular distance to the
chord; ties break toward the LARGER beta (more regularisation).

Outputs:
    outputs/session28/bvae_lcurve.json    full sweep table + rule + provenance
    outputs/session28/bvae_beta_pin.json  {"beta": ..., ...} read by _run_one.sh

Usage:
    python scripts/session28/pick_bvae_beta.py [--runs-root outputs/runs/session28]
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

SWEEP_CELLS = {
    "bvae_lcurve_b5em4": 5e-4,
    "bvae_lcurve_b1em3": 1e-3,
    "bvae_lcurve_b2p5em3": 2.5e-3,
    "bvae_lcurve_b5em3": 5e-3,
    "bvae_lcurve_b1em2": 1e-2,
}


def knee_index(rate: np.ndarray, distortion: np.ndarray) -> int:
    """Max-chord-distance knee of a rate-distortion curve, ordered by beta.

    Points must be ordered by ascending beta. Both axes are min-max
    normalised; the chord runs from the first to the last point; ties in
    perpendicular distance break toward the larger beta (later index).
    """
    r = np.asarray(rate, dtype=float)
    d = np.asarray(distortion, dtype=float)
    if r.size != d.size or r.size < 3:
        raise ValueError("need >= 3 sweep points with matching shapes")

    def norm(v: np.ndarray) -> np.ndarray:
        span = v.max() - v.min()
        return np.zeros_like(v) if span == 0 else (v - v.min()) / span

    x, y = norm(r), norm(d)
    p0 = np.array([x[0], y[0]])
    p1 = np.array([x[-1], y[-1]])
    chord = p1 - p0
    n = np.linalg.norm(chord)
    if n == 0:
        return r.size - 1
    # perpendicular distance of each point to the chord
    rel = np.stack([x, y], axis=1) - p0
    dist = np.abs(rel[:, 0] * chord[1] - rel[:, 1] * chord[0]) / n
    best = 0
    for i in range(r.size):
        if dist[i] >= dist[best]:  # >= : ties go to the larger beta
            best = i
    return best


def harvest_run(run_dir: Path) -> dict | None:
    """Pull (rate, distortion) for one sweep run from its metrics.jsonl."""
    mp = run_dir / "metrics.jsonl"
    if not mp.exists():
        return None
    kl, recon_tb = [], []
    with open(mp) as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event") != "log":
                continue
            if "L_kl" in rec:
                kl.append((rec["step"], float(rec["L_kl"])))
            if "diag/L_recon_test_b" in rec:
                recon_tb.append((rec["step"], float(rec["diag/L_recon_test_b"])))
    if len(kl) < 5 or len(recon_tb) < 3:
        return None
    kl.sort(key=lambda t: t[0])
    recon_tb.sort(key=lambda t: t[0])
    return {
        "rate_kl": float(np.median([v for _, v in kl[-5:]])),
        "distortion_recon_test_b": float(np.median([v for _, v in recon_tb[-3:]])),
        "n_kl_points": len(kl),
        "n_recon_points": len(recon_tb),
        "last_step": int(max(kl[-1][0], recon_tb[-1][0])),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", default="outputs/runs/session28")
    p.add_argument(
        "--warmup-frac",
        type=float,
        default=0.02,
        help="Beta warmup fraction pinned alongside beta (released-recipe shape).",
    )
    args = p.parse_args()

    rows = []
    missing = []
    for tag, beta in sorted(SWEEP_CELLS.items(), key=lambda kv: kv[1]):
        rec = harvest_run(REPO / args.runs_root / tag)
        if rec is None:
            missing.append(tag)
            continue
        rows.append({"tag": tag, "beta": beta, **rec})
    if missing:
        raise SystemExit(
            f"[pick-bvae-beta] FATAL: incomplete sweep, missing/short runs: {missing}. "
            "Run the bvae_lcurve_* cells first (launch_queue.sh GPU-1 tail)."
        )

    rate = np.array([r["rate_kl"] for r in rows])
    dist = np.array([r["distortion_recon_test_b"] for r in rows])
    k = knee_index(rate, dist)
    pick = rows[k]

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO
    ).stdout.strip()
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()

    out_dir = REPO / "outputs/session28"
    out_dir.mkdir(parents=True, exist_ok=True)
    lcurve = {
        "_doc": "T8 beta-VAE L-curve sweep (canonical sum-KL convention). "
        "Knee rule: max perpendicular distance to the normalised "
        "rate-distortion chord, ties toward larger beta.",
        "rows": rows,
        "knee_tag": pick["tag"],
        "knee_beta": pick["beta"],
        "git_commit": commit,
        "generated": now,
    }
    (out_dir / "bvae_lcurve.json").write_text(json.dumps(lcurve, indent=2))

    pin = {
        "beta": pick["beta"],
        "beta_warmup_frac": args.warmup_frac,
        "convention": "canonical sum-KL (Higgins 2017 / Nat. Commun. Eq. (4)); "
        "released KTH code mean-KL is a known typo (author-confirmed)",
        "picked_by": "L-curve knee, scripts/session28/pick_bvae_beta.py",
        "candidates": sorted(SWEEP_CELLS.values()),
        "git_commit": commit,
        "generated": now,
    }
    (out_dir / "bvae_beta_pin.json").write_text(json.dumps(pin, indent=2))

    print("[pick-bvae-beta] sweep table:")
    for r in rows:
        mark = "  <-- KNEE" if r["tag"] == pick["tag"] else ""
        print(
            f"  beta={r['beta']:.1e}  KL={r['rate_kl']:.4f}  "
            f"recon_test_b={r['distortion_recon_test_b']:.5f}{mark}"
        )
    print(
        f"[pick-bvae-beta] PINNED beta = {pick['beta']:.1e} " f"-> {out_dir / 'bvae_beta_pin.json'}"
    )
    print(
        "[pick-bvae-beta] review outputs/session28/bvae_lcurve.json before "
        "trusting downstream tables (idempotent: re-run after any fix)."
    )


if __name__ == "__main__":
    main()
