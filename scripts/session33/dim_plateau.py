"""Dimension plateau {16, 32, 64} + min-d readability curves (item 2, part 1).

SESSION_33_MANUSCRIPT_V3.md Section 11 item 2 (Section 3.1 robustness).

Reads the Session 33 Q1/Q2 evals (q1_d.json, q2_d.json) plus the frozen d=32
anchors (outputs/session31/q1_ablation.json / q2_ablation.json, model
jepa_pool), and reports wake readability (windowed linear probe), matched-
predictor merit and C_L closure at h8, and decode SSIM, per latent dimension.
Case-clustered bootstrap CIs on the wake readability via the cached latents
(same r2_contribs/agg_r2 math as track_p_gates P1).

Run (CPU):
    taskset -c 16-23 python -m scripts.session33.dim_plateau
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

from scripts.session32.track_p_gates import q1_cells, q2_cells  # noqa: E402
from src.evaluation.report_session31 import agg_r2, r2_contribs  # noqa: E402

PLATEAU = (16, 32, 64)


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def _load(p):
    return json.loads(_resolve(p).read_text())


def wake_ci(cache_dir: Path, model: str, *, n_boot=10000, seed=0) -> dict:
    """Case-clustered bootstrap CI of the windowed test_b wake-enstrophy linear
    probe R2 (probe fit on train, scored on windowed test_b rows)."""
    from src.evaluation.pressure_infer import load_gap_split
    from src.evaluation.represent import fit_linear_probe

    tr = load_gap_split(cache_dir / f"latents_{model}_train.npz")
    tb = load_gap_split(cache_dir / f"latents_{model}_test_b.npz")
    m = tb["window_mask"]
    probe = fit_linear_probe(tr["z_gap"], tr["targets"]["wake_enstrophy"])
    pred = probe.predict(tb["z_gap"][m])
    y = tb["targets"]["wake_enstrophy"][m]
    contribs = r2_contribs(y, pred)
    cases = np.asarray(tb["case_id"])[m]
    uniq = np.array(sorted(set(cases.tolist())))
    rows = {c: np.where(cases == c)[0] for c in uniq}
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = uniq[rng.integers(0, len(uniq), size=len(uniq))]
        idx = np.concatenate([rows[c] for c in pick])
        sst = contribs["ss_tot"][idx].sum()
        boots[b] = 1.0 - contribs["ss_res"][idx].sum() / sst if sst > 0 else np.nan
    return {
        "r2": agg_r2(contribs),
        "ci95": [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))],
        "n_rows": int(m.sum()),
        "n_cases": int(len(uniq)),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="dimension plateau + readability-vs-d")
    ap.add_argument("--q1-d", default="outputs/session33/q1_d.json")
    ap.add_argument("--q2-d", default="outputs/session33/q2_d.json")
    ap.add_argument("--cache-d", default="outputs/session33/q1_d_latents")
    ap.add_argument("--q1-anchor", default="outputs/session31/q1_ablation.json")
    ap.add_argument("--q2-anchor", default="outputs/session31/q2_ablation.json")
    ap.add_argument("--cache-anchor", default="outputs/session31/q1_latents_ablation")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/session33/dim_plateau.json")
    args = ap.parse_args(argv)

    q1d, q2d = _load(args.q1_d), _load(args.q2_d)
    q1a, q2a = _load(args.q1_anchor), _load(args.q2_anchor)
    cache_d, cache_a = _resolve(args.cache_d), _resolve(args.cache_anchor)

    by_d = {}
    for d in PLATEAU:
        if d == 32:
            model, q1, q2, cache = "jepa_pool", q1a, q2a, cache_a
        else:
            model, q1, q2, cache = f"jepa_pool_d{d}", q1d, q2d, cache_d
        cells = {**q1_cells(q1, model)}
        try:
            cells.update(q2_cells(q2, model))
        except KeyError:
            cells["note_q2"] = "missing"
        wake = q1["models"][model]["probes"]["windowed"]["wake_enstrophy"]
        cells["wake_linear_r2"] = float(wake["linear_r2"])
        cells["wake_mlp_r2"] = float(wake["mlp_r2"])
        cells["wake_linear_ci"] = wake_ci(cache, model, n_boot=args.n_boot, seed=args.seed)
        by_d[str(d)] = {"model": model, **cells}

    r2s = {d: by_d[d]["wake_linear_r2"] for d in map(str, PLATEAU)}
    merits = {d: by_d[d].get("merit_mean_obs_h8") for d in map(str, PLATEAU)}
    spread_r2 = max(r2s.values()) - min(r2s.values())
    fin = [v for v in merits.values() if v is not None and np.isfinite(v)]
    spread_merit = (max(fin) - min(fin)) if len(fin) == len(PLATEAU) else None
    payload = {
        "task": "SESSION 33 -- pooled dimension plateau {16, 32, 64} (item 2)",
        "params": {
            "anchors": "d=32 = frozen jepa_pool (q1/q2_ablation)",
            "probe": "windowed test_b wake-enstrophy linear probe + case-clustered CI",
            "n_boot": args.n_boot,
            "seed": args.seed,
        },
        "by_d": by_d,
        "plateau": {
            "wake_linear_r2": r2s,
            "merit_mean_obs_h8": merits,
            "spread_wake_r2": spread_r2,
            "spread_merit": spread_merit,
            "flat_within_0.05_wake": bool(spread_r2 <= 0.05),
        },
    }
    out = _resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=float))
    for d in map(str, PLATEAU):
        c = by_d[d]
        print(
            f"[plateau] d={d:>2s} wake_lin={c['wake_linear_r2']:.3f} "
            f"CI{[round(x, 3) for x in c['wake_linear_ci']['ci95']]} "
            f"merit_h8={c.get('merit_mean_obs_h8', float('nan'))} "
            f"ssim={c['floor_ssim']:.3f}",
            flush=True,
        )
    flat = payload["plateau"]["flat_within_0.05_wake"]
    print(f"[plateau] wake spread={spread_r2:.3f} flat<=0.05: {flat}", flush=True)
    print(f"[plateau] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
