"""Minimum-dimension panel: smallest d reaching wake R2 >= 0.5 (item 2, part 2).

SESSION_33_MANUSCRIPT_V3.md Section 11 item 2 / Section 4.6 (the d = 32 defence).

Per family, the windowed test_b wake-enstrophy LINEAR probe R2 (the Table X
convention) as a function of latent dimension:
  jepa_pool     : d in {4, 8, 16} from q1_d.json + d=32 anchor (q1_ablation).
  fukami_wake   : d in {4, 8, 16} from q1_d.json + d=32 anchor (q1_reference).
  POD truncation: d in {1, 2, 4, 8, 16, 32} computed here by truncating the
                  cached POD coefficients to the leading d columns (the basis
                  is singular-value ordered; truncation is exact, no retrain).

Run (CPU):
    taskset -c 16-23 python -m scripts.session33.min_d_panel
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

THRESH = 0.5
POD_DS = (1, 2, 4, 8, 16, 32)
NN_DS = (4, 8, 16, 32)


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def _load(p):
    return json.loads(_resolve(p).read_text())


def wake_lin(q1: dict, model: str) -> float:
    return float(q1["models"][model]["probes"]["windowed"]["wake_enstrophy"]["linear_r2"])


def pod_truncation_curve(cache_dir: Path) -> dict:
    from src.evaluation.pressure_infer import load_gap_split
    from src.evaluation.represent import fit_linear_probe, r2_score_np

    tr = load_gap_split(cache_dir / "latents_pod_train.npz")
    tb = load_gap_split(cache_dir / "latents_pod_test_b.npz")
    m = tb["window_mask"]
    y_tr = tr["targets"]["wake_enstrophy"]
    y_tb = tb["targets"]["wake_enstrophy"][m]
    out = {}
    for d in POD_DS:
        probe = fit_linear_probe(tr["z_gap"][:, :d], y_tr)
        out[str(d)] = float(r2_score_np(y_tb, probe.predict(tb["z_gap"][m][:, :d])))
    return out


def smallest_d(curve: dict, thresh: float):
    for d in sorted(int(k) for k in curve.keys()):
        v = curve[str(d)]
        if v is not None and np.isfinite(v) and v >= thresh:
            return d
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description="min-d wake readability panel")
    ap.add_argument("--q1-d", default="outputs/session33/q1_d.json")
    ap.add_argument("--q1-anchor", default="outputs/session31/q1_ablation.json")
    ap.add_argument("--q1-reference", default="outputs/session31/q1_reference.json")
    ap.add_argument("--pod-cache", default="outputs/session31/q1_latents")
    ap.add_argument("--jepa-prefix", default="jepa_pool",
                    help="d != 32 model-name prefix (D250: jepa_pool_vec).")
    ap.add_argument("--anchor-model", default="jepa_pool",
                    help="d = 32 anchor model name (D250: jepa_pool_vec).")
    ap.add_argument("--q1-d-ref", default=None,
                    help="fukami_wake d-sweep Q1 (reconstructive baseline, no vec "
                         "variant); defaults to --q1-d.")
    ap.add_argument("--out", default="outputs/session33/min_d_panel.json")
    args = ap.parse_args(argv)

    q1d = _load(args.q1_d)
    q1dref = _load(args.q1_d_ref) if args.q1_d_ref else q1d
    q1a = _load(args.q1_anchor)
    q1r = _load(args.q1_reference)

    curves = {
        "jepa_pool": {
            **{str(d): wake_lin(q1d, f"{args.jepa_prefix}_d{d}") for d in NN_DS if d != 32},
            "32": wake_lin(q1a, args.anchor_model),
        },
        "fukami_wake": {
            **{str(d): wake_lin(q1dref, f"fukami_wake_d{d}") for d in NN_DS if d != 32},
            "32": wake_lin(q1r, "fukami_wake"),
        },
        "pod_truncated": pod_truncation_curve(_resolve(args.pod_cache)),
    }
    panel = {
        fam: {"curve": c, "smallest_d_r2_ge_0.5": smallest_d(c, THRESH)}
        for fam, c in curves.items()
    }
    payload = {
        "task": "SESSION 33 -- minimum-dimension wake-readability panel (item 2)",
        "params": {
            "probe": "windowed test_b wake-enstrophy LINEAR probe (Table X convention)",
            "threshold": THRESH,
            "pod": "exact truncation of the singular-value-ordered coefficients",
        },
        "panel": panel,
    }
    out = _resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=float))
    for fam, p in panel.items():
        c = {k: round(v, 3) for k, v in sorted(p["curve"].items(), key=lambda kv: int(kv[0]))}
        print(f"[min-d] {fam:14s} curve={c} smallest_d>=0.5: {p['smallest_d_r2_ge_0.5']}",
              flush=True)
    print(f"[min-d] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
