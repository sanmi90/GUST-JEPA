"""Paired per-encounter tests + Holm for the v3 estimation endpoints (item 7).

SESSION_33_MANUSCRIPT_V3.md Section 11 item 7; HANDOFF D239 (PRE-REGISTERED
before this script ran).

Data: the frozen per-encounter records in outputs/session32/envelope_by_gust.json
(jepa_pool, frozen filter). Primary split test_b. Statistics via
scripts/session28/stats_lib.py VERBATIM (nothing reimplemented). Direction
pre-registered: delta = R2_filter_analysis - R2_baseline > 0 (filter better).

Family F1 (PRIMARY, 4 tests, Holm within family): filter vs STATIC single-frame
recovery, {C_L, E_w} x {impact, relaxation}.
Family F2 (SECONDARY, 3 tests, Holm within family): filter vs FIELD-FREE
OPEN-LOOP forecast, {C_L impact, C_L relaxation, E_w impact} (the E_w-relaxation
cell does not exist in the frozen record schema; pre-registered at 3 tests).
Annexes (descriptive, not Holm-adjusted): |G| >= 1 test_b subset; test_c.

Run (CPU):
    taskset -c 16-23 python -m scripts.session33.paired_stats_v3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

from scripts.session28.stats_lib import (  # noqa: E402
    case_cluster_bootstrap,
    case_level_paired_stats,
    case_means,
    holm,
    mixedlm_intercept,
)

# (test name, filter key, (lens, baseline key))
FAMILY_F1 = [
    ("CL_impact_vs_static", "CL_analysis_r2_impact", ("recovery", "CL_r2_impact")),
    ("CL_relax_vs_static", "CL_analysis_r2_relax", ("recovery", "CL_r2_relax")),
    ("Ew_impact_vs_static", "Ew_analysis_r2_impact", ("recovery", "Ew_r2_impact")),
    ("Ew_relax_vs_static", "Ew_analysis_r2_relax", ("recovery", "Ew_r2_relax")),
]
FAMILY_F2 = [
    ("CL_impact_vs_openloop", "CL_analysis_r2_impact", ("forecast", "fieldfree_CL_r2_impact")),
    ("CL_relax_vs_openloop", "CL_analysis_r2_relax", ("forecast", "fieldfree_CL_r2_relax")),
    ("Ew_impact_vs_openloop", "Ew_analysis_r2_impact", ("forecast", "fieldfree_Ew_r2_impact")),
]


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def run_family(records: list[dict], family, *, seed: int) -> dict:
    out = {"tests": {}, "holm": None}
    pvals = {}
    for name, fkey, (lens, bkey) in family:
        a = np.array([r["filter"][fkey] for r in records], dtype=float)
        b = np.array([r[lens][bkey] for r in records], dtype=float)
        cases = np.array([r["case_id"] for r in records])
        finite = np.isfinite(a) & np.isfinite(b)
        delta = a[finite] - b[finite]
        cid = cases[finite]
        if delta.size == 0:
            out["tests"][name] = {"n": 0, "note": "no finite pairs"}
            continue
        boot = case_cluster_bootstrap(delta, cid, rng=np.random.default_rng(seed))
        _, cm = case_means(delta, cid)
        paired = case_level_paired_stats(cm)
        mlm = mixedlm_intercept(delta, cid)
        out["tests"][name] = {
            "n_pairs": int(delta.size),
            "n_dropped_nonfinite": int((~finite).sum()),
            "bootstrap": boot,
            "case_level": paired,
            "mixedlm": mlm,
        }
        pvals[name] = paired["wilcoxon_p_one_sided"]
    usable = {k: v for k, v in pvals.items() if isinstance(v, float) and np.isfinite(v)}
    if usable:
        adj, survive = holm(usable)
        out["holm"] = {
            "p_raw": usable,
            "p_holm": adj,
            "survive_0.05": survive,
            "p_source": "case-level Wilcoxon one-sided (delta > 0 = filter better)",
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="v3 paired stats (D239 pre-registered families)")
    ap.add_argument("--envelope-json", default="outputs/session32/envelope_by_gust.json")
    ap.add_argument("--model", default="jepa_pool")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/session33/paired_stats_v3.json")
    args = ap.parse_args(argv)

    env = json.loads(_resolve(args.envelope_json).read_text())
    records = env["models"][args.model]["records"]
    test_b = [r for r in records if r["split"] == "test_b"]
    test_b_g1 = [r for r in test_b if r["abs_G"] >= 0.9]
    test_c = [r for r in records if r["split"] == "test_c"]

    payload = {
        "task": "SESSION 33 -- paired per-encounter tests + Holm (v3 endpoints)",
        "pre_registration": "HANDOFF D239 (recorded before this script ran)",
        "params": {
            "source": args.envelope_json,
            "model": args.model,
            "seed": args.seed,
            "stats": "scripts/session28/stats_lib.py verbatim",
            "direction": "delta = R2_filter_analysis - R2_baseline > 0 (one-sided)",
            "n_test_b": len(test_b),
            "n_test_b_absG_ge1": len(test_b_g1),
            "n_test_c": len(test_c),
        },
        "primary_test_b": {
            "F1_filter_vs_static": run_family(test_b, FAMILY_F1, seed=args.seed),
            "F2_filter_vs_openloop": run_family(test_b, FAMILY_F2, seed=args.seed),
        },
        "annex_test_b_absG_ge1": {
            "F1_filter_vs_static": run_family(test_b_g1, FAMILY_F1, seed=args.seed),
            "F2_filter_vs_openloop": run_family(test_b_g1, FAMILY_F2, seed=args.seed),
            "note": "descriptive (weak-signal R2 caveat, D236); not Holm-family primary",
        },
        "annex_test_c": {
            "F1_filter_vs_static": run_family(test_c, FAMILY_F1, seed=args.seed),
            "F2_filter_vs_openloop": run_family(test_c, FAMILY_F2, seed=args.seed),
            "note": "characterisation only (frozen filter; D236 precedent)",
        },
    }

    out_path = _resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=float))

    print("[stats-v3] ===== PRIMARY (test_b) =====", flush=True)
    for fam_name, fam in payload["primary_test_b"].items():
        print(f"  -- {fam_name} --", flush=True)
        for tname, t in fam["tests"].items():
            if "bootstrap" not in t:
                print(f"    {tname}: {t}", flush=True)
                continue
            b = t["bootstrap"]
            c = t["case_level"]
            h = fam["holm"]
            print(
                f"    {tname}: enc_mean={b['enc_mean']:+.3f} "
                f"CI{[round(x, 3) for x in b['enc_mean_ci']]} "
                f"case_mean={b['case_mean']:+.3f} "
                f"wilcoxon_p={c['wilcoxon_p_one_sided']} "
                f"holm={h['p_holm'].get(tname) if h else None} "
                f"survive={h['survive_0.05'].get(tname) if h else None}",
                flush=True,
            )
    print(f"[stats-v3] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
