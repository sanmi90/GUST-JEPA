"""Assemble the three-family x dimension DA grid (Session 34 closing deliverable).

Reads the phase-resolved DA evaluations produced by scripts/session34/da_phase_eval.py
(one JSON per model, all with the same four recipes: rex_enkf, linear_lae, eobs,
openloop) and assembles the POD vs Fukami vs JEPA dimension grid requested by Carlos
("Before finish we should do the data assimilation study for all dimensions", "Add also
POD so we can have POD vs Fukami vs JEPA").

Every cell is an OWN-STACK evaluation: the model's own OSP tap staircase (W=30 TCSI),
its own delay-embedded observation encoder, its own latent-REX forecast operator, and
its own decode-floor decoder. test_b, K=8 taps, every-frame pressure, no noise.

Output: outputs/session34/da_dims_grid.json + a markdown table on stdout.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "outputs/session34"

# (family, d, label, json path) -- order defines table rows
ROWS = [
    ("POD", 4, "pod_d4", "da_phase_dim_pod_d4.json"),
    ("POD", 8, "pod_d8", "da_phase_dim_pod_d8.json"),
    ("POD", 16, "pod_d16", "da_phase_dim_pod_d16.json"),
    ("POD", 32, "pod_d32", "da_phase_dim_pod_d32.json"),
    ("Fukami AE", 4, "fukami_wake_d4", "da_phase_dim_fukami_wake_d4.json"),
    ("Fukami AE", 8, "fukami_wake_d8", "da_phase_dim_fukami_wake_d8.json"),
    ("Fukami AE", 16, "fukami_wake_d16", "da_phase_dim_fukami_wake_d16.json"),
    ("Fukami AE", 32, "fukami_wake_d32", "da_phase_dim_fukami_wake.json"),
    ("JEPA CLW", 4, "jepa_pool_vec_d4", "da_phase_dim_jepa_pool_vec_d4.json"),
    ("JEPA CLW", 8, "jepa_pool_vec_d8", "da_phase_dim_jepa_pool_vec_d8.json"),
    ("JEPA CLW", 16, "jepa_pool_vec_d16", "da_phase_dim_jepa_pool_vec_d16.json"),
    ("JEPA CLW", 32, "jepa_pool_vec", "da_phase_dim_jepa_pool_vec_d32.json"),
    ("JEPA CLN-rex", 4, "cln_rexpred_d4_s0", "da_phase_dim_cln_rexpred_d4_s0.json"),
    ("JEPA CLN-rex", 32, "jepa_pool_ln_rexpred_s0", "da_phase_dim_jepa_pool_ln_rexpred_s0.json"),
    ("kit AE-LW", 32, "ae_wake_pool", "da_phase_dim_ae_wake_pool_d32.json"),
]

RECIPES = ("rex_enkf", "linear_lae", "eobs")


def best_recipe(summary: dict) -> tuple[str, dict]:
    """Best of the three assimilating recipes by impact-phase C_L RMSE."""
    key = min(RECIPES, key=lambda r: summary[r]["impact"]["cl_rmse"])
    return key, summary[key]


def main() -> None:
    grid = []
    for family, d, label, fname in ROWS:
        p = OUT_DIR / fname
        if not p.exists():
            print(f"[grid] MISSING {fname} -- skipped")
            continue
        summary = json.loads(p.read_text())["summary"]
        rec_name, rec = best_recipe(summary)
        grid.append(
            {
                "family": family,
                "d": d,
                "model": label,
                "best_recipe": rec_name,
                "impact_cl_rmse": rec["impact"]["cl_rmse"],
                "relax_cl_rmse": rec["relax"]["cl_rmse"],
                "pre_cl_rmse": rec["pre"]["cl_rmse"],
                "impact_cl_r2": rec["impact"]["cl_r2"],
                "peak_rel_error_pct": rec["peak_rel_error_pct_median"],
                "peak_timing_error_tc": rec["peak_abs_timing_error_tc_median"],
                "impact_ssim_nearbody": rec["impact"]["ssim_nearbody"],
                "impact_ssim_full": rec["impact"]["ssim_full"],
                "relax_ssim_full": rec["relax"]["ssim_full"],
                "per_recipe": {
                    r: {
                        "impact_cl_rmse": summary[r]["impact"]["cl_rmse"],
                        "peak_rel_error_pct": summary[r]["peak_rel_error_pct_median"],
                    }
                    for r in RECIPES
                },
                "source": fname,
            }
        )
    out = OUT_DIR / "da_dims_grid.json"
    out.write_text(
        json.dumps(
            {
                "protocol": {
                    "split": "test_b",
                    "taps": "own OSP staircase K=8, W=30",
                    "stack": "own E_obs + own latent-REX + own decode-floor decoder",
                    "best_recipe_criterion": "min impact-phase C_L RMSE over "
                    + "/".join(RECIPES),
                    "obs": "every-frame wall pressure, no added noise",
                },
                "grid": grid,
            },
            indent=2,
        )
    )
    print(f"[grid] wrote {out} ({len(grid)} rows)\n")

    hdr = (
        "| family | d | recipe | impact RMSE | relax RMSE | peak err % | "
        "SSIM nb (imp) | SSIM full (imp) |"
    )
    print(hdr)
    print("|" + "---|" * 8)
    for g in grid:
        print(
            f"| {g['family']} | {g['d']} | {g['best_recipe']} | "
            f"{g['impact_cl_rmse']:.3f} | {g['relax_cl_rmse']:.3f} | "
            f"{g['peak_rel_error_pct']:.1f} | {g['impact_ssim_nearbody']:.3f} | "
            f"{g['impact_ssim_full']:.3f} |"
        )


if __name__ == "__main__":
    main()
