"""Welch t-test on the 3 JEPA d=64 + SL seed extended_metrics vs the
production D99 number (S12_E_d64 + SL).

Reads:
  outputs/runs/session14/thrust6/jepa_d64_seed{0,1,2}/encoder/decoder_specloss_recipe/eval/extended_metrics.json
  outputs/runs/session12/S12_E_d64/encoder/decoder_specloss_recipe/eval/extended_metrics.json  (production reference)

Writes:
  outputs/session14/thrust6_welch_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from scipy import stats


REPO = Path(__file__).resolve().parents[1]
SEED_PATHS = [
    REPO / "outputs/runs/session14/thrust6" / f"jepa_d64_seed{s}/encoder/decoder_specloss_recipe/eval/extended_metrics.json"
    for s in (0, 1, 2)
]
PROD_PATH = REPO / "outputs/runs/session12/S12_E_d64/encoder/decoder_specloss_recipe/eval/extended_metrics.json"


HEADLINE_KEYS = [
    "ssim_mean_mean", "ssim_mean_median",
    "enstrophy_rel_err_wake_mean", "enstrophy_rel_err_wake_median",
    "radial_spectrum_l2_wake_mean", "radial_spectrum_l2_wake_median",
    "spectrum2d_mean_contour_iou_mean", "spectrum2d_mean_contour_iou_median",
    "spectrum2d_max_wavelength_ratio_median",
    "mse_full_mean", "mse_wake_mean",
]


def main() -> None:
    seeds = []
    for p in SEED_PATHS:
        if not p.exists():
            print(f"MISSING: {p}")
            continue
        with p.open() as f:
            seeds.append(json.load(f))
    with PROD_PATH.open() as f:
        prod = json.load(f)
    print(f"seeds loaded: {len(seeds)}/{len(SEED_PATHS)}")

    out = {"production_reference": str(PROD_PATH.relative_to(REPO)),
           "n_seeds": len(seeds), "splits": {}}

    for split in ("test_a", "test_b", "test_c"):
        rows = {}
        for key in HEADLINE_KEYS:
            seed_vals = []
            for s in seeds:
                v = s.get(split, {}).get(key)
                if v is not None and np.isfinite(v):
                    seed_vals.append(float(v))
            if len(seed_vals) < 2:
                rows[key] = {"n_seeds": len(seed_vals), "skipped": "fewer than 2 finite seeds"}
                continue
            prod_v = prod.get(split, {}).get(key)
            rows[key] = {
                "n_seeds": len(seed_vals),
                "seed_mean": float(np.mean(seed_vals)),
                "seed_std": float(np.std(seed_vals, ddof=1)),
                "seed_values": seed_vals,
                "production_value": float(prod_v) if prod_v is not None else None,
                "delta_vs_prod": float(np.mean(seed_vals) - (prod_v or 0.0)) if prod_v is not None else None,
            }
            if prod_v is not None and len(seed_vals) >= 2:
                t, p = stats.ttest_1samp(seed_vals, prod_v)
                rows[key]["one_sample_t"] = float(t)
                rows[key]["one_sample_p"] = float(p)
        out["splits"][split] = rows

    out_path = REPO / "outputs/session14/thrust6_welch_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)

    print(f"\nsaved -> {out_path}")
    print("\n=== HEADLINE WELCH t-TEST (Test B) ===")
    print(f'{"key":<45} {"seed mean":>10} {"seed std":>10} {"prod":>10} {"t":>7} {"p":>8}')
    for key, r in out["splits"]["test_b"].items():
        if "seed_mean" not in r:
            continue
        print(f'{key:<45} {r["seed_mean"]:>10.4f} {r["seed_std"]:>10.4f} '
              f'{r["production_value"] or 0:>10.4f} '
              f'{r.get("one_sample_t", float("nan")):>7.2f} '
              f'{r.get("one_sample_p", float("nan")):>8.4f}')


if __name__ == "__main__":
    main()
