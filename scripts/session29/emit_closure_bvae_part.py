"""Emit the beta-VAE row of the six-observable closure table (Phase D, Session 29).

The Phase-D closure part (scripts/session28/emit_closure_floor_part.py) bound the
per-observable closure (representational and forecast) for jepa_tf_noc, fukami,
and pod, but NOT for the beta-VAE Solera-Rico baseline. This adds that row so a
beta-VAE line can sit alongside the predictive / reconstructive / linear rows in
tab:closure (paper/sections/tables/results_tables.tex).

Protocol (IDENTICAL to emit_closure_floor_part.py): the values are read from the
committed closure matrix (outputs/session28/closure_matrix/matrix.csv, produced
by closure_matrix.py against the canonical DNS physical metrics, fixed linear
ridge probe fitted on the family's TRAIN per-frame latents, case-clustered 5-fold
readout), as the seed-mean of every cell at horizon H=16, split test_b, tier
pooled, latent d=64, probe ridge. No re-encoding, no rollout recomputation: the
matrix cells already exist for bvae_match (representational) and bvae_d64
(forecast).

Recipe split (documented, NOT a bug; it mirrors the existing paper's own usage):
    representational -> bvae_match_d64_s{42,0,1}  (the MATCHED Solera-Rico head;
        wake-repr seed-mean = NumReprWakeBvaeMatch = 0.53, which this row MUST
        cite for its wake-repr cell rather than minting a duplicate macro;
        sensing_cf.py / closure_headline.json also use bvae_match for the plain
        "Bvae" cell).
    forecast         -> bvae_d64, the ONLY beta-VAE matched-predictor rollout on
        disk, trained and rolled out from bvae_faith_d64_s42 (drift_ce1.py
        PRED_LAT, families_closure.yaml bvae_faith.rollouts). No matched-recipe
        predictor was trained; bvae_match carries no rollouts entry in the
        families manifest. Forecast is therefore single-seed (s42).

Before trusting the beta-VAE outputs the script re-derives published comparator
cells from the SAME matrix.csv with the SAME filter and asserts they round to the
macros already in paper/macros.tex (NumReprWakeFukami = -0.25, the JEPA / Fukami /
POD closure cells, NumReprWakeBvaeMatch = 0.53). CPU, idempotent.

    python scripts/session29/emit_closure_bvae_part.py
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MATRIX = REPO / "outputs/session28/closure_matrix/matrix.csv"
OUT = REPO / "outputs/session28/numbers_parts/closure_bvae.json"

# observable -> macro stem (matches emit_closure_floor_part.py)
OBS = {
    "C_L": "CL",
    "C_D": "CD",
    "I_y": "Iy",
    "wake_enstrophy": "Wake",
    "circulation_pos": "CircPos",
    "circulation_neg": "CircNeg",
}


def seed_mean(rows, family, observable, endpoint):
    """Seed-mean of the closure cell at H=16, test_b pooled, d=64, ridge.

    Verbatim filter from emit_closure_floor_part.py so the beta-VAE row uses the
    exact protocol the jepa_tf_noc / fukami / pod rows use.
    """
    vals = [
        float(r["value"])
        for r in rows
        if r["family"] == family
        and r["observable"] == observable
        and r["endpoint"] == endpoint
        and r["split"] == "test_b"
        and r["tier"] == "pooled"
        and r["H"] == "16"
        and r["probe"] == "ridge"
        and r["d"] == "64"
    ]
    return statistics.mean(vals) if vals else None


def reproduce_checks(rows) -> list[str]:
    """Re-derive published comparator cells; fail loudly on any mismatch."""
    checks = [
        # (family, observable, endpoint, published rounded value, macro)
        ("fukami", "wake_enstrophy", "representational", -0.25, "NumReprWakeFukami"),
        ("bvae_match", "wake_enstrophy", "representational", 0.53, "NumReprWakeBvaeMatch"),
        ("jepa_tf_noc", "C_L", "representational", 0.86, "NumCloReprCLJepaTf"),
        ("fukami", "C_L", "representational", 0.56, "NumCloReprCLFukami"),
        ("pod", "C_D", "representational", 0.64, "NumCloReprCDPod"),
        ("jepa_tf_noc", "wake_enstrophy", "forecast", 0.43, "NumCloFcstWakeJepaTf"),
        ("pod", "wake_enstrophy", "forecast", -0.60, "NumCloFcstWakePod"),
    ]
    log: list[str] = []
    for family, obs, endpoint, published, macro in checks:
        v = seed_mean(rows, family, obs, endpoint)
        if v is None:
            raise SystemExit(f"reproduce-check: no cell for {family}/{obs}/{endpoint}")
        got = round(v, 2)
        ok = got == published
        verdict = "OK" if ok else "MISMATCH"
        log.append(f"{macro}: recomputed {got:+.2f} vs published {published:+.2f} -> {verdict}")
        if not ok:
            raise SystemExit(f"reproduce-check FAILED for {macro}: {got} != {published}")
    return log


def main() -> None:
    rows = list(csv.DictReader(open(MATRIX)))

    print("[closure-bvae] reproduce-checks against published macros:")
    for line in reproduce_checks(rows):
        print(f"  {line}")

    numbers = {}

    # Representational: matched recipe (bvae_match), 3 seeds. Wake column is
    # DELIBERATELY omitted: the table cites NumReprWakeBvaeMatch (0.53) for the
    # beta-VAE wake-repr cell; minting NumCloReprWakeBvae would duplicate it.
    six_repr = []
    for obs, ostem in OBS.items():
        v = seed_mean(rows, "bvae_match", obs, "representational")
        if v is None:
            raise SystemExit(f"missing bvae_match representational cell for {obs}")
        six_repr.append(v)  # all six enter the mean-over-six
        if obs == "wake_enstrophy":
            continue  # wake-repr cell reuses NumReprWakeBvaeMatch
        numbers[f"clo_repr_{ostem.lower()}_bvae"] = {
            "macro": f"NumCloRepr{ostem}Bvae",
            "value": round(v, 2),
            "fmt": "%.2f",
            "split": "test_b",
            "endpoint": "representational",
            "probe": "ridge",
            "observable": obs,
            "horizon": 16,
            "run_tags": ["bvae_match_d64_s42", "bvae_match_d64_s0", "bvae_match_d64_s1"],
            "source": "emit_closure_bvae_part.py (matrix.csv seed-mean, matched recipe)",
        }

    # Mean over the six representational observables (matched recipe), to mirror
    # NumCloReprMeanSixJepaTf. Includes the matched wake-repr value in the mean.
    numbers["clo_repr_meansix_bvae"] = {
        "macro": "NumCloReprMeanSixBvae",
        "value": round(statistics.mean(six_repr), 2),
        "fmt": "%.2f",
        "split": "test_b",
        "endpoint": "representational",
        "probe": "ridge",
        "horizon": 16,
        "source": "emit_closure_bvae_part.py (mean over six matched-recipe repr cells)",
        "note": "includes the matched wake-repr value (NumReprWakeBvaeMatch) in the six-mean",
    }

    # Forecast: the only beta-VAE matched-predictor rollout (bvae_d64), trained
    # and rolled out from bvae_faith_d64_s42 (single seed; no matched-recipe
    # predictor exists). Full six observables incl. wake.
    for obs, ostem in OBS.items():
        v = seed_mean(rows, "bvae_faith", obs, "forecast")
        if v is None:
            raise SystemExit(f"missing bvae forecast cell for {obs}")
        numbers[f"clo_fcst_{ostem.lower()}_bvae"] = {
            "macro": f"NumCloFcst{ostem}Bvae",
            "value": round(v, 2),
            "fmt": "%.2f",
            "split": "test_b",
            "endpoint": "forecast",
            "probe": "ridge",
            "observable": obs,
            "horizon": 16,
            "run_tags": ["bvae_d64"],
            "source": (
                "emit_closure_bvae_part.py (matrix.csv; bvae_d64 rollout "
                "from bvae_faith_d64_s42)"
            ),
        }

    part = {"part": "closure_bvae", "numbers": numbers}
    OUT.write_text(json.dumps(part, indent=2))
    print(f"[closure-bvae] wrote {len(numbers)} numbers -> {OUT}")
    for name, rec in numbers.items():
        print(f"  {rec['macro']:24s} = {rec['value']:+.2f}")


if __name__ == "__main__":
    main()
