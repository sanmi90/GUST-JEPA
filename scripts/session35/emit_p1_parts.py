"""Emit the Session 35 P1 seed-band numbers part for the macros pipeline.

Reads the P1 artifacts under outputs/session35 (plus the Session 34 s0
anchors) and writes outputs/session33/numbers_parts/p1_bands.json in the
eval_all_v3 part schema. Idempotent; missing inputs are SKIPPED with a
warning so the part can be regenerated incrementally as runs land.

Run (CPU): python -m scripts.session35.emit_p1_parts
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
S35 = REPO_ROOT / "outputs/session35"
S34 = REPO_ROOT / "outputs/session34"
OUT = REPO_ROOT / "outputs/session33/numbers_parts/p1_bands.json"


def band(vals: list[float]) -> dict:
    a = np.asarray(vals, float)
    return {
        "seed_mean": float(a.mean()),
        "seed_sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "n": int(len(a)),
    }


def load(p: Path) -> dict | None:
    if not p.exists():
        print(f"[p1-emit] SKIP missing {p.relative_to(REPO_ROOT)}")
        return None
    return json.loads(p.read_text())


def main() -> int:
    numbers: dict = {}

    # ---- T1 rexpred d32 band -------------------------------------------------
    d = load(S35 / "rexpred_d32_band.json")
    if d:
        b = d["band"]
        numbers["p1_rexpred_d32_peak"] = {
            "value": b["seed_mean"], "macro": "PoneRexpredPeak", "fmt": "%.3f",
            "seed_mean": b["seed_mean"], "seed_sd": b["seed_sd"], "n": b["n"],
            "split": "test_b", "note": "frozen linear probe, pooled peak-region R2",
        }
        numbers["p1_rexpred_d32_gate"] = {
            "value": 1.0 if b["gate_pass_mean_above_cln"] else 0.0,
            "macro": "PoneRexpredGate", "fmt": "%.0f",
            "note": f"T1 gate: band mean >= 0.862 -> {'PASS' if b['gate_pass_mean_above_cln'] else 'FAIL'}",
        }

    # ---- T4 fukami d16 band --------------------------------------------------
    d = load(S35 / "fk16_seed_band.json")
    if d:
        ib, pb = d["impact_cl_rmse_band"], d["peak_rel_error_pct_band"]
        numbers["p1_fk16_impact_rmse"] = {
            "value": ib["seed_mean"], "macro": "PoneFkSixteenImpactRmse",
            "fmt": "%.3f", "seed_mean": ib["seed_mean"], "seed_sd": ib["seed_sd"],
            "n": ib["n"], "split": "test_b",
            "note": f"own-stack DA grid protocol; verdict {d['gate']['verdict']}",
        }
        numbers["p1_fk16_peak_err"] = {
            "value": pb["seed_mean"], "macro": "PoneFkSixteenPeakErr",
            "fmt": "%.1f", "seed_mean": pb["seed_mean"], "seed_sd": pb["seed_sd"],
            "n": pb["n"], "unit": "percent", "split": "test_b",
        }

    # ---- T2 conditioning-null bands -------------------------------------------
    arms: dict[str, list[float]] = {"none": [], "phase": [], "phase_gdy": []}
    for src in (S34 / "rex2_cov.json", S35 / "rex2_cov_s1.json", S35 / "rex2_cov_s2.json"):
        d = load(src)
        if d:
            for k in arms:
                arms[k].append(d[k]["decoded_cl_r2"])
    if all(len(v) for v in arms.values()):
        for k, macro in (("none", "PoneCovNone"), ("phase", "PoneCovPhase"),
                         ("phase_gdy", "PoneCovOracle")):
            numbers[f"p1_cov_{k}"] = {
                "value": band(arms[k])["seed_mean"], "macro": macro, "fmt": "%.3f",
                **band(arms[k]), "split": "test_b",
                "note": "decoded C_L R2, latent-REX2 covariate arm",
            }
        worst_oracle = max(arms["phase_gdy"])
        best_none = min(arms["none"])
        numbers["p1_cov_oracle_hurts_all_seeds"] = {
            "value": 1.0 if all(o < n for o, n in zip(arms["phase_gdy"], arms["none"])) else 0.0,
            "macro": "PoneCovOracleHurtsAll", "fmt": "%.0f",
            "note": f"oracle<none per seed; worst oracle {worst_oracle:.3f} vs best none {best_none:.3f}",
        }

    # ---- T3 filter-seed band ---------------------------------------------------
    vals = []
    d0 = load(S34 / "rex_filter_tuned.json")
    if d0:
        vals.append(d0["aggregates"]["median_CL_r2_impact"])
    for s in (1, 2, 3, 4):
        d = load(S35 / f"rex_filter_tuned_s{s}.json")
        if d:
            vals.append(d["aggregates"]["median_CL_r2_impact"])
    if len(vals) >= 3:
        numbers["p1_rexenkf_impact"] = {
            "value": band(vals)["seed_mean"], "macro": "PoneRexEnkfImpact",
            "fmt": "%.3f", **band(vals), "split": "test_b",
            "note": "member-noise seeds, band 1.77 protocol-clean",
        }

    # ---- T3 streaming bands (protocol-clean 1.77 arm is the headline) ----------
    for tag, macro_stub, srcs in (
        ("b177", "PoneStreamClean", [(S35 / f"rex_stream_b177_noise{nz}_s{s}.json", nz, s)
                                     for nz in ("0.0", "0.05", "0.1", "0.2")
                                     for s in (0, 1, 2)]),
    ):
        by_noise: dict[str, list[float]] = {}
        for p, nz, s in srcs:
            d = load(p)
            if d:
                by_noise.setdefault(nz, []).append(
                    d["aggregates"]["median_CL_r2_impact"])
        for nz, v in by_noise.items():
            if len(v) >= 2:
                key = f"p1_stream_{tag}_n{nz.replace('.', 'p')}"
                mac = macro_stub + {"0.0": "Zero", "0.05": "Five",
                                    "0.1": "Ten", "0.2": "Twenty"}[nz]
                numbers[key] = {
                    "value": band(v)["seed_mean"], "macro": mac, "fmt": "%.3f",
                    **band(v), "split": "test_b",
                    "note": f"streaming, band 1.77, tap noise {nz}",
                }

    # ---- T6 d4 filter band ------------------------------------------------------
    vals = []
    d0 = load(S34 / "rex_filter_d4.json")
    if d0:
        vals.append(d0["aggregates"]["median_CL_r2_impact"])
    for s in (1, 2, 3, 4):
        d = load(S35 / f"rex_filter_d4_s{s}.json")
        if d:
            vals.append(d["aggregates"]["median_CL_r2_impact"])
    if len(vals) >= 3:
        numbers["p1_d4_filter_impact"] = {
            "value": band(vals)["seed_mean"], "macro": "PoneDFourFilterImpact",
            "fmt": "%.3f", **band(vals), "split": "test_b",
            "note": "jepa_pool_vec_d4, member-noise seeds, band 1.77",
        }

    # ---- lowd d4 encoder bands (P0, now banded macros) ---------------------------
    d = load(S34 / "lowd_d4_seedband.json")
    if d:
        for fam, macro in (("fukami_wake_d4", "PoneLowdFukami"),
                           ("jepa_clw_d4", "PoneLowdJepa"),
                           ("aero_lift_d4", "PoneLowdAero")):
            numbers[f"p1_lowd_{fam}"] = {
                "value": band(d[fam])["seed_mean"], "macro": macro, "fmt": "%.3f",
                **band(d[fam]), "split": "test_b",
                "note": "peak-region R2, frozen linear probe, d=4",
            }

    # ---- T5 NIS tuning + frozen two-stage -----------------------------------------
    d = load(S35 / "nis_band_tuning.json")
    if d:
        numbers["p1_nis_c_star"] = {
            "value": d["c_star"], "macro": "PoneNisCStar", "fmt": "%.2f",
            "split": "test_a", "note": "argmin |pooled impact NIS - 1| on test_a only",
        }
        numbers["p1_nis_at_c_star"] = {
            "value": d["c_star_nis"], "macro": "PoneNisAtCStar", "fmt": "%.2f",
            "split": "test_a",
        }
    d = load(S35 / "two_stage_envelope.json")
    if d:
        for arm, macro in (("rex_test_b", "PoneFrozenRexImpact"),
                           ("two_stage_test_b", "PoneFrozenTwoStageImpact")):
            a = d["arms"][arm]["aggregates"]
            numbers[f"p1_{arm}_impact"] = {
                "value": a["median_CL_r2_impact"], "macro": macro, "fmt": "%.3f",
                "split": "test_b", "note": f"frozen run at c*={d['c_star']:g}",
            }
        numbers["p1_two_stage_relax"] = {
            "value": d["arms"]["two_stage_test_b"]["aggregates"]["median_CL_r2_relax"],
            "macro": "PoneFrozenTwoStageRelax", "fmt": "%.3f", "split": "test_b",
        }
        numbers["p1_f20_gate"] = {
            "value": 1.0 if d["f20_gate"]["verdict"] == "F20-A" else 0.0,
            "macro": "PoneFTwentyGate", "fmt": "%.0f",
            "note": f"verdict {d['f20_gate']['verdict']}; anchor reproduces="
                    f"{d['anchor_check']['reproduces']}",
        }

    OUT.write_text(json.dumps({"part": "p1_bands", "numbers": numbers}, indent=1))
    print(f"[p1-emit] wrote {OUT.relative_to(REPO_ROOT)} with {len(numbers)} numbers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
