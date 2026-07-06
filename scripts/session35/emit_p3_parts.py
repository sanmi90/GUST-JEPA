"""Emit the Results A-D numbers part (Session 35 P3).

Everything the v4 Results prose cites that was not yet macro-bound:
Track C paired-gate deltas with case-clustered CIs, freshly computed E_w
probes per cell (3 encoder seeds, linear + MLP, test_b), the dimension and
SSIM ladders, the probe-dilution control, the shared-REX family bands
(now n = 3 per family), the 40-step forecast contrast, and the Part D
phase/smoother/own-stack/grid numbers at the val-calibrated band.

Writes outputs/session33/numbers_parts/p3_results.json.
Run (CPU): taskset -c 16-23 python -m scripts.session35.emit_p3_parts
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
S34 = REPO / "outputs/session34"
S35 = REPO / "outputs/session35"
OUT = REPO / "outputs/session33/numbers_parts/p3_results.json"

N: dict = {}

DWORD = {"4": "Four", "8": "Eight", "16": "Sixteen", "32": "ThirtyTwo"}


def dword(s: str) -> str:
    return DWORD[str(s)]



def add(key, value, macro, fmt="%.3f", **kw):
    N[key] = {"value": float(value), "macro": macro, "fmt": fmt, **kw}


def band(vals):
    a = np.asarray(vals, float)
    return {"seed_mean": float(a.mean()),
            "seed_sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "n": int(len(a))}


def main() -> int:
    # ---- A. paired-gate deltas (case-clustered bootstrap CIs) ----------------
    gates = json.loads((S34 / "trackc_gates.json").read_text())
    packs = {
        "CwVsCl": gates["Q2_D255"]["cw_vs_cl"],
        "ClwVsCl": gates["Q2_D255"]["clw_vs_cl"],
        "CnVsCl": gates["Q2alt_D256"]["cn_vs_cl"],
        "ClnVsCl": gates["Q2alt_D256"]["cln_vs_cl"],
        "CwnVsCw": gates["Q1_D257"]["cwn_vs_cw"],
    }
    metrics = {"peak_r2": ("Peak", "%.2f"), "ssim_nearbody": ("SsimNb", "%.3f"),
               "filter_cl_r2": ("Filt", "%.2f")}
    for cmp_name, pack in packs.items():
        for mkey, (mtag, fmt) in metrics.items():
            if mkey not in pack:
                continue
            b = pack[mkey]["boot"]
            base = f"TcDelta{cmp_name}{mtag}"
            add(f"tc_delta_{cmp_name}_{mkey}", b["case_mean"], base, fmt,
                split="test_b", note="case-mean paired delta")
            add(f"tc_delta_{cmp_name}_{mkey}_lo", b["case_mean_ci"][0],
                base + "Lo", fmt)
            add(f"tc_delta_{cmp_name}_{mkey}_hi", b["case_mean_ci"][1],
                base + "Hi", fmt)

    # ---- A. E_w probes per cell (computed here; 3 seeds, test_b) --------------
    from scripts.session34.trackc_lift_eval import group_encounters
    from src.evaluation.represent import fit_linear_probe, fit_mlp_probe

    def load_ew(run, split):
        z = np.load(S34 / f"trackc_latents/latents_{run}_{split}.npz",
                    allow_pickle=True)
        return (z["z_gap"], z["target_wake_enstrophy"].astype(np.float64))

    def r2(y, p):
        return 1.0 - ((y - p) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)

    cells = {"Clw": ["jepa_pool_vec", "jepa_pool_vec_s1", "jepa_pool_vec_s2"],
             "Cln": ["jepa_pool_ln_s0", "jepa_pool_ln_s1", "jepa_pool_ln_s2"],
             "AeLw": ["ae_wake_pool", "ae_wake_pool_s1", "ae_wake_pool_s2"]}
    for tag, runs in cells.items():
        lin, mlp = [], []
        for i, run in enumerate(runs):
            ztr, ytr = load_ew(run, "train")
            ztb, ytb = load_ew(run, "test_b")
            lin.append(r2(ytb, fit_linear_probe(ztr, ytr).predict(ztb)))
            mlp.append(r2(ytb, fit_mlp_probe(ztr, ytr, seed=i).predict(ztb)))
        for probe, vals in (("Lin", lin), ("Mlp", mlp)):
            bb = band(vals)
            add(f"ew_{tag}_{probe}".lower(), bb["seed_mean"],
                f"Ew{tag}{probe}", "%.2f", **bb, split="test_b",
                note="wake-enstrophy probe, per-frame z")
            print(f"[p3] Ew {tag} {probe}: {np.round(vals,3).tolist()}")

    # ---- B. dimension ladder + SSIM ladder + probe dilution -------------------
    lad = json.loads((S34 / "lift_dimension_ladder.json").read_text())
    for fam, tag in (("cln_rexpred", "Rexpred"), ("flagship_clw", "Clw"),
                     ("fukami_wake", "Fukami")):
        for d, blob in lad[fam].items():
            if d == "note":
                continue
            vals = blob["lin"]
            bb = band(vals)
            add(f"lad_{fam}_{d}", bb["seed_mean"], f"Lad{tag}D{dword(d[1:])}",
                "%.3f", **bb, split="test_b")
    ssim = json.loads((S34 / "cln_rexpred_ssim_ladder.json").read_text())
    for d, blob in ssim.items():
        if not d.startswith("d"):
            continue
        for mkey, mtag in (("full", "Full"), ("nearbody", "Nb")):
            v = blob[mkey]
            vals = v if isinstance(v, list) else [v]
            bb = band(vals)
            add(f"ssimlad_{mkey}_{d}", bb["seed_mean"],
                f"SsimLad{mtag}D{dword(d[1:])}", "%.3f", **bb, split="test_b")
    pd_ = json.loads((S34 / "probe_dilution_test.json").read_text())["results"]
    for d, blob in pd_.items():
        add(f"pd_mlp_d{d}", blob["mlp_test"], f"PdMlpD{dword(d)}", "%.2f",
            split="test_b")
        add(f"pd_best4_d{d}", blob["best4_lin"], f"PdBestFourD{dword(d)}", "%.2f",
            split="test_b")

    # ---- C. shared-REX family bands (n = 3 each) + 40-step contrast -----------
    fam_files = {
        "Clw": ["latent_rex_jepa_pool_vec.json", "latent_rex_jepa_pool_vec_s1.json",
                "latent_rex_jepa_pool_vec_s2.json"],
        "Cln": ["latent_rex_jepa_pool_ln_s0.json", "latent_rex_jepa_pool_ln_s1.json",
                "latent_rex_jepa_pool_ln_s2.json"],
        "AeLw": ["latent_rex_ae_wake_pool.json", "latent_rex_ae_wake_pool_s1.json",
                 "latent_rex_ae_wake_pool_s2.json"],
    }
    for tag, files in fam_files.items():
        vals = [json.loads((S34 / f).read_text())["decoded_cl_r2"] for f in files]
        bb = band(vals)
        add(f"rexfam_{tag}".lower(), bb["seed_mean"], f"RexFam{tag}", "%.3f",
            **bb, split="test_b", note="shared default-REX operator")
    fc = json.loads((S34 / "trackc_forecast.json").read_text())
    add("ar40_clw", fc["clw"]["rolled_cl_r2"], "ArFortyClw", "%.2f",
        split="test_b")
    add("ar40_cln", fc["cln"]["rolled_cl_r2"], "ArFortyCln", "%.2f",
        split="test_b")
    add("rex_tuned_cl", json.loads((S34 / "latent_rex_tuned_testb.json")
        .read_text())["decoded_cl_r2"], "RexTunedCl", "%.3f", split="test_b")

    # ---- D. phase table at band 1.77, smoother, own-stack, grid ---------------
    ph = json.loads((S35 / "da_phase_eval_b177.json").read_text())["summary"]
    for rec, rtag in (("openloop", "Ol"), ("eobs", "Eobs"),
                      ("linear_lae", "Lae"), ("rex_enkf", "Rex")):
        for phase, ptag in (("pre", "Pre"), ("impact", "Imp"), ("relax", "Rel")):
            add(f"dap_{rec}_{phase}_rmse", ph[rec][phase]["cl_rmse"],
                f"Dap{rtag}{ptag}Rmse", "%.2f", split="test_b")
            add(f"dap_{rec}_{phase}_r2", ph[rec][phase]["cl_r2"],
                f"Dap{rtag}{ptag}RTwo", "%.2f", split="test_b")
        add(f"dap_{rec}_peakerr", ph[rec]["peak_rel_error_pct_median"],
            f"Dap{rtag}PeakErr", "%.1f", unit="percent", split="test_b")
    sm = json.loads((S34 / "da_smoother.json").read_text())["summary"]
    for key, tag in (("kf", "Kf"), ("rts_lag5", "Rts"), ("rex_enkf", "RexF"),
                     ("rex_enks", "Enks")):
        if key not in sm:
            continue
        add(f"sm_{key}_imp_rmse", sm[key]["impact"]["cl_rmse"],
            f"Sm{tag}ImpRmse", "%.3f", split="test_b")
        add(f"sm_{key}_rel_rmse", sm[key]["relax"]["cl_rmse"],
            f"Sm{tag}RelRmse", "%.3f", split="test_b")
    own = {"Clw": "da_phase_eval.json", "Cln": "da_phase_jepa_pool_ln_s0.json",
           "AeLw": "da_phase_ae_wake_pool.json"}
    for tag, f in own.items():
        s = json.loads((S34 / f).read_text())["summary"]
        best = min(("rex_enkf", "linear_lae", "eobs"),
                   key=lambda r: s[r]["impact"]["cl_rmse"])
        for phase, ptag in (("pre", "Pre"), ("impact", "Imp"), ("relax", "Rel")):
            add(f"own_{tag}_{phase}".lower(), s[best][phase]["cl_rmse"],
                f"Own{tag}{ptag}Rmse", "%.2f", split="test_b",
                note=f"best recipe {best}, single filter seed")
    grid = json.loads((S34 / "da_dims_grid.json").read_text())["grid"]
    fam_tag = {"POD": "Pod", "Fukami AE": "Fk", "JEPA CLW": "Jep",
               "JEPA CLN-rex": "Rexp", "kit AE-LW": "KitAe"}
    for g in grid:
        t = fam_tag[g["family"]]
        add(f"grid_{t}_d{g['d']}".lower(), g["impact_cl_rmse"],
            f"Grid{t}D{dword(g['d'])}Rmse", "%.3f", split="test_b",
            note=f"best recipe {g['best_recipe']}")
        add(f"grid_{t}_d{g['d']}_pk".lower(), g["peak_rel_error_pct"],
            f"Grid{t}D{dword(g['d'])}Pk", "%.1f", unit="percent", split="test_b")

    OUT.write_text(json.dumps({"part": "p3_results", "numbers": N}, indent=1))
    print(f"[p3-emit] wrote {OUT.relative_to(REPO)} with {len(N)} numbers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
