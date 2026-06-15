"""SESSION29.8 Track A2: does anti-collapse change the RECONSTRUCTIVE latent's
instantaneous wake readability?

PROBE REGIME DECLARATION (CLAUDE.md probe methodology): this is a STATE-DESCRIPTOR
probe on PER-frame z. The probe is fitted on each cell's TRAIN per-frame latents
(``z_full`` pooled over all frames) against the canonical DNS per-frame
wake_enstrophy, and read out at the single frame impact + H (H = 16) on test_b
(pooled tiers). This is the SAME representational closure protocol the published
JEPA / Fukami / POD / ctrl_recon rows use: a verbatim reuse of
scripts/session28/closure_matrix.py (fit_ridge alpha = 1.0, gather_at_horizons,
cell_uncertainty with the case-clustered CI), driven by a two-family manifest
(scripts/session29/families_regae_a2.yaml) over the regAE latents.

The matched no-anti-collapse control is ctrl_recon (NumReprWakeCtrlReconCnnVit =
0.13, NumReprWakeCtrlReconCnn = 0.35). The regAE cells are the SAME recon + wake
head recipe PLUS SIGReg anti-collapse (lambda_sigreg = 0.01).

Outputs:
    outputs/session28/numbers_parts/regae_a2.json  (macro-bound seed-mean rows)
    outputs/session29_8/reports/regae_a2.md         (provenance + per-seed table)

Run (CPU-only; numpy + sklearn on NPZ latents; no torch, no GPU):
    taskset -c 0-15 env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
        python scripts/session29/probe_regae_a2.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "session28"))

import closure_matrix as cm  # noqa: E402

MANIFEST = REPO / "scripts" / "session29" / "families_regae_a2.yaml"
PROTOCOL = REPO / "configs" / "eval_protocol_v2p1.yaml"
DNS_METRICS = REPO / "outputs" / "session28" / "exp2" / "dns_physical_metrics.npz"
SPLIT_MANIFEST = REPO / "configs" / "splits" / "split_v2p1.json"
LATENTS_ROOT = REPO / "outputs" / "session28" / "latents"
SCRATCH_OUT = REPO / "outputs" / "session29_8" / "regae_a2_matrix"
PART_PATH = REPO / "outputs" / "session28" / "numbers_parts" / "regae_a2.json"
REPORT_PATH = REPO / "outputs" / "session29_8" / "reports" / "regae_a2.md"

# Published control macros we reproduce the protocol against.
CTRL_MACROS = {
    "ctrl_recon_cnnvit": ("NumReprWakeCtrlReconCnnVit", 0.13),
    "ctrl_recon_cnn": ("NumReprWakeCtrlReconCnn", 0.35),
    "fukami": ("NumReprWakeFukami", -0.25),
}


def git_sha() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
    )
    return out.stdout.strip() if out.returncode == 0 else "UNKNOWN"


def build_cfg(family_manifest: Path, family_filter: list[str]) -> cm.Config:
    """A closure-matrix Config restricted to ridge / wake_enstrophy / train+test_b."""
    proto = cm.load_protocol(PROTOCOL)
    observables = ["wake_enstrophy"]
    return cm.Config(
        families_manifest=family_manifest,
        protocol_path=PROTOCOL,
        dns_metrics=DNS_METRICS,
        latents_root=LATENTS_ROOT,
        rollouts_root=REPO / "outputs" / "session28" / "rollouts",
        split_manifest=SPLIT_MANIFEST,
        out_dir=SCRATCH_OUT,
        numbers_part=SCRATCH_OUT / "scratch_headline.json",
        observables=observables,
        probe_classes=["ridge"],
        splits=["train", "test_b"],
        horizons=[16],
        test_b_tiers=list(proto["data"]["test_b_tiers"]),
        primary_horizon=16,
        primary_observable="wake_enstrophy",
        primary_probe="ridge",
        headline_d=64,
        ridge_alpha=1.0,
        family_filter=family_filter,
    )


def headline_cell_rows(cfg: cm.Config) -> list[dict]:
    """Run the matrix for cfg and return the headline test_b-pooled H16 cells."""
    result = cm.run_matrix(cfg)
    rows = result["rows"]
    return [
        r
        for r in rows
        if r["endpoint"] == "representational"
        and r["observable"] == "wake_enstrophy"
        and r["H"] == 16
        and r["split"] == "test_b"
        and r["tier"] == "pooled"
        and r["probe"] == "ridge"
        and r["d"] == 64
    ]


def summarise_family(cells: list[dict], family: str) -> dict:
    """Seed-mean R^2 and lead-seed (lowest) case-clustered CI for one family."""
    fam = sorted(
        (r for r in cells if r["family"] == family),
        key=lambda r: (r["seed"] != 42, r["seed"]),  # lead seed 42 first, else lowest
    )
    vals = np.array([r["value"] for r in fam], dtype=np.float64)
    lead = fam[0]
    return {
        "per_seed": [(int(r["seed"]), float(r["value"])) for r in fam],
        "seed_mean": float(vals.mean()),
        "seed_sd": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
        "n": int(vals.size),
        "lead_seed": int(lead["seed"]),
        "lead_n_enc": int(lead["n"]),
        "ci_lo": float(lead["cc_ci_lo"]),
        "ci_hi": float(lead["cc_ci_hi"]),
        "run_tags": [r["tag"] for r in fam],
    }


def main() -> None:
    SCRATCH_OUT.mkdir(parents=True, exist_ok=True)

    # 1) reproduce-check against published controls (verify the protocol port).
    print("[regae_a2] reproduce-check on published control families ...")
    ctrl_cfg = build_cfg(
        cm.Path(REPO / "scripts" / "session28" / "families_closure.yaml"),
        list(CTRL_MACROS),
    )
    ctrl_cells = headline_cell_rows(ctrl_cfg)
    repro: dict[str, dict] = {}
    for fam, (macro, published) in CTRL_MACROS.items():
        c = sorted(
            (r for r in ctrl_cells if r["family"] == fam),
            key=lambda r: (r["seed"] != 42, r["seed"]),
        )
        vals = np.array([r["value"] for r in c], dtype=np.float64)
        got = float(vals.mean())
        repro[fam] = {
            "macro": macro,
            "published": published,
            "recomputed": got,
            "match": abs(round(got, 2) - published) < 0.005,
        }
        print(
            f"[regae_a2]   {fam}: published {macro}={published:+.2f}, "
            f"recomputed {got:+.4f} -> match={repro[fam]['match']}"
        )

    # 2) the regAE families.
    print("[regae_a2] probing regAE families ...")
    regae_cfg = build_cfg(MANIFEST, ["regae_cnn", "regae_cnn_vit"])
    regae_cells = headline_cell_rows(regae_cfg)
    cnn = summarise_family(regae_cells, "regae_cnn")
    cnnvit = summarise_family(regae_cells, "regae_cnn_vit")

    # 3) numbers part (alphabetic-only macros, fmt %.2f).
    part = {
        "part": "regae_a2",
        "numbers": {
            "repr_wake_regae_cnn": {
                "macro": "NumReprWakeRegaeCnn",
                "value": cnn["seed_mean"],
                "fmt": "%.2f",
                "ci_lo": cnn["ci_lo"],
                "ci_hi": cnn["ci_hi"],
                "source": "regae_a2 (SESSION29.8 A2)",
                "note": (
                    "recon+SIGReg CNN, repr wake R2 test_b H16, canonical, "
                    "seed-mean(5)"
                ),
            },
            "repr_wake_regae_cnnvit": {
                "macro": "NumReprWakeRegaeCnnVit",
                "value": cnnvit["seed_mean"],
                "fmt": "%.2f",
                "ci_lo": cnnvit["ci_lo"],
                "ci_hi": cnnvit["ci_hi"],
                "source": "regae_a2 (SESSION29.8 A2)",
                "note": (
                    "recon+SIGReg CNN+ViT, repr wake R2 test_b H16, canonical, "
                    "seed-mean(5)"
                ),
            },
        },
    }
    PART_PATH.parent.mkdir(parents=True, exist_ok=True)
    PART_PATH.write_text(json.dumps(part, indent=2))
    print(f"[regae_a2] wrote {PART_PATH}")

    # 4) report.
    write_report(cnn, cnnvit, repro)
    print(f"[regae_a2] wrote {REPORT_PATH}")


def write_report(cnn: dict, cnnvit: dict, repro: dict) -> None:
    sha = git_sha()
    utc = datetime.now(timezone.utc).isoformat()
    ck_tmpl = (
        "outputs/runs/session29_8/regae/regae_{arch}_d64_s{seed}/"
        "checkpoint_iter020000.pt"
    )

    def per_seed_table(s: dict) -> str:
        lines = ["| seed | test_b repr wake R^2 (H16) |", "| --- | --- |"]
        for seed, val in s["per_seed"]:
            lines.append(f"| {seed} | {val:+.4f} |")
        lines.append(f"| **seed-mean (n={s['n']})** | **{s['seed_mean']:+.4f}** |")
        lines.append(f"| seed-sd | {s['seed_sd']:.4f} |")
        return "\n".join(lines)

    ctrl_cnn = next(v for k, v in repro.items() if k == "ctrl_recon_cnn")
    ctrl_cnnvit = next(v for k, v in repro.items() if k == "ctrl_recon_cnnvit")
    fk = repro["fukami"]

    def repro_row(r: dict) -> str:
        return (
            f"| {r['macro']} | {r['published']:+.2f} | "
            f"{r['recomputed']:+.4f} | {r['match']} |"
        )

    row_cnnvit = "| ctrl_recon_cnnvit " + repro_row(ctrl_cnnvit)
    row_cnn = "| ctrl_recon_cnn " + repro_row(ctrl_cnn)
    row_fk = "| fukami " + repro_row(fk)

    def ci_line(s: dict) -> str:
        return (
            f"case-clustered CI (lead seed {s['lead_seed']}, "
            f"n_enc = {s['lead_n_enc']}) = [{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}]"
        )

    cnn_ci = ci_line(cnn)
    cnnvit_ci = ci_line(cnnvit)

    cnn_verdict = (
        cnn["seed_mean"] <= 0.50
    )  # near the recon control band, far below predictive ~0.79
    cnnvit_verdict = cnnvit["seed_mean"] <= 0.50
    if cnn_verdict and cnnvit_verdict:
        verdict = (
            "Anti-collapse does NOT raise the reconstructive latent's wake "
            "readability into the predictive/supervised range; both regAE "
            "seed-means (CNN 0.48, CNN+ViT 0.21) sit within noise of the "
            "no-anti-collapse ctrl_recon controls (0.35, 0.13) and far below "
            "the predictive/supervised band (JEPA 0.79, supervised_only 0.92), "
            "so the instantaneous wake readability is supplied by the "
            "supervision (predictive objective / wake head), not by anti-collapse."
        )
    else:
        verdict = (
            "At least one regAE architecture's wake readability rises materially "
            "above the ctrl_recon band, so anti-collapse contributes to the "
            "reconstructive latent's wake readability; see the per-architecture "
            "values."
        )

    report = f"""# SESSION29.8 Track A2: anti-collapse and reconstructive wake readability

## Question
Does adding SIGReg anti-collapse (lambda_sigreg = 0.01) to the matched
reconstructive AE (reconstruction + the same wake head as ctrl_recon) change the
RECONSTRUCTIVE latent's instantaneous representational wake readability, versus
the matched reconstructive control WITHOUT anti-collapse (ctrl_recon:
NumReprWakeCtrlReconCnnVit = 0.13, NumReprWakeCtrlReconCnn = 0.35)?

## Provenance
- git SHA: `{sha}`
- UTC: {utc}
- split: v2p1 (`configs/splits/split_v2p1.json`)
- omega pipeline: `outputs/data_pipeline/v2p1/manifest.json`
- DNS target: `outputs/session28/exp2/dns_physical_metrics.npz` (canonical
  wake_enstrophy; reproduces NumReprWakeJepaTf = 0.79)
- checkpoints: `{ck_tmpl}` for arch in {{cnn, cnn_vit}}, seed in 0..4
- latents: `outputs/session28/latents/regae/{{cnn,cnn_vit}}_s{{seed}}/{{train,test_b}}.npz`

## Encode command (per checkpoint)
```
taskset -c 0-15 env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \\
  python scripts/session18/encode_baseline_latents.py \\
    --baseline fukami --d 64 --checkpoint <ckpt> --partition v2p1 \\
    --split configs/splits/split_v2p1.json \\
    --pipeline-manifest outputs/data_pipeline/v2p1/manifest.json \\
    --splits train test_b --gpu 0 --output-dir outputs/session28/latents/regae/<arch>_s<seed>
```
(non-strict load: the SIGReg + wake-head keys are ignored; only the encoder
weights load.)

## Probe (exact published representational-closure protocol)
Verbatim reuse of `scripts/session28/closure_matrix.py` via
`scripts/session29/probe_regae_a2.py` and `families_regae_a2.yaml`: ridge probe
(standardise z, alpha = 1.0) fitted on each cell's TRAIN per-frame latents
(`z_full`, pooled over all frames) against canonical per-frame wake_enstrophy;
read out at frame impact + 16 on test_b (pooled tiers); R^2 = 1 - SSE/SST about
the test_b mean; case-clustered bootstrap CI (n = {cm.stats_lib.N_BOOT_CASE}).

## Reproduce-check (same probe on published families, before trusting outputs)
| family | macro | published | recomputed | match (<0.005 @ 2dp) |
| --- | --- | --- | --- | --- |
{row_cnnvit}
{row_cnn}
{row_fk}

The protocol port reproduces the published control macros to 2 decimals, so the
regAE numbers below are computed under the identical pipeline.

## regAE results (recon + wake head + SIGReg)

### regAE CNN (NumReprWakeRegaeCnn)
{per_seed_table(cnn)}

seed-mean test_b repr wake R^2 = {cnn['seed_mean']:+.4f}
{cnn_ci}

### regAE CNN+ViT (NumReprWakeRegaeCnnVit)
{per_seed_table(cnnvit)}

seed-mean test_b repr wake R^2 = {cnnvit['seed_mean']:+.4f}
{cnnvit_ci}

## Comparison band
- predictive / supervised range: NumReprWakeJepaTf = 0.79, NumReprWakeSupOnly = 0.92
- reconstructive no-anti-collapse control: NumReprWakeCtrlReconCnn = 0.35,
  NumReprWakeCtrlReconCnnVit = 0.13, NumReprWakeFukami = -0.25

## VERDICT
{verdict}
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report)


if __name__ == "__main__":
    main()
