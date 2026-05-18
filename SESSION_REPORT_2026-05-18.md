# Session report — 2026-05-18 (Session 5)

Session 5 ran from the evening of 2026-05-17 through the morning of
2026-05-18, straddling the date boundary. The Session 4 report
(`SESSION_REPORT_2026-05-17.md`) covers the prior session; this report
uses the date the work landed to disambiguate.

Scope: Session 5 meaningful 5k-iter smoke run on 5 training cases. The session
is methodological: it answers "does SIGReg work on this low-intrinsic-dim
physics data, in any reasonable configuration, on a small case subset?" The
deliverable is a decision string (HEALTHY / PARTIAL / TRIVIAL / WEAK / DEAD)
plus the corresponding HANDOFF D-entries; the numerical pass criteria are
operational targets, not session acceptance criteria.

Two housekeeping items also landed this session: a citation correction for
PLDM (D32) and a third v1 absorption of overnight run3 files (D33).

## Starting state

| Item | State |
|---|---|
| Last commit on `origin/main` | `58155eb` (Session 4 report) |
| Local `main` | 0 commits ahead |
| `src/models/` | encoder.py, predictor.py, jepa.py, sigreg.py, vicreg.py, adaln.py, rope.py |
| `src/training/` | scheduled_sampling.py, diagnostics.py, auto_fallback.py, train_jepa.py |
| `tests/` | 71 tests passing, 1 slow integration test gated by `--runslow` |
| Partition v1 | 47 cases / 230 encounters; split SHA256 `6fa9fd14…` |
| HANDOFF.md decision log | last entry was D23 |
| Untracked planning docs | `SESSION5_MEANINGFUL_SMOKE_5K.md`, `SESSION5_PLDM_BASELINE.md` |

## What ran end-to-end

### Housekeeping

1. **PLDM citation correction (D32)** — `HANDOFF.md` D8 originally cited PLDM
   as Sobal, Jyothir, Jalagam, Carion, Cho, LeCun (2022), arXiv:2211.10831
   "Joint Embedding Predictive Architectures Focus on Slow Features". This is
   the 4-page NeurIPS SSL workshop precursor, not the actual PLDM paper. The
   primary PLDM reference is Sobal, Zhang, Cho, Balestriero, Rudner, LeCun
   (February 2025), arXiv:2502.14819, "Learning from Reward-Free Offline Data:
   A Case for Planning with Latent Dynamics Models". Updates landed in three
   places: D8 header now carries "(citation corrected 2026-05-17, see D32)"
   and the inline citation; HANDOFF Key references / Direct baselines section
   re-orders PLDM, PLDM precursor (workshop), PLDM (stress-tested); CLAUDE.md
   "Baselines to implement" item 4. The "7-term loss" language in D8 is
   approximate and is deferred for verification against arXiv:2502.14819
   Appendix C.1.1 if and when Session 5.PLDM triggers.

2. **D33: absorb Gust_027 and Gust_031 into v1** — two overnight run3 files
   (`Gust_027_x-1.965_y-0.387_s-2.0_d1.5.h5` and
   `Gust_031_x-1.844_y-0.872_s-3.0_d0.5.h5`, both timestamped 2026-05-17 21:17).
   Decoded with the alpha=14 degree rotation:
   `G-2.00_D1.50_Y+0.10` and `G-3.00_D0.50_Y-0.40`, both `run3`, both
   default to `train`. Same precedent as D12/D14/D15/D20: v1 still has no
   paper-reportable training checkpoint, so absorption stays in v1.
   Cumulative effect since D20:
   - Train cases: 37 -> 39 (+2 run3).
   - Train encounters: 126 -> 132 (+6).
   - Test A encounters: 52 -> 54 (+2).
   - Total cases: 47 -> 49.
   - Total encounters: 230 -> 238.
   New inventory SHA256:
   `dd984588be553a28285a35fed7328cfcf9b482329e6f346b4f1e9a0574f764bc`;
   new split SHA256:
   `7f8f60428e13b7c2fe4063e15bd99ea9e08e5e6cecf0e8883f8fb6a4875e2331`.
   The Session 5 5-case smoke subset (D24) is unaffected; the new cases will
   be available to Session 6 lambda bisection.

### Session 5 infrastructure

3. **TDD: `tests/test_sanity_checks.py` (6 tests)** — three happy-path checks
   plus three failure-mode checks for `check_1`, `check_2`, `check_3`. Fast
   suite went from 71 passing to 77 passing (6 added, plus 2 from the
   encoder modification below).

4. **`src/training/sanity_checks.py` (5 checks)** — Session 5 pre-variant gates:

   - check_1: projection BatchNorm running stats are healthy after warmup
     (finite, |mean| < 10, var in `(1e-4, 100)`). The deeper "case label
     leakage through running stats" concern from HANDOFF "Warnings and
     pitfalls" is theoretical and not testable here; left to the linear
     probe diagnostic during training.
   - check_2: predictor is identity-on-residual at init (L_pred in
     `[0.01, 10]`); after one Adam step at least one AdaLN.linear.weight
     has moved off zero; after a few overfitting steps on one batch
     L_pred has decreased.
   - check_3: `SIGReg(z).backward()` produces a finite, nonzero gradient
     on the encoder's projection BatchNorm bias.
   - check_4: `predictor.rollout(steps=1)` at index 1 equals teacher-forced
     prediction at index 0 within atol 1e-4.
   - check_5: data loader emits `omega.shape == (16, 32, 1, 192, 96)`,
     `c.shape == (16, 3)`, omega is finite, `|omega|.max() < 10000`.
     (Plan's original `|omega| < 100` bound was wrong: DNS vorticity at
     Re=5000 peaks at ~4000 in vortex cores; survey of all 49 cases gave
     median |omega|.max() = 1482. The 10000 ceiling catches Inf without
     false-failing legitimate intense vorticity events.)

   Script entrypoint: `python -m src.training.sanity_checks --all
   --require-gpu`. All five checks PASS on the 5-case smoke subset.

5. **`HybridCNNViTEncoder.projection_norm` constructor arg (D25)** —
   default `"batchnorm"`; `"layernorm"` swaps `nn.LayerNorm(latent_dim)`
   in at `proj[-1]`. Test `test_encoder_projection_is_batchnorm` renamed to
   `test_encoder_projection_is_batchnorm_by_default`; new
   `test_encoder_projection_can_be_layernorm` and
   `test_encoder_projection_norm_rejects_unknown` added.

6. **`train_jepa.py` flags (D25, D26)** — `--projection-norm`,
   `--anticollapse`, `--cases-from`, `--tag-suffix`. Mutual-exclusion
   between `--cases` and `--cases-from` enforced in `resolve_cases`.
   W&B tag list becomes `['hybrid_cnn_vit', '{sigreg,vicreg}']` per
   `--anticollapse`, plus `run:<suffix>` when `--tag-suffix` is set.
   New keys logged to W&B config: `projection_norm`, `anticollapse`,
   `tag_suffix`. The `auto_fallback_triggered` field stays in summary
   logging but the SIGReg -> VICReg auto-fallback path is silently
   skipped when `--anticollapse vicreg` is selected.

7. **JSONL side-logger** — `train_jepa.py` now writes
   `<output_dir>/metrics.jsonl` alongside the W&B run. One JSON line per
   `_log_metrics` call. Offline-friendly: the analysis notebook reads
   JSONL directly without requiring `wandb sync` first.

8. **`configs/cases/smoke_5cases.yaml` (D24)** — 5-case smoke subset.
   The Session 5 plan named four periodic cases plus one run3 case; two
   of the planned periodic ids (`G+3.00_D0.50_Y+0.20` and
   `G+1.00_D1.50_Y+0.10`) do not exist in `configs/splits/split_v1.json`
   because periodic has no |G|=3 cases and no D=1.5 cases. Substitutes:
   `G+3.00_D0.50_Y+0.40` (run3) and `G+1.00_D1.50_Y+0.20` (run3),
   preserving the G/D/Y coverage at the cost of a 1-periodic + 4-run3
   split instead of the planned 4 + 1. Total: 16 train encounters + 5
   test_a held-out encounters across 5 cases.

9. **HANDOFF.md D24-D26 + D32 + D33 appended** — six new entries:
   D24 (5-case smoke subset), D25 (`--projection-norm` flag),
   D26 (`--anticollapse` and `--tag-suffix` flags), D32 (PLDM citation
   correction), D33 (Gust_027 and Gust_031 absorption). D27, D28, D29
   appended at session close once the variant outcomes are known.

### Variant runs (5k iters each, RTX PRO 6000 Blackwell Max-Q, cuda:2)

10. **Sanity checks pre-run** — `python -m src.training.sanity_checks --all
    --seed 0 --require-gpu`: 5/5 PASS in under a minute. The five checks
    were exercised once on the 5-case smoke subset; no variant run was
    started before this gate passed.

11. **Run A: SIGReg + BatchNorm (default config)** — 5000 iterations on
    the RTX PRO 6000 Blackwell Max-Q (cuda:2). Final: PR=1.025,
    r2_overall=0.779 (r2_G=0.923, r2_D=0.775, r2_Y=0.637), L_anti=0.081,
    L_pred ~ 0. Trajectory: PR drops from 1.44 -> 1.03 over the run
    (latent collapses to rank ~1); r2_overall starts at 0.54, peaks at
    0.94 around iter 250, then oscillates around 0.8-0.9. Quadrant:
    **TRIVIAL** (collapse to c; encoder = f(c)). PR < 9.6 -> Run B fires.

12. **Run B: SIGReg + LayerNorm** — 5000 iterations. Final: PR=1.135,
    r2_overall=0.452 (r2_G=0.645, r2_D=0.419, r2_Y=0.293), L_anti=0.124.
    Trajectory: PR stays near 1 throughout; r2_overall oscillates
    violently (range -0.86 to +0.86 across iterations). Quadrant:
    **DEAD** (latent collapsed AND probe non-functional). Different
    failure mode from A: LayerNorm prevented the clean memorisation but
    didn't recover PR. PR < 9.6 -> Run C fires.

13. **Run C: VICReg + BatchNorm** — 5000 iterations. Final: PR=17.463,
    r2_overall=0.887 (r2_G=0.914, r2_D=0.889, r2_Y=0.858), L_anti=0.083.
    Trajectory: PR climbs progressively 1.4 -> 4.7 -> 9.9 -> 13.6 ->
    17.5; r2_overall hovers in 0.86-0.97 throughout. Quadrant:
    **TRIVIAL_LITE** (PR healthy but r2 in collapse-to-c range). New
    failure mode not strictly named by the plan: VICReg's variance
    hinge forces dimension spread, but the encoder fills the extra
    dims with c-correlated noise. Predictor still trivially overfits.
    Strict trigger logic says Run D does NOT fire (C cleared PR), but
    Run D was launched anyway with explicit user approval to complete
    the 2x2 grid.

14. **Run D: VICReg + LayerNorm** — 5000 iterations. Final: PR=7.588,
    r2_overall=0.803 (r2_G=0.929, r2_D=0.784, r2_Y=0.696), L_anti=4.007.
    Trajectory: PR climbs slowly (1.4 -> 4 -> 6 -> 8), capped around
    PR=8 by the LN/VICReg interaction; L_anti stays large (~4) because
    the per-sample LayerNorm normalisation fights the per-dim VICReg
    variance hinge. Quadrant: **TRIVIAL** (partial dim spread, still
    leaks c). Confirms that LayerNorm halves VICReg's PR recovery
    compared to BatchNorm under matched data.

15. **Analysis notebook `notebooks/01_smoke_5k_analysis.ipynb`** —
    executed end-to-end on all four variants. 6 sections: load JSONL,
    loss curves, latent health 2x2, combinatorial 2x2 quadrant table,
    PCA latent exploration (on Run C's checkpoint, the variant the
    notebook picks as "best PR"), decision string. Notebook prints
    `MIXED: quadrants ['TRIVIAL', 'DEAD', 'TRIVIAL_LITE', 'TRIVIAL']`
    because no single one of the plan's named outcomes applies
    strictly; the methodological reading in D27 calls this
    **TRIVIAL-DOMINANT** (3 of 4 variants have r2 > 0.7, the
    "encoder leaks c" failure signature).

16. **HANDOFF.md D27 / D28 / D29 appended** —
    D27 records the four variants' outcomes and the TRIVIAL-DOMINANT
    classification; D28 proposes three concrete revisions to the
    auto-fallback rule (drop the r2 conjunct, case-conditional probe,
    or L_pred overfitting indicator), with the case-conditional
    probe (b) recommended as most principled; D29 records the PLDM
    priority-on-TRIVIAL rule (always-record per the Session 5 plan).

## Files

### New

| Path | Purpose |
|---|---|
| `src/training/sanity_checks.py` | Five Session 5 pre-variant gates (D24) |
| `tests/test_sanity_checks.py` | 6 unit tests for checks 1, 2, 3 |
| `configs/cases/smoke_5cases.yaml` | 5-case smoke subset (D24) |
| `scripts/run_smoke_5k_variants.sh` | Sequential variant launcher with conditional triggers |
| `notebooks/01_smoke_5k_analysis.ipynb` | Loss / latent-health / 2x2 / PCA / decision-string |
| `SESSION_REPORT_2026-05-17_session5.md` | This report |
| `SESSION5_MEANINGFUL_SMOKE_5K.md` | Session 5 plan (was untracked at start) |
| `SESSION5_PLDM_BASELINE.md` | Conditional Session 5.PLDM plan (was untracked at start) |

### Modified

| Path | Change |
|---|---|
| `src/models/encoder.py` | `projection_norm: str = "batchnorm"` ctor arg (D25) |
| `src/training/train_jepa.py` | 4 new flags + JSONL side-logger + VICReg construction-time selection |
| `tests/test_encoder.py` | Renamed BN-by-default test; added LayerNorm-path and validation tests |
| `HANDOFF.md` | D8 citation fix; D24-D26 + D32 + D33 appended; key references PLDM block |
| `CLAUDE.md` | "Baselines to implement" item 4 PLDM citation fix |
| `data_manifest/raw_cases_inventory.yaml` | Regenerated (49 cases) |
| `configs/splits/split_v1.json` | Regenerated for 49 cases / 238 encounters |

## Numerical outcomes (running tally; filled at session close)

| Variant | Anti-collapse | Encoder norm | PR@5k | r2_overall@5k | L_anti@5k | Quadrant |
|---|---|---|---|---|---|---|
| A | SIGReg | BatchNorm | 1.025 | 0.779 (G=0.92, D=0.78, Y=0.64) | 0.081 | TRIVIAL (collapse to c) |
| B | SIGReg | LayerNorm | 1.135 | 0.452 (G=0.64, D=0.42, Y=0.29) | 0.124 | DEAD (collapsed AND uninformative; r2 oscillated, dropped to -0.86 mid-run) |
| C | VICReg | BatchNorm | 17.463 | 0.887 (G=0.91, D=0.89, Y=0.86) | 0.083 | PR cleared (>16) but r2 in collapse-to-c range (>0.7); not strict HEALTHY |
| D | VICReg | LayerNorm | 7.588 | 0.803 (G=0.93, D=0.78, Y=0.70) | 4.007 | TRIVIAL (partial spread, still leaks c) |

## Decision string

Notebook algorithmic output:
`MIXED: quadrants ['TRIVIAL', 'DEAD', 'TRIVIAL_LITE', 'TRIVIAL']; manual
inspection required.`

Methodological reading (see HANDOFF.md D27):
**TRIVIAL-DOMINANT** (3 of 4 variants land in r2 > 0.7 = encoder leaks c).
The form of the leak varies across the grid:
- under SIGReg + BN: rank-1 collapse, z = f(c)
- under VICReg + BN: dim spread but c leaks into every dim
- under VICReg + LN: partial spread (PR ~ 8), still leaks c
- under SIGReg + LN: collapsed AND probe non-functional (the only non-leak
  case is also the only DEAD case)

Common signature across all four: **L_pred reaches near zero by iter 100**.
With only 16 train sub-trajectories and 5 distinct c values, the easy
thing for the encoder to learn is c itself; no useful dynamics
representation is required for L_pred to vanish.

H4 confirmed at the 5-case scale: the LeWM Two-Room failure mode
replicates on physics data. The four-variant grid additionally shows
that **prevention of rank-1 collapse is necessary but not sufficient**:
VICReg recovers PR (Run C) but the encoder still memorises c.

## Suggested next steps

**Session 5.PLDM** (per D29). The full PLDM 7-term loss (Sobal et al.,
arXiv:2502.14819) adds an inverse-dynamics term that explicitly forces
the latent to capture dynamics, not just case labels. This is exactly
the missing constraint in the four 2-term variants here. The
methodological prediction is:

- if PLDM lands in HEALTHY (PR > 16, 0.5 < r2 < 0.7), the regime-
  dependent SIGReg-vs-PLDM contrast is confirmed; Session 6 proceeds
  with PLDM as the primary trained model.
- if PLDM also lands in TRIVIAL / TRIVIAL_LITE, the failure mode is
  data-scale-bound and Session 5.5 (expand to 10-12 cases) follows.

Before Session 5.PLDM starts, verify the 7-term loss against
arXiv:2502.14819 Appendix C.1.1 and the official code at
github.com/vladisai/PLDM (the D8 description was approximate; D32
records the citation correction but did NOT re-verify the loss
formulation).

## Open items

- The DNS vorticity scale (peak |omega| up to 4000 across the cache) is
  much larger than the Session 5 plan's "O(50)" estimate. The encoder's
  first GroupNorm normalises this away, so no bf16 overflow risk, but if
  any future variant produces NaN losses early, this input scale is the
  first hypothesis to check.
- D17's "predictor.out_proj also uses BatchNorm" comment is unchanged by
  D25. In the LayerNorm-encoder variants (Run B, Run D), the predictor's
  BatchNorm-projected output and the LayerNorm-projected encoder target
  have different normalisation styles. The L_pred MSE still computes but
  measures something less clean. Whether to also pipe `projection_norm`
  through the predictor is recorded as a deferred decision in D25; the
  variant data will tell us if this matters.
