# Session 26 artifact map (Track 0)

Maps each in-scope manuscript table/figure (compiled number from `paper/main.aux`)
to its generating script and the cached output file it reads. Paths relative to repo
root `/home/carlos/GUST-JEPA`. All artifacts are v2-era (Sessions 18-25); the v1
`outputs/session14/...` paths in the stale CLAUDE.md are NOT the paper source.

## Split (paper runs on v2, not v1)

- Manifest: `configs/splits/split_v2.json` (84 cases: 70 train / 10 test_b / 4 test_c).
- Encounters: train 226, val 86, test_b 42, test_c 24.
- test_b = 10 distinct cases behind 42 encounters (9 run3 cases x 4 enc + 1 periodic x 6 enc);
  5 interior + 5 boundary tier. **Substantial within-case clustering: only 10 independent cases.**
- test_c = 4 distinct cases (all G=+4 periodic) behind 24 encounters (6 enc each).
- Case-to-encounter mapping committed at `outputs/session26/split_v2_case_map.json`.

## Tables

| # | label | generating script | cached output |
|---|-------|--------------------|---------------|
| 4 | tab:closure | scripts/session20/exp_closure_r2.py | outputs/session23_closure/closure_r2_dimsweep_d16.csv; outputs/session20/closure_r2/closure_r2_heldout.csv |
| 5 | tab:conditioning_floor | scripts/session23/exp_conditioning_floor_plus.py | outputs/session23/conditioning_floor_plus/h16/floor.csv |
| 6 | tab:latent_drift | scripts/_oneoff_latent_drift_diagnostic.py | outputs/session18/exp_b1_test3/latent_drift_diagnostic.json |
| 7 | tab:controls_2x2 | scripts/session20/track_a_closure.py | outputs/session20/track_a/controls_2x2.csv,.json |
| 8 | tab:jepa_dimsweep | scripts/session23/closure_dsweep.py + build_jepa_table.py | outputs/session23_closure/closure_r2_dsweep.csv; closure_r2_noc.csv |
| 9 | tab:b1_closure_train_r2 | scripts/session18 b1 pipeline | outputs/session18/exp_b1_test3/physical_closure_noBN_unified.csv (train rows) |
| 10 | tab:paired_closure | scripts/session21/session21_paired_closure_stats.py | regenerated from latents+rollouts+DNS (no cached array file) |

Table 10 note: `session21_paired_closure_stats.py` reuses `scripts/session20/exp_closure_r2.py`
(LATENTS_ROOT=outputs/session18/exp_b1, ROLLOUTS_ROOT=outputs/session18/exp_b1_test3,
DNS_METRICS_PATH=outputs/session17/exp2/dns_physical_metrics.npz). VALIDATED this session:
reproduces the manuscript paired numbers exactly (repr wake 31/42, dErr +43.1, CI [+23.5,+66],
sign p=1.4e-3; forecast wake 27/42, dErr +32, CI [+10.8,+54.8], sign p=4.4e-2).

## Per-encounter absolute-error arrays (predictive JEPA d=64 vs reconstructive Fukami d=64)

NOT cached as standalone arrays. Regeneration path (no training, no GPU):
`session21_paired_closure_stats.load_per_encounter_abs_error(observable, mode, family)`
returns the per-encounter |probe-DNS| on test_b at H=16 for all six observables in both
modes (repr=z_dns, forecast=z_markov), in canonical sorted (case_id, encounter) order so the
arrays are index-aligned across families. For case-clustered statistics (Track 1) this loader
must also expose the per-element case_id (it currently drops the dict keys); Track 1 extends it.

## Figures

| # | label | generating script | cached output |
|---|-------|--------------------|---------------|
| 8 | fig:persistence | scripts/session20/exp_persistent_homology.py | outputs/session20/persistent_homology/persistent_homology.json |
| 12 | fig:ot | scripts/session20/exp_ot_field_and_alignment.py | outputs/session20/ot/ot_results.json |
| 13 | fig:scale_decomp | scripts/session20/exp_scale_decomposition.py | outputs/session20/scale_decomp/scale_decomp.json |
| 14 | fig:observability | (session21 pressure_v2 pipeline) | outputs/session21/pressure_v2/pressure_obs_v2.csv |
| 15 | fig:wake_code | scripts/session25_fig_wake_code.py | outputs_causal/jepa_modes/cross_encoder3.json |

Figure PDFs live under `paper/sections/figures/results/` (git-tracked via the .gitignore
exception). Fig 8 PDF = sections/figures/results/figC* / fig persistence; Fig 14 PDF =
figF_observability.pdf.

## Build and conventions

- Source of truth: `paper/sections/*.tex`, `paper/main.tex`, `paper/sections/tables/*.tex`.
  The `.md` under `sections/_v2_md_archive/` are ARCHIVED. Do NOT run the md->tex converter
  (`scripts/_oneoff_md_to_tex.py`); it would overwrite hand-authored .tex.
- Build: `cd paper && latexmk -pdf -interaction=nonstopmode main.tex`. Baseline: exit 0,
  40 pages, no undefined refs/citations.
- Conventions checker: `~/.claude/skills/academic-paper-writer-vortex-jepa/scripts/enforce_conventions.py <file>`.
  Baseline at `outputs/session26/baseline_conventions.txt`: 0 em-dashes anywhere; pre-existing
  R^2-coverage/uncertainty heuristic flags (abstract 1, sec2 1, sec4 46, sec5 4, appA 3, appB 13)
  are false positives (numbers live in tables the line-checker cannot see; D159/D160). Do NOT chase
  them; the load-bearing gate is em-dashes == 0 and no new forbidden phrasings.

## Outputs are gitignored; force-add small summaries

`outputs/` and `outputs_causal/` are gitignored, as are `*.csv/*.npz/*.npy/*.h5/*.pt`. Prior
sessions force-add (`git add -f`) the small `.txt/.json` summaries for traceability (17 such
files tracked). Session 26 follows that: force-add `.txt/.json/.tsv/.md` summaries under
`outputs/session26/`, leave large arrays/CSVs untracked but referenced.
