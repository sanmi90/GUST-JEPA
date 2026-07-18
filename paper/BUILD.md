# Manuscript build instructions (main_30, session-39 comparison-led revision)

Canonical branch: `session39-comparison-lead`. The manuscript is authored
directly in LaTeX (the v1-era Markdown-to-TeX pipeline is retired; the old
instructions live in git history of this file). Output: `paper/main.pdf`
(55 pp) and `paper/supplementary.pdf` (3 pp).

## One-shot build

```bash
cd /home/carlos/GUST-JEPA && source .venv/bin/activate
cd paper
latexmk -pdf -interaction=nonstopmode main.tex
latexmk -pdf -interaction=nonstopmode supplementary.tex
```

Both must exit 0. Expected page counts: main 55, supplementary 3 (check with
`pdfinfo`). `supplementary.tex` cross-references `main.aux` via the `xr`
package, so build main first.

## Source layout

- `paper/main.tex`                 skeleton: class `JFM-FLM_Au` with options
                                   `[lineno,paper]`, preamble (`\fittab`,
                                   notation macros), section includes.
- `paper/sections/abstract.tex`, `section_1_introduction.tex` ...
  `section_6_conclusions.tex`, `appendix_{a,b,c}_*.tex`   top-level sections.
- `paper/sections/v4/s3_*.tex, s4_*.tex`   LOAD-BEARING subfiles input from
                                   sections 3 and 4 (nine files). Do not
                                   archive or delete them.
- `paper/sections/tables/*.tex`    tables, all numbers macro-bound.
- `paper/sections/figures/`        TikZ (`tikz/`) and matplotlib (`results/`)
                                   assets.
- `paper/sections/supp_*.tex`      supplementary sections (S1 ledger, S2
                                   failure modes, S3 figures, S4
                                   suited-operator table).
- `paper/nomenclature.tex`         hand-maintained model/estimator name macros
                                   (\PredState, \DirectFC, \TwoStageKF, ...).
- `paper/refs.bib` (+`refs_to_add.bib`)  bibliography.

## Numbers pipeline (no number is ever hand-typed)

```
outputs/session3{1,2,3,9}/...            frozen evaluation artifacts (v2p2)
  -> scripts/session33/emit_numbers_parts.py   (+ per-track emitters:
     scripts/session35/emit_p{1,3}_parts.py,
     scripts/session39/emit_critical_parts.py, ...)
  -> outputs/session33/numbers_parts/*.json    (25 parts)
  -> scripts/session33/eval_all_v3.py          merge + validate + v2.1
     macro-collision check   ->  outputs/session33/numbers.json
  -> scripts/session33/emit_macros_v3.py       ->  paper/macros_v3.tex
```

`main.tex` inputs `macros.tex` (v2.1-legacy, retained and collision-checked)
then `macros_v3.tex` (canonical, 968 macros). Regeneration is deterministic:
re-running `eval_all_v3.py` + `emit_macros_v3.py` reproduces both files
byte-identically except the generated-timestamp/provenance-commit header
lines (verified 2026-07-18). If a diff shows anything beyond those two
header lines, an input artifact changed: stop and audit before rebuilding.

## Verification gates (run all before any freeze/push)

```bash
cd /home/carlos/GUST-JEPA
.venv/bin/python scripts/session36/audit_numbers.py   # wraps the tracer
.venv/bin/python scripts/session36/wordcount.py       # informational budgets
cd paper && latexmk -pdf -interaction=nonstopmode main.tex
```

Pass criteria:
- `audit_numbers` prints `[trace_numbers] PASS` (zero hand-typed result
  numerals in content .tex; whitelist = protocol constants) and
  `[audit_numbers] PASS` (every macros_v3.tex macro matches numbers.json,
  zero mismatches / missing / orphans).
- `wordcount`: abstract must be <= 250; s2/s4/s5 run over their nominal
  budgets by standing decision D268 (clarity over budget) -- report, do not
  force.
- Build log: 0 LaTeX errors, 0 undefined references/citations,
  0 "Float too large" warnings. Accepted residual warnings: ~26 overfull
  hboxes of ~1.6 pt from the class's running head/footer machinery, and one
  33.95 pt overfull vbox on page 1 caused by the class's draft banner
  placeholder (replaced in production typesetting). Anything larger is a
  regression.
- `grep -c "—" sections/*.tex sections/v4/*.tex` must be 0 (no em-dashes).

Known class trait (do not "fix"): booktabs `\toprule`/`\midrule` do not
render under `JFM-FLM_Au.cls`; every table shows a single heavy rule at the
block end. Wide tables must be wrapped in `\fittab{...}` (conditional
resizebox, defined in main.tex); an unwrapped wide tabular under the
`lineno` option produces a large phantom overfull hbox and a stray rule
fragment in the right margin.

## Figure lineage (label -> generator -> notes)

All data figures are v2p2 except fig:staging (v2p1 asset, data-identical:
the v2p2 cache symlinks those encounters; deliberate, do not regenerate).
TikZ sources live in `paper/sections/figures/tikz/`.

| Figure | Asset | Generator |
|---|---|---|
| 1 fig:staging | fig_staging_v2p1.pdf | scripts/session29/fig_staging_v2p1.py |
| 2 fig:paramspace | fig_paramspace_v3.pdf | scripts/session33 |
| 3 fig:method | tikz/fig1_jepa_architecture.pdf + tikz/fig_estimation_loop.pdf | tikz sources |
| 4 fig:cube | fig_cube_*_v4.pdf | scripts/session35 (split at session 39) |
| 5 fig:dimrace | dimension-axis panels | scripts/session34/35 |
| 6 fig:atlas | fig_atlas_dmd_v3.pdf + fig_t1_spectra_v4.pdf | scripts/session33/fig_atlas_dmd_v3.py + scripts/session39/t1_spectral_flatness.py |
| 7-9 forecast/mechanism/trade | fig_forecast*, fig_mechanism_hroll_v3, fig_t_trade | scripts/session33/35/39 |
| 10 fig:hero | fig_hero_traces_v3.pdf + fig_cl_envelope_traces_v4.pdf | scripts/session33/38 |
| 11 fig:envelope | fig_envelope_v3.pdf | scripts/session33 |
| 12 tracking / 13 fig:dimsgrid | fig_deployment-family panels / DA grid | scripts/session35 |
| 14-29 appendix | fig_*_v4.pdf, portraits, decode, calibration | scripts/session35/39 (grep scripts/ for the asset name) |

To find any figure's generator: `grep -rn "<asset name>" scripts/`.
Regenerate only on frozen inputs; figure edits never change numbers (the
tracer guards the text side).

## Editorial ledgers

`editorial/CHANGELOG.md` (current), `editorial/FIGURE_PLAN.md` (session-39
dispositions + caption contract), `editorial/NUMBER_AUDIT.md` (gate
tooling), `editorial/PROVENANCE.md`, `editorial/MANUSCRIPT_AUDIT.md` and
`editorial/CLAIM_MAP.md` (Session-36 baselines, historical; line numbers
refer to the pre-session-39 build), `editorial/REVISION_PLAN.md` (executed).
`paper/HEADLINE_NUMBERS.md` is superseded (numbers.json is canonical).

## What still needs the authors (submission blockers)

- Table 1: seven `\pending{}` DNS solver-resolution rows, filled from
  `paper/dns_metadata.yaml` by the simulation collaborators
  (`scripts/session28/DNS_COLLABORATOR_PACKAGE.md`). THE blocker.
- Zenodo DOI, license, CRediT, funding statements in the front matter.
- Retitle the Table 1 value column once filled (main_30 review item #16).
