# MANUSCRIPT_AUDIT.md (Session 36, Stage 0)

Baseline: branch `jfm-rewrite-v2` at the Session 35 close state plus the M5
closure edit below. Build `latexmk -pdf main` rc=0, 50 pages, snapshot at
`paper/build/baseline.pdf` (untracked). Zero undefined references or
citations (check_refs.py). One latexmk warning: the JFM class version
request (`You have requested document class 'JFM-FLM_Au'`), benign.
Number tracer: PASS at zero hits (was 1; the M5 sentinel is closed, see
section 8). Macro chain: 774 macros in macros_v3.tex against 714
numbers.json entries, zero value mismatches, zero missing, zero orphans
(lo/hi CI variants accounted). Em-dashes: zero.

Audit tooling this stage: `scripts/session36/{audit_numbers,lint_language,
check_refs,wordcount}.py`. audit_numbers wraps the existing build gate
`scripts/session35/trace_numbers.py` (not forked) and adds the literal dump
(`editorial/number_literals.csv`, untracked per the repo *.csv ignore,
regenerate via the script; 105 decimal literals, all whitelisted
protocol/configuration constants) and the macros/json cross-check.
lint_language uses word-boundary regexes rather than the upstream substring
match so `prove` does not hit `improve` (precision adaptation).

## 1. Concordance (memo numbering vs current build)

Carlos confirmed `paper/main.pdf` IS the exported main_25.pdf, so the memo's
numbering maps 1:1. Verified against main.aux:

| # | Figure label | # | Table label |
|---|---|---|---|
| 1 | fig:staging | 1 | tab:dns_pending |
| 2 | fig:paramspace | 2 | tab:architecture |
| 3 | fig:phi | 3 | tab:enkf |
| 4 | fig:method (4a fig:method_arch, 4b fig:method_eval) | 4 | tab:filter_params |
| 5 | fig:estimation_loop | 5 | tab:baselines |
| 6 | fig:cube | 6 | tab:closure |
| 7 | fig:readability | 7 | tab:mechanism |
| 8 | fig:decode | 8 | tab:recovery |
| 9 | fig:dimrace | 9 | tab:family_filter |
| 10 | fig:readability_matrix | 10 | tab:envelope |
| 11 | fig:forecast | 11 | tab:filter_error |
| 12 | fig:mechanism_hroll | 12 | tab:paired_closure |
| 13 | fig:trade | 13 | tab:prepsens |
| 14 | fig:hero | | |
| 15 | fig:envelope | | |
| 16 | fig:centerpiece | | |
| 17 | fig:relerr | | |
| 18 | fig:ownstack | | |
| 19 | fig:deploy | | |
| 20 | fig:dimsgrid | | |
| 21 | fig:atlas | | |
| 22-25 | fig:predictive_vs_reconstructive, fig:field_recovery, fig:recon, fig:pooling_cost (appendix) | | |

25 figure environments (21 main + 4 appendix), 13 tables. Matches the memo
inventory exactly.

## 2. Word counts vs Stage 4 budgets (texcount)

| Group | Words | Budget | Delta |
|---|---|---|---|
| abstract | 281 | 250 | +31 |
| s1 introduction | 1387 | 1300 | +87 |
| s2 flow and data | 1972 | 1600 | +372 |
| s3 methods (incl. v4 s3_*) | 4647 | 2600 | +2047 |
| s4 results (incl. v4 s4_*; aggregate 4.1-4.4) | 6516 | 5300 | +1216 |
| s5 discussion | 1830 | 1200 | +630 |
| s6 conclusions | 419 | 450 | -31 |

Main-text total ~17.1k words against the ~12.5k target. The two heavy
compressions are Methods (estimator sub-variants and placement details move
to appendix B per the structure map) and Discussion (three-mechanism
reorganisation, no re-quoted numbers).

## 3. The memo's 15 catches, located (Stage 4/5 work unless noted)

1. False superlative "combines the highest boundary closure":
   `sections/section_4_results.tex:271`. Rewrite as the combination claim.
2. Mismatched glosses "near unity ... nearly twenty":
   `sections/section_4_results.tex:344` (values are macros \VnisGZero,
   \VnisGFour; the GLOSS WORDS are the defect). State numbers plainly.
3. Duplicated "near 3.0 ... near 3.0": `sections/section_4_results.tex:345-346`.
   STAGE 0 FINDING: the tex uses two DIFFERENT macros (\VdivThreshDOne,
   \VdivThreshDOneHalf) and BOTH are exactly 3.0 in numbers.json (source
   part `table_v_envelope`, fmt %.1f). Not a copy bug in the tex. M2 item:
   inspect the unrounded per-D thresholds in the source part; if they
   genuinely coincide, the prose contrast is wrong FRAMING (the honest
   statement is that the half-divergence threshold is insensitive to core
   diameter) and D304 retires without author input; if they differ before
   rounding, raise the fmt precision.
4. Static-inverse 0.83 vs figure 0.825: STAGE 0 FINDING: same source value
   `dap_eobs_impact_r2` = 0.8254 (macro \DapEobsImpRTwo, fmt %.2f -> "0.83");
   the figure prints 0.825 from the same JSON. Formatting drift, not a
   provenance defect. Fix: one display precision at Stage 4/5.
5. RMSE at gust ratio three vs tab:filter_error columns:
   `sections/section_4_results.tex:266` (\DirectCLKeightGThree) and :361;
   table has |G| = 1, 2, 4 columns only. Add the column or quote the
   table's values (Stage 4 + possibly a numbers part addition).
6. Abstract "with no divergences": `sections/abstract.tex:25`. Scope to the
   boundary set or drop; the adopted front-matter draft (L4) drops
   divergence counts entirely, so this resolves when the draft binds.
7. Recoverability interval stated twice: `sections/section_4_results.tex:177-181`
   and `sections/section_5_discussion.tex:71-73`. Keep once in Results.
8. Split naming drift "test A": absent from the tex (aliases needed once in
   s2.2 at Stage 2/4); the remaining hit is INSIDE figure 16(e), drawn by
   `scripts/session35/fig_da_centerpiece_v4.py`. FIGURE-TODO (M3e).
9. Archive case IDs in figure 14 column headers: not in any caption; drawn
   by `scripts/session33/fig_hero_traces_v3.py`. FIGURE-TODO (M3, redraw
   with physical G labels; s = -G rule states in the data appendix only).
10. "dimension-invariant": 4 hits (abstract, results, conclusions; see lint
    log). Replace with the two-probe two-claim wording.
11. "beyond any wall-limited filter": 2 hits. Narrow to tap counts, delay
    windows and estimators considered.
12. Envelope narrative tension (boundary at three vs tracking at four):
    `sections/section_4_results.tex:361` states the boundary; the abstract
    celebrates |G| = 4. One two-part sentence at Stage 4.
13. Static-inverse honesty clause must survive into s6: currently present;
    the adopted concluding-remarks draft keeps it. Verify at Stage 4 close.
14. "pre-registered" 10 hits: say once in s3.5 ("fixed in advance of
    evaluation and archived with the data record"), remove the rest.
15. Table 1 framing: `sections/tables/table_dns_metadata.tex` carries 7
    \pending{} rows, the only \pending{} in the build. Submission blocker,
    Carlos-owned (DNS collaborators).

## 4. Banned-language inventory (lint_language.py)

85 banned hits, 0 em-dashes, 42 review-only "boundary" hits (mostly the
legitimate boundary-test split alias and physical boundary layer; vet at
Stage 4). Breakdown: flagship 13, pre-registered 10, honest* 9, refut* 6,
as-built 6, buys 6, load-bearing 5, catastrophic* 5, own-stack 4,
dimension-invariant 4, protocol-clean 3, knob-free 3, wall-limited 2,
erratic 2, settle* 2, specialist 1, kit strength 1, earns its keep 1,
celebrated 1, carries particular force 1. Full log reproducible via the
script; replacements per the memo section 8 table at Stage 4.

## 5. Acronym inventory (whole-word counts across sections+tables)

JEPA 12, POD 12, REX 11, CLN 10, SIGReg 7, EnKF 4, CLW 3, TiRex 3, NIS 3,
LAE 1, RTS 1, OSP 1, qDEIM 1, VICReg 1. Stage 2 confines JEPA to the s1/s3
lineage discussion and migrates CLW/CLN/REX/LAE and the filter names to the
\PredState/\LiftState/\DirectFC/... macros; cube codes survive only on the
figure 6/7 axes plus one legend row.

## 6. Placeholder / TODO inventory

- 7 \pending{} rows in `table_dns_metadata.tex` (catch 15, Carlos-owned).
- No other PENDING/TODO/placeholder markers in the built tex.

## 7. Pinned inputs for Track M1 (D301 execution)

The ten `tab:closure` rows (the memo's table 6), top block = controlled
matrix, bottom = reference recipes: JEPA (wake) | supervised only |
AE (wake) | JEPA (no wake) | AE (no wake) | anti-collapse AE | beta-VAE |
Fukami | Fukami (wake) | POD. Merit macros \Xmerit*. The caption itself
documents the suited-operator confound (matched transformer on the pooled
controlled-matrix states, residual U-Net on the reference latents).
Shared operator pinned from s3.2.1 (`paper/sections/v4/s3_3_rex.tex`):
LSTM width 512, nine quantiles, pinball loss, AdamW 1e-3, weight decay
1e-4, cosine decay, batch 64, gradient clip 1.0, 6000 iterations, arcsinh
per-window normalisation; selection provenance `scripts/session34/rex_tune.py`.
`scripts/session34/latent_rex.py` hardcodes width 256 / 3 quantiles, so M1
runs through a parameterised `scripts/session36/rex_families_m1.py`.
Latent caches: `outputs/session34/trackc_latents/` (38 variants); the
family-to-cache mapping is pinned in the M1 script and echoed in
PROVENANCE.md.

## 8. Carried Session 35 freeze items

- M5 (0.2 strong-effect bar): CLOSED THIS STAGE. Git archaeology found no
  before-results commit declaring the bar (earliest Gate O commit 9610035,
  2026-07-02, is the results commit itself), so per cite-or-drop and the
  review_closure.md disposition the bar clause is DROPPED from the one
  surviving sentence (`sections/section_4_results.tex`, Gate O paragraph,
  % REVIEW-CLAIM comment in place; conclusion unchanged: advantage
  significant but modest, no strong-form gate claimed). Tracer now at zero
  hits; mc_provenance.md MC-11 note updated.
- Retained-v3 caption seed-provenance audit: rolls into Stage 1
  PROVENANCE.md (every caption's n and seed provenance tagged there).
- POD-vs-AE intro tension and sign-convention placement: Stage 4 items,
  listed here so they cannot be dropped (s1 budget work and the physical-G
  rule respectively).

## 9. Stage 0 disposition

No scientific prose beyond the M5 closure was touched. Stage 1 (CLAIM_MAP,
PROVENANCE) is ready to start on approval; Track M1 is fully pinned; the
M2 candidate list with existing v2p2 artifacts is in SESSION_36.md.
