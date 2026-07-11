# SESSION 38 REPORT (2026-07-11): JFM rewrite phase 3 (D268 decisions, Stage 5 subset, Stage 6)

Branch `jfm-rewrite-v2`, commits 3d28e17..94ee715 (continues the Session
36/37 conversation). All gates GREEN at close: main 48pp rc=0,
supplementary 3pp rc=0, refs 0 both logs, tracer PASS, audit_numbers PASS,
language lint ZERO banned hits, abstract 249/250.

## Carlos's decisions applied (D268)

Budgets relaxed for clarity, filter equations stay in Methods (the MC
contract wins; wordcount s3 gate moved to 5000, reported not enforced).
Abstract trimmed to 249 texcount words with micro-trims only. Two
D310-consistency corrections forced en route: the abstract's pre-M1
"predictive coordinates are the most forecastable" became "the
wake-supervised states are the most forecastable", and s4_c's three-family
shared-REX passage was scoped to the LIFT readout and explicitly reconciled
with the ten-family five-observable tie of tab:closure (both
% REVIEW-CLAIM-marked).

## Stage 5 (feasible subset)

fig:readability_matrix DROPPED (memo 1.8). fig:hero regenerated: the
archive-signed case_id headers replaced by |G|, D, Y (the G sign is
withheld pending the s3.5 sign audit). fig:centerpiece regenerated: panel
(e) "test A" -> "validation", ladder annotations at the prose precision
(memo catch 4 closed). Discussion no longer re-quotes any Results number.
editorial/FIGURE_PLAN.md carries the full dispositions; the M3a-M3f merges
and moves (pipeline schematic, cube flatten, forecasting merge,
envelope/phase merge, ladder+grid merge, atlas slim) are deferred to the
figure-engineering pass where D302/D305 land.

## Stage 6

Four wide tables (envelope, mechanism, filter_error, recovery) wrapped in
\fittab; only two >20pt overfulls remain, both inside the pre-existing
TikZ schematic assets (M3b scope). editorial/NUMBER_AUDIT.md documents the
tooling-enforced number audit (tracer zero hand-typed numerals; 835/735
macro cross-check zero mismatches; per-table split provenance in
PROVENANCE.md). Nomenclature greps: the single CLN caption pairing is the
allowed legend exception; zero archive-signed identifiers outside the data
appendix; zero em-dashes.

## Remaining program work

- M3 figure-engineering pass (M3a-M3f + fig 2 relabel + moves; D302/D305).
- The deferred deep prose compressions (s2 -357, s4 -1266, s5 -601) if
  Carlos wants them beyond the D268 clarity standard.
- The s3.5 sign-convention audit (one % REVIEW-CLAIM open).
- Carlos-owned: session35-branch merge decision, DNS Table 1 (7 \pending{}
  rows, the submission blocker), Zenodo DOI, license/CRediT/funding,
  ledger read-through (CLAIM_MAP, PROVENANCE), keyword list check.
