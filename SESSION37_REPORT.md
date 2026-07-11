# SESSION 37 REPORT (2026-07-11, overnight autonomous): JFM rewrite phase 2, Stages 3 + 4 core

Branch `jfm-rewrite-v2` (continues Session 36 in the same conversation at
Carlos's "keep going"). Commits a8d93fa..fe7f2e4. All gates GREEN at close:
main 49pp rc=0, supplementary 3pp rc=0, zero undefined refs both logs,
tracer PASS, macros/json cross-check PASS, language linter ZERO banned hits.

## Stage 3 (complete)

Results reassembled into the four target subsections, labels preserved,
absorbed headers demoted: 4.1 Constructing a physically useful state
(construction cube + decode floor + what-the-state-carries + the physics
subsubsection with the DMD shedding clock and the atlas); 4.2 Compression
and forecastability (dimension tiers split out of s4_b into
v4/s4_b2_dimension.tex + distributed-code + spatial-trade + forecasting +
rollout mechanism); 4.3 What wall pressure observes (retitled, labels
kept); 4.4 Sequential estimation and operating limits (tracking + envelope
+ estimator ladder). sec:res_code moved with the distributed-code content;
two references re-pointed.

Appendices: A and B unchanged; appendix D dissolved: D.3 -> NEW in-paper
appendix C (calibration disclosure, per the memo/D305 recommendation);
D.1/D.2 -> supplementary S1/S2. NEW paper/supplementary.tex (JFM class, xr
cross-refs into main, S-numbering): S1 forecaster ledger, S2 failure
modes, S3 supplementary figures (decode gallery + pooling cost), S4
suited-operator merit table (documents the D310-superseded protocol).
Sections 2 and 3 retitled to the target structure.

## Stage 4 (front matter + claim repairs + language table complete; compression stopped deliberately)

Front matter bound (D303/D306): title now "Predictive reduced-order states
for wall-pressure estimation of extreme vortex gust--airfoil interactions";
abstract, intro requirements-and-roadmap paragraph and Concluding remarks
replaced by the approved front_matter_rewrite.tex drafts, every number
macro-bound; the abstract's divergence-count claim is gone; the n = 4
qualifier added to the 3D-observability sentence; roadmap bound to labels;
"Concluding remarks" per lineage convention.

Mandatory claim repairs, all % REVIEW-CLAIM-marked: (1) false superlative ->
combination claim (closure among the highest, divergence 0.72 vs 0.93-0.97,
cross-strength consistency incl. the reconstructive baseline's -0.39 at
|G| = 1, wake readout); (2) glosses -> plain macro values; (3) duplicated
"near 3.0" -> grid-resolution statement (D304 retired: widest core reaches a
one-half rate at |G| = 2, most compact never crosses); (5) tab:envelope
gains the filter C_L RMSE column so the ratio-three value is table-anchored;
(7) recoverability interval quoted once; (12) envelope reconciled in one
two-part sentence; (4) centerpiece precision FIGURE-TODO; (6)/(13) resolved
by the draft binding.

Language table swept to ZERO banned hits (85 at Stage 0). The divergent
criterion (impact-phase R^2 below -1, from two_stage_envelope.py) is defined
once and used thereafter. The single "fixed in advance of evaluation"
statement lives in s3.5. The s3.5 inventory-sign-convention sentence is
flagged % REVIEW-CLAIM against the physical-G rule (audit at Stage 5).

## Deliberately stopped: word-budget compression (Carlos decision required)

s1 (1235/1300) and s6 (376/450) are inside budget; abstract 264/250 is a
macro-token counting artifact on the approved draft. The remaining
compressions collide with prior locked work or need author judgment:

- s3 Methods 4641/2600: the budget collides with the Session 35 Gupta MC
  completeness contract (numbered equations per filter; MC-1..MC-12 mapped
  in outputs/session35/mc_provenance.md). Proposal: move OSP placement, the
  smoother configuration detail and eq-free config prose to appendix B
  (~-700 words) and keep the filter-defining equations; the remaining
  ~-1300 needs a relaxed budget or undoing MC. CARLOS DECIDES.
- s4 6590/5300: prune the multiply-stated claims (CLAIM_MAP: P1 x7, P4 x8
  statement sites) and the demoted subsubsection redundancies.
- s5 1805/1200: the memo's three-mechanism reorganisation.
- s2 1957/1600.

## Open for the next session (Stage 4 tail + Stage 5/6 = Session 38)

Compression per the decisions above; Stage 5 figure plan (M3a-M3f redraws,
FIGURE-TODOs: fig:readability_matrix drop-or-regen, fig:hero archive-ID
headers, fig 16e test-A wording, centerpiece precision; D302/D305 at the
STOP); Stage 6 consistency (NUMBER_AUDIT.md, nomenclature grep, visual diff
vs build/baseline.pdf). Carlos-owned unchanged: session35-branch merge, DNS
Table 1, Zenodo DOI, license/CRediT/funding; ledgers CLAIM_MAP/PROVENANCE
still await his read-through, plus the abstract 264-word acceptance and the
sign-convention audit decision.
