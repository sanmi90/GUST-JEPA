# CLAIM_MAP.md (Session 36, Stage 1)

The four primary claims, each with its statement locations, evidence anchors,
split and headline macros, as extracted from the current build (baseline
`paper/build/baseline.pdf`, branch jfm-rewrite-v2). Nothing is repaired here;
repairs are Stage 4 work. Line numbers refer to the Stage-0 commit state.

## P1 State construction

Claim: under the predictive objective, the scalar lift target is necessary
against latent collapse; the near-body (Chang) force-element target sharpens
the peak loads on top of it; the wake target is required for the state to
carry the wake observables; the multi-step predictive objective supplies
rollout stability, not instantaneous readability.

- Stated at 7 locations: abstract.tex:14-16; section_1_introduction.tex:126-128;
  v4/s4_a_construction.tex:15-19, 30-39, 49-50; section_4_results.tex:99-116
  (rollout-stability leg); section_6_conclusions.tex:8-11.
- Evidence: fig:cube (6), fig:readability (7), tab:closure (6),
  tab:mechanism (7), fig:mechanism_hroll (12).
- Split: test_b (v4/s4_a_construction.tex:12); cube gates pre-registered.
- Macros: \TcPeakRTwoCLN, \TcRhoCZero, \TcRhoCWN, \TcDeltaClnVsClPeak,
  \ZmeritSixteenHone, \ZmeritSixteenHeight, \ZdriftHeight, \ZdriftHone.
- Stage-4 note: 7 statement sites exceed the target (abstract + intro +
  one Results site + conclusions); the s4 restatements compress into 4.1.

## P2 Compression

Claim: load information is linearly accessible at d = 4 and approximately
unchanged over the tested dimensions under a nonlinear probe; wake-field
reconstruction requires at least d = 16.

- Stated at 5 locations: abstract.tex:17-19; section_1_introduction.tex:131-134;
  v4/s4_b_reconstruction.tex:24-37, 40-50; section_6_conclusions.tex:13-15.
- Evidence: fig:dimrace (9), including the probe-dilution control.
- Split: test_b (v4/s4_b_reconstruction.tex:25).
- Macros: \PoneLowdJepa, \PoneLowdAero, \PoneLowdFukami, \SsimLadFullDFour,
  \SsimLadFullDThirtyTwo, \SsimLadNbDFour, \SsimLadNbDThirtyTwo, \PdMlpDFour,
  \PdMlpDThirtyTwo, \PdBestFourDEight, \PdBestFourDThirtyTwo.
- Stage-4 note: the "dimension-invariant" shorthand (4 hits, lint log) must
  become the two-probe two-claim wording everywhere (memo catch 9).

## P3 Common-operator forecasting

Claim: the direct multi-horizon forecaster remains usable over 40 steps where
autoregressive rollout degrades through error accumulation; supplying the true
gust parameters provides no benefit; the predictive coordinates are the most
forecastable.

- Stated at 6 locations: abstract.tex:19-21; section_1_introduction.tex:135-138;
  v4/s4_c_prediction.tex:12-19, 21-32, 34-42; section_6_conclusions.tex:16-19.
- Evidence: fig:forecast (11; direct-vs-AR contrast + oracle leg).
- Split: test_b (v4/s4_c_prediction.tex:24).
- Macros: \RexFamClw, \RexFamCln, \RexFamAeLw, \RexTunedCl, \ArFortyClw,
  \ArFortyCln, \PoneCovOracle, \PoneCovNone, \PoneCovPhase.
- Stage-4 notes: (i) the abstract-level oracle claim must be "no benefit",
  not "degrades" (the degradation is attributed to overfitting at the
  present case count in the appendix ledger; memo correction 1.5).
  (ii) The cross-family merit column of tab:closure is currently
  suited-operator (caption documents the confound); Track M1 (D301/D310)
  replaces it with the shared-operator merit. (iii) The tab:closure caption
  says "horizon sixteen" while the emit code computes H=8
  (% REVIEW-NUMBER; MANUSCRIPT_AUDIT.md; M1 reports both).

## P4 Wall-pressure estimation

Claim: a two-stage ensemble filter sensing eight wall-pressure taps tracks the
lift through the encounter (impact R2 \PoneTwoStageCovImpact on test_b,
\PoneTwoStageCovImpactC on test_c); the wake enstrophy is not recovered by the
wall-sensor configurations considered; the static delay-embedded inverse
matches the filter's in-distribution impact median; the filter's uncertainty
consistency degrades from |G| of about 3 while median tracking holds through
|G| = 4.

- Stated at 8 locations: abstract.tex:21-25; section_1_introduction.tex:139-148;
  v4/s4_d_assimilation.tex:26-35, 120-131; section_4_results.tex:241-278,
  300-325; section_5_discussion.tex:59-61; section_6_conclusions.tex:20-30.
- Evidence: fig:hero (14), tab:family_filter (9), fig:centerpiece (16),
  tab:filter_error (11), fig:envelope (15), tab:envelope (10), fig:trade (13),
  tab:recovery (8), fig:relerr (17), fig:ownstack (18), fig:deploy (19),
  fig:dimsgrid (20).
- Splits: test_b primary (v4/s4_d_assimilation.tex:59); test_c reported-only
  (:33); the noise scale c = 1.77 validation-calibrated (:139).
- Macros: \PoneTwoStageCovImpact, \PoneTwoStageCovImpactC,
  \PoneTwoStageCovRmse, \DapEobsImpRTwo, \DapRexImpRmse, \DapRexPeakErr,
  \VfiltCLGOne, \VfiltCLGThree, \VfiltCLGFour, \VdivGFour, \VnisGZero,
  \VnisGFour, \SmRtsImpRmse, \GridJepDFourRmse.
- Stage-4 notes: memo catches 1 (false superlative at
  section_4_results.tex:271), 2 (glosses :344), 3 (threshold framing
  :345-346, D304 resolution below), 5 (|G|=3 RMSE vs tab:filter_error
  columns), 6 (abstract divergence scoping), 7 (recoverability interval
  duplicated in s5), 12 (envelope two-part sentence), 13 (static-inverse
  clause in s6).

## Multiply-stated claims

All four primary claims exceed two statement sites (7/5/6/8). Messaging is
consistent (no contradictions found), so this is a compression problem, not a
correctness problem: Stage 4 removes the Results-internal restatements and
the Discussion re-quotes (Discussion references sections, quotes no number
already in Results).

## Unsupported claims

None. Every cited evidence label resolves in the build; every primary claim
carries at least one figure or table anchor.

## Test-set selection statements

One disclosed instance: the test-selected NIS noise-band variant
(v4/s4_d_assimilation.tex:49-54), explicitly excluded from the headline by
the freeze rule and reported only in appendix (app:calibration), consistent
with the protocol statement at section_4_results.tex:11-13. Stage 5 keeps
the figure clean (the test-peek annotation moves from fig:centerpiece into
the appendix text, figure plan M3e). No other test-set-justified selection
statements were found.

## D304 resolution (recorded here for the Stage 4 rewrite)

The duplicated "near 3.0 ... near 3.0" contrast (memo catch 3) is real data,
not a macro bug: the half-divergence threshold is defined as the first grid
point of the discrete |G| ladder {0.5, 1, 1.5, 2, 3, 4} where the divergence
rate strictly exceeds 0.5, and both D = 1.0 and D = 1.5 quantize to 3.0
under that rule (D = 1.5 already reaches a rate of exactly 0.5 at |G| = 2;
D = 0.5 never exceeds 0.5 on the tested envelope, threshold null). The
Stage 4 fix is to drop the false core-diameter contrast and state the
threshold once with the grid resolution caveat. Author input not required.
