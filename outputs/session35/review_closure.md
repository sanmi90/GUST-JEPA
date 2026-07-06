# Review closure map (SESSION_35_MANUSCRIPT_V4.md Part 1 -> v4 text)

Session 35 P4. Each finding of the v4 honest review mapped to where the new
text answers it. File pointers are tex sources on branch
session35-manuscript-v4 (state at HANDOFF D263+).

## Critical findings

- C1 (operator confound: transformer/U-Net split claimed "same operator"):
  FIXED. The split is retitled a legacy suited-operator protocol in
  section_3_methods.tex (closure-protocol subsection, bridging sentence) and
  the primary comparison is the shared REX operator
  (paper/sections/v4/s3_3_rex.tex); both protocols reported, ordering
  operator-robust in v4/s4_c_prediction.tex paragraph 1 (RexFam* macros,
  n = 3 per family).
- C2 (head/supervision asymmetry buried in an appendix): FIXED. The
  2x2x2 cube IS the construction narrative: v4/s4_a_construction.tex with
  pre-registered gates, paired CIs, and the three-heads Methods rewrite
  (section_3_methods.tex 3.1 + v4/s3_1_chang_head.tex). The old "not a
  clean isolation" caveat sentence was replaced by a pointer to the cube.
- C3 (DNS metadata table author-owned): UNCHANGED by design. The
  \pending{} rows stay; author-owned per the spec.
- C4 (under-dispersion conflation: calibration vs model error): FIXED with
  data. Two-regime statement written separately in v4/s4_d_assimilation.tex
  (closing paragraph) and quantified: impact side WAS calibration (the
  state-dependent Q removes the divergences), relax side is model error
  (median contraction; the smoother, not the noise model, addresses it);
  strengthened by the Session 35 NIS refutation (same file, calibration
  paragraph) and the disclosure appendix (appendix_d_ledgers.tex,
  app:calibration).

## Major findings

- M1 (forecast closure demoted): FIXED. Part C elevates forecasting to a
  co-headline with the direct-vs-autoregressive mechanism
  (v4/s4_c_prediction.tex); the fitted-merit sleight-of-hand is gone
  because the shared-REX protocol is primary.
- M2 (tuned-baseline provenance unstated): FIXED in Methods: the
  reconstructive references use their best-documented configuration
  sentence retained (section_3_methods.tex closure subsection) and the kit
  freeze is the symmetric answer, now stated via the per-filter/per-family
  parameter provenance (v4/s3_4_estimators.tex, table tab:filter_params);
  MC-9 records the config hashes (outputs/session35/mc_provenance.md).
- M3 (inflation prose single-policy): FIXED. Per-family rho stated in
  v4/s3_4_estimators.tex (base-filter subsection) and in
  tab:filter_params.
- M4 (parameter floor without the combined variant): CLOSED more
  decisively by the conditioning null with seed bands:
  v4/s4_c_prediction.tex final paragraph (PoneCovOracle vs PoneCovNone,
  non-overlapping; phase leg a wash). R12 correction recorded in
  SESSION35_REPORT.md.
- M5 (0.2 strong-effect bar uncited): PARTIALLY CLOSED. The v4 text no
  longer invokes the bare 0.2 bar; pre-registration citations are in
  Methods 3.5 (v4/s3_5_protocol.tex, uncertainty paragraph: gates written
  and committed before results, shipping with the data record). The
  archived v3-era plan commit for the original bar remains
  [PENDING: author] in mc_provenance.md MC-11; if not located, the bar
  stays dropped, which the current text already reflects.
- M6 (plateau/pooling numbers referenced not quoted): PARTIALLY CLOSED.
  The v4 dimension story quotes its numbers via macros (Lad*/SsimLad*/Pd*
  in v4/s4_b_reconstruction.tex). The v3 \PlateauSpread mention lives in
  the retained physics subsection; verify at the final freeze that every
  retained v3 sentence still quotes its macro.
- M7 (wake-enstrophy filter null unexplained): FIXED. Mechanistic
  explanation written in v4/s4_d_assimilation.tex closing paragraph (Chang
  visibility + delay-window sensing geometry + the K x W trade pointer)
  per locked decision D6.

## Moderate / prose findings

- Abstract opaque macros / undefined "static inverse": FIXED (abstract
  rewritten; static inverse introduced with its meaning in Part D and the
  contributions define leakage-free at first use).
- Intro three-contribution mixed levels + leakage-free undefined: FIXED
  (four contributions, one per stage; leakage-free defined inline;
  section_1_introduction.tex).
- Conclusions opener stronger than caveats: FIXED (softened, error + R2
  together, three bounding facts including the static-inverse bound).
- POD-vs-AE intro tension, sign-convention footnote, 120-frame window,
  case-clustering rationale placement: NOT YET swept; queued for the
  fresh-eyes jfm_project_writing_style pass (P4 remainder) with the v2.1
  recovery-ordering split-brain note (Appendix B).

## Strengths-to-preserve check

Honest nulls kept and now explained (wake null with mechanism, merit-tie
fragility superseded by the operator-robust protocol); division-of-labour
attribution upgraded to the three-head form; triple uncertainty protocol
extended (member-noise seeds added); delay-embedding organizing principle
retained verbatim in v4/s3_4_estimators.tex; |G|=4 boundary framing now
carries the two-stage positive result without softening the 3D
observability limit.
