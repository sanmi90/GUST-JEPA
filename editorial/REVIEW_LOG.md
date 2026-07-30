# REVIEW_LOG.md

Ledger for the 2026-07-27 senior critical assessment of `paper/main.tex`
(branch `solera`, 57 pp, gates green). One row per finding. Severity is
blocker / major / minor / nit. Disposition is `claude-fix` (mechanical or
low-risk, applied on approval), `carlos-decides` (scientific or scope call),
or `dismissed` (hypothesis tested and rejected).

Passes run: 0 (review front), 1 (argument spine), 2 (superfluity), 5 (gates).
Pass 3 (line-level read) partially executed on §1, §3, §4, §5, §6.
Pass 4 (floats) partially executed.

## Findings

| id | pass | file:line | sev | finding | disposition |
|---|---|---|---|---|---|
| F01 | 1 | `sections/abstract.tex:28-30` | blocker | The abstract's only result number is the boundary R^2 (0.837), quoted with no RMSE. §4.6.2 states that this metric flatters extreme-gust tracking. See below. | carlos-decides |
| F02 | 1 | `sections/appendix_c_calibration.tex:27-40` | major | The abstract's static-inverse qualifier is evidenced only by fig:calibration(a), which is figure 30 of 30 in an appendix titled "Calibration disclosure". | carlos-decides |
| F03 | 0 | `sections/tables/table_dns_metadata.tex:9` | major | Table 1's third column is headed "Why a referee needs it" and the caption still says pending values "are filled ... before submission". Author-facing scaffolding in a submitted table. This is main_30 review item #16, queued in `paper/BUILD.md:140` to run "once filled"; the rows were filled 2026-07-24 and the retitle was not done. | claude-fix |
| F04 | 2 | `section_3_methods.tex:65-69` | minor | The two-network summary restates the encoder/predictor description given 45 lines earlier in the same subsection. ~55 words, pure duplication. | claude-fix |
| F05 | 2 | `section_3_methods.tex:187-192` vs `:278-281` | minor | The legacy suited-operator protocol (transformer on pooled states, residual U-Net on reference latents, "numerically stable on it") is described twice, near-verbatim, in §3.2 and §3.3. ~45 words. | claude-fix |
| F06 | 2 | `section_3_methods.tex:118-124` vs `:262-281` | minor | The fairness controls of §3.1 ("the reconstructive baselines carry the identical heads") are restated as the control description in §3.3. ~80 words. | claude-fix |
| F07 | 2 | `section_3_methods.tex:95-110` | minor | The wake-descriptor justification runs ~180 words and five citations to defend a descriptor the same passage says was "chosen by design rather than tuned". The wavelet-scattering sentence is decorative. Compressible to ~70 words. | carlos-decides |
| F08 | 2 | `section_3_methods.tex:54-64` | minor | Figure-pointer thicket: four cross-references plus a 40-word inline gloss of a contrast the referenced appendix figure exists to show. ~60 words. | claude-fix |
| F09 | 2 | §1:70, §4:106, §4:133, §4:152, §5:75 | major | "The POD coefficients do not reach the wake at any dimension tested" is stated at five sites, twice near-verbatim (`section_4_results.tex:136-138` and `section_5_discussion.tex:75-77` share the phrasing "not that POD is obsolete but that the POD coefficients do not achieve comparable linear readability"). | claude-fix |
| F10 | 2 | §4.2:148-152, §4.2:156-161, §4.4:327-339, §5.1:18-20 | major | "Field decode does not separate the families" is concluded four times, twice in consecutive paragraphs. ~100 words recoverable. | claude-fix |
| F11 | 2 | `section_3_methods.tex:196-199`, `:19-20`, `section_4_results.tex:195-198` | minor | The dimension-plateau / "robustness statement rather than a tuned operating point" claim appears three times, twice verbatim in that phrase. | claude-fix |
| F12 | 2 | `section_4_results.tex:200-205` | minor | Decodability-versus-estimability is previewed in §4.2 and then stated in full in §5.3. The §4 preview is ~55 words of forward pointer. | claude-fix |
| F13 | 2 | `section_4_results.tex:589-595` | minor | Prose narrates the lower panels of figure 11 in terms the caption (`:541-551`) already carries. ~55 words. | claude-fix |
| F14 | 2 | `section_4_results.tex:527-528` | nit | One-sentence forward pointer to the immediately following subsubsection. | claude-fix |
| F15 | 2 | `section_4_results.tex:559-561`, `:658-660`, fig:tracking caption | major | The base frozen filter and the production two-stage filter both appear in §4.6, forcing three separate disambiguation notes. Structural, not prose: the repetition is a symptom of reporting two estimators in the body. | carlos-decides |
| F16 | 3 | `section_5_discussion.tex:29-31` | major | "the predictive filter posts by far the fewest innovation-consistency failures (0.72 against 0.93 to 0.97)". "By far" describes a gap between failure rates of 72% and 93%. Every filter fails on most encounters; the superlative reads as spin. D-W7 already corrected a sibling claim in this sentence's neighbourhood. | carlos-decides |
| F17 | 3 | `section_5_discussion.tex:96-97` vs `section_6_conclusions.tex:46-47` | minor | §5.2 says the filter "extends the usable gust-intensity range beyond every static inverse"; §6 says a static inverse "matches the filter's in-distribution impact median". Both true at different scopes, but adjacent in a referee's reading. | claude-fix |
| F18 | 3 | `section_1_introduction.tex:68`, `section_5_discussion.tex:80` | major | `solerarico_compactness_underreview` is load-bearing twice: it supplies §1's motivating tension and §5.1's inherited caveat. An unpublished, under-review self-citation carrying the paper's central motivation. | carlos-decides |
| F19 | 1 | `section_1_introduction.tex` | minor | §1 is 982 words against ~8,000 of Results and states its contributions as prose. CLAIM_MAP records an enumerated findings list (i)-(iv) at P7 that is no longer present. | carlos-decides |
| F20 | 3 | `section_1_introduction.tex:117-118` | nit | "Section 2 describes ..." then "\S 3 the reduced states". Mixed section-reference style within one sentence. | claude-fix |
| F21 | 4 | `section_4_results.tex:280-301`, `:530-554` | minor | Figures 7 and 11 each combine two graphics under aliased labels. The text disambiguates with "top"/"bottom", so references resolve, but figure 7 letters its upper panels (d),(e) while its lower half is unlettered, and figure 11 uses Top/Bottom with no letters at all. Inconsistent with the (a),(b) scheme elsewhere. | claude-fix |
| F22 | 0 | CLAIM_MAP P3/P4, HANDOFF D-W7 | major | Six items carried as OPEN since the 2026-07-18 audit are untouched by the current review pass: fig 7 seed-axis attribution; no printed bands for the RexFam lift forecasts and the 0.755 co-trained value; per-family wake-closure and ladder-median macros; the two-stage "removes the consistency failures" claim has no bound macro; the §3.5 sign-convention audit. | carlos-decides |
| F23 | 2 | `editorial/CHANGELOG.md` (Session 37 tail) | major | The Methods compression decision Carlos was asked for at Session 37 is still open. The recorded impasse assumed cutting Methods meant moving equations out. F04-F08 show ~340 words of pure duplication that touch no equation, so the impasse is false. | carlos-decides |
| F24 | 4 | `sections/appendix_c_calibration.tex` | minor | Appendix C is titled "Calibration disclosure" but carries the estimator ladder, the smoother comparison and the band calibration. The title describes one of its three panels. | claude-fix |
| H1 | 2 | §5, §6 | dismissed | Hypothesis: the Discussion restates Results numbers. Tested by macro intersection: §5 shares 3 result macros with §4, §6 shares 3. The Discussion restates conclusions, not numbers. Cutting §5/§6 cannot move the word count. | dismissed |
| H2 | 4 | figures 4, 7, 11 | dismissed | Hypothesis: aliased float labels produce bare figure numbers with no panel pointer. Figure 4's sub-labels are genuine `subfigure` environments resolving to 4a/4b; figures 7 and 11 are disambiguated in prose. Downgraded to F21. | dismissed |
| H3 | 5 | gates | dismissed | Hypothesis: gates have drifted since the last recorded pass. Re-run 2026-07-27: `trace_numbers` PASS, `audit_numbers` PASS (968 macros, 868 json, 0 mismatches), 0 em-dashes, 57 pp, no undefined refs beyond the benign hyperref `\thepage` note. | dismissed |

## Applied 2026-07-27

Approved and landed: F03, F04, F05, F06 (reduced scope), F08, F09, F10, F11,
F12, F13, F14, F17, F20.

Not applied, with reasons:

- **F01** Both repairs drafted in `F01_abstract_drafts.md`, neither applied at
  the time. Checked while drafting: no boundary RMSE exists anywhere in the
  pipeline (`p1_bands.json` has `p1_two_stage_test_b_b177_rmse_imp` but no
  `test_c` counterpart), so keeping the boundary number in the abstract needs a
  regeneration, not an edit. **Resolved 2026-07-30, see below.**
- **F06** Overstated in the original entry and corrected here. The §3.1
  fairness list and the §3.3 control matrix describe different things and do
  not duplicate at the ~80-word scale claimed. The real duplication was
  narrower: §3.1's fifth fairness point restated the cube/three-seeds/gates
  sentence from the paragraph immediately above it. That clause was cut,
  keeping the one genuinely new statement (probe fit/score splits). ~18 words.
- **F07** Left alone. Trimming the wake-descriptor justification removes
  citations, which is a scientific judgement rather than a mechanical cut.
- **F21** Left alone. Consistent panel lettering in figures 7 and 11 requires
  regenerating the figures through their scripts, not editing captions.
- **F02, F15, F16, F18, F19, F22, F23, F24** All in the carlos-decides bucket,
  untouched.

Result: body 19,404 to 19,109 words (-295), main 57 to 56 pp. Gates after:
`trace_numbers` PASS, `audit_numbers` PASS (968/868, 0 mismatches), 0
em-dashes, 0 undefined references, main rc=0, supplementary rebuilt after main,
3 pp rc=0.

## Applied 2026-07-30: F01, option B

The abstract and §6 now lead with the in-distribution estimation result,
`\PoneTwoStageCovImpact` paired with `\PoneTwoStageCovRmse`, satisfying D252.
The boundary result is stated qualitatively in both ("the filter's advantage is
... at the held-out $|G| = 4$ boundary") and keeps its number where §4.6 and §5
report it with the surrounding caveat. No new macro and no regeneration run.
`\PoneTwoStageCovImpactC` remains bound and is still consumed by
`v4/s4_d_assimilation.tex:33`.

Two compensating trims kept the abstract inside the 250-word limit, which the
longer sentence would otherwise have broken (259 at the intermediate draft):
"remain suitable for forward prediction" to "remain forecastable" (the term §1
and §6 already use), and "using eight wall pressure taps tracks the lift through
the impact and relaxation phases" to "with eight wall-pressure taps tracks the
lift through impact and relaxation". Abstract now 249/250.

Gates after: main rc=0 at 56 pp, supplementary rc=0 at 3 pp rebuilt after main,
`trace_numbers` PASS, `audit_numbers` PASS (968 macros / 868 json, 0 mismatches),
0 em-dashes, 0 undefined references beyond the benign hyperref `\thepage` note.

## F01 in detail

`sections/v4/s4_d_assimilation.tex:63-64` reports the in-distribution result as
R^2 0.794 **with** its RMSE. `:33` reports the boundary result 0.837 with no
error in lift units. The abstract promotes the second and drops the first.

`section_4_results.tex:574-582` states the mechanism explicitly:

> the accuracy that matters at deployment is the error in real lift units, not
> the variance-normalised score [...] the strong gust inflates the lift
> excursion faster than the error grows, so the normalised score flatters the
> extreme-gust tracking while the absolute error roughly doubles.

So the paper's own Results section warns that R^2 flatters exactly the regime
the abstract's single number is drawn from, and that number (0.837) is higher
than the in-distribution one (0.794), which invites the reader to conclude the
estimator works better out of distribution than in it. Project decision D252
(CLAUDE.md) independently requires physical error to be reported alongside R^2
for this reason.

The variance-ratio caveat is deployed on the phase axis
(`s4_d_assimilation.tex:10-13`, "the relaxation variance is small and the
ratio, not the error, collapses") but never on the gust-intensity axis in the
abstract or §6.

Minimal repair: pair 0.837 with its lift-unit error in the abstract and §6, or
lead the abstract with the in-distribution pair. Requires an RMSE macro for the
boundary two-stage run if one is not already emitted.

## Word budget

Comment- and macro-stripped counts, 2026-07-27.

| Unit | Words |
|---|---|
| §1 | 982 |
| §2 | 2450 |
| §3 (+ v4/s3_*) | 5255 |
| §4 (+ v4/s4_*) | 7996 |
| §5 | 2019 |
| §6 | 455 |
| abstract | 247 |
| **body** | **19404** |
| appendices | 4075 |

Internal budget of record: ~12,500 body words (Session 37/38), relaxed at D268
without a re-baseline. Identified duplication in F04-F14 totals roughly 700
words. Reaching 13-15k requires the structural calls in F15, F19 and F23, not
prose trimming alone.
