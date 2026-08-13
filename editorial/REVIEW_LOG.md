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

## Applied 2026-07-30: open-issue walkthrough, rounds 1 and 2

Six findings closed with Carlos in the loop, one decision each.

- **F16 rescoped.** "the predictive filter posts by far the fewest
  innovation-consistency failures" became "every filter is stressed: all of
  them fail innovation consistency on most encounters, least often the
  predictive filter". The ordering survives, the superlative does not, and the
  shared failure is now stated rather than left for a referee to compute from
  the macros (0.72 against 0.93 to 0.97).
- **F24 applied, F02 premise corrected.** Appendix C retitled "Estimator
  selection and calibration disclosure", which is what its three panels carry.
  F02's original framing was wrong and is withdrawn: the abstract's
  static-inverse qualifier is *not* evidenced only in an appendix.
  `v4/s4_d_assimilation.tex:24-40` states the claim in the body with a bound
  macro (\DapEobsImpRTwo) and rebuts it three ways (tails, boundary, state
  tracking beyond a single read-out). Only the plot is remote. Promoting it
  would have added a 14th body figure to a paper flagged as long.
- **F18 verified, no change.** The companion paper is still under review at
  JFM. The `@unpublished` entry renders "Under review, Journal of Fluid
  Mechanics" in `main.bbl`; §1:24 attributes a specific empirical finding
  rather than an authority claim, and §5.1:23 flags it as an inherited caveat
  and then argues past it with three controls. Author-owned residue: confirm
  coauthor spelling and final title before submission (`refs_to_add.bib:85`).
- **F15 resolved without demotion.** The base/production split is declared once
  at the head of §4.6 and the two repeats are gone (the `fig:envelope` caption
  parenthetical and the `tab:envelope` caption clause). Both captions keep
  "under the base frozen filter" so they stay self-contained. Demotion was
  rejected on evidence: the base-filter consistency strip in `fig:envelope`(a)
  is the exhibit for §6's "the filter's uncertainty consistency degrades even
  as the median tracking holds", so moving it to the supplementary would have
  cost a §6 claim. The "one filter in the body" line in CRITICAL_ASSESSMENT
  overreached and is withdrawn.
- **F23 CLOSED, budget re-baselined.** The Session 37 impasse assumed ~700
  words of Methods were movable. After F04-F08 landed, §3 is 4997 words and
  what remains movable is the fixed-lag smoother (115 w) plus sensor placement
  (192 w) into appendix B: 307 words, which fragments the estimator
  description across body and appendix while §4.6 keeps referring to the
  smoother. **Budget of record is now ~19,100 body words.** The 12,500 figure
  was a Session 37/38 internal target, relaxed at D268 and never re-derived
  against any JFM constraint; JFM sets no hard limit. F23 is not to be
  re-litigated without a new reason.
- **F19 applied as an enumeration.** §1's findings sentence became "The
  results support four claims: (i) ... (iv) ...", inline rather than as an
  itemize list, so the prose register holds. Note for the record: my finding
  claimed CLAIM_MAP carried a lost (i)-(iv) list at P7. It does not; that
  premise was unverified and is withdrawn. The change stands as a deliberate
  scannability choice, not a restoration.

Gates after rounds 1 and 2: main rc=0 at 56 pp, `trace_numbers` PASS,
`audit_numbers` PASS (968 macros / 868 json, 0 mismatches), 0 em-dashes, 0
undefined references beyond the benign hyperref `\thepage` note.

## Applied 2026-07-30: open-issue walkthrough, round 3

- **F07 dismissed.** The wake-descriptor justification stays. Re-reading
  `section_3_methods.tex:25`, the wavelet-scattering sentence is not
  decorative: it is the passage's argument that pairing a spatial pooling with
  a radial spectrum is a principled construction rather than an ad hoc one,
  and it cites a turbulence application doing it for that reason. Cutting it
  would weaken the defence of a descriptor the same passage calls "chosen by
  design rather than tuned", which is where a referee would press.
- **F21 applied in LaTeX, no figure regeneration.** Both floats stack two
  independent graphics, so the lettering is a float-level edit. Figure 7's top
  graphic already carries internal panels (a)-(e), so its lower graphic became
  (f) and the caption's "Top:"/"Bottom:" became "(a-e)"/"(f)". Figure 11 had
  no internal letters, so its two graphics became (a) and (b). Two prose call
  sites were updated to the new letters: `section_4_results.tex:79`
  ("figure~\ref{fig:t1_spectra}, bottom" to "...}f") and `:225` ("The lower
  panels of figure~..." to "Figure~...b shows"). Note: the labels are set in
  the body font, so figure 7's (f) does not match the sans lettering of its
  own (a)-(e). In-figure lettering would be cleaner and is now possible (see
  the artefact note below); it was not done because it needs a regeneration of
  `fig_t1_spectra_v4.pdf` off the 63 GB latent cache.
  Gate note: the first attempt used `\makebox[0.78\linewidth]` to align the
  label to the graphic, which `trace_numbers` correctly flagged (it strips
  `\includegraphics` widths but not `\makebox`). Changed to `\linewidth`
  rather than whitelisting a decimal.

### F22.1 RESOLVED, and the ledger's premise was backwards

CLAIM_MAP recorded this as "fig 7's panel says 'encoder seeds' where
text/caption say operator seeds", implying the text needed fixing. The
opposite is true, and the figure was the defect.

Evidence: `scripts/session34/latent_rex.py` loads a **frozen** encoder run by
`--run` (`load_cache(CACHE, args.run, ...)`) and uses `--seed` only to seed
operator training (`torch.manual_seed`), writing the `_s{n}` suffix. So
`latent_rex_jepa_pool_vec{,_s1,_s2}.json` are one encoder under three
**operator** seeds. The generator's own docstring called the identical
mechanism "operator seeds" for panel (b) while calling it "encoder-seed
retrains" for panel (a), which is the internal contradiction that settles it.

So `v4/s4_c_prediction.tex:18` and `:68` ("three operator seeds") were
**correct** and are unchanged. Fixed instead:
`scripts/session35/fig_forecastability_v4.py` panel (a) title
("encoder seeds" to "operator seeds") and its docstring, and the figure asset
was regenerated. Data is unchanged: the CLW per-seed values
[0.6646, 0.5713, 0.6792] have mean 0.638382, matching `\RexFamClw` to seven
digits, and `fig_forecast_null_v4.pdf` came back text-identical.

Checked and found correct, not changed: `tables/table_closure.tex:23` and
`appendix_a_regularisation.tex:267,350` all say "operator seeds" and are
right, because `scripts/session36/rex_families_m1.py` also seeds operator
training (`train_operator(Zt, seed, ...)`).

### Artefact availability, discovered while resolving F22.1

Ten of the twelve JSONs this figure plots, the upstream
`outputs/session34/trackc_latents` cache and the session33 encoder
checkpoints are all **absent from this working tree**. They exist in
`/home/carlos/GUST-JEPA/outputs/` (read-only), which is where the ten JSONs
were recovered from. `outputs/` is gitignored, so nothing was lost from
version control, but a figure regeneration cannot be assumed to work from a
fresh clone of this tree alone. Anyone regenerating a body figure should check
Carlos's tree first and must not simply run a generator that says "n from the
files present on disk": with the inputs missing it would have silently emitted
a degraded n=1 figure instead of failing.

Still open: F22 items 2-5 (RexFam/co-trained bands, per-family wake-closure
and ladder-median macros, the unbound two-stage consistency claim, the §3.5
sign-convention audit).

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

## Applied 2026-07-30: the operator-seed axis was never declared (F22.1b)

Raised by Carlos while reviewing the F22.1 fix: "operator seeds" is not
defined anywhere the reader can find it. Investigating it found a
Methods/Results inconsistency rather than a wording problem.

`v4/s3_5_protocol.tex` declared uncertainty "at three levels": encoder
training variance (three seeds), filter member-noise variance (five seeds),
and the case-clustered bootstrap. Operator training variance was not among
them, yet "three operator seeds" appears at five sites (the fig:forecast
caption, `v4/s4_c_prediction.tex:18` and `:46`, `tables/table_closure.tex:23`,
`appendix_a_regularisation.tex:267` and `:350`). Worse,
`section_3_methods.tex:88` told the reader that every reported band is "the
standard deviation across three encoder retrains", so a reader arrives at the
Results primed to misread every operator-seed band as an encoder-seed band.

Applied:

- `s3_5_protocol.tex`: "three levels" to "four levels", with operator
  training variance declared as its own level, scoped explicitly: it isolates
  the forecaster's training noise on a frozen representation and does NOT
  include encoder variability. That scoping matters because the family gaps in
  fig:forecast(a) are therefore not bracketed by encoder-retrain uncertainty.
- `section_3_methods.tex:88`: no longer claims every band is an encoder
  retrain; now "three independent retrains ... of the encoder or, where
  families are compared under one shared forecaster, of that forecaster on a
  frozen representation".
- Figure title regenerated from "shared direct forecaster, operator seeds" to
  "shared direct forecaster, 3 seeds (encoder frozen)", which states what
  varies and what is held fixed without the jargon.
- fig:forecast caption made self-contained: "three operator seeds each, the
  forecaster retrained on a frozen representation".

Ledger numbering note: CLAIM_MAP calls this "fig 7". The figure carrying the
title is `fig:forecast`, not figure 7 (which is fig:atlas/fig:t1_spectra). The
CLAIM_MAP numbering predates the Session 39 figure renumber and should not be
trusted for figure identity.

Gates after: main rc=0 at 56 pp, `trace_numbers` PASS, `audit_numbers` PASS
(968/868, 0 mismatches), 0 em-dashes, 0 undefined references beyond the benign
hyperref note.

---

## Relevance overlay (2026-07-31, branch `review-colour-map`)

A colour-coded relevance map over the whole manuscript, requested to answer one
question directly: what has to stay, and what can go. It is an overlay, not an
edit. The manuscript text is never rewritten; every marker is inserted inline
and `--strip` restores the sources byte for byte, verified by a round-trip
assertion inside the apply step and by `git diff` returning clean.

**Apparatus.** `paper/reviewmarks.tex` (master switch plus three sub-switches,
five colour classes, margin-note emitter, inline ID tags, legend, review-mode
page widening) and `scripts/review/annotate_relevance.py` (`--inventory`,
`--apply`, `--strip`). The ledger is `editorial/RELEVANCE_MAP.md`, one row per
block, and it is the thing to read; the PDF is how to read it in context.

**Coverage.** 164 blocks: every prose paragraph and every figure or table
caption that `main.tex` typesets, across the abstract, the six sections, the
`v4` subfiles, the three appendices and the eleven table files. Classified
K 135, T 16, A 6, S 4, D 3.

**What the map says.** The three deletions are duplication, not content:
`S3-08` is a two-sentence analogy nothing refers to, `S3-12` states the
uncertainty protocol a third time (and at three levels where `S35-02` says
four), `S4-10` is a bare forward pointer. The six annex marks are the Chang
head construction (`S31-01`), the fairness enumeration (`S3-05`), the smoother
recursion and the placement criterion (`S34-06`, `S34-07`), the phase-split
narration whose figure is already in appendix A (`S4-15`), and the calibration
narrative that appendix C exists to hold (`S4D-03`). The four supplementary
marks are the run4 sign-check (`S2-05`) and the topology and preprocessing
robustness checks on the superseded d = 64 encoders (`APA-05`, `APA-06`,
`APA-C3`). Of the sixteen trims, six are the duplication F09 to F14 already
identified in `CRITICAL_ASSESSMENT.md`, now located to the paragraph.

**Three findings that came out of building it.**

1. `sections/protocol_box.tex` is input by nothing. It is an orphan file
   carrying an evaluation-protocol box that no longer reaches the PDF.
2. `sections/appendix_c_supplementary_figures.tex` is input by
   `supplementary.tex`, not by `main.tex`, while its name says it is in-paper
   appendix C. Its three siblings in that document are named `supp_s1_*`,
   `supp_s2_*`, `supp_s4_*`; this is supplementary section S3 and should be
   `supp_s3_figures.tex`.

   CORRECTION (2026-07-31): the first version of this entry also claimed the
   file defined `app:decode_figures` and collided with appendix A. It does not.
   `app:decode_figures` is defined once, in `appendix_a_regularisation.tex:297`,
   and the body's pointers resolve there correctly. The file defines only
   `fig:recon` and `fig:pooling_cost`, both referenced only from inside itself.
   The defect is the name and nothing else.
3. `APA-03` and `S35-02` disagree on the number of uncertainty levels, three
   against four. `S3-12` states it a third time. One of the three has to go and
   the other two have to agree.

**Two bugs found and fixed while building it, both worth recording.**

- The strip step matched `\re` inside `\ref`, turning every cross-reference
  into `f{...}` across 30 files. Caught by inspecting the diff before building,
  reverted with `git checkout`, and fixed with a negative lookahead. The apply
  step now round-trips through strip and refuses to write if the result differs
  from the input.
- Colour reached the captions but not the body prose. Two causes, found by
  bisection: `\rb` runs in vertical mode at the head of a paragraph, where
  `\color` does not reach the paragraph that follows, and `\marginnote` leaves
  the surrounding text back at black. Fixed with `\leavevmode` and by
  re-asserting the colour after the note.

`scripts/session35/trace_numbers.py` gained one strip rule so that the section
references and block identifiers inside `\rb{}{}{}` are not read as manuscript
numerals. The pattern matches nothing on a branch without the overlay.

Gates with the overlay on: main rc=0 at 57 pp (56 plus the legend),
`trace_numbers` PASS, `audit_numbers` PASS (968/868, 0 mismatches), 0
em-dashes, 0 undefined references beyond the benign hyperref note. With
`\reviewmarksfalse`: 56 pp at the JFM trim, and `pdftotext` output identical to
the committed `main.pdf` on branch `solera`.

### Resolutions (2026-07-31)

**Finding 1, the orphan protocol box: deleted.** `sections/protocol_box.tex`
was input by nothing and stale on top of it, quoting headline numbers at
d = 64, horizons {4, 8, 16, 32} and the v2.1-era D130/D165 primary endpoint.
It was the only user of `\CommitHash`; that macro is emitted into `macros.tex`
from `numbers.json` and is left alone. Removed from the `trace_numbers`
content globs. Git history preserves the box.

**Finding 2, the misnamed file: renamed.**
`sections/appendix_c_supplementary_figures.tex` is now
`sections/supp_s3_figures.tex`, matching its three siblings in
`supplementary.tex` and its actual role as supplementary section S3. The
claimed label collision was not real; see the correction in the finding itself.

**Finding 3, the uncertainty protocol: one accounting, six sources.** The
disagreement was worse than a count. The two statements enumerated different
things and neither was a superset: `S3-12` and `APA-03` listed bootstrap, seed
retrains and probe cross-validation, while `S35-02` listed encoder seeds,
operator seeds, filter member seeds and the case-clustered bootstrap. Deleting
either side would have dropped a real disclosure, the probe cross-validation on
one side and the filter member seeds on the other. There are six sources, not
three or four. A second defect ran through all three sites independently of the
count: "every held-out R^2 and mean absolute error carries three independent
uncertainty signals" is false, since filter numbers carry member-noise seeds
and no probe cross-validation, and `S35-02` itself concedes single-seed cells.

Resolved as (author's choice among three options) one table in appendix A:

- New `tab:uncertainty`, six rows, each naming the source, the unit varied or
  resampled, the count, and the results that carry it. Placed with the other
  configuration tables at the end of appendix A, deliberately, so that no
  existing caption identifier in `RELEVANCE_MAP.md` shifts.
- `S3-12` deleted. Its subsection was titled "Diagnostics, controls, and
  uncertainty" and is now "Diagnostics and controls", since the heading was
  naming content that had left.
- `S35-02` keeps the protocol's own two sentences and points at the table; the
  four-level enumeration is gone, the single-seed disclosure stays.
- `APA-03` no longer re-enumerates. It now carries only what a table cannot:
  which source dominates at this sample size, the n values, why the plain
  encounter bootstrap overstates the effective sample size, and the
  community-standard defence.
- The universal quantifier is gone from all three sites: "no single number
  carries all six; which apply depends on what was varied to produce it".

Net effect on the body: one paragraph out of Methods, one enumeration
compressed to a pointer, one table into the appendix. Ledger updated
(`S35-02` and `APA-03` re-noted, `S3-12` dropped, `APA-C16` added, still 164
blocks).

Gates: main rc=0, 57 pp with the overlay on and 56 pp with `\reviewmarksfalse`,
supplementary rc=0 at 3 pp, `trace_numbers` PASS, `audit_numbers` PASS
(968/868, 0 mismatches), 0 em-dashes, 0 undefined references beyond the benign
hyperref note. The clean build no longer matches the `solera` baseline, by
design: a word-level diff shows exactly the five intended changes above and
nothing else.

## Applied 2026-08-04: PDF comments on S2-05 and S3-01..S3-04

Read back with `scripts/review/read_annotations.py`, applied, and closed with
`carry_comments.py --resolve`. Five comments, twelve requests.

**S2-05** (`section_2_flow_and_data.tex`). Dropped "We verified the additional
cases against the archive ... rather than by trusting the file metadata", which
narrated our own process rather than the data, and folded the surviving
mirror-transient evidence into one sentence. Block stays classified `S`.

**S3-01** (`section_3_methods.tex:13`). The $d = \LatentDim$ justification
asserted a bound without stating it. Now gives the arithmetic: $K$ taps over $m$
delays yield $Km$ measurements, the bound admits $d$ once $Km > 2d$, and the
production budget of $K = \FilterTaps$ over a ten-frame window admits $d$ up to
$40$, so $32$ sits just inside the ceiling. Added the schematic pointer
(figure~\ref{fig:method}a) and the configuration pointer
(table~\ref{tab:architecture}).

**S3-02**. Said outright that the training predictor is a recursive single-step
model and not a sequence-to-sequence one, and named the direct forecaster as the
one-shot alternative that cannot compound. Gave the estimator a self-contained
gloss at first mention ("an ensemble Kalman filter that advances a cloud of
candidate states through a prediction model and corrects them against sparse
wall-pressure measurements") while keeping the forward reference.

**S3-03**. The two loss terms are now two explained terms rather than one
compound clause. Added why collapse happens at all: a constant encoder drives
the rollout error to zero while carrying no information, so it is an attractor
of the objective, and the regulariser bites because a constant has zero variance
along every projection.

**S3-04**. Rewrote the opening, added the motivation (predictability does not
fix what the state *contains*), and enumerated the three heads (i)-(iii).

**Appendix links were broken, and this was the one real defect.** Every
`\ref{app:...}` resolved to the anchor `section.1/2/3`, already owned by
sections 1, 2 and 3, so all three appendix links jumped into the body. hyperref
patches `\appendix` for exactly this; the JFM class uses a `\begin{appen}`
environment, so the patch never fired. Fixed in `main.tex` with
`\renewcommand{\theHsection}{app.\Alph{section}}`; `\theHsubsection` is defined
in terms of it and followed. Verified in the PDF: `section.app.A/B/C` now
resolve to distinct page objects in the appendix, not to page object 75.

Two deviations from the comments as written, both deliberate:

- S3-04 asked for "map **predicted latent** variables to aerodynamic
  observables". The heads read the encoder output $\zvec_t$, not the predictor
  output, so the text says "map the latent state" and adds a sentence making the
  distinction explicit. Writing it the other way would have misdescribed the
  model.
- (superseded the same day, see below) S3-04's enumeration was first written
  inline because the displayed list rendered wrong. It is now a displayed
  enumerated list, as asked.

Two tool bugs surfaced and were fixed:

- `annotate_relevance.py` split a paragraph at `\begin{enumerate}`. TeX ends a
  paragraph at a blank line, not at `\end{itemize}`, so prose resuming after a
  displayed list is still that paragraph. The split renumbered every later block
  in the file and silently misaligned the ledger rationales by one. `LIST_ENVS`
  is now spanned the way `MATH_ENVS` already was.
- `carry_comments.py --restore` re-added every resolved comment on each rebuild:
  `key()` compared raw contents, but a resolved comment is written with the
  `[done] ` prefix and stored without it, so it never matched. S1-08 had
  accumulated three copies. `key()` now normalises the prefix; verified
  idempotent over three consecutive restores.

Gates: main rc=0 at 57 pp, supplementary rc=0 at 3 pp, `audit_numbers` PASS
(968/868, 0 mismatches), `trace_numbers` PASS, 0 em-dashes, 1 undefined (the
benign hyperref `\thepage` note), ledger 164 blocks / 0 unclassified / no ID
churn.

### Correction, 2026-08-04: the displayed list

Carlos clarified that S3-04 wanted a real enumerated list, so substituting an
inline (i)/(ii)/(iii) was wrong. The class defect was real but I had misdiagnosed
the knob. `\enummax` (`JFM-FLM_Au.cls` l.992) measures the label and sets
`\leftmargini` to exactly that width, so `\labelwidth` comes out *narrower* than
the label; TeX sets an oversized label flush and drops the separation, which is
why neither `\labelsep` nor `\labelsepi` nor `\item[...]` overrides changed
anything. `\listtextleftmargin` (l.1003) is the class's own override for
`\leftmargini`; at 1.6em the label has room. Set locally, so it is scoped by the
`\rb` group and no other list in the document is affected.

That change exposed a latent bug in `scripts/session35/trace_numbers.py`:
`\setlength` and `\addtolength` take two arguments, but the strip rule matched
only one, so the register name was removed and the length `1.6em` was left behind
and reported as a hand-typed manuscript number. No existing call used a decimal
length, so it had never fired. Given its own two-argument rule.

## Applied 2026-08-04, second round: S3-04, S3-05, S31-01

**S3-04, the wake target.** The descriptor was described only by what it was
made of, not by what its numbers are. Now stated concretely and checked against
`src/data/wake_observables.py`: $80$ numbers, $64$ patch energies on an
$8 \times 4$ grid taken separately for positive and negative $\omegaz$, plus a
$16$-bin radially-averaged power spectrum. The five-citation provenance stays
but is attached to the part of the construction each citation justifies rather
than run together in one sentence.

**S3-04, the closing sentence.** "The head-supervision axis is no longer a
confound" was an internal-history statement, and "gates fixed in advance of
evaluation" is stated at the point of use in \S4.1.1 anyway. Rewritten to say
what the cube is and what it buys: the heads switch independently, so
\S4.1.1 trains all eight on-off combinations and each head's contribution is
read off the cube rather than inferred from one configuration.

**S3-05.** Five points, four of which are stated where they bind (byte-matched
targets and shared head class in S3-04, identical heads on the reconstructive
anchors in S3-11 and table 12, the probe rule in \S3.5). Cut to the one point
that is not stated elsewhere, the POD asymmetry, plus a pointer to the full
control set. Reclassified `A` -> `K`: at three sentences there is nothing left
to demote.

**S31-01.** The near-body head was the only one of the three with its own
`\subsubsection`, and carried two equations, a discretisation, a residual, a
band definition, a quality gate and a proxy control while the other two got a
clause each. Levelling up was the wrong direction for a paper already 19.4k
words against a 12.5k budget, so the construction moved to appendix A, into the
`app:nearbody` subsection that already holds this head's own result (APA-08) and
its figure (fig:phi). The body keeps what the observable is and why it is not
the wake target, at the same depth as the other two heads. Heading removed;
`\label{sec:methods_chang}` stays in \S3.1 so the three body references still
resolve, and the enumerate item now points at appendix~\ref{app:nearbody} for
the construction instead of at its own subsection. Reclassified `A` -> `K`.

### Gate coverage gap found while doing this (not fixed here)

Moving the construction into `appendix_a_regularisation.tex` made the tracer
fail on the $\phi_L$ Laplacian residual $6.4 \times 10^{-13}$. The number had
not changed; it had moved into scanned territory. `CONTENT_GLOBS` in
`scripts/session35/trace_numbers.py` covers `sections/*.tex`,
`sections/tables/*.tex` and `sections/captions/*.tex` but **not**
`sections/v4/*.tex`, which holds four Methods subsections and five Results
subsections, roughly a third of the manuscript's prose. That prose has never
been number-audited.

Measured: scanning `sections/v4/*.tex` yields 129 raw hits, 8 after the existing
whitelist. Of those 8:

- 4 are false positives from decimal subscripts. `SUPERSUB` is
  `[\^_]\{?\d+\}?`, which does not handle a decimal, so `q_{0.9}` leaves `.9}`
  behind (`s3_4_estimators.tex:132,135`).
- 1 is a false positive from a LaTeX optional argument, `\\[1.2ex]`
  (`s4_c_prediction.tex:61`).
- 3 are genuine: the Fukami $d = 16$ lucky-seed values $0.18$, $0.65$, $5.93$
  are hand-typed in `s4_d_assimilation.tex:111` and are result values, so they
  should be macro-bound.

Not fixed in this commit, because closing it means a numbers-pipeline change
(three new macros through `emit_numbers_parts.py` -> `numbers.json` ->
`macros_v3.tex`) plus two tracer regex fixes, and expanding `CONTENT_GLOBS`
before those land would fail the gate. The residual itself is whitelisted with
justification: it is a solver verification constant, not a pipeline result.

Gates: main rc=0 at 57 pp, supplementary rc=0 at 3 pp, `audit_numbers` PASS,
`trace_numbers` PASS, 0 em-dashes, 1 benign hyperref note, ledger 165 blocks / 0
unclassified. Appendix A block IDs shifted by one (APA-07..APA-09 ->
APA-08..APA-10) because a block was inserted mid-appendix; the four affected
ledger rows were realigned by hand and verified against the preview.

## 2026-08-11: comment mis-attribution on figures (tooling)

Carlos's figure-5 comment was reported under S3-C1, the caption of figure 4, and
I acted on figure 4 before he corrected me. The cause is structural, not a
one-off: a float prints its artwork **above** its caption, so a comment written
on a figure sits after the preceding block's tag and before the caption's own,
and the reader's reading-order rule hands it to whatever text precedes the
float.

Geometry cannot settle it. A comment on the last line of that preceding
paragraph lands in the same gap, and a vector figure carries real text of its
own, so "are there words in between" separates nothing. Both candidate fixes,
prefer-the-following-caption and require-empty-space, trade one silent
misattribution for another.

So the reader now says so instead of guessing: it attaches by reading order as
before and flags the annotation with the caption it might belong to and the
command that moves it. `carry_comments.py --move FROM TO` is that command, new
here. Verified by simulating a sticky note 60 pt above the figure-5 caption tag:
resolves to S3-06, flagged for S31-C1.

Both figure-5 comments moved from S3-C1 to S31-C1 and redrawn there. Nothing in
the manuscript changed; the figure-4 edit made on the wrong reading was reverted
in the previous commit and its artwork is byte-identical to what was committed.

## Applied 2026-08-11, second round: S3-06 and S31-C1, plus two tooling fixes

**S3-06, the legacy operator.** The body carried a full description of the
superseded suited-operator protocol, and appendix~A carried a longer one. Both
are gone. \S3.2 now says only that a superseded protocol fitted a different
operator to each family and points at the supplementary; appendix~A keeps the
shared operator and the filter's use of the matched transformer. Supplementary
S4 absorbed the stability split that justified the protocol, the transformer
diverging on the reference latents while the U-Net is stable on everything, and
opens by stating it is the self-contained home for the topic. This is the
disposition the ledger already carried for APA-01 (`T`, "spends twenty lines
narrating a suited-operator protocol the paper no longer uses").

**S3-06, the closing sentences.** Both read as defensive without saying what was
done. Rewritten: the two choices are introduced as choices that could be read as
favouring our own model, the strict-published variant is stated as something we
trained rather than something we "found", "tuned baseline, not a hobbled one"
becomes "the comparison here is against the stronger of the two", and the
head-form sentence says what the shared form actually is, prediction at the
current frame.

**S31-C1.** "Lower branch" was simply wrong: the trained row is lower still, so
the geometry branch is the middle one, and it is the only grey one. Relabelled.
The $16$-bin radial power spectrum is now defined where it is used: the
two-dimensional power spectrum averaged over rings of constant wavenumber
magnitude, binned from the largest scales to the smallest, so each bin reports
energy at a scale irrespective of direction, with one sentence saying what the
two blocks of the target contribute (where the structure is, at which scales).

### Two tooling defects, both mine

**Icons drew on top of each other.** Every comment on a block got the identical
rectangle, and poppler draws a fixed-size icon at the rect origin, so the second
comment on a block was invisible underneath the first. Carlos lost comments to
this. They now stack down the empty margin at `ICON + 3pt`, and the per-block
counter is seeded from what is already in the file so a partial restore still
lands in the next free slot. Verified: the three blocks carrying two comments
each now report distinct icon rectangles.

**`--move` was all-or-nothing.** Correcting the figure-4/figure-5 mix-up moved
both comments off S3-C1 when only one belonged to figure 5. `--move` now takes
`--match TEXT`, a case-insensitive substring, so a single comment can be moved
off a block that collected several. The "separate (a) and (b)" comment is back
on S3-C1 where it belongs.

## Applied 2026-08-11, third round: S33-01 (three) and S3-07

**S33-01, the context window.** Carlos read "$L$ drawn uniformly from $16$ to
$30$ during training, fixed at $L = 25$ at evaluation" and had to ask whether $L$
is the lookback length. It is, and the sentence now says so before giving the
numbers, with the reason for randomising it (so the operator is not tuned to one
context length).

**S33-01, the companion citation.** arXiv:2607.24569 is now cited where the
one-shot design is justified. The bib already held this paper as
`solerarico_compactness_underreview`, an `@unpublished` marked "under review, no
arXiv index" per an earlier instruction, dated 2025. That instruction is
superseded: it was posted 2026-07-27. Entry updated to `@article` with the arXiv
ID and the correct year, title and author list confirmed against the abstract
page; the key is unchanged so the two existing call sites in \S1 and \S5 do not
move. The claim is scoped to what the paper actually shows, that latent forecasts
in actuated wake flows degrade with horizon and can diverge catastrophically. It
does not itself compare direct against autoregressive heads, so the text says
emitting the horizon at once removes compounding as *one route* to that failure
rather than attributing the finding to the citation.

**S33-01, the pinball loss.** Now referenced (Koenker \& Bassett 1978, added to
the bib) and motivated: $\rho_q$ is minimised in expectation by the $q$th
quantile, so the nine outputs are calibrated quantiles rather than an arbitrary
band, which is what lets the inter-quantile width be read as the filter's
state-dependent process noise. The sLSTM comparison is gone from the body, per
Carlos's point that citing the alternative is noise when the adopted model is a
plain LSTM; supplementary S1 still records it, which is where a selection
catalogue belongs.

**S3-07.** "We do not headline a forecast-$R^2$-versus-horizon curve;" removed.
The following clause already says what the rollout is for.

### Pre-existing defect surfaced: the supplementary had no bibliography

`supplementary.tex` cites work of its own but carried no `\bibliography`, so
`\citep{xlstm}` rendered as a literal "(?)" in the built PDF, visible on page 1.
This predates this session's work (the file has not changed since `fe68b4c`) and
was missed because the supplementary gate only ever checked the return code and
the page count, never the log for undefined citations. Fixed with the same style
and `.bib` files as the main document: 0 undefined citations, still 3 pages, and
the reference now renders as Beck et al. 2024 with a reference list.

Worth adding to the standing gate: `grep -c "Citation.*undefined" supplementary.log`.

## 2026-08-13: S34-01, S34-02, S34-C1 (the estimator subsection)

**Terminology: "taps" becomes "sensors" throughout.** Carlos raised it on S34-01
but the preference is global, and a partial rename would have been worse than
none: 90 occurrences across fourteen files, all of them meaning the same thing.
"Tap" is standard experimental-aerodynamics vocabulary for a physical pressure
port, but there is no experiment here, only sampled surface locations in a DNS,
and the paper already used "sensor" for the same object in "sensor count",
"sensor placement" and "sparse-sensor". The rename removes that split. The
`\FilterTaps` macro name is untouched, since it is internal and renaming it
would churn the numbers pipeline for no reader-visible gain.

Three body figures carry the old word in their axis labels and titles
(`fig_deployment_v4`, `fig_ownstack_da_v4`, `fig_t_trade`). Their generating
scripts are patched, but the figures **could not be regenerated**: the result
JSONs they read are absent from this checkout. See FIGURE_PLAN.md; this is
Carlos-owned and blocks nothing else.

**S34-01.** Reworded to an academic register. What "up to a diffeomorphism"
means is now stated rather than assumed: the delay vector and the true state are
related by a smooth, smoothly invertible change of coordinates, so the
reconstruction preserves the dynamics and every smooth function of the state but
does not return the state in its original coordinates, which is why a learned
map is still needed. The two closing claims are separated and each given its
mechanism: temporal resolution is exchangeable for spatial resolution, and the
wake circulation is recovered because it is dynamically coupled to the
near-surface flow the sensors resolve and so leaves a signature in the pressure
history.

**S34-02.** The $f$ and $a$ superscripts were used in equation (5) and never
defined; they now are, as forecast and analysis, in the sentence following the
equation pair. The time base is stated as an interval between successive frames
rather than as a parenthetical "time base", and is macro-bound (`\DtTc`), as are
the assimilation bounds (`\AssimPre`, `\AssimPost`), which were hand-typed. The
two protocols are renamed from "evaluation frames", which collided with "frame"
as the time index everywhere else in the paper, to "assimilation protocols", and
each is given in full. The unit test is dropped from the leakage sentence; the
guarantee now rests on the construction alone, which is what a referee can check.

**S34-C1, the figure.** Carlos was right that there was an error. The observation
model maps the *forecast* to a *predicted* observation, and the innovation is
that prediction subtracted from the measurement; the old drawing hung the label
$y - \mathcal{H}z^f$ on the incoming measurement arrow with no node where the
subtraction happened, so the measurement read as though it arrived carrying the
innovation. The difference is now an explicit junction: predicted observation in
from above, measurement in from below, innovation out to the left into the
update. Moving the measurement below rather than to the right freed the whole
right-hand side for the predicted-observation label and made the figure narrower,
so at `\linewidth` the scale rose from $0.70$ to $0.87$ and every glyph is
larger than before. The state boxes are tagged "analysis" and "forecast", which
answers the S34-02 superscript question from the artwork as well as the prose.

**S34-C1, the caption.** $\FilterMembers = 64$ verified against three independent
places: the `enkf.py` signature default, the argparse defaults of both driver
scripts, and the numbers pipeline entry. The caption is now split into Predict
and Correct and walks the loop in order, including what the Kalman step actually
does with the innovation. "at inflation $\rho = 1$" is gone: it was unexplained
and also wrong for one family, since §3.4.1 sets $\rho = 1.05$ for the
reconstructive stack. The caption points at the table for it instead.

## 2026-08-13, second batch: twelve comments on Methods 3.4, 3.5 and the diagnostics

Sequenced deliberately: every content edit and every resolve happened first,
while the block identifiers still meant what they meant when Carlos wrote the
comments, and the three structural changes (one delete, two moves) went last in
one pass. Three of the twelve remove a block, and each removal renumbers every
later block in its file, so the reverse order would have closed comments against
paragraphs they were not written about.

**S34-C1, the figure.** The "predicted observation" label sat outside the dashed
Correct box because `fit=` does not enumerate anonymous nodes attached to a
`\draw`; naming the node and adding it to the fit list is the whole fix. The
production-configuration note is out of the artwork and into the caption, where
it was already half-stated.

**S34-03, now S34-04 to S34-06.** It opened "The frozen production filter",
which contradicts every other use of that phrase in the paper: §4.7 explicitly
separates the base diagnostic filter from the production estimator, and the
production estimator is the two-stage one. Now "the base filter ... the reference
configuration against which the others are read, not the production estimator".
The equations were correct but written in the textbook $P^{f}H^{\mathsf{T}}(HP^{f}
H^{\mathsf{T}} + R)^{-1}$ form, which implies a $d \times d$ forecast covariance
is formed; `src/estimation/enkf.py` forms ensemble cross-covariances instead and
never builds it. Rewritten in the cross-covariance form actually implemented,
with a note that the two agree for the linear $h$ used here, plus a
plain-language reading of what the gain does and why the observation is perturbed
per member. The perturbed-observation form is credited to \citet{burgers1998},
which the code's own docstring already names; new bib entry. The four-fifths
split is justified: it is by whole encounter, and a frame-wise split would put
near-copies of the fitted data in the held-out set and return an $R$ far smaller
than the error the filter meets. $\rho$ is explained where it is used.

**S34-04, now S34-07.** "Penalty 1.0" appeared three times across the section
and all three are gone from the prose, stated once instead in the
`tab:filter_params` caption, which already collects what is common to every row.
The "hard analogue of their spectral hinge penalty" is replaced by what the
projection does and why: a singular value above one is a direction the operator
amplifies every step. The observation-rate protocol is deleted outright, not
trimmed: it was defined in Methods and referred to nowhere in the paper.

**S34-05, now S34-08 and S34-09.** $1.28155$ is $\Phi^{-1}(0.9)$, verified
against `scipy` and against `BAND_TO_SIGMA = 2.5631` in both filter drivers; the
text now says so, so the reader can check the conversion rather than trust it.
NIS is given as a numbered equation with each term named and the meaning of the
expected value of one spelled out in both directions, overdispersed below and
overconfident above.

**S3-08 deleted** (Carlos: "delete"), which agrees with its own class D.

**S34-06 to appendix C, S34-07 to appendix B.** The smoother recursion sits in
appendix C beside the figure that already carries the smoother comparison; the
placement criterion sits in appendix B immediately after APB-02, which argues
that placement does not drive the ordering. Both keep a one-sentence form in the
body, and the labels moved with the content: `sec:methods_rts` and
`sec:methods_osp` are gone and every call site now points at `app:rts` or
`app:osp`. One of those call sites was written earlier in this same session.

**S35-01.** SSE and SST defined, and used to say what $R^2$ measures and where it
misleads, which is the reason the paper reports physical error beside it. The
SSIM constants are dropped here and pointed at §2, where they were already given
in full; the block's own ledger rationale had flagged that duplication.

**S35-02.** The six uncertainty sources are enumerated in prose with the reason
each is separated. The table stays. Carlos suggested it might not be needed, but
it is referenced twice from appendix A and it is the artefact that resolved three
mutually inconsistent statements of the same protocol (see the source comment
above `tab:uncertainty`); dropping it would reopen that. Prose enumeration plus
the table as the authoritative accounting is the reading of his comment that
keeps both. **Carlos to overrule if he wants the table gone.**

**Net word delta: body +768, appendices +477.** Six of the twelve comments asked
for clarification or justification and four asked for cuts, so this batch moves
the wrong way against the length finding. The two appendix moves took roughly 380
words out of the body and the clarifications put more back. Worth stating
plainly, because the manuscript is already well over its own budget.

### Tooling note

The block parser ends a block at a display equation unless the text after
`\end{equation}` begins lowercase. Writing "In words: ..." after the EnKF gain
split S34-03 in two and silently renumbered everything after it, which the
`--inventory` count caught. Changed to "where ...", which is the correct
typography after an equation anyway. Ledger realignment after the structural pass
was done by matching each block's opening text against the previous ledger rather
than by identifier, since identifiers are positional.
