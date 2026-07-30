# F01: two drafts for the abstract and §6 headline number

**RESOLVED 2026-07-30: option B applied**, with the two compensating trims
recorded in `REVIEW_LOG.md` (the option B wording below runs the abstract to 259
words against a 250 limit, so it landed in the tighter form now in
`sections/abstract.tex`). Option A is kept here as the record of what a boundary
headline would have cost.

## The problem, restated

`section_4_results.tex` (envelope subsection) says:

> the accuracy that matters at deployment is the error in real lift units, not
> the variance-normalised score [...] the strong gust inflates the lift
> excursion faster than the error grows, so the normalised score flatters the
> extreme-gust tracking while the absolute error roughly doubles.

The abstract's only result number is that flattered score, at the boundary,
with no lift-unit error: $R^2 = 0.837$. §6 quotes 0.794 (test_b) and 0.837
(boundary), neither with an error. §4 does pair the test_b value with its RMSE
(0.286).

## Availability constraint, checked

There is **no boundary RMSE anywhere in the numbers pipeline**.
`outputs/session33/numbers_parts/p1_bands.json` carries
`p1_two_stage_test_b_b177_rmse_imp` but no `test_c` counterpart, and
`numbers.json` has no such entry. So:

- **Option A needs a pipeline regeneration** (emit the boundary impact RMSE
  from the same frozen two-stage run, then re-emit macros). Not a prose edit.
- **Option B needs no new number.**

## Option A: keep the boundary result, pair it with its error

Requires the new macro `\PoneTwoStageCovRmseC`.

Abstract, replacing the current final result clause:

> ... the filter's advantage is in the tails, in continuous state tracking, and
> at the held-out $|G| = 4$ boundary, where it reaches impact-phase
> $R^2 = \PoneTwoStageCovImpactC$ at a lift-unit error of
> \PoneTwoStageCovRmseC{} with the final two-stage configuration. The
> normalised score rises with gust strength because the lift excursion grows
> faster than the error does.

§6, replacing the estimation sentence:

> On estimation, a two-stage ensemble filter sensing eight wall-pressure taps
> tracks the lift through the encounter, phase-resolved and in physical units,
> with an impact-phase median $R^2 = \PoneTwoStageCovImpact$ (lift-unit error
> \PoneTwoStageCovRmse) on the \TestSplit{} set and
> $R^2 = \PoneTwoStageCovImpactC$ (\PoneTwoStageCovRmseC) at the $|G| = 4$
> boundary, where the higher score reflects the larger lift excursion rather
> than better tracking.

Cost: one regeneration run. Benefit: keeps the boundary result as the headline,
which is the paper's strongest claim, and defuses the objection in the same
breath.

## Option B: lead with the in-distribution pair

No new macro. Uses `\PoneTwoStageCovImpact` and `\PoneTwoStageCovRmse`, both
already bound.

Abstract, replacing the current final result clause:

> ... the filter's advantage is in the tails, in continuous state tracking, and
> at the held-out $|G| = 4$ boundary. With the final two-stage configuration it
> reaches impact-phase $R^2 = \PoneTwoStageCovImpact$ at a lift-unit error of
> \PoneTwoStageCovRmse, and holds its median tracking out to the boundary.

§6, replacing the estimation sentence:

> On estimation, a two-stage ensemble filter sensing eight wall-pressure taps
> tracks the lift through the encounter, phase-resolved and in physical units,
> with an impact-phase median $R^2 = \PoneTwoStageCovImpact$ at a lift-unit
> error of \PoneTwoStageCovRmse{} on the \TestSplit{} set, and holds its median
> tracking at the $|G| = 4$ boundary.

Cost: none. The boundary number stops being a headline and stays where it is
already reported in §4.6 and §5. Benefit: the abstract's number is the one the
paper defends with a physical error, and the counterintuitive
0.837-beats-0.794 reading disappears entirely.

## Recommendation

B if you want this closed today with no compute; A if you want to keep the
boundary result in the abstract, which is the stronger scientific claim and
worth the regeneration.
