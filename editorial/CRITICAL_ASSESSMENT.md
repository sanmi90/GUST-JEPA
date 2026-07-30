# Critical assessment of main.tex

2026-07-27, branch `solera` at `52cc5a1`. Findings ledger: `REVIEW_LOG.md`.

## Verdict

The manuscript is mechanically submission-ready and scientifically sound. Gates
are green, every primary claim resolves to an exhibit, the language discipline
holds, and the honesty register is unusually good for a paper of this kind:
negative results are reported, a test-peeked variant is quarantined in an
appendix and named as such, and the estimator is called a feasibility
demonstration rather than a deployed system.

Three things stand between it and a clean first-round review, and only the
first is urgent.

**One. The abstract's single number is the one the paper's own Results section
warns against.** §4.6.2 states that the variance-normalised score "flatters the
extreme-gust tracking while the absolute error roughly doubles". The abstract's
only result number is that flattered score at the extreme-gust boundary, quoted
without its lift-unit error, and it is numerically higher (0.837) than the
in-distribution result (0.794) which *is* reported with an RMSE in §4. A referee
who reads the abstract and then §4.6.2 will conclude the abstract is oversold,
and will be technically correct. Project decision D252 already requires physical
error alongside R^2 for exactly this reason. This is a two-sentence repair and it
should happen before anything else.

**Two. The paper is long in a way that is fixable without argument loss.** The
body is 19,404 words against an internal budget of ~12,500 that has never been
re-baselined since D268 relaxed it. Roughly 700 words are literal duplication:
the POD-does-not-reach-the-wake finding is stated at five sites, the
field-decode-does-not-separate conclusion at four, the dimension-plateau claim
at three, the suited-operator protocol twice near-verbatim within Methods. None
of these cuts touch an equation. That matters, because the Session 37 compression
decision has been parked for months on the premise that cutting Methods meant
undoing the Gupta MC contract by moving filter equations to an appendix. That
premise is false: ~340 words of Methods are duplication.

**Three. An abstract-level claim is evidenced only in an appendix.** The
abstract's static-inverse qualifier, the one that scopes the whole estimation
result, rests on figure 30 of 30, in an appendix titled "Calibration
disclosure". The appendix title also under-describes its contents: it carries
the estimator ladder and the smoother comparison, not just the disclosure.

## What the read confirmed and what it dismissed

Confirmed: over-length; mass concentrated in §3 and §4 (two thirds of the body);
the load-bearing appendix; author-facing scaffolding surviving in Table 1.

Dismissed: that the Discussion pads by restating results (it shares three
result macros with §4 and restates conclusions, not numbers, so it is not a cut
target); that the aliased float labels break cross-references (the prose
disambiguates); that the gates have drifted.

## Ranked fix list

### Blocker

1. **F01 RESOLVED 2026-07-30, option B.** Abstract and §6 now lead with the
   in-distribution pair (`\PoneTwoStageCovImpact` with `\PoneTwoStageCovRmse`);
   the boundary result stays qualitative in both and keeps its number in §4.6
   and §5. No new macro, no regeneration. Abstract 249/250 words.

### Major, scientific or scope (yours)

2. **F02/F24** Decide whether the estimator ladder is promoted into §4.6 or the
   abstract's static-inverse qualifier is rescoped. Retitle appendix C either way.
3. **F15** Decide whether the base frozen filter's envelope diagnosis stays in
   the body. It currently costs three disambiguation notes.
4. **F16** "By far the fewest consistency failures" compares 72% against 93%.
   Every filter fails on most boundary encounters. Rescope or drop "by far".
5. **F18** `solerarico_compactness_underreview` carries §1's motivating tension
   and §5.1's caveat. Confirm its status before submission.
6. **F19** §1 is 982 words and states its contributions as prose; the enumerated
   (i)-(iv) list CLAIM_MAP records is gone. Restore or accept.
7. **F22** Six items open since the 2026-07-18 audit remain untouched, including
   the §3.5 sign-convention audit and three missing-band/missing-macro items.
8. **F23** The Session 37 Methods compression decision, now unblocked.

### Mechanical (mine, on approval)

9. **F03** Retitle Table 1's third column and drop the "before submission"
   clause from its caption. This is review item #16, which BUILD.md queued to run
   once the rows were filled; they were filled on 2026-07-24.
10. **F09-F14** Remove the duplicated statements: POD/wake (5 sites to 2),
    field-decode (4 to 2), dimension plateau (3 to 1), the §4.2 decodability
    preview, the figure-11 narration, the one-line forward pointer. ~330 words.
11. **F04-F08** Methods duplication: the two-network restatement, the
    suited-operator second description, the fairness/controls overlap, the
    figure-pointer thicket. ~340 words, no equation touched.
12. **F17** Scope the §5.2 "beyond every static inverse" against §6's "matches".
13. **F20/F21** Section-reference style; panel-lettering consistency in
    figures 7 and 11.

## Reaching a target

F09-F14 and F04-F08 together recover roughly 700 words, taking the body to
about 18,700. Getting to 13-15k is not a copy-editing problem: it needs the
structural calls in F15 (one filter in the body, not two), F02 (where the
estimator ladder lives) and F23 (whether §3.4's per-estimator equations stay).
Those are yours, and none of them is forced.

## State of record

Gates at assessment time: `trace_numbers` PASS, `audit_numbers` PASS (968
macros, 868 json entries, 0 mismatches), 0 em-dashes, 57 pp, no undefined
references beyond the benign hyperref `\thepage` note, 28 overfull hboxes all
at 1.61 pt.

Standing submission blockers, unchanged: Table 1's convergence row (the last
`\pending{}`), Zenodo DOI, licence/CRediT/funding.
