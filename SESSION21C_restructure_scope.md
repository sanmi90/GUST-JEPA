# Session 21C: restructure, scope, final polish (Path A, JFM Standard Article)

Paste into Claude Code with the repo open, AFTER 21A and 21B. Editing and
recompiling only. Path A keeps the deployment material but compressed; it does not
cut it entirely (that would be the Rapid path).

-----

## 1. Consolidate the closure tables  (Tier 3.2)

- Merge Table 2 (representational MAE) and Table 3 (forecast R2) into ONE table
  with both modes side by side, so the reader sees representation-vs-forecast in a
  single exhibit. Keep the paired-difference/sign-test column added in 21A on the
  wake observables.
- Move Table 4 (training-set fit, explicitly “for reference only”) to the
  appendix.
- Net: table count drops from seven to about five.

-----

## 2. Compress the deployment / closed-loop material  (Tier 4, Path A choice)

Keep exactly ONE deployment figure and ONE paragraph in the Discussion; move the
rest to a short appendix or cut.

- KEEP: Fig 14 (pressure-to-state recoverability across families), because it
  reinforces the main result that the predictive latent is more recoverable than
  POD. Keep one Discussion paragraph: the predictor is the action-conditioned
  world model a controller would use, the latent is more pressure-recoverable than
  the energy-optimal POD, and a closed-loop pilot does not yet meet its tolerances
  with the rollout (not the estimator) as the bottleneck. State the gap honestly
  in one or two sentences; do not expand it.
- CUT from the main text: the sensor-selection figure (current Fig 13) and the
  lead-time figure (current Fig 15). Either drop, or compress both into the short
  appendix.
- TRIM: the closed-loop tolerance discussion to a few sentences. Remove the
  pre-registered-gate bookkeeping from the main text.
- Because Fig 13 (sensor selection / qDEIM) is cut, you do NOT need the Drmac &
  Gugercin citation; keep Manohar et al. for the one sparse-sensing sentence if
  the paragraph mentions placement.

Rationale to keep in mind while trimming: the control story is the authors’
broader programme (a companion MPC paper is under review), but a pilot that fails
its own gates cannot strengthen this paper; it is compressed here and developed in
the follow-up. Do not frame the deployment material as “preliminary” in a way that
weakens the standalone contribution.

-----

## 3. Resolve Section 4.4 “Where the error concentrates”  (Tier 3.3)

JFM papers do not include a subsection describing an analysis that was not done.
Two acceptable outcomes:

- PREFERRED: if the optional “closure vs gust regime” panel was built in 21B
  (closure vs signed G from the per-encounter errors and (G,D,Y) labels), then
  Section 4.4 reports it as a real result; delete the “left to a study with more
  cases per cell” hedge.
- ELSE: cut the not-done paragraph to a single sentence noting that the per-axis
  probe already localises the expected residual (weak Y) and leaving the full
  per-stratum table to future work.

-----

## 4. Sentence-length and readability pass  (Tier 3.5)

Split any sentence running more than about three printed lines. Target example:
the Section 5.1 sentence beginning “The predictive objective, regularised against
collapse, instead organises…”. The reference papers rarely stack more than two
subordinate clauses. This is a global light pass, not a rewrite.

-----

## 5. Final internal-vocabulary and house-style sweep

- Re-grep the whole .tex for: “Track”, “A1”..“A5”, “D-i”, “D-ii”, “B1”, “Session”,
  “original draft”, “original analysis”, “pending”, “TODO”. Zero reader-facing
  occurrences should remain (except a TODO you deliberately left for the authors,
  e.g. the resolution numbers in Section 2.2, which must be filled before
  submission).
- Em-dash sweep: replace with commas/colons/parentheses per house preference.
- Confirm the abstract is one paragraph, about 230-270 words, ends on significance,
  and uses the DNS/SOD2D wording (not LES, not “simulations of Fukami”).

-----

## 6. Recompile and final targets

- Compile clean: no undefined refs, no “??”, no leftover placeholders.
- Main-text figures about 8; tables about 5; the deployment material is one figure
  plus one paragraph.
- The core result (forward closure + drift/topology/transport mechanism + the 2x2
  - the d=32 minimal-state reading) should read as the spine, with the deployment
    as a short outlook.
- Sanity check every figure caption against the style guide: declarative, carries
  the takeaway, no internal labels.