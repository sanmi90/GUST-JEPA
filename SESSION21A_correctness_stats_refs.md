# Session 21A: correctness, statistics, references (Path A, JFM Standard Article)

Paste this into Claude Code with the manuscript repo open. All post-processing or
editing; no retraining. Do these in order. Items marked [EXACT] give verbatim
replacement text. Recompile at the end and confirm no broken refs.

Read `SESSION21_STYLE_GUIDE.md` only if you touch captions; otherwise this session
is text and numbers.

-----

## 1. Data provenance and DNS/solver correctness  (Tier 0.1, REVISED)

The reviewer assumed the data were Fukami/CharLES LES. They are NOT. The data are
the authors’ own DNS computed with the SOD2D spectral-element solver, reproducing
the Fukami, Smith & Taira (2025) configuration. Two problems in the current text:
(a) it misattributes the data to ref [6] (“the governing data are the
spanwise-periodic simulations of Fukami, Smith & Taira [6]”), and (b) it never
names the solver.

Fixes:

- Demote ref [6] from data source to configuration/characterisation source
  everywhere it is cited as the data origin.
- Name the solver. Add the Gasparino et al. 2024 citation (in refs_to_add.bib).
- Keep “DNS” (the authors confirm a fully resolved DNS), and add the resolution
  evidence a JFM referee will expect, since SOD2D is a scale-resolving code used
  for both LES and DNS and stabilises with entropy viscosity. Search the repo
  (sim config, mesh files, any sim README/HANDOFF) for: polynomial order, element
  count, spanwise extent, grid spacing vs the local small scale, Mach number. If
  present, fill the bracketed sentence below; if not present, leave the bracket as
  a visible TODO for the authors to fill before submission. DO NOT invent numbers.

[EXACT] Replace the data-provenance sentence in Section 2.2 with:
“The flow configuration follows Fukami, Smith & Taira (2025): a NACA 0012 at
angle of attack $14^\circ$ and chord-based Reynolds number $5000$, perturbed by
a spanwise-oriented discrete Taylor vortex on a spanwise-periodic domain. The
data analysed here are our own direct numerical simulations of this
configuration, computed with the GPU-enabled spectral-element solver SOD2D
(Gasparino, Spiga & Lehmkuhl 2024) at low Mach number, so the flow is
effectively incompressible. [Resolution: polynomial order $p=_$, $_$ elements,
spanwise extent $_c$, with grid spacing $_$ relative to the local small scale;
TODO fill from sim metadata.] Reference [Fukami, Smith & Taira 2025] provides
the configuration and its physical characterisation, not the data.”

- Global sweep: change “DNS” wording stays correct, but check the abstract,
  Section 2.1, and Section 2.2 do not call the data “simulations of Fukami…” or
  imply CharLES. Also fix CLAUDE.md and HANDOFF.md in the repo if they say the data
  came from Fukami or are LES (hygiene; a future reader of the repo should not be
  misled).

-----

## 2. Remove every “original draft” / “original analysis”  (Tier 0.2)

Three occurrences. [EXACT] replacements:

- Page 7, “The original draft reported only a training-set R2 (Table 4 …), which
  measures in-distribution fit.” ->
  “The training-set fit (Table 4, mean 0.835 at $d=64$) measures in-distribution
  accuracy, not generalisation; the held-out forecast is markedly lower in
  absolute terms, as it must be, but it is where the families separate.”
- Table 6 caption, delete the clause “the original draft compared this floor
  against JEPA’s training-set fit.” The caption then states only that the last
  column is JEPA’s held-out forecast R2, the like-for-like comparison.
- Section 4.1, “two held-out questions that the original analysis conflated” ->
  “two held-out questions: how well the representation carries each observable
  (Table 2), and how well the predictor forecasts it (Table 3).”

-----

## 3. Fix the bibliography placeholder  (Tier 0.3)

Reference [13] prints “verify volume/pages.” The intended paper (urban-canyon /
building-wake motivation) is almost certainly:
Mohamed, A., Marino, M., Watkins, S., Jaworski, J. & Jones, A. 2023 Gusts
encountered by flying vehicles in proximity to buildings. Drones 7(1), 22.
This entry is in refs_to_add.bib. Replace [13] with it and delete the note.
If the authors specifically intended a different “gust-aware flight” paper, they
should confirm the exact handle; do not keep the placeholder.

-----

## 4. Add Key words  (Tier 0.4)

[EXACT] After the abstract, add:
“Key words: vortex shedding, low-dimensional models, machine learning”

-----

## 5. Strip internal vocabulary from prose, captions, and the 2x2  (Tier 3.6)

- grep the .tex for: “Track”, “A1”, “A2”, “A3”, “A4”, “A5”, “D-i”, “D-ii”, “B1”,
  “Session”, “session20”, “D13”. Remove or relabel all reader-facing occurrences.
- In Section 4.5 and Table 7, relabel the 2x2 cells descriptively: “predictive
  CNN+ViT”, “predictive CNN”, “reconstructive CNN+ViT”, “reconstructive CNN”,
  “predictive CNN+ViT, no wake head”. Never ask the reader to remember A1..A5.
- “test_a/b/c” may stay as defined split names, but introduce them once with words
  (“in-distribution test set”, “out-of-distribution |G|=4 set”).

-----

## 6. Reframe the statistics from weakness to strength  (Tier 1, highest value)

Run `scripts/session21/session21_paired_closure_stats.py` (already written; wire
its `load_per_encounter_abs_error` to the per-encounter absolute-error arrays
behind Tables 2 and 3, index-aligned across families). It prints, for all six
observables in both modes: the per-encounter paired improvement (reconstructive
minus predictive error), a 2000-resample paired bootstrap 95% CI, and a one-sided
sign test, plus a LaTeX block and the abstract headline line.

Then:

- Add the paired-difference column and sign-test to Table 3 (or as a short
  companion panel) for the wake observables. Report wake enstrophy explicitly:
  “the predictive latent has the smaller wake-enstrophy error on k of 42 held-out
  encounters (paired mean improvement [delta], 95% CI [..], one-sided sign
  p=[..]).”
- In Section 4.1, replace the concession that the per-observable separation “is
  not individually significant at this sample size” with the paired result. Keep
  the mechanistic evidence (topology p=4.4e-8) as the clincher, but stop conceding
  the primary comparison. Keep the wide MARGINAL interval reported honestly as the
  reason the PAIRED statistic is the right one (shared encounter difficulty
  cancels under pairing).
- Add once, in Section 3.4 or 4.1, the no-cherry-picking sentence: [EXACT]
  “All six observables are reported in both modes, not a selected subset, so the
  consistency of the ordering across every observable is itself part of the
  evidence.”
- Reframe the abstract sentence accordingly (see the replacement abstract in the
  review; insert the paired-significance clause and keep the DNS/SOD2D wording).

-----

## 7. Representation-vs-forecast framing sentence  (Tier 3.4, drop-in)

[EXACT] At the top of Section 4.1, before the tables are discussed:
“We distinguish two questions. Does the latent, encoded directly from a held-out
field, carry each observable (representation quality, Table 2)? And does the
predictor, rolled out from the impact frame, forecast it (forecast quality,
Table 3)? The first isolates the encoder; the second tests the encoder and the
learned dynamics together. The predictive latent leads on both, and the forecast
is the load-bearing test.”

-----

## 8. References and situating the work  (Tier 5)

Append refs_to_add.bib to the bibliography. Add these in-text citations:

- The companion-paper bridge (this is the authors’ own under-review JFM paper, and
  it strengthens the framing by positioning the predictive objective as escaping a
  documented tradeoff). [EXACT] add to the Introduction, after the JEPA paragraph:
  “Reconstruction-trained nonlinear encoders compress aggressively but yield latent
  dynamics that are harder to forecast than an energy-optimal linear basis, a
  compactness-versus-forecast tradeoff documented for controlled wake flows
  (Solera-Rico et al., under review). We show this tradeoff is a consequence of the
  reconstruction objective: a predictive latent recovers nonlinear compactness
  without surrendering forecast stability.”
  And [EXACT] one sentence in the Discussion (drift mechanism, Section 5.1):
  “The manifold departure of the reconstructive rollout is the parametric-gust
  counterpart of the catastrophic-divergence tail reported for reconstruction-based
  latent dynamics in controlled wakes (Solera-Rico et al., under review); the
  predictive objective is the common cure.”
- Constante-Amores & Graham (JFM 984, R9, 2024): cite where latent dynamics on
  manifolds is introduced (Section 1 or 5.1). CONFIRM the exact title against your
  library; the entry in the .bib has a title placeholder flagged.
- Solver: Gasparino, Spiga & Lehmkuhl 2024 (Section 2.2, done in step 1).
- If the deployment/pressure paragraph is kept in Path A (it is, compressed, see
  Session 21C): cite Manohar, Brunton, Kutz & Brunton (IEEE CSM 2018) for sparse
  sensor placement in that paragraph. Add Drmac & Gugercin (qDEIM) ONLY if the
  sensor-selection figure is retained (it is being cut in 21C), otherwise omit.
- Optional, strengthens the fluids framing: a classical unsteady-gust model
  (Kussner/Sears) as the baseline the data-driven approach supersedes, and one
  TDA-in-fluids citation beyond Smith et al. Add if space allows; not mandatory.

-----

## 9. Recompile and verify

- Compile; confirm no undefined references, no “??”, no leftover “verify” notes.
- grep for em-dashes and replace per house preference (commas/colons/parentheses).
- Confirm the abstract is one paragraph, about 230-270 words, ending on
  significance.