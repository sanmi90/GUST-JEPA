# SESSION_MS: manuscript-rewrite compute dependencies

Renumber to the next free SESSION_NN on adoption. This session produces or
confirms the numbers the JFM rewrite (Claude Code session v2) depends on.
It trains no new encoders. All fits are on frozen v2.2 latents.

## Pre-flight

- [ ] `source .venv/bin/activate`; `PREVENT_ROOT` and `WANDB_PROJECT` set.
- [ ] v2.2 split manifest present and hash-logged (102 cases, 450
      encounters: 268 train / 100 validation / 42 in-distribution test /
      40 boundary test, symmetric |G| = 4).
- [ ] Frozen checkpoints for all ten table-6 families at d = 32 resolvable
      from the run registry; record checkpoint hashes.
- [ ] `numbers.json` writable; every new key carries a `provenance` field
      `{split: v2.2, session: MS, date}`.
- [ ] figstyle.py importable; outputs to `outputs/runs/sessionMS/`.

## Track M1: shared-operator forecast merit for table 6

Purpose: the primary cross-family forecast comparison must use one
operator; table 6's merit column currently mixes suited operators.

Protocol: fit the direct multi-horizon quantile forecaster (s3.2.1
configuration: LSTM width 512, nine quantiles, context 16-30 train / 25
eval, arcsinh per-window normalisation, pinball loss, AdamW 1e-3, 6000
iterations) identically on each of the ten frozen latent families, three
operator seeds each. Score the five-observable mean merit at H = 16 on the
in-distribution test split through the frozen probes; case-clustered
bootstrap CIs (2000 resamples on encounter means).

Outputs: `numbers.json` keys `merit_shared_<family>_{mean,lo,hi}`; a
long-form CSV per seed; W&B run group `MS-M1`.

Gates:
- Strong: family ordering matches the suited-operator ordering with
  non-overlapping case-clustered CIs at the top block. Action: replace the
  table 6 merit column; prose unchanged except operator description.
- Weak: ordering preserved but CIs overlap. Action: replace column; prose
  states the ordering is directional, mirroring the existing seed-variance
  caution of s5.5.
- Null: ordering changes. Action: report the shared-operator column as the
  primary result, move the suited-operator table to supplementary, and
  reword the s4.3/s4.4 attribution accordingly (% REVIEW-CLAIM to the
  editorial session; new HANDOFF entry).

## Track M2: v2.2 provenance re-runs

Purpose: retire every `% PROVENANCE-TODO` from the editorial Stage 1
ledger. For each item, confirm the quoted number was produced on v2.2; if
it was v2.1, re-run on v2.2 and compare.

Items (from PROVENANCE.md; expected list):
- M2a near-null / Mahalanobis departure fractions and condition numbers
  (table 7, figure 12a).
- M2b parameter-only floor (kernel-ridge (G, D, Y) -> observables at
  H = 16).
- M2c latent DMD spectrum and atlas probes (figure 21, s4.10 quantities:
  St = 0.666 / modulus 0.993 and the per-family comparison).
- M2d distributed-code gap (full-latent minus best-single-coordinate wake
  readability).
- M2e topology (Vietoris-Rips H1 fractions).
- M2f pressure-recovery pillar (table 8 state/observable recovery;
  figure 13 K x W grid) on the v2.2 taps and split.

Per-item gate: v2.2 value within the quoted value's case-clustered CI ->
keep the quoted value, update provenance tag. Outside the CI -> update the
macro, flag `% REVIEW-CLAIM` if any sentence's direction changes, log a
HANDOFF entry. If a v2.1 number is retained deliberately (cost), the
manuscript discloses it with the table-13 wording pattern.

## Track M3: figure regeneration

Regenerate through figstyle.py, to the editorial FIGURE_PLAN specs:
- M3a figure 6 replacement (two panels: PR per cell; paired case-mean
  forest vs lift-only), single-level (a)/(b) labels.
- M3b pipeline schematic (merged figures 4+5).
- M3c forecasting figure (merged 11 + 12b,c).
- M3d envelope/phase figure (merged 15 + 16a,b + 17a).
- M3e ladder + dimension grid (16c + 20), no in-panel prose; the
  test-peek note becomes appendix text.
- M3f slim physics figure (fig 21 DMD spectrum + one atlas panel).
Gate per figure: labels legible at JFM column width (check at 100 %),
caption data (split, n, uncertainty) supplied to the editorial session.

## HANDOFF stubs

- D310 M1 outcome branch taken (strong/weak/null) and table 6 disposition.
- D311 Per-item M2 dispositions (kept / updated / disclosed-v2.1).
- D312 Any claim rewordings triggered by M1/M2, with section references.

## Out of scope

New encoder training; Track C conditioning matrix; EBC evaluation;
multiscale observability (T4); sensor-reduction Track T. This session
exists solely to make the manuscript's numbers current and its primary
comparison operator-consistent.
