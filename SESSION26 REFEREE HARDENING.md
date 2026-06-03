# SESSION 26: Referee hardening of the JFM manuscript

Plan document for Claude Code. Read `CLAUDE.md` and `HANDOFF.md` first, then this file, then
execute track by track.

Last updated: 2026-06-03. Author of this plan: external-review pass on `main.pdf`.

## Why this session exists

An external JFM-referee-grade review of the current manuscript (the “forward physical closure”
paper, paper PDF built from `paper/`) found the science to be at or above the bar of the
Taira-lineage comparators in the project, but flagged a set of issues that a hostile JFM referee
would use to push the paper to major-revision or reject. This session fixes every one of those
issues that is inside our control. It does NOT touch the DNS solver-resolution numbers or the
grid and time-step convergence study, which the simulation collaborators own.

None of the in-scope work requires new model training. It is re-analysis of cached outputs,
plus rewriting, plus a reproducibility-package preparation. The one exception is an optional
alternative in Track 8 that is gated behind explicit user approval.

## Operating rules (do not violate)

1. Environment, every shell:
   `source .venv/bin/activate` then `export PREVENT_ROOT=$HOME/PREVENT WANDB_PROJECT=vortex-jepa`.
1. No em-dashes anywhere, in the paper or in any document you write. Run the repo conventions
   checker before every commit (find it; D160 refers to an `enforce_conventions` check).
1. Do NOT fabricate, estimate, or fill any DNS resolution number. The Table 1 author-fill rows
   (free-stream Mach number, computational domain and span, element and solution-point counts,
   minimum wall-normal spacing and wall units, time step and CFL, gust-release station, grid and
   time-step sensitivity) stay exactly as `[PENDING]`. Leave them untouched. They are out of scope.
1. The paper runs on split v2 (D130, D131), not v1. `CLAUDE.md` still documents v1 and is stale on
   this point. Confirm the actual split the paper numbers come from in Track 0 and use it. Do not
   edit any split manifest by hand; regenerate via the build script if a split ever needs changing
   (it should not this session).
1. Do not touch Test C (the |G| = 4 set) for any model selection or tuning. It is reported only.
1. Any new or revised manuscript figure imports `scripts/session21/figstyle.py` and is sized at
   the measured JFM text width. Keep the four-colour family key and per-family markers consistent.
1. The build must stay clean. After any edit to `paper/`, run the full build (`latexmk`, the
   project’s documented command) and confirm: exit 0, no undefined references, no undefined
   citations, no em-dashes, and record the page count. A track is not done until the build is clean.
1. Traceability. Every number that you newly introduce into the manuscript this session must trace
   to a committed output file under `outputs*/`. Maintain a running `outputs/session26/new_numbers_manifest.tsv`
   mapping each new paper number to the script and output file that produced it. A number with no
   committed source is treated the same as a fabricated DNS number: it does not go in the paper.
1. Honesty over preservation. The statistics work in Track 1 can weaken a claim (for example the
   forecast wake result may not survive case-level clustering or family-wide correction). If a claim
   weakens, reword the paper to match the weaker evidence. Do not suppress the result, do not
   re-pick a test to recover significance, and flag every such change prominently in the session
   report and the relevant decision-log entry.
1. Reuse cached latents and eval outputs instead of re-encoding or retraining. If a per-encounter
   array you need does not exist, regenerate it from cached latents plus the existing probes
   (no training). If you believe a step needs GPU training, stop and report before launching it.
1. Append a decision-log entry to `HANDOFF.md` for each track (continue the numbering from the
   last D-entry, which is D163). Record what changed, the rationale, and any claim that weakened.

## Definition of done

All in-scope gates below are either passed or honestly marked failed-with-explanation; the build is
clean and em-dash-free with the page count recorded; every new paper number is in the traceability
manifest; the DNS `[PENDING]` rows are untouched; `HANDOFF.md` has a D-entry per track; `CLAUDE.md`
is corrected where it is stale (at minimum the split-version note); and `SESSION26_REPORT.md` is
written summarising gate outcomes, claim changes, residual risks, and what remains for the
collaborators (the DNS numbers and the convergence study).

-----

## Track 0: Ground-truth audit and clean baseline build

Objective: establish where everything is before changing anything, and a clean baseline build to
diff against.

Check and do:

- Build the manuscript as-is and confirm it currently compiles clean. Record the baseline page
  count and stash the baseline PDF.
- Locate the manuscript source root (the main `.tex` and `paper/sections/`). Confirm the build
  command (latexmk) and the conventions checker. Run the checker on the current tree to capture the
  baseline state of any pre-existing flags (D160 noted some pre-existing R^2 flags in section 5 that
  are not real violations; record which flags are pre-existing so you do not chase them).
- Confirm the split the paper’s numbers come from. Find the v2 manifest (for example
  `configs/splits/split_v2.json`, confirm the actual name). From it, extract and write down:
  the number of distinct CASES behind test_b and behind test_c, and the case-to-encounter mapping
  (which of the 42 test_b encounters and 24 test_c encounters belong to which case). This is the
  input Track 1 needs.
- Build an artifact map. For each manuscript table and figure that carries a number this session
  touches, record the generating script and the cached output file it reads:
  Table 4 (held-out closure), Table 5 (conditioning floor), Table 6 (drift), Table 7 (2x2 controls),
  Table 8 (dimension sweep and ablations), Table 9 (training fit), Table 10 (paired closure),
  Figure 8 (persistent homology), Figure 12 (optimal transport), Figure 13 (scale decomposition),
  Figure 14 (pressure observability), Figure 15 (wake-forecast code). Locate the v2-era eval outputs
  (Sessions 18 to 25; do not assume the v1 `outputs/session14/...` paths from the stale `CLAUDE.md`).
- Locate the per-encounter absolute-error arrays for predictive (JEPA d=64) and reconstructive
  (Fukami d=64) on test_b at H=16, for all six observables, in both the representational and the
  forecast modes. These produced Table 10. If they are not cached as arrays, identify the script
  that would regenerate them from cached latents plus probes.

Gate 0: clean baseline build with page count recorded; `outputs/session26/artifact_map.md` mapping
every in-scope table and figure to its (script, output file); the test_b and test_c case counts and
case-to-encounter mapping written to a committed file; the location of the per-encounter error
arrays confirmed or a regeneration path identified. If any table or figure source cannot be found,
list it explicitly as a blocker in the audit note.

Write: no paper edits in Track 0. Open the Session 26 decision-log block in `HANDOFF.md` with the
audit findings.

-----

## Track 1: Statistics hardening (do this first; it can change claims)

Objective: make the headline statistics survive a methods referee. Three sub-tasks.

### 1a. Case-level dependence

The paper reports encounter-level bootstraps and a per-encounter paired test, but never states how
many distinct cases the test encounters come from. If several encounters share a case they share the
baseline shedding dynamics and are not independent, so encounter-level intervals and p-values are
optimistic.

Do:

- Report n_cases for test_b and test_c in the paper (Section 2.2 and the relevant table captions).
- For the wake-enstrophy paired comparison (predictive d=64 vs reconstructive d=64, test_b, H=16),
  in BOTH modes, recompute with a case-clustered procedure: resample CASES with replacement
  (block bootstrap, at least 10000 resamples, averaging within case before the case enters the
  resample), and report the case-clustered 95 percent CI on the mean per-encounter improvement and a
  case-level paired statistic (Wilcoxon signed-rank on per-case means, or a mixed-effects model with
  case as a random effect). Lead the paper with the magnitude-based case-clustered CI, not the
  sign-test p-value.
- Re-express the other headline statistics that are currently encounter-level (the drift ratio is a
  single number so it is fine; the topology Mann-Whitney, the transport Spearman, and the
  scale-decomposition correlations should be checked for whether the encounter is the right unit, and
  re-reported at the case level where a case contributes multiple encounters).

### 1b. Multiple comparisons and a pre-registered primary endpoint

Table 10 reports twelve paired tests (six observables, two modes). The forecast wake p of 0.044
does not survive a family-wide correction (Bonferroni or Holm) over twelve tests; the
representational wake p of 0.0014 does.

Do:

- Designate wake enstrophy as the single pre-registered PRIMARY endpoint, justified a priori on
  physical grounds (the wake observables are the most demanding and most discriminating; this
  rationale is already in Section 2.2, make the designation explicit there). State that the other
  five observables are secondary and descriptive.
- For the secondary observables, report Holm-corrected p-values in Table 10 or its caption, and
  state plainly that the forecast wake result does not survive a family-wide correction, which is
  exactly why wake enstrophy is pre-registered as the primary endpoint and why the representation
  and mechanism evidence (Track 6) is the anchor rather than the marginal forecast number.

### 1c. Predictive versus the conditioning floor

The paired tests answer predictive versus reconstructive. They do not answer predictive versus
parameters-alone. The floor on wake enstrophy is 0.17 and the forecast is 0.449, but with a CI to
negative values the margin over the floor is not established at the marginal level.

Do:

- Per encounter, compute predictive-forecast wake error against the conditioning-floor wake error
  (the frame-matched kernel-ridge floor from Table 5 at H=16) and report a paired CI (case-clustered).
  Do the same for the representational closure against the floor (this should be comfortably positive).
- State the result: whether the forecast is reliably above the floor, and that the representational
  closure is clearly above it. This supports re-anchoring on representation in Track 6.

Gate 1: a committed `outputs/session26/stats/` directory with: n_cases per split; case-clustered CIs
and p-values for the wake comparison in both modes; Holm-corrected secondary p-values; the
predictive-versus-floor paired CIs; and the re-checked topology, transport, and scale statistics at
the appropriate unit of analysis. A `stats_summary.md` stating, claim by claim, which survive
case-level clustering and family-wide correction and which need rewording. Test C untouched for
selection.

Write: Section 2.2 (case counts, primary-endpoint designation), Table 10 and caption (Holm-corrected
secondary p-values, case-clustered primary CI), and a note feeding Track 6. Decision-log entry.

-----

## Track 2: Persistent-homology robustness

Objective: give the topology result (the median generator count, Mann-Whitney p around 4e-8, cited
as the decisive confirmation) the threshold and sampling robustness study that the Smith et al.
(2024) appendix in the project sets as the bar, and that a referee from that group will expect.

Check and do:

- Find the Session 20 Track C persistent-homology script and the four latent point clouds per
  encounter (predictive encoded, predictive rollout, reconstructive encoded, reconstructive rollout)
  and the generator-count computation behind Figure 8.
- Report explicitly how the significance threshold (the noise floor separating real H1 generators
  from noise) is currently set.
- Add a sensitivity of the predictive-versus-reconstructive generator-count separation to that
  threshold over a defensible range, and a sensitivity to the number of points sampled per encounter
  (subsample the trajectory), following the convergence protocol of Smith et al. (2024).
- Recompute the median generator counts and the Mann-Whitney p across the threshold and sampling
  grid. Confirm the separation holds; if the order-of-magnitude p is threshold-fragile, report that
  honestly and soften the word “decisive.”

Gate 2: a committed output table of generator-count separation across threshold and sampling
settings, and a drafted appendix paragraph (or a Section 4.3 sentence with appendix backing) that
reports the robustness and cites Smith et al. (2024).

Write: Section 4.3 or Appendix A, plus decision-log entry.

-----

## Track 3: Physical-definition caveats

Objective: close three physics holes a fluid mechanician finds immediately. All three are about how
the observables are defined and computed from the cache, not about the DNS setup, so they are in
scope.

### 3a. The impulse Iy does not satisfy the impulse-lift relation

The mid-plane 2D vorticity misses the bound circulation, so dIy/dt does not track CL on the DNS
(the project measured r near minus 0.028; D124c). A referee will try the impulse theorem and find
it fails.

Do:

- Recompute that DNS correlation from the cache so it is traceable, then add an explicit statement in
  Section 2.2 that Iy is a mid-plane 2D diagnostic of wake-vortex transport and is NOT the
  impulse-theorem lift, citing the measured decorrelation. Ensure no passage in Section 4 implies
  otherwise.

### 3b. The wake observables are mid-plane 2D proxies; the forces are 3D

The four wake and flow observables (wake enstrophy, the two circulations, Iy) are computed on the
mid-plane slice, consistent with the encoder input, and so omit out-of-plane content. The project
measured that roughly 20 percent of the spanwise-vorticity enstrophy is not in the spanwise mean even
in-distribution (chi_3D near 0.20; D147). CL and CD by contrast are true 3D surface-integrated forces.

Do:

- State in Section 2.2 that the wake and flow observables are mid-plane 2D quantities, quantify the
  in-distribution out-of-plane content with chi_3D near 0.20 (verify the number is traceable from the
  D147 output), and acknowledge the definitional asymmetry with the 3D forces. This also tightens the
  existing |G| = 4 observability-boundary argument.

### 3c. The circulation threshold is arbitrary

The signed circulations use a fixed threshold omega_c = 1.

Do:

- Recompute the circulation closure at a small set of thresholds (for example omega_c in {0.5, 1, 2})
  and report a one-line sensitivity, or give a physical justification for omega_c = 1.

Gate 3: drafted Section 2.2 paragraphs with the Iy non-impulse caveat, the 2D-proxy statement with
chi_3D, and the omega_c sensitivity result committed. Every number traceable.

Write: Section 2.2, plus decision-log entry.

-----

## Track 4: Resolve the decoder confound

Objective: remove the internal tension between “the predictive decoder is blurry by design” and the
quantitative physical-space claims in Section 4.6 (the leading-edge vortex localised to 0.32 chords,
circulation to within 5 percent).

Check and do:

- State clearly that all physical-space claims are restricted to the validated large-scale band
  (sigma/c = 0.05); the decoder is blurry only at small scale, which is why large-scale tracking is
  meaningful while small-scale fidelity is not.
- Find the oracle-decode numbers (the visualisation decoder applied to the simulation-encoded latent,
  Session 20 Track F or Session 23 Track E artifacts) and report them as the CEILING for the Section
  4.6 metrics (LEV centroid error, circulation error, large-scale enstrophy correlation), so the
  predictive-rollout numbers are read against the oracle ceiling rather than against the true field.

Gate 4: Section 4.6 reworded to scope every physical-space claim to the large-scale band and to
report the oracle-decode ceiling alongside the rollout numbers; the ceiling numbers traceable.

Write: Section 4.6, plus decision-log entry.

-----

## Track 5: Baseline-tuning transparency

Objective: pre-empt the “you compared against an under-tuned or unstable baseline” objection.

Check and do:

- The reconstructive AE wake-forecast R^2 is non-monotonic in d and worst at d=64
  (3: -0.082, 16: -0.395, 32: +0.007, 64: -0.478 in Table 4b). Make this explicit as a consequence
  of the drift mechanism (a reconstruction objective leaves more directions unconstrained at higher
  d, so the rollout has more room to drift), and confirm it is seed-robust using the existing retrains.
- Report the per-seed values for the high-variance 2x2 control cell (reconstructive CNN+ViT,
  standard deviation near 0.27 on the wake R^2), and frame the instability itself as evidence
  (unconstrained latent geometry yields seed-dependent wake closure), not as an artifact to hide.
- Confirm in the text that the AE baseline uses its best documented configuration (ReLU plus GroupNorm
  plus future-CL head, per `CLAUDE.md`, not the strict-paper variant that gives a worse probe), so a
  referee cannot claim the baseline was hobbled.

Gate 5: a sentence in Section 4.1 or 4.5 explaining the non-monotonicity mechanistically, per-seed
values for the high-variance cell committed, and an explicit statement of the AE baseline
configuration. Numbers traceable.

Write: Section 4.1 and 4.5, plus decision-log entry.

-----

## Track 6: Headline reframe to the transport-consistency principle

Objective: convert the paper from “a predictive model beats a reconstructive baseline” (a
representation-learning result that invites a wrong-venue rejection) into “we identify the geometric
property a reduced coordinate must have to forecast gust encounters, and show which encoder
objectives confer it” (a fluid-mechanics result). Do this only after Tracks 1 to 5, so the numbers
and caveats are settled.

Do:

- Lead with the principle: the forecastability of a reduced coordinate for these flows is governed by
  whether the latent metric is an isometry of the optimal-transport geometry of the field; a
  reconstruction objective does not impose this, so its rollout leaves the data manifold (the drift),
  whereas the predictive objective, regularised against collapse, produces a transport-consistent
  metric so iterating the predictor stays on the manifold. JEPA is the instrument that realises the
  principle, not the subject of the paper. This reframe is supported by the strongest evidence already
  in the paper (drift, topology, optimal transport, scale decomposition).
- Re-anchor the wake claim on representation plus mechanism. The predictive latent CARRIES the wake
  structure that the reconstructive latent does not (representational closure R^2 near 0.75,
  case-clustered paired CI from Track 1, topology, transport, scale decomposition), and the drift
  mechanism explains WHY a reconstruction objective cannot. Present the forecast as consistent
  confirmation, not as the load-bearing test. Lead with the case-clustered paired improvement CI from
  Track 1, not the sign-test p-value.
- Sharpen the objective-plus-supervision finding. State it as: the predictive objective is what
  converts wake-observable supervision into a forecastable wake state; the same supervision on a
  reconstructive objective drifts away (the reconstructive cells carry the same wake head and do not
  reach the predictive closure, Table 7). This is crisper than the current phrasing and matches every
  control.
- Touch: abstract, Section 1 contributions, the framing sentences in Sections 3.4 and 4.3, Section
  5.1, and Section 6. Keep the changes consistent with the Track 1 numbers.

Suggested principle statement to adapt (do not paste verbatim; align with the final numbers):
“A reduced state forecasts these encounters well when its latent metric is an isometry of the
optimal-transport geometry of the flow, so that one predictor step is a transport-consistent move and
iterating it stays on the data manifold. A reconstruction objective does not impose this property and
its rollout drifts off the manifold; a predictive objective regularised against collapse does, which
is why the predictive latent keeps the wake observables close under rollout while the reconstructive
latent does not.”

Gate 6: revised abstract, intro contributions, Section 5.1, and Section 6 consistent with the
principle framing and with the Track 1 numbers; build clean.

Write: abstract, Section 1, Sections 3.4 and 4.3, Section 5.1, Section 6, plus decision-log entry.

-----

## Track 7: Demote the world-model framing and rewrite the abstract close

Objective: stop the disclaimed world-model framing from priming the reader for something the paper
does not deliver, and stop amplifying the machine-learning reading that risks the venue.

Do:

- Reduce the world-model material to one sentence of motivation in Section 1 and one sentence in
  Section 5.4 or the outlook. The interventional test fails and is disclaimed, so it should not frame
  the paper.
- Do not end the abstract on the world-model disclaimer. Move that to a single mid-abstract clause and
  end on the positive forecastable-and-observable result.

Suggested abstract close to adapt:
“The same predictive state is the most recoverable of the three from sparse wall pressure, so it is
observable as well as forecastable. We frame the predictive objective as conferring a
transport-consistent latent metric; the action-conditioned world-model reading is motivation only,
since a direct interventional test does not hold.”

Gate 7: world-model mentions reduced to motivation; abstract ends on the positive result; build clean.

Write: abstract, Section 1, Section 5.4, plus decision-log entry.

-----

## Track 8: Decide the closed-loop pilot

Objective: stop Section 4.7 and Section 5.4 from undercutting each other (a pressure-recoverable
forecastable state whose control payoff is also shown not to materialise).

Default action (no compute):

- Cut the closed-loop pilot to a single honest sentence. Remove the deployment framing from Section
  4.7 and keep only the representational property: the predictive state is the most pressure-observable
  of the three. Move any remaining pilot detail to a short appendix note or delete it. The honest
  reading (representation and pressure observability are in place; the closed loop does not yet meet a
  tight tolerance and is rollout-limited) stays, but compressed, so it does not read as a capability
  that fails to deliver.

Optional alternative (requires explicit user approval before any GPU run):

- Add one open-loop latent-planning comparison in which the predictive latent beats the reconstructive
  and linear latents on a control-relevant surrogate, so the forecastable state demonstrably pays off
  even without full closed-loop control. If the user approves this, scope it, then stop and report the
  plan before launching anything on the RTX 6000.

Gate 8: pilot reduced to an honest scope statement and the Section 4.7 deployment claim removed (or,
only on explicit approval, the open-loop result added); build clean.

Write: Section 4.7, Section 5.4, Appendix B, plus decision-log entry.

-----

## Track 9: Manuscript economy and internal consistency

Objective: fix the one number-location mismatch, trim the table load, and tighten the register toward
the lineage style.

Do:

- The abstract headlines the representational wake R^2 near 0.75, but Table 4(a), which it points to,
  reports mean absolute error, not R^2. Add an R^2 column or a companion panel to Table 4(a) so the
  abstract’s number appears in the table it references. Verify the number.
- Move Table 9 (training fit, explicitly “for reference only”) to the appendix if it is not already,
  and consider merging the representational and forecast blocks of Table 4 into one table.
- Register pass: in each Results subsection, state the finding cleanly in the first sentence, then
  qualify, and move the heaviest scope bookkeeping into Sections 5.2 and 5.3. Reduce the density of
  meta-commentary (phrases like “we are explicit that,” “the honest claim is,” “one caveat is
  load-bearing”) so the Results read like the lineage comparators: clean result first, qualification
  after.
- Spot-check every cross-referenced number for internal consistency after all edits.

Gate 9: Table 4(a) shows the R^2 the abstract cites; tables trimmed; register tightened; no internal
number mismatches; build clean.

Write: abstract and Table 4 area, Results subsections, plus decision-log entry.

-----

## Track 10: Reproducibility package preparation (no DOI minting)

Objective: replace “available on reasonable request,” which JFM increasingly treats as a soft
rejection trigger, with a deposited analysis package. Do not mint the DOI (that needs the user’s
account) and do not deposit raw DNS (collaborator-owned).

Do:

- Prepare a clean release of the analysis artifacts: a top-level README with reproduction steps, a
  LICENSE (confirm or choose), a CITATION.cff, and a Zenodo metadata file (`.zenodo.json`) with
  authors, title, and keywords. List exactly what will be deposited: the ROM and evaluation code, the
  v2 split manifest, and the cached outputs needed to regenerate the tables and figures.
- Update the data-availability statement to: code and evaluation pipeline deposited at a DOI
  placeholder, the processed per-encounter cache (or a representative subset) available, and the raw
  DNS owned by the simulation collaborators. Leave the DOI as a clearly marked placeholder for the
  user to fill after minting.

Gate 10: release scaffolding committed (README, LICENSE, CITATION.cff, `.zenodo.json`, a release-candidate
tag); the data-availability statement updated with a DOI placeholder; build clean.

Write: the data-availability statement, plus decision-log entry.

-----

## Track 11: Final verification and handoff

Objective: prove the session is done and leave the project in a clean, traceable state.

Do:

- Full build: latexmk exit 0, no undefined references or citations, no em-dashes (run the conventions
  checker), final page count recorded.
- Traceability sweep: confirm every new number in the manuscript appears in
  `outputs/session26/new_numbers_manifest.tsv` with a committed source. Any number without a source is
  removed or sourced before done.
- Decision log: append one D-entry per track to `HANDOFF.md` (continuing from D163), each recording
  what changed, the rationale, and any claim that weakened under Track 1.
- Correct `CLAUDE.md` where it is stale: at minimum note that the paper runs on split v2 (the locked
  decisions section still describes v1). Do not rewrite locked decisions; only fix statements that are
  now factually wrong.
- Write `SESSION26_REPORT.md`: gate outcomes, claim changes (especially any weakened by case-level
  clustering or family-wide correction), residual risks, and the explicit list of what remains for the
  collaborators (the Table 1 DNS resolution numbers and the grid and time-step convergence study).

Gate 11 (definition of done): clean em-dash-free build with page count; all in-scope gates passed or
honestly failed-with-explanation; traceability manifest complete; decision log updated; `CLAUDE.md`
de-staled; `SESSION26_REPORT.md` written; DNS `[PENDING]` rows untouched.

-----

## Out of scope (leave for the collaborators)

Do not work on, and do not fabricate values for, any of the following:

- The Table 1 DNS solver-resolution numbers: free-stream Mach number or incompressible-limit
  confirmation, computational domain and spanwise extent, element and solution-point counts, minimum
  wall-normal spacing and wall units, time step and maximum CFL, gust-release station.
- The grid and time-step sensitivity (convergence) study.
  These stay as `[PENDING]` author-fill. Surface them in `SESSION26_REPORT.md` as the remaining
  collaborator action, nothing more.

## Suggested execution order

Track 0, then Track 1 (it can change which claims survive and everything downstream depends on the
settled numbers), then Tracks 2 through 5 (the remaining evidence and caveats), then Track 6 and Track
7 (the reframe and abstract, which depend on the settled numbers and caveats), then Track 8 (pilot
decision), then Track 9 (economy and consistency), then Track 10 (reproducibility package), then Track
11 (verification and handoff). Build clean and commit at the end of each track.