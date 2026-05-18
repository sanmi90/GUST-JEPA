# SESSION5_MEANINGFUL_SMOKE_5K.md

Session 5 plan for the vortex-jepa project.

Last updated: 2026-05-17.

## Session goal

Run the 5k-iteration meaningful smoke on 5 training cases and learn from
whatever it produces. The output of this session is a methodological finding,
not a passing build: we are answering the question “does SIGReg work on this
low-intrinsic-dim physics data, in any reasonable configuration, on a small
case subset?” The answer determines what Session 6 looks like.

Pass-criteria-as-target (from HANDOFF step 5):

- SIGReg loss below 5.0 at iter 5000
- Participation ratio PR(z) > 0.5 * d = 16 at iter 5000
- Linear probe R^2 for c on Test B > 0.5 at iter 5000

Pass-criteria-as-acceptance: these numerical thresholds are the operational
target. **They are not the only outcome that counts as a successful session.**
A clean negative result (the methodologically interesting outcome of “SIGReg
collapses even with LayerNorm, VICReg recovers PR but probe R^2 stays low at
this data scale”) is also a successful Session 5: it tells us what Session 6
must look like and gives us a real datapoint for the paper’s contribution
3 (the SIGReg-on-low-intrinsic-dim diagnostic). The failure mode IS the
contribution; we just need to be confident about which failure mode it is.

## Why this is harder than it sounds

Six convergent reasons:

**There is a published precedent that predicts the failure mode.** LeWM
(arXiv:2603.19312) reports on its project page and in the paper itself:
“LeWM underperforms on Two-Room; we suspect this is due to the intrinsic
dimensionality of the task being too low, which may hinder the Gaussian
regularizer from producing a well-structured latent space. This highlights
a potential limitation of the SIGReg regularization in very low-complexity
environments.” Our pre-registered H4 hypothesis is essentially that this
LeWM failure mode generalises to physics data with intrinsic dimension ~5
to 10. The 5-case smoke is the smallest experiment that can replicate the
LeWM Two-Room failure on a new domain. Session 5 should be framed as a
direct replication-and-extension: replicate the failure under default
LeWM-faithful settings (Run A), then test interventions (LayerNorm in Run
B, VICReg in Run C, both in Run D) that LeWM did NOT systematically try.
This is the “honest reporting of the SIGReg-on-low-intrinsic-dim
diagnostic” promised in the project summary item 3.

**The Session 4 smoke already showed the pathology, just with fewer samples.**
The 200-iter run on 11 training samples produced PR = 1.12 (out of d = 32,
i.e. effective rank ~ 1) and probe R^2 = 0.711 at iter 100. That is
near-complete latent collapse paired with high decodability of c, which is
the “encoder memorises the case label” trivial solution. Going from 11 to
~20 training samples (5 cases x 4 train encounters average) does not
mechanically change why this happens; the collapse mechanism is structural,
not sample-count-bound. Session 5 must allow for the case that the default
configuration does NOT clear the pass criteria, and that this is a real
signal, not a bug. The expected outcome under H4 is TRIVIAL (LeWM Two-Room
replicated). The interesting question is whether the LayerNorm and VICReg
interventions recover anything.

**The Two-Room failure is also a precedent for PLDM beating LeWM at low
intrinsic dimensionality.** LeWM Section 5 reports: “In the simpler
Two-Room environment, PLDM and DINO-WM outperform LeWM, which may be
explained by the SIGReg regularization encouraging a Gaussian distribution
in a high-dimensional latent space, while the intrinsic dimensionality of
the environment is much lower.” This is more specific than “SIGReg
struggles on low intrinsic dim”; it says PLDM’s VICReg-derived 7-term
objective actually WINS in that regime. Our paper’s central methodological
contrast (D8) is SIGReg+2-term vs PLDM’s VICReg+7-term. The LeWM finding
implies the contrast may not favor SIGReg on our data, since our estimated
intrinsic dimension (~5 to 10) is closer to Two-Room than to Push-T. This
reshapes Session 5’s priorities for what comes next: if the TRIVIAL outcome
materialises, **PLDM becomes the priority comparator immediately after
Session 5**, before Session 5.5 or Session 6. The contrast we then report
is regime-dependent (SIGReg wins on Push-T-like data; PLDM wins on
Two-Room-like data; the PR diagnostic tells you which regime you are in).
That regime-dependent framing is more defensible than the original
“SIGReg always wins on physics” framing.

**The auto-fallback rule, as currently designed, will not fire on the
expected failure.** D5 fires when `iter >= 20000 AND PR(z) < 0.3 * d AND probe_R^2 < 0.7`. On the 200-iter smoke, PR was deeply below threshold but
probe R^2 was already ABOVE 0.7. The conjunctive design is intentional: it
catches the worst case (PR low AND probe low, “the latent is both collapsed
AND useless”). The trivial-solution failure mode (PR low AND probe HIGH,
“the latent collapses TO c”) falls outside the rule. Whether this design
choice is correct on physics data is a separate methodological question.
The plan treats this as a question to answer with data, not a bug to fix
ahead of time. If Session 5 demonstrates the trivial-solution mode and we
believe the rule should also catch it, that proposal becomes D28; if the
LayerNorm intervention recovers a healthy latent without needing the rule
to fire, the rule stays as is.

**D17’s “first intervention” is manual.** If PR is low at the end of the
5k run, the documented first action is to swap BatchNorm for LayerNorm at
the encoder projection. This is a config/code change, not an automated
toggle. Session 5 must be set up to do this swap as one of several
controlled variant runs, with comparable W&B logging and a single analysis
notebook that loads all of them.

**The pass criteria collide with a stated warning.** HANDOFF “Warnings and
pitfalls” reads: “High probe R^2 on the encoder for c is a red flag, not a
success. The encoder is unconditional by design; if it can decode c, c is
leaking from somewhere.” The Session 5 pass criteria want probe R^2 > 0.5
for c. Both can be true (some decodability is the encoder doing its job,
near-perfect decodability with collapsed PR is leakage), so the real
acceptance condition is the *combination*, not any single metric. The plan
encodes this combinatorial criterion explicitly.

## What this session is NOT

Scope-limiting matters here.

This session is **not** the place to introduce Hydra configs. Hydra is a
refactoring task. Done well it takes a session of its own; done poorly
mid-session it blocks the methodologically important smoke run. HANDOFF
step 5 bundles Hydra with the 5k smoke; this plan unbundles them. Hydra
becomes Session 6.5 or its own short session before the lambda bisection.

This session is **not** the place to turn on `torch.compile()`. Same
argument: a performance lever, not a methodological one. Enable it after
we have a non-compiled baseline wall-clock to compare against. Defer to
the same Hydra session.

This session is **not** the place to build the visualisation decoder. The
original HANDOFF step 4 mentioned “the visualisation decoder produces
recognisable fields” as a smoke criterion, but Session 4 report dropped
this criterion correctly: the decoder is its own training pipeline on a
frozen encoder. Build the decoder only after a JEPA checkpoint clears
Session 5’s combinatorial criterion, otherwise the decoder is decoding a
collapsed latent and we learn nothing from it.

This session is **not** the place to start the baselines (PLDM, Fukami AE,
Solera-Rico, POD). Those are parallel paper-work that does not block.

## Files to create

```
notebooks/01_smoke_5k_analysis.ipynb
scripts/run_smoke_5k_variants.sh
src/training/sanity_checks.py
configs/cases/smoke_5cases.yaml             # case list, plain YAML, NOT a Hydra config
tests/test_sanity_checks.py
SESSION5_MEANINGFUL_SMOKE_5K.md             # this file
```

The intent is to keep new code small. The bulk of Session 5 is RUNS plus
ANALYSIS, not new modules. Five files, all small.

No changes to:

- The 7 model modules (encoder, predictor, sigreg, adaln, rope, jepa,
  vicreg). All of Session 5’s variants are configuration changes, not
  module modifications.
- The data loader (`src/data/episode_dataset.py`).
- The training entrypoint (`src/training/train_jepa.py`) is NOT modified
  unless we discover a bug in the sanity-check pass. If we do, log the
  fix as a D-entry; otherwise the entrypoint is stable.

## arXiv MCP plugin

Enabled. Recommended consultation order if needed during analysis. All
references below have been verified against the primary sources, not
inferred from memory.

|Reference                                                     |arXiv ID  |Used for                                                                                                                                                                                                                                                                                                                                              |
|--------------------------------------------------------------|----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|LeWM: Maes et al.                                             |2603.19312|Two-Room failure mode quote (paper text plus le-wm.github.io project page); Section 4 ablations on M projections and integration knots; Appendix G on lambda bisection                                                                                                                                                                                |
|LeJEPA: Balestriero, LeCun                                    |2511.08544|Section 4 (SIGReg derivation, isotropic-Gaussian assumption); Section 6 (experiments at scale); official code at github.com/galilai-group/lejepa, formerly github.com/rbalestr-lab/lejepa                                                                                                                                                             |
|VICReg: Bardes, Ponce, LeCun                                  |2105.04906|Variance hinge intuition for low-rank latents; Section 5.4 on dropping the invariance term when there is no paired view                                                                                                                                                                                                                               |
|V-JEPA 2-AC: Assran et al.                                    |2506.09985|Training recipe: teacher-forced one-step over T=15 plus rollout over T=2 (this is the source of “H_roll=2” in V-JEPA 2-AC, which our project deliberately deviates from with H_roll=8 per D21)                                                                                                                                                        |
|PLDM: Sobal, Zhang, Cho, Balestriero, Rudner, LeCun (Feb 2025)|2502.14819|7-term VICReg-derived (or related multi-term) anti-collapse, comparator for the paper’s central methodological contrast. NOTE: D8 in HANDOFF.md originally cited the wrong arXiv ID (2211.10831, which is a 2022 workshop precursor with a different author list). The correct primary reference is 2502.14819; see D32 (to be added after Session 5).|

LeJEPA Figure 1 is an overview figure (training-loss-vs-probe correlation,
training stability at scale, PCA semantic features, Galaxy10 results); it
is NOT a collapse-modes diagnostic. For PR-interpretation context, consult
LeJEPA Section 4 (where the isotropic-Gaussian assumption is derived) and
Section 6 (where empirical collapse is reported across architectures).

If the smoke produces an unexpected failure mode (not collapse, not
trivial solution, but something else), consult LeWM Section 4 and the
LeWM project page (le-wm.github.io) for the Two-Room failure mode
discussion before diagnosing on instinct.

## Pre-run sanity checks (mandatory, ~30 min)

Before running ANY of the 5k variants, validate that the 200-iter smoke’s
behaviour is the expected pathology and not a wiring bug masquerading as
collapse. Build `src/training/sanity_checks.py` as a small module with
five checks, each runnable as a script entrypoint. The point is to rule
out trivial explanations before we invest 60+ minutes of GPU time on the
variant runs.

### Check 1: BatchNorm statistics do not correlate with c

Run the encoder in `eval()` mode on a small batch from 3 cases (one
encounter each). Extract the `running_mean` and `running_var` of the
encoder’s projection BatchNorm BEFORE any training. They should be the
PyTorch defaults (mean 0, var 1) since no training has run; this is a
sanity check that we are looking at the right module and that the cache
read is correct.

Then run the encoder in `train()` mode for 100 steps of forward-only
(no backward), feeding distinct batches drawn from one case at a time
in alternation. The BatchNorm running statistics will update. After
100 steps, the running mean and var should NOT have learned to encode
the case label, because we are feeding randomly mixed batches. If they
do (per-batch BatchNorm leaks case identity through the running stats),
that is a real bug to fix before the 5k run.

This is the test for the “BatchNorm statistics correlating with c”
warning in HANDOFF.

### Check 2: predictor is identity-on-residual at init, but NOT at iter 1

The Session 3 unit test verifies the predictor is identity-on-residual
at init (AdaLN-Zero). At init, `predictor(z, c) == embed(z)`, so
teacher-forced L_pred is large (the embed is not identity to the
latent). After one optimizer step, the AdaLN gates should move off
zero and L_pred should start decreasing. Verify on a single batch
under torch.manual_seed(0) that:

- At iter 0, L_pred is between 0.01 and 10.0 (matching the
  test_jepa_identity_predictor_gives_zero_pred_loss_at_init test).
- After one Adam step on this batch, the AdaLN gates have nonzero
  values (e.g. assert at least one parameter in
  `[m for m in jepa.predictor.modules() if isinstance(m, AdaLN)][0] .final_linear.weight` is nonzero).
- After ten steps on this single repeated batch, L_pred has
  decreased monotonically. This is overfitting on one batch, which
  is the cheapest possible test that learning is wired correctly.

### Check 3: SIGReg gradient is meaningful on the actual encoder output

Run a forward through encoder + predictor, extract `z` of shape
`(B*T, d=32)`, compute SIGReg(z), backward. Verify:

- The gradient at the encoder’s output projection BatchNorm bias is
  nonzero (SIGReg actually moves the encoder).
- The gradient magnitude is in the expected ballpark from the Session
  2 SIGReg unit tests (not enormously larger or smaller).

If SIGReg’s gradient on real encoder output is orders of magnitude
different from the synthetic-Gaussian unit test, that is a signal that
the encoder output distribution is so degenerate that even the
regulariser is degenerate.

### Check 4: rollout matches teacher-forced single-step at horizon 1

On a single batch, with the encoder in `eval()` mode (so z is
deterministic), assert:

```
jepa.predictor.rollout(z[:, :1, :], cond, steps=1)[:, 1:, :]
    \approx jepa.predictor(z, cond)[:, :1, :]
```

within bf16 tolerance. This is a consistency check that the two
forward modes agree at horizon 1 (where they must by construction).
Failure here is a Session 3 bug we missed.

### Check 5: data loader emits the correct (B, T, 1, 192, 96) and (B, 3) shapes

Already covered by Session 2’s `episode_dataset.py` smoke test, but
re-verify on the exact 5-case subset Session 5 will use. The check
asserts:

- omega shape is `(16, 32, 1, 192, 96)` for batch_size=16
- c shape is `(16, 3)` with values in the (G, D, Y) ranges from the
  inventory
- omega has no NaN, no Inf
- omega magnitudes are roughly in `[-100, 100]` (vorticity at this
  Re; sanity check on the cache preprocessing)

If any of these fail, the variant runs WILL produce useless results.

### Unit tests for the sanity checks

`tests/test_sanity_checks.py`:

```python
def test_check_1_runs_on_synthetic_encoder():
    """Build a tiny encoder and verify check_1 runs and reports the
    expected status on synthetic data (no real cache needed)."""

def test_check_2_runs_on_jepa_instance():
    """Verify check_2 runs on a fresh JEPA instance and reports the
    AdaLN gate-movement assertion correctly when the optimizer step
    succeeds vs fails."""

def test_check_3_runs_against_synthetic_z():
    """Verify check_3 runs on synthetic z and reports finite gradient
    magnitudes."""

# Checks 4 and 5 are integration tests; they live in the slow path
# and are exercised by the variant runs themselves.
```

Run all sanity checks via `python -m src.training.sanity_checks --all`
before the first variant. Total wall-clock: under 5 minutes.

## The case subset

Five cases, deliberately chosen rather than picked randomly:

|case_id              |Source  |G |D  |Y/c  |Rationale                           |
|---------------------|--------|--|---|-----|------------------------------------|
|`Baseline`           |periodic|0 |0  |0    |No-gust limit; calibration reference|
|`G+3.00_D0.50_Y+0.20`|periodic|+3|0.5|+0.20|Strong positive gust, near top of   |
|`G-3.00_D1.00_Y-0.20`|periodic|-3|1.0|-0.20|Strong negative gust (sign coverage)|
|`G+1.00_D1.50_Y+0.10`|periodic|+1|1.5|+0.10|Moderate gust, larger diameter      |
|`G+1.00_D1.00_Y-0.20`|run3    |+1|1.0|-0.20|Run3 source group representation    |

Stored in `configs/cases/smoke_5cases.yaml` as a plain YAML list of
case_ids, NOT a Hydra config:

```yaml
cases:
  - Baseline
  - G+3.00_D0.50_Y+0.20
  - G-3.00_D1.00_Y-0.20
  - G+1.00_D1.50_Y+0.10
  - G+1.00_D1.00_Y-0.20
```

The training entrypoint loads this with PyYAML and passes the list to
`--cases`. Replace exact case_ids if any of the four parameter-encoded
ones do not exist in `configs/splits/split_v1.json`; the manifest is the
authority. The Baseline case is named literally `Baseline`, NOT the
parameter-encoded form (Session 4 report flagged this discrepancy).

Why this subset:

- Spans G from -3 to +3, the full training G axis except |G|=4 (Test C).
- One case at each of the standard D values used in the runs (0, 0.5,
  1.0, 1.5).
- Both signs of Y/c (positive and negative offset from chord).
- One run3 case so we exercise the source-group variation that Session 4
  did not.
- 5 distinct c values is still small enough that case-memorisation
  remains a risk; this is intentional, since the whole point of Session 5
  is to surface that risk.

Expected encounter count: 5 cases x 4 train encounters average (some
periodic have 4, run3 have 3) = roughly 19 training encounters.
Confirmed by inspecting `configs/splits/split_v1.json` against this list.

## The variant runs

Up to four variant runs, branched on outcome. Each run takes 15 to 25
minutes wall-clock at 5000 iterations on the RTX PRO 6000 Blackwell.

### Run A: baseline default

The default Session 4 configuration on the 5-case subset for 5000
iterations. This is the run the pass criteria are written against.

```bash
python -m src.training.train_jepa \
    --partition v1 \
    --cases-from configs/cases/smoke_5cases.yaml \
    --max-iters 5000 \
    --seed 0 \
    --diagnostic-every 250 \
    --checkpoint-every 1000 \
    --log-every 25 \
    --output-dir outputs/runs/smoke5k/run_a_sigreg_bn \
    --wandb-mode online \
    --tag-suffix run_a_sigreg_bn_seed0
```

The `--cases-from` argument is new (small modification to `train_jepa.py`
if it does not already accept a YAML case list; if it does, use the
existing flag). The `--tag-suffix` adds a per-run identifier to the W&B
tag list so the analysis notebook can disaggregate runs.

Note: `--diagnostic-every 250` is denser than Session 4’s default 1000.
We want twenty diagnostic snapshots over 5k iters to actually see the
trajectory of PR and probe R^2, not just the final value.

### Run B: SIGReg + LayerNorm (conditional)

Run B fires if Run A’s PR at iter 5000 is below 0.3 * d = 9.6. This is
D17’s “first diagnostic intervention.” The only change vs Run A is the
encoder’s projection norm.

This requires a one-line conditional in `src/models/encoder.py` or a
config flag (`--projection-norm layernorm`). Implementation choice: add
the flag to the entrypoint and pass it down. Keep BatchNorm as default
so D17 stays the canonical recipe.

```bash
python -m src.training.train_jepa \
    [...same as Run A...] \
    --projection-norm layernorm \
    --output-dir outputs/runs/smoke5k/run_b_sigreg_ln \
    --tag-suffix run_b_sigreg_ln_seed0
```

### Run C: VICReg + BatchNorm (conditional)

Run C fires if Run B’s PR at iter 5000 is below 0.3 * d, OR if Run A
clears the PR criterion but Run B does too (in which case BatchNorm vs
LayerNorm is uninformative and we want to know whether SIGReg itself is
the limit). The change is the anti-collapse module.

```bash
python -m src.training.train_jepa \
    [...same as Run A...] \
    --anticollapse vicreg \
    --output-dir outputs/runs/smoke5k/run_c_vicreg_bn \
    --tag-suffix run_c_vicreg_bn_seed0
```

The `--anticollapse vicreg` flag swaps the JEPA wrapper’s anti-collapse
module at construction time. The auto-fallback controller is still
instantiated but should never fire (already VICReg). This is just
another “off-default” configuration switch.

### Run D: VICReg + LayerNorm (conditional, last resort)

Run D fires only if Runs A, B, and C all fail the PR criterion. At
that point, neither the regulariser nor the projection norm is the
cause; the data scale or the encoder architecture is the cause, and
that becomes the diagnostic.

```bash
python -m src.training.train_jepa \
    [...same as Run A...] \
    --anticollapse vicreg \
    --projection-norm layernorm \
    --output-dir outputs/runs/smoke5k/run_d_vicreg_ln \
    --tag-suffix run_d_vicreg_ln_seed0
```

### Run E (optional): seed variance

If Run A (or whichever of A-D first clears the PR criterion) is
successful, run a second seed to check variance:

```bash
python -m src.training.train_jepa \
    [...same as the winning variant...] \
    --seed 42 \
    --output-dir outputs/runs/smoke5k/run_a_sigreg_bn_seed42 \
    --tag-suffix run_a_sigreg_bn_seed42
```

This adds 15 to 25 minutes and confirms the result is not a lucky seed.
Single-seed conclusions are common in deep-learning papers but bad
science; the variance run is cheap insurance and is the right scope
for an honest paper.

### Stopping conditions

For each run, the entrypoint runs the full 5000 iters even if a metric
crashes; we want the full trajectory. There is no early termination.
Total wall-clock for the worst case (A + B + C + D + E variance) is
roughly 2 to 2.5 hours, which fits inside one focused session.

## Analysis framework

The interesting work happens after the runs. Build
`notebooks/01_smoke_5k_analysis.ipynb` to:

### Section 1: Load all variant runs from W&B

Use `wandb.Api()` to fetch the runs whose tag-suffix matches
`run_[a-e]_*` in the `partition_v1` group. Pull the full history of
all logged metrics. Confirm all runs have the same step indices.

### Section 2: Loss curve comparison

One figure with three panels:

- L_pred over iters, one line per variant
- L_roll over iters, one line per variant
- L_anticollapse over iters, one line per variant (SIGReg and VICReg
  have different scales, label clearly)

Y-axis log scale. Mark iter 5000.

### Section 3: Latent health diagnostics

A 2x2 figure:

- PR(z) over iters, one line per variant, horizontal lines at 0.3 * d
  (auto-fallback threshold) and 0.5 * d (Session 5 pass criterion).
- Probe R^2 overall over iters, one line per variant, horizontal line
  at 0.5 (Session 5 pass criterion) and 0.7 (auto-fallback ceiling).
- Per-dimension variance histogram at iter 5000, one panel per variant
  (small multiples).
- Per-c-component probe R^2 (r2_G, r2_D, r2_Y) over iters, three lines
  per variant. This disaggregation is informative: r2_G is often easy
  (G is the dominant axis of variation) and r2_Y is often hard (Y is
  fine-grained); seeing only r2_G high tells us something different
  from all three high.

### Section 4: The combinatorial pass criterion

For each variant, compute and report a 2x2 outcome table at iter 5000:

|        |probe_R2 > 0.5                  |probe_R2 <= 0.5                        |
|--------|--------------------------------|---------------------------------------|
|PR > 16 |**healthy**                     |regulariser too strong / encoder broken|
|PR <= 16|trivial solution (collapse to c)|dead encoder                           |

Print which quadrant each variant landed in. This is the methodological
finding of Session 5.

### Section 5: Latent space exploration (qualitative)

Load the final checkpoint of the winning variant (or the best of the
four). Encode all 5 cases worth of Test A held-out encounters. Run
PCA on the resulting (N, d=32) latent set. Plot the first two
principal components, coloured by (G, D, Y).

If the latent is healthy, points coloured by G should occupy clearly
separated regions in PCA space, and the within-case (impact phase)
variation should be a smooth trajectory through those regions.

If the latent is collapsed-to-c, the PCA will show 5 tight clusters
with no within-case structure.

If the latent is broken, the PCA will show no structure related to
either c or impact phase.

This is qualitative but it is the cheapest visualisation we have until
the decoder exists.

### Section 6: Decision and next session

The notebook’s final cell prints a decision string of the form:

```
Session 5 outcome: <one of>

  HEALTHY      - variant <X> clears both PR > 16 and 0.5 < probe_R^2 < 0.7
                 -> proceed to Session 6 (Hydra + lambda bisection)
                 -> the WINNING variant becomes the Session 6 default
                 -> note: this would partially DISCONFIRM H4 at the 5-case
                    scale; the LeWM Two-Room precedent did not have the
                    LayerNorm/VICReg interventions tried systematically

  TRIVIAL      - all variants land in the PR <= 16 AND probe_R^2 > 0.7 quadrant
                 -> H4 confirmed at 5-case scale; the LeWM Two-Room failure
                    replicates on physics data, AND the LayerNorm and VICReg
                    interventions do not recover from it.
                 -> this is the methodologically interesting NEGATIVE result
                    and a direct contribution to the paper.
                 -> proceed to Session 5.5 (expand to 10-12 cases) to test
                    whether the failure is data-scale-bound or structural.
                 -> if Session 5.5 still TRIVIAL, the failure is structural
                    in SIGReg for low-intrinsic-dim physics; proceed to
                    Session 6 with this as a documented limitation, propose
                    the rule revision in D28 (drop the probe_R^2 conjunct,
                    OR use case-conditional probe), and continue to lambda
                    bisection with the understanding that no lambda value
                    will rescue the trivial-solution mode.

  PARTIAL      - some but not all variants clear PR > 16. Variants that do
                 clear it are healthy; the others reveal which axis matters.
                 -> if BatchNorm-to-LayerNorm fixes it: D17 confirmed at
                    smoke scale, paper claims this as a contribution.
                 -> if SIGReg-to-VICReg fixes it: SIGReg specifically (not
                    BatchNorm) is the issue at this data scale; paper
                    reports this comparison.
                 -> proceed to Session 6 with the winning variant.

  WEAK         - variant <X> has healthy PR but probe_R^2 below 0.5
                 -> the encoder is spreading but not capturing c
                 -> proceed to Session 5.5 with phi_t added (D16 alternative)
                 -> new D-entry recording phi_t enabled

  DEAD         - all variants land in PR <= 16 AND probe_R^2 < 0.5
                 -> structural problem (data, encoder, predictor, or loss)
                 -> stop, debug, do not proceed to Session 6
```

This decision tree IS the deliverable. The 5k smoke is not “did it pass” but
“what did it tell us about the methodology”. Under H4, TRIVIAL is the
expected outcome and is a successful Session 5 in its own right; it is a
positive datapoint for the paper’s contribution claim 3.

## Pass criteria for Session 5 as a session

Three things must be true at the end of Session 5 for the session itself
to count as complete:

1. All four sanity checks (1-4) pass cleanly. If any fails, it is a code
   bug, not a smoke result, and we fix it before any variant runs.
1. At least Run A completes 5000 iterations with finite loss and a clean
   W&B run upload. We must have data, regardless of whether the data shows
   pass or fail.
1. The analysis notebook produces a decision string (HEALTHY / TRIVIAL /
   WEAK / DEAD), and a corresponding HANDOFF D-entry is appended.

Note that “the smoke meets the original pass criteria” is NOT in this
list. The session can succeed methodologically while the original
numerical criteria fail. That is the central point.

## Modifications to `src/training/train_jepa.py`

Three small additions, all backwards-compatible with Session 4’s command
line:

1. `--cases-from <path-to-yaml>`: read a YAML file with a `cases` list
   and use that as the case subset. If both `--cases` and `--cases-from`
   are given, raise.
1. `--projection-norm {batchnorm,layernorm}`: pass this through to the
   encoder constructor. Default `batchnorm` (D17). The encoder needs a
   matching constructor argument; if it does not have one, add it now
   (a small change to `src/models/encoder.py`, with a corresponding
   tweak to `test_encoder_projection_is_batchnorm` to make the assertion
   conditional on the constructor flag).
1. `--anticollapse {sigreg,vicreg}`: pass this through to the JEPA
   constructor. Default `sigreg`. The JEPA constructor already takes an
   `anticollapse` module argument; the entrypoint just selects which
   one to instantiate.
1. `--tag-suffix <str>`: append to the W&B tag list. Default empty.

These four flags are additive; no Session 4 behaviour changes. The slow
integration test still passes unmodified (it does not pass any of the
new flags, so all defaults apply).

## D-entries to record (depending on outcome)

### Always (regardless of outcome)

**D24**: Session 5 case subset is the five cases listed in
`configs/cases/smoke_5cases.yaml`, deliberately chosen to span the G
axis, both Y signs, and both source groups. Rationale: random selection
gives unstable comparison across sessions; pinning the subset means
Session 6+ debugging is reproducible. The subset is NOT part of the
partition manifest (it is not a split), it is a runtime case selector.

**D25**: `--projection-norm` flag added to `train_jepa.py`. Default
remains BatchNorm per D17. The flag exists so D17’s “first diagnostic
intervention” is reachable without code edits.

**D26**: `--anticollapse` flag added to `train_jepa.py`. Default
remains SIGReg per D5. The flag exists so manual VICReg comparison runs
do not require code edits.

### Conditional (depending on outcome)

**D29 (always, regardless of outcome)**: PLDM baseline priority is
conditional on Session 5 outcome. The LeWM paper (Maes et al.,
arXiv:2603.19312, Section 5) reports: “In the simpler Two-Room
environment, PLDM and DINO-WM outperform LeWM, which may be explained by
the SIGReg regularization encouraging a Gaussian distribution in a
high-dimensional latent space, while the intrinsic dimensionality of the
environment is much lower.” Our estimated intrinsic dimension (~5 to 10
per D4) is closer to Two-Room than to Push-T. Therefore: if Session 5
shows the TRIVIAL outcome (the LeWM Two-Room failure mode replicates on
our data and no intervention recovers from it), PLDM becomes the
priority comparator IMMEDIATELY after Session 5, before either Session
5.5 (expand cases) or Session 6 (Hydra plus lambda bisection). This is
recorded ahead of time because it changes the implicit ordering of
“baselines are parallel work” (D8) into “PLDM is conditional priority”
when the trivial-solution mode appears. The paper’s contribution claim 3
correspondingly sharpens from “SIGReg as a JEPA-for-science methodology”
to “the regime-dependent SIGReg-PR diagnostic, with PLDM as the
recommended fallback for low-intrinsic-dim domains.”

**D27 (if outcome is HEALTHY)**: Session 5 5k smoke cleared pass criteria
under variant X. Numbers are <PR, probe R^2, SIGReg loss>. Session 6
proceeds with variant X as the default.

**D27 (if outcome is TRIVIAL)**: Session 5 5k smoke collapsed under all
variants at the 5-case data scale. The PR <= 16 AND probe_R^2 > 0.7
pattern is consistent across the regulariser axis (SIGReg vs VICReg)
and the projection norm axis (BatchNorm vs LayerNorm), indicating the
pathology is data-scale-bound, not regulariser- or norm-bound.
Per D29, Session 5.PLDM follows.

**D27 (if outcome is PARTIAL)**: Session 5 5k smoke cleared the PR
criterion under variant X but not under the default. Numbers are
<PR, probe R^2, SIGReg loss> for each variant. Session 6 proceeds
with variant X.

**D27 (if outcome is WEAK)**: Session 5 5k smoke has healthy PR but
low probe R^2 under variant X. The encoder is anti-collapsed but does
not capture c-relevant information at this data scale. Next step:
Session 5.5 with phi_t conditioning (D16 alternative) enabled.

**D27 (if outcome is DEAD)**: Session 5 5k smoke produced no healthy
variant. Structural problem; the wiring assumption from Session 4 is
violated somewhere. Do not proceed to Session 6 until root-caused.

**D28 (if rule revision is judged worthwhile)**: The auto-fallback condition
`PR < 0.3 * d AND probe_R^2 < 0.7` is conjunctive by design (catches the
“latent collapsed AND useless” worst case). Session 5 data shows the
trivial-solution mode (PR low AND probe_R^2 HIGH) falls outside the rule.
Two possible revisions:
(a) `PR < 0.3 * d` alone (drop the probe_R^2 conjunct, fire whenever the
latent collapses regardless of probe behaviour).
(b) Use a case-conditional probe: measure probe R^2 on held-out CASES
(Test B), not on held-out encounters within seen cases (Test A
sub-batches). The trivial-solution mode should drop probe R^2 on
Test B because the encoder has memorised seen c-values; held-out
cases would not be in the memorisation lookup.
Defer either revision to a discussion before Session 6; record now so it
is on the next session’s agenda. Cite LeWM Section 4 for the precedent
discussion of SIGReg failure modes.

## Out of scope for Session 5

- **Hydra configs**. Defer to a dedicated short session before Session 6,
  or fold into Session 6 itself. The argparse interface is sufficient for
  Session 5’s variant runs.
- **`torch.compile()`**. Same reasoning; defer to the Hydra session so
  the speedup is measured against an apples-to-apples non-compiled
  baseline.
- **The visualisation decoder**. The decoder is the right tool for asking
  “what does the latent represent” once we have a non-collapsed latent.
  Build it after Session 5 produces such a latent, not before.
- **Baselines** (POD, Fukami AE, Solera-Rico, PLDM). POD, Fukami AE, and
  Solera-Rico are parallel work and do not block. **PLDM is conditional**:
  if Session 5’s outcome is TRIVIAL, PLDM is promoted to the priority
  comparator (Session 5.PLDM, see “After Session 5 lands”). The LeWM
  paper documents PLDM outperforming LeWM on Two-Room (the low-intrinsic-
  dim environment). If our data shows the same regime, running PLDM
  immediately is more informative than expanding the case subset, because
  it directly tests the regime-dependent contrast the paper claims.
- **Symmetry augmentation** (Open Q6). Adding paired symmetry samples
  could roughly double the effective data and might rescue a TRIVIAL
  outcome. Consider for Session 5.5 only if data-scale-bound collapse is
  the diagnosis.
- **Frame-skip sweep** (Open Q2). Frame-skip 2 is the default; an
  ablation to frame-skip 1 is one configuration knob worth checking, but
  only if Session 5 cleared its pass criteria and we want a finer
  understanding. Defer.
- **Auxiliary observable head** (Open Q4). Same logic: a feature for
  later if the main path works.

## Expected duration

The session is long and exploratory.

- Reading and pre-run sanity checks: 30 to 45 minutes.
- Run A (5k iters, 5 cases): 15 to 25 minutes wall-clock.
- Run B (conditional): 15 to 25 minutes if triggered.
- Run C (conditional): 15 to 25 minutes if triggered.
- Run D (conditional): 15 to 25 minutes if triggered.
- Run E (optional seed variance): 15 to 25 minutes.
- Analysis notebook + decision string: 45 to 90 minutes.
- HANDOFF entries + session report: 30 minutes.

Total realistic span: 3 to 5 hours. This is longer than Sessions 2-4 because
the session is methodological, not just coding. Plan for this; do not
attempt it as a quick add-on between meetings.

Wall-clock for the 5k runs is based on Session 4’s 200-iter timing of
roughly 30 seconds. Naive scaling gives 5000/200 * 30 = 750 seconds =
12.5 minutes per run, plus 20 percent overhead for the denser diagnostics
and the W&B online logging. So 15 to 18 minutes is a reasonable estimate
per run.

## If something is unclear

The arXiv MCP plugin is enabled. The most likely sources of confusion
during analysis are:

1. **Interpreting PR**: a low PR can mean complete collapse (PR = 1) or
   low-rank collapse to a few dimensions (PR = 4 to 6, the intrinsic
   dim guess). The LeJEPA paper Fig 1 distinguishes these.
1. **Probe R^2 disagreement across c components**: r2_G high but r2_D
   low does not mean the encoder is broken. G has only 7 levels (-3 to
   +3 in our split); D has 4 levels. Some components are easier to
   probe because of cardinality and physical effect size, not because
   of the regulariser.
1. **VICReg’s variance term firing on standardised latents**: BatchNorm
   at the projection forces var = 1 per dimension, which makes VICReg’s
   variance hinge always satisfied. This is a noted interaction; the
   covariance term remains informative. The VICReg unit tests already
   exercise this; just be aware in interpretation.

If after consulting the source there is genuine ambiguity about how to
interpret a result, record the question as an OPEN ITEM in the session
report and proceed; do not block analysis on a paper consultation.

## After Session 5 lands

The next session depends on the outcome:

- HEALTHY -> Session 6: Hydra + torch.compile + lambda bisection. The
  PLDM baseline can wait; it is a comparator, not a blocker.
- PARTIAL -> Session 6 with the winning variant as default; PLDM still
  parallel work.
- TRIVIAL -> **Session 5.PLDM**: train the PLDM baseline on the same 5-case
  subset, before Session 5.5 or Session 6. LeWM Section 5 documents that
  PLDM outperforms LeWM on Two-Room precisely because of the low
  intrinsic dimensionality. If PLDM also collapses on our data at this
  scale, the failure mode is not SIGReg-specific and Session 5.5
  (expand to 10-12 cases) is the right next move. If PLDM trains
  successfully where SIGReg failed, that is the headline result for
  contribution claim 3 (the regime-dependent SIGReg-PR diagnostic).
  Either way the answer reshapes Session 6.
- WEAK -> Session 5.5: add phi_t conditioning (D16 alternative), rerun
  smoke on the same 5 cases.
- DEAD -> debug session, no number. Stop the forward march until the
  problem is diagnosed.

Session 2, 3, and 4 unit tests must remain green throughout whatever
session comes next. Session 5 adds the sanity-check tests and possibly
relaxes `test_encoder_projection_is_batchnorm` to be conditional on the
constructor flag; that is the only test modification expected.

## Decision references (existing)

- D5 (HANDOFF): SIGReg with auto-fallback to VICReg at iter 20k.
- D17 (HANDOFF): BatchNorm at encoder projection per LeWM; LayerNorm
  is the FIRST diagnostic intervention if H4 (partial SIGReg collapse)
  bites. Run B operationalises this.
- D16 (HANDOFF): predictor cond_dim = 3, no phi_t default. WEAK outcome
  triggers the alternative (phi_t in conditioning).
- D19 (HANDOFF): RTX 6000 Blackwell only.
- D20, D14, D15 (HANDOFF): partition v1 has 47 cases / 230 encounters.
- D21 (HANDOFF): V-JEPA 2-AC-faithful scheduled sampling, H_roll = 8.
- D22 (HANDOFF): VICReg coefficients with invariance term dropped.
- D23 (HANDOFF): slow integration tests via `--runslow`.
- D24-D29 (this session): see above.
