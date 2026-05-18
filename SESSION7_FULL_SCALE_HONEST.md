# SESSION7_FULL_SCALE_HONEST.md

Session 7 plan.

Last updated: 2026-05-18.

## Framing

Session 6 (D39) characterised PLDM+OBS as the smoke-scale winner under
a 5-case subset, with the regulariser-asymmetry finding (“observable
head rescues SIGReg, marginally helps PLDM”) as the headline. Sessions
5, 5.PLDM, and 6 collectively characterised the failure modes of pure
JEPA at small scale across 8 axis variations. Test B (parametric
interpolation) and Test C (extrapolation to G=4) have not been
evaluated in any session.

Session 7 commits to three production-scale 20k-iter runs on the full
v1.2 train partition (49 cases, ~138 train encounters per D35) with
full Test A / Test B / Test C evaluation on each. The session’s
deliverable is the complete metric table across all three test splits
for all three configurations.

Three runs span the methodologically interesting axes:

- **R1: PLDM + observable head, BatchNorm projection.** D39’s bundled
  smoke winner scaled to 41 train cases. The headline configuration.
- **R2: PLDM only, eta = 0, BatchNorm projection.** No observable
  head. Tests whether the observable head is necessary or decorative
  on top of PLDM at full scale. D39’s smoke comparison was inconclusive
  on this; Test B at scale should resolve it.
- **R3: SIGReg + observable head, BatchNorm projection.** Tests whether
  the OBS-rescues-SIGReg finding from D39 persists at full scale.
  Critical for the paper’s regulariser-asymmetry contribution claim.

Test B is the primary success metric. Test A is a sanity check (the
model has seen these cases; it should do well in-distribution). Test C
is a stretch goal (extrapolation to G=4; failure is the expected
outcome at this training scale).

Honest checkpoint discipline: the full metric table for all three runs
across all three test splits is reported with no cherry-picking. If
R1 looks great on Test A and Test C but bad on Test B, the report
says so clearly. If all three runs underperform on Test B, the paper
reframes around the diagnostic contribution; we do not hide that
finding inside an optimistic narrative.

R0 (pure SIGReg+BN at full scale) is **deferred** to Session 8 as a
contingent task. If Session 7’s ALL_FAIL outcome lands, R0 strengthens
the “pure JEPA fails across scales” paper claim. If Session 7 lands
healthy on any configuration, R0 is unnecessary because the smoke-scale
8-axis evidence is already overwhelming.

## Session goal

Three parallel/sequential 20k-iter runs on the full v1.2 train partition
(41 cases / 138 train encounters per D35). Three test splits evaluated
on all three checkpoints. One decision string at the end.

|Run|Card            |Configuration       |Hypothesis tested                                                                                                                                               |
|---|----------------|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
|R1 |cuda:2          |PLDM + OBS, BN      |D39’s bundled winner scaled to 41 cases. The headline configuration.                                                                                            |
|R2 |cuda:3 first 5h |PLDM only, BN, eta=0|Is the observable head doing the work, or does PLDM-A alone generalise? Disambiguates D39’s “OBS marginally helps PLDM” reading at full scale.                  |
|R3 |cuda:3 second 5h|SIGReg + OBS, BN    |Does OBS rescue SIGReg at scale? D39 found this rescue at 5 cases. R3 tests whether it persists at 41 cases. Critical for the regulariser-asymmetry paper claim.|

Pass criteria for the session:

1. All three runs complete 20k iterations with finite losses and clean
   W&B uploads.
1. The full evaluation matrix (3 runs × 3 test splits × all metrics)
   completes without errors.
1. The decision string is computed and a session report is written.

The substantive outcome (which configurations generalise to Test B; do
they generalise to Test C; is the regulariser asymmetry confirmed at
scale) is the deliverable. A clean negative result is a successful
session.

## What I am explicitly NOT doing in this session

These are useful and may land eventually, but they would confound the
central comparison.

- **No BN-vs-LN axis.** If R1 fails, Session 7B tests LN as a
  remediation. If R1 passes, BN is validated.
- **No eta sweep.** All three runs use eta = 0.01 (when applicable).
  The eta sweep is Session 8 contingent on R1 success.
- **No L sweep.** All three runs use L = 32 (1.6 t/c at frame-skip 1
  per D34).
- **No VICReg runs.** D27 already characterised VICReg’s SPREAD_TRIVIAL
  failure at smoke scale; re-testing at full scale would burn 5 hours
  on a known-failing axis.
- **No pure SIGReg run** (R0 deferred to Session 8 contingent).
- **No Hydra refactor, no torch.compile, no lambda bisection.** All
  deferred.

## Pre-flight checks (mandatory, ~30 min)

Before launching anything, four sanity checks.

### Check A: full-partition data loader

```bash
python -m src.training.sanity_checks --check data_loader \
    --cases all_train --num-batches 20
```

Confirm: 138 train encounters load cleanly, no NaN/Inf, omega magnitudes
in (-100, 100), CL_future magnitudes in (-2, 3). If any check fails,
STOP.

### Check B: `--all-train` flag works on both entrypoints

If `train_baseline.py` and `train_jepa.py` do not currently accept
`--all-train`, add it. Roughly 15 lines per entrypoint plus one unit
test each. Reads the train bucket from the partition manifest directly
rather than requiring an explicit 138-line YAML.

### Check C: two-card configuration

```bash
python -c "import torch; print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
```

Should return two RTX 6000 names per D40. If only one card is visible,
the session extends to ~15 hours of sequential execution; consider
postponing R3 to Session 7B in that case.

### Check D: Test B and Test C splits are intact

```bash
python -c "
from src.data.episode_dataset import EpisodeDataset
import json
manifest = json.load(open('configs/splits/split_v1.json'))
print('Test B cases:', len(manifest['test_b_cases']))
print('Test C cases:', len(manifest['test_c_cases']))
print('manifest sha256:', manifest.get('sha256', '<not stored>'))
"
```

Should report 6 Test B cases (~28 encounters) and 4 Test C cases (~24
encounters) per D7/D9. The sha256 should match the v1.2 partition
recorded in D35. If counts disagree, STOP and investigate; the
evaluation is meaningless without correct splits.

## Step 1: launch three production runs

### R1: PLDM + OBS, BN projection

```bash
python -m src.training.train_baseline \
    --baseline pldm \
    --gpu 0 \
    --partition v1 \
    --all-train \
    --max-iters 20000 \
    --seed 0 \
    --observable-head cl_future \
    --observable-head-weight 0.01 \
    --observable-head-deltas 8 16 24 \
    --projection-norm batchnorm \
    --diagnostic-every 500 \
    --checkpoint-every 2000 \
    --log-every 50 \
    --output-dir outputs/runs/session7/run_r1_pldm_obs_bn \
    --wandb-mode online \
    --tag-suffix run_r1_pldm_obs_bn_seed0_full
```

Headline configuration. D39’s bundled winner extended to full data.
~5 hours on the first RTX 6000 (`--gpu 0`, D40).

### R2: PLDM only, eta=0, BN projection (no observable head)

```bash
python -m src.training.train_baseline \
    --baseline pldm \
    --gpu 1 \
    --partition v1 \
    --all-train \
    --max-iters 20000 \
    --seed 0 \
    --observable-head none \
    --projection-norm batchnorm \
    --diagnostic-every 500 \
    --checkpoint-every 2000 \
    --log-every 50 \
    --output-dir outputs/runs/session7/run_r2_pldm_only_bn \
    --wandb-mode online \
    --tag-suffix run_r2_pldm_only_bn_seed0_full
```

The OBS-vs-no-OBS control. Same PLDM configuration as R1 but without
the observable head. If R2 matches R1 on Test B, PLDM alone is
sufficient and the observable head is decorative (paper claim shifts
to “PLDM works at scale; observable augmentation is helpful but not
necessary”). If R1 substantially beats R2 on Test B, the observable
head is doing real work and the integrated bundle is the right recipe.

~5 hours on the second RTX 6000 (`--gpu 1`, D40) starting concurrently with R1.

### R3: SIGReg + OBS, BN projection

```bash
python -m src.training.train_jepa \
    --gpu 1 \
    --partition v1 \
    --all-train \
    --max-iters 20000 \
    --seed 0 \
    --observable-head cl_future \
    --observable-head-weight 0.01 \
    --observable-head-deltas 8 16 24 \
    --projection-norm batchnorm \
    --anticollapse sigreg \
    --lambda-sigreg 0.1 \
    --diagnostic-every 500 \
    --checkpoint-every 2000 \
    --log-every 50 \
    --output-dir outputs/runs/session7/run_r3_sigreg_obs_bn \
    --wandb-mode online \
    --tag-suffix run_r3_sigreg_obs_bn_seed0_full
```

Regulariser-asymmetry test at full scale. D39 found OBS rescued SIGReg
at 5 cases (CL r² from -0.02 to 0.95). R3 tests whether this rescue
persists at 41 cases. Sequential on the second RTX 6000 (`--gpu 1`, D40)
after R2 finishes. ~5 hours.

### Launcher script

`scripts/launch_session7.sh` runs all pre-flight checks, launches R1
with `--gpu 0` in background, launches R2 with `--gpu 1` in background,
waits for R2, launches R3 with `--gpu 1`, waits for all to finish, then
calls Step 2. Logs everything to `outputs/runs/session7/launch.log` with
explicit PIDs and W&B run IDs.

## Step 2: full evaluation suite (~2 hours)

Single notebook `notebooks/05_session7_full_evaluation.ipynb`. The
notebook produces ONE complete table that is the session’s deliverable.

### The complete table (the honest checkpoint)

For each run × each test split × each metric:

|Run             |Split |PR_all|PR_within|r2(z→c)|r2_dyn_phase|r2(CL_future)|(c,t) baseline|delta|
|----------------|------|------|---------|-------|------------|-------------|--------------|-----|
|R1 PLDM+OBS+BN  |Test A|?     |?        |?      |?           |?            |?             |?    |
|R1 PLDM+OBS+BN  |Test B|?     |?        |?      |?           |?            |?             |?    |
|R1 PLDM+OBS+BN  |Test C|?     |?        |?      |?           |?            |?             |?    |
|R2 PLDM only BN |Test A|?     |?        |?      |?           |?            |?             |?    |
|R2 PLDM only BN |Test B|?     |?        |?      |?           |?            |?             |?    |
|R2 PLDM only BN |Test C|?     |?        |?      |?           |?            |?             |?    |
|R3 SIGReg+OBS+BN|Test A|?     |?        |?      |?           |?            |?             |?    |
|R3 SIGReg+OBS+BN|Test B|?     |?        |?      |?           |?            |?             |?    |
|R3 SIGReg+OBS+BN|Test C|?     |?        |?      |?           |?            |?             |?    |

The (c, t) baseline column is computed PER SPLIT. At 41 train cases the
baseline r²(c, t → CL_future) on Test A will be different from the 5-case
0.902; expected to be in 0.5 to 0.7 range. On Test B the baseline
generalisation reflects how well a tiny MLP can interpolate (c, t) to
unseen c values; expected lower, perhaps 0.2 to 0.5. On Test C the
baseline extrapolation will be very low, possibly negative.

The “delta” column is `r2(z → CL_future) - r2(c, t → CL_future)`. This
is the actual signal: positive delta means the latent is doing something
the parametric lookup cannot. Negative delta means the latent is worse
than a tiny MLP at this task.

The full table is the honest checkpoint. Every number gets reported.

### Section 1: training trajectory comparison

Three sub-panels (R1, R2, R3) showing all loss components over 20k iters
with diagnostic markers every 500 iters. Confirms training stability.
If any run plateaued or crashed, mark it visibly in the figure.

### Section 2: in-sample audit (Test A)

The same static-vs-dynamic audit as Session 6 D39. PR_all, PR_within,
r2(z → c), r2_dyn_phase, r2(CL_future). Plus the (c, t) baseline at 41
cases.

Expected outcome reasoning: Test A asks “given a case the model has
seen, does it handle held-out encounters from that case?” This is
essentially an in-distribution generalisation test and should be the
easiest of the three. All three runs should clear PR_within > 4 and
have positive delta on Test A; if any does not, that run is broken
and we should not trust its Test B / Test C numbers.

### Section 3: parametric interpolation (Test B) – THE PRIMARY METRIC

The 6 Test B cases span (G, D, Y) values intermediate to training
ranges per D7. For each run:

```python
# Pseudocode
z_test_b = encoder(omega_test_b)            # (N_test_b, T, d)
# Train probe on Test A latents (in-sample)
probe_z = LinearRegression().fit(z_test_a.reshape(-1, d), cl_future_test_a.reshape(-1, 3))
# Evaluate on Test B latents (out-of-sample c values)
r2_z_test_b = probe_z.score(z_test_b.reshape(-1, d), cl_future_test_b.reshape(-1, 3))
# Same with the (c, t) baseline
probe_ct = LinearRegression().fit(np.hstack([c_test_a, t_test_a]), cl_future_test_a.reshape(-1, 3))
r2_ct_test_b = probe_ct.score(np.hstack([c_test_b, t_test_b]), cl_future_test_b.reshape(-1, 3))
delta_test_b = r2_z_test_b - r2_ct_test_b
```

This is the headline number. If delta_test_b > 0.05, the latent has
learned flow physics that generalises beyond case lookup. If delta_test_b
is 0 to 0.05, the latent has learned something marginal that we report
honestly. If delta_test_b < 0, the latent is worse than (c, t) at
generalising; this is the honest negative result.

For the paper, the key plot is delta_test_b across the three runs.

### Section 4: extrapolation (Test C, secondary)

The 4 Test C cases at |G| = 4 are out-of-distribution. Computed the same
way as Test B but with the harder split. Expected: lower numbers across
the board. Reporting honestly even if results are uniformly poor.

If any run shows delta_test_c > 0, that is a strong extrapolation
result worth a dedicated paper section. If all runs are at delta_test_c
~ 0 or negative, we note “extrapolation to G=4 is beyond the model’s
capability at this training scale, consistent with deep-learning models
on bounded parameter ranges” and move on.

### Section 5: case-conditional CL probe (cross-validation diagnostic)

Leave-one-case-out across the 41 train cases. Average r²(z → CL_future)
on each held-out case. Compare to average r²(c, t → CL_future) on the
same. Provides a second view of generalisation, this time within the
train partition rather than to Test B’s parametric-interpolation cases.

If Test B delta and leave-one-out delta agree, the result is robust. If
they disagree (Test B much better than leave-one-out or vice versa),
investigate: the Test B cases may be unusually close to or far from
training in (G, D, Y) space, biasing the result.

### Section 6: decision string

Print one of:

```
Session 7 outcome: <one of>

  STRONG_GO         - R1 delta_test_b > 0.10 AND R1 PR_within > 4 on
                       Test B. PLDM+OBS+BN generalises to interpolation.
                       Session 8: lambda bisection on PLDM's 4 weights,
                       decoder training, eta sweep, paper finalisation.

  WEAK_GO           - R1 delta_test_b in [0.03, 0.10]. Generalisation
                       is marginal but real. Session 8: same as STRONG_GO
                       but with explicit discussion of marginal Test B
                       performance in the paper.

  PLDM_ALONE_VIABLE - R2 delta_test_b > R1 delta_test_b - 0.02 (i.e.
                       PLDM without OBS matches PLDM+OBS on Test B).
                       The observable head is not necessary; PLDM alone
                       is sufficient. Paper simplifies. Session 8:
                       lambda bisection on R2's config.

  OBS_NECESSARY     - R1 delta_test_b > R2 delta_test_b + 0.05 AND
                       R3 delta_test_b > 0. The observable head is
                       doing real work, AND OBS+SIGReg also generalises.
                       Paper's regulariser-asymmetry claim becomes
                       "observable augmentation works for both
                       regularisers at full scale; PLDM is slightly
                       better but the OBS choice is independent of
                       the regulariser."

  REGULARISER_      - R1 delta_test_b > R3 delta_test_b + 0.05.
  ASYMMETRY           Observable rescue is regulariser-specific; PLDM
                       benefits more than SIGReg. Paper's contribution
                       claim 3 stands as written in D39.

  R3_WINS           - R3 delta_test_b > R1 delta_test_b. SIGReg+OBS
                       beats PLDM+OBS at scale. Unexpected; would
                       suggest the PLDM machinery interferes at scale
                       and the simpler observable-augmented SIGReg is
                       the right path. Session 8 reframes around
                       SIGReg+OBS.

  ALL_FAIL          - All three runs have delta_test_b < 0.03. None
                       generalises to Test B meaningfully beyond the
                       (c, t) baseline. The 5-case "active" finding
                       in D39 was a small-data artifact. Paper
                       reframes around the diagnostic contribution
                       (failure-mode taxonomy, case-conditional CL
                       probe) and the failure of JEPA-for-science at
                       this data quantity. Session 8 runs R0 (pure
                       SIGReg+BN at full scale) to close the loop on
                       "pure JEPA also fails at scale" before the
                       paper draft.

  TEST_B_TEST_A_    - At least one run has Test A delta > 0.1 AND
  DISCREPANCY         Test B delta < 0.03. The model overfits the
                       training cases without learning transferable
                       flow physics. Honest failure mode worth reporting.
                       Session 8 investigates why (case-similarity
                       analysis, regularisation strength, possibly the
                       "the model memorises 41 cases too" hypothesis).
```

The decision tree has 8 branches because the three-run × Test B × Test
A × Test C space genuinely has that many qualitatively distinct
outcomes. The session report explicitly names which branch the data
landed in.

## Step 3: paper drafting (parallel, ~3-4 hours during R1+R2 compute)

While R1 and R2 run (the first 5 hours of GPU), draft three sections
in `paper/sections/`:

- `section_3_methods.md`: data pipeline (D34, D35), architecture,
  loss compositions, diagnostic suite. ~4 pages.
- `section_4_failure_modes.md`: the rank-1 vs SPREAD_TRIVIAL vs DEAD
  taxonomy from Sessions 5/5.PLDM/6. ~3 pages with Figure 1 being the
  2x2 outcome table.
- `section_5_full_scale_results.md`: SKELETON. The actual results from
  Step 2 fill in after Step 2 lands. But the table templates, figure
  layouts, and narrative structure should exist as placeholders so the
  final write-up is mechanical.

Real writing, in LaTeX-friendly markdown. Step 3 overlaps Step 1’s
compute and adds no wall-clock to the session.

## Risk register (honest assessment)

|Risk                                                                           |Probability|Mitigation                                                                                                       |If it fires                                                                                            |
|-------------------------------------------------------------------------------|-----------|-----------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
|F-S regression replicates on PLDM at 41 cases (any run crashes or fails Test A)|medium     |None for R1 (BN). Session 7B can retry with LN if R1 fails.                                                      |Session 8 runs PLDM+OBS+LN at full scale as remediation. ~5 hours additional.                          |
|20k iters insufficient for convergence                                         |low-medium |Checkpoints every 2k iters. If iter-20k metrics still improving, Session 7B does a 40k extension on the best run.|One additional run continuation, no replanning.                                                        |
|Source heterogeneity (run3 + periodic + D35 cases) destabilises                |medium     |Pre-flight check A catches gross issues.                                                                         |Session 7B investigates per-source normalisation.                                                      |
|Test B is too easy / too hard (sample of 6 cases is small)                     |medium-high|Cross-check with the leave-one-out probe (Section 5).                                                            |Section 5’s leave-one-out gives a 41-fold cross-validation that is robust to Test B sampling artifacts.|
|Two-card configuration breaks                                                  |low        |Pre-flight check C catches a missing card; `--gpu` flag (D40) falls back to `--gpu 0` sequentially.              |Session extends by ~10 hours, no replanning.                                                           |
|Test C is uniformly disappointing                                              |high       |Expected. Report honestly.                                                                                       |No remediation; Test C failure is the expected outcome and is reported as a limitation.                |
|All three runs land in the same outcome cluster (e.g. all WEAK_GO)             |medium     |The three configurations are deliberately spread across the regulariser-asymmetry axis.                          |Less methodological signal but still a clean result.                                                   |

## Decisions to record (in HANDOFF.md)

**D44**: Session 7 launched three production-scale runs on the v1.2
train partition. Configurations: R1 (PLDM+OBS+BN), R2 (PLDM only,
eta=0, BN), R3 (SIGReg+OBS+BN). 20k iters each, seed 0, frame-skip 1,
L=32, eta=0.01 where applicable.

**D45**: Full evaluation suite (Test A, Test B, Test C, case-conditional
CL probe) landed in `notebooks/05_session7_full_evaluation.ipynb`. Test
B is the primary success metric per the “honest checkpoint” framing.
The complete metric table for all three runs is the session’s
deliverable.

**D46** (conditional on Step 2): Session 7 decision string is [one of
the eight outcomes]. Session 8 commits to [corresponding action].

**D47** (conditional): R1-vs-R2 delta is [reported number]. The
observable head is [necessary / decorative / helpful but not necessary]
at full scale.

**D48** (conditional): R1-vs-R3 delta is [reported number]. The
regulariser-asymmetry finding from D39 [holds / does not hold / is
regime-dependent] at full scale.

**D49** (housekeeping, in same commit): if any pre-flight check
revealed a data pipeline issue at the full 41-case scale, document
and resolve before training. CLAUDE.md “Hardware” already reflects
two RTX 6000 cards and the `--gpu {0,1}` pattern per D40.

## What I want to flag clearly before launch

Two things, in order of importance.

**Test B is the metric that matters for the paper.** Every previous
session evaluated on Test A held-out encounters within seen cases. Test
A tells us whether the model can interpolate WITHIN a case it has seen,
which is essentially an in-distribution test. Test B tells us whether
the model can interpolate ACROSS cases it has not seen, which is the
parametric-generalisation question the partition was built for.

A model can clear Test A and fail Test B by memorising 41 case labels
just like it memorised 5. The fact that we have 41 cases instead of 5
makes the memorisation shortcut harder but not impossible. Test B is
the test of whether the encoder has actually learned (G, D, Y) as
continuous parameters rather than as discrete labels.

If R1, R2, R3 all clear Test A but only some clear Test B, the report
must say so plainly. The previous “honest checkpoint” gap was treating
in-sample r²(z → CL_future) = 0.96 on a 5-case smoke as evidence of
flow-state encoding when it was at minimum consistent with c-mediated
case lookup. Session 7’s full-scale evaluation with Test B closes that
gap permanently.

**The decision tree has 8 branches.** The R1-vs-R2 axis adds the
“is observable head doing the work” question that was buried in
D39’s smoke analysis. The R1-vs-R3 axis tests the regulariser-
asymmetry claim at scale. Either could land in unexpected ways. The
PLDM_ALONE_VIABLE outcome (R2 matches R1 on Test B) would simplify the
paper. The R3_WINS outcome (SIGReg+OBS beats PLDM+OBS) would force a
rethink. The TEST_B_TEST_A_DISCREPANCY outcome explicitly catches the
“model memorises 41 cases too” failure mode that Session 7 is built
to detect.

## Expected duration

- Pre-flight checks: 30 min.
- R1 + R2 in parallel: 5 hours.
- R3 alone on cuda:3: 5 hours, starting at hour ~5.5.
- Step 2 evaluation: 2 hours, after R3 finishes.
- Step 3 paper drafting: 3-4 hours, overlaps R1+R2 compute.
- Session report + D-entries: 1 hour.

Total wall-clock: ~12-13 hours from launch. Agent-active time:
~6-7 hours (Step 3 happens during compute).

The session is too long for a single sitting. Plan for a launch in the
morning, paper drafting during compute, audit and report writing the
following day.

## After Session 7

The decision string maps deterministically to Session 8:

- **STRONG_GO**: Session 8 is lambda bisection (6-8 evaluations on R1’s
  config), eta sweep on R1, decoder training, paper finalisation.
- **WEAK_GO**: Same as STRONG_GO but with explicit Test C investigation
  and shorter horizon (40k iters not 80k full).
- **PLDM_ALONE_VIABLE**: Session 8 reframes around R2’s config; lambda
  bisection on PLDM weights without observable head; paper simplifies.
- **OBS_NECESSARY**: Session 8 does eta sweep for R1 AND R3; tests
  whether OBS is optimisable in a regulariser-independent way.
- **REGULARISER_ASYMMETRY**: Session 8 is lambda bisection on R1
  (PLDM+OBS); R3 documented as comparator; paper’s contribution claim
  3 stands.
- **R3_WINS**: Session 8 reframes around R3 (SIGReg+OBS); lambda
  bisection on SIGReg lambda; PLDM documented as failed-at-scale.
- **ALL_FAIL**: Session 8 runs R0 (pure SIGReg+BN at full scale) to
  close the “JEPA fails at scale” claim; paper reframes around the
  diagnostic contribution.
- **TEST_B_TEST_A_DISCREPANCY**: Session 8 investigates the
  memorise-at-scale failure mode (case-similarity analysis,
  regularisation strength variation, possibly more data via the run3
  expansion noted in D35).

Session 9 in all cases: decoder training, full Section 7 evaluation
suite per the architecture spec, paper figures, JFM-track manuscript
draft.

## Decision references

- D2 (HANDOFF): d = 32. Locked.
- D5, D17: anti-collapse with auto-fallback; BatchNorm projection.
- D27 (HANDOFF): Session 5 TRIVIAL-dominant outcome.
- D29, D30, D32 (HANDOFF): PLDM is conditional priority, citation
  corrected, 5-term loss verified.
- D34 (HANDOFF): frame-skip 1, dt_eff = 0.05, L = 32 = 1.6 t/c.
- D35 (HANDOFF): 49 cases / ~138 train encounters in v1.2 partition.
- D36, D37 (HANDOFF): CL canonical observable target, eta = 0.01
  placeholder.
- D40 (HANDOFF): two RTX 6000 cards canonical hardware; `--gpu {0,1}` flag.
- D39 (HANDOFF): COMBINED_REMEDIATION outcome of Session 6; PLDM as
  recommended base; regulariser asymmetry. **Session 7 tests all three
  at full scale with honest Test B reporting.**
- D44-D49 (this session): see above.
