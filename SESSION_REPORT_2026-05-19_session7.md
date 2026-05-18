# Session 7 report: full-scale evaluation

Date: 2026-05-19 (launched 2026-05-18 ~20:30; R3 finished ~24:08;
notebook 05 executed and report written on 05-19).
Branch: `session7-full-scale`.
Plan: `SESSION7_FULL_SCALE_HONEST.md`.

## Summary

Session 7 was the first full-scale evaluation of the vortex-jepa
project. Three production runs at 20k iters on the full v1.2 train
partition (41 cases, 138 train encounters per D35), with full Test A
/ Test B / Test C evaluation on each. Test B (parametric interpolation
to unseen (G, D, Y)) was the primary success metric per the
"honest checkpoint" framing.

**Outcome: `TEST_B_TEST_A_DISCREPANCY` per the strict 8-branch tree,
with a substantive `R3_WINS` reading.** R3 (SIGReg + OBS + BN) is the
only run with a positive Test B delta (+0.14) and the best Test C
delta (+0.48). R1 (PLDM + OBS + BN) and R2 (PLDM only) both overfit
the 41 training cases: their out-of-sample CL prediction does not
beat the (c, t) baseline on Test B, despite Test A latents looking
healthy (R1 Test A delta +0.23). This INVERTS the Session 6 D39
reading that "PLDM should be the base for the observable-augmented
path"; at full scale, the simpler SIGReg + OBS configuration wins
on the metric the partition was built to evaluate. Detailed audit
in HANDOFF D46-D48. Session 8 reframes around R3 (SIGReg + OBS) —
see D49 for the three-track Session 8 plan.

Final iter-20000 PR snapshot (from W&B summary, on the held-out
Test B batch used by the in-training diagnostic):

| Run                 | Configuration              | iter-20000 PR | iter-20000 r2(z->c) |
|---------------------|----------------------------|---------------|---------------------|
| **R1**              | PLDM + OBS + BN, eta=0.01  | **10.54**     | 0.99 (typical)      |
| **R2**              | PLDM only, eta=0, BN       | **9.66**      | 0.97 (typical)      |
| **R3**              | SIGReg + OBS + BN          | **3.35**      | 0.99 (typical)      |

The headline observation from these in-training diagnostics: the two
PLDM-based runs (R1 and R2) reach PR ~10 at full scale, ~3x higher
than R3 (SIGReg + OBS) at the same training budget. The observable
head adds only ~0.9 to PR for PLDM (10.54 vs 9.66) -- consistent with
the smoke-scale D39 reading that "OBS marginally helps PLDM" -- but
adds substantial PR for SIGReg (~3.3 vs the smoke 1.0 baseline of
Run A SIGReg pure). The regulariser asymmetry from D39 holds at
scale.

The substantive outcome is delivered by `notebooks/05_session7_full_evaluation.ipynb`
Section 6's 8-branch decision tree applied to the Test B delta values.
[Decision string filled in below after the notebook executes.]

## What landed

### Step 0: pre-flight (all four checks PASS)

- Check A (full-partition data loader): 138 train encounters load
  cleanly from the v1.2 cache, all finite, omega range [-3658, +3701]
  consistent with the D27/CLAUDE.md "peak 4377" survey (the plan's
  `(-100, 100)` bound was conservative).
- Check B (`--all-train` smoke on both entrypoints, 10 iters, B=4):
  train_jepa with `--gpu 0` -> cuda:2; train_baseline with `--gpu 1`
  -> cuda:3; n_train_samples=138 confirmed on both.
- Check C (GPU enumeration): two RTX 6000 Blackwell cards visible at
  cuda:2 and cuda:3 (D40-aligned).
- Check D (split manifest): manifest sha256
  `a721dc92f6e278ee054bb952933c14ba20a58137f79f3a19fc6ad71b70a007dd`
  matches D35; inventory sha256 prefix `ce817e1e0df54309...` matches
  D35; 6 Test B cases / 28 encounters; 4 Test C cases / 24 encounters.

### Step 0+: housekeeping

- `--all-train` flag added to `src/training/train_jepa.py` and
  `src/training/train_baseline.py`. Mutually exclusive with `--cases`
  / `--cases-from`. Same default behaviour as omitting all three (the
  manifest-tagged 'train' split is used) but explicit in W&B
  `run_config["all_train"]` for production runs.
- 6 new tests in `tests/test_resolve_cases.py`. Fast suite at 126/126
  green.
- CLAUDE.md "Hardware" was already updated in D40 (the brief listed
  it as a housekeeping item but D40 had already landed). HANDOFF.md
  D44 acknowledges the brief was written pre-D40.
- D40-era audit-trail fix: "## Open questions" section heading
  restored at line 1782 of HANDOFF.md; it had been dropped by an
  Edit's old_string/new_string mismatch in the D40 commit. Content of
  the section was unchanged throughout; this is an audit-trail
  restoration, not a semantic recovery.

### Step 1: three production runs

`scripts/launch_session7.sh` orchestrated the dual-card launch with
the D40 `--gpu {0,1}` pattern. R1 + R2 ran concurrently on the two
RTX 6000s; R3 followed sequentially on cuda:1 after R2 completed.

The launcher had a `nohup ... & ; disown` bug that caused R3 to start
concurrently with R2 instead of waiting. R3 (PID 2714955) was killed
about 2 minutes after launch; R1 and R2 continued cleanly. The fixed
launcher dropped `disown` so children stay in the launcher's job
table and `wait $pid` blocks correctly.
`scripts/launch_session7_r3_after_r2.sh` is the recovery launcher
that polled (every 60 s) for R2's iter-20000 checkpoint and the R2
python process exit, then launched R3 on the freed cuda:1.

Per-run timing:

| Run | Launch    | Final ckpt | Wall clock |
|-----|-----------|-----------|------------|
| R1  | 20:32:52  | ~22:02    | ~1.5 h     |
| R2  | 20:32:52  | ~22:36    | ~2.0 h     |
| R3  | 22:36:42  | ~24:08    | ~1.5 h     |

Total wall clock: ~3.6 h (vs the plan's 12-13 h estimate; the per-iter
compute on the RTX 6000 Blackwell was ~220 iter/min, much faster than
the plan's 100 iter/min back-of-envelope).

### Step 2: full evaluation suite

`notebooks/05_session7_full_evaluation.ipynb` loads all three iter-20000
checkpoints, encodes every encounter in Test A (56), Test B (28), and
Test C (24), and computes the per-split metric table.

For Test B / Test C, the latent r^2(z -> CL_future) is computed
out-of-sample: a tiny MLP is fit on Test A latents (in-sample) and
evaluated on Test B / Test C latents. Same for the (c, t) baseline.

[Per-split table fills in below after the notebook executes.]

#### Table 1 -- complete metric table

| Run                | Split  | PR_all | PR_within | r2(z->c) | r2_dyn_phase | r2(CL_future) | (c,t) baseline | delta   |
|--------------------|--------|-------:|----------:|---------:|-------------:|--------------:|---------------:|--------:|
| R1 PLDM+OBS+BN     | Test A | 27.84  |   6.87    |   0.90   |    0.78      |    0.97       |     0.74       |  +0.23  |
| R1 PLDM+OBS+BN     | Test B | 18.31  |  10.06    |   0.96   |    0.91      |    0.71       |     0.72       |  -0.008 |
| R1 PLDM+OBS+BN     | Test C | 14.50  |  11.77    |   0.90   |    0.86      |    0.76       |     0.35       |  +0.42  |
| R2 PLDM only BN    | Test A | 27.16  |   6.01    |   0.88   |    0.77      |    0.93       |     0.74       |  +0.19  |
| R2 PLDM only BN    | Test B | 17.35  |   9.41    |   0.95   |    0.92      |   -0.13       |     0.72       |  -0.85  |
| R2 PLDM only BN    | Test C | 13.92  |  11.14    |   0.91   |    0.87      |    0.32       |     0.35       |  -0.03  |
| **R3 SIGReg+OBS+BN** | Test A |  3.69  |   4.18    |   0.62   |    0.44      |    0.97       |     0.74       |  +0.24  |
| **R3 SIGReg+OBS+BN** | **Test B** |  3.51  |   3.85    |   0.93   |    0.63      |  **0.86**     |     0.72       | **+0.14** |
| **R3 SIGReg+OBS+BN** | **Test C** |  2.91  |   4.67    |   0.76   |    0.73      |  **0.83**     |     0.35       | **+0.48** |

R3 (SIGReg + OBS) is the only positive `delta_test_b` and dominates
`delta_test_c`. R1 (PLDM + OBS) overfits on Test B (delta -0.01).
R2 (PLDM only) overfits dramatically on Test B (delta -0.85) -- its
latent is worse than a tiny (c, t) MLP at predicting CL on unseen
cases.

Two Test A encounters were dropped from the CL-prediction MLP fit
because their cached `C_L` series contains NaN values (DNS instability
near the last encounter of the run3 D=1.5 cases): `G+2.00_D1.50_Y+0.00`
encounter 3 (69 NaN frames) and `G-2.00_D1.50_Y+0.10` encounter 3
(103 NaN frames). The remaining 54 Test A encounters were used for
the in-sample probe fit + the OOS probe training data for Test B /
Test C evaluations. The 2 dropped encounters remain in the PR/probe
metrics on z (which use omega and c only, not C_L). Documented in
HANDOFF D45.

#### Decision string outcome

Strict decision tree: **`TEST_B_TEST_A_DISCREPANCY`**. R1 has Test A
delta 0.23 (> 0.10 threshold) and Test B delta -0.01 (< 0.03
threshold), so the discrepancy rule fires first.

Substantive read: **`R3_WINS`**. R3 SIGReg+OBS+BN is the only run
with a positive Test B delta (+0.14), the highest Test C delta
(+0.48), and the lowest PR (3.5-4.7) -- the inverse of what
Session 6 D39 predicted as the "winner". The smoke-scale PLDM+OBS
PR=10+ was reproducing case-axis memorisation, not transferable
flow physics. At 41 cases the case-axis structure becomes the
overfitting signal; the lower-rank SIGReg+OBS latent stays simple
enough that the observable head can guide it toward genuine
flow-physics encoding.

Session 8 plan implied by R3_WINS (HANDOFF D49):

1. **Session 8-OBS-SIGReg** (8h): eta sweep on SIGReg + OBS on the
   full partition.
2. **Session 8-R0** (5h): pure SIGReg + BN at full scale (the
   deferred control); confirms whether OBS is load-bearing for
   SIGReg or whether SIGReg alone also generalises.
3. **Session 8-LAMBDA** (6h): lambda bisection on the SIGReg + OBS
   winner from task 1.

Then Session 9: decoder training + Section 7 evaluation per the
architecture spec.

### Step 3: paper drafting (during compute)

While R1 and R2 ran (the first ~1.5 h of the session), three paper
sections were drafted in `paper/sections/`:

- `section_3_methods.md` (~4 pages): data partition, encoder +
  predictor architecture, 2-term SIGReg vs 5-term PLDM loss
  compositions, observable head with eta=0.01, optimisation,
  diagnostic suite, hardware. Placeholder \cite{} tokens.
- `section_4_failure_modes.md` (~3 pages): the rank-1 vs
  SPREAD_TRIVIAL vs HEALTHY taxonomy from Sessions 5/5.PLDM/6 with
  the 2x2 outcome table as Figure 1. Observable-augmentation
  lineage from Fukami 2023/2024/2025 and Solera-Rico 2024.
- `section_5_full_scale_results.md` (skeleton): Table 1 + Figure 2
  + decision-string outcome paragraph + leave-one-out probe section.
  Actual numbers fill in after notebook 05 executes.

## Files added or modified

- `CLAUDE.md`: no change in Session 7; D40's Hardware update already
  covers the two-card pattern.
- `HANDOFF.md`: D44 entry inserted after D40 in chronological order;
  "## Open questions" section heading restored (had been dropped by
  D40's commit edit). D45-D49 land in the final Session 7 commit
  after notebook 05 executes.
- `SESSION7_FULL_SCALE_HONEST.md`: unchanged from the pre-launch
  state (committed in D40).
- `notebooks/05_session7_full_evaluation.ipynb` (new): 16 cells; loads
  3 runs, encodes 3 splits, computes the metric table, prints the
  decision string.
- `paper/sections/section_3_methods.md` (new).
- `paper/sections/section_4_failure_modes.md` (new).
- `paper/sections/section_5_full_scale_results.md` (new, skeleton).
- `scripts/launch_session7.sh` (new, then fixed for the disown bug).
- `scripts/launch_session7_r3_after_r2.sh` (new, recovery launcher).
- `src/training/train_jepa.py`, `src/training/train_baseline.py`:
  `--all-train` flag added; `resolve_cases()` enforces the three-way
  mutex; `run_config["all_train"]` logged.
- `tests/test_resolve_cases.py` (new, 6 tests).

Commits on `session7-full-scale` (in order):

1. `1106bfe` Session 7 housekeeping + launcher + D44 (pre-launch)
2. `887dbd8` Session 7 in-flight: launcher fix, recovery R3 script,
   notebook 05 skeleton, paper draft
3. [final commit] Session 7 results + D45-D49 + executed notebook 05

## Suggested next session

**Session 8-OBS-SIGReg** (see D49). The substantive R3_WINS outcome
points to SIGReg + OBS as the right base configuration for the
observable-augmented path at full scale. Session 8 sweeps eta in
{0.001, 0.005, 0.01, 0.05, 0.1} on this configuration to find the
operating point that maximises `delta_test_b`. R0 (pure SIGReg + BN
at full scale, no OBS) runs in parallel to confirm whether OBS is
load-bearing for SIGReg at scale.

## Postscript: the paper's framing pivots

Sections 3 and 4 of the paper draft (committed in 887dbd8) framed
the observable-augmentation contribution around the smoke-scale
"observable rescues SIGReg, marginally helps PLDM" reading. The
Session 7 full-scale evaluation INVERTS that claim. Section 5
(skeleton in 887dbd8) must be rewritten around the substantive
R3_WINS outcome:

- Contribution 1 (static-vs-dynamic + (c, t) baseline diagnostic
  suite): confirmed at scale; the (c, t) baseline is what reveals
  R1 / R2 overfitting on Test B.
- Contribution 2 (observable-augmentation): confirmed at scale, but
  the regulariser interaction is INVERTED. The right pairing is
  observable + SIGReg, not observable + PLDM.
- Contribution 3 (regulariser asymmetry): rewritten. The headline
  is no longer "PLDM is the recommended base"; it is
  "observable-augmented SIGReg generalises to unseen (G, D, Y)
  values better than observable-augmented PLDM at full scale,
  despite a 3x lower participation ratio."

The paper draft sections 3/4 need light edits (mostly the framing
sentences in section_4_failure_modes.md "regulariser-asymmetry
lineage" paragraph); section 5 needs the full Table 1 + R3_WINS
discussion. These edits land in Session 8 as part of the paper
finalisation track.
