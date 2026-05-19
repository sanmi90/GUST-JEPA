# SESSION8_R3_VALIDATION_GRID_SWEEP.md

Session 8 plan, revised with Option B (2D eta × lambda grid) plus the
latent-dimension sweep.

Last updated: 2026-05-19.

## Framing

Session 7 produced R3_WINS (D46-D48): R3 SIGReg+OBS+BN at full scale
generalises to Test B (+0.14 delta) where R1 PLDM+OBS+BN does not
(-0.01) and R2 PLDM only fails dramatically (-0.85). The Session 8
trajectory plot of L_anti (the SIGReg anti-collapse loss) showed
something the smoke evaluation missed: R3’s L_anti *rises* from
iter-0 to roughly iter-5000, then plateaus at ~7×10^-2 for the rest
of training. The lambda=0.1 weight means the L_anti cost at
convergence is small (~0.007), so the encoder is choosing to sacrifice
SIGReg compliance in exchange for lower L_pred and L_obs.

This is the “controlled collapse” regime: SIGReg pressure (wants
higher PR) balanced against the observable head’s pressure (wants the
latent to predict CL well in a low-dimensional space), with lambda
small enough that SIGReg is essentially overridden. R3’s PR of 3.35
is the trained equilibrium, not noise.

The controlled-collapse reading raises two questions the Session 7
plan did not anticipate:

1. **Lambda is doing real work, not acting as a placeholder.** A lambda
   sweep is now at least as important as the eta sweep. We need a 2D
   (eta, lambda) grid.
1. **The latent dimension d=32 may be too large.** LeWM Two-Room
   failure mechanism (D27/D39) predicts SIGReg fights itself when
   d > intrinsic dimension. Our intrinsic dim estimate is 5-10. At
   d=32 the encoder must add redundancy or noise to satisfy SIGReg in
   a space the data cannot fill. At d=8 (closer to intrinsic dim), the
   Gaussianity request is achievable and the encoder may not need to
   collapse to satisfy it. This was Ablation 1 in the original
   architecture spec, deferred until the JEPA was healthy enough to
   ablate; R3 makes it the right time.

Session 8 therefore has three production sweeps: validation
diagnostics, the (eta × lambda) 2D grid, and the d sweep at the best
(eta, lambda) point. Plus the R0 control and the paper section rewrite.
Total: 14 production runs of 20k iters each, ~8-9 hours wall-clock
with two-card parallel execution.

## Why the 2D grid and not sequential 1D sweeps

A sequential eta sweep at fixed lambda=0.1 then lambda sweep at best
eta is cheaper (~9 runs instead of 9 for the grid alone). But it
assumes the optimum (eta, lambda) is separable: the best eta does not
depend on lambda. The L_anti trajectory shows lambda and eta interact
through the loss balance. If lambda is too low, even a heavy
observable head (high eta) cannot prevent collapse. If lambda is too
high, the encoder is forced to maintain PR and the observable head’s
signal is diluted. The 2D grid catches this interaction; the
sequential 1D sweep misses it.

The grid is 9 runs at 1.5h each. Two cards in parallel reduces this
to ~7h wall-clock for Step 4 alone. Comparable to the sequential 1D
budget but with the interaction effect resolved.

## What this session does NOT do

- Lambda bisection per the original Session 6 plan (the LeWM bisection
  of arXiv:2603.19312 Appendix G, 6-8 evaluations at 24k iters each).
  Deferred to Session 9 contingent on Session 8 finding a healthy
  (eta, lambda, d) operating point.
- Full Section 7 evaluation suite (15 ablations from architecture
  spec). Session 9 or 10.
- Observable-target ablations (drag, LE pressure peak). Session 10+;
  requires data pipeline extension.
- Visualisation decoder training. After lambda bisection.
- Multi-seed averages on every grid point (too expensive at 9-12 runs).
  Single seed=0 for the grid; Step 3 covers seed variance for the
  Session 7 R3 baseline.

## Session goal and step structure

Six steps. Steps 1-3 are validation (~3 hours), Step 4 is the 2D grid
(~7 hours wall-clock with parallelism), Step 5 is the d sweep at the
best grid point (~3 hours), Step 6 is the R0 control (~1.5 hours
sequential after Step 4), Step 7 is paper section rewrite (~3-4 hours
overlapping compute).

|Step  |Output                                                                                                                                                                                 |Wall-clock    |GPU?           |
|------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------|---------------|
|Step 1|Trajectory analysis for R1, R2, R3 across all saved Session 7 checkpoints. Test B delta and PR_within over iters. Determines convergence and identifies any peak-then-decline patterns.|~1h           |No             |
|Step 2|Auxiliary-head ablation on R3. Three CL prediction methods compared on Test B (fresh probe, trained head, fresh probe for another observable if available).                            |~45 min       |No             |
|Step 3|R3 seed=42 run at full scale. Seed-variance bound on the +0.14 Test B delta.                                                                                                           |~1.5h         |cuda:0         |
|Step 4|2D (eta × lambda) SIGReg grid: 9 runs at 3 etas × 3 lambdas on R3 config. Plus run E10: PLDM+OBS+BN with paper-tuned weights for head-to-head comparison.                              |~7h           |cuda:0 + cuda:1|
|Step 5|Latent-dimension sweep at the best (eta, lambda) point. 3 runs at d in {8, 16, 32} (d=32 is the Session 7 R3 baseline, included for direct comparison).                                |~3h wall-clock|cuda:0 + cuda:1|
|Step 6|R0 control (pure SIGReg+BN+no OBS at full scale).                                                                                                                                      |~1.5h         |cuda:0         |
|Step 7|Paper Section 5 rewrite during compute windows. Sections 3 and 4 light edits per D49.                                                                                                  |~3-4h overlap |No             |

Total session wall-clock: roughly 12-14 hours from launch including
the validation diagnostics, all sweeps, R0, and the paper writing.
The compute is parallelisable, so agent-active time is ~6-8 hours.
Plan for a long day with the launcher running overnight if needed.

## Pass criteria for Session 8

1. Step 1 trajectory analysis completes for all three Session 7 runs.
1. Step 2 auxiliary-head ablation produces a definitive read on
   whether R3’s latent contains CL-relevant flow state independent of
   the trained head.
1. Step 3 R3-seed=42 lands Test B delta in [+0.05, +0.25] (a tight
   bracket but still allowing for ±50% seed noise around +0.14).
1. Step 4 grid completes; the (eta, lambda) that maximises Test B
   delta is identified.
1. Step 5 d sweep completes; the d value that maximises Test B delta
   at the best (eta, lambda) is identified.
1. Step 6 R0 control produces a Test B delta number.
1. Step 7 commits a rewritten Section 5 and light edits to Sections 3
   and 4.

Per the Session 7 honest-checkpoint discipline, ALL results get
reported. If the grid has a clear peak at a different (eta, lambda)
than Session 7’s (0.01, 0.1), the new operating point becomes the
production default. If the grid is flat (Test B delta roughly constant
across the grid), that is a publishable robustness finding.

## Step 1: convergence and stability audit (~1 hour, no GPU)

Load Session 7’s checkpoints at iter 2000, 4000, 6000, 8000, 10000,
12000, 14000, 16000, 18000, 20000 for R1, R2, R3. 30 checkpoints
total. Compute Test A and Test B metrics for each (Test C skipped to
save time).

`notebooks/06_session7_trajectory_audit.ipynb`. Three panels per run
(PR_within over iters, r2(CL_future) Test A over iters,
delta_test_b over iters). The L_total / L_pred / L_obs / L_anti
trajectory panels from the existing notebook 05 / current screenshot
are kept for reference but supplemented with the test-set metrics
which are not visible in the training loss.

Sub-deliverable: R2 anomaly investigation. R2’s Test B delta of -0.85
is anomalous. Two diagnostic checks:

- Fit the Test B probe on a random 75% of Test B latents and evaluate
  on the held-out 25%. If R2’s “within-Test-B” probe r2 is similar to
  its (c, t) baseline within-Test-B, R2’s negative delta on Test B was
  a probe-distribution-shift artifact between Test A and Test B. If
  R2’s within-Test-B probe r2 is still negative, the latent itself is
  anti-generalising.
- Plot R2’s delta_test_b over iters. If it was positive at iter 5000-
  10000 and degraded by iter 20000, the PLDM 5-term loss is actively
  destroying generalisation in late training. This would be a
  publishable failure mode independent of the R3 finding.

## Step 2: auxiliary-head ablation on R3 (~45 min, no GPU)

Take R3’s iter-20000 checkpoint. Encode all Test A and Test B
encounters. `notebooks/07_session8_head_ablation.ipynb`.

Compare three CL_future prediction methods on Test B:

1. Fresh linear probe fit on Test A latents (the Session 7 method,
   delta=+0.14).
1. The trained auxiliary head from R3’s checkpoint applied directly
   to Test B latents without retraining.
1. Fresh linear probe fit on Test A latents, but for a DIFFERENT
   aerodynamic observable. The cache has lift coefficient already
   (C_L); check whether it also contains drag coefficient (C_D) or
   leading-edge pressure (p_LE). If yes, predict that observable
   instead. If no, skip method 3 (the cache change is a Session 10
   task).

Interpretation matrix:

|Method 1|Method 2   |Method 3 (if available)|Reading                                                                                                               |
|--------|-----------|-----------------------|----------------------------------------------------------------------------------------------------------------------|
|+0.14   |similar    |similar positive       |R3 encodes general flow state; the head shaped it but other observables are also predictable. Strongest paper finding.|
|+0.14   |much higher|n/a                    |The head extracts non-linear structure a linear probe misses. Latent has rich CL-relevant content.                    |
|+0.14   |similar    |near zero or negative  |R3 encodes CL specifically, not general flow state. The win is observable-target-specific. Paper claim narrows.       |
|+0.14   |much lower |n/a                    |The head adds substantial value at inference. Latent alone is less informative.                                       |

## Step 3: R3 seed=42 (~1.5 hours, cuda:0)

```bash
python -m src.training.train_jepa \
    --gpu 0 \
    --partition v1 \
    --all-train \
    --max-iters 20000 \
    --seed 42 \
    --observable-head cl_future \
    --observable-head-weight 0.01 \
    --observable-head-deltas 8 16 24 \
    --projection-norm batchnorm \
    --anticollapse sigreg \
    --lambda-sigreg 0.1 \
    --diagnostic-every 500 \
    --checkpoint-every 2000 \
    --log-every 50 \
    --output-dir outputs/runs/session8/run_r3_seed42 \
    --wandb-mode offline \
    --tag-suffix run_r3_seed42_validation
```

Launches concurrently with Step 1 (which is CPU-only). Pass criterion:
Test B delta in [+0.05, +0.25]. If outside this bracket, Session 8
pauses to investigate seed variance before proceeding to the grid.

## Step 4: 2D (eta × lambda) grid plus PLDM reference (~7 hours wall-clock)

Nine SIGReg runs at three etas and three lambdas. The Session 7 R3
point (eta=0.01, lambda=0.1) is included so the grid has a known
reference. Plus one PLDM reference point (E10) with paper-tuned
hyperparameters for a direct head-to-head comparison.

|                         |lambda=0.01|lambda=0.1 (S7 default)|lambda=1.0|
|-------------------------|-----------|-----------------------|----------|
|**eta=0.001**            |run E1     |run E2                 |run E3    |
|**eta=0.01** (S7 default)|run E4     |run E5 (= Session 7 R3)|run E6    |
|**eta=0.1**              |run E7     |run E8                 |run E9    |

Run E5 is the Session 7 R3 itself (already trained); the grid uses its
existing results rather than re-training.

Plus run **E10: PLDM+OBS+BN with paper-tuned weights** from
arXiv:2502.14819 Appendix J.2 Tables 13-17 (Two-Rooms config). The
Session 7 R1 used the D30 placeholder defaults (all weights 1.0); E10
uses the paper’s tuned values (alpha=4.0, beta=6.9, delta=0.75,
omega=0.0). This is a single PLDM reference point, not a full PLDM
sweep, included to rule out the “PLDM was just badly tuned” objection.
A full PLDM hyperparameter sweep is deferred to Session 9 if the
results warrant it.

Eight new SIGReg runs at 1.5h each plus one PLDM run = 13.5h of
compute. Two-card parallelism divides this to ~7h wall-clock. Plus
the existing R3 anchor at the center, the full grid is complete in
~6-7h wall-clock.

### Launch pattern

`--gpu 0` sequence: E1, E2, E3, E4 (four runs × 1.5h = 6h).
`--gpu 1` sequence: E6, E7, E8, E9 (four runs × 1.5h = 6h), then E10
(1.5h). Total on `--gpu 1`: ~7.5h.
(E5 already exists from Session 7.)

```bash
# Example for run E1 (eta=0.001, lambda=0.01)
python -m src.training.train_jepa \
    --gpu 0 \
    --partition v1 --all-train --max-iters 20000 --seed 0 \
    --observable-head cl_future --observable-head-weight 0.001 \
    --observable-head-deltas 8 16 24 \
    --projection-norm batchnorm --anticollapse sigreg \
    --lambda-sigreg 0.01 \
    --diagnostic-every 500 --checkpoint-every 2000 --log-every 50 \
    --output-dir outputs/runs/session8/run_e1_eta0p001_lam0p01 \
    --wandb-mode offline --tag-suffix run_e1_eta0p001_lam0p01
```

Vary `--observable-head-weight` and `--lambda-sigreg` per run. Other
flags identical to Session 7’s R3.

```bash
# Run E10: PLDM+OBS+BN with paper-tuned weights (arXiv:2502.14819 Appendix J.2 Two-Rooms).
# Mapping of PLDM paper notation to existing train_baseline.py CLI flags:
#   alpha (L_var weight)        = --lambda-var
#   beta  (L_cov weight)        = --lambda-cov
#   delta (L_time_sim weight)   = --lambda-time-sim
#   omega (L_idm weight)        = --lambda-idm
#   gamma (L_sim weight)        = --pldm-gamma
python -m src.training.train_baseline \
    --baseline pldm \
    --gpu 1 \
    --partition v1 --all-train --max-iters 20000 --seed 0 \
    --lambda-var 4.0 --lambda-cov 6.9 --lambda-time-sim 0.75 --lambda-idm 0.0 \
    --observable-head cl_future --observable-head-weight 0.01 \
    --observable-head-deltas 8 16 24 \
    --projection-norm batchnorm \
    --diagnostic-every 500 --checkpoint-every 2000 --log-every 50 \
    --output-dir outputs/runs/session8/run_e10_pldm_paper_tuned \
    --wandb-mode offline --tag-suffix run_e10_pldm_paper_tuned
```

The PLDM weight flags already exist in `train_baseline.py` under the
`--lambda-*` naming convention (verified pre-launch). No code change
needed for E10; only the plan's flag names had to align with the
codebase.

### Analysis

After all 9 SIGReg grid points exist (8 new + Session 7 R3 anchor)
and E10 PLDM completes, build `notebooks/08_eta_lambda_grid.ipynb`.
Compute Test A, Test B, Test C delta for each grid point. Four
figures:

- Heatmap of delta_test_b across the (eta, lambda) SIGReg grid. The
  headline result.
- Heatmap of PR_all across the grid (to see whether high-delta points
  cluster in a particular PR range).
- Heatmap of r2(z->c) across the grid (to see whether c-encoding
  varies with the regularisation balance).
- A “champion table” comparing the best SIGReg grid point (eta*,
  lambda*), R3 (Session 7 default eta=0.01 lambda=0.1), R1 (Session 7
  PLDM with placeholder weights), and E10 (PLDM with paper-tuned
  weights). Test A / Test B / Test C delta for each, plus PR_within
  and r2(z->c). This is the head-to-head SIGReg-vs-PLDM-at-best-effort
  comparison that the paper needs.

Identify (eta*, lambda*) maximising delta_test_b for SIGReg. This
becomes the production operating point for Step 5 and Session 9.

### What the grid could reveal

Expected outcome (most likely): the grid has a smooth interior peak
that may or may not be exactly at the Session 7 default. The L_anti
trajectory plot suggests lambda=0.1 with eta=0.01 was a reasonable but
not necessarily optimal choice; an adjacent grid point might be better.

Surprise outcome A: the peak is at the eta=0.1 or lambda=0.01 corner.
This would mean the observable head doing most of the regularisation
work and SIGReg essentially turned off (lambda=0.01) or that very
heavy observable supervision (eta=0.1) recovers what SIGReg cannot.
Either way, R3’s success is more about the observable head than about
SIGReg.

Surprise outcome B: the peak is at lambda=1.0. SIGReg actually
benefits from stronger weight at scale, contrary to the L_anti rising
pattern observed at lambda=0.1. The encoder needs the stronger
Gaussianity pressure to avoid the rising-L_anti state. This would
contradict the controlled-collapse reading.

Surprise outcome C: the grid is flat. delta_test_b roughly constant
across the 9 grid points. R3’s win is robust to (eta, lambda) tuning,
which is a positive paper finding.

### Three possible E10 outcomes

The PLDM paper-tuned reference run resolves the “PLDM was just badly
tuned” objection. Three readings:

- **E10 delta_test_b < best SIGReg grid point - 0.05**: even with
  paper-tuned weights, PLDM underperforms SIGReg+OBS at scale. The
  D46 R3_WINS finding is robust to PLDM hyperparameter choice. No
  full PLDM sweep needed; paper claim 3 stands strongly.
- **E10 delta_test_b within 0.05 of best SIGReg**: PLDM with proper
  tuning is competitive. Session 9 must include a proper PLDM
  hyperparameter sweep before the paper can claim either method is
  better. The paper claim weakens to “SIGReg+OBS and properly tuned
  PLDM both work at full scale; further investigation needed.”
- **E10 delta_test_b > best SIGReg + 0.05**: paper-tuned PLDM is
  actually the winner. Session 8 found the wrong operating point;
  Session 9 becomes a proper PLDM sweep. Paper claim 3 inverts back
  toward the Session 6 D39 reading at scale, with the caveat that
  PLDM needs paper-tuned (not default) weights.

## Step 5: latent-dimension sweep at best (eta*, lambda*) (~3 hours)

After Step 4 identifies (eta*, lambda*), sweep d in {8, 16, 32}. Three
runs total; d=32 is the best-grid-point result (already trained as
part of Step 4), so only two new runs at d=8 and d=16.

```bash
# Example for d=8
python -m src.training.train_jepa \
    --gpu 0 \
    --partition v1 --all-train --max-iters 20000 --seed 0 \
    --latent-dim 8 \
    --observable-head cl_future --observable-head-weight {eta*} \
    --observable-head-deltas 8 16 24 \
    --projection-norm batchnorm --anticollapse sigreg \
    --lambda-sigreg {lambda*} \
    --diagnostic-every 500 --checkpoint-every 2000 --log-every 50 \
    --output-dir outputs/runs/session8/run_d8_best \
    --wandb-mode offline --tag-suffix run_d8_best_eta_lambda
```

`--latent-dim` is now wired as an alias for `--d` on both
`train_jepa.py` and `train_baseline.py` (committed in the Session 8
pre-launch housekeeping, alongside this plan). Behavior with the
existing `--d` flag is unchanged.

Two runs in parallel on `--gpu 0` (d=8) and `--gpu 1` (d=16). Total
1.5h wall-clock for the sweep.

### Analysis

`notebooks/09_latent_dim_sweep.ipynb`. For each d value, compute
delta_test_b, delta_test_c, PR_all, PR_within, r2(z->c). The headline
figure is delta_test_b vs d.

Three plausible outcomes:

- **d small wins** (d=8 best): the LeWM intrinsic-dimension mechanism
  is real; smaller latent gives SIGReg a space it can fill. Paper
  recommendation includes “use d close to estimated intrinsic
  dimension.” Strongest finding.
- **d invariant** (all three within ±0.02): the model is robust to d.
  Paper notes the result and uses d=8 or d=16 going forward for compute
  efficiency.
- **d large wins** (d=32 best): the controlled-collapse mechanism
  works best when the latent has room to allocate. Counterintuitive
  but possible; suggests the auxiliary head needs latent space to
  shape independent of the rest of z.

If the d sweep shows d=8 wins meaningfully, Session 9’s lambda
bisection runs at d=8 not d=32. The paper’s headline architecture
diagram changes accordingly.

## Step 6: R0 control (~1.5 hours, cuda:0 after Steps 4 or 5)

Pure SIGReg + BN, no observable head, at full scale. The “is OBS
load-bearing for SIGReg at full scale” question.

```bash
python -m src.training.train_jepa \
    --gpu 0 \
    --partition v1 --all-train --max-iters 20000 --seed 0 \
    --observable-head none \
    --projection-norm batchnorm --anticollapse sigreg \
    --lambda-sigreg 0.1 \
    --diagnostic-every 500 --checkpoint-every 2000 --log-every 50 \
    --output-dir outputs/runs/session8/run_r0_sigreg_only \
    --wandb-mode offline --tag-suffix run_r0_sigreg_only_full
```

Runs after Step 4 OR after Step 5 (cuda:0 becomes free once the d
sweep completes). Lambda=0.1 here is the Session 7 default; if Step 4
found a much higher lambda*, also run R0 at lambda*. Two R0 runs
(0.5h each on top of the 1.5h primary) is cheap insurance.

Three substantive outcomes from R0:

- **R0 delta_test_b < 0**: pure SIGReg at full scale fails; OBS is
  essential. Paper claim 3 robust: “observable augmentation is
  necessary for SIGReg to generalise on this data.”
- **R0 delta_test_b in [0, 0.05]**: pure SIGReg barely beats baseline.
  OBS adds substantial value but is not strictly required. Paper
  claim shifts to “observable augmentation substantially improves
  SIGReg generalisation.”
- **R0 delta_test_b > 0.1**: pure SIGReg at full scale generalises
  comparably to SIGReg+OBS. OBS is decorative at scale. Paper claim
  shifts to “SIGReg JEPA generalises at full scale; observable
  augmentation provides modest improvement.” Unlikely given the
  Session 6 evidence across 8 axes, but worth checking.

## Step 7: paper Section 5 rewrite (~3-4 hours, during compute)

During the ~7 hours of grid compute on Step 4 and the ~3 hours of d
sweep compute on Step 5, write Section 5 of the paper around the
validated findings. Per D49 this is the central paper work for
Session 8.

Section structure:

- 5.1 Full-scale evaluation setup (v1.2 partition, Test A/B/C splits,
  the (c, t) baseline as the central diagnostic). Carry forward from
  Session 7’s skeleton.
- 5.2 The three Session 7 production runs (R1, R2, R3) with the
  Table 1 metric table. State the TEST_B_TEST_A_DISCREPANCY decision
  tree outcome and the substantive R3_WINS reading.
- 5.3 The regulariser-asymmetry inversion: smoke-scale D39 said “PLDM
  is the recommended base”; full-scale Session 7 inverts this. Mechanism
  discussion grounded in the L_anti trajectory: SIGReg+OBS operates in
  a controlled-collapse regime; PLDM+OBS produces a high-PR latent
  that absorbs case-axis variation. Cite Sessions 5/5.PLDM/6 and the
  LeWM Two-Room precedent (arXiv:2603.19312 Section 5) explicitly.
- 5.4 Session 8 validation work: trajectory analysis (Step 1),
  auxiliary head ablation (Step 2), seed-variance bound (Step 3).
- 5.5 (eta × lambda) grid results: the heatmaps from Step 4. Discuss
  whether the Session 7 default was optimal or whether the grid found
  a better operating point. State the production (eta*, lambda*).
- 5.6 Latent-dimension sweep results: the delta_test_b vs d figure
  from Step 5. Discuss the LeWM intrinsic-dimension prediction and
  whether our data confirms it. State the production d.
- 5.7 R0 control: what it implies for the necessity-of-OBS claim.
- 5.8 Recommendation summary: the production configuration going into
  Session 9 (lambda bisection) and the paper’s contribution claims as
  validated by Session 8.

Approximate length: 10-12 pages. Sections 3 and 4 also get light
edits per D49: section_4_failure_modes.md “regulariser-asymmetry
lineage” paragraph and any abstract draft.

## Risk register

|Risk                                                     |Probability|Mitigation                                                                                                                                             |If it fires                                                                  |
|---------------------------------------------------------|-----------|-------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
|Step 1 shows R3 was still climbing at iter-20000         |low        |The L_anti screenshot suggests convergence; full Test B trajectory will confirm. If still climbing, the d sweep uses 40k iters; doubles step 5 compute.|+3h on Step 5                                                                |
|Step 2 reveals R3’s win is auxiliary-head-mediated       |medium     |Paper claim adjusts to “the head shapes z toward CL prediction”; the SIGReg+OBS recipe still works but the interpretation narrows                      |Section 5.3 wording changes                                                  |
|Step 3 seed=42 lands outside [+0.05, +0.25]              |medium     |Run seed=123 as a third datapoint; report seed variance honestly                                                                                       |+1.5h GPU                                                                    |
|Grid is flat (no interaction)                            |low-medium |Document as a robustness finding; report median (eta*, lambda*)                                                                                        |Sections 5.5 framing shifts                                                  |
|Grid peak at edge (eta=0.001 corner or lambda=1.0 corner)|medium     |Run 2-3 extension points outside the grid in Session 9                                                                                                 |+2-3h GPU in Session 9                                                       |
|d=8 wins by a lot                                        |low-medium |Paper’s architecture figure changes; D2 is updated; Session 9 lambda bisection at d=8 not d=32                                                         |Paper changes are easy; compute savings going forward                        |
|d=8 loses by a lot                                       |low        |Counterintuitive; investigate before paper claims; consider d=16 as a compromise                                                                       |Section 5.6 framing emphasises why d=32 is needed despite intrinsic dim of ~8|
|R0 generalises comparably to R3                          |low        |OBS is decorative; paper simplifies                                                                                                                    |Section 5.7 reframes; eta sweep becomes a robustness finding                 |
|E10 (paper-tuned PLDM) beats best SIGReg grid point      |low-medium |Session 9 becomes a proper PLDM hyperparameter sweep; paper claim 3 framing inverts back toward D39 with the caveat that PLDM needs proper tuning      |+4-6h GPU in Session 9 plus paper reframing                                  |

## D-entries to record

**D50**: Step 1 trajectory audit results. Convergence status and any
peak-then-decline patterns for R1, R2, R3.

**D51**: Step 2 auxiliary-head ablation result. Whether R3’s latent
contains CL-relevant flow state independent of the trained head, and
whether the win generalises to other observables (if testable).

**D52**: Step 3 seed=42 result. R3 seed-variance bound.

**D53**: Step 4 (eta × lambda) grid result. The optimal SIGReg
operating point (eta*, lambda*) and the grid shape (peaked, flat,
edge-peaked).

**D53b**: Step 4 E10 PLDM paper-tuned reference run. Whether
paper-tuned PLDM is competitive with the best SIGReg grid point.
Determines whether Session 9 needs a full PLDM hyperparameter sweep.

**D54**: Step 5 d sweep result. The optimal d at the best (eta*,
lambda*) and the delta_test_b vs d shape.

**D55**: Step 6 R0 result. Whether OBS is load-bearing for SIGReg at
full scale.

**D56**: Paper Section 5 rewrite committed. Sections 3 and 4 light
edits per D49 also committed.

**D57** (always): Session 8 outcome summary. Either VALIDATED (Steps
1-3 pass, grid has a clear peak, d sweep is interpretable, R0
informative; Session 9 is lambda bisection at the production config),
CONCERN (one of Steps 1-3 returned an unexpected result; investigate
before scaling), or PAPER_PIVOT (R0 or Step 2 returned results that
change the paper’s core claim).

## After Session 8

Most likely path: Session 9 is the lambda bisection at the validated
(eta*, lambda*, d*) configuration. 6-8 evaluations of 20k iters each
to find the precise lambda maximising delta_test_b. Plus the
visualisation decoder training on the winning encoder. Plus the start
of the full Section 7 evaluation suite per the architecture spec.

Alternative paths:

- **VALIDATION_CONCERN**: Session 9 is targeted further diagnostics
  (longer horizons, multi-seed, observable-target ablations).
- **PAPER_PIVOT**: Session 9 rewrites paper claims and runs whichever
  configuration the new claims point to.

Session 10 in all cases: full 15-ablation matrix per architecture spec
(now including the d ablation already done in Session 8), final paper
figures, JFM manuscript draft.

## Decision references

- D2: d=32 per LeWM. **Session 8 Step 5 directly ablates this.**
- D5, D17: SIGReg with BatchNorm projection.
- D27, D31, D39: failure-mode taxonomy.
- D34, D35: frame-skip 1, partition v1.2 (41 train cases, 138 train
  encounters).
- D40 (HANDOFF): two RTX 6000 cards; `--gpu {0,1}` flag.
- D44-D49: Session 7.
- D50-D57: this session.
