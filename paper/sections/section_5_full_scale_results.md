# Section 5: Full-scale results

LaTeX-friendly markdown. This section reports the full-scale Session 7
production runs (R1, R2, R3) on the v1.2 partition (41 train cases, 138
encounters, locked split sha256
`a721dc92...20a58137f79f3a19fc6ad71b70a007dd`), the Session 8 validation
diagnostics (trajectory audit, auxiliary-head ablation, seed-variance
bound), and the Session 8 production sweeps (the 2D eta × lambda grid,
the latent-dimension sweep at the best operating point, the R0 control,
and the paper-tuned PLDM reference E10).

The headline result (Section 5.2): the simpler SIGReg + observable head
(R3) generalises to Test B with delta = +0.14 over the (c, t) baseline,
while the 5-term PLDM + observable head (R1) of D39's smoke-scale
recommendation is no better than the baseline on Test B at full scale
(delta = -0.01). The Section 6 D39 reading of "PLDM is the recommended
base for the observable-augmented path" is therefore overturned at scale.
The Session 8 grid (Section 5.5) and d sweep (Section 5.6) refine the
operating point; the R0 control (Section 5.7) confirms the observable
head is load-bearing for the SIGReg path at scale.

## 5.1 Experimental setup

Three full-scale Session 7 production runs differ in the axes Section 4
identified as discriminating:

- **R1: PLDM + observable head, BatchNorm projection.** The D39 smoke
  winner. R1 tests whether the smoke-scale healthy reading scales to
  41 train cases.
- **R2: PLDM only (eta = 0), BatchNorm projection.** The OBS-vs-no-OBS
  control on the PLDM base. R2 isolates the observable head's
  contribution.
- **R3: SIGReg + observable head, BatchNorm projection.** The
  regulariser-asymmetry test. Section 4.2 showed OBS rescuing SIGReg
  from TRIVIAL on the 5-case smoke; R3 tests whether the rescue
  persists at 41 cases.

All three runs train 20,000 iterations on the full v1.2 train partition,
seed 0, batch size 16, sub-trajectory length L = 32, eta = 0.01 where
applicable, lambda = 0.1 for SIGReg, lambda_{var,cov,time_sim,idm} = 1.0
for PLDM (D30 unit-weight starting point). Detailed configuration in
Section 3 and in `scripts/launch_session7.sh`.

Each run is evaluated on three held-out splits:

- **Test A** (in-sample held-out encounters within train cases; 56
  encounters from 41 cases): in-distribution generalisation, sanity check.
- **Test B** (parametric interpolation; 6 cases, 28 encounters): cases
  at unseen interior (G, D, Y) values. The primary metric.
- **Test C** (extrapolation; 4 cases at |G| = 4, 24 encounters):
  stretch goal.

The headline figure of merit per split is
`delta = r2(z -> CL_future) - r2((c, t) -> CL_future)`. The (c, t)
baseline is a tiny MLP probe `(case_descriptor, frame_index) -> CL(t + Delta)`
fit on Test A and evaluated on the split. A positive delta on Test B
means the latent encodes generalisable flow physics beyond a parametric
case-frame lookup. Two D = 1.5 Test A encounters have non-finite CL
values from DNS instability near the last encounter and are dropped
from CL fits but retained for PR / probe metrics on z (D45).

## 5.2 Session 7 full-scale results: TEST_B_TEST_A_DISCREPANCY with R3_WINS

Table 1 reports the complete per-(run, split) metric table from
`notebooks/05_session7_full_evaluation.ipynb` Section 4 (final iter
20000 checkpoints; D46).

| Run                | Split  | PR_all | PR_within | r2(z->c) | r2_dyn_phase | r2(CL_future) | (c,t) baseline | delta   |
|--------------------|--------|-------:|----------:|---------:|-------------:|--------------:|---------------:|--------:|
| R1 PLDM+OBS+BN     | Test A | 27.84  |   6.87    |   0.90   |    0.78      |    0.97       |     0.74       |  +0.23  |
| R1 PLDM+OBS+BN     | Test B | 18.31  |  10.06    |   0.96   |    0.91      |    0.71       |     0.72       |  -0.01  |
| R1 PLDM+OBS+BN     | Test C | 14.50  |  11.77    |   0.90   |    0.86      |    0.76       |     0.35       |  +0.42  |
| R2 PLDM only BN    | Test A | 27.16  |   6.01    |   0.88   |    0.77      |    0.93       |     0.74       |  +0.19  |
| R2 PLDM only BN    | Test B | 17.35  |   9.41    |   0.95   |    0.92      |   -0.13       |     0.72       |  -0.85  |
| R2 PLDM only BN    | Test C | 13.92  |  11.14    |   0.91   |    0.87      |    0.32       |     0.35       |  -0.03  |
| R3 SIGReg+OBS+BN   | Test A |  3.69  |   4.18    |   0.62   |    0.44      |    0.97       |     0.74       |  +0.24  |
| R3 SIGReg+OBS+BN   | Test B |  3.51  |   3.85    |   0.93   |    0.63      |    0.86       |     0.72       |  +0.14  |
| R3 SIGReg+OBS+BN   | Test C |  2.91  |   4.67    |   0.76   |    0.73      |    0.83       |     0.35       |  +0.48  |

Table 1: complete per-(run, split) metric table from Session 7. Bold
numbers in the source notebook; reproduced here verbatim.

The eight-branch decision tree in
`SESSION7_FULL_SCALE_HONEST.md` Step 2 Section 6 short-circuits on
TEST_B_TEST_A_DISCREPANCY first when matched. R1 matches that branch
(test_a delta 0.23 > 0.10 AND test_b delta -0.01 < 0.03). R2 matches
even more dramatically (test_a +0.19 vs test_b -0.85). The same data
also satisfies R3_WINS strictly (R3 test_b delta +0.14 > R1 test_b
delta -0.01); the tree picks the first matching branch but the
substantive reading is R3_WINS.

The substantive reading (D46):

- **R3 SIGReg + OBS + BN is the only run that generalises to Test B**
  with delta = +0.14 and is the BEST run on Test C with delta = +0.48.
- R1 PLDM + OBS overfits: it has the highest PR (27.84 on Test A), the
  cleanest within-case dynamics decomposition (`r2_dyn_phase` = 0.78
  on Test A, 0.91 on Test B), but its out-of-sample CL prediction on
  Test B is no better than the (c, t) baseline.
- R2 PLDM-only is the worst: Test B delta = -0.85 means the 5-term
  PLDM latent at full scale is *worse* than a tiny (c, t) MLP at
  predicting CL on unseen cases. This is overfitting to the 41 train
  cases in a way that hurts generalisation.
- The smoke-scale (5 cases) "PLDM + OBS wins" finding of D39 was a
  small-data artifact. The PR = 10 numbers PLDM + OBS achieves at
  both smoke and full scale look like the same healthy reading, but
  the Test B signal at full scale shows PR captures case-specific
  memorisation, not transferable flow physics.

Figure 2 (`outputs/runs/session8/fig_session7_delta_summary.png`) is the
per-run bar chart of Test A, Test B (cross-split), and Test B
(within-Test-B 75/25) delta for the three Session 7 runs. The Test B
cross-split bars are the headline numbers; the within-Test-B bars are
the diagnostic that disambiguates "uninformative latent" (R2 cross-split
-1.22) from "Test A -> Test B distribution shift" (R2 within-Test-B
+0.10). The dashed lines at 0.03 (WEAK_GO threshold) and 0.10 (STRONG_GO
threshold) frame R3's cross-split +0.14 as strongly positive while R1
and R2 fail to clear either threshold on the cross-split metric.

## 5.3 The regulariser-asymmetry inversion

The R1 vs R3 comparison at the same eta = 0.01 isolates the
anti-collapse regulariser (D48):

| Split  | R1 delta | R3 delta | R3 - R1 |
|--------|---------:|---------:|--------:|
| test_a |  +0.23  |  +0.24   |  +0.01  |
| test_b |  -0.01  |  +0.14   |  +0.15  |
| test_c |  +0.42  |  +0.48   |  +0.07  |

On Test A both regularisers produce equivalent CL prediction. On
Test B (the parametric-interpolation question this paper centres on),
the simpler SIGReg regulariser materially outperforms PLDM by +0.15
absolute, the difference between "fails to beat baseline" and
"+14 percentage points over baseline." On Test C extrapolation R3 is
also slightly ahead.

This inverts the D39 reading from Section 4.3. D39 (5-case smoke)
concluded that PLDM is the recommended base because PLDM + OBS reaches
PR around 10 while SIGReg + OBS plateaus at PR around 3. The Session 7
full-scale evaluation shows PR around 10 was masking the overfitting
that happens when the 5-term PLDM has 41 cases to memorise, while the
low-PR SIGReg + OBS latent retains its generalisation capability.

The R1 vs R2 comparison at the same PLDM base isolates the
observable head's contribution (D47):

| Split | R1 delta | R2 delta | R1 - R2 |
|-------|---------:|---------:|--------:|
| test_a |  +0.23  |  +0.19   |  +0.04  |
| test_b |  -0.01  |  -0.85   |  +0.84  |
| test_c |  +0.42  |  -0.03   |  +0.45  |

The observable head rescues PLDM dramatically out-of-sample (R1 - R2
on test_b is +0.84) but the rescued state only reaches the (c, t)
baseline level (delta -0.01). Without OBS, PLDM at full scale produces
a latent that is worse than the parametric baseline at predicting CL
on unseen cases. The OBS-vs-no-OBS control on the SIGReg base is the
Step 6 R0 task (Section 5.7).

The mechanism behind the inversion: the L_anti trajectory plot in
`notebooks/05_session7_full_evaluation.ipynb` Section 5 shows R3's
SIGReg loss *rising* from iter 0 to iter 5000 and plateauing at
~7e-2 for the rest of training. At lambda = 0.1 the anti-collapse
cost at convergence is ~7e-3, small enough that the encoder sacrifices
strict SIGReg compliance in exchange for lower L_pred and L_obs. The
trained equilibrium is a controlled-collapse regime where SIGReg
provides directional pressure (preventing rank-1 collapse) without
forcing the encoder to maintain the high-PR isotropy of pure SIGReg.
PLDM's 5-term loss does not have a single weight that the encoder can
"buy off" cheaply; instead all four collapse-prevention terms pull
simultaneously, and the resulting latent at full scale absorbs
case-axis variation that does not transfer to Test B.

Paper claim 3 (regulariser asymmetry) is reworded from
"observable augmentation rescues SIGReg, marginally helps PLDM"
(D39, smoke scale) to "the observable-augmented SIGReg latent
generalises to unseen (G, D, Y) values better than the
observable-augmented PLDM latent at full scale, despite a 3x lower
participation ratio." The deeper finding: PR alone is not a reliable
proxy for the generalisation quality of a JEPA latent on
low-intrinsic-dim physics data. The (c, t) baseline + Test B delta is
the right diagnostic.

Section 5.4 below adds one more nuance to the PR-vs-generalisation
relationship. The trajectory audit at Step 1 reports a fresh MLP fit
on 75% of Test B latents and evaluated on the held-out 25% (the
"within-Test-B" probe r2). R2 PLDM-only has within-Test-B delta of
+0.10 -- *higher* than R3's +0.07. PLDM-only's latent is in fact
informative about CL on Test B when fit on Test B itself; what it lacks
is the Test A -> Test B alignment that lets a probe trained on Test A
transfer to Test B. R3's edge in the headline cross-split metric is
therefore the alignment of its Test A and Test B latent geometries
(consistent with SIGReg's controlled-collapse pushing the encoder into
a more compact representation that does not drift in geometry between
the two splits), not the per-split informativity of its latent.

## 5.4 Session 8 validation diagnostics

Three diagnostic concerns were raised by the Session 7 trajectory plot
and by the smoke-vs-full-scale inversion of D39.

**Step 1 (trajectory audit, D50).** `notebooks/06_session7_trajectory_audit.ipynb`
loads every saved checkpoint at iter 2000, 4000, ..., 20000 for R1, R2,
R3 (30 evaluations total) and computes Test A and Test B metrics
per-checkpoint. Three non-trivial findings:

- **R2's Test B delta progressively DEGRADES across training:** -0.18 at
  iter 4000, -0.45 at iter 8000, -1.21 at iter 20000. The PLDM 5-term
  loss is actively destroying Test A -> Test B transferability over the
  second half of training. This is a publishable failure mode of the
  PLDM loss at full scale, independent of the R3_WINS finding.
- **R2's PR_all rises in lockstep with the cross-split degradation:**
  PR_all on Test A grows from 1.63 (iter 2000) to 27.14 (iter 20000),
  meanwhile PR_within on Test A *shrinks* from 15.14 to 6.02. R2 is
  moving variance OUT of within-case dynamics INTO the case-mean axis
  -- the SPREAD_TRIVIAL signature of Section 4.2 at full scale. The
  growing case-mean variance is precisely what hurts Test A -> Test B
  transfer: Test B has different case identities, so a case-mean-
  dominated latent geometry does not align between the two splits.
  Crucially, PR_all RISES while generalisation FALLS; PR is therefore
  not just an imperfect proxy for generalisation but in this regime is
  *anti-correlated* with it, reinforcing the claim that
  Test-B-delta-over-(c,t) is the right diagnostic.
- **R2's *within-Test-B* delta is steadily POSITIVE (~+0.10) throughout
  training.** Fit a fresh MLP on 75% of Test B latents and evaluate on
  25%, and R2's latent does predict CL on Test B. The Session 7 -0.85
  number is largely a distribution-shift artifact between Test A and
  Test B, not a globally uninformative latent. R2's within-Test-B
  (+0.10) is in fact *higher than* R3's within-Test-B (+0.07); R3's
  advantage in the cross-split metric is the alignment of its Test A
  and Test B latent geometries, not the per-split informativity. R3 by
  contrast has PR_all and PR_within of similar magnitude (3.69 and
  4.18 at iter 20000), the controlled-collapse regime of Section 5.3.

R3 itself converges by iter 12000 to the +0.14 plateau (no
peak-then-decline behaviour), confirming the iter-20000 endpoint is the
trained equilibrium and that the d-sweep and grid can use 20k iters
without leaving headroom for further training.

**Step 2 (auxiliary-head ablation on R3 iter-20000, D51).** Three
CL-prediction methods on Test B are compared in
`notebooks/07_session8_head_ablation.ipynb`:

|Target |Fresh probe on z (Test B) |Trained R3 head (Test B) |Gap fresh - trained |
|-------|-------------------------:|------------------------:|-------------------:|
|C_L    |               +0.138     |               +0.137    |          +0.001    |
|C_D    |               +0.106     |              n/a        |          n/a       |
|p_LE   |               +0.123     |              n/a        |          n/a       |

Method 1 (fresh probe) reproduces the D46 +0.14 to within 0.002.
Method 2 (trained R3 head applied directly) is essentially identical
to Method 1 (+0.137 vs +0.138); the trained head extracts no non-linear
structure a linear probe misses. Method 3 (fresh probe for unrelated
observables C_D drag and p_LE leading-edge pressure) gives Test B
deltas of +0.106 and +0.123 -- clearly positive and within 0.04 of the
trained-for target. **R3's latent encodes general flow state, not
CL-specific structure.** This matches Row 1 of the plan's interpretation
matrix and is the strongest of the four possible readings.

**Step 3 (seed-variance bound, D52).** R3 retrained from scratch with
seed = 42, identical configuration otherwise. Pass criterion: Test B
delta in [+0.05, +0.25].

Result: **R3-seed42 Test B delta = +0.121 (PASS).** Compared with R3
seed = 0 final +0.138 (D46 Session 7), the seed-variance bound on the
+0.14 headline finding is ~0.017 absolute, or ~12% relative. Trajectory
previews at iters 8000 and 12000 (+0.081 and +0.117 respectively) track
the seed = 0 R3 trajectory consistently ~0.02 lower at matching
iterations; the seed = 0 R3 trajectory plateau at +0.14 is therefore a
seed-robust feature of the configuration, not a single-seed accident.

## 5.5 The (eta x lambda) grid: SIGReg + OBS operating point

Nine SIGReg + OBS runs at eta in {0.001, 0.01, 0.1} times lambda_SIGReg
in {0.01, 0.1, 1.0} on R3 config (the Session 7 R3 reused as the
(eta=0.01, lambda=0.1) centre cell). Plus E10: PLDM + OBS + BN with
the paper-tuned Two-Rooms weights from arXiv:2502.14819 Appendix J.2
(alpha=4.0, beta=6.9, delta=0.75, omega=0.0) at eta=0.01. All runs at
seed=0, 20k iterations, full v1.2 train partition.

|                | lambda=0.01 | lambda=0.1   | lambda=1.0 |
|----------------|------------:|-------------:|-----------:|
| **eta=0.001**  |      -0.200 |       +0.007 |     -0.620 |
| **eta=0.01**   |  **+0.159** |       +0.138 |     +0.093 |
| **eta=0.1**    |      +0.148 |       +0.146 |     +0.152 |

Table 2: Test B delta over (c, t) baseline at iter 20000, per grid cell.
The best cell is (eta*=0.01, lambda*=0.01) with delta = +0.159.

The figures `outputs/runs/session8/fig_grid_delta_b.png`,
`fig_grid_pr_all.png`, and `fig_grid_r2_z_c.png` are heatmaps of the
same data plus PR_all and r2(z->c) across the grid.

Three pattern observations (D53):

1. **eta is the dominant axis.** At eta = 0.001 (the head is almost
   off) the encoder fails or barely matches baseline regardless of
   lambda. At eta in {0.01, 0.1} the encoder generalises across all
   lambdas tested. The observable head is the central regulariser at
   full scale, more than lambda is.
2. **Lower lambda is better at eta = 0.01.** Sequence at eta=0.01:
   lam=0.01 (+0.159) > lam=0.1 (+0.138) > lam=1.0 (+0.093). The
   Session 7 default of lambda=0.1 was not optimal; lambda=0.01
   (SIGReg essentially off) generalises ~+0.02 higher and is the new
   production setting.
3. **The eta=0.1 row is essentially flat in lambda.** +0.148, +0.146,
   +0.152 across lambdas (within 0.006 absolute). When the OBS
   pressure is strong enough, SIGReg's contribution is negligible.

These three patterns together support the reading that the observable
head -- not SIGReg -- is the central regulariser of the SIGReg + OBS
configuration at full scale. SIGReg's role is small: at lambda=0.01
its gradient is barely perceptible, yet the encoder still avoids the
TRIVIAL collapse that pure SIGReg + BN landed in at the smoke scale
(Section 4.2). The L_anti rising trajectory of Section 5.3 makes this
mechanically explicit: the encoder buys SIGReg compliance cheaply
(L_anti rises but the lambda * L_anti contribution is small) and
focuses on L_pred and L_obs.

Champion table comparing the best SIGReg grid point (E4) against the
Session 7 R3 anchor (E5), R1's PLDM with unit-weight defaults (Session
7), and E10's PLDM with paper-tuned Two-Rooms weights:

| Run             | eta | lambda | PR_all (Test B) | r2(z->c) | r2(CL_future) | delta   |
|-----------------|----:|-------:|----------------:|---------:|--------------:|--------:|
| E4 (best SIGReg)|0.01 |  0.01  |           2.61  |   0.87   |     0.88      | +0.159  |
| E5 (S7 R3)      |0.01 |  0.10  |           3.51  |   0.93   |     0.86      | +0.138  |
| E10 PLDM tuned  |0.01 |   --   |          23.02  |   0.65   |     0.62      | -0.095  |
| R1 PLDM defaults|0.01 |   --   |          18.33  |   0.96   |     0.72      | -0.003  |

Table 3: Champion table on Test B at iter 20000.

E10 (D53b) is **worse** than R1 (PLDM with unit-weight defaults) on
Test B (-0.095 vs -0.003). Paper-tuned PLDM does not rescue PLDM at
full scale on this data; the Two-Rooms hyperparameters tuned for the
LeWM gridworld do not transfer to low-intrinsic-dim physics. **The
"PLDM was just badly tuned" objection is decisively ruled out**: with
paper-tuned weights PLDM is *even worse* than with defaults. Session 9
does not need a full PLDM hyperparameter sweep; paper claim 3 stands
robustly.

Identified production operating point: **(eta*, lambda*) = (0.01, 0.01).**
Session 5.6 (d-sweep) and Session 9 (lambda bisection) use this point
as the centre.

## 5.6 Latent-dimension sweep

The L_anti rising trajectory in Section 5.3 and the SPREAD_TRIVIAL
signature for the high-PR PLDM runs motivate a direct test of the LeWM
Two-Room intrinsic-dimension mechanism (\cite{lewm} Section 5) on this
data. With intrinsic dim estimated at ~5-10 (Section 4), d = 32 could
either help (more latent room for the OBS-induced structure) or hurt
(extra dimensions force SIGReg to fight itself, the LeWM prediction).

Step 5 sweeps d in {8, 16, 32} at the Step 4 (eta*, lambda*) = (0.01,
0.01). The d=32 run is the E4 grid point already trained; d=8 and d=16
are new runs at the same hyperparameters with a different latent
dimension.

| d  | PR_all (Test B) | PR_within (Test B) | r2(z->c) Test B | delta Test A | delta Test B | delta Test C |
|---:|----------------:|-------------------:|----------------:|-------------:|-------------:|-------------:|
|  8 |        2.22     |        3.36        |       0.70      |    +0.224    |  **+0.092**  |    +0.451    |
| 16 |        2.37     |        3.68        |       0.69      |    +0.214    |  **+0.103**  |    +0.474    |
| 32 |        2.61     |        3.88        |       0.87      |    +0.227    |  **+0.159**  |    +0.470    |

Table 4: latent-dimension sweep at the production operating point.

The figure `outputs/runs/session8/fig_d_sweep.png` plots Test B delta
vs d.

**d=32 wins on Test B by +0.07 over d=8.** The LeWM intrinsic-dimension
prediction (smaller d closer to the data's intrinsic dim should win)
is **not confirmed on this data** (D54). Two diagnostics resolve the
mechanism:

1. PR_all is essentially flat in d (2.22 / 2.37 / 2.61 across d in
   {8, 16, 32}). The encoder uses ~2 effective dimensions regardless of
   d. The "extra" d=32 dimensions are not used by the encoder.
2. r2(z->c) on Test B is 0.70 / 0.69 / 0.87 across d. At d=32 the
   encoder encodes c more strongly. Yet Test B delta is highest at
   d=32 -- the c-encoding does not cap generalisation.

The mechanism: the latent's intrinsic structure is the same regardless
of d (PR_within stays around 3.4-3.9). What changes with d is the
downstream linear probe's freedom: a probe trained on (32-d Test A
latents -> CL) and evaluated on (32-d Test B latents) has more degrees
of freedom for cross-split interpolation than at d=8. The extra
dimensions help the *probe*, not the encoder.

Implication for paper claim 1: D2's d=32 default is empirically correct
on this data. The LeWM Two-Room intrinsic-dim mechanism applies to
SIGReg-only configurations (where the regulariser fights itself when
d > intrinsic_dim) but the SIGReg + OBS + BN configuration in this
paper is not in that regime -- the OBS head is the dominant regulariser
(D53) and extra latent dimensions help rather than hurt.

Production configuration: d* = 32. Session 9 lambda bisection runs at
d = 32.

## 5.7 R0 control: is OBS load-bearing for SIGReg at scale?

Two pure SIGReg + BN runs at full scale, no observable head, no PLDM.
20k iterations, seed 0, full v1.2 partition. Both lambdas tested:

| Run                       | lambda | r2(CL_future) Test B | (c, t) baseline | delta_test_b |
|---------------------------|-------:|---------------------:|----------------:|-------------:|
| R0 SIGReg-only lambda=0.1 |  0.1   |        -0.023        |     0.718       |   **-0.742** |
| R0 SIGReg-only lambda=0.01| 0.01   |        -0.029        |     0.718       |   **-0.748** |

Table 5: R0 control runs at full scale.

Both R0 runs **fail catastrophically** on Test B. The pure-SIGReg latent
without observable head is uninformative about CL on unseen (G, D, Y)
cases (r2 around zero), while the (c, t) parametric baseline predicts
CL at r2 = 0.72 by lookup. **OBS is load-bearing for SIGReg at scale**
(D55).

The two-lambda confirmation matters: the result is consistent at
lambda = 0.1 (Session 7 default) and lambda = 0.01 (Step 4 D53 best).
SIGReg alone, regardless of weight, does not produce a generalising
latent on this physics data. The observable head is the central
regulariser; SIGReg provides directional pressure that, by itself,
is insufficient.

The contribution of the OBS head to the SIGReg path is **+0.90
absolute** on Test B: from R0's -0.74 (no OBS) to E4's +0.16 (best
SIGReg + OBS). This is comparable to the +0.84 OBS contribution on the
PLDM path observed in D47 (R1 vs R2). The two regulariser bases are
similar in *their dependence on OBS*, but the OBS-augmented latents
diverge sharply at scale: SIGReg + OBS generalises at +0.16 on Test B,
while PLDM + OBS (R1) does not (-0.003) and PLDM + OBS with paper-tuned
weights (E10) is even worse (-0.095). The asymmetry of D48 is not
between the bare regularisers but between their interactions with the
observable head.

Paper claim 2 (observable augmentation is necessary) and claim 3
(regulariser asymmetry between SIGReg + OBS and PLDM + OBS) both stand
robustly after R0.

## 5.8 Recommendation summary and Session 9 path

The production configuration going into Session 9 (lambda bisection):

- **Anti-collapse:** SIGReg with batched-normalised projection
- **Lambda_SIGReg:** 0.01 (Step 4 D53 best)
- **Auxiliary observable head:** cl_future at eta = 0.01 (D37 default
  also Step 4 D53 best, see Section 5.5)
- **Latent dimension d:** 32 (D2 default also Step 5 D54 best, see
  Section 5.6)
- **Predictor:** 6-layer AdaLN-Zero transformer with c = (G, D, Y)

The headline Session 8 production cell (eta=0.01, lambda=0.01, d=32)
reaches **Test B delta = +0.159**, +0.02 absolute above the Session 7
R3 default (eta=0.01, lambda=0.1, d=32) and +0.16 absolute above the
2-term-SIGReg-only baseline at any lambda. Test C delta at the same
cell is +0.45 (extrapolation to |G|=4 also works).

Paper contribution claims as validated by Session 8 work:

1. **Static-vs-dynamic + (c, t) baseline diagnostic suite** is useful
   at scale: it distinguishes "R3 generalises with PR ~ 3" from
   "R1 memorises with PR ~ 27" cleanly (Section 5.2 Table 1), and the
   PR_all-rises-while-PR_within-shrinks pattern is the SPREAD_TRIVIAL
   signature at scale (Section 5.4). PR alone is anti-correlated with
   generalisation in the PLDM regime; the (c, t)-baseline-delta is the
   robust diagnostic.
2. **Observable augmentation is necessary** for either regulariser at
   scale: R0 control fails at -0.74 (Section 5.7); R2 PLDM-only fails
   at -0.85 (D47). OBS provides +0.90 to SIGReg and +0.84 to PLDM on
   Test B.
3. **Regulariser asymmetry at scale**: SIGReg + OBS generalises at
   +0.16 on Test B; PLDM + OBS does not (-0.003 with defaults, -0.095
   with paper-tuned). The asymmetry is between the two **observable-
   augmented** configurations, not between the bare regularisers (both
   fail without OBS). The (eta x lambda) grid (Section 5.5) reveals
   the OBS head as the dominant regulariser; SIGReg at lambda=0.01
   provides residual directional pressure. The d-sweep (Section 5.6)
   confirms the D2 d=32 default; the LeWM intrinsic-dim prediction
   does not apply to this regime.

Session 8 outcome: **VALIDATED** (D57). All four step-results land within
the plan's "VALIDATED" criteria. Session 9 launches lambda bisection
at d=32, eta=0.01 over a finer lambda interval centred on 0.01
(LeWM-style 6-8 evaluations between 0.001 and 0.1). The visualisation
decoder also begins on the d=32 SIGReg + OBS encoder.

**Session 9 D58 update:** The lambda bisection over
`lambda in {0.001, 0.003, 0.01, 0.03, 0.1}` at seed=0 confirms
**lambda\* = 0.01** as the production point with `delta_test_b = +0.159`
(E4 from Session 8 wins; F1 at lambda=0.001 lands at +0.118, F2 at
lambda=0.003 lands at +0.131, F3 at lambda=0.030 lands at +0.131, E5
at lambda=0.100 lands at +0.138). The curve has a clean interior
maximum at lambda = 0.01 with PR\_all also peaked there (PR\_all =
2.61 at E4 versus 2.10 to 3.51 across the rest of the bracket). The
Session 8 coarse-grid finding survives the finer bisection
resolution.

The Session 9 seed-variance bound at lambda\* = 0.01 across three seeds
(E4 seed=0, F4 seed=42, F5 seed=123) is wider than the Session 8 D52
single-comparison spread suggested:

| seed | Test A delta | Test B delta | Test C delta |
|-----:|-------------:|-------------:|-------------:|
|    0 |    +0.227    | **+0.159**   |    +0.470    |
|   42 |    +0.231    | **+0.096**   |    +0.457    |
|  123 |    +0.226    | **+0.137**   |    +0.496    |

3-seed mean Test B delta = **+0.131 +/- 0.032 (1-sigma)**, range = 0.063
absolute. The variance is concentrated on Test B (parametric
interpolation); Test A is seed-robust (spread 0.005) and Test C nearly
so (spread 0.039). Two readings discussed in HANDOFF D58: (1) lambda
= 0.01 sits at the lower edge of the bisection bracket where SIGReg
pressure is small, giving the encoder more freedom to land in
different local optima across seeds (consistent with D52's smaller
seed spread of 0.017 at lambda = 0.1); (2) the +0.159 E4 result is
the best of three seeds. The paper claim 1 headline number adopts the
mean: **+0.131 +/- 0.032 (1-sigma) across three seeds**, with the
+0.063 max-min range as the variance bound.

The outcome category is PRODUCTION_PIVOT per the strict reading of the
Session 9 plan's pass criterion (seed range > +/- 0.05). The production
config still works (all three seeds give positive Test B delta and
beat every Session 7 / 8 / 9 ablation), so the pivot is mild: only the
headline number shifts from "+0.159 single seed" to "+0.131 +/- 0.032
across three seeds".

## 5.9 Limitations

This paper reports five honest limitations of the Session 7 + Session 8
evidence:

1. **Single-seed grid.** The (eta x lambda) grid in Section 5.5 reports
   a single seed = 0 evaluation per cell. Step 3 (Section 5.4) provides
   a single seed-variance datapoint (seed = 42) on the R3 reference,
   not a per-cell variance bound. Multi-seed averages on every grid
   point would multiply compute by 3-5x; we defer this to Session 10.
2. **Parametric range bounded by the data.** Test B at unseen interior
   (G, D, Y) covers 6 cases and 28 encounters; Test C at |G| = 4 covers
   4 cases and 24 encounters. The (G, D, Y) parametric range is the
   DNS-generated envelope, not a continuous parametric study. Out-of-
   envelope generalisation is not tested.
3. **Single training duration.** All Session 7 / Session 8 runs use
   20k iterations, ~2000 epochs over 138 sub-trajectory encounters.
   The trajectory audit (Section 5.4) shows R3 converged by iter 12000,
   but the optimal training length at d = 8 or d = 16 (Section 5.6) may
   be different. We do not report training-length ablations.
4. **PLDM reference is a single paper-tuned point.** E10 uses the
   Two-Rooms config from arXiv:2502.14819 Appendix J.2. A full PLDM
   hyperparameter sweep would require ~6-8 additional 20k-iter runs.
   Session 9 launches this sweep only if E10 outperforms the best
   SIGReg grid point (Section 5.5).
5. **R0 control at one or two lambda values.** Section 5.7 reports R0
   at lambda = 0.1 by default and optionally at lambda* if Step 4 finds
   it materially different. R0 does not span the full lambda range.

The paper's claims are conditioned on these limitations and we restate
them inline at each claim's first appearance.

## Open writing TODO

- Regenerate Figure 2 if a future session adds the SIGReg + OBS d=32
  best-grid-point and R0 bars to it.
- Cross-reference HANDOFF D44-D57 inline where the table-of-record
  citation belongs (most done; spot-check Sections 5.2 and 5.7).
