# Section 5 — Full-scale results (SKELETON)

LaTeX-friendly markdown. This section is a skeleton; the actual
numbers fill in after `notebooks/05_session7_full_evaluation.ipynb`
executes against the three completed Session 7 runs (R1/R2/R3,
20k iters each, full 41-train-case v1.2 partition).

The deliverable of this section is one complete table (Table 1
below) and one figure (Figure 2 below). The text walks through the
table, names the Session 7 decision-string outcome, and discusses
what the outcome implies for the paper's three contribution claims.

## 5.1 Experimental setup

The three full-scale runs of Section 5 differ in exactly the axes
Section 4 identified as discriminating:

- **R1: PLDM + observable head, BatchNorm projection.** The bundle
  Section 4.2 identified as healthy on the 5-case smoke. R1 tests
  whether that healthiness scales to 41 train cases.
- **R2: PLDM only, eta=0, BatchNorm projection.** The OBS-vs-no-OBS
  control. R2 isolates the observable head's contribution at scale.
- **R3: SIGReg + observable head, BatchNorm projection.** The
  regulariser-asymmetry test. Section 4.2 showed OBS rescuing SIGReg
  from TRIVIAL on the 5-case smoke; R3 tests whether the rescue
  persists at 41 cases.

All three runs train for 20,000 iterations on the full v1.2 train
partition (138 train sub-trajectory encounters across 41 cases),
with seed 0, batch size 16, sub-trajectory length L=32, eta=0.01
where applicable. Detailed configuration in Section 3.4 and in the
launcher script `scripts/launch_session7.sh`.

Each run is evaluated on three held-out splits:

- **Test A** (in-sample held-out encounters within train cases; 56
  encounters from 41 cases): the encoder has seen these cases, just
  not these specific encounters. Test A is essentially an in-
  distribution generalisation test and is a sanity check, not the
  headline metric.
- **Test B** (parametric interpolation; 6 cases, 28 encounters):
  cases at unseen (G, D, Y) values intermediate to training ranges.
  Test B is the primary success metric — it tests whether the encoder
  learned (G, D, Y) as continuous parameters rather than as discrete
  labels.
- **Test C** (extrapolation; 4 cases at |G|=4, 24 encounters): cases
  beyond the training G range. Test C is a stretch goal; failure is
  the expected outcome at this training scale.

## 5.2 Table 1 — complete metric table (TBD)

For each of the three runs and each of the three test splits, the
following metrics are computed by `notebooks/05_session7_full_evaluation.ipynb`
Section 4.

| Run             | Split | PR_all | PR_within | r2(z->c) | r2_dyn_phase | r2(CL_future) | (c,t) baseline | delta |
|-----------------|-------|-------:|----------:|---------:|-------------:|--------------:|---------------:|------:|
| R1 PLDM+OBS+BN  | Test A| TBD    | TBD       | TBD      | TBD          | TBD           | TBD            | TBD   |
| R1 PLDM+OBS+BN  | Test B| TBD    | TBD       | TBD      | TBD          | TBD           | TBD            | TBD   |
| R1 PLDM+OBS+BN  | Test C| TBD    | TBD       | TBD      | TBD          | TBD           | TBD            | TBD   |
| R2 PLDM only BN | Test A| TBD    | TBD       | TBD      | TBD          | TBD           | TBD            | TBD   |
| R2 PLDM only BN | Test B| TBD    | TBD       | TBD      | TBD          | TBD           | TBD            | TBD   |
| R2 PLDM only BN | Test C| TBD    | TBD       | TBD      | TBD          | TBD           | TBD            | TBD   |
| R3 SIGReg+OBS+BN| Test A| TBD    | TBD       | TBD      | TBD          | TBD           | TBD            | TBD   |
| R3 SIGReg+OBS+BN| Test B| TBD    | TBD       | TBD      | TBD          | TBD           | TBD            | TBD   |
| R3 SIGReg+OBS+BN| Test C| TBD    | TBD       | TBD      | TBD          | TBD           | TBD            | TBD   |

The (c, t) baseline column is a tiny MLP probe `(case_descriptor,
frame_index) -> CL(t + Delta)` fit on Test A and evaluated on each
split. The delta column is `r2(z -> CL_future) - r2(c, t -> CL_future)`
on the same split. A positive delta on Test B means the encoder learned
generalisable flow physics beyond a case-frame lookup.

## 5.3 Figure 2 — delta on Test B across the three runs (TBD)

Bar chart of `delta_test_b` for R1, R2, R3, with the (c, t) baseline at
zero and horizontal dashed lines at 0.03 (the WEAK_GO threshold) and
0.10 (the STRONG_GO threshold). The figure is the headline plot of
the paper's quantitative section.

## 5.4 Decision-tree outcome (TBD)

The decision-string outcome from `notebooks/05_session7_full_evaluation.ipynb`
Section 6 belongs here as a single sentence: "The Session 7 outcome
is **<one of>**:
STRONG_GO / WEAK_GO / PLDM_ALONE_VIABLE / OBS_NECESSARY /
REGULARISER_ASYMMETRY / R3_WINS / ALL_FAIL / TEST_B_TEST_A_DISCREPANCY".

The accompanying paragraph translates the outcome into the implication
for each of the paper's three contribution claims:

- Contribution 1 (the static-vs-dynamic + (c, t) baseline diagnostic
  suite): is it useful at scale? (Almost certainly yes regardless of
  the outcome.)
- Contribution 2 (observable-augmentation rescue regime): does the
  Session 6 finding (OBS rescues SIGReg from TRIVIAL) persist at
  scale? Answered by R3's Test B delta.
- Contribution 3 (regulariser asymmetry): does PLDM+OBS materially
  beat SIGReg+OBS at scale? Answered by R1-vs-R3 delta on Test B.

## 5.5 Case-conditional leave-one-out probe (TBD)

`notebooks/05_session7_full_evaluation.ipynb` Section 5 will compute a
leave-one-case-out cross-validation across the 41 train cases for each
of the three runs. The result is reported here as a robustness check
against the small (6-case) Test B sample: if the leave-one-out delta
agrees with the Test B delta, the Test B result is robust.

## 5.6 Discussion (TBD)

Discussion paragraph fills in after the table lands. Likely structure:

- "The headline finding from Table 1 is X."
- "Test C results across all three runs are <weak/strong>;
  extrapolation to |G|=4 is <not / surprisingly> achievable at the
  current training scale."
- "The R1 vs R2 delta on Test B is <positive/zero/negative>, which
  <confirms/refutes> the necessity of the observable head at full scale."
- "The R1 vs R3 delta on Test B is <positive/zero/negative>, which
  <confirms/refutes> the regulariser asymmetry claim at full scale."
- One paragraph mapping the eight-branch decision outcome to the
  Session 8 follow-up.

## Open writing TODO

- After notebook 05 lands, replace every TBD with the actual number.
- Generate Figure 2 (delta_test_b bar chart). Use matplotlib so the
  figure regenerates with the notebook.
- Add a Limitations subsection if ALL_FAIL or TEST_B_TEST_A_DISCREPANCY
  lands; the honest checkpoint discipline requires explicit limitations
  language.
- Cross-reference the Session 6 D39 audit table for the smoke-scale
  comparison; one inline citation to that decision entry is sufficient.
