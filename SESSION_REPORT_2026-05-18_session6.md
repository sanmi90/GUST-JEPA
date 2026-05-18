# Session 6 report: factorial diagnostic

Date: 2026-05-18.
Branch: `session6-factorial`.
Plan: `SESSION6_FACTORIAL_DIAGNOSTIC.md`.

## Summary

Session 6 was the diagnostic session that disambiguates the Session 5
TRIVIAL / SPREAD_TRIVIAL outcome (D27) across five candidate root
causes. The plan defined four pass criteria for the session itself
(D35 absorption, CL data pipeline, audit notebook, five factorial runs
plus the decision-string analysis); the numerical pass criteria from
Session 5 (PR > 16, probe R^2 in 0.5 to 0.7) explicitly did not apply.

Outcome: **COMBINED_REMEDIATION** by the strict canonical_for_axis logic
(no JEPA axis fully active; partial axes = F-NC and F-OBS). The substantive
read is more nuanced: **PLDM-A is *already* active by the same audit
criteria** (PR_within=4.01, r2(z_dyn->phase)=0.58, r2(CL_future)=0.96 vs
the c+t baseline of 0.90), contrary to the Session 5 D31 reading of
"DATA_SCALE_BOUND". The Session 6 observable head:

  - **rescues SIGReg from TRIVIAL** (Run A's r2(CL_future)=-0.02 ->
    F-OBS's 0.95);
  - **marginally improves PLDM** (PR_within 4.01 -> 4.77, other metrics
    unchanged).

The regulariser asymmetry (observable is necessary for SIGReg, marginal
for PLDM) is the headline finding. Session 7 commits to a PLDM-focused
deepening rather than the strict COMBINED_REMEDIATION combination of
F-NC and F-OBS on SIGReg JEPA. Full reasoning in HANDOFF.md D39.

## What landed

### Step 0: absorb two new run3 cases (D35)

The collaborator left two new run3 files
(`Gust_032_x-1.844_y-0.872_s-1.5_d1.5.h5` and
`Gust_033_x-1.844_y-0.872_s3.0_d0.5.h5`). Regenerated the inventory and
the manifest:

- `data_manifest/raw_cases_inventory.yaml` SHA256
  `ce817e1e0df54309...` (from `dd984588...`).
- `configs/splits/split_v1.json` SHA256
  `a721dc92f6e278ee054bb952933c14ba20a58137f79f3a19fc6ad71b70a007dd`
  (from `7f8f6042...`).
- Counts: 49 -> 51 cases total; 39 -> 41 train cases; 132 -> 138 train
  encounters; 54 -> 56 Test A encounters. Test B (6 cases / 28 enc) and
  Test C (4 cases / 24 enc) unchanged.
- New cases default to `train` per `build_split_manifest.py`. The two
  new case_ids are `G-1.50_D1.50_Y-0.40` and `G+3.00_D0.50_Y-0.40`.
- Cache: 8 new encounter files written under `${VORTEX_JEPA_CACHE}/v1/`.

CLAUDE.md "Data" section updated to reflect 51 cases / 41 train / 56
test_a encounters. The v1 cache directory is unchanged (kept under `v1/`
because the binary format and preprocessing version are the same); the
post-D35 partition is called "v1.2" in session reports only.

The D-number renumbering note is recorded inline at the head of D35:
the Session 6 plan referred to this entry as "D33", but D33 and D34
were already taken by the prior run3 absorption (2026-05-17) and the
frame-skip housekeeping note (2026-05-18), so this session uses D35
through D39 instead.

### Step 1: CL(t+Delta) in the data pipeline (D36)

`src/data/episode_dataset.py` gained an `emit_cl_future` flag
(default off, backwards compatible) and a `cl_future_deltas` tuple
(default `(8, 16, 24)`). When enabled, each sample carries a `(L,
n_deltas)` float32 tensor with `CL(t + delta)` for each frame index
in the sub-trajectory, clamped to the last valid `C_L` when
`t + delta >= encounter_length`. The collate function in
`src/training/train_jepa.py` passes `cl_future` through the batch when
present.

Three new tests in `tests/test_episode_dataset.py`:

- `test_cl_future_shape_contract`     : verifies (L, n_deltas) float32 and
  the backwards-compatible default.
- `test_cl_future_no_nans_on_smoke_subset` : confirms finite values across
  all sub-trajectories on the 5-case smoke subset; CL envelope on the
  smoke subset spans roughly [-2.5, 7.8] because G = +/-3 cases spike
  during impact (the plan's "typically -1 to +3" prior was conservative).
- `test_cl_future_clamps_at_encounter_end` : forces a near-end start
  and verifies that the most-clamped column (`delta = 24`) is constant
  at `C_L[-1]` for the tail of the sub-trajectory.

CL was already cached per encounter (`C_L (120,)` in every cache file),
so no preprocessing changes were required. Only the dataset and
collate were touched.

### Step 2: static-vs-dynamic audit notebook

`notebooks/02_audit_static_dynamic.ipynb` loads the Run A
(SIGReg + BN) and Run C (VICReg + BN) iter=5000 checkpoints, encodes
all Test A encounters from the 5 smoke cases, decomposes the latent
into per-case mean and per-encounter dynamic parts, and reports eight
metrics. Findings (notebook outputs are committed):

| Metric                | Run A (SIGReg + BN) | Run C (VICReg + BN) |
|-----------------------|---------------------|---------------------|
| `PR_all`              |  1.02               | 27.00               |
| `PR_case_mean`        |  1.00               |  4.62               |
| `PR_within_case`      |  2.25               |  8.25               |
| `r2(z -> c)`          |  0.73               |  0.88               |
| `r2(z_dyn -> c)`      | -0.17               | -0.17               |
| `r2(z -> phase)`      |  0.13               |  0.25               |
| `r2(z_dyn -> phase)`  |  0.13               |  0.49               |
| `||z_dyn|| / ||z||`   |  0.10               |  0.83               |

Interpretation:

- Run A is clean TRIVIAL: 90 percent of the latent norm is in the case
  means, within-case PR is just above the "useful-dynamics" lower bound
  of 2, and there is no phase signal in the dynamic part.
- Run C is **not** a clean SPREAD_TRIVIAL: the dynamic part carries
  meaningful within-encounter phase information (r2 = 0.49, close to
  the 0.5 "active" threshold) and has zero residual c leak. The
  encoder uses 27 effective dimensions, has 8.25 within-case PR, and
  most of the latent magnitude is in the dynamic part. Closer to
  "high-rank with case-axis spread + nontrivial dynamics" than to a
  pathological pattern.

This refines the Step 4 decision-tree reading: F-NC and F-OBS (the
two axes most diametrically opposed to "case-axis spread") need to
clear the same 0.5 phase threshold that Run C is brushing against,
not the much higher bar Run C would imply if it were a true SPREAD_TRIVIAL.

### Step 3: factorial wiring and five F-* runs

Code surfaces landed:

- `src/models/observable_head.py` (new): the ObservableHead MLP and
  the `observable_loss` function. 7 unit tests in
  `tests/test_observable_head.py`.
- `src/models/predictor.py`: `AutoregressivePredictor` accepts
  `cond_dim=0`, which swaps the AdaLN-Zero blocks for standard pre-norm
  transformer blocks (`UnconditionedPredictorBlock`). 3 new tests in
  `tests/test_predictor.py` cover the unconditioned construction,
  one-step gradient flow (no warm-up needed without AdaLN), and the
  negative-cond_dim validation.
- `src/models/jepa.py`: `c_dropout_prob`, `observable_head`, and
  `observable_weight` parameters; new `loss_obs` key in the forward
  output dict. 4 new tests in `tests/test_jepa.py` cover c-dropout
  matching the cond=0 reference in train mode, c-dropout being a
  no-op in eval, observable-head loss + gradient flow, and the
  missing-cl_future error path.
- `src/training/train_jepa.py`: five new CLI flags
  (`--sub-trajectory-length`, `--rollout-horizon`, `--c-dropout-prob`,
  `--predictor-cond-dim`, `--observable-head` plus
  `--observable-head-weight` and `--observable-head-deltas`).
  `run_config` and the iteration log now include the new fields.
- `configs/cases/smoke_24cases.yaml` (new): 24-case F-S subset; 6
  periodic + 18 run3 spanning 12 G levels.

Latent dataset fix: `EpisodeDataset` now clamps `uniform_start_range`
and `impact_overlap_start_range` upper bounds to `(n_frames -
subtraj_len)`. The manifest was authored for L = 32 and hard-coded the
upper bound at 88; at L = 64 the unconstrained start would have asked
for frames [88, 152) from a 120-frame encounter. Behaviour at L = 32
is unchanged.

Test suite: 114 fast tests pass (97 pre-session + 17 new for D35/D36/
D37/D38/D39-supporting code). The slow GPU smoke test
(`tests/test_train_jepa_smoke.py`) was exercised in two short manual
smokes (F-NC with cond_dim=0 and F-OBS with the observable head) at
B=4, 30 iters, on the RTX 6000 before the long runs were kicked off.

### Step 3: five 5k-iter factorial runs

Each variant trains for 5000 iterations on the RTX 6000 with seed 0,
SIGReg + BN, diagnostic cadence every 250 iters, checkpoint every
1000. Output layout: `outputs/runs/session6/run_f_{l,cd,nc,s,obs}/`.

Runs launched via `scripts/run_session6_factorial.sh` (the script
writes per-variant `train.log` files and a top-level summary at
`outputs/runs/session6/_session6_summary.log`).

| Variant | Axis changed | PR_final | r2(z->c)_final | Reading |
|---------|--------------|---------:|---------------:|---------|
| F-L     | sub-trajectory length 64 + rollout horizon 16 |  1.01 |  0.79 | TRIVIAL (L axis inactive) |
| F-CD    | per-batch c-dropout 0.5                        |  1.03 |  0.60 | TRIVIAL (c-dropout inactive; lower r2 but still collapsed) |
| F-NC    | predictor cond_dim = 0                         |  1.02 |  0.76 | TRIVIAL (D6 arch not the root cause) |
| F-S     | 24-case scale-up                               |  1.03 |  0.64 | TRIVIAL (not data-scale-bound at 24 cases) |
| F-OBS   | observable CL head, eta = 0.01                 |  3.11 |  0.99 | partially_active (PR climbs above 1) |

Run order on the two RTX 6000 cards (Hardware D19, confirmed two
RTX-6000-Blackwell cards on the workstation; CLAUDE.md only mentioned
one before this session):

- cuda:2 (GPU 0): F-L (0-50 min) -> F-CD (51-90 min) -> F-S (91-120 min)
- cuda:3 (GPU 1): F-NC (0-32 min) -> F-OBS (33-65 min)

Parallelism on the second RTX 6000 was proposed mid-session; the wrapper
script was killed after F-L finished and two new scripts were launched
to split the remaining four variants across the two cards. Total wall
clock dropped from ~2.5 hours (sequential) to ~2.0 hours, with the
saving used to fit two in-session extensions (F-OBS @ 10k and PLDM+OBS).

In-session extensions (motivated by F-OBS being the only obvious escape
at the time the extensions were proposed):

| Variant   | iters | PR_all_final | r2(z->c)_final | Reading |
|-----------|------:|-------------:|---------------:|---------|
| F-OBS @ 10k (resume) | 10000 |  3.83 |  0.98 | still partially_active (slow PR drift ~+0.2/1k confirmed) |
| PLDM+OBS             |  5000 | 12.42 (last diag) |  0.96 | active by the Session 6 bar; only marginally above PLDM-A on the audit-style metrics |

`--resume-from` was added to `train_jepa.py` mid-session to support the
F-OBS @ 10k extension; the observable head was wired into `PLDMWrapper`
and `train_baseline.py` to support the PLDM+OBS extension. Both changes
came with unit tests (2 new tests in `tests/test_pldm_wrapper.py`).

The PLDM+OBS extension was originally framed as "does the observable
head also break PLDM's collapse?" The audit answer is more interesting:
PLDM was not collapsed in the first place (the Session 5 D31 "DATA_SCALE_BOUND"
reading was too pessimistic), and the observable head provides only a
marginal improvement on top. The asymmetric value of observable
augmentation (SIGReg vs PLDM) is the substantive Session 6 finding
that the strict decision-tree string does not capture.

### Step 4: factorial analysis notebook + decision string

`notebooks/03_factorial_analysis.ipynb` was executed with all 9 rows
(8 variants + 1 PLDM-A baseline) and produced the audit table above,
loss-curve panel, per-axis classify() reading, and the decision string
`COMBINED_REMEDIATION` (per the strict canonical_for_axis logic). The
notebook outputs are committed so reviewers can inspect the loss
curves and the metric table without rerunning. The `r2(CL_future)`
column shows that the four observable-coupled or PLDM rows
(PLDM-A, PLDM+OBS, F-OBS, F-OBS @ 10k) all beat the
`baseline_ct(c, t) -> CL_future` MLP at r2=0.902, while the four
pure-SIGReg JEPA axes (Run A, F-L, F-CD, F-NC, F-S) score below zero
(worse than predicting the mean CL).

### Step 4: factorial analysis notebook and decision string

`notebooks/03_factorial_analysis.ipynb` loads all six runs (Run A +
the five Session 6 variants), repeats the static-vs-dynamic audit on
each, fits the two CL-prediction baselines (`baseline_ct(c, t) ->
CL_future` and `baseline_jepa(z_t) -> CL_future`), classifies each
axis as `active / partially_active / inactive / regressed`, and prints
the Session 6 decision string from the six-outcome menu.

**Decision string** (strict, from `notebooks/03_factorial_analysis.ipynb`
Section 5): `COMBINED_REMEDIATION`. Partial axes: F-NC and F-OBS. No
JEPA axis fully active. Strict reading: Session 7 should run F-NC + F-OBS
combinations on SIGReg JEPA.

**Substantive reading** (uses the full audit table, not just the JEPA
axes): PLDM-A and PLDM+OBS are both **active** by the same bar. The
observable head's bigger role is rescuing SIGReg from TRIVIAL than
boosting PLDM. Session 7 should therefore investigate PLDM more
deeply rather than focus on the SIGReg-side combination.

**Session 7 plan**: see D39 for the three-track proposal:

  1. **Session 7-PLDM-DEEP** (4h): PLDM-A at 20k iters on the full
     41-train-case partition. Confirms the active-by-default reading.
  2. **Session 7-OBS-PLDM** (8h): sweep eta in {0, 0.001, 0.005, 0.01,
     0.05} on PLDM at 20k iters on the full partition. Picks the operating
     point that maximises `r2(z -> CL_future)` on Test A with PR_within
     in the healthy 0.3d to 1.0d window.
  3. **Session 7-COMB-NC-OBS (optional, ~2h)**: the strict
     COMBINED_REMEDIATION path. Combines F-NC and F-OBS on SIGReg JEPA
     as a control check that the JEPA-side combination does not
     unexpectedly outperform PLDM.

## Pre-registered methodological note

Session 6 deliberately tests five axes that were not all part of the
original architectural specification. The original spec committed to
`SIGReg + BN + L = 32 + c-at-predictor + no observable` as the recipe.
Session 5 showed this recipe lands in TRIVIAL or SPREAD_TRIVIAL at the
5-case scale. Session 6 tests which axis of that recipe is the
rate-limiting one. Each of the five axes corresponds to a published
precedent:

- F-L: longer sub-trajectories in V-JEPA 2 (arXiv:2506.09985).
- F-CD: classifier-free guidance via condition dropout (Ho and
  Salimans, arXiv:2207.12598).
- F-NC: Brain-JEPA / Echo-JEPA, where the encoder is fully
  responsible for encoding subject-level information.
- F-S: standard data-scale ablation in self-supervised learning.
- F-OBS: observable augmentation (Fukami and Taira JFM 2023; Fukami,
  Nakao, Taira JFM 2024; Fukami 2025 transonic buffet JFM 2025).

The session is not an ad-hoc rescue attempt; it is a controlled
factorial test of methodological choices each of which is defensible
in isolation.

## What did NOT land

- Hydra configs, `torch.compile()`, dynamic-IDM PLDM with CL target,
  symmetry augmentation, frame-skip ablation, the other three baselines
  (POD, Fukami AE, Solera-Rico beta-VAE), and the 80k full training are
  all explicitly out of scope per the plan's "Out of scope for Session 6"
  section.
- The Solera-Rico beta-VAE comparator session is planned as a separate
  parallel session and was not started.

## Files added or modified

- `CLAUDE.md`: data counts bumped to 41 / 51 / 56. Hardware section
  remains unchanged from this session even though we discovered there
  are actually TWO RTX 6000 Blackwell cards on the workstation, not
  one; the Session 6 use of cuda:3 for parallelism is recorded in
  D38/D39 and the run-script comments. A separate housekeeping pass
  should update CLAUDE.md "Hardware" to acknowledge the second card
  and document the CUDA_VISIBLE_DEVICES=3 pattern used here.
- `HANDOFF.md`: entries D35, D36, D37, D38 added; D39 added after the
  decision string lands.
- `SESSION6_FACTORIAL_DIAGNOSTIC.md`: the plan, checked in.
- `configs/cases/smoke_24cases.yaml` (new).
- `configs/splits/split_v1.json`: regenerated for the v1.2 manifest.
- `data_manifest/raw_cases_inventory.yaml`: regenerated.
- `notebooks/02_audit_static_dynamic.ipynb` (new, with executed outputs).
- `notebooks/03_factorial_analysis.ipynb` (new, executed after the
  factorial runs complete).
- `scripts/run_session6_factorial.sh` (new): the sequential five-run
  launcher.
- `src/data/episode_dataset.py`: `cl_future` emission + start-range clamping.
- `src/models/jepa.py`: c-dropout, observable head, `loss_obs` key.
- `src/models/observable_head.py` (new).
- `src/models/predictor.py`: cond_dim=0 path with `UnconditionedPredictorBlock`.
- `src/training/train_jepa.py`: 5 new CLI flags + run_config + logging.
- `tests/test_episode_dataset.py` (new): 3 dataset-level tests.
- `tests/test_observable_head.py` (new): 7 tests.
- `tests/test_predictor.py`: +3 tests for the unconditioned variant.
- `tests/test_jepa.py`: +4 tests for c-dropout and observable head.

Commits on `session6-factorial` (in order):

1. `4e77ff9` Session 6 Steps 0-1: absorb 2 run3 cases (v1.2) and add CL(t+Delta) to loader
2. `ef1e965` Session 6 Step 2: static-vs-dynamic audit notebook for Run A and Run C
3. `889a002` Session 6 Step 3 wiring: observable head, c-dropout, cond_dim=0 predictor, 5 new CLI flags
4. `d739919` Clamp dataset uniform_start_range to (0, n_frames - subtraj_len)
5. `216429d` Session 6 Steps 0-3 prep: D36/D37/D38 entries, run script, report skeleton
6. `592f2f0` Add --resume-from to train_jepa.py for restarting from a checkpoint
7. `e967494` Add observable head to PLDMWrapper + train_baseline.py for PLDM+OBS variant
8. [final commit] Session 6 Step 4 + D39 + PLDM+OBS observable head tests + report fill-in

## Suggested next session

See D39's "Session 7 follow-up" line. The session-7 plan is one of:

- Session 7-CRC   (lambda bisection on a single fixed-axis configuration)
- Session 7-COMB  (factorial combinations of partial axes)
- Session 7-OBS   (observable head sweep, then lambda bisection)
- Session 7-SCALE (full-train-cases scale run, then lambda bisection)
- Session 7-PIVOT (dynamic-IDM PLDM + Solera-Rico comparator)
- Session 7-AMBIG (replicate active axes with different seed)
