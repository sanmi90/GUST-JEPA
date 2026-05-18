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

Outcome: TBD (filled in after the factorial analysis notebook executes
and the decision string is recorded as D39).

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

| Variant | Axis changed | Status |
|---------|--------------|--------|
| F-L     | sub-trajectory length 64 + rollout horizon 16 | TBD |
| F-CD    | per-batch c-dropout 0.5 | TBD |
| F-NC    | predictor cond_dim = 0 | TBD |
| F-S     | 24-case scale-up | TBD |
| F-OBS   | observable CL head, eta = 0.01 | TBD |

[Per-variant final metrics filled in after `_session6_summary.log`
records exit codes.]

### Step 4: factorial analysis notebook and decision string

`notebooks/03_factorial_analysis.ipynb` loads all six runs (Run A +
the five Session 6 variants), repeats the static-vs-dynamic audit on
each, fits the two CL-prediction baselines (`baseline_ct(c, t) ->
CL_future` and `baseline_jepa(z_t) -> CL_future`), classifies each
axis as `active / partially_active / inactive / regressed`, and prints
the Session 6 decision string from the six-outcome menu.

[Decision string and the recommended Session 7 plan filled in after
the notebook executes.]

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

- `CLAUDE.md`: data counts bumped to 41 / 51 / 56.
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

Commits on `session6-factorial`:

```
[hash] Session 6 Steps 0-1: absorb 2 run3 cases (v1.2) and add CL(t+Delta) to loader
[hash] Session 6 Step 2: static-vs-dynamic audit notebook for Run A and Run C
[hash] Session 6 Step 3 wiring: observable head, c-dropout, cond_dim=0 predictor, 5 new CLI flags
[hash] Clamp dataset uniform_start_range to (0, n_frames - subtraj_len)
[hash] Session 6 Step 3 results + Step 4 analysis + D39 decision string
```

The final commit lands the executed `notebooks/03_factorial_analysis.ipynb`
and the D39 entry in `HANDOFF.md` once the runs complete.

## Suggested next session

See D39's "Session 7 follow-up" line. The session-7 plan is one of:

- Session 7-CRC   (lambda bisection on a single fixed-axis configuration)
- Session 7-COMB  (factorial combinations of partial axes)
- Session 7-OBS   (observable head sweep, then lambda bisection)
- Session 7-SCALE (full-train-cases scale run, then lambda bisection)
- Session 7-PIVOT (dynamic-IDM PLDM + Solera-Rico comparator)
- Session 7-AMBIG (replicate active axes with different seed)
