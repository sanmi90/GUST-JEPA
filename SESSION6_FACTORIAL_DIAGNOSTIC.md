# SESSION6_FACTORIAL_DIAGNOSTIC.md

Session 6 plan for the vortex-jepa project.

Last updated: 2026-05-18.

## Replaces the previous Session 6

The previous Session 6 plan was "Hydra + torch.compile + lambda bisection."
That session is deferred. The Session 5 results (Run A landed TRIVIAL with
PR = 1.025 and probe R^2 = 0.779; Run C landed in a new SPREAD_TRIVIAL mode
with PR = 17.46 and probe R^2 = 0.887) make the lambda bisection premature:
we do not yet know whether the trivial pattern is regulariser-bound,
data-scale-bound, c-leakage-bound, sub-trajectory-length-bound, or
self-supervision-insufficient-bound. Bisecting lambda on a model that fails
all reasonable configurations would be tuning the wrong knob.

This session is the factorial diagnostic that disambiguates the five
candidate root causes. Each candidate gets one controlled single-axis run.
The output is a clean attribution of which axis (or combination) restores
a healthy latent, plus a decision string mapping each outcome to the
next specific session.

## Session goal

Five factorial single-axis runs, plus a static-vs-dynamic audit on Run A
and Run C checkpoints, plus a data-pipeline change to add lift coefficient
CL(t+Δ) as an observable target. Then an analysis notebook that produces
the decision string.

|Step|Output|
|---|---|
|Step 0|Two new run3 cases absorbed into v1 partition. Partition becomes v1.2, sha256 updated, counts updated, D33 recorded.|
|Step 1|`src/data/episode_dataset.py` emits CL(t+Δ) alongside omega and c. Backward-compatible: existing entrypoints work unchanged.|
|Step 2|`notebooks/02_audit_static_dynamic.ipynb` audits Run A and Run C checkpoints. Produces PR_all, PR_case_mean, PR_within_case, plus three probe metrics on z and z_dyn.|
|Step 3|Five factorial runs: F-L (L=64), F-CD (c-dropout=0.5), F-NC (no c at predictor), F-S (scale to 24 cases), F-OBS (observable CL head, η=0.01). Each at 5k iters.|
|Step 4|`notebooks/03_factorial_analysis.ipynb` reads all five W&B runs plus Session 5's Run A baseline, produces the decision string, and recommends the next session.|

This session has no module-level new code beyond the data pipeline addition
and one observable head class. Everything else is configuration changes
and analysis. Estimated wall-clock: 6 to 8 hours including reading.

## Why not run all five axes plus combinations?

Five single-axis runs at 5k iters costs ~2 hours of GPU. Combinations
(L=64 AND c-dropout=0.5, etc.) double or quadruple the cost without
adding interpretability until we know which single axes matter. The
factorial design here is deliberately incomplete: it tests one axis at
a time, and only after Step 4's analysis do we know which combinations
to run in the next session.

## Locked decisions baked into Session 6

1. **CL is the canonical dynamic observable target.** Replaces "time-to-
   impact" and "vortex centroid" from the collaborator report. Rationale:
   CL is the aerodynamically meaningful quantity, aligns with Fukami's
   lift-augmented AE (Fukami and Taira, Journal of Fluid Mechanics 2023,
   arXiv:2305.18394; observable-augmented manifold learning, Fukami,
   Nakao, Taira, JFM 2024), and aligns with Solera-Rico, Sanmiguel Vila
   et al. (Nat Commun 2024, arXiv:2304.03571). It is also what the
   eventual digital-twin objective cares about. Recorded as D34.

2. **Observable head as auxiliary loss, NOT as primary loss.** Weight
   η = 0.01 keeps the JEPA self-supervised character while pressuring
   the encoder to retain CL-relevant information. The architectural
   inheritance is from Fukami: the encoder maps omega -> z, an auxiliary
   head maps z -> CL_future, and the JEPA prediction + anti-collapse
   loss continues to operate on z. Recorded as D35.

3. **Five factorial runs at 5 cases each (except F-S which is 24).** The
   F-S run is the scale test; all others hold case count at 5 to isolate
   the relevant axis. Recorded as D36.

4. **Decision tree maps outcomes to next session.** Each of the five runs
   produces a (PR_within_case, r2(z_dyn -> CL_future), r2(z -> c))
   triple. The analysis notebook prints which axes are "active"
   (improve PR_within_case and CL skill, reduce c-leakage). The decision
   string commits Session 7 to a specific configuration. Recorded as
   D37 (conditional on outcome).

## arXiv references (verified)

| Reference | arXiv ID / DOI | Use for |
|---|---|---|
| Fukami, Taira, "Grasping extreme aerodynamics on a low-dimensional manifold" | arXiv:2305.18394 (Nat Commun 2023) | The lift-augmented AE recipe: encoder + decoder + auxiliary CL head, β-VAE flavor |
| Fukami, Nakao, Taira, "Observable-augmented manifold learning for multi-source turbulent flow data" | JFM 2024, arXiv: TBD (verify in session if needed) | Generalization of the lift-augmented framework; relevant if the agent needs to read more on observable supervision |
| Solera-Rico, Sanmiguel Vila, et al. | arXiv:2304.03571 (Nat Commun 2024) | β-VAE + transformer ROM; Carlos's own group, the strongest internal-coherence baseline. The paper's compact-near-orthogonal-ROM framing is the precedent for D2's d=32 latent. |
| Fukami, Taira, "Compact representation of transonic airfoil buffet flows with observable-augmented machine learning" | arXiv:2509.17306 (JFM 2025) | Lift-augmented 3D latent on transonic buffet; demonstrates that observable supervision reduces intrinsic dim from ~10 to 3 on a structured fluid problem. Direct precedent for our η=0.01 choice. |
| LeWM | arXiv:2603.19312 | Two-Room failure mode (Section 5); we already replicated this in Run A. Re-cite if needed for paper framing. |
| LeJEPA | arXiv:2511.08544 | SIGReg derivation. The agent should NOT re-derive SIGReg behaviour from this paper; consult Section 4 only if interpretation of unexpected anti-collapse dynamics is needed. |
| PLDM | arXiv:2502.14819 | Multi-step L_sim (Equation 3.3); use for the IDM-with-CL design in F-OBS rather than the static-c IDM (which was the original D8 error, now corrected in D32). |
| Ho, Salimans, "Classifier-Free Diffusion Guidance" | arXiv:2207.12598 | Background for c-dropout (Run F-CD). The technique is well-established in diffusion; we are extrapolating to JEPA conditioning. Cite if the paper text describes c-dropout's lineage. |

If during analysis the agent finds an interpretation that depends on a
paper section, use the arXiv MCP plugin to verify the section content
directly. Several prior sessions had references that turned out to be
incorrect when checked against the primary source (the D8 PLDM citation,
the LeJEPA Figure 1 citation). Do not assume references are correct
without checking.

## Step 0: absorb two new run3 cases (mandatory, BEFORE anything else)

The collaborator left two new run3 cases in the run3 folder. They must be
absorbed into v1 before the experiments run, so the F-S scale run (which
uses up to 24 cases) and any future scale curve uses the most recent
partition.

### Cases to absorb

The agent must:

1. Identify the two new cases. Look in `${PREVENT_ROOT}/data/raw/run3/`
   for case directories whose names are not already in
   `configs/splits/split_v1.json`. There should be exactly two.

2. Verify each case has the standard run3 structure (frame indices,
   alpha = 14 deg metadata, vorticity in `omega_z.h5`, force coefficients
   in `force.csv` or equivalent). If the structure differs, STOP and
   report; do not absorb non-standard cases.

3. Run `python -m src.data.build_split --absorb-new-cases` (or the
   equivalent script if the entrypoint differs). The script should:
   - assign each new case to a split bucket (default: train, since v1's
     ratio is roughly 78 percent train);
   - compute encounter counts per case using the existing
     `impact_overlap_start_range` logic;
   - update `configs/splits/split_v1.json` in place;
   - recompute the manifest sha256;
   - write `data_manifest/raw_cases_inventory.yaml` with the updated
     case list.

4. Verify counts. Expected after absorption: 49 cases total (47 + 2),
   train encounters ~134 to 138 (depending on per-case encounter count;
   exact number to be reported in the session report). Update CLAUDE.md
   "Locked decisions, Data" section line 4 with the new totals.

5. Record D33 in HANDOFF.md:

```
### D33: Absorbed two new run3 cases into v1.2 (2026-05-18)

Two new run3 cases identified by the collaborator: <case_id_1>, <case_id_2>
(filled in by the agent after Step 0 of Session 6). The absorption is in-place
on v1 rather than triggering a v2 cut, consistent with the D14/D15/D20
absorption rule (in-place absorption while v1 has no paper-reportable
checkpoint yet). Partition counts after absorption: 49 cases total (37 + 2
train cases identified; 50 Test A, 28 Test B, 24 Test C, 1 Baseline
calibration reference). Total train encounters: <X> (to be filled by
agent). Split sha256: <new_sha256> (to be filled).
```

### Pitfalls

- The script must update the sha256 BEFORE recording D33's hash, not after.
- The training entrypoint reads the manifest at startup; any cached
  manifest in memory from a previous session is stale.
- Run `pytest tests/test_episode_dataset.py` after absorption to confirm
  the data loader still passes the four-check smoke test with the new
  partition.

## Step 1: add CL(t+Δ) to the data pipeline (mandatory)

The encoder takes vorticity. The new auxiliary task takes a future lift
coefficient. The data loader must emit both.

### Where CL comes from

Per-encounter HDF5 metadata or sibling files contain the time series of
force coefficients. Each frame has a CL value computed from pressure
integration over the airfoil surface. The data pipeline must:

1. Read the force-coefficient series for each encounter at preprocessing
   time. Cache as `cl.h5` (one 1D array per encounter, length L_full).
2. At `__getitem__` time, extract a sub-trajectory of vorticity (the
   current behaviour) AND the corresponding sub-trajectory of CL plus
   the CL values at indices (t + Δ_1, t + Δ_2, t + Δ_3) for the auxiliary
   prediction.

### Configuration

- Δ values: `delta_frames = (8, 16, 24)`. At `dt_eff = 0.1`, these are
  0.8, 1.6, 2.4 t/c into the future. Covers short, medium, and long
  forecasting horizons relative to the impact dynamics (5 to 20 t/c).
- All three Δ predictions are produced by separate observable heads
  (see Step 3, F-OBS). The data loader simply emits a `(B, T, 3)`
  tensor of "CL at +8, +16, +24 frames ahead of each frame in the
  sub-trajectory."

### Implementation note

`EpisodeDataset.__getitem__` should be modified to optionally emit CL,
gated by a constructor flag `emit_cl_future: bool = True`. Existing
training scripts that did not request CL continue to work; the new field
is simply absent from the batch dict. Backwards compatibility matters
because Session 5's Run A/B/C/D checkpoints will be re-evaluated in
Step 2 and the audit code reads them without CL.

The batch contract becomes:

```
{
    'omega': (B, T, 1, 192, 96),
    'c':     (B, 3),
    'cl_future': (B, T, 3),    # CL at frame index + (8, 16, 24)
}
```

### Sanity check

After implementing, run a smoke `python -m src.training.train_jepa
--cases-from configs/cases/smoke_5cases.yaml --max-iters 10
--diagnostic-every 5 --output-dir /tmp/cl_smoke --wandb-mode disabled`
and confirm the batch contains `cl_future` with finite values in a
plausible range (CL for our gust dataset is typically -1 to +3).

### Unit tests

`tests/test_episode_dataset.py` gets two new tests:

```python
def test_cl_future_shape_contract():
    """With emit_cl_future=True, batch contains cl_future of shape
    (B, T, 3) and dtype float32."""

def test_cl_future_no_nans():
    """No NaN or Inf in cl_future across the full 5-case smoke subset.
    Edge cases near the end of an encounter (where t+24 exceeds the
    encounter length) are handled by clamping to the last valid frame,
    with a logged warning."""
```

The "clamp at the end" choice is a design call. Alternative: drop
frames where t+24 exceeds encounter length. Clamping is the cheaper
choice and is documented as a known limitation; CL near the end of
the encounter is in the post-impact relaxation regime and is roughly
stationary, so clamping introduces small bias rather than spurious
signal. Record this choice in the session report.

## Step 2: static-vs-dynamic audit on Run A and Run C (mandatory)

Existing checkpoints from Session 5 are at
`outputs/runs/smoke5k/run_a_sigreg_bn/checkpoint_iter005000.pt` and
`outputs/runs/smoke5k/run_c_vicreg_bn/checkpoint_iter005000.pt`.
Audit them in a single notebook before running any new experiments.

### What the audit computes

For each checkpoint, on a held-out batch (use Test A encounters from
the 5 Session 5 cases; do not need Step 1's CL pipeline for this audit
because all metrics here are about z, not about CL prediction):

```python
z_all          : (N_encounters, T, d)  encoder outputs across all frames
z_case_mean    : (N_cases,      1, d)  mean of z over time and encounters per case
z_dyn          : (N_encounters, T, d)  z_all minus z_case_mean (broadcast)
```

Then report eight metrics:

```python
PR_all          = participation_ratio(z_all.reshape(-1, d))
PR_case_mean    = participation_ratio(z_case_mean.reshape(-1, d))
PR_within_case  = mean over cases of participation_ratio(z_dyn restricted to one case)

r2_z_to_c           = linear_probe(z_all, c)
r2_z_dyn_to_c       = linear_probe(z_dyn, c)
r2_z_to_phase       = linear_probe(z_all, frame_index_within_encounter)
r2_z_dyn_to_phase   = linear_probe(z_dyn, frame_index_within_encounter)
```

### Interpretation

- If `PR_within_case < 2`: the encoder has compressed all temporal
  variation into the case mean. There is no within-case dynamic signal.
- If `r2_z_dyn_to_phase > 0.5`: the dynamic part of z captures temporal
  structure within an encounter (the encoder is doing useful work).
- If `r2_z_dyn_to_c > 0.5`: even the "dynamic" part still leaks c.
  This is the SPREAD_TRIVIAL signature.
- A clean SPREAD_TRIVIAL pattern from Run C would be:
  PR_within_case low (say 2-4), r2_z_to_c high (~0.9), r2_z_dyn_to_c
  still moderate (~0.5), r2_z_dyn_to_phase near zero.

### Implementation

`notebooks/02_audit_static_dynamic.ipynb` has six sections:

1. Load checkpoint, load 5-case Test A subset, encode all frames.
2. Compute z_case_mean and z_dyn.
3. Compute the eight metrics for Run A and Run C.
4. Side-by-side bar chart comparing the metrics.
5. PCA visualization of z_all colored by c, then z_dyn colored by
   frame_index_within_encounter.
6. Short markdown summary: "Run A status:" and "Run C status:" with
   one-line interpretation.

Run this BEFORE Step 3's new experiments. The audit may reveal that Run
C's SPREAD_TRIVIAL is partially redeemed by some within-case structure,
which would change Step 3's design priorities.

## Step 3: five factorial single-axis runs

After Steps 0, 1, 2 are complete, run five JEPA training runs that each
change exactly one axis from the Session 5 Run A baseline.

### Common setup (all five runs)

|Parameter|Value|
|---|---|
|Anti-collapse|SIGReg with M=256 (Session 4 default)|
|Projection norm|BatchNorm (D17 default)|
|Seed|0|
|Max iters|5000|
|Diagnostic every|250|
|Checkpoint every|1000|
|Log every|25|
|Output dir|`outputs/runs/session6/<run_id>`|
|W&B mode|online|
|W&B group|`session6_factorial`|

### The five runs

#### F-L: longer sub-trajectory

```bash
python -m src.training.train_jepa \
    --partition v1 \
    --cases-from configs/cases/smoke_5cases.yaml \
    --max-iters 5000 \
    --seed 0 \
    --sub-trajectory-length 64 \
    --rollout-horizon 16 \
    --diagnostic-every 250 \
    --output-dir outputs/runs/session6/run_f_l \
    --tag-suffix run_f_l_seed0_L64
```

Hypothesis: L=32 covers only 1.6 t/c, which is too short for impact
dynamics (5 to 20 t/c). With L=64 the encoder sees more pre-impact and
more post-impact, so dynamic information becomes more important to
prediction and the trivial shortcut is less attractive.

Required code change: `--sub-trajectory-length` and `--rollout-horizon`
flags must be wired through to the encoder constructor and the JEPA
wrapper. If they are not yet, add them. The encoder's positional
embeddings need to accommodate L=64 (currently sized for L=32).

#### F-CD: c-dropout

```bash
python -m src.training.train_jepa \
    --partition v1 \
    --cases-from configs/cases/smoke_5cases.yaml \
    --max-iters 5000 \
    --seed 0 \
    --c-dropout-prob 0.5 \
    --diagnostic-every 250 \
    --output-dir outputs/runs/session6/run_f_cd \
    --tag-suffix run_f_cd_seed0_p0p5
```

Hypothesis: with c-dropout, during training half of all batches see
c=0 at the predictor. The predictor cannot rely on c being available
and must learn to forecast from z alone. The encoder is then under
pressure to encode dynamics into z.

Required code change: a `c_dropout_prob` parameter in the JEPA wrapper.
Each forward pass, with probability `c_dropout_prob`, replace `c` with
zeros before passing to the predictor. The encoder is unchanged (it
never sees c). Implementation is roughly 5 lines in `src/models/jepa.py`.

Note: this is inspired by classifier-free guidance in diffusion (Ho,
Salimans, arXiv:2207.12598), but applied here as an anti-shortcut
regulariser rather than as a guidance mechanism.

#### F-NC: no c at predictor

```bash
python -m src.training.train_jepa \
    --partition v1 \
    --cases-from configs/cases/smoke_5cases.yaml \
    --max-iters 5000 \
    --seed 0 \
    --predictor-cond-dim 0 \
    --diagnostic-every 250 \
    --output-dir outputs/runs/session6/run_f_nc \
    --tag-suffix run_f_nc_seed0_cond0
```

Hypothesis (the most diagnostic single change): the trivial pattern is
rooted in D6 (c enters at predictor, encoder is unconditional). If c is
removed entirely from the predictor, the encoder MUST encode c into z
for the predictor to forecast c-dependent dynamics. This is the strongest
test of whether the c-conditioning architecture is the root cause.

Required code change: `cond_dim=0` already exists as a valid value in
the predictor signature (Session 3); just pass it through. AdaLN-Zero
with cond_dim=0 reduces to identity-on-residual permanently, which is
not what we want; the predictor must instead skip AdaLN entirely when
cond_dim=0 and rely only on the residual stream. This is a small code
change in `AutoregressivePredictor.__init__`.

#### F-S: scale up to 24 cases

```bash
python -m src.training.train_jepa \
    --partition v1 \
    --cases-from configs/cases/smoke_24cases.yaml \
    --max-iters 5000 \
    --seed 0 \
    --diagnostic-every 250 \
    --output-dir outputs/runs/session6/run_f_s \
    --tag-suffix run_f_s_seed0_24cases
```

Hypothesis: with 24 cases instead of 5, the encoder has more distinct
c values to memorize. Memorizing becomes less attractive relative to
learning physics. PR_within_case should rise; if it does, the failure is
data-scale-bound.

Required artifact: `configs/cases/smoke_24cases.yaml`, a deliberate
selection of 24 cases (24 = roughly two-thirds of the 37 train cases)
that spans G from -3 to +3 with several D and Y values. The agent
should build this YAML by inspection of the partition manifest.

Alternative: use ALL 37 train cases (the F-S-full variant). I do not
recommend this for the first run because (a) wall-clock doubles
(37*4=148 train encounters vs 24*4=96), and (b) if a 24-case run lands
healthy, the marginal information from going to 37 is small. If 24
lands TRIVIAL, then running F-S-full at 37 is the natural follow-up.

#### F-OBS: observable head

```bash
python -m src.training.train_jepa \
    --partition v1 \
    --cases-from configs/cases/smoke_5cases.yaml \
    --max-iters 5000 \
    --seed 0 \
    --observable-head cl_future \
    --observable-head-weight 0.01 \
    --observable-head-deltas 8 16 24 \
    --diagnostic-every 250 \
    --output-dir outputs/runs/session6/run_f_obs \
    --tag-suffix run_f_obs_seed0_eta0p01
```

Hypothesis: the pure self-supervised JEPA loss is insufficient to
prevent collapse on low-intrinsic-dim physics data. A small auxiliary
supervised signal (CL at +8, +16, +24 frames) pressures the encoder to
retain dynamic information and breaks the case-memorization shortcut.

Required code change: a new module `src/models/observable_head.py`
implementing a small MLP `nn.Linear(d=32, hidden=64) -> GELU ->
nn.Linear(hidden=64, 3)` mapping z_t to three future CL values. The
JEPA wrapper composes this in the loss as

```python
L_obs = mse_loss(head_cl(z), cl_future)  # cl_future: (B, T, 3)
L_total = L_pred + 0.5 * L_roll + lambda_sigreg * L_anticollapse + eta * L_obs
```

The eta = 0.01 weight is small enough that the JEPA self-supervised
character is preserved; the head provides weak guidance rather than
direct supervision. Inspired by Fukami's β=auxiliary-weight in the
lift-augmented AE (Fukami and Taira 2023, JFM 2023; observable
augmentation in Fukami, Nakao, Taira, JFM 2024).

Implementation cost: ~50 lines for the module, ~10 lines in the JEPA
wrapper, ~30 lines in three new unit tests
(`tests/test_observable_head.py`).

### What about combinations?

Combinations (L=64 AND c-dropout=0.5, L=64 AND observable head, etc.)
are out of scope for Session 6. Session 7 will run combinations of the
axes that Step 4 identifies as active. This deliberately limits the
search space.

Total Session 6 GPU cost: 5 runs × 25 min = ~125 minutes plus the 24-case
run (estimated 35 min due to larger dataset) = ~140 minutes total.

### Code changes summary

|File|Change|Lines|
|---|---|---|
|`src/data/episode_dataset.py`|Emit cl_future (Step 1)|~40|
|`src/data/preprocessing.py` (or equivalent)|Cache cl.h5 alongside omega_z.h5|~30|
|`src/training/train_jepa.py`|Add 5 new CLI flags|~25|
|`src/models/predictor.py`|Handle cond_dim=0 cleanly|~5|
|`src/models/jepa.py`|Add c-dropout, observable head wiring|~20|
|`src/models/observable_head.py`|New module|~50|
|`tests/test_episode_dataset.py`|cl_future tests|~30|
|`tests/test_observable_head.py`|New test file|~80|
|`configs/cases/smoke_24cases.yaml`|New, hand-curated|~30|

Total: roughly 310 lines of new code across 9 files. Modest scope.

## Step 4: factorial analysis notebook

`notebooks/03_factorial_analysis.ipynb` has the following structure.

### Section 1: load all six runs

Six W&B runs: Session 5 Run A (the SIGReg baseline TRIVIAL result) plus
the five new Session 6 runs. Pull full history. Confirm all runs have
the same step indices and the same diagnostic cadence.

### Section 2: side-by-side metric table at iter 5000

|Run|PR_all|PR_within_case|r2(z→c)|r2(z_dyn→c)|r2(z_dyn→phase)|r2(z→CL_future)|
|---|---|---|---|---|---|---|
|Run A (Session 5 baseline)|...|...|...|...|...|N/A|
|F-L (L=64)|...|...|...|...|...|...|
|F-CD (c-dropout=0.5)|...|...|...|...|...|...|
|F-NC (no c at predictor)|...|...|...|...|...|...|
|F-S (24 cases)|...|...|...|...|...|...|
|F-OBS (observable head)|...|...|...|...|...|...|

The Session 5 Run A row's last column (r2(z→CL_future)) requires
re-evaluating Run A's checkpoint with the new CL data loader. This is a
one-shot evaluation, not a retraining; the checkpoint is unchanged.
Adds ~5 min to the analysis.

### Section 3: per-axis interpretation

For each axis, print one of:

- **active**: the axis materially improves at least two of
  (PR_within_case > 4, r2(z_dyn → phase) > 0.5, r2(z → CL_future) >
  r2_baseline(c, t → CL_future)). The baseline (c, t → CL_future) is
  trained as a tiny MLP in this notebook section.
- **partially_active**: improves one of the three metrics but not two.
- **inactive**: no material improvement over Run A.
- **regressed**: worse than Run A on PR_within_case (the axis hurt).

### Section 4: lift-prediction baseline comparison

This is the central skill metric per the collaborator's report. Train
two tiny MLPs:

```
baseline_ct: MLP(c, frame_index) -> CL(t+Δ)   # for Δ in {8, 16, 24}
baseline_jepa: MLP(z_t)             -> CL(t+Δ)
```

For each variant's final checkpoint, compute r²(z → CL_future) and
compare to r²(c, t → CL_future). If the JEPA is materially better,
the latent has learned useful aerodynamic state. If it is not, the
latent is at best a fancy lookup of c.

### Section 5: decision string

Print one of:

```
Session 6 outcome: <one of>

  CLEAN_ROOT_CAUSE      - exactly one axis is "active" and that axis,
                          applied alone, produces a healthy latent. Most
                          parsimonious explanation. Session 7: lambda
                          bisection on that single-axis configuration.
                          (Examples: F-NC alone fixes it -> root cause was
                          c-conditioning architecture.)

  COMBINED_REMEDIATION  - no single axis fixes it but two or three are
                          "partially_active". Session 7: factorial
                          combinations of the partial axes.
                          (Example: F-L and F-CD both partially help ->
                          combine them.)

  OBSERVABLE_REQUIRED   - F-OBS is the only "active" axis. Pure self-
                          supervised JEPA is insufficient on this data;
                          observable augmentation is necessary. Session
                          7 reframes as observable-augmented JEPA, with
                          paper framing matching the Fukami / Solera-
                          Rico precedent. This is publishable.

  SCALE_BOUND           - F-S is the only "active" axis. Pure JEPA works
                          but only with enough cases. Session 7:
                          full-train-cases scale run, then lambda
                          bisection.

  ALL_AXES_FAIL         - no run is "active". Structural failure not
                          attributable to any single tested cause.
                          Session 7: pivot to either dynamic-IDM PLDM
                          (with CL as the IDM target, per D34) or to
                          the Solera-Rico β-VAE comparison run, framed
                          as a careful negative result.

  AMBIGUOUS             - two or more axes look fully active. Session
                          7: replicate the apparently-active axes with
                          a different seed and run combinations to
                          disambiguate.
```

### Section 6: numerical sanity report

A bullet list of "things that look suspicious in the data" for the
agent's report. Examples to check:

- L_pred not monotone decreasing (training instability).
- F-OBS's L_obs increasing while L_pred decreases (head is sacrificing
  CL prediction for self-supervised performance; head weight may be
  too low).
- F-NC's L_pred much higher than Run A's (predictor has lost
  capacity without c).
- Probe R^2 oscillating (eval batch too small; rerun with larger eval
  batch).

## Pass criteria for Session 6 as a session

Four conditions, ALL of which must be met:

1. Two new run3 cases absorbed; partition is v1.2 with new sha256; D33
   recorded; all data-pipeline unit tests pass.
2. CL is in the data pipeline; `cl_future: (B, T, 3)` is in the batch
   contract; the two new unit tests pass.
3. The audit notebook produces side-by-side metrics for Run A and Run C
   that quantitatively characterize the TRIVIAL and SPREAD_TRIVIAL
   patterns.
4. All five factorial runs complete 5000 iterations with finite losses
   and clean W&B uploads. The analysis notebook produces a decision
   string from the menu above.

The numerical pass criteria of Session 5 (PR > 16, probe R² in 0.5 to
0.7) are NOT pass criteria for Session 6. Session 6 is a diagnostic
session; the deliverable is the decision string, not a particular
numerical outcome.

## Pre-registered methodological note

This session deliberately tests five axes that were not all part of the
original architectural specification. The original spec committed to
SIGReg+BN+L=32+c-at-predictor as the recipe. Session 5 showed this
recipe lands in trivial pattern at the 5-case scale. Session 6 tests
which axis of that recipe is the rate-limiting one.

Each of the five axes corresponds to a published precedent:

- F-L: longer sub-trajectories are standard in video-based JEPA (V-JEPA
  2 uses 64-frame windows at 4 fps; V-JEPA 2-AC at arXiv:2506.09985).
- F-CD: classifier-free guidance via condition dropout (Ho, Salimans,
  arXiv:2207.12598) and BERT-style masking precedents.
- F-NC: matches Brain-JEPA and Echo-JEPA where the encoder is fully
  responsible for encoding all subject-level information without
  separate conditioning.
- F-S: standard data-scale ablation in self-supervised learning.
- F-OBS: observable augmentation in fluid mechanics (Fukami and Taira
  2023, JFM; Fukami, Nakao, Taira 2024, JFM; Fukami 2025 transonic
  buffet, JFM).

The session is therefore not an ad-hoc rescue attempt; it is a
controlled factorial test of methodological choices each of which is
defensible in isolation.

## Out of scope for Session 6

- **Hydra configs**. Argparse continues to be sufficient. Defer to a
  small dedicated session before Session 8 (the lambda bisection
  session).
- **torch.compile()**. Defer to the same session as Hydra.
- **Dynamic-IDM PLDM**. The CL-as-IDM design is locked (D34) but the
  PLDM training pipeline is built only if Session 6 lands ALL_AXES_FAIL
  or AMBIGUOUS. Session 5.PLDM (if triggered) replaces the static-c
  IDM with the dynamic-CL IDM per D34.
- **Symmetry augmentation** (Open Q6 in HANDOFF). Stays parked.
- **Frame-skip ablation** (Open Q2). Stays parked.
- **The other three baselines** (POD, Fukami AE, Solera-Rico β-VAE).
  POD and the Solera-Rico β-VAE are now elevated to "implement before
  paper submission, in parallel with the main JEPA path." They do not
  block Session 6 but they should be drafted as separate sessions
  parallel to Session 7. Specifically: a dedicated Solera-Rico β-VAE
  comparator session, using the existing KTH-FlowAI code at
  github.com/KTH-FlowAI/beta-Variational-autoencoders-and-transformers
  -for-reduced-order-modelling-of-fluid-flows (Carlos's group's own
  repo), adapted to the gust dataset.
- **80k full training**. Absolutely not until Session 7 or Session 8.
  The Session 6 result rules out lambda bisection on a failed
  configuration.

## Expected duration

- Step 0 (absorb 2 new cases): 30 to 60 minutes including unit-test
  reruns and the manifest sha256 update.
- Step 1 (CL data pipeline): 90 to 150 minutes. The bulk is in the
  preprocessing script (caching cl.h5) and the dataset class change.
- Step 2 (audit notebook): 60 to 90 minutes for the existing-checkpoint
  audit. Pure analysis, no training.
- Step 3 (five factorial runs): 140 minutes of GPU plus ~30 minutes of
  setup. The runs can be launched sequentially (no parallel GPU on the
  single RTX 6000); the agent supervises them but does not need to be
  actively coding during the runs.
- Step 4 (factorial analysis notebook): 60 to 90 minutes.
- HANDOFF entries (D33, D34, D35, D36, D37) plus session report: 45
  minutes.

Total realistic span: 6 to 8 hours, similar to Session 5. Plan for one
focused day.

## D-entries to record

**D33** (Step 0, always): Two new run3 cases absorbed into v1.2.
Specific case IDs and counts filled in by the agent.

**D34** (Step 1, always): CL(t+Δ) is the canonical dynamic IDM and
observable-head target. Replaces "time-to-impact" and "vortex centroid"
from the collaborator report. Rationale: CL is aerodynamically
meaningful, aligns with Fukami's lift-augmented AE precedent (JFM 2023,
2024, 2025), aligns with Solera-Rico β-VAE (Nat Commun 2024).
Implementation: `head_CL: z_t -> CL(t+Δ)` for Δ in {8, 16, 24} frames
(0.8, 1.6, 2.4 t/c).

**D35** (Step 3, always): observable head added as auxiliary loss with
weight η = 0.01. Justification: small η preserves JEPA self-supervised
character; observable provides weak dynamics guidance. To be tuned in
Session 7 if F-OBS lands active.

**D36** (Step 3, always): five factorial single-axis variants, each
tests one hypothesis about the trivial-pattern root cause. Combinations
deferred to Session 7.

**D37** (Step 4, conditional on outcome): the decision string. Records
which axis is active, what Session 7 looks like.

## What this session enables for the paper

Two specific outcomes shift the paper's framing materially:

**If F-OBS is active (OBSERVABLE_REQUIRED branch)**: the paper becomes
"observable-augmented JEPA succeeds where pure self-supervision fails
on low-intrinsic-dim fluid data; we propose the within-case probe and
the SPREAD_TRIVIAL diagnostic as additions to the JEPA evaluation
suite." This connects directly to Fukami's published work and Solera-
Rico's own group's prior work. It is a defensible, novel, replicable
methodological contribution.

**If F-NC is active (CLEAN_ROOT_CAUSE branch with c-conditioning as
cause)**: the paper becomes "static-descriptor conditioning of JEPA
predictors creates a memorization shortcut that bypasses representation
learning; removing predictor conditioning fixes it." This is a more
narrowly architectural finding but it has implications for other JEPA
work that uses static labels (Brain-JEPA, Echo-JEPA), and it is
directly testable in those domains.

**If F-S is active (SCALE_BOUND)**: standard finding, less novel, but
still publishable as "JEPA on small fluid datasets requires X training
cases to avoid trivial collapse." Less methodological impact.

**If ALL_AXES_FAIL**: the paper's framing pivots from "JEPA-for-science
succeeds with X adaptations" to "we honestly tested JEPA on low-
intrinsic-dim physics and found it does not work even with five
defensible adaptations; β-VAE and observable-augmented AE remain
superior at this scale." Still publishable; the negative result is
clean because of the factorial design.

## After Session 6 lands

Per the decision string, Session 7 is one of:

- **Session 7-CRC** (CLEAN_ROOT_CAUSE): lambda bisection on the
  single-axis-fixed configuration. Roughly the original Session 6 plan
  but rooted in evidence. ~6 hours.
- **Session 7-COMB** (COMBINED_REMEDIATION): factorial combinations of
  partial axes. Run 4 to 6 combination runs at 5k iters, identify the
  best, then lambda bisection. ~8 hours.
- **Session 7-OBS** (OBSERVABLE_REQUIRED): refine the observable head
  (sweep η, sweep Δ, sweep the head architecture), then lambda
  bisection on the augmented JEPA. ~10 hours.
- **Session 7-SCALE** (SCALE_BOUND): full-train-cases scale run (37
  cases), then lambda bisection. ~8 hours.
- **Session 7-PIVOT** (ALL_AXES_FAIL): dynamic-IDM PLDM with CL target
  (Session 5.PLDM rewritten per D32 and D34), and parallel Solera-Rico
  β-VAE comparator session. ~10 hours of paper restructuring on top.

## Decision references (carried forward)

- D2 (HANDOFF): d = 32 per LeWM. Still locked.
- D5 (HANDOFF): SIGReg with auto-fallback to VICReg.
- D6 (HANDOFF): encoder is unconditional, c enters only the predictor.
  **F-NC tests this directly.**
- D8 (HANDOFF, corrected in D32): PLDM as comparator, with dynamic-CL
  IDM per D34.
- D14, D15, D20 (HANDOFF): partition v1 absorption history.
- D17 (HANDOFF): BatchNorm projection per LeWM; F-L, F-CD, F-NC, F-S,
  F-OBS all keep BatchNorm.
- D19 (HANDOFF): RTX 6000 Blackwell only.
- D21 (HANDOFF): V-JEPA 2-AC-faithful scheduled sampling with H_roll=8.
  F-L changes H_roll to 16 to maintain the same ratio with L=64.
- D22 (HANDOFF): VICReg coefficients. Not exercised in Session 6
  (all runs use SIGReg).
- D24-D28 (HANDOFF, Session 5): variant flags and case subset.
- D29 (HANDOFF): PLDM-on-TRIVIAL was the original priority. Session 6
  overrides this with factorial diagnostics first; Session 5.PLDM is
  deferred to Session 7-PIVOT if and only if Session 6 lands
  ALL_AXES_FAIL.
- D32 (HANDOFF, Session 5): PLDM citation correction.
- D33-D37 (this session): see above.
