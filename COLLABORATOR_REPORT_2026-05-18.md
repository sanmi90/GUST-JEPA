# vortex-jepa Pipeline Report (2026-05-18)

Collaborator-facing snapshot of the project state. Self-contained; reading
this should not require browsing the HANDOFF or session reports. For the
authoritative decision history with rationales and alternatives considered,
see `HANDOFF.md` (decisions D1 through D33).

Lead: Carlos Sanmiguel Vila (INTA, UC3M).

## 1. Executive Summary

We are building an end-to-end Joint-Embedding Predictive Architecture
(JEPA) for parametric vortex-gust airfoil interactions at Re = 5000.
Architectural template: LeWM (Maes et al., arXiv:2603.19312, March 2026).
Anti-collapse template: LeJEPA / SIGReg (Balestriero and LeCun,
arXiv:2511.08544, November 2025). Training-recipe template: V-JEPA 2-AC
(Assran et al., arXiv:2506.09985, 2025). Direct baselines for the paper:
POD, Fukami AE (Phys. Rev. Fluids 10, 084703, 2025), Solera-Rico beta-VAE
plus transformer (Nat. Commun. 15, 1361, 2024), and PLDM (Sobal, Zhang,
Cho, Balestriero, Rudner, LeCun, arXiv:2502.14819, February 2025).

State as of 2026-05-18:

- Data pipeline complete. Partition v1 contains 49 cases (21 periodic
  plus 28 run3) and 238 encounters in cache. Split locked at
  `configs/splits/split_v1.json` with 39 train cases (132 encounters),
  6 Test B cases (28 encounters), and 4 Test C cases (24 encounters).
  Test A (impact-instant generalization) is 54 held-out encounters drawn
  from the same cases as train.
- Model primitives complete and unit-tested. Encoder, predictor, SIGReg,
  VICReg, AdaLN-Zero, RoPE, JEPA wrapper. Test suite: 97 passing in
  ~3.5 minutes (CPU-friendly with bf16 paths skipped if no RTX 6000).
  One slow integration test gated behind `pytest --runslow`.
- Training scaffold complete. `src/training/train_jepa.py` for SIGReg and
  VICReg JEPA; `src/training/train_baseline.py` for PLDM (and stubs for
  Fukami, Solera-Rico, POD).
- Five 5k-iteration smoke runs landed on a 5-case subset. None achieved
  the healthy regime (PR > 16 and 0.5 < probe R-squared < 0.7). All
  five variants exhibit the "encoder collapses to the static episode
  descriptor c" failure mode in different forms. Methodological reading:
  H4 (the LeWM Two-Room failure mode replicates on physics data) is
  confirmed at the 5-case data scale and survives both the 2-term
  (SIGReg, VICReg) and the 5-term (PLDM) anti-collapse families.
- Next session is Session 5.5: expand the smoke subset to 10 to 12 cases
  and re-run SIGReg-BN and PLDM, with the methodological question being
  "is the failure data-scale-bound (transition at higher case count) or
  structural (plateau even at 10-12 cases)?".

## 2. Scientific Goal

The paper targets three claims:

1. End-to-end JEPA-style self-supervised representation learning is
   viable on fluid mechanics data with low intrinsic dimensionality
   (estimated 5 to 10 for our manifold).
2. The latent forecast horizon and probing R-squared at matched latent
   dimension d match or beat Fukami et al. (PRF 2025) and Solera-Rico
   et al. (Nat. Commun. 2024).
3. We contribute the participation-ratio-based SIGReg failure diagnostic
   as a reusable methodology for JEPA-for-science. Per the LeWM
   Two-Room precedent (LeWM Section 5: PLDM and DINO-WM outperform LeWM
   when the intrinsic dimension is low), the framing is regime-dependent:
   SIGReg as the default plus PLDM as the recommended fallback for
   low-intrinsic-dim domains.

## 3. Data Pipeline

### 3.1. DNS source

The raw direct numerical simulation (DNS) data lives in an external
project (Carlos's PREVENT effort on ML turbulence detection that
produced these runs). vortex-jepa accesses it by path, not by copy:

- Set `PREVENT_ROOT` to the PREVENT root before any pipeline step.
  Default on the workstation: `$HOME/PREVENT`.
- Raw files: `${PREVENT_ROOT}/data/raw/periodic/` and
  `${PREVENT_ROOT}/data/raw/periodic/run3/`.
- Cache (built by `scripts/preprocess.py`):
  `${PREVENT_ROOT}/data/processed/vortex-jepa/v1/{case_id}/encounter_{kk}.h5`.

The DNS is a NACA 0012 airfoil at angle of attack alpha = 14 degrees,
Reynolds number Re = 5000. The flow is perturbed by Taylor vortices
parametrized by gust strength G, gust diameter D, and crosswise offset
Y/c. The two source groups are:

- periodic (800-frame runs, 6 encounters of 120 frames each plus an 80-frame
  trailing remainder that is discarded by the loader);
- run3 (480-frame runs, 4 encounters of 120 frames each, no trailing
  remainder).

Each encounter is one "gust event": gust release at frame 0, impact at
frame ~40 (vortex centroid crosses leading edge at t ~ 1.965 t/c), and
post-impact dynamics through frame ~80. Time step is dt_tc = 0.05 in
chord-convective units. The full impact window for sampler purposes is
frames [25, 55].

The HDF5 schema (verified in `outputs/schema_inspection/schema.yaml`)
holds velocity at `/u` with shape `(T, 192, 96, 32, 3)` and component order
`(u_x, u_y, u_z)`; vorticity at `/curlU` with the same spatial layout and
component order matching the velocity vorticity components (omega_z is
index 2, sign convention `du/dy - dv/dx`, opposite of the standard
right-hand rule); wall pressure at `/sensors/p` shape `(1536, T)` which
unfolds as `(192 surface points) x (8 z-stations)` (the inner axis is z,
so spanwise averaging is `reshape((192, 8, T)).mean(axis=1)`); and forces
at `/forces/CL` and `/forces/CD` already non-dimensionalized
(`CL = 2 * lift exact`). Pitching moment is not stored; integrate over
`/airfoil_xy` if needed. The `/u` and `/curlU` arrays carry NaN in the
2624 cells where `/inside_solid > 0`; the cache fills these with 0.

### 3.2. Filename to (G, D, Y) decode

Each run3 filename encodes the gust parameters via the form
`Gust_NNN_x{X}_y{Y}_s{S}_d{D}.h5`. Example: `Gust_001_x-1.965_y-0.387_s1.0_d0.5.h5`
decodes to `(G = +1.0, D = 0.5, Y = +0.10)` under the alpha = 14 degree
rotation locked in `data_manifest/raw_cases_inventory.yaml`
(`parser.formula_inverse`). The Baseline case (no gust, G = D = Y = 0)
has filename `Baseline.h5`.

`scripts/100c_raw_cases_inventory.py` re-parses the raw directory and
emits the YAML manifest. When a collaborator drops a new file (this has
happened five times during the project), the workflow is: rerun the
inventory script, copy the YAML into vortex-jepa's `data_manifest/`,
rerun `scripts/preprocess.py` to extract any new encounter cache files,
rerun `python build_split_manifest.py` to refresh
`configs/splits/split_v1.json`, and append a D-entry to HANDOFF.md (this
is the pattern named D12, D14, D15, D20, and D33).

### 3.3. Cache format

Each encounter is stored as one HDF5 file with the following fields:

- `omega_z`: `(120, 192, 96)` float32. Mid-plane spanwise vorticity
  extracted from `/curlU[..., 16, 2]` (z-index 16 is the mid-span plane;
  component index 2 is omega_z). NaN-filled cells from `/inside_solid`
  are replaced with 0.
- `p_wall`: `(120, 192)` float32. Spanwise-averaged wall pressure.
- `C_L`, `C_D`: `(120,)` float32 each. Force coefficients.
- 17 attributes: `case_id`, `G`, `D`, `Y`, `source_group`,
  `encounter_index`, `frame_start`, `frame_end`, `dt_tc`,
  `impact_frame_estimate`, `mid_span_index`, `omega_z_sign_convention`,
  `preprocessing_version`, `partition_version`, `raw_relative_path`,
  `n_frames`.

The partition is frozen at creation. To introduce schema or parameter
changes the right move is to bump `preprocessing_version` or
`partition_version`, not to edit v1 in place. The exception (D12, D14,
D15, D20, D33) is absorption: adding new cases to v1 is allowed while v1
has no paper-reportable training checkpoint. Once a v1 checkpoint
ships, the next absorption MUST go to v2.

Per the most recent absorption (D33), partition v1 holds 49 cases /
238 encounters; new inventory SHA256 is
`dd984588be553a28285a35fed7328cfcf9b482329e6f346b4f1e9a0574f764bc` and
new split SHA256 is
`7f8f60428e13b7c2fe4063e15bd99ea9e08e5e6cecf0e8883f8fb6a4875e2331`.

### 3.4. Split design

The split is locked at `configs/splits/split_v1.json` with hashes
recorded above. Counts:

- **train**: 39 cases, 132 encounters (first 4 of 6 periodic, first 3 of
  4 run3). The no-gust Baseline case is included in train (encounters
  0 to 3); the metadata flags it `is_calibration_reference: true` so
  calibration tools can find it, but it participates in training like
  any other case (D9).
- **Test A** (impact-instant generalization, contiguous holdout):
  54 encounters drawn from the same 39 cases as train (last 2 of 6 for
  periodic cases, last 1 of 4 for run3 cases). Tests whether the model
  generalizes across shedding phases within seen (G, D, Y).
- **Test B** (parametric interpolation): 6 interior cases pooled across
  source groups, 28 encounters. Tests whether the model interpolates in
  (G, D, Y) within the training envelope.
- **Test C** (extrapolation, |G| = 4 only): 4 cases, 24 encounters.
  Reserved for end-of-paper reporting only. Never used for model
  selection.

The extrapolation axis is asymmetric: |G| = 3 stays in training; only
|G| = 4 is held out. Periodic trailing partials are discarded. The
impact frame estimate of 40 was validated on the cached partition v1
(vorticity-domain argmax mean = 40.8, force-domain argmax mean = 38.8,
both over [25, 55]).

Sub-trajectory sampling: length L = 32 frames with a two-branch start
mixture. With probability `impact_aware_fraction = 0.7` the start is
uniform on `impact_overlap_start_range = [8, 40]` (this guarantees
the sub-trajectory intersection with the [25, 55] impact window contains
at least 7 frames). Otherwise the start is uniform on the full episode.
The observed impact-overlap fraction on the cached partition is 0.814,
matching the predicted 0.811.

## 4. Cases

### 4.1. Inventory snapshot

Partition v1 (post-D33) holds 49 cases:

- 21 periodic (Baseline plus 20 gust cases with |G| in {0.25, 0.5, 1.0,
  2.0, 4.0}; |G| = 4 cases are in Test C).
- 28 run3 (no |G| = 0; cases at |G| in {0.5, 1.0, 1.5, 2.0, 3.0} with
  D in {0.5, 1.0, 1.5} and Y in {-0.4, -0.2, -0.1, 0, 0.1, 0.2, 0.4}).

Periodic does not have any |G| = 3 case and does not have any D = 1.5
case; both regimes live only in run3.

### 4.2. The smoke 5-case subset (D24)

For Session 5 (the methodological smoke), we pinned a 5-case subset to
stress-test the training loop. The subset was chosen to span the G axis
from -3 to +3, cover all four D values (0, 0.5, 1.0, 1.5), exercise both
signs of Y/c, and include both source groups. Stored at
`configs/cases/smoke_5cases.yaml`:

```yaml
cases:
  - Baseline                   # periodic, G=0,  D=0,   Y=0
  - G+3.00_D0.50_Y+0.40        # run3,    G=+3, D=0.5, Y=+0.4
  - G-3.00_D1.00_Y-0.20        # run3,    G=-3, D=1.0, Y=-0.2
  - G+1.00_D1.50_Y+0.20        # run3,    G=+1, D=1.5, Y=+0.2
  - G+1.00_D1.00_Y-0.20        # run3,    G=+1, D=1.0, Y=-0.2
```

Train encounter count: 4 + 3 + 3 + 3 + 3 = 16 sub-trajectories. With
sub-trajectory length 32 and the impact-aware sampler, this is small;
the methodological question Session 5 asks is whether SIGReg works at
this scale.

The Session 5 plan originally named two periodic ids
(`G+3.00_D0.50_Y+0.20` and `G+1.00_D1.50_Y+0.10`) that do not exist in
the manifest (periodic has no |G| = 3 and no D = 1.5). The closest
manifest cases were substituted; both happen to be run3, so the source
group balance shifted from the planned 4 periodic + 1 run3 to the actual
1 periodic + 4 run3. The G / D / Y coverage was preserved.

### 4.3. Full-data training plan

Session 5 is the smoke; Session 6 (after Session 5.5 produces a healthy
variant) uses the full train split: 132 train encounters across 39
cases. The model is trained for 80,000 iterations with lambda chosen by
bisection over [0.001, 1.0] (six evaluations of 24k iterations each).

## 5. Algorithm Design

### 5.1. Encoder (Hybrid CNN + ViT)

The encoder is a hybrid CNN stem followed by a small Vision Transformer.
Module: `src/models/encoder.py:HybridCNNViTEncoder`.

Input shape: `(B, T, 1, 192, 96)`. The single channel is the mid-plane
spanwise vorticity omega_z. T defaults to 32.

CNN stem (~3M parameters):

- Stage 0: Conv2d 1 to 64, kernel 7, stride 2, padding 3, then GroupNorm
  (8 groups) and GELU. Downsamples to `(64, 96, 48)`.
- Stage 1: two Conv2d-GroupNorm-GELU blocks at 64 channels.
- Down1: Conv2d 64 to 128, stride 2. Downsamples to `(128, 48, 24)`.
- Stage 2: two Conv2d-GroupNorm-GELU blocks at 128 channels.
- Down2: Conv2d 128 to 256, stride 2. Downsamples to `(256, 24, 12)`.
- Stage 3: two Conv2d-GroupNorm-GELU blocks at 256 channels.

The result is a `(B*T, 256, 24, 12)` feature map. Flatten to 288 spatial
tokens of channel dim 256.

ViT body (~7M parameters):

- A learnable [CLS] token is prepended (`(B*T, 289, 256)`).
- A deterministic 2D sin-cos positional embedding (no learned positions)
  is added.
- 6 pre-norm transformer blocks at hidden dim 256, 8 heads, MLP ratio 4,
  dropout 0.0 (no dropout in the default encoder).

Projection head (the LeWM-specific piece):

- Final LayerNorm on the token sequence.
- Read [CLS] only.
- `nn.Linear(256, d=32)`.
- `nn.BatchNorm1d(32)` (D17). This is the key LeWM choice: the
  projection norm at the encoder bottleneck must be BatchNorm because
  the preceding ViT LayerNorm would otherwise prevent the anti-collapse
  objective from being optimized cleanly (LeWM Section 3.1).

The Session 5 Run B variant exercised a `projection_norm = "layernorm"`
alternative to test whether the BatchNorm choice is responsible for
SIGReg's failure (D25); the smoke result is that LayerNorm degrades the
probe rather than recovering PR.

Total encoder parameters: ~6.7M. Conditioning: NONE. The encoder is
unconditional by design (D6). The static episode descriptor c = (G, D, Y)
enters only the predictor.

### 5.2. Predictor (AdaLN-Zero transformer with RoPE)

The predictor is a 6-layer causal autoregressive transformer with
DiT-style AdaLN-Zero conditioning on c and 1D rotary position embeddings
(RoPE) on the temporal axis. Module:
`src/models/predictor.py:AutoregressivePredictor`.

- Input: `z` of shape `(B, T, d=32)`, `cond` of shape `(B, 3)`.
- Embed: `Linear(32, 384)`.
- Conditioning MLP: 2-layer MLP `Linear(3, 384) -> SiLU -> Linear(384, 384)`
  produces a `(B, 384)` conditioning vector; the predictor blocks broadcast
  this across t.
- 6 blocks, each a DiT block: LayerNorm with `elementwise_affine=False`
  followed by AdaLN-Zero modulation `(shift, scale, gate)`, multi-head
  causal self-attention with RoPE on Q and K only (not on V, per
  RoFormer Section 3.4), then a second AdaLN-Zero and MLP. Hidden 384,
  16 heads (head dim 24), MLP ratio 4, dropout 0.1.
- Output projection: `Linear(384, 32) -> BatchNorm1d(32)`. BatchNorm
  matches the encoder so predicted and target latents live in the same
  distribution (D17).

The AdaLN-Zero init (D11 in DiT) sets the final linear in each AdaLN
to zero, which makes the predictor identity-on-residual at iter 0. The
predictor learns dynamics by moving the AdaLN gates off zero during
training. Sanity check 2 (Session 5) verifies that one Adam step moves
at least one AdaLN linear weight off zero.

Total predictor parameters: ~14M (heavier than the encoder by design).
Conditioning at default: static c = (G, D, Y), no time-varying phase
phi_t (D16). The architecture supports cond_dim = 4 with a one-line
change if Session 6 needs to add phi_t.

### 5.3. SIGReg (LeWM appendix-A Epps-Pulley statistic)

`src/models/sigreg.py:SIGReg`. The LeWM anti-collapse objective. Computes
the Epps-Pulley statistic at M = 256 random projections of the latent
batch, integrated on 17 knots over t in [0.2, 4]. Returns a non-negative
scalar that goes to zero on isotropic Gaussian latents and grows on
non-Gaussian distributions (collapsed, low-rank, heavy-tailed).

Three implementation notes:

- The body runs in fp32 even under bf16 autocast (the characteristic
  function involves complex exponentials whose differences are not
  well-bounded under bf16). D13.
- No leading N multiplier on the statistic. The LeWM appendix-A
  definition omits it; the official LeJEPA PyTorch listing includes it.
  The LeWM convention is the more authoritative source for this project;
  D13 records the convention choice and the empirically calibrated unit
  test thresholds (Gaussian < 0.01, Student-t df=2 > 0.05, Uniform(-1, 1)
  > 0.02 at B = 4096).
- Knot range [0.2, 4] not [-5, 5]. The half-axis choice is harmless
  (the integrand is symmetric in t, and the integrand at t in [0, 0.2)
  is negligible).

### 5.4. VICReg (Bardes et al. ICLR 2022)

`src/models/vicreg.py:VICReg`. The anti-collapse fallback. Per D22, the
default uses `mu = 25.0, lambda_ = 25.0, nu = 1.0, gamma = 1.0`. The
invariance term parameterized by `lambda_` is dropped: it requires a
paired second view of each sample, which JEPA without paired
augmentations does not have, and the H-JEPA reference plus PLDM
precedent both drop it. The default forward computes
`mu * L_var + nu * L_cov` only. The full four-argument constructor is
kept for forward-compatibility.

The variance hinge target is the per-dimension standard deviation
(`sqrt(var + eps)`), not the variance itself (Bardes et al. equation 1).
The `eps = 1e-4` default prevents infinite gradients when a latent dim
approaches zero variance.

### 5.5. PLDM 5-term loss (verified 2026-05-18 against arXiv:2502.14819)

`src/baselines/pldm.py:PLDMLoss`. The PLDM anti-collapse objective from
Sobal et al. arXiv:2502.14819. Per D30 the paper actually has FIVE
terms, not seven as the D8 reading from project bootstrap incorrectly
claimed; the equations were verified verbatim from Appendix D.1.1:

```
L_JEPA = L_sim
       + alpha * L_var
       + beta  * L_cov
       + delta * L_time_sim
       + omega * L_IDM
```

- `L_sim` is the multi-step rollout MSE between the predictor's
  open-loop rollout from a seed frame and the ground-truth encoder
  latents over the horizon H (Section 3.3, Equation 3).
- `L_var` is the VICReg variance hinge applied per-time-slice then
  averaged over (time, dim).
- `L_cov` is the off-diagonal covariance Frobenius norm per-time-slice
  then averaged over time.
- `L_time_sim` is the temporal smoothness MSE ||z_t - z_{t+1}||^2
  averaged across (B, T-1, d).
- `L_IDM` is the inverse-dynamics regression: a small MLP predicts
  the per-step action from (z_t, z_{t+1}). In our setup we adapt to
  the static episode descriptor c (the JEPA has no per-step action);
  the IDM MLP predicts c from each consecutive (z_t, z_{t+1}) pair,
  broadcast across the T-1 pairs per batch sample.

Four tunable weights (alpha, beta, delta, omega) plus L_sim with
implicit weight 1. The IDM MLP is `(2*d) -> 128 -> 128 -> c_dim`.

Paper hyperparameter values per Appendix J.2:

| Environment       | alpha | beta | delta | omega |
|-------------------|-------|------|-------|-------|
| Two-Rooms         |  4.0  |  6.9 |  0.75 | 0.0   |
| Diverse PointMaze | 35.0  | 12.0 |  0.1  | 5.4   |
| Ant-U-Maze        | 26.2  |  0.5 |  8.1  | 0.58  |

None matches our regime cleanly; the Session 5.PLDM smoke used
all-1.0 placeholders.

### 5.6. Training recipe (V-JEPA 2-AC-faithful scheduled sampling)

The JEPA total loss is

```
L_total = L_pred + 0.5 * L_roll + lambda * L_anticollapse
```

where `L_pred` is the teacher-forced one-step MSE on `z` over the full
T - 1 = 31 positions of the sub-trajectory, `L_roll` is the open-loop
rollout MSE over H_roll = 8 steps from one random start position per
forward pass, and `L_anticollapse` is SIGReg by default with a runtime
VICReg fallback (D5).

Two transpositions from the V-JEPA 2-AC original (Assran et al.,
arXiv:2506.09985, Section 6):

- We use T - 1 = 31 teacher-forced positions because we have access to
  the full sub-trajectory. V-JEPA 2-AC uses 15 because its architecture
  exposes 16 frame slots at a time.
- We use H_roll = 8 because the vortex impact dynamics last 5 to 20
  t/c (40 to 160 effective frames at dt_eff = 0.1). V-JEPA 2-AC uses
  H_roll = 2 which is too short for this domain.

The two-loss sum is the simplest faithful translation of the LeWM
`L_pred + lambda * L_sigreg` extended with rollout from V-JEPA 2-AC.
Bengio probabilistic mixing was rejected because it adds a
hyperparameter axis (the teacher-forcing probability schedule) with no
published precedent for JEPA-style models.

Rollout start strategies (D21): `fixed_zero` for unit tests,
`uniform_random` for training, `impact_aware` reserved for Session 5+
ablation.

The PLDM wrapper uses a different composition. PLDM's `L_sim` is a
multi-step rollout that subsumes both `L_pred` and `L_roll`. The
PLDMWrapper rolls out from `z[:, :1, :]` for H = 8 steps and computes
MSE against ground-truth `z[:, :H+1, :]`. There is no separate
teacher-forced loss; the regularisation terms (var, cov, time-sim, idm)
are applied to the full encoded sequence z, not to z_hat.

### 5.7. Diagnostics

`src/training/diagnostics.py` defines three pure functions called every
`diagnostic_every` iterations on a held-out Test B sub-batch:

- `participation_ratio(z_batch)`: PR = `(sum_i s_i)^2 / sum_i s_i^2`
  over singular values of the batch latent matrix. Equals d for
  isotropic latents, 1 for rank-1 collapse.
- `linear_probe_r2(z, c, fit_indices, eval_indices)`: closed-form
  linear least-squares fit on the fit indices and R-squared on the eval
  indices. Returns per-component R-squared (r2_G, r2_D, r2_Y) plus
  r2_overall (unweighted mean).
- `per_dim_variance_histogram(z_batch)`: histogram of per-dimension
  variances over the batch.

Auto-fallback controller (D5, `src/training/auto_fallback.py`):

```
if iteration >= 20000 AND PR < 0.3 * d AND r2_overall < 0.7:
    swap SIGReg -> VICReg (idempotent state machine)
    log event prominently to W&B and stdout
    continue training, do not restart
```

The conjunctive design catches the worst case (latent collapsed AND
useless). It does NOT catch the trivial-solution mode where PR is
collapsed but r2 is high; that pathology was observed in Run A at the
smoke scale (D27) and the rule may need revision before Session 6 (D28
proposes three concrete options).

## 6. Training Infrastructure

### 6.1. Hardware contract (D19)

All training, smoke-test, and benchmark runs MUST use the RTX 6000
Blackwell (sm_120) GPU. The workstation also exposes two NVIDIA L40S
(sm_89) cards; those must NOT be used so paper compute is on a single
named accelerator class. Silent CPU fallback is also forbidden.

Enforcement: `src/utils/device.py:require_rtx6000()` walks
`torch.cuda.device_count()`, picks the first device whose name contains
both "RTX" and "6000", runs a tiny probe kernel
(`torch.zeros(4, device=d) + 1`) to confirm the installed PyTorch wheel
ships kernels for sm_120, and returns a `torch.device` or raises
`NoRTX6000Error` with a clear message that lists what torch DID see and
the suggested reinstall command. Training entrypoints call this at
startup; tests that genuinely exercise CUDA paths call it and
`pytest.skip` if it raises.

PyTorch was upgraded from 2.1.2+cu121 (sm_50..sm_90 only, silently fell
back to L40S on Blackwell) to 2.12.0+cu130 on 2026-05-17. The cu130
wheels on the default PyPI index ship kernels for sm_120 and pass the
probe.

### 6.2. W&B logging contract

W&B is the primary logger. Four REQUIRED keys logged on every run:

- `preprocessing_version` (from `configs/preprocessing.yaml`)
- `partition_version` (e.g. "v1")
- `lambda_sigreg` (anti-collapse weight; null for PLDM runs which use
  the four PLDM lambdas instead)
- `seed` (full deterministic seed)

Additional keys for paper-grade runs:

- `split_sha256`, `inventory_sha256`, `code_sha256` (git commit hash)
- `auto_fallback_triggered`, `gpu_name`, `wandb_run_id`

W&B run group: `partition_v1`. W&B tags: `["hybrid_cnn_vit", regularizer_name]`
with optional `"run:<tag_suffix>"` to disaggregate Session 5 variants.

Offline mode (the Session 5 default) writes each run to
`outputs/runs/.../wandb/offline-run-*/` plus a side JSONL log at
`outputs/runs/.../metrics.jsonl`. The JSONL is self-contained (one JSON
event per W&B log call) and is what the analysis notebook reads; W&B
offline can be synced with `wandb sync` after `wandb login`.

### 6.3. Sanity checks (Session 5, D24-D26)

Before any 5k-iter variant run, `src/training/sanity_checks.py --all`
runs five gates in under 5 minutes:

1. Projection BatchNorm running stats are healthy after warmup (finite,
   `|mean| < 10`, `var` in `(1e-4, 100)`).
2. The predictor is identity-on-residual at init (`L_pred` in
   `[0.01, 10]`), at least one AdaLN linear weight moves off zero after
   one Adam step, and L_pred decreases over a few overfitting steps.
3. `SIGReg(z).backward()` produces a finite, nonzero gradient on the
   projection BatchNorm bias.
4. `predictor.rollout(steps=1)` at index 1 equals the teacher-forced
   prediction at index 0 within atol 1e-4.
5. The data loader emits `omega.shape == (16, 32, 1, 192, 96)`,
   `c.shape == (16, 3)`, omega is finite, and `|omega|.max() < 10000`.

The original `|omega| < 200` bound from the Session 5 plan was wrong;
DNS vorticity at Re = 5000 peaks at ~4000 in vortex cores. A survey
across all 49 cases gave median `|omega|.max()` = 1482 (the strongest
case, G+4.00_D0.50_Y-0.10 from Test C, hits 4377 at frame 52). The
10,000 ceiling catches Inf without false-failing legitimate intense
vorticity events.

## 7. Smoke Results (Session 5 plus Session 5.PLDM)

### 7.1. Five-variant comparison

All five variants were run on the same 5-case smoke subset (D24), with
seed 0, 5000 iterations, the same hybrid CNN+ViT encoder, the same
AdaLN-Zero predictor, the same data loader, and the same W&B
diagnostic cadence (every 250 iterations on a Test B sub-batch). Only
the anti-collapse loss and the encoder projection norm vary.

Final state at iter 5000:

| Variant            | Anti-collapse     | Proj | PR     | r2_overall | r2_G  | r2_D  | r2_Y  | Quadrant      |
|--------------------|-------------------|------|--------|------------|-------|-------|-------|---------------|
| A: SIGReg + BN     | 2-term LeWM       | BN   |  1.025 | 0.779      | 0.923 | 0.775 | 0.637 | TRIVIAL       |
| B: SIGReg + LN     | 2-term LeWM       | LN   |  1.135 | 0.452      | 0.645 | 0.419 | 0.293 | DEAD          |
| C: VICReg + BN     | 2-term VICReg     | BN   | 17.463 | 0.887      | 0.914 | 0.889 | 0.858 | TRIVIAL_LITE  |
| D: VICReg + LN     | 2-term VICReg     | LN   |  7.588 | 0.803      | 0.929 | 0.784 | 0.696 | TRIVIAL       |
| PLDM-A             | 5-term VICReg+IDM | BN   |  5.966 | 0.970      | 0.986 | 0.970 | 0.953 | TRIVIAL       |

Quadrant definitions:

- HEALTHY: PR > 0.5 * d = 16 AND 0.5 < r2_overall < 0.7. The encoder
  is anti-collapsed AND learns something useful about c without
  memorising it. **No variant reached this quadrant.**
- TRIVIAL: PR <= 16 AND r2_overall > 0.5. The encoder collapses to a
  low-rank representation of c (rank ~1 in Run A; nearly constant per
  case in PLDM-A).
- TRIVIAL_LITE (a quadrant not strictly named in the Session 5 plan):
  PR > 16 AND r2_overall > 0.7. The encoder anti-collapses but spreads
  c-correlated noise across many dimensions.
- WEAK: PR > 16 AND r2_overall <= 0.5. Anti-collapsed but not capturing c.
- DEAD: PR <= 16 AND r2_overall <= 0.5. Collapsed and uninformative.

### 7.2. Loss decomposition signatures

The variants share one signature: L_pred (or L_sim for PLDM) drops to
near zero by iter 100 (16 train sub-trajectories is small enough that
the predictor overfits the dynamics trivially regardless of regulariser).
The anti-collapse loss decompositions tell the story of HOW each variant
collapsed:

- **Run A (SIGReg+BN)**: SIGReg loss converges to ~0.08, near its
  Gaussian-on-unit-variance asymptote. The latent is rank 1 in d-dim
  space, which makes the SIGReg projections degenerate; the regulariser
  is satisfied because each projected scalar is Gaussian after the
  BatchNorm normalisation. The encoder is f(c) plus tiny noise.
- **Run B (SIGReg+LN)**: SIGReg loss settles around 0.12; the per-sample
  LayerNorm at the projection makes the latent direction-per-sample-dependent
  rather than collapsed-to-a-line, but the latent is still rank 1
  in the across-batch sense. The probe r2 oscillates violently across
  iterations (range -0.86 to +0.86) because the LayerNorm-normalised
  representations are nearly orthogonal per sample, which destroys the
  linear-probe fit.
- **Run C (VICReg+BN)**: L_var converges to ~0 (variance hinge is
  satisfied per-dim), L_cov stays around 0.08. The variance hinge forces
  every dimension to have variance >= 1, so PR climbs to 17.5. But the
  encoder satisfies this by spreading c-correlated noise across all 32
  dimensions; the linear probe still extracts c with r2 = 0.89.
- **Run D (VICReg+LN)**: L_var stays at ~4 throughout (the per-sample
  LayerNorm prevents the per-dim variance hinge from converging
  cleanly). PR climbs partway (to ~8) but not to the healthy threshold.
  r2 stays at ~0.80.
- **PLDM-A (5-term)**: L_sim ~ 0.014, L_var ~ 0.51, L_cov ~ 0.10,
  **L_time_sim ~ 0.002, L_idm ~ 0.0005**. The PLDM signature is that
  L_time_sim and L_idm both go to zero simultaneously, which means the
  encoder produces almost-constant latents over time AND the IDM head
  decodes c from any (z_t, z_{t+1}) pair near-perfectly. The IDM term
  PRESSURES this rather than preventing it: the easiest way to make c
  decodable from any consecutive pair is to make z constant per case.

### 7.3. Decision string

Session 5 (notebook algorithmic): `MIXED: quadrants ['TRIVIAL', 'DEAD',
'TRIVIAL_LITE', 'TRIVIAL']; manual inspection required.`

Session 5 (methodological, D27): TRIVIAL-DOMINANT. 3 of 4 variants
land at r2_overall > 0.7, which is the "encoder leaks c" failure
signature predicted by hypothesis H4 (the LeWM Two-Room failure mode
extended to physics data). The plan's strict TRIVIAL definition required
"all variants land in PR <= 16 AND r2 > 0.7", and Run C broke that by
clearing PR; the spirit of TRIVIAL was preserved.

Session 5.PLDM (D31): `DATA_SCALE_BOUND: PLDM-A PR=5.97 (<= 16); both
regularisers collapse on 5 cases. r2=0.970. -> Session 5.5 (expand to
10-12 cases) on BOTH SIGReg and PLDM.`

## 8. Main Problems

### 8.1. Data scale insufficiency

With 5 cases and ~3 train encounters per case (~16 sub-trajectories
total at L = 32) the encoder has insufficient data to learn anything
beyond the static episode descriptor c. The only consistent local
minimum is z = f(c) plus noise, in some form.

Five regularisation families now confirm this:

- SIGReg (Gaussian prior, 2-term loss): collapses to rank 1, encodes
  c on one axis.
- SIGReg + LayerNorm (per-sample normalisation at projection): collapses
  to rank 1 AND destroys the probe.
- VICReg (per-dim variance hinge, 2-term loss): forces dim spread but
  fills the extra dims with c-correlated noise.
- VICReg + LayerNorm: partial dim spread, still leaks c.
- PLDM (5-term, variance + covariance + temporal smoothness + IDM):
  ALL terms can be near-zero simultaneously when z is constant per case;
  the IDM term actively pressures this.

The methodological finding survives the bootstrap session's 200-iter
smoke (which already showed PR=1.12, r2=0.71 on 11 train samples) and
generalises across the 2-term and 5-term regulariser families. **Session
5.5 must answer whether expanding to 10-12 cases produces a transition
to HEALTHY** or whether the failure persists structurally.

### 8.2. The trivial-solution failure mode is below the auto-fallback radar

The auto-fallback rule (D5) fires on `PR < 0.3 * d AND r2_overall < 0.7`.
In Run A (PR=1.025, r2=0.779) the conjunct is NOT satisfied because r2
is above 0.7; the rule decides "the encoder is collapsed BUT the probe
still works, so don't fall back yet". That decision is correct under
the rule's design intent (catch the worst case: latent both collapsed
AND uninformative). But the trivial-solution mode (latent collapsed AND
probe high because the encoder memorised c) is exactly what we want to
catch in physics data with low intrinsic dim, and the rule misses it.

D28 records three concrete revisions to consider before Session 6:

(a) Drop the r2 conjunct entirely (fire on PR alone). Catches the
trivial-solution mode but false-fires on slow-spreading variants
during the early training transient.

(b) Switch the probe to a CASE-conditional split (fit on K Test B
cases, evaluate on the other 6-K). The trivial-solution mode should
drop r2 sharply when the encoder has only memorised seen c values.
This is the most principled and operationalises the original
"memorisation vs generalisation" intent of the rule. Cost: higher
variance on the small Test B set (6 cases total).

(c) Add an overfitting indicator: fire on
`PR < 0.3 * d AND L_pred_running < 1e-3`. Run A's L_pred is below
1e-3 by iter 100; this signature is unambiguous. Cost: another
threshold to tune; requires running-average bookkeeping.

Decision deferred to Session 6 start.

### 8.3. The PLDM 5-term vs 7-term correction (D30)

This is a process problem, not a science problem, but it changes how
the paper's PLDM claim should be written. The original D8 (from project
bootstrap) read the PLDM loss as a "7-term VICReg-derived objective"
with six tunable weights. Direct verification of arXiv:2502.14819
Appendix D.1.1 on 2026-05-18 (using the arxiv MCP plugin to download
the paper text and grep-verifying the LaTeX equations at chars 18700-
19800 and 75130-77100) shows that the paper has FIVE terms:

```
L_JEPA = L_sim + alpha * L_var + beta * L_cov + delta * L_time-sim + omega * L_IDM
```

Four tunable weights (alpha, beta, delta, omega). D8's "term 5" (var
on temporal differences dz) and "term 6" (cov on dz) were spurious;
the paper does NOT regularise dz, only z. D30 records the full
correction and updates HANDOFF and CLAUDE.md. The
SESSION5_PLDM_BASELINE.md plan was written under the original D8
misreading and stays as a historical record; D30 supersedes.

Why this matters for the paper: the central methodological contrast in
contribution claim 3 (D8) was "SIGReg + 2-term (proposed) vs VICReg +
7-term (PLDM)". The corrected framing is "SIGReg + 2-term vs PLDM +
5-term". The hyperparameter complexity comparison changes from
O(log n) bisection over 1 weight vs O(n^6) grid search over 6 weights
to O(log n) over 1 vs grid search over 4. The methodological story is
unchanged: SIGReg is simpler; PLDM's IDM term is the key additional
piece for low-intrinsic-dim regimes per LeWM Section 5; Session 5.PLDM
confirms PLDM also fails at 5 cases.

### 8.4. The predictor BatchNorm vs LayerNorm-encoder mismatch

When the encoder is set to `projection_norm = "layernorm"` (Run B,
Run D), the encoder's per-sample-normalised z is compared against the
predictor's per-feature-normalised z_hat (the predictor's `out_proj`
is BatchNorm in all variants). The L_pred MSE still computes but
measures something less clean. The Session 5 plan was explicit:
"`--projection-norm` passes through to the encoder constructor"
(only). The predictor `out_proj` BatchNorm is left in place.

If Session 6 confirms LayerNorm-encoder variants are interesting, the
predictor `out_proj` should also be made configurable and the two
norms should be matched.

## 9. Open Questions

Inherited from earlier sessions and refreshed for the collaborator:

1. **Frame-skip resolution.** Default frame-skip 2 (dt_eff = 0.1).
   Frame-skip 1 (no skipping) is viable on the 96 GB GPU and provides
   2x more frames per sub-trajectory. Verify against impact dynamics
   resolution as part of Session 6.
2. **Lambda bisection budget.** Six evaluations over [0.001, 1.0] for
   SIGReg-JEPA. If the optimum is near LeWM's default 0.1, stop early
   and log this as a robustness result.
3. **Auxiliary observable head.** Should the JEPA optionally produce
   wall pressure or C_L as a side prediction? Default is no (per LeWM).
   Reserve as an ablation. If it substantially helps probe r2, it is a
   reportable hybrid contribution.
4. **C-JEPA-style gust masking ablation.** Requires defining the "gust
   object" region per episode. The vortex centroid is computable
   analytically from launch position plus U_inf * t. A circular mask of
   radius D around the centroid would zero out the gust in selected
   frames. Optional ablation.
5. **Symmetry augmentation.** The flow has approximate
   Y -> -Y, G -> -G, omega_z -> -omega_z symmetry. Adding this as a
   paired augmentation roughly doubles the effective training data.
   May rescue a TRIVIAL outcome at small data scale. Implement and
   ablate to verify it does not destabilize SIGReg.

New from Session 5.PLDM:

6. **Stricter IDM variant.** Our static-c IDM (predicts c from any
   z-pair) is too easy to satisfy and may PRESSURE collapse rather
   than prevent it. A stricter IDM that predicts a time-varying
   quantity (e.g., vortex centroid position, computable analytically
   from frame index plus c) would carry a real temporal signal.
7. **The predictor architectural difference.** PLDM uses a single-step
   GRU / Conv / MLP; we use a causal transformer. Per the plan, this
   is the second-order ablation; first answer the data-scale question.

## 10. Next Steps

In order:

1. **Session 5.5** (~3-5 hours). Expand the smoke subset to 10-12
   cases (the candidate list is in `configs/cases/smoke_5cases.yaml`
   plus 5-7 more spanning the rest of the (G, D, Y) cube). Re-run
   SIGReg + BN and PLDM-A. The PR / r2 outcome at the new case count
   answers the data-scale-bound vs structural question.

2. If Session 5.5 produces a HEALTHY variant: **Session 6** runs Hydra
   configs plus `torch.compile()` plus the lambda bisection (six
   evaluations of 24k iterations each on the full train split). The
   winning variant produces the first paper-reportable v1 checkpoint;
   from that moment on, the partition-immutability rule (D5)
   bites and the next absorption goes to v2.

3. If Session 5.5 plateaus: the failure is structural, not
   data-scale-bound. Possible interventions in order of cost:
   symmetry augmentation (Open Q5), phi_t conditioning (D16
   alternative), stricter IDM (Open Q6, new), frame-skip 1 (Open Q1),
   auxiliary observable head (Open Q3). The first to recover a healthy
   variant becomes the canonical recipe.

4. **Full 80k training** of the chosen lambda. Train the visualization
   decoder on the frozen encoder. Run the full Section-7 evaluation
   suite (forecasting horizon vs Fukami AE and Solera-Rico, probe
   R-squared vs both, decoded reconstruction MSE, qualitative latent
   exploration).

5. **Baselines in parallel.** POD, Fukami AE, Solera-Rico beta-VAE.
   PLDM is already implemented and lands as the priority comparator
   under D29. Each baseline takes 30 min to 4 hours of GPU time on
   the full train split.

6. **Ablation matrix.** The 15 ablations from the architecture spec.
   Mandatory: d sweep, SIGReg vs VICReg vs none, teacher forcing vs
   scheduled sampling vs full rollout, the four baselines.

7. **Paper.** Once a healthy variant exists and the ablation matrix
   is complete.

## 11. Code Map

```
vortex-jepa/
├── CLAUDE.md                        # Operational guide for Claude Code sessions
├── HANDOFF.md                       # Decision history (D1-D33)
├── README.md
├── requirements.txt
├── build_split_manifest.py          # Regenerates configs/splits/split_v1.json
│
├── configs/
│   ├── preprocessing.yaml           # Cache schema (v1.0.0)
│   ├── cases/
│   │   └── smoke_5cases.yaml        # Session 5 smoke subset (D24)
│   └── splits/
│       └── split_v1.json            # Locked split manifest (SHA in D33)
│
├── data_manifest/
│   └── raw_cases_inventory.yaml     # PREVENT-side inventory snapshot
│
├── scripts/
│   ├── 100c_raw_cases_inventory.py  # Filename to (G, D, Y) parser
│   ├── inspect_raw_hdf5.py          # Step 0 schema inspector
│   ├── preprocess.py                # Per-encounter cache extractor
│   └── run_smoke_5k_variants.sh     # Sequential variant launcher
│
├── src/
│   ├── baselines/
│   │   ├── __init__.py
│   │   └── pldm.py                  # PLDMLoss 5-term (D30)
│   ├── data/
│   │   └── episode_dataset.py       # Impact-aware sub-trajectory sampler
│   ├── models/
│   │   ├── adaln.py                 # AdaLN-Zero (DiT)
│   │   ├── encoder.py               # HybridCNNViTEncoder (D17)
│   │   ├── jepa.py                  # JEPA wrapper (L_pred + 0.5*L_roll + lambda*L_anti)
│   │   ├── pldm_wrapper.py          # PLDMWrapper (D30)
│   │   ├── predictor.py             # AutoregressivePredictor (AdaLN+RoPE)
│   │   ├── rope.py                  # 1D temporal RoPE
│   │   ├── sigreg.py                # LeWM Appendix-A Epps-Pulley (D13)
│   │   └── vicreg.py                # Bardes et al. variance + covariance (D22)
│   ├── training/
│   │   ├── auto_fallback.py         # SIGReg -> VICReg state machine (D5)
│   │   ├── diagnostics.py           # PR + linear probe R-squared + var hist
│   │   ├── sanity_checks.py         # Session 5 pre-variant gates (D24)
│   │   ├── scheduled_sampling.py    # V-JEPA 2-AC two-loss recipe (D21)
│   │   ├── train_baseline.py        # PLDM training entrypoint
│   │   └── train_jepa.py            # JEPA training entrypoint
│   └── utils/
│       └── device.py                # require_rtx6000() (D19)
│
├── tests/                           # 97 passing, 1 slow (--runslow)
│   ├── conftest.py                  # --runslow CLI option (D23)
│   ├── test_adaln_zero.py
│   ├── test_auto_fallback.py
│   ├── test_device.py
│   ├── test_diagnostics.py
│   ├── test_encoder.py
│   ├── test_jepa.py
│   ├── test_pldm_loss.py            # PLDM 5-term tests (D30)
│   ├── test_pldm_wrapper.py
│   ├── test_predictor.py
│   ├── test_rope.py
│   ├── test_sanity_checks.py
│   ├── test_scheduled_sampling.py
│   ├── test_sigreg.py
│   ├── test_train_jepa_smoke.py     # Slow, opt-in via --runslow
│   └── test_vicreg.py
│
├── notebooks/
│   ├── 00_qc_partition_v1.ipynb     # Cache integrity QC
│   └── 01_smoke_5k_analysis.ipynb   # Session 5 + 5.PLDM analysis (Sections 1-7)
│
├── SESSION_DATA_PREP.md             # Preprocessing plan + Step 0 status
├── SESSION_REPORT_2026-05-15.md     # Bootstrap session
├── SESSION_REPORT_2026-05-16.md     # Session 2 (model primitives)
├── SESSION_REPORT_2026-05-17.md     # Session 4 (training scaffold)
├── SESSION_REPORT_2026-05-18.md     # Session 5 (5k smoke + variants)
├── SESSION_REPORT_2026-05-18_session5pldm.md  # Session 5.PLDM
├── SESSION2_MODEL_PRIMITIVES.md     # Session 2 plan
├── SESSION5_MEANINGFUL_SMOKE_5K.md  # Session 5 plan
├── SESSION5_PLDM_BASELINE.md        # Session 5.PLDM plan (under D8 misreading; D30 supersedes)
└── COLLABORATOR_REPORT_2026-05-18.md  # This document
```

## 12. References

Direct architectural template:

- LeWM: Maes, Le Lidec, Scieur, LeCun, Balestriero. "LeWorldModel: Stable
  End-to-End Joint-Embedding Predictive Architecture from Pixels."
  arXiv:2603.19312, March 2026.

Anti-collapse theory:

- LeJEPA / SIGReg: Balestriero and LeCun. "LeJEPA: Provable and
  Scalable Self-Supervised Learning Without the Heuristics."
  arXiv:2511.08544, November 2025.
- VICReg: Bardes, Ponce, LeCun. ICLR 2022.

Direct baselines:

- PLDM (primary, verified): Sobal, Zhang, Cho, Balestriero, Rudner,
  LeCun. "Learning from Reward-Free Offline Data: A Case for Planning
  with Latent Dynamics Models." arXiv:2502.14819, February 2025.
  Project page: latent-planning.github.io. Code:
  github.com/vladisai/PLDM.
- PLDM workshop precursor: Sobal, Jyothir, Jalagam, Carion, Cho,
  LeCun. "Joint Embedding Predictive Architectures Focus on Slow
  Features." arXiv:2211.10831, NeurIPS SSL workshop 2022.
- Solera-Rico, Sanmiguel Vila, Gomez-Lopez, Wang, Almashjary, Dawson,
  Vinuesa. "beta-Variational Autoencoders and Transformers for
  Reduced-Order Modelling of Fluid Flows." Nat. Commun. 15, 1361, 2024.
- Fukami, Iwatani, Maejima, Asada, Kawai. "Compact Representation of
  Transonic Airfoil Buffet Flows with Observable-Augmented Machine
  Learning." J. Fluid Mech. 1021, A39, 2025 (arXiv:2509.17306).
- Fukami, Smith, Taira. "Extreme Vortex-Gust Airfoil Interactions at
  Reynolds Number 5000." Phys. Rev. Fluids 10, 084703, 2025.

Related JEPA work:

- V-JEPA 2 / V-JEPA 2-AC: Assran et al. arXiv:2506.09985, 2025.
  Multi-step training recipe with scheduled sampling.
- C-JEPA: Nam, Le Lidec, Maes, LeCun, Balestriero. arXiv:2602.11389,
  February 2026. Object-centric masking.

Latent dynamics on manifolds:

- Constante-Amores and Graham. "Data-Driven State-Space and Koopman
  Operator Models of Coherent State Dynamics on Invariant Manifolds."
  J. Fluid Mech. 984, R9, 2024 (arXiv:2312.03875).

## 13. How to Re-Run Anything

Environment setup:

```bash
export PREVENT_ROOT=$HOME/PREVENT
export WANDB_PROJECT=vortex-jepa
source .venv/bin/activate
```

Verify the pipeline is healthy:

```bash
pytest tests/                                # 97 passed, 1 skipped, ~3.5 min
python -m src.training.sanity_checks --all --require-gpu  # 5/5 PASS, <5 min
```

Re-run the Session 5 + 5.PLDM smoke (offline W&B, ~5 hours total GPU
time across 5 variants):

```bash
bash scripts/run_smoke_5k_variants.sh 0      # Runs A, B, C, D conditionally

python -m src.training.train_baseline \      # Run PLDM-A
    --baseline pldm \
    --partition v1 \
    --cases-from configs/cases/smoke_5cases.yaml \
    --max-iters 5000 --seed 0 \
    --diagnostic-every 250 --checkpoint-every 1000 --log-every 25 \
    --output-dir outputs/runs/smoke5k/run_pldm_a \
    --wandb-mode offline --tag-suffix run_pldm_a_seed0
```

Open the analysis:

```bash
jupyter notebook notebooks/01_smoke_5k_analysis.ipynb
```

Absorb a new run3 file (D12 / D14 / D15 / D20 / D33 pattern):

```bash
# 1. Confirm the file is at $PREVENT_ROOT/data/raw/periodic/run3/Gust_NNN_*.h5
# 2. Regenerate the inventory:
PYTHONPATH=src python scripts/100c_raw_cases_inventory.py
# 3. Copy to vortex-jepa side:
cp $PREVENT_ROOT/data_manifest/raw_cases_inventory.yaml data_manifest/
# 4. Extract the new encounter cache (skips existing):
python scripts/preprocess.py --partition v1
# 5. Regenerate the split manifest:
python build_split_manifest.py
# 6. Append a D-entry to HANDOFF.md.
```

For any non-trivial change to a model module, run the slow integration
test before committing:

```bash
pytest tests/ --runslow   # 98 passed, ~3.5 min
```
