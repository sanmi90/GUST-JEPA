# Session 18 Experiment B1: baseline comparison fairness protocol

Date: 2026-05-28
Lead: Carlos Sanmiguel Vila (INTA, UC3M)
Hardware: 2x RTX 6000 Blackwell (sm_120), bf16 mixed precision
Target: produce a 7-row by 4-column physical Markov closure comparison table
where the only varying quantity is the latent-extractor (JEPA, Fukami AE,
POD), the latent dimension d, and the test split. All other knobs are
locked at the values listed below.

The paper's headline claim depends on this. Any reviewer who finds that a
baseline was disadvantaged by an unequal training recipe can reject the
comparison. The locked recipe below is the single source of truth.

## Data and split

Partition: `v1` (60 cases). Manifest `configs/splits/split_v1.json`
sha256 anchored. Train/test_a/test_b/test_c assignment is whatever the
manifest already encodes. The Baseline case sits in train + test_a per
the project memory.

Preprocessing: `outputs/data_pipeline/v1/manifest.json` (canonical
three-stage pipeline: spatial mask of 140 inside-solid + 1-cell-adjacent
cells, per-encounter p99.99 clip with 282 thresholds in [52, 178],
3-sigma scale by train_std = 3.5526; mean-preserving, no shift). All
baselines see exactly this normalised omega_z field. Cache directory
`${VORTEX_JEPA_CACHE}/v1/{case_id}/encounter_{k:02d}.h5`.

Resolution: native `(192, 96)` mid-plane vorticity (no upsampling, no
downsampling). All baselines work in the same pixel space.

Frame sampling: identical impact-aware sampler used by the JEPA training
loop (70% impact-aware, 30% uniform, sub-trajectory length L = 32).

Observable head targets: per-baseline canonical lift target. The
JEPA production checkpoint and the Fukami AE retrains both predict
the CURRENT-FRAME C_L (delta = 0); see "Loss configurations and the
documented asymmetry" section below for the per-baseline objective.

## Loss configurations and the documented asymmetry

The three baselines (JEPA, Fukami AE, POD) use DIFFERENT TRAINING LOSSES
because each method has a canonical recipe that defines what the method
IS. Forcing them onto a single loss would mean "this is no longer the
JEPA / Fukami AE we are claiming to compare against." The fairness rule
is therefore:

1. Each baseline uses its OWN published / project-locked loss recipe.
2. All baselines see the SAME data, the SAME ω-pipeline preprocessing,
   the SAME train/test_a/test_b/test_c split, the SAME impact-aware
   sampler.
3. The DOWNSTREAM comparison is on a COMMON transformer predictor
   (identical architecture and recipe per "Common transformer
   predictor" section).
4. Every term in every loss is documented below with its motivating
   HANDOFF.md entry. Reviewers can audit term-by-term.

### JEPA d=64 production loss

Total objective at training time, with weights pulled from
`outputs/runs/session12/S12_E_d64/encoder/checkpoint_iter020000.pt`:

```
L_JEPA = L_pred                                              (weight 1.0; teacher-forced 1-step in latent space)
       + 0.5  * L_roll                                       (open-loop rollout, H_roll = 8; V-JEPA 2-AC recipe)
       + 0.01 * SIGReg(z)                                    (anti-collapse on the BatchNorm-projected latent)
       + 0.01 * MSE(C_L(t), C_L_hat(t))                      (lift-augmentation head, current-frame; delta = 0)
       + 1.0  * SmoothL1(wake_target_t, wake_hat_t)          (patch_signed_spectrum 80-D wake observable head)
```

Justifications, per HANDOFF.md entry:

- L_pred + 0.5 * L_roll: CLAUDE.md "Locked decisions, Training"; the
  V-JEPA 2-AC recipe transposed to our setting per
  `src/training/scheduled_sampling.py` docstring. Teacher-forced one-step
  loss over T-1 = 31 positions of the sub-trajectory plus an H_roll = 8
  open-loop rollout with fixed coefficient 0.5.

- 0.01 * SIGReg(z): D5 (SIGReg with auto-fallback to VICReg, locked
  Session 2), D13 (LeWM Appendix A formulation, no N multiplier),
  D17 (BatchNorm projection required by SIGReg). The lambda_sigreg
  value is the locked production weight; the auto-fallback to VICReg
  did NOT fire on the S12_E_d64 run.

- 0.01 * MSE(C_L(t), C_L_hat(t)) at delta = 0: D37 (Session 6,
  observable head added as auxiliary loss with weight eta = 0.01).
  Quote: "with eta = 0.01 ... the observable term contributes about
  a percent of the total loss at convergence -- the head is a weak
  guidance signal, not a primary supervision target." Delta = 0
  (current-frame C_L) matches Fukami's published recipe.

- 1.0 * SmoothL1(wake_target_t, wake_hat_t) on patch_signed_spectrum:
  D81-D84 (Session 11 Track 1 W0_C_lam100, the winning JEPA + decoder
  configuration on this flow). Lambda_wake = 1.0 was reached by
  extending the original Session 11 lambda ladder beyond its max of
  0.30 after the W0_C_lam30 result showed visible wake reconstruction
  for the first time. The patch_signed_spectrum mode produces an 80-D
  target by sampling signed-vorticity patches on a coarse wake grid.

### Fukami AE training loss (paper-faithful)

Reference: Fukami and Taira, "Grasping Extreme Aerodynamics on a
Low-Dimensional Manifold," arXiv:2305.08024 (Phys. Rev. Fluids 10,
084703, 2025), equation 3. Verified via arxiv MCP during Session 18.

```
L_Fukami = ||q - q_hat||_2^2                                  (L2 vorticity-field reconstruction)
         + beta * ||C_L - C_L_hat||_2^2                       (lift-augmentation; beta = 0.05 per L-curve analysis)
```

with beta = 0.05 and the lift head outputting a single scalar C_L at
the same frame as the input vorticity (Table S.1, "Output 2 (Lift
coefficient)", Data size (1)). No anti-collapse term, no rollout term,
no wake observable head in the original Fukami AE.

Identical knobs across d = 3, 32, 64 (only `--d` varies):

- ω-pipeline preprocessing (matches JEPA).
- recon_loss_type = mse (matches the L2 reconstruction term in eqn 3).
- lambda_recon = 1.0, lambda_lift = 0.05.
- observable_head_deltas = [0] (single-scalar current C_L).
- ReLU + GroupNorm + bf16 instead of strict-paper tanh + no GroupNorm
  + fp32. The strict-paper variant gives Test B probe delta of -0.45
  on this Re=5000 flow (CLAUDE.md "Things to NOT do"); the activation
  + norm + precision adaptation preserves Fukami's objective while
  making training numerically stable on RTX 6000 Blackwell.

### POD training loss (none)

POD is closed-form. Snapshot SVD on the same ω-pipeline-normalised
train frames produces the d-rank truncated basis. No training loss
exists; nothing to be "fair" to.

### The asymmetries that survive

After enforcing same data + same preprocessing + same split, three
loss-side asymmetries remain:

1. JEPA has L_pred + 0.5 * L_roll. Fukami AE has none. POD has none.
   Each method's objective IS its identity; matching them would
   require constructing a new method that isn't the published one.

2. JEPA has 0.01 * SIGReg(z). Fukami AE has none (its latent
   regularisation comes from the bottleneck dimension and lift loss).
   POD has none (the basis is orthonormal by construction).

3. JEPA has the wake observable head (Session 11 W0_C_lam100
   W0_C_lam100, lambda_wake = 1.0). Fukami AE does not (the original
   paper does not have this; the Session 11 attempt to add it to
   Fukami AE in D86 broke training and was abandoned). POD does not.

These asymmetries are inherent to the method comparison the paper
is making. Each baseline gets ITS CANONICAL CONFIGURATION. The
methods appendix tabulates all weights with HANDOFF.md citations.

The CONSEQUENCE: the comparison is not "which loss is better at the
same objective"; it is "which low-dimensional representation, trained
under its method-defining recipe, produces the best downstream
physical-Markov closure when the same transformer predictor is
trained on top." This is the cleanest scientifically meaningful
comparison given the methods themselves are structurally different.

The downstream transformer-predictor stage IS strictly uniform (see
"Common transformer predictor" section): same architecture, same
optimiser, same iter budget, only the input latent_dim varies. This
is where the fairness gates apply.

## Fukami AE (B1 Part a)

Implementation: `src/baselines/fukami_ae.py` `FukamiAEWrapper`.

PAPER-FAITHFUL recipe across d = 3, 32, 64 (only `--d` varies).
Reference: Fukami and Taira, "Grasping Extreme Aerodynamics on a Low-
Dimensional Manifold," arXiv:2305.08024 (Phys. Rev. Fluids 10, 084703,
2025).

The Fukami training objective (their equation 3):

    w* = argmin_w [ ||q - q_hat||_2 + beta * ||C_L - C_L_hat||_2 ]

Quoted from the paper: "we choose beta = 0.05 based on the L-curve
analysis [56]". The lift decoder outputs a single scalar C_L at the
SAME frame as the input vorticity (their Table S.1, "Output 2 (Lift
coefficient)" with Data size (1)). NOT multi-horizon C_L.

Concrete settings:

- omega_pipeline: `outputs/data_pipeline/v1/manifest.json` (canonical
  three-stage pipeline; same as JEPA encoder input).
- recon_loss_type: `mse` (matches the L2 reconstruction term in
  equation 3).
- lambda_recon = 1.0, lambda_lift = 0.05 (Fukami's published beta).
- observable_head = `cl_future`, observable_head_deltas = [0]
  (single-scalar current-frame C_L, matching Fukami's Output 2).
- omega_clip = None, omega_clip_pct = None (pipeline clips already).
- No active-pixel mask (`recon_active_threshold = 0.0`).
- No wake observable head (`wake_observable_weight = 0.0`).

Activation / normalisation adaptation (stability for bf16 mixed
precision; this is the load-bearing variant per CLAUDE.md):

- Activation: `relu` (Fukami used `tanh` in fp32; switching to ReLU
  + GroupNorm stabilises the bf16 training).
- Conv blocks: GroupNorm enabled (`use_conv_norm=True`), n_groups=4.

The strict-paper variant (`tanh` + no GroupNorm + fp32) gives Test B
probe delta of -0.45 on this Re=5000 flow per CLAUDE.md, so it is
documented in the methods appendix as a known-broken variant and is
NOT used in the headline comparison. The (ReLU + GroupNorm + bf16)
adaptation preserves Fukami's objective while making training
numerically stable on RTX 6000 Blackwell.

All three Fukami AE share architecture (encoder
`FukamiCNNEncoder` with 1->32->16->8->4 channels and four 2x maxpools,
288 -> 256 -> 64 -> 32 -> 16 -> d FC chain; mirror decoder; 3-layer
MLP lift head with output 3 = len(deltas)).

Optimizer (same as Fukami's original recipe):

- AdamW, betas=(0.9, 0.95), weight_decay=0.0.
- Peak learning rate 1.0e-3.
- Warmup 5%, cosine decay to 5% of peak.
- Gradient clip 1.0.
- bf16 mixed precision on RTX 6000 Blackwell.
- 20,000 iterations, B = 16, T = 32, partition v1, --all-train.

Verification gate before predictor training:

- Test A SSIM_mean >= 0.60 OR Test A ratio_mean (MSE_mean / floor_mean)
  < 2.0. Either threshold passing is acceptable.

Outputs: `outputs/session18/exp_b1/fukami_ae_d{3,32,64}/checkpoint_iter020000.pt`
+ `final_eval.json` (Test A, B, C per-encounter MSE, SSIM, eps_volume).

Two pre-existing Fukami AE checkpoints (Session 9 `run_a11_fukami_ae_
d3_beta005` and Session 11 `D4_fukami_ae_d32_matched`) were initially
considered as drop-in d=3 and d=32 baselines but were retrained under
this unified recipe to eliminate cross-d preprocessing inconsistencies
(no-pipeline at d=3, lambda_lift=1.0 at d=32). The retrained checkpoints
are the ones used in B1.

## POD (B1 Part b)

Implementation: `scripts/session11_pod_baseline.py`, parameterised by `--d`.

Method: snapshot POD on pipeline-normalised train frames. Centered
truncated SVD via `torch.svd_lowrank` (q = d + 10, niter = 4); take top
d singular triplets.

Settings (locked across d = 16, 32, 64):

- Frame stride 1 (use all train frames).
- Center by global train mean before SVD.
- omega_pipeline: same as Fukami AE.

Outputs: `outputs/session18/exp_b1/pod_d{16,32,64}/{pod_basis.npz,
pod_summary.json}`.

Note: POD is closed-form, no training loop. Reconstruction quality at
d = 16, 32, 64 is reported in the same final_eval.json schema as
Fukami AE.

## Common transformer predictor (B1 Part c)

Implementation: `src/models/predictor.py` `AutoregressivePredictor`.

Architecture (locked across all 6 baseline + predictor pairs):

- latent_dim = baseline's d (the only quantity that varies; this is
  unavoidable for a matched-d comparison and is the only acceptable
  per-baseline knob).
- cond_dim = 3 (G, D, Y).
- hidden_dim = 384.
- depth = 6 layers.
- heads = 16.
- mlp_ratio = 4.0.
- dropout = 0.1.
- max_seq_len = 32.
- RoPE temporal positional encoding on Q and K (not V).
- AdaLN-Zero conditioning on (G, D, Y), one AdaLN before attention and
  one before the MLP per block; identity at initialization.
- BatchNorm at the output projection (matches the encoder projector).
- Causal mask via `F.scaled_dot_product_attention(is_causal=True)`.

Training loss:

- L_pred (teacher forcing): MSE between z_hat[:, t, :] and z[:, t+1, :]
  over the sub-trajectory.
- L_roll (scheduled sampling, H_roll = 8): MSE over an autoregressive
  rollout of length 8 from a randomly chosen anchor inside the
  sub-trajectory.
- Total: L = L_pred + 0.5 * L_roll. No anti-collapse term (the
  latents are frozen from the upstream encoder; the predictor has no
  way to collapse them).

Optimizer (matches the JEPA predictor):

- AdamW, betas = (0.9, 0.95).
- Peak learning rate 5.0e-4.
- weight_decay = 0.05.
- Warmup fraction 0.05, cosine decay to 0.05 of peak.
- Gradient clip 1.0.
- bf16 mixed precision.

Training duration:

- 20,000 iterations. Matches the JEPA training duration in iterations.
- Batch B = 16, sub-trajectory T = 32.
- Identical impact-aware sampler as Fukami AE.

Latents fed to the predictor:

- Fukami AE: take `wrapper.encoder(omega_norm)` for every frame in the
  sub-trajectory. Detached from the autoencoder graph.
- POD: `coeffs[t] = (omega_norm[t].reshape(-1) - mean) @ Phi`. Closed-form.
- JEPA: use the existing `outputs/session14/latents/S12_E_d64/*.npz`
  pre-extracted full-trajectory latents to avoid re-encoding.

Important: no per-baseline tuning of the predictor's hyperparameters.
The same recipe is used for d = 3, 16, 32, 64. Reviewers will check this.

Outputs: `outputs/session18/exp_b1/predictor_{baseline}_d{d}/checkpoint.pt`
where `baseline in {fukami, pod, jepa}` and `d in {3, 16, 32, 64}` as
appropriate.

## Physical Markov closure evaluation (B1 Part d)

Re-uses Session 17 Experiment 2 pipeline. Physical observables computed
per frame from the decoded omega field (Fukami AE has a decoder; POD
has the closed-form projection; JEPA uses the SL decoder
`outputs/runs/session12/S12_E_d64/encoder/decoder_specloss_recipe/
decoder_iter012000.pt`).

Observables:

- C_L: surface integration of pressure (from probe trained on per-frame
  DNS data; same probe across all baselines).
- I_y^w: wake vorticity impulse (the wake-only integral of x * omega over
  a fixed downstream domain). Replaces I_y throughout the paper to
  acknowledge the bound-circulation exclusion noted in Session 17 D124c.
- Wake enstrophy E_w: integrated (omega^2) over the wake domain.
- Spectral lambda ratio: ratio of low-wavenumber energy in the radial
  spectrum at H = 16, 32.

Rollout modes:

- Markov-only: attention masked to (z_impact, self) at every layer;
  ground-truth (G, D, Y) is passed to AdaLN.
- Full-context: standard rollout with up to 32-frame seed ending at
  impact; ground-truth (G, D, Y) is passed.

Horizons: H = 8, 16, 32.

Splits: Test B (28 encounters, in-distribution held-out cases) and
Test C (24 encounters, G = +4 OOD).

Aggregation: bootstrap 2,000 resamples for 95 percent CIs on per-encounter
absolute error. Paired bootstrap when comparing two rollout modes within
the same baseline; independent bootstrap when comparing across baselines.

The headline table is 7 x 4: rows are baselines (JEPA d=64, Fukami AE
d=3, Fukami AE d=32, Fukami AE d=64, POD d=16, POD d=32, POD d=64),
columns are (C_L absolute error at H=16, I_y^w absolute error at H=16,
wake enstrophy absolute error at H=16, lambda ratio at H=16). A second
table reports the same on Test C.

Output: `outputs/session18/exp_b1/physical_closure_comparison.csv` with
columns `(baseline, d, split, horizon, mode, C_L_abs_err, I_y_w_abs_err,
E_w_abs_err, lambda_ratio, n_encounters, ci_low_C_L, ci_high_C_L, ...)`.

## Epiplexity (B1 Part e, optional)

Re-uses Session 14 D100 pipeline. Per-token loss-curve area for each
baseline + predictor pair, sampled at the same iteration grid. Reports
the same metric Session 14 used to give the JEPA-versus-Fukami AE
2.16x absorption ratio at d = 32.

Output: `outputs/session18/exp_b1/epiplexity_comparison.csv`.

## Hardware enforcement

Every training entrypoint must call
`from src.utils.device import require_rtx6000` and exit non-zero if not
on an RTX 6000 Blackwell. CUDA visible device selection is via
`--gpu {0, 1}` for the two cards; never via shell `CUDA_VISIBLE_DEVICES`.

Per CLAUDE.md, the L40S cards (sm_89) must not be used for vortex-jepa
runs. Silent CPU fallback is forbidden.

## W&B logging

Required keys logged for every run (per CLAUDE.md):

- `preprocessing_version` (from `configs/preprocessing.yaml`)
- `partition_version` ("v1")
- `lambda_sigreg` (null for Fukami AE and predictor; n/a for POD)
- `seed`
- `split_sha256`
- `inventory_sha256`
- `code_sha256` or `git_commit`
- `gpu_name` (must contain "RTX" and "6000")
- `wandb_run_id`

Tags: `[fukami_ae, charbonnier_relu_groupnorm]` for Fukami AE runs;
`[pod, snapshot_svd]` for POD; `[predictor_baseline, jepa_recipe]` for
the common transformer predictor on each baseline.

Group: `partition_v1`.

Project: `vortex-jepa` (`WANDB_PROJECT` env var).

## Reproducibility

All seeds locked at 0 for the headline runs. The variance characterization
(Thrust 6 seed retrains) is already done for the JEPA encoder; the B1
comparison does not need to repeat it for Fukami AE and POD in this
session. A seed-variance analysis of the baselines is a follow-up if
reviewers ask for it.

## Acceptance and case decision

Per the Session 18 plan, B1 ends with a Case A versus Case B decision:

- Case A: JEPA wins on physical Markov closure at H = 16 by more than
  20 percent absolute error reduction on C_L AND wake enstrophy versus
  the best baseline. The paper's Section 5 leads with the comparison
  table; the differentiation is on the centerpiece result.

- Case B: Fukami AE at d = 64 matches JEPA on physical Markov closure
  (within 20 percent on both metrics). Section 5 reports the shared
  closure result; the differentiation moves to Sections 6 and 7 (cross-
  seed reproducibility, response-relevant non-Q structure discovery,
  superior pressure observability).

Either outcome is publishable. The decision is documented in D129 of
HANDOFF.md and the manuscript framing is updated accordingly.

## What B1 does NOT include

- Per-baseline hyperparameter tuning. The fairness protocol forbids it.
- Solera-Rico beta-VAE + transformer baseline. Deferred to a follow-up
  paper; the JFM scope is Fukami AE + POD only.
- PLDM end-to-end JEPA-from-pixels comparison. Already shown to
  collapse at the 5-case data scale in Session 5.PLDM smoke (HANDOFF D31);
  reported in the methods appendix as a documented regime-dependent
  fallback rather than a quantitative entry in the comparison table.
- Strict-paper Fukami variant (tanh + no GroupNorm + fp32). Documented
  in the methods appendix as a negative result (Test B probe delta of
  -0.45 in our flow; CLAUDE.md). Excluded from the headline comparison
  to avoid stacking a known-broken variant against the JEPA.
- 3D DNS reanalysis. The 2D mid-plane Wu's theorem limitation stands;
  I_y^w replaces I_y throughout the paper.

## Sign-off

This protocol is the truth for B1 fairness. Any deviation requires a
documented entry in HANDOFF.md and a written justification in the
manuscript's methods section. The locked recipe is in place to make
the comparison defensible to reviewers; the paper's headline depends on
it.
