# SESSION11_WAKE_RESULTS_FIRST.md

Session 11 plan: get wake reconstruction working. Results first, GPU
budget not a constraint. Multiple parallel attack tracks with explicit
escalation if early tracks fail.

Last updated: 2026-05-21.

## Framing

Session 10’s D73 outcome (ALL_THREE_PARTIAL) is “neither LapFiLM nor
CoordMLP recovers wake on the frozen JEPA d=32 latent.” The Session 10
report attributes this to the encoder, the GPT collaborator proposes
wake observable head retraining, and my Session 11 v1 plan added
diagnostic steps before encoder retraining. All three readings agree
the wake problem is real and needs to be solved.

Session 11 commits to solving the wake reconstruction. The plan is
ordered into five tracks with automatic escalation. Tracks fire in
priority order; later tracks launch automatically if earlier ones
do not clear the success criteria. The session ends when wake is
reconstructed OR all five tracks are exhausted.

The five tracks:

- **Track 0**: validation diagnostics (LapFiLM upper bound,
  temporal probe, perturbation probe). Cheap, run first, scope the
  problem.
- **Track 1**: GPT collaborator’s wake observable triplet sweep.
  patch_signed_spectrum, patch_signed, enstrophy_scalar at multiple
  lambda_wake values. Several runs in parallel.
- **Track 2**: my alternative wake observable design: wake-coarse-pool
  (24x12 spatial average of wake ROI vorticity field). Auxiliary head
  predicts the full 288D coarse field.
- **Track 3**: spatial-latent encoder fallback. If Tracks 1 and 2
  both fail, the 32D global latent is structurally inadequate. Replace
  with a spatial latent of shape (12, 6, c) where c is per-cell
  channel count.
- **Track 4**: decoder family swap fallback. If Tracks 1-3 all fail,
  the issue may be neither encoder bottleneck nor encoder objective
  but decoder architecture. Try a vision transformer decoder.

Plus the long-deferred matched-d=32 Fukami AE baseline as a separate
deliverable.

The session does NOT have a wall-clock budget. It has an outcome
budget: stop when wake reconstruction Test B SSIM exceeds 0.50 AND
wake enstrophy relative error drops below 0.45 AND wake structure
is visually identifiable in Figure 3. Whichever track produces this
result is the production configuration; the others become documented
ablations.

If all five tracks fail to meet the success criteria, the session
report includes the full ablation table and Session 12 considers
whether the project’s central claim should pivot from “reconstruction
quality” to “diagnostic suite + downstream prediction” (the latter is
already a publishable contribution per Sessions 5-8).

## What this session does NOT do

- E3 params_phase conditioning ablation. Deferred again.
- bilinear_conv upsampling ablation. Not the bottleneck.
- Hydra refactor, torch.compile.
- Section 7 full evaluation suite. Comes after wake is solved.

## Locked decisions carried forward

- Frame-skip 1, L=32 = 1.6 t/c (D34).
- Partition v1.2 (D35).
- Two RTX 6000 Blackwell cards on cuda:2 and cuda:3 (D38, D40).
- Omega pipeline v1; losses in normalized space; raw scale for
  metrics and figures (Session 9 discipline; D71 bug-fix).
- Frozen JEPA encoder for Track 0 only:
  `outputs/runs/session9/run_jepa_pipeline_lam0p01_seed42/checkpoint_iter020000.pt`.
- Test B is the primary success metric (Test A in-distribution sanity;
  Test C extrapolation stretch).

## Success criteria (the outcome budget)

A configuration “wins” Session 11 if it satisfies all three:

1. Test B SSIM median >= 0.50 (Session 10 best was E2 at 0.391;
   Session 11 target is +25 percent over E2).
1. Test B wake enstrophy relative error median <= 0.45 (Session 10
   best was E4 at 0.568; Session 11 target is -20 percent over E4).
1. Visual Figure 3 inspection on the canonical Test B encounter
   `G+1.00_D1.00_Y+0.10 encounter 00` shows wake shedding at frame
   55 (post-impact) that is recognisable as vortex-street structure,
   not a smudge. This is a qualitative check by the human reviewer;
   the report should include the figure for direct comparison.

Met by ANY of the tracks = session success. None of the tracks meeting
all three = session is a clean negative result; the paper considers
repositioning around the diagnostic contribution.

Test C is reported but not gating. Extrapolation to G=4 is hard and no
project session has cleared meaningful Test C generalization.

## arXiv references (verified)

|Reference                          |arXiv ID  |Use for                                                                                                                                           |
|-----------------------------------|----------|--------------------------------------------------------------------------------------------------------------------------------------------------|
|LapSRN                             |1704.03915|Carry forward from Session 10.                                                                                                                    |
|FiLM                               |1709.07871|Carry forward.                                                                                                                                    |
|PixelShuffle                       |1609.05158|Carry forward.                                                                                                                                    |
|Focal Frequency Loss               |2012.12821|Carry forward.                                                                                                                                    |
|SIREN                              |2006.09661|Carry forward (Track 0 perturbation test, also Track 4 alternative if VAE-style decoder considered).                                              |
|Vision Transformer                 |2010.11929|Track 4 decoder family swap if Tracks 1-3 fail.                                                                                                   |
|MAE (Masked Autoencoders)          |2111.06377|Track 4 alternative; the MAE decoder is a ViT trained for pixel reconstruction from sparse tokens, a natural fit for our problem if Track 4 fires.|
|Latent Diffusion (Stable Diffusion)|2112.10752|Track 4 alternative if simpler decoder swaps fail; the LDM decoder is the strongest known image-from-latent decoder family.                       |
|DINOv2                             |2304.07193|Reference architecture for the spatial-latent Track 3 redesign (DINOv2 uses spatial tokens, not a single vector latent).                          |

The first five are carry-forward from Session 10 and are verified
real. The latter four are Track 4 fallback candidates and should be
consulted via arXiv MCP if Track 4 fires.

## Track 0: validation diagnostics (mandatory, run first)

Three diagnostics in parallel on cuda:2 and cuda:3. All cheap relative
to Tracks 1-4.

### Track 0.1: LapFiLM upper bound test on omega input

Train LapFiLM with omega directly as input (not through the encoder).
This is the upper bound on what LapFiLM can do given perfect input
information. The Session 10 plan called for 5k iters; with no budget
constraint, run 20k iters to full convergence.

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/session9_train_decoder.py \
    --omega-pipeline-manifest outputs/data_pipeline/v1/manifest.json \
    --input-mode omega_direct \
    --decoder-type lapfilm \
    --decoder-upsample pixelshuffle \
    --decoder-loss region_pyr_ffl \
    --lambda-region 1.0 --lambda-pyramid 0.4 --lambda-ffl 0.05 \
    --ffl-warmup-iters 2000 \
    --lambda-enstrophy 0.02 --lambda-circulation 0.01 \
    --max-iters 20000 \
    --B 16 --T 32 --seed 42 \
    --output-dir outputs/runs/session11/T0_1_lapfilm_omega_direct
```

Implementation note: requires adding `--input-mode {latent, omega_direct}` flag to the training script. The omega_direct mode
bypasses the JEPA encoder and feeds omega through a small
`PatchPoolEncoder` (16x16 patch average over 192x96 -> 12x6 spatial
grid with 64 channels via learned linear projection). About 30 lines
of new code.

Pass criteria (interpreted at session end):

- Test B SSIM > 0.65: LapFiLM is very capable given good input. The
  encoder is the bottleneck (H1 confirmed strongly). Tracks 1-3 are
  the right approach.
- Test B SSIM in [0.45, 0.65]: LapFiLM is moderately capable. Mixed
  H1+H2. Tracks 1-3 may help but Track 4 should not be dismissed.
- Test B SSIM < 0.45: LapFiLM cannot reconstruct wake from any
  input at our 192x96 resolution. Track 4 (decoder family swap)
  should fire FIRST, before encoder retraining.

### Track 0.2: temporal-window probe on existing E2 checkpoint

No new training. `scripts/session11_temporal_probe.py` loads E2 and
evaluates three input modes on Test B:

- Single frame: decode(z_t).
- Temporal mean: decode(mean(z_{t-2..t+2})).
- Future window: decode(mean(z_t..z_{t+5})).

If future window improves Test B SSIM by >= 0.05 over single frame,
H3 (temporal context matters) is supported. Then Track 4 should
include a temporal-aware decoder, not just a different per-frame
architecture.

### Track 0.3: latent perturbation probe

No new training. `scripts/session11_perturbation_probe.py` adds
Gaussian noise (sigma in {0.01, 0.05, 0.1, 0.5}) to encoded z_seq,
decodes, and measures wake quality vs sigma.

Robust wake through sigma 0.1 = wake info is in broad latent
directions = encoder is fine. Fragile wake through sigma 0.05 =
wake info is in narrow latent directions = encoder retraining
justified.

## Track 1: GPT collaborator’s wake observable head sweep

The collaborator proposed three wake observable forms. The Session 10
report committed to one of them (radial spectrum). Run all three with
multiple lambda_wake values. Total: 6-9 runs depending on what looks
worth doubling.

### Wake observable target definitions

Compute from pipeline-normalized omega over the wake ROI
(x in [0, 4.5], y in [-1.25, 1.25], excluding airfoil/solid/adjacent
mask). Three modes:

**Mode A: enstrophy_scalar (1 dim)**. Mean omega^2 over the wake ROI,
log1p’d. Cheapest, lowest information.

**Mode B: patch_signed (64 dim)**. 8x4 patch grid over the wake ROI.
Positive-vorticity patch energy `log1p(mean(relu(omega)^2))` gives 32
dims. Negative-vorticity patch energy `log1p(mean(relu(-omega)^2))`
gives 32 dims. Concatenated: 64 dims. Captures coarse vortex placement
including sign.

**Mode C: patch_signed_spectrum (80 dim)**. Mode B’s 64 dims plus a
16-bin radial wake spectrum: `log1p(radial_power_bin_k(omega * wake_mask * hann_window))` for k in [0, 15]. Concatenated: 80 dims.
Captures both spatial coarse structure and spectral content.

All three modes are standardized using train split statistics only.

### Implementation

Per the GPT collaborator’s spec:

```
src/data/wake_observables.py
scripts/session11_precompute_wake_observables.py
src/models/observable_heads.py  (add WakeObservableHead)
```

WakeObservableHead:

```python
class WakeObservableHead(nn.Module):
    def __init__(self, latent_dim=32, out_dim=80, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, z): return self.net(z)
```

Loss: Smooth L1 (Huber) with beta=0.5. Applied at deltas {0, 8, 16,
24} so the encoder at z_t and the predictor at future z_hat both feel
the wake supervision.

### Sweep grid

```
Mode A: enstrophy_scalar
  lambda_wake = 0.03 (W0_A_lam03)

Mode B: patch_signed
  lambda_wake = 0.03 (W0_B_lam03)
  lambda_wake = 0.10 (W0_B_lam10)

Mode C: patch_signed_spectrum (THE COLLABORATOR'S PREFERRED MODE)
  lambda_wake = 0.03 (W0_C_lam03)
  lambda_wake = 0.10 (W0_C_lam10)
  lambda_wake = 0.30 (W0_C_lam30)
```

Six runs total. Parallel on the two cards: three at a time, each ~1.5h.
Total wall-clock ~3-4h. Each run trains the JEPA encoder for 20k iters
on the full v1.2 train partition with the wake observable head added
to the existing CL observable head.

### Evaluation per run

Build a wake-probe summary per run:

- Test B linear probe r2(z -> G, D, Y)
- Test B linear probe r2(z -> CL_future)
- Test B linear probe r2(z -> wake_patch_signed)
- Test B linear probe r2(z -> wake_spectrum)
- Test B linear probe r2(z -> wake_enstrophy)
- Participation ratio of z

The pre-decoder gate (per the GPT collaborator’s design):

- wake_patch_r2_test_b improvement >= 0.10 vs S9 baseline
- wake_spectrum_r2_test_b improvement >= 0.05 vs S9 baseline
- G, D, Y probe delta does not drop by more than 0.02
- CL observable error does not worsen by more than 5 percent
- PR(z) does not collapse below the S9 operating point

For the wake-aware encoder configurations that pass the gate, train
the LapFiLM decoder on the new frozen latent. Use the E1 recipe (no
FFL, region+pyramid+enstrophy+circulation). 20k iters each.

The best Track 1 result (call it W*) is the run that maximizes Test
B SSIM after decoder retraining, subject to the gate.

### Track 1 success / failure

- W* meets the Session 11 success criteria (Test B SSIM >= 0.50,
  wake enstrophy <= 0.45, visible wake in Figure 3): SESSION DONE.
  Document the other Track 1 runs as ablations. Track 2-4 do not
  fire.
- W* improves wake but does not meet criteria (Test B SSIM in
  [0.42, 0.50] for example): partial success. Fire Track 2 (my
  alternative wake target) to see if a different observable form
  reaches the bar.
- No Track 1 run improves over the S9 + decoder baseline meaningfully:
  the wake-observable-head mechanism does not work at all. Fire
  Tracks 3-4 (encoder structural redesign or decoder family swap).

## Track 2: wake-coarse-pool target (parallel to Track 1, or fires if Track 1 partial)

An alternative wake target design: predict the wake-ROI vorticity
field downsampled to 24x12 via average pooling. That is 288 scalar
values per frame, capturing the full spatial structure at coarse
resolution, without locking the encoder to a hand-designed feature
(unlike patch energies or radial spectrum).

Why 24x12: the wake ROI is roughly 86x40 pixels at full resolution.
Pooling to 24x12 gives roughly 3.5 pixels per cell, or 1.6 cells per
chord-length unit in x and 1.0 per chord-length in y. Coarse enough
to be tractable as a low-dim head output; fine enough to capture
vortex-street spacing (which is roughly 2 chord-lengths in our
gust-airfoil problem).

### Implementation

Same WakeObservableHead architecture as Track 1, but with out_dim=288
and a different target preprocessing function in
`src/data/wake_observables.py`:

```python
def wake_coarse_pool_target(omega_norm, wake_mask, out_shape=(24, 12)):
    """Mean-pool wake-ROI vorticity to (24, 12). Standardize."""
    # Apply wake mask, then 2D adaptive_avg_pool2d to (24, 12).
    # Then flatten to 288D and standardize via train split statistics.
    ...
```

### Sweep

```
W2_coarse_lam03  (lambda_wake = 0.03)
W2_coarse_lam10  (lambda_wake = 0.10)
W2_coarse_lam30  (lambda_wake = 0.30)
```

Three runs. After Track 1 sweep, run these on the freed cards. ~4-5h
total wall-clock.

Then decoder retrain on the best Track 2 encoder, evaluate, compare.

### Track 2 success / failure

Same criteria as Track 1. If Track 2 reaches the bar, SESSION DONE.

If both Track 1 and Track 2 are partial (some improvement, none
meets criteria), the wake-observable approach has a ceiling at our
data scale. Fire Track 3.

If both Track 1 and Track 2 are total failures (no improvement),
the encoder structural design is fundamentally inadequate; the 32D
global latent cannot hold both prediction-relevant features AND
wake-region structure. Fire Track 3.

## Track 3: spatial-latent encoder fallback (fires if Tracks 1+2 fail)

If wake observable supervision through a 32D global latent cannot
recover wake reconstruction, the latent bottleneck itself is the
problem. The Session 7-8 architectural work locked d=32 per LeWM
on intrinsic-dim grounds. But intrinsic dim arguments assume the
latent has to summarize the WHOLE flow into a single vector. The
wake reconstruction problem may need a spatial latent.

### Design

Replace the encoder’s final pooling with a spatial latent of shape
(12, 6, c) where c is per-cell channel count. Try c=4 (288 total
dims, comparable budget to 32D x 9 features) and c=8 (576 dims).

The JEPA prediction loss still applies but now on the spatial
latent: the predictor takes z_t (12x6xc) and c=(G,D,Y) and predicts
z_{t+1} (12x6xc). The auxiliary CL head pools the spatial latent to
a vector before predicting CL. The auxiliary wake head (if used)
predicts wake_coarse_pool directly from z without further pooling.

The decoder LapFiLM is unchanged structurally but now starts from a
12x6 spatial input rather than a 32D vector. This is actually
closer to its natural input shape than the current configuration.

Implementation cost: ~200 lines of encoder + predictor modification
plus a few training-script flag changes. The hybrid CNN-ViT encoder’s
final stage changes from “flatten + linear to 32” to “keep spatial
tokens at 12x6”.

### Runs

```
T3_spatial_c4_no_wake   (spatial latent, no wake head)
T3_spatial_c4_wake_C    (spatial latent + Mode C wake head, lambda_wake=0.1)
T3_spatial_c8_wake_C    (larger spatial latent + Mode C wake head)
```

Three runs. After Tracks 1-2 land, ~5-6h wall-clock.

Then decoder retrain (LapFiLM expects 12x6 input naturally; this
should be cleaner than the 32D-vector starting point).

### Track 3 success / failure

If T3_spatial_c4_wake_C or T3_spatial_c8_wake_C clears the success
criteria, SESSION DONE. The paper claim shifts: “JEPA with spatial
latent + wake observable head recovers wake reconstruction; the
global-vector latent was the bottleneck.”

If even spatial latent + wake supervision fails, Track 4 fires.

## Track 4: decoder family swap fallback (fires if Tracks 1+2+3 fail)

If three different encoder objectives plus a spatial-latent redesign
cannot recover wake reconstruction, the issue may not be the encoder
at all. The decoder family may be the wrong tool. LapFiLM is a CNN
pyramid; CoordMLP is a SIREN-style coordinate field. Two natural
alternatives:

### Track 4.1: vision transformer decoder

A ViT decoder treats the output as a grid of tokens, processes them
with self-attention, and projects to pixels. Strongly different
inductive bias from CNN pyramid: long-range dependencies are
first-class, not implicit through receptive-field growth. MAE
(arXiv:2111.06377) is the canonical reference; its decoder is a small
ViT that reconstructs from sparse tokens.

For our problem: input is z (either 32D global or 12x6 spatial from
Track 3), output is 192x96 grid. The ViT decoder upsamples 12x6 to
192x96 via patch-token unrolling and per-patch MLPs, with self-
attention across all 72 patches.

Implementation cost: ~300 lines of new code. Roughly comparable to
LapFiLMDecoder.

### Track 4.2: diffusion-style refinement decoder

Run LapFiLM to produce a coarse omega prediction, then run a small
denoising network conditioned on the LapFiLM output to refine wake
details. The denoiser is trained as a standard DDPM with a
small number of timesteps (say 10) to keep inference fast. This is
NOT full latent diffusion (arXiv:2112.10752) which would require
much more infrastructure, but a “diffusion refinement” head.

Implementation cost: ~400 lines plus the noise scheduling logic.

### Track 4 choice

Run Track 4.1 first (cleaner conceptually and cheaper to implement).
If Track 4.1 fails to meet criteria, try Track 4.2.

## Matched-d=32 Fukami AE baseline (independent of Tracks 0-4, runs in parallel)

This has been deferred since Session 9 Section 7b. Run it now
regardless of which track is firing.

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/session9_train_fukami.py \
    --omega-pipeline-manifest outputs/data_pipeline/v1/manifest.json \
    --latent-dim 32 \
    --max-iters 20000 \
    --B 16 --T 32 --seed 42 \
    --output-dir outputs/runs/session11/D4_fukami_ae_d32_matched
```

Wall-clock ~2 hours on cuda:3. Compare against Session 10 E2 on Test
B SSIM, eps, wake metrics.

This produces a result regardless of how Tracks 0-4 land. It is
paper-essential for the JEPA-vs-AE contribution claim.

## Sequencing

Track 0 runs first (~1-2h wall-clock for the upper-bound + probes
in parallel).

Track 1 launches concurrent with Track 0 if results are clear
already, or after Track 0 completes if diagnostics inform Track 1
design (e.g. Track 0 suggests LapFiLM is fine, so trust the wake
observable head approach more).

Track 1 takes ~4-5h wall-clock for 6 runs and decoder retrains.

Track 2 runs after Track 1 unless Track 1 already met success
criteria. ~4-5h.

Tracks 3 and 4 fire only if needed.

Matched-d Fukami baseline runs in parallel on whichever card is
free, ~2h.

Estimated total wall-clock if Track 1 succeeds: ~7-8 hours.
If Track 2 also needed: ~12-13 hours.
If Track 3 needed: ~18-19 hours.
If Track 4 needed: ~24-26 hours.

The session terminates when success criteria are met OR all tracks
are exhausted. The agent should not stop early; the goal is wake
reconstruction.

## Step-by-step launch procedure for the agent

After Track 0 diagnostics complete (1-2h):

1. Implement `src/data/wake_observables.py` with all three GPT modes
   plus the wake_coarse_pool target. Tests pass.
1. Implement `src/models/observable_heads.py:WakeObservableHead`.
   Tests pass.
1. Implement `scripts/session11_precompute_wake_observables.py`.
   Run it for all four target types, save HDF5 files.
1. Modify `src/training/train_jepa.py` with wake observable flags.
   Test with –max-iters 10 smoke. Tests pass.
1. Launch Track 1 sweep (6 runs in parallel across two cards).
   ~4-5h.
1. For each Track 1 run, evaluate the wake-probe summary. Identify
   those passing the gate.
1. For gate-passing runs, retrain the LapFiLM decoder on the new
   frozen encoder. Evaluate Test B SSIM and wake metrics.
1. Compute Figure 3 for each gate-passing decoder.
1. If any Track 1 configuration meets Session 11 success criteria,
   STOP and write session report. SESSION DONE.
1. Otherwise launch Track 2 sweep (3 runs).
1. Repeat evaluation + decoder retrain.
1. If success, STOP.
1. Otherwise launch Track 3 (spatial latent encoder). 3 runs +
   decoder retrains.
1. If success, STOP.
1. Otherwise launch Track 4 (ViT decoder, then if needed diffusion
   refinement).

At each STOP point, write a partial session report documenting
which tracks fired, what succeeded, what was learned.

## D-entries to record

**D78**: Track 0 results (LapFiLM upper bound + temporal probe +
perturbation probe). The H1-vs-H2-vs-H3 disambiguation.

**D79**: Matched-d=32 Fukami AE baseline result.

**D80**: Track 1 sweep results (6 runs). The wake-observable-head
sweep landscape.

**D81 (conditional)**: Track 2 results.

**D82 (conditional)**: Track 3 results (spatial latent).

**D83 (conditional)**: Track 4 results (ViT decoder or diffusion).

**D84**: Session 11 outcome. Which track produced the production
configuration, plus the full ablation table.

## Risk register

|Risk                                                                                  |Probability|Mitigation                                                                                                                                |If it fires                                                                                                               |
|--------------------------------------------------------------------------------------|-----------|------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
|Wake observable hurts CL prediction in Track 1                                        |medium-high|lambda_wake sweep includes 0.03 (small enough to not dominate); the gate explicitly checks CL doesn’t drop                                |Lower lambda_wake; if 0.03 still hurts, the wake supervision is fundamentally incompatible with CL prediction at our scale|
|Track 1 sweep finds NO configuration that passes the gate                             |medium     |The gate is strict but reasonable; if all fail, the wake observable head mechanism may be wrong                                           |Track 2 alternative target                                                                                                |
|Track 1 best run passes the gate but the decoder retrain still doesn’t reach SSIM 0.50|medium-high|The wake observable adds information to z, but a 32D global latent may not have capacity for all of (G,D,Y) + CL_future + wake_field      |Track 3 spatial latent                                                                                                    |
|Spatial latent in Track 3 destabilises JEPA prediction                                |medium     |The predictor on spatial input is more complex; may need its own retraining recipe                                                        |Reduce spatial channels (c=2) before giving up                                                                            |
|ViT decoder in Track 4 produces worse results than LapFiLM                            |low-medium |The MAE-style decoder is well-validated on images but our wake-vorticity domain may not transfer                                          |Try diffusion refinement (Track 4.2)                                                                                      |
|Matched-d Fukami AE d=32 beats JEPA+decoder by large margin                           |low-medium |Paper claim 3 may need to weaken; the JEPA contribution shifts from “best reconstruction” to “best prediction + reasonable reconstruction”|Paper Section 5 rewrite                                                                                                   |
|All five tracks fail                                                                  |low        |Clean negative result; the paper repositions around the diagnostic contribution                                                           |Session 12 is paper rewrite                                                                                               |

## Three predictions worth pre-registering

Track 0 (LapFiLM on omega) reaches Test B SSIM > 0.65: credence ~65%.
Direct omega input is much richer than 32D latent; LapFiLM has 707k
params; should exploit it well at full 20k iter training.

Track 1 (best wake observable head config) clears Session 11 success
criteria: credence ~30%. The wake observable mechanism is plausible
but the 32D global latent may not have capacity. The GPT collaborator
is more optimistic than I am about this.

Track 3 (spatial latent encoder, if it fires) clears criteria where
Track 1 did not: credence ~50% given Track 1 partial success. The
spatial latent gives the encoder a much more natural representational
shape for spatial structure like wake.

Net session-success credence (any track meeting criteria): ~70%.
The 30% failure scenario is the one where the paper repositions
around diagnostic + prediction contributions, which is already
publishable.

## Decision references

Carry forward: D34, D35, D38, D40, D44-D49, D50-D57, D60-D69,
D70-D77.

This session: D78-D84.