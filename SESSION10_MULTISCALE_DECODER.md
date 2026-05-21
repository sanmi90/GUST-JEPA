# SESSION10_MULTISCALE_DECODER.md

Session 10 plan: improve reconstruction quality of the JEPA visualisation
decoder via a multiscale Laplacian-pyramid architecture, with a coordinate-
neural-field audit to disentangle decoder capacity from latent information
content.

Last updated: 2026-05-21.

## Framing

Session 9 established that the JEPA decoder on the d=32 pipeline-trained
latent achieves Test A SSIM 0.503, Test B SSIM 0.358, Test C SSIM 0.243.
This is strictly above every Fukami AE configuration tested at d=3 or
d=8. But Figure 3 still shows clear deficits in wake-scale structure:
the gust core is reproduced convincingly through impact, but the shedding
behind it is partially absent. The paper would benefit from a Figure 3
that captures both core dynamics AND wake shedding faithfully.

The Session 9 report attributes the wake-erasure failure mode at the
Fukami AE level to "decoder-capacity-limited" rather than loss-design-
limited. The seven-variant AE loss study found no loss tuning recovered
publication-quality reconstruction at d=3 or d=8. This evidence directs
Session 10 toward decoder architecture rather than more loss tweaks on
the existing decoder.

Two questions the session must answer:

1. **Does a multiscale (Laplacian-pyramid) decoder architecture improve
   reconstruction quality on the JEPA d=32 latent?** If yes, the
   current Fukami-style decoder was the bottleneck. If no, the JEPA
   latent itself lacks wake-scale information.

2. **Is the small-scale wake information present in the JEPA latent
   at all?** A coordinate neural field decoder (coord_mlp with Fourier
   features) is a much stronger reconstructor than any pyramid CNN
   for high-frequency signals. If coord_mlp also fails to recover wake
   structure on the JEPA latent, the limitation is in the encoder, not
   the decoder.

These two questions structure the three production runs of the session.
Other promising directions (conditioned decoder, frozen-Fukami-d=32
latent comparison, matched-d Fukami end-to-end baseline) are deferred
to Session 11 to keep Session 10 focused.

## How this differs from the GPT collaborator's proposal

The collaborator's plan proposed six experiments (E0-E5) plus three
decoder types plus a 5-term loss with five lambdas. This is the right
direction but too large for one session.

Five specific changes from the collaborator's plan, in order of
importance.

**E0 (Fukami decoder MSE reproduction) dropped.** Session 9 already
produced this exact checkpoint at outputs/runs/session9/decoder_pipeline
_mse with SSIM=0.503/0.358/0.243. Re-running is 1.5h of GPU producing a
known number. The existing checkpoint serves as the baseline.

**E3 (params_phase conditioning) deferred to Session 11.** The
conditioning question is methodologically important but it bundles two
design choices: (a) whether FiLM is the right mechanism for latent
conditioning, and (b) whether adding (G, D, Y, phase) on top of z helps.
Session 10 isolates (a) with a no_film ablation; Session 11 tests (b)
after the FiLM mechanism is validated.

**E5 (LapFiLM on frozen Fukami d=32 latent) deferred to Session 11.**
This is the right question for the paper but it answers "is decoder
architecture the limitation for Fukami too" which is a Session 11 task
once we know whether the architecture helps JEPA.

**Enstrophy and circulation losses fixed.** The collaborator's code
compares the scalar mean enstrophy/circulation between predicted and
target, which is a global integral constraint. A model can satisfy
this with uniform noise. The intended physics constraint is the
spatial enstrophy/circulation field match. Session 10 implements the
pixelwise version: `(weight * (pred.pow(2) - target.pow(2))).abs().mean()`
for enstrophy and `(weight * (pred - target)).abs().mean()` for
circulation density.

**A `no_film` ablation added.** The collaborator's plan uses FiLM
conditioning by default but never tests whether FiLM is necessary
versus simpler concat-and-conv. Adding one no_film ablation
(LapFiLM-no-film with z concatenated as initial channels but no FiLM
modulation) closes a defensibility gap for the paper. Roughly 1.5h
extra GPU.

## arXiv references (verified)

| Reference | arXiv ID | Use for |
|---|---|---|
| LapSRN: Lai, Huang, Ahuja, Yang | 1704.03915 | Laplacian pyramid super-resolution; sub-band residual prediction at each pyramid level, Charbonnier loss. Architecture verified. |
| FiLM: Perez, Strub, de Vries, Dumoulin, Courville | 1709.07871 | Feature-wise linear modulation; the canonical mechanism for conditioning convolutional features on an external vector. Verified. |
| PixelShuffle / sub-pixel CNN: Shi et al. | 1609.05158 | Learned upsampling via channel-to-space rearrangement; avoids bilinear smoothing. Verified. |
| CoordConv: Liu et al. | 1807.03247 | Adding coordinate channels to convolutional inputs; addresses convolution's translation equivariance limitation on global geometry. Verified. |
| Fourier Features: Tancik et al. | 2006.10739 | Coordinate MLPs learn high-frequency functions only when inputs are Fourier-encoded. Used in the coord_mlp audit. Verified. |
| Focal Frequency Loss: Jiang, Dai, Wu, Loy (ICCV 2021) | 2012.12821 | Adaptive focus on hard-to-synthesise frequency components by down-weighting easy ones. Verified, demonstrated on VAE/pix2pix/SPADE. |
| SIREN: Sitzmann et al. | 2006.09661 | Implicit neural representations with sinusoidal activations; designed for fine-detail signals and PDE-related fields. Optional for the coord_mlp audit. Verified. |

All seven references are real, present on arXiv, and used correctly
in the implementation. The collaborator's citations are accurate.

## Locked decisions carried forward

- Frame-skip = 1, dt_eff = 0.05, L = 32 = 1.6 t/c (D34).
- Partition v1.2: 49 cases, 138 train encounters (D35).
- Two RTX 6000 cards on cuda:2 and cuda:3 per D38/D40.
- Omega pipeline v1: spatial mask + p99.99 clip + 3-sigma normalize.
  Loss computed in normalized space; metrics and figures in raw scale
  per Session 9.
- JEPA encoder: pipeline-trained at lambda*=0.01, seed 42, d=32.
  Checkpoint at outputs/runs/session9/run_jepa_pipeline_lam0p01_seed42.
- Test B is the primary success metric. Test A is in-distribution
  sanity. Test C is a stretch goal (extrapolation to G=4).

## Files to add

Two new source files:

```text
src/models/lap_film_decoder.py
src/models/decoder_losses.py
```

One new architecture file for the audit:

```text
src/models/coord_mlp_decoder.py
```

Plus tests:

```text
tests/test_lap_film_decoder.py
tests/test_decoder_losses.py
tests/test_coord_mlp_decoder.py
```

## Files to modify

```text
scripts/session9_train_decoder.py  (extend with new flags)
scripts/session9_decoder_fig3_pipeline.py  (handle lapfilm + coord_mlp)
```

The Session 9 entrypoint and figure pipeline are extended rather than
replaced. New CLI flags get added but existing flags keep their
behaviour.

## Step 1: implement LapFiLMDecoder (~3 hours)

`src/models/lap_film_decoder.py`. Five-level pyramid (12x6, 24x12,
48x24, 96x48, 192x96). FiLMResBlock with GroupNorm + SiLU, 2 res
blocks per level. Coordinate and Fourier features concatenated per
level. Optional airfoil-adjacent-mask channel from
`outputs/data_pipeline/v1/airfoil_adjacent_mask.npy`.

The architecture spec follows the collaborator's proposal exactly:

```python
decoder_type = "lapfilm"
latent_dim = 32
base_hw = (12, 6)
base_ch = 64
channels = [64, 64, 48, 32, 24]
resblocks_per_level = 2
norm = "groupnorm"
activation = "silu"
upsample = "pixelshuffle"  # also support bilinear_conv
fourier_bands = 4
use_coord_channels = True
use_airfoil_mask_channel = True
use_film = True              # for ablation: False = concat-only
final_activation = None
```

The forward pass returns a dict with `pred` (the final 192x96
prediction) and `pyramid` (list of all 5 pyramid-level predictions in
order coarse-to-fine). The pyramid loss in Step 2 uses both.

Two upsampling modes supported via `upsample` argument:
- `pixelshuffle`: nn.Conv2d to 4x channels then nn.PixelShuffle(2).
  Learns the upsampling kernel.
- `bilinear_conv`: F.interpolate(mode='bilinear', align_corners=False)
  followed by nn.Conv2d. Safer against checkerboard artifacts.

Run pixelshuffle by default. The bilinear_conv variant becomes a
Session 11 ablation if pixelshuffle shows artifacts.

### Unit tests for LapFiLMDecoder

`tests/test_lap_film_decoder.py`:

```python
def test_lap_film_decoder_shape_contract():
    """Input z (2, 32), output pred (2, 1, 192, 96).
    Pyramid has 5 entries with shapes (2,1,12,6), (2,1,24,12),
    (2,1,48,24), (2,1,96,48), (2,1,192,96).
    torch.manual_seed(0)."""

def test_lap_film_decoder_no_nan_at_init():
    """Forward pass on random z produces finite outputs.
    No NaN, no Inf at initialization."""

def test_lap_film_decoder_gradient_flows():
    """Backward through MSE on the final prediction produces
    nonzero gradients on all FiLM linear weights and the per-level
    convolutional weights."""

def test_lap_film_decoder_no_film_ablation():
    """use_film=False produces a working decoder that takes z as
    initial channels rather than as FiLM modulation. Same output
    shape. Different parameter count (no FiLM linear layers)."""

def test_lap_film_decoder_airfoil_mask_optional():
    """use_airfoil_mask_channel=True adds an input channel at every
    pyramid level. use_airfoil_mask_channel=False produces a working
    decoder with one fewer input channel per level."""

def test_lap_film_decoder_bf16_autocast():
    """Forward+backward under torch.autocast bf16 succeeds on the
    RTX 6000 Blackwell. Skip if no GPU."""
```

## Step 2: implement decoder_losses.py (~2 hours)

`src/models/decoder_losses.py`. Five loss functions plus the combined
`region_pyr_ffl` loss type. Loss is always computed in normalized
space per the Session 9 discipline (raw scale for metrics and
figures only).

```python
def charbonnier(x, eps=0.05):
    return torch.sqrt(x * x + eps * eps) - eps

def region_weight(target_norm, coord, solid_or_airfoil_mask=None):
    # Soft active mask + wake ROI weight. Floor 0.05, never zero.
    # Wake ROI: x in (0, 4.5), |y| < 1.25.
    # Returns (B, 1, H, W) normalized so mean is 1.
    ...

def weighted_mse(pred, target, weight):
    return (weight * (pred - target).pow(2)).mean()

def pyramid_residual_loss(pred_pyr, target, eps=0.05):
    # Charbonnier loss on residuals at each pyramid level.
    # Weights: [0.10, 0.20, 0.40, 0.80, 1.00] coarse-to-fine.
    ...

def local_focal_frequency_loss(pred, target, patch=32, alpha=1.0):
    # Per-patch FFT, with weight = |diff|.pow(alpha).detach().
    # Weight is normalized per-patch.
    ...
```

The enstrophy and circulation losses are implemented spatially, not
as global integral comparisons. This is the bug fix from the
collaborator's proposal.

```python
def enstrophy_field_loss(pred, target, weight=None):
    """L2 difference of spatial enstrophy fields (not integrals).
    Enstrophy is omega^2 pointwise. Comparison is spatial, so a
    model cannot pass this with uniform noise of the right total
    enstrophy."""
    if weight is None:
        weight = 1.0
    diff = pred.pow(2) - target.pow(2)
    return (weight * diff.pow(2)).mean()

def circulation_density_loss(pred, target, weight=None):
    """L1 difference of vorticity (circulation density) at each
    point. Same spatial comparison logic as enstrophy_field_loss.
    Vorticity is signed so use L1 rather than L2 to avoid
    cancellation issues at large positive/negative pairs."""
    if weight is None:
        weight = 1.0
    diff = pred - target
    return (weight * diff.abs()).mean()
```

Both losses optionally take a weight mask. Use the wake_roi mask
(from region_weight's wake component) so the enstrophy and
circulation constraints apply only to the wake region. The
freestream and pre-impact regions are by construction approximately
uniform; enforcing zero-mean enstrophy difference there adds noise.

### Combined `region_pyr_ffl` loss

```python
def region_pyr_ffl_loss(
    pred_pyr,            # list of pyramid predictions, coarse to fine
    target,              # (B, 1, 192, 96) in normalized space
    coord,               # dict with 'x', 'y' grids
    solid_or_airfoil_mask=None,
    lambda_region=1.0,
    lambda_pyramid=0.4,
    lambda_ffl=0.05,
    lambda_enstrophy=0.02,
    lambda_circulation=0.01,
    ffl_alpha=1.0,
    ffl_patch=32,
    ffl_warmup_factor=1.0,  # 0.0 during warmup, 1.0 after
):
    final_pred = pred_pyr[-1]
    weight = region_weight(target, coord, solid_or_airfoil_mask)

    L_region = weighted_mse(final_pred, target, weight)
    L_pyramid = pyramid_residual_loss(pred_pyr, target)
    L_ffl = local_focal_frequency_loss(final_pred, target,
                                        patch=ffl_patch, alpha=ffl_alpha)
    L_enstrophy = enstrophy_field_loss(final_pred, target, weight=weight)
    L_circulation = circulation_density_loss(final_pred, target,
                                              weight=weight)

    L_total = (lambda_region * L_region
               + lambda_pyramid * L_pyramid
               + lambda_ffl * ffl_warmup_factor * L_ffl
               + lambda_enstrophy * L_enstrophy
               + lambda_circulation * L_circulation)

    return {
        "L_total": L_total,
        "L_region": L_region,
        "L_pyramid": L_pyramid,
        "L_ffl": L_ffl,
        "L_enstrophy": L_enstrophy,
        "L_circulation": L_circulation,
    }
```

FFL warmup: `ffl_warmup_factor = 0.0` for the first 2000 iterations,
then linearly ramp to 1.0 over 1000 iterations. This prevents the
decoder from chasing high-frequency noise before it has learned the
gust core. Implementation: outside the loss function, the training
loop computes the warmup factor from the current iteration and
passes it as an argument.

### Unit tests for decoder_losses

`tests/test_decoder_losses.py`:

```python
def test_charbonnier_zero_at_zero():
    """charbonnier(0) = 0 exactly. charbonnier(x) > 0 elsewhere.
    charbonnier(x).backward() produces finite gradients."""

def test_region_weight_floor():
    """Weight has floor 0.05 everywhere. Wake ROI region has
    weight > 0.5. Solid mask region has weight 0 if mask provided."""

def test_pyramid_loss_zero_on_perfect_pyramid():
    """If predictions equal targets at all 5 pyramid levels,
    pyramid_residual_loss is exactly zero."""

def test_local_ffl_zero_on_perfect_reconstruction():
    """If pred == target, local_focal_frequency_loss is zero."""

def test_local_ffl_finite_on_perfect_freestream():
    """If both pred and target are zero everywhere, FFL is zero
    (no NaN from division by zero norm)."""

def test_enstrophy_field_loss_zero_on_perfect():
    """If pred == target, enstrophy_field_loss is zero. Even
    when pred and target are both nonzero noise."""

def test_enstrophy_field_loss_nonzero_on_uniform_noise():
    """Pred = uniform noise, target = structured wake. Both have
    same scalar mean enstrophy by construction (matched variance).
    The collaborator's mean-comparison loss is zero on this case;
    enstrophy_field_loss is nonzero. This is the bug-fix test."""

def test_circulation_density_loss_zero_on_perfect():
    """If pred == target, circulation_density_loss is zero."""
```

The `test_enstrophy_field_loss_nonzero_on_uniform_noise` test is
the explicit regression check against the collaborator's proposed
scalar-mean comparison.

## Step 3: implement CoordMLPDecoder (~2 hours)

`src/models/coord_mlp_decoder.py`. Implicit neural representation
that maps (x, y) coordinates plus latent z to vorticity values.
Used only for the diagnostic audit, not as a production decoder.

```python
class CoordMLPDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 32,
        hidden: int = 128,
        layers: int = 5,
        fourier_bands: int = 8,
        activation: str = "sine",  # or "gelu_fourier"
        chunk_pixels: int = 4096,
        H: int = 192,
        W: int = 96,
    ):
        ...

    def forward(self, z, coords=None):
        """z: (B, latent_dim)
        coords: (N, 2) or None (full grid built from H, W)
        Returns: (B, 1, H, W) reconstruction.
        """
        ...
```

Activation = sine corresponds to SIREN (Sitzmann et al.,
arXiv:2006.09661). The gelu_fourier variant uses GELU activations
on Fourier-encoded inputs (Tancik et al., arXiv:2006.10739). Test
both; SIREN typically performs better on PDE-like fields, but GELU
is more standard in modern ML.

Chunked forward: for a 192x96 = 18432 pixel field at batch 16, the
total memory for unchunked forward is ~16 * 18432 * (32 + Fourier
features) which can be large. Process pixels in chunks of 4096.

### Unit tests for CoordMLPDecoder

`tests/test_coord_mlp_decoder.py`:

```python
def test_coord_mlp_shape_contract():
    """Input z (2, 32), output (2, 1, 192, 96)."""

def test_coord_mlp_chunking_invariant():
    """Output is bitwise identical for chunk_pixels in {1024,
    4096, 18432 (no chunking)}. The chunk size is a memory
    optimization only, not a numerical choice."""

def test_coord_mlp_siren_vs_gelu_both_train():
    """Both activation modes produce finite gradients and the
    parameter counts differ in the expected way."""
```

## Step 4: extend the training entrypoint (~1 hour)

Modify `scripts/session9_train_decoder.py` to accept the new flags.
The plan flags:

```text
--decoder-type {fukami, lapfilm, coord_mlp}
--decoder-upsample {pixelshuffle, bilinear_conv}
--decoder-fourier-bands  (default 4 for lapfilm, 8 for coord_mlp)
--decoder-base-ch        (default 64 for lapfilm)
--decoder-resblocks-per-level  (default 2 for lapfilm)
--decoder-use-film {true, false}  (default true; false for ablation)

--decoder-loss {mse, charbonnier, region_pyr_ffl}
--lambda-region          (default 1.0)
--lambda-pyramid         (default 0.4)
--lambda-ffl             (default 0.05)
--lambda-enstrophy       (default 0.02)
--lambda-circulation     (default 0.01)
--ffl-warmup-iters       (default 2000)
--ffl-ramp-iters         (default 1000)
--active-tau             (default 0.10)
--active-softness        (default 0.03)
--inactive-weight        (default 0.05)
--wake-weight            (default 0.50)
```

Existing flags (--encoder-run, --omega-pipeline-manifest, --max-iters,
--B, --T, --seed, --output-dir) keep their behaviour.

The `--decoder-cond` flag is added but defers to Session 11 for the
params and params_phase modes. Session 10 uses `--decoder-cond none`
exclusively.

Roughly 70 lines of argparse plus 30 lines for the dispatch to
decoder_type and decoder_loss. Plus one unit test in
`tests/test_decoder_cli_args.py`.

## Step 5: extend the figure pipeline (~30 min)

Modify `scripts/session9_decoder_fig3_pipeline.py` to handle the new
decoder types. The figure layout for the canonical Test B encounter
G+1.00_D1.00_Y+0.10 encounter 00 stays the same (frames 25, 40, 55;
fixed colorbar ±3; airfoil overlay). The script gains a dispatch on
decoder_type to load the right checkpoint.

## Step 6: three production runs

After Steps 1-5 land and unit tests pass, three production runs in
parallel/sequential on the two RTX 6000 cards.

### Run E1: LapFiLM, no FFL, pyramid + region + enstrophy + circulation

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/session9_train_decoder.py \
    --omega-pipeline-manifest outputs/data_pipeline/v1/manifest.json \
    --encoder-run outputs/runs/session9/run_jepa_pipeline_lam0p01_seed42 \
    --decoder-type lapfilm \
    --decoder-upsample pixelshuffle \
    --decoder-use-film true \
    --decoder-loss region_pyr_ffl \
    --lambda-region 1.0 \
    --lambda-pyramid 0.4 \
    --lambda-ffl 0.0 \
    --lambda-enstrophy 0.02 \
    --lambda-circulation 0.01 \
    --max-iters 20000 \
    --B 16 --T 32 --seed 42 \
    --output-dir outputs/runs/session10/E1_jepa_lapfilm_pyr_noffl
```

Tests the multiscale architecture without the focal frequency loss.
Isolates the architecture contribution from the FFL contribution.
Wall-clock ~1.5h on cuda:2.

### Run E2: LapFiLM with FFL

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/session9_train_decoder.py \
    --omega-pipeline-manifest outputs/data_pipeline/v1/manifest.json \
    --encoder-run outputs/runs/session9/run_jepa_pipeline_lam0p01_seed42 \
    --decoder-type lapfilm \
    --decoder-upsample pixelshuffle \
    --decoder-use-film true \
    --decoder-loss region_pyr_ffl \
    --lambda-region 1.0 \
    --lambda-pyramid 0.4 \
    --lambda-ffl 0.05 \
    --ffl-warmup-iters 2000 \
    --ffl-ramp-iters 1000 \
    --lambda-enstrophy 0.02 \
    --lambda-circulation 0.01 \
    --max-iters 20000 \
    --B 16 --T 32 --seed 42 \
    --output-dir outputs/runs/session10/E2_jepa_lapfilm_pyr_ffl
```

The full LapFiLM + region + pyramid + FFL + enstrophy + circulation
combination on the JEPA latent. Parallel with E1 on cuda:3. Wall-clock
~1.5h.

### Run E4: CoordMLPDecoder audit

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/session9_train_decoder.py \
    --omega-pipeline-manifest outputs/data_pipeline/v1/manifest.json \
    --encoder-run outputs/runs/session9/run_jepa_pipeline_lam0p01_seed42 \
    --decoder-type coord_mlp \
    --decoder-fourier-bands 8 \
    --decoder-loss region_pyr_ffl \
    --lambda-region 1.0 \
    --lambda-pyramid 0.0 \
    --lambda-ffl 0.03 \
    --ffl-warmup-iters 2000 \
    --max-iters 20000 \
    --B 16 --T 32 --seed 42 \
    --output-dir outputs/runs/session10/E4_jepa_coordmlp_audit
```

Sequential on cuda:2 after E1 finishes. The audit: does a coordinate
neural field decoder, given unlimited spatial resolution and Fourier
features, recover wake-scale structure from the frozen JEPA latent?
If yes, the latent has the information and the LapFiLM result tells
us how much of that information is recoverable by a CNN decoder. If
no, the latent itself lacks wake structure and Session 11 will need
to retrain the encoder with a wake-aware objective. Wall-clock ~1.5h.

### Optional Run E_noFiLM: LapFiLM without FiLM (ablation, conditional)

If E2 substantially beats the Session 9 baseline, run a no_film
ablation on the same configuration:

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/session9_train_decoder.py \
    [...same as E2 but with --decoder-use-film false] \
    --output-dir outputs/runs/session10/E_noFiLM_jepa_lapfilm_concat
```

Tests whether the FiLM conditioning specifically contributes, vs
simpler concat-and-conv. If no_FiLM performs comparably, the paper
description simplifies. If FiLM substantially helps, the paper makes
the architectural claim explicitly.

Runs after E2 on cuda:3 if conditions met. Wall-clock ~1.5h.

### Sequencing

cuda:2: E1 (1.5h), then E4 (1.5h). Total 3h.
cuda:3: E2 (1.5h), then optional E_noFiLM (1.5h conditional). Total
1.5-3h.

Total compute wall-clock: 3-4 hours.

## Step 7: full evaluation and Figure 3 (~2 hours)

For each of E1, E2, E4 (and E_noFiLM if run), compute the full
decoder evaluation suite plus generate the Figure 3 panel.

### Metrics in decoder_summary.json

| Metric | Test A | Test B | Test C |
|---|---|---|---|
| SSIM (full field) | * | * | * |
| epsilon_vol | * | * | * |
| MSE (full, normalized) | * | * | * |
| MSE (active, raw scale) | * | * | * |
| MSE (inactive, raw scale) | * | * | * |
| MSE (wake ROI, raw scale) | * | * | * |
| Enstrophy relative error (full) | * | * | * |
| Enstrophy relative error (wake) | * | * | * |
| Circulation absolute error (wake) | * | * | * |
| Local FFT error (mean) | * | * | * |
| Radial spectrum L2 (wake ROI) | * | * | * |

The radial spectrum metric is the physics-grounded addition: compute
the 1D radial power spectrum of vorticity in the wake ROI for
prediction vs target, report L2 error across the wavenumber range.
This is a JFM-reviewer-relevant metric beyond SSIM.

Plus per-frame metrics for the canonical Test B encounter:

| Metric | Frame 25 | Frame 40 | Frame 55 |
|---|---|---|---|
| Wake MSE | * | * | * |
| Wake enstrophy err | * | * | * |

### Figure 3

The canonical Test B encounter G+1.00_D1.00_Y+0.10 encounter 00,
frames 25, 40, 55, fixed colorbar ±3, airfoil overlay.
4-column figure: target | Session 9 baseline (existing checkpoint) |
E1 LapFiLM | E2 LapFiLM+FFL. Plus a 5th column for E4 coord_mlp.

### Success criteria

A run is "successful" relative to the Session 9 baseline if it
improves at least two of:

- Test B SSIM: 0.358 → target >= 0.39
- Test B epsilon_vol: 0.978 → target <= 0.94
- Wake enstrophy relative error: reduce by >= 20 percent
- Wake ROI MSE: reduce by >= 20 percent

AND does not regress on:

- Inactive MSE (must not increase by more than 10 percent)
- Test A SSIM (must not decrease by more than 5 percent vs the
  Session 9 baseline 0.503)

The Test C metrics are reported but not gating; Test C is hard and
no improvement is expected at this stage.

## Decision outcomes after Step 7

The session's deliverable is the decision string mapping the three
experiments to Session 11 priorities.

```
Session 10 outcome: <one of>

  LAPFILM_WINS_ARCHITECTURE_LIMITED
                       - E1 and E2 both clear the success criteria
                         meaningfully beyond the Session 9 baseline.
                         The previous decoder was the bottleneck.
                         coord_mlp audit (E4) also improves but
                         less than E2; the CNN pyramid is sufficient.
                         Session 11 runs E3 (params_phase
                         conditioning) and E5 (frozen Fukami d=32
                         comparison) plus matched-d=32 Fukami AE.

  FFL_IS_KEY           - E1 marginal, E2 clearly better than E1.
                         Focal frequency loss is doing real work.
                         Session 11 sweeps FFL hyperparameters
                         (patch size, alpha, ramp schedule).

  COORD_MLP_BEST       - E4 outperforms E1 and E2 on wake metrics.
                         The JEPA latent has more information than
                         a CNN decoder can extract; coordinate
                         neural fields are the right reconstruction
                         tool. Session 11 explores SIREN
                         alternatives and chunked inference for
                         production use.

  ALL_THREE_PARTIAL    - E1, E2, E4 all show modest improvements
                         (~0.01-0.02 SSIM gain) but none clears
                         the success criteria fully. The wake-scale
                         information is partially in the latent but
                         hard to recover. Session 11 attacks the
                         encoder: retrain JEPA with a wake-region
                         observable head (CL was the dynamic target;
                         add wake enstrophy or wake spectrum as a
                         second observable).

  ALL_THREE_FAIL       - E1, E2, E4 all stay near the Session 9
                         baseline. The JEPA latent fundamentally
                         lacks wake-scale information. Session 11
                         pivots to encoder retraining with explicit
                         wake-region supervision; the visualisation
                         decoder cannot fix what the encoder did
                         not encode. The paper's reconstruction
                         claim narrows to "the JEPA decoder captures
                         gust core dynamics but not wake shedding."
```

## Out of scope for Session 10

- E3 (params_phase conditioning). Deferred to Session 11.
- E5 (frozen Fukami d=32 latent comparison). Deferred to Session 11.
- Matched-d=32 end-to-end Fukami AE baseline. Deferred to Session 11.
- bilinear_conv ablation. Conditional on Session 10 outcome; if
  pixelshuffle shows artifacts in Figure 3, Session 11 includes
  bilinear_conv as a remediation ablation.
- Hydra refactor, torch.compile, additional lambda sweeps.
- 3-seed averages on the new decoders. The Session 11 plan can
  include this once we know which decoder is the right one.

## Decisions to record (in HANDOFF.md)

**D70**: Session 10 dropped E0 (Fukami decoder MSE reproduction)
because the Session 9 checkpoint serves as the baseline. Deferred E3
(params_phase) and E5 (frozen Fukami d=32) to Session 11.

**D71**: Enstrophy and circulation losses implemented as spatial
field comparisons, not scalar mean comparisons. The bug fix from
the collaborator's proposal: a model with uniform noise of the
right total enstrophy would pass the scalar comparison but fail
the field comparison. The field comparison is the correct physics
constraint.

**D72**: FiLM use_film=False added as an architectural ablation
flag. The no_film variant lets the paper claim FiLM specifically
is the right conditioning mechanism rather than concatenation.

**D73**: Session 10 outcome decision string (one of the five
outcomes above). Determines Session 11 scope.

**D74-D77**: per-run results (E1, E2, E4, optional E_noFiLM).
Test A/B/C SSIM, epsilon, wake MSE, enstrophy error. Plus the
Figure 3 path.

## Expected duration

- Step 1 (LapFiLMDecoder + tests): 3 hours.
- Step 2 (decoder_losses + tests): 2 hours.
- Step 3 (CoordMLPDecoder + tests): 2 hours.
- Step 4 (entrypoint extension): 1 hour.
- Step 5 (figure pipeline): 30 min.
- Step 6 (three production runs): 3-4 hours wall-clock with two-card
  parallelism.
- Step 7 (evaluation suite + Figure 3): 2 hours.
- HANDOFF + session report: 1 hour.

Total wall-clock: 12-14 hours from start. Agent-active time:
8-10 hours.

Implementation work (Steps 1-5) takes the bulk of the agent-active
time. Steps 6-7 overlap GPU compute with analysis work.

## Pre-flight checks

1. Session 9 JEPA encoder checkpoint exists at
   `outputs/runs/session9/run_jepa_pipeline_lam0p01_seed42`.
2. Omega pipeline manifest exists at
   `outputs/data_pipeline/v1/manifest.json`.
3. Airfoil adjacent mask exists at
   `outputs/data_pipeline/v1/airfoil_adjacent_mask.npy` for the
   optional mask channel.
4. Both RTX 6000 Blackwell cards visible at cuda:2 and cuda:3.
5. All Sessions 2-9 unit tests pass before any new code lands.

## Decision references

- D5, D17: SIGReg with BatchNorm projection.
- D34, D35: frame-skip 1, partition v1.2.
- D38, D40: two RTX 6000 cards.
- D44-D49: Session 7 full-scale evaluation and R3_WINS.
- D50-D57: Session 8 validation grid sweep.
- D60-D69: Session 9 omega pipeline + decoder retrain (approximate;
  the actual D-numbers may differ).
- D70-D77: this session.
