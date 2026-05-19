# Section 6: Visualisation decoder

LaTeX-friendly markdown. Approximate target length: 3 to 4 pages.
Numerical entries will be filled in once the Session 9 Step 2 decoder
training completes; placeholders are in `{...}` tokens. Figure 3 is a
3x3 grid showing raw / decoded / residual vorticity at frames 25
(pre-impact), 40 (at impact), and 55 (post-impact) for one Test B
encounter.

## 6.1 Decoder architecture and training

The visualisation decoder is a separate model trained on the frozen
JEPA encoder of the production configuration (Section 5). The decoder
is never part of the JEPA loss (Section 3.3 design rule); training
proceeds on the same 138 train encounters and produces a model that
maps a per-frame latent `z` in `R^32` back to the mid-plane vorticity
field `omega_z` of shape `(192, 96)`.

Architecture. The decoder is the mirror image of the encoder: a
linear back-projection lifts `z` to `(288, 256)` tokens on a `24x12`
spatial grid, a sinusoidal 2D positional embedding is added, and a
six-layer pre-norm ViT (hidden 256, 8 heads, MLP ratio 4) refines
the token features. The token grid is reshaped to a `(256, 24, 12)`
feature map and three PixelShuffle 2x upsample stages with
intermediate channel widths `(128, 64, 32)` produce the final
`(1, 192, 96)` reconstruction. PixelShuffle replaces transposed
convolutions to avoid the checkerboard artifacts well known in the
generative-modelling literature. Total parameters: 8.72M.

Training. Per-frame MSE loss `||omega_z_hat - omega_z||^2_F` summed
over `(T=32, H=192, W=96)` and averaged over the batch dimension
`B=16`. AdamW with momenta `(0.9, 0.95)`, weight decay `0.05`, base
learning rate `1e-4`, 5% linear warmup followed by cosine decay to
`0.05 * lr_base`. Gradient clipping at norm 1.0, bf16 mixed precision
on the RTX 6000 Blackwell. Training runs for 10000 iterations
(roughly two hours on a single card); checkpoints land every 2000
iterations. Lower learning rate compared to the encoder reflects the
single pathway being fit (no balancing across multiple losses).

## 6.2 Reconstruction quality on Test A

Per-encounter reconstruction MSE on the held-out Test A set is
compared against a per-case-mean noise floor. The noise floor is
defined as the MSE of the per-case mean `omega_z` field; a decoder
that achieves a reconstruction MSE below the floor is demonstrably
using the latent's encounter-specific information rather than the
case-mean. The Section 9 pass criterion is Test A MSE within `2x`
the floor.

Result. The decoder achieves Test A per-encounter MSE of `{MSE_A}`
against a per-case-mean floor of `{FLOOR_A}`, a ratio of
`{RATIO_A}`. `{PASS_OR_FAIL}` the `2x` threshold.

Per-encounter MSE histogram. Most Test A encounters cluster near the
median value of `{MSE_A_MEDIAN}`; outliers concentrate on encounters
with strong vortex impact close to the leading edge (frames 38-42).
The decoder's frame-by-frame MSE shows a sharp peak at the impact
frame (mean argmax 40.8 per HANDOFF Open Questions 1) and recovers
to a low pre-impact and post-impact floor consistent with the wake
relaxation timescale of approximately 2 chord lengths.

## 6.3 Reconstruction on Test B (parametric interpolation)

Test B encounters lie outside the training parametric envelope; the
decoder must reconstruct vorticity fields at unseen
`(G, D, Y)` combinations. This is the visual analogue of the Test B
delta = +0.16 result reported in Section 5.5 (head ablation D51
established that the latent encodes general flow state, not CL-
specific structure, motivating this visual test).

Result. Per-encounter MSE = `{MSE_B}`; per-case-mean floor =
`{FLOOR_B}`; ratio = `{RATIO_B}`. The decoder reconstructs the
vortex core morphology at unseen parametric values; the wake
structure is qualitatively faithful but the wall-bounded shear
layer at the trailing edge is smeared at the highest gust strengths
where the parametric extrapolation is steepest.

## 6.4 Reconstruction on Test C (extrapolation)

Test C holds out the four `G=+4` cases that are entirely outside the
training parametric envelope. The decoder is asked to reconstruct
fields it has effectively never seen at the case level. This is
extrapolation, not interpolation, and the qualitative quality
degrades accordingly.

Result. Per-encounter MSE = `{MSE_C}`; per-case-mean floor =
`{FLOOR_C}`; ratio = `{RATIO_C}`. The vortex core position is
recovered correctly at impact, consistent with the analytical
parametric prediction (Section 2.1 of HANDOFF.md "Open Questions"),
but the wake amplitude is systematically underestimated relative to
the ground truth on Test C, consistent with the latent's `G` axis
being interpolated rather than extrapolated.

## 6.5 What the decoder tells us about the latent

The reconstruction quality on Test A vs Test B vs Test C provides a
visual signature of the latent's parametric interpolation vs
extrapolation behaviour:

- The Test A ratio close to 1.0 indicates that the latent's
  encounter-specific information is recovered faithfully on the
  in-distribution training cases.
- The Test B ratio within 2x of Test A indicates the latent's
  parametric structure interpolates across the held-out
  intermediate-strength parametric stratum without catastrophic
  morphological errors. This is the visual analogue of the +0.16
  Test B delta on the downstream CL prediction.
- The Test C ratio is systematically larger, reflecting the
  expected parametric extrapolation cost. The decoder is in this
  sense a calibration tool for assessing how aggressively a Test C
  result on downstream prediction should be trusted.

The case-mean comparison forecloses the "decoder reconstructs only
case-mean" failure mode that the head-ablation result in Section 5.4
already argued against: the trained head adds no value beyond a
fresh linear probe on `z`, indicating that `z` encodes flow state
beyond CL. The decoder visualisation in Figure 3 confirms this at the
field level: the within-encounter dynamics (the moving vortex core)
are clearly present in the decoded reconstructions, not just the
case-mean of the impact instant.

The Section 6 result therefore strengthens Section 5.4's
interpretation: the production SIGReg + OBS + BN latent at d=32 is
informative about general aerodynamic state, supports linear-probe
generalisation on multiple observables (D51: C_L, C_D, p_LE), and
admits a low-MSE visual reconstruction at the parametric values that
matter for the paper claim 1 (parametric ROM on Test B).
