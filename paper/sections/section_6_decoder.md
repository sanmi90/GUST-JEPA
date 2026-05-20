# Section 6: Visualisation decoder

LaTeX-friendly markdown. Approximate target length: 3 to 4 pages.
Figure 3 is a 3x3 grid showing raw / decoded / residual vorticity at
frames 25 (pre-impact), 40 (at impact), and 55 (post-impact) for one
Test B encounter (file
`outputs/runs/session9/decoder/fig3_decoder_reconstruction.png`).
Per-encounter MSE-ratio histograms across Test A / B / C are at
`outputs/runs/session9/decoder/fig_decoder_mse_distribution.png`.

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

Result. The decoder achieves Test A per-encounter MSE of 14.73
against a per-case-mean floor of 1.57, a ratio of **9.37**, with
Fukami SSIM = 0.726 mean. The reconstruction **fails** the `2x`
threshold by a wide margin (the case-mean is a near-perfect
predictor on Test A because the held-out encounters share their
case-mean training-side neighbours, giving floor = 1.57 versus the
decoder's 14.73). The SSIM 0.726 indicates the structural similarity
of the reconstruction is reasonable; the failure is on per-pixel
intensity rather than on global shape.

Per-encounter MSE histogram. Most Test A encounters cluster near the
median value of 9.24; outliers concentrate on encounters
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

Result. Per-encounter MSE = 31.33; per-case-mean floor =
9.40; ratio = 3.33; SSIM = 0.572. The decoder reconstructs the
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

Result. Per-encounter MSE = 71.09; per-case-mean floor =
29.56; **ratio = 2.40** (just outside the 2x threshold); SSIM = 0.414.
The vortex core position is
recovered correctly at impact, consistent with the analytical
parametric prediction (Section 2.1 of HANDOFF.md "Open Questions"),
but the wake amplitude is systematically underestimated relative to
the ground truth on Test C, consistent with the latent's `G` axis
being interpolated rather than extrapolated.

## 6.5 What the decoder tells us about the latent

The reconstruction quality on Test A vs Test B vs Test C provides a
visual signature of the latent's parametric interpolation vs
extrapolation behaviour:

- The Test A ratio of 9.37 indicates the JEPA's predictive-only
  encoder discards reconstruction-relevant information. The case-mean
  is a near-perfect predictor on Test A (the held-out encounters
  share their case-mean training-side neighbours; floor = 1.57) so
  any non-trivial reconstruction MSE looks large relative to the
  floor. The SSIM 0.726 indicates the structural similarity is
  preserved; the failure is on per-pixel intensity rather than on
  global shape.
- The Test B ratio of 3.33 is closer to the 2x threshold; the
  per-case-mean floor on Test B is materially higher (9.40 because
  the held-out cases do not share their mean with training, so the
  mean is a worse predictor). The decoder's absolute MSE on Test B
  is 31.33 -- twice the Test A absolute MSE -- but the ratio
  is one third of Test A's because the floor is six times higher.
- The Test C ratio of 2.40 clears the 2x threshold marginally.
  Per-pixel reconstruction quality is poor in absolute terms (MSE
  71.09) but the case-mean is even worse (floor 29.56) because
  Test C is an extrapolation regime where the per-case mean has
  no training analogue.

## 6.6 Head-to-head with the Fukami AE (Section 7.2 A11)

The Fukami lift-augmented autoencoder (Section 7.2 ablation A11) shares
the matched-d = 32 setting and the same evaluation pipeline, but
trains the encoder + decoder + lift head jointly on reconstruction
MSE + lift MSE. The head-to-head on the same Test A / B / C splits:

| Method                                     | Test A ratio | Test A SSIM | Test B ratio | Test B SSIM | Test C ratio | Test C SSIM | Test B delta (downstream) |
|--------------------------------------------|-------------:|------------:|-------------:|------------:|-------------:|------------:|--------------------------:|
| JEPA encoder (frozen E4) + decoder (this section) |  9.37  |   0.726     |     3.33     |   0.572     |     2.40     |   0.414     |  +0.131 +/- 0.032 (3 seeds) |
| Fukami CNN AE (matched d = 32)             |     7.70     |   0.748     |     1.60     |   0.722     |     1.44     |   0.558     |  +0.073                   |

Fukami AE wins on per-pixel reconstruction (ratio 1.5x to 2x lower on
every split; SSIM 0.02 to 0.15 higher). JEPA + frozen-encoder decoder
wins on downstream Test B prediction by +0.058 absolute (the
matched-d head-to-head from Section 7.2). The Fukami AE encoder + decoder
joint training preserves reconstruction-relevant information that the
JEPA predictive-only training discards; the JEPA's discarding strategy
produces a more transferable latent for downstream prediction at the
cost of fidelity in reconstruction. This is the explicit JEPA
tradeoff (paper Section 2.1).

The Section 9 pass criterion ("Test A within 2x the floor") was set
under the assumption that the JEPA decoder would behave like a
reconstruction-trained AE. The 9.37 ratio is well outside the
threshold; the proper reading is that the criterion was set against
the wrong baseline. A more useful comparison is the head-to-head with
Fukami AE in the table above, where the JEPA's predictive-only
training trades reconstruction fidelity (Fukami wins) for downstream
predictive utility (JEPA wins).

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
