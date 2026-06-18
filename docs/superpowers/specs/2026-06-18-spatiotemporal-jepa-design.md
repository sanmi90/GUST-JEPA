# Spatio-temporal JEPA encoder: design spec

Date: 2026-06-18
Status: design, awaiting user review before implementation plan
Owner: Carlos Sanmiguel Vila

## Question

Does making the JEPA *encoder* temporal help? Today the encoder is per-frame
(`HybridCNNViTEncoder.forward`, `src/models/encoder.py:214` flattens
`(B, T, 1, H, W)` to `(B*T, 1, H, W)` and encodes every frame independently).
All temporal structure lives in the separate autoregressive predictor. So each
latent `z_t` is a snapshot: it cannot distinguish a vortex moving up through a
point from one moving down through it, because it never sees a neighbouring
frame. This spec tests whether a latent that integrates a short causal motion
window forecasts, reconstructs, and disentangles better.

This is variant A of a staged plan. If A beats the per-frame JEPA on the metric
suite below, we escalate to B (full V-JEPA: tubelet tokenizer + masked
space-time feature objective, reusing A's stem as the tokenizer). If A is a
clean null, we stop and report that temporal encoding does not help inside the
autoregressive-JEPA framing, which is itself an honest, publishable result.

## The one architectural change

Replace the per-frame 2D conv stem with a **causal 3D-conv tubelet stem**:

- Input `(B, T, 1, H, W)` is left-padded in time by replication of frame 0.
- 3D convs (temporal kernel 3, temporal stride 1, spatial stride 2), stacked 3
  deep, downsample space (192x96 -> 24x12) while preserving one feature-frame
  per input frame. Effective causal temporal receptive field ~7 frames
  (~0.35 t/c at dt_tc = 0.05).
- Output is per-frame spatio-temporal feature maps `(B, T, c3, 24, 12)`.
- The existing per-frame ViT + `[CLS]` readout + BatchNorm projection run
  unchanged on top.

The module honours the exact `(B, T, 1, H, W) -> (B, T, d)` contract, so it
drops into the existing training loop, predictor, and eval with no other
changes. Each `z_t` now integrates frames `<= t` instead of frame `t` alone.

### Why causal, not centered

A centered window lets `z_t` peek at future frames, which leaks future
information into the predictor's teacher-forcing targets and inflates the
forecast metric. Causal keeps the comparison honest: `z_t` depends only on
frames `<= t`, the predictor and the entire forecast/SSIM/drift/probe eval are
identical to the per-frame JEPA, and the encoder is used only to produce
initial/target latents, never inside the rollout.

### Safety rail

A unit test asserts causality: perturbing frame `t+1` must not change `z_t`.
This catches the single most likely bug, an off-by-one in the causal pad that
leaks the future. Also a shape-contract test on `(B, T, 1, 192, 96)` inputs.

## What is held fixed (apples-to-apples)

Same predictor (autoregressive, RoPE, causal), same recipe (SIGReg 0.01, lift
0.01, wake 1.0 patch_signed_spectrum, `H_roll` 8), `--predictor-cond-dim 0`,
split v2.1, omega pipeline manifest `outputs/data_pipeline/v2p1/manifest.json`,
20k iterations, same SSIM decoder recipe, same forecast probe. The only moving
part is the encoder stem.

### Iteration count is convergence-driven, not a fixed budget

20k matches the existing per-frame band seeds (which only ever trained to 20k:
checkpoints at 10k and 20k), and 20k is verified at or past convergence for this
recipe: the regAE/JEPA training loss is flat from roughly iter 4k onward (batch
noise aside). So 20k is both the fair comparison point and not underfit. We do
NOT extend to 80k for a headline number. The only thing that would justify more
iterations is a 3D-conv ST encoder whose convergence diagnostic (loss + probe
R^2 + participation ratio) is still trending upward at 20k, and then only as far
as the plateau, never a fixed budget. Underfitting is decided from the curve,
not assumed away; overtraining for its own sake is not done.

## Runs

- ST encoder via `src/training/train_jepa.py --encoder st_hybrid`, at
  **d = 64 and d = 16**, seeds {0, 1, 2, 42}.
- Baselines already on disk:
  `outputs/runs/session28/jepa_tf_noc_d64_s{0,1,2,42}/encoder/checkpoint_iter020000.pt`
  and `jepa_tf_noc_d16_s{0,1,42}`.

d = 16 is the pointed test. The per-frame JEPA was weak and erratic at d = 16
(the measured reversal: regAE-matched +0.78[0.69,0.85] vs JEPA-own
+0.30[0.07,0.45] at h = 1). Hypothesis: a snapshot encoder spends its tiny
budget on instantaneous structure and has nothing left for dynamics; a
spatio-temporal latent should help most exactly where the per-frame model
struggles.

## Evaluation suite (the full set, every checkpoint feeds all six)

1. Forecast band: wake-enstrophy R^2 at h = 1,2,4,8,12,16, JEPA-own predictor
   rollout, seed band (`scripts/session29/m_seed_forecast_band.py`,
   `roll_own_predictor.py`).
2. SSIM: T9 SL decoder on the frozen ST encoder, test_a/test_b/test_c, windowed
   Wang (K1 0.01, K2 0.03, L = 8.45) (`scripts/session29/dec_posthoc_launch.sh`).
3. Drift: on-manifold Mahalanobis ratio + rel-l2 of the rolled latent
   (`scripts/session29/m2_drift_nowake.py`).
4. Parameter probe: CV-honest KernelRidge(RBF) R^2 for (G, D, Y) from
   impact-frame z, test_b, bootstrap CI.
5. Participation ratio + near-null-dim count (ties to the d16<->d64 story).
6. Instantaneous wake readability: ridge z->wake R^2 at the impact frame.

## Decision gate (A -> B)

Escalate to the full V-JEPA build B if the ST encoder beats the per-frame JEPA
on the forecast band at either d by a margin outside the seed bands, OR
materially improves drift or SSIM. Otherwise stop and report A as a clean null.

## Implementation surface

- New `SpatioTemporalCNNViTEncoder` in `src/models/encoder.py` (3D-conv stem,
  reuses `_ViTBlock`, per-frame `[CLS]` readout, BatchNorm projection).
- Wire `--encoder st_hybrid` into `src/training/train_jepa.py` (the existing
  family switch at lines 174 / 727-731), with a `--temporal-kernel` /
  `--temporal-depth` knob.
- Unit tests in `tests/`: causality + shape contract.
- A launcher script under `scripts/session29/` analogous to `d16_then_d32.sh`:
  train ST encoders (both cards, 2-packed, CPU-capped, RTX 6000 only) ->
  extract latents -> JEPA-own rollouts + matched-predictor rollouts -> decoders
  -> run the six-metric suite and emit a comparison table vs the per-frame band.

## Constraints (standing, non-negotiable)

RTX 6000 Blackwell only (`require_rtx6000`); L40S forbidden (asolera runs SOD2D
there); no CPU fallback; CPU-capped with taskset so >= 64 cores stay free; W&B
logs the four required keys + gpu_name; loss in 3-sigma normalised space; no
em-dashes anywhere; numbers via macros in any paper-facing output. ST runs queue
behind the in-flight d32 band (both cards currently busy).

## Risks

- 3D conv raises encoder FLOPs/memory; mitigate with temporal stride 1 but small
  temporal kernel and the same spatial downsampling schedule.
- At d = 16 a richer encoder could collapse harder under SIGReg; the PR/null-dim
  diagnostic and the auto-fallback rule already guard this.
- Causal-pad off-by-one (future leak); the causality unit test is the rail.
