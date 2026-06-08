# JEPA architecture auxiliary: conditioned and unconditioned models

Full architectural specification of the two JEPA model families compared in this
work: the conditioned production JEPA (the paper baseline, gust parameters
c = (G, D, Y) fed to the predictor) and the unconditioned JEPA (no c anywhere).
The unconditioned family has two predictor variants, transformer (tf-no-c) and
recurrent (lstm-no-c). All three share the same encoder, the same visualisation
decoder design, and the same loss and training recipe; they differ only in the
predictor and its conditioning.

All quantities below are read from the actual checkpoints (parameter counts are
exact state-dict tensor counts), not from the design spec.

## 1. Shared encoder (unconditional in every model)

The encoder is a pure state map omega(t) -> z. The gust never enters it in any
model; conditioning, when present, is confined to the predictor.

- Input: mid-plane spanwise vorticity omega_z at the native cache resolution
  (192, 96), single channel (the spanwise component of /curlU at the mid-span
  plane), pipeline-normalised (3-sigma scaling, no mean shift).
- Hybrid CNN stem: three downsampling stages producing a 24 x 12 feature map at
  256 channels, i.e. 288 spatial tokens.
- Vision transformer: 6 layers, hidden width 256, 8 attention heads, operating on
  the 288 tokens plus a learned [CLS] token.
- Latent projection: the [CLS] token is mapped by a single-layer MLP with
  BatchNorm (not LayerNorm; the SIGReg anti-collapse term requires BatchNorm at
  the latent boundary) to the latent z.
- Latent dimension: d = 64 (a d = 32 variant exists with identical structure).
- Parameter count: 6.68 M (identical across all three models; the unconditioned
  encoders are independent end-to-end retrains, same architecture).

## 2. Predictor (the architectural difference)

The predictor advances the latent autoregressively, z(t) -> z_hat(t+1), over a
context window of T = 32 frames. Three configurations are used.

### 2a. Conditioned transformer (production baseline)
- AutoregressivePredictor: 6-layer causal transformer, hidden width 384, 16
  attention heads, dropout 0.1.
- RoPE temporal positional encoding on Q, K; causal attention mask (each step
  sees only past steps).
- Conditioning: AdaLN-Zero on the static gust descriptor c = (G, D, Y). The
  conditioning vector rescales the transformer's internal (affine-free) LayerNorms
  through two AdaLN modules per block, initialised so the gate is zero (identity
  on the residual at initialisation, so conditioning perturbs training gently).
  cond_dim = 3.
- Parameter count: 16.15 M (the AdaLN conditioning modules account for the
  difference versus the unconditioned transformer below).

### 2b. Unconditioned transformer (tf-no-c)
- The same AutoregressivePredictor: 6 layers, hidden 384, 16 heads, dropout 0.1,
  RoPE, causal mask.
- cond_dim = 0: there is no conditioning vector, so the AdaLN-Zero path collapses
  to the identity (the gate stays zero forever) and the gust never reaches the
  predictor. Output head is a plain Linear (no output BatchNorm).
- Parameter count: 10.69 M.

### 2c. Unconditioned recurrent (lstm-no-c)
- LSTMLatentPredictor: a 3-layer LSTM over the latent sequence, hidden width 256,
  dropout 0.1, predicting the next latent from the causal history; plain Linear
  output head (no output BatchNorm).
- cond_dim = 0: the gust is never seen.
- Parameter count: 1.40 M (about 12x smaller than the conditioned transformer
  predictor, yet the stronger in-envelope forecaster on test_b).

At inference the unconditioned models are fully self-contained: the encoder maps
omega(t) -> z, the model's own predictor rolls z forward with no gust input
(cond = zeros of width 0), seeded either from the single impact-frame latent
(markov) or from the observed pre-impact history window (full-context). The
predictor is rolled on the raw encoder output (the BatchNorm-normalised latent),
with no separate z-scoring.

## 3. Visualisation decoder (separate stage, frozen encoder, never in the JEPA loss)

The decoder exists only to turn latents into fields for figures and animations.
It is trained after the JEPA encoder is frozen and is never part of the JEPA
objective.

- LapFiLMDecoder: a 5-level Laplacian pyramid with FiLM modulation and pixelshuffle
  upsampling, decoding z -> omega_z field (192, 96) in normalised space.
- Parameter count: 0.91 M.
- Training loss: region + Laplacian-pyramid + enstrophy + circulation + gradient +
  spectral-amplitude (the region_pyr_specloss recipe), 30k iterations on the frozen
  encoder, train split only.
- Each model has its own decoder trained on its own frozen encoder (the production
  decoder for the conditioned model; decoder_noc_tf and decoder_noc_lstm for the
  unconditioned models). Held-out reconstruction SSIM lands near 0.73 on the
  validation split for all three.

## 4. Loss and training recipe (shared)

The JEPA objective lives entirely in latent space:

  L = L_pred + 0.5 * L_roll + lambda * SIGReg(Z)

- L_pred: teacher-forced one-step latent prediction error, target
  z(t+1) = Encoder(omega(t+1)). There is no EMA and no stop-gradient on the target
  encoder; gradients flow through the target.
- L_roll: scheduled-sampling open-loop rollout error over H_roll = 8 steps (the
  model is fed its own predictions so it tolerates its own error).
- SIGReg(Z): the LeWM Epps-Pulley anti-collapse regulariser, M = 256 random
  projections, weight lambda = 0.01. This is the only thing preventing latent
  collapse (no decoder, no stop-gradient), which is why a latent-space predictor
  with no anti-collapse term would degenerate to a constant.
- Auxiliary observable heads read z during training with small weights: a
  future-C_L head (weight 0.01) and a wake-observable head (patch_signed_spectrum,
  weight 1.0). These are heads on z, not reconstruction of the field.
- Optimiser: AdamW (0.9, 0.95), weight decay 0.05, 5% linear warmup then cosine to
  0.05x peak. Encoder learning rate 1.5e-4, predictor learning rate 5e-4. bf16
  mixed precision, gradient clip 1.0.
- Iterations: 20k for the production conditioned encoder (checkpoint iter020000)
  and for both unconditioned retrains, which use the identical recipe with
  cond_dim = 0 (and predictor_type transformer or lstm).

## 5. Conditioning: the one controlled difference

- Conditioned (production): c = (G, D, Y) enters only the predictor, via AdaLN-Zero.
  The encoder is unconditional. The model is a conditional forward-closure model:
  in deployment it would require estimating or marginalising over c.
- Unconditioned (tf-no-c, lstm-no-c): c enters nothing. The predictor advances the
  latent from its own history alone. The model is fully self-contained at inference
  and needs no gust parameters.

tf-no-c is the cleaner controlled ablation: it differs from the conditioned model
in exactly one variable (conditioning removed, same transformer), so its figures
read as a direct test of whether conditioning is load-bearing. lstm-no-c changes
two things at once (predictor architecture and conditioning) but is the stronger
in-envelope forecaster and is far cheaper (1.40 M predictor).

## 6. Summary

| component | conditioned (prod) | tf-no-c | lstm-no-c |
|---|---|---|---|
| encoder | HybridCNNViT, 6.68 M (unconditional) | same | same |
| predictor | AR transformer 384/16/6 + AdaLN(c), 16.15 M | AR transformer 384/16/6, cond_dim 0, 10.69 M | LSTM 256/3, cond_dim 0, 1.40 M |
| conditioning | c=(G,D,Y) -> predictor (AdaLN-Zero) | none | none |
| decoder | LapFiLM 0.91 M (frozen-encoder stage) | LapFiLM 0.91 M | LapFiLM 0.91 M |
| latent d | 64 | 64 | 64 |
| context T | 32 (RoPE) | 32 (RoPE) | 32 (windowed history) |
| loss | L_pred + 0.5 L_roll + 0.01 SIGReg | same | same |
| iterations | 20k | 20k | 20k |

## 7. Checkpoint provenance

- conditioned: outputs/runs/session12/S12_E_d64/encoder/checkpoint_iter020000.pt
  (+ decoder_specloss_recipe/decoder_iter030000.pt)
- tf-no-c: outputs/session27/JEPA_d64_noc_tf/checkpoint_iter020000.pt
  (+ outputs/session27/decoder_noc_tf/decoder_iter030000.pt)
- lstm-no-c: outputs/session27/JEPA_d64_noc_lstm/checkpoint_iter020000.pt
  (+ outputs/session27/decoder_noc_lstm/decoder_iter030000.pt)

The encoder/predictor live under the "jepa_state_dict" key (prefixes "encoder."
and "predictor."); the predictor configuration is recorded in the checkpoint
"args" (predictor_type, predictor_cond_dim, d, T).
