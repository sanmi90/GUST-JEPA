# Faithful V-JEPA on the gust data: design spec (variant B)

Date: 2026-06-19
Status: design approved in chat 2026-06-19; awaiting plan
Branch: continue on `spatiotemporal-jepa` (or a fresh `vjepa` branch)

## Context

Variant A (causal spatio-temporal encoder inside our autoregressive-JEPA framing)
was a clean null on all six metrics (see
`2026-06-19-spatiotemporal-jepa-findings.md`): spatio-temporal *encoding* does
not help. Variant B tests the other axis the user wants to see regardless of A:
does the V-JEPA *objective* (masked space-time feature prediction with an EMA
target) beat our autoregressive-rollout objective? Expected payoff is low given
A's null; this is an explicit user-requested exploration ("just for fun", full
faithful). Reference: Bardes et al., "Revisiting Feature Prediction for Learning
Visual Representations from Video", arXiv:2404.08471, 2024.

## What is faithful vs adapted

Faithful: tubelet tokenizer, multi-block 3D masking, EMA target encoder with
stop-gradient, mask-infilling predictor, smooth-L1 masked-feature loss, no
SIGReg (EMA+masking is the anti-collapse).

Adapted (so the result lands in our comparison table): encoder sized to our
~10-12M budget (not ViT-L); a d=64 per-frame latent obtained by attentive pooling
for evaluation; our split v2.1, omega pipeline, and the six-metric suite.

## Architecture

### Tokenizer
3D-conv tubelet embed of `(B, T=32, 1, 192, 96)`. Tubelet `(2, 16, 16)` (t,h,w),
stride = tubelet size, so the token grid is `(T/2, 192/16, 96/16) = (16, 12, 6) =
1152 tokens`, each embedded to `hidden=384`. Add fixed 3D sin-cos positional
embeddings (factorized t,h,w).

### Context and target encoders
ViT: depth 8, hidden 384, 6 heads, MLP ratio 4, no [CLS]. The **context
encoder** processes only the visible (unmasked) tokens. The **target encoder** is
an EMA copy (cosine momentum 0.996 -> 1.0 over training) that processes the FULL
token set; its outputs are the prediction targets with stop-gradient. LayerNorm
the target features per token before the loss (V-JEPA normalises targets).

### Masking (multi-block 3D, per clip)
Sample blocks on the (16,12,6) grid: 2 long-range spatial blocks spanning all 16
temporal positions (each ~ (16, large-h, large-w)) + 4 short-range blocks (few
frames, smaller spatial), union ~75-85% of tokens masked. Context = complement
(>= ~15% visible). Masks differ per sample in a batch.

### Predictor (mask-infilling)
Narrow ViT: depth 6, hidden 192. Input = projected context-encoder tokens +
learnable mask tokens placed at the masked positions, each carrying the target
position's 3D positional embedding. Output projected back to 384; smooth-L1
(beta 0.5) to the EMA-target features at masked positions only.

### Training loop (new entrypoint `src/training/train_vjepa.py`)
AdamW (0.9,0.95), wd 0.05, encoder lr 1.5e-4, predictor lr 5e-4, cosine schedule
+ 5% warmup, grad-clip 1.0, bf16, 20k iters (convergence-matched; verify the loss
plateau, do NOT pad to a round number). EMA update every step. RTX 6000 only via
require_rtx6000. CPU-capped. W&B logs the four required keys + gpu_name + a
`collapse` diagnostic (token-feature std / rank) since there is no SIGReg.

## Eval adapter (the bridge to the six-metric suite)
V-JEPA yields per-token features, not a per-frame d-dim latent or a rollout. To
score it apples-to-apples at d=64:
1. Freeze the context encoder; encode a clip; reshape tokens back to
   `(T/2, 12, 6, 384)`.
2. Per output-frame, **attentive-pool** the spatial tokens (learned query, or
   mean-pool as a fallback) and a small head **project to d=64**; upsample the
   T/2 temporal axis to 120 frames (nearest/linear) to match the per-frame latent
   convention. This pooling head is fit briefly on train (frozen encoder) to a
   light proxy target, or is a fixed mean-pool+PCA-to-64; choose the simplest
   that yields a usable per-frame `z_t`. (Plan task: pick and justify.)
3. Run the ESTABLISHED protocol on `z_t`: a separately-trained **matched**
   predictor (V-JEPA has no co-trained rollout predictor; matched is the only
   apples-to-apples choice, identical to the AE-matched baseline) ->
   forecast/drift; the T9 decoder -> SSIM; impact-frame `z` -> G/D/Y probe; PR.

## Runs and comparison
d=64, seeds {0,1,2}, split v2.1. Compare on all six metrics to: per-frame JEPA-own
(+0.89->+0.61 forecast h1->h16; drift 1.60; SSIM ~0.50 test_b; Y 0.57), ST
(+0.82->+0.37; drift 1.90; Y 0.49), regAE-matched (+0.54->+0.37 forecast; SSIM
0.476). Report HONESTLY whether the V-JEPA objective helps.

## Implementation surface
- `src/models/vjepa_tokenizer.py` (3D tubelet embed + 3D pos-embed)
- `src/models/vjepa.py` (context/target ViT encoders, EMA, predictor, masked loss)
- `src/models/vjepa_masking.py` (multi-block 3D mask sampler)
- `src/training/train_vjepa.py` (training loop, EMA, W&B, collapse diagnostic)
- `src/models/vjepa_pool.py` (attentive-pool -> d=64 eval head)
- extend `scripts/session18/encode_baseline_latents.py` with a `--baseline vjepa`
  branch (encode + pool -> per-frame z_t in the standard npz format)
- a launcher `scripts/session29/vjepa_band.sh` (train -> extract -> matched
  predictor + roll -> decoder -> metrics), RTX 6000, CPU-capped, idempotent
- unit tests: tokenizer shape, mask coverage/complement, EMA updates &
  no-grad-to-target, predictor predicts only masked positions, encoder I/O
  contract, a 50-iter overfit-one-batch smoke (loss decreases).

## Risks
- Collapse without SIGReg if EMA momentum/masking are off: the collapse
  diagnostic + overfit-batch smoke guard this; fall back to higher mask ratio or
  target LayerNorm if token-feature rank collapses.
- The pooling-to-d64 eval head is a judgement call; a bad pooling could
  understate V-JEPA. Pick the simplest defensible head and state it.
- Multi-session build; cards are shared (stop the redundant ST decoders before
  training). Expected payoff low (A null) -- this is exploratory.

## Constraints (standing)
RTX 6000 Blackwell only; L40S forbidden (asolera); no CPU fallback; CPU-capped
(>=64 cores free); loss in 3-sigma normalised space; no em-dashes; numbers via
macros in any paper output; commit only when asked; do not push / do not merge
without the user; train to convergence not a fixed budget.
