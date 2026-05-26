# Session 18 scripts

This directory implements Experiment B1 from the Session 18 plan:
Fukami AE and POD baseline comparison on physical Markov closure, with a
common transformer predictor trained on top of each baseline's latents
under the locked B1 fairness protocol.

Read first:
- `SESSION18 PLAN.md` (repo root) for the session-wide plan.
- `SESSION18_B1_PROTOCOL.md` (repo root) for the locked fairness rules.

The scripts intentionally encode the protocol into the entrypoints so
the launch sequence below is mechanical. Any deviation is a deliberate
choice that should land in HANDOFF.md.

## Launch sequence

```bash
# Required at the top of every shell session.
source .venv/bin/activate
export PREVENT_ROOT=$HOME/PREVENT
export WANDB_PROJECT=vortex-jepa

# Always confirm we are on RTX 6000 cards.
python -c "from src.utils.device import require_rtx6000; print(require_rtx6000(gpu_index=0))"
```

### 1. Fukami AE training (B1 Part a)

All three d values are trained with the SAME recipe (omega_pipeline +
MSE + lambda_lift=0.05 + ReLU + GroupNorm + observable deltas
(8,16,24)). See `SESSION18_B1_PROTOCOL.md` Fukami AE section for the
locked recipe. The driver `train_fukami_baselines.sh` originally used
Charbonnier; for B1 fairness use the explicit invocations below
(`recon-loss-type mse`, `lambda-lift 0.05`).

```bash
# Single card, sequential:
for d in 3 32 64; do
    python scripts/session9_train_fukami.py \
        --gpu 0 --partition v1 --all-train \
        --max-iters 20000 --seed 0 \
        --d $d \
        --B 16 --T 32 \
        --observable-head cl_future --observable-head-deltas 8 16 24 \
        --observable-head-weight 1.0 \
        --lambda-recon 1.0 --lambda-lift 0.05 \
        --omega-pipeline-manifest outputs/data_pipeline/v1/manifest.json \
        --recon-loss-type mse \
        --activation relu \
        --lr 1e-3 --weight-decay 0.0 --warmup-frac 0.05 --grad-clip 1.0 \
        --num-workers 4 \
        --tag-suffix session18_b1_fukami_d${d}_unified \
        --wandb-mode offline \
        --output-dir outputs/session18/exp_b1/fukami_ae_d${d}
done

# Two cards, parallel pair (d=3 + d=32 on GPU 1, d=64 on GPU 0):
# Same invocation, vary --gpu and --d, launch in background.
```

After each d completes, run the verification gate:

```bash
python scripts/session18/verify_fukami_gate.py \
    --eval-json outputs/session18/exp_b1/fukami_ae_d${d}/final_eval.json --d $d
```

The gate passes when Test A SSIM_mean >= 0.60 OR Test A ratio_mean < 2.0.

Two pre-existing checkpoints (Session 9 d=3 beta005; Session 11 D4 d=32)
were rejected for B1 because their recipes deviated from the unified
protocol (Session 9 d=3 had no pipeline; Session 11 d=32 used
lambda_lift=1.0).

### 2. POD basis computation (B1 Part b, ~1h CPU)

```bash
bash scripts/session18/compute_pod_baselines.sh "16 32 64"
```

Closed-form snapshot SVD on pipeline-normalised train frames. Writes
`pod_basis.npz` and `pod_summary.json` per d.

### 3. Encode baseline latents (B1 Part c prep, ~10 min each)

```bash
# Fukami AE
for d in 3 32 64; do
  python scripts/session18/encode_baseline_latents.py \
    --baseline fukami --d $d \
    --checkpoint outputs/session18/exp_b1/fukami_ae_d${d}/checkpoint_iter020000.pt
done

# POD
for d in 16 32 64; do
  python scripts/session18/encode_baseline_latents.py \
    --baseline pod --d $d \
    --basis outputs/session18/exp_b1/pod_d${d}/pod_basis.npz
done

# JEPA (reuse the production E d=64 encoder)
python scripts/session18/encode_baseline_latents.py \
    --baseline jepa --d 64 \
    --checkpoint outputs/runs/session12/S12_E_d64/encoder/checkpoint_iter020000.pt
```

Outputs land in `outputs/session18/exp_b1/latents_{baseline}_d{d}/{train,test_a,test_b,test_c}.npz`.

The JEPA d=64 step can be skipped if reusing the pre-extracted latents
already at `outputs/session14/latents/S12_E_d64/`; just symlink them
into `outputs/session18/exp_b1/latents_jepa_d64/`.

### 4. Train transformer predictor on each baseline (B1 Part c, ~3h each)

```bash
# Six predictor trainings (3 Fukami + 3 POD). JEPA predictor reuses the
# jointly-trained predictor inside the production JEPA checkpoint.
for d in 3 32 64; do
  python scripts/session18/train_baseline_predictor.py \
    --latents-dir outputs/session18/exp_b1/latents_fukami_d${d} \
    --tag fukami_d${d} --gpu 0
done

for d in 16 32 64; do
  python scripts/session18/train_baseline_predictor.py \
    --latents-dir outputs/session18/exp_b1/latents_pod_d${d} \
    --tag pod_d${d} --gpu 0
done
```

Each predictor uses identical recipe: AdamW lr=5e-4 weight_decay=0.05
betas=(0.9, 0.95), 20K iters, B=16, T=32, H_roll=8, hidden_dim=384,
depth=6, heads=16, dropout=0.1, AdaLN-Zero on (G, D, Y), RoPE on Q/K.
No per-baseline knob varies (CLAUDE.md "Locked decisions", B1 protocol).

### 5. Rollouts on Test B / Test C (B1 Part d step 1, ~20 min each)

```bash
for tag in fukami_d3 fukami_d32 fukami_d64 pod_d16 pod_d32 pod_d64; do
  python scripts/session18/eval_baseline_rollouts.py \
    --latents-dir outputs/session18/exp_b1/latents_${tag} \
    --predictor outputs/session18/exp_b1/predictor_${tag}/checkpoint_iter020000.pt \
    --tag ${tag}
done
```

Saves `outputs/session18/exp_b1/rollouts_{tag}/{test_b,test_c}.npz` with
`z_dns`, `z_markov`, `z_full`, `G`, `D`, `Y`, `case_ids`,
`encounter_indices`, `impact_frame`.

### 6. Physical metrics from rollouts (B1 Part d step 2, NOT YET BUILT)

The next-session task is `physical_metrics_from_rollouts.py`:

1. Load `outputs/session17/exp2/dns_physical_metrics.npz` (already exists).
2. For each baseline, fit z->observable probes on train per-frame data
   (z_full from `latents_{tag}/train.npz`, observables from DNS metrics).
   Probe choice: linear ridge (Session 17 D124 baseline) and MLP-reg
   (Session 17 D127 best).
3. Apply probes to `rollouts_{tag}/{test_b,test_c}.npz`'s z_markov and
   z_full to get predicted C_L, I_y^w, wake enstrophy, lambda ratio
   per encounter.
4. Compute absolute error vs DNS at H=8, 16, 32.
5. Bootstrap 2000-resample 95% CI per baseline x metric x split x horizon.
6. Write headline 7x4 table to `physical_closure_comparison.csv` and
   produce `figures/exp_b1_markov_closure_baselines.png`.

The JEPA row in the table reuses Session 17 exp2 results without
recomputing.

## Two-card parallel pattern

The workstation has two RTX 6000 Blackwell cards. Use them for parallel
fan-out (D40):

```bash
# d=3 + d=32 in parallel:
bash scripts/session18/train_fukami_baselines.sh "3"  0 &
bash scripts/session18/train_fukami_baselines.sh "32" 1 &
wait

# Predictor on Fukami d=64 in parallel with POD d=32:
python scripts/session18/train_baseline_predictor.py \
    --latents-dir outputs/session18/exp_b1/latents_fukami_d64 \
    --tag fukami_d64 --gpu 0 &
python scripts/session18/train_baseline_predictor.py \
    --latents-dir outputs/session18/exp_b1/latents_pod_d32 \
    --tag pod_d32 --gpu 1 &
wait
```

Do NOT use shell-level `CUDA_VISIBLE_DEVICES` to choose between the two
RTX 6000s; the `--gpu` flag plus `require_rtx6000(gpu_index=N)` handle
device selection correctly and the W&B device.index reflects the choice.

## Outputs tree

```
outputs/session18/exp_b1/
  fukami_ae_d{3,32,64}/
    checkpoint_iter{004000..020000}.pt
    final_eval.json
    metrics.jsonl
    train.log
  pod_d{16,32,64}/
    pod_basis.npz
    pod_summary.json
    pod.log
  latents_fukami_d{3,32,64}/
    {train,test_a,test_b,test_c}.npz
  latents_pod_d{16,32,64}/
    {train,test_a,test_b,test_c}.npz
  latents_jepa_d64/
    {train,test_a,test_b,test_c}.npz   (or symlink to outputs/session14/latents/S12_E_d64)
  predictor_{fukami,pod}_d{...}/
    checkpoint_iter{005000..020000}.pt
    metrics.jsonl
    train.log
  rollouts_{fukami,pod,jepa}_d{...}/
    {test_b,test_c}.npz
  physical_closure_comparison.csv      (B1 Part d step 2)
  epiplexity_comparison.csv            (B1 Part e, optional)
```

## Realistic compute estimate (two RTX 6000 cards parallel)

| Step                              | Wall time       | Notes                                   |
|-----------------------------------|-----------------|-----------------------------------------|
| Fukami AE x 3 d (parallel 2 cards)| 4-6 h           | d=3 + d=32 in parallel, then d=64       |
| POD x 3 d                         | 15-30 min       | CPU, snapshot SVD via torch.svd_lowrank |
| Encode latents (7 baselines)      | 15-30 min       | Forward passes on 252 encounters each   |
| Predictor training x 6 (parallel) | 3-5 h           | Small predictor on precomputed latents  |
| Rollouts x 6 (parallel)           | 15-30 min       | Fast: predictor-only, no encoder        |
| Physical metrics + bootstrap      | 30-60 min       | NOT YET BUILT (B1 Part d step 2)        |
| Epiplexity (optional)             | 1-2 h           | NOT YET BUILT                           |
| **B1 total**                      | **~9-12 h**     | **One working day, parallel cards**     |

The pace assumes both RTX 6000 Blackwell cards are used. The "Week 1"
framing in the SESSION18 plan is calendar-conservative; this directory's
infrastructure makes the whole B1 sequence run in a focused single day.

## Notes

- The JEPA baseline reuses the production E d=64 checkpoint at
  `outputs/runs/session12/S12_E_d64/encoder/checkpoint_iter020000.pt`
  (jointly trained encoder + predictor). No new JEPA training.

- The Markov closure literature gate from the plan was a 10-percent C_L
  tolerance; Session 17 D127 revised showed the predictor + probe
  pipeline has irreducible ~30 percent relative error at H=16. Use the
  Mode-degradation-vs-Mode-A gate (factor 0.7 to 1.3 versus oracle).

- The Wu's impulse-lift correlation expectation (r > 0.95 on DNS) does
  not hold on 2D mid-plane omega (DNS itself gives r = -0.03; Session
  17 D124c). Use `I_y^w` (wake-only vorticity impulse) instead of `I_y`
  throughout the paper.

- The strict-paper Fukami variant (`tanh` + no GroupNorm + fp32) is
  excluded from the headline comparison and documented in the methods
  appendix as a known-broken variant on this flow (CLAUDE.md).
