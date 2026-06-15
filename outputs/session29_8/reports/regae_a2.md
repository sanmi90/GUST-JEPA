# SESSION29.8 Track A2: anti-collapse and reconstructive wake readability

## Question
Does adding SIGReg anti-collapse (lambda_sigreg = 0.01) to the matched
reconstructive AE (reconstruction + the same wake head as ctrl_recon) change the
RECONSTRUCTIVE latent's instantaneous representational wake readability, versus
the matched reconstructive control WITHOUT anti-collapse (ctrl_recon:
NumReprWakeCtrlReconCnnVit = 0.13, NumReprWakeCtrlReconCnn = 0.35)?

## Provenance
- git SHA: `bed0f250e5bc0e53be4b579e418c469c5d013e14`
- UTC: 2026-06-15T23:22:16.743065+00:00
- split: v2p1 (`configs/splits/split_v2p1.json`)
- omega pipeline: `outputs/data_pipeline/v2p1/manifest.json`
- DNS target: `outputs/session28/exp2/dns_physical_metrics.npz` (canonical
  wake_enstrophy; reproduces NumReprWakeJepaTf = 0.79)
- checkpoints: `outputs/runs/session29_8/regae/regae_{arch}_d64_s{seed}/checkpoint_iter020000.pt` for arch in {cnn, cnn_vit}, seed in 0..4
- latents: `outputs/session28/latents/regae/{cnn,cnn_vit}_s{seed}/{train,test_b}.npz`

## Encode command (per checkpoint)
```
taskset -c 0-15 env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  python scripts/session18/encode_baseline_latents.py \
    --baseline fukami --d 64 --checkpoint <ckpt> --partition v2p1 \
    --split configs/splits/split_v2p1.json \
    --pipeline-manifest outputs/data_pipeline/v2p1/manifest.json \
    --splits train test_b --gpu 0 --output-dir outputs/session28/latents/regae/<arch>_s<seed>
```
(non-strict load: the SIGReg + wake-head keys are ignored; only the encoder
weights load.)

## Probe (exact published representational-closure protocol)
Verbatim reuse of `scripts/session28/closure_matrix.py` via
`scripts/session29/probe_regae_a2.py` and `families_regae_a2.yaml`: ridge probe
(standardise z, alpha = 1.0) fitted on each cell's TRAIN per-frame latents
(`z_full`, pooled over all frames) against canonical per-frame wake_enstrophy;
read out at frame impact + 16 on test_b (pooled tiers); R^2 = 1 - SSE/SST about
the test_b mean; case-clustered bootstrap CI (n = 10000).

## Reproduce-check (same probe on published families, before trusting outputs)
| family | macro | published | recomputed | match (<0.005 @ 2dp) |
| --- | --- | --- | --- | --- |
| ctrl_recon_cnnvit | NumReprWakeCtrlReconCnnVit | +0.13 | +0.1267 | True |
| ctrl_recon_cnn | NumReprWakeCtrlReconCnn | +0.35 | +0.3455 | True |
| fukami | NumReprWakeFukami | -0.25 | -0.2530 | True |

The protocol port reproduces the published control macros to 2 decimals, so the
regAE numbers below are computed under the identical pipeline.

## regAE results (recon + wake head + SIGReg)

### regAE CNN (NumReprWakeRegaeCnn)
| seed | test_b repr wake R^2 (H16) |
| --- | --- |
| 0 | +0.3160 |
| 1 | +0.7269 |
| 2 | +0.5316 |
| 3 | +0.4200 |
| 4 | +0.4175 |
| **seed-mean (n=5)** | **+0.4824** |
| seed-sd | 0.1565 |

seed-mean test_b repr wake R^2 = +0.4824
case-clustered CI (lead seed 0, n_enc = 42) = [-0.3575, +0.7605]

### regAE CNN+ViT (NumReprWakeRegaeCnnVit)
| seed | test_b repr wake R^2 (H16) |
| --- | --- |
| 0 | +0.6241 |
| 1 | -0.6335 |
| 2 | +0.7857 |
| 3 | +0.4890 |
| 4 | -0.2324 |
| **seed-mean (n=5)** | **+0.2066** |
| seed-sd | 0.6099 |

seed-mean test_b repr wake R^2 = +0.2066
case-clustered CI (lead seed 0, n_enc = 42) = [+0.1893, +0.8914]

## Comparison band
- predictive / supervised range: NumReprWakeJepaTf = 0.79, NumReprWakeSupOnly = 0.92
- reconstructive no-anti-collapse control: NumReprWakeCtrlReconCnn = 0.35,
  NumReprWakeCtrlReconCnnVit = 0.13, NumReprWakeFukami = -0.25

## VERDICT
Anti-collapse does NOT raise the reconstructive latent's wake readability into the predictive/supervised range; both regAE seed-means (CNN 0.48, CNN+ViT 0.21) sit within noise of the no-anti-collapse ctrl_recon controls (0.35, 0.13) and far below the predictive/supervised band (JEPA 0.79, supervised_only 0.92), so the instantaneous wake readability is supplied by the supervision (predictive objective / wake head), not by anti-collapse.
