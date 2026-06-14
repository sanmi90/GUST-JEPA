# Latent drift: supervised_only family (SESSION29.8 A1)

Adds the supervised_only encoder to the manuscript latent-drift table (`tab:latent_drift`) using the identical horizon, reference cloud, covariance estimator, and rollout protocol as the existing families (jepa, fukami, pod, bvae) in `scripts/session28/drift_ce1.py`. The three metrics are computed by importing and calling `drift_ce1.compute_key`, not by reimplementing any formula, so they are apples-to-apples.

## Provenance

- git sha: `273744179a582c5f680d47bd9249805ff1de3537`
- command: `python scripts/session29/drift_supervised_only.py`
- UTC timestamp: 2026-06-14T21:42:08Z
- horizon H = 24, latent d = 64, split = test_b, full-context rollout (`z_full`)
- inputs:
    - encoded latents: `/home/carlos/GUST-JEPA/outputs/session28/latents/supervised_only_d64_s42/{train,test_b}.npz`
    - matched rollout: `/home/carlos/GUST-JEPA/outputs/session28/rollouts/supervised_only_d64/test_b.npz`
    - predictor checkpoint: `/home/carlos/GUST-JEPA/outputs/session28/predictors/supervised_only_d64/checkpoint_iter020000.pt`
- part file: `/home/carlos/GUST-JEPA/outputs/session28/numbers_parts/drift_supervised_only.json`

## supervised_only headline metrics (H = 24, d = 64, test_b)

- `DriftRelLtwoSupOnly` = 0.9140 (relative-l2 rollout deviation vs DNS-encoded latent)
- `DriftMahaSupOnly` = 0.8269 (Mahalanobis ratio to per-frame TRAIN encoded cloud, v2 floored regime)
- `DriftSpecNearNullSupOnly` = 0.0000 (near-null departure fraction; 0 near-null encoded eigen-directions)

## 1-row comparison (same protocol, H = 24, d = 64, test_b)

| family | rel-l2 | Maha ratio | near-null dep frac | n null dirs | enc top-5 var |
|---|---|---|---|---|---|
| supervised_only | 0.914 | 0.827 | 0.000 | 0 | 0.500 |
| jepa (predictive, on-manifold) | 0.537 | 0.702 | 0.883 | 42 | 0.919 |
| fukami (reconstructive, off-manifold) | 0.318 | 8.998 | 0.995 | 58 | 0.998 |

## Interpretation

The reconstructive Fukami AE keeps a tiny rel-l2 but blows up the Mahalanobis ratio because its rollout exits along near-null encoded directions (high near-null departure fraction, many near-null dirs, variance concentrated in the top-5). The predictive JEPA wanders within the broad isotropised cloud (larger rel-l2, in-distribution Mahalanobis, few near-null dirs). On the (Mahalanobis ratio, near-null fraction) plane supervised_only is closer to jepa (predictive, on-manifold).
