# Topology fairness (C-E2; closes referee M4)

Vietoris-Rips persistence (ripser, maxdim=1) on the per-encounter latent trajectory point cloud (z over 120 frames traces the shedding limit cycle; a clean cycle is one persistent H1 generator). Generators counted at the canonical 5 percent diameter floor and over the robustness grid [2, 5, 10, 20] percent.

## Metrics (per family)

- raw: learned-latent coordinates as stored (continuity with v2).
- stdz: per-family standardisation, z-score each latent dim by that family's TRAIN per-dim std (the cheap whitening; the headline fair metric).
- maha: full Mahalanobis whitening by the family's TRAIN covariance (Sigma^{-1/2}); reported for completeness.

Headline whitened metric for the GE2 decision: stdz.

## Gusted encounters (test_b + test_c), H1 at the 5 percent floor

Median H1 generator count and the fraction of encounters read as a single clean cycle (count == 1), raw vs the headline whitened metric. Gusted encounters are kept whole: the impact event concentrates the trajectory into one dominant excursion.

| family | median (raw) | median (whitened) | frac single (raw) | frac single (whitened) |
|---|---|---|---|---|
| jepa_tf_noc_d64_s42 | 1.0 | 1.0 | 0.59 | 0.61 |
| fukami_d64_s42 | 1.0 | 1.0 | 0.80 | 0.82 |
| pod_d64 | 1.0 | 2.0 | 0.71 | 0.42 |
| bvae_faith_d64_s42 | 1.0 | 3.0 | 0.48 | 0.20 |

## No-gust Baseline control, H1 at the 5 percent floor

The full 120-frame Baseline encounter traces ~2 shedding periods (St_full = 0.338, dt_tc = 0.05, period ~59 frames), so it fragments for EVERY family, including the predictive one; that full-encounter number is reported but is not the clean-cycle null. The fair clean-limit-cycle reference is the SINGLE-PERIOD segmentation, where a clean cycle is one generator.

| family | period (frames) | full-encounter median (raw) | single-period median (raw) | single-period median (whitened) | single-period frac single (whitened) |
|---|---|---|---|---|---|
| jepa_tf_noc_d64_s42 | 30 | 5.0 | 1.0 | 1.0 | 0.56 |
| fukami_d64_s42 | 58 | 2.0 | 2.0 | 2.5 | 0.00 |
| pod_d64 | 29 | 5.5 | 1.0 | 1.0 | 0.62 |
| bvae_faith_d64_s42 | 30 | 1.5 | 0.0 | 0.5 | 0.31 |

## Decisive comparative gap (predictive minus reconstructive)

Single-cycle fraction, JEPA minus fukami, under stdz whitening at the 5 percent floor:

- gusted encounters: -0.21 (jepa 0.61, fukami 0.82)
- single-period no-gust control: +0.56 (jepa 0.56, fukami 0.00)

## GE2 branch

Branch: WEAK-MIXED

split verdict. On the WHOLE gusted encounters per-family standardisation closes/reverses the gap (gap = -0.21, fukami is read as a single cycle at least as often as the predictive family), so the v2 raw-coordinate gusted fragmentation claim was an anisotropic-scale artefact and Figure 5 must reframe from topology to metric organisation. BUT the single-period no-gust limit-cycle control DOES separate the families (gap = +0.56: the reconstructive encoding fragments the clean shedding cycle while the predictive encoding keeps it), and this survives whitening; the clean-cycle topology statement holds on the no-gust control, scoped to the limit cycle

## Encoding-vs-rollout attribution fix (abstract)

The v2 abstract said the reconstructive ROLLOUT 'fragments the encounter topology', but Sec 4.3 establishes fragmentation for the simulation-ENCODED latent and says the rollout lifetime ratio does NOT separate families. This analysis computes persistence on the simulation-ENCODED latent trajectories. Fragmentation therefore belongs to the ENCODING; manifold departure (the drift result, Sec 4.3) belongs to the ROLLOUT. The abstract must attribute fragmentation to the encoding and manifold departure to the rollout, regardless of the GE2 branch.

