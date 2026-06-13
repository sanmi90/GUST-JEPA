# Sparse wall-pressure FLOW-FIELD recovery comparison (v2.1)

Author reframe 2026-06-13: the sensing track is a cross-method comparison of
which representation, estimated from sparse wall pressure, best recovers the
STATE and the FLOW FIELD. The earlier pressure -> wake-enstrophy claim is
dropped. This file is the PHYSICAL-SPACE companion to the STATE recovery in
`outputs/session28/sensing/` (state R^2 at K=8 test_b: jepa 0.78 > fukami 0.66
> bvae 0.51 > pod 0.34).

## The chain

Per family, per encounter on test_b and test_c:

    sparse wall pressure (K qDEIM taps over the pre-impact window W=30)
      --KRR(RBF)-->  estimated latent z_hat(impact+16)
      --family decoder (frozen operating point)-->  recovered omega field
    scored against the DNS truth field at impact+16.

The KRR map, qDEIM tap picks, pressure window W and readout horizon are reused
verbatim from `scripts/session28/sensing_cf.py`. The decoders are loaded at
their frozen operating iters from `decode_operating_points.py`.

## Metrics

At K=2/4/8/16 on test_b and test_c, per family: field SSIM (Wang K1=0.01
K2=0.03 on pipeline-normalized omega, data range L=8.45 from the
`configs/ssim_data_range.json` registry), volume L2 relative error
(eps_volume), and peak-vorticity retention (recovered/DNS peak |omega|), of
the pressure-recovered field vs DNS. MSE/eps/peak are on the raw scale; SSIM
normalizes both sides into the pipeline-normalized units of L.

## Decode-ceiling reference (the decoder-quality confound)

For every family we ALSO score the field decoded from the TRUE encoded latent
z_dns(impact+16) (no pressure step). The gap `decode-ceiling SSIM minus
pressure-recovered SSIM` isolates how much field error is the pressure->state
recovery versus the decoder itself. POD's LINEAR decoder has the best raw
ceiling (test_a SSIM ~0.68) while Fukami's is ~0.50, so a field-from-pressure
ranking can be partly decoder quality, not state recovery. Both numbers are
reported side by side; the ceiling is never allowed to silently flip the
ranking.

## Headline (test_b, K=8)

| family | pressure-recovered SSIM [95% CI] | decode-ceiling SSIM [95% CI] | gap |
| --- | --- | --- | --- |
| jepa_tf_noc | 0.572 [0.467, 0.673] | 0.618 [0.550, 0.688] | +0.047 |
| fukami | 0.481 [0.383, 0.581] | 0.538 [0.471, 0.608] | +0.057 |
| pod | 0.583 [0.484, 0.674] | 0.709 [0.659, 0.762] | +0.126 |

## Which method recovers the FIELD best (with the ceiling gap)

Pressure-recovered field SSIM ranking (best first): pod, jepa_tf_noc, fukami.
Decode-ceiling SSIM ranking (best first): pod, jepa_tf_noc, fukami.
STATE-recovery ranking (sensing_cf, for reference): jepa_tf_noc, fukami, pod.


VERDICT (adversarial, reported AS MEASURED): the FIELD winner is pod, which
is NOT the STATE winner jepa_tf_noc. The decode-ceiling order (pod, jepa_tf_noc, fukami) shows this is
driven at least in part by decoder quality, not state recovery: the family
with the stronger raw decoder recovers a more SSIM-similar field even when
its pressure->state recovery is weaker. We report the field result as
measured and do NOT force the state ordering onto it; the gap column makes
the confound explicit.


## Statistics

5-fold CASE-level CV on the pressure->z map (sensing_cf folds; no case
straddles a fold); case-clustered bootstrap CI on the headline field SSIM per
family; paired JEPA-vs-reconstructive differences via a one-sided sign test +
a case-clustered CI on the per-encounter SSIM difference (NOT
case_permutation_p, the B6 lesson).

## Notes

- bvae has no trained decoder under `outputs/session28/decode`, so it cannot
  enter the FIELD comparison; its STATE recovery is in `sensing_cf.json`.
- Decoder forward ran on: RTX6000_gpu1 (RTX 6000 gpu1 if available, else CPU; the L40S
  cards are forbidden).
- Outputs: `results.{npz,json}`, `fig_field_recovery.{pdf,png}`, and the
  numbers part `outputs/session28/numbers_parts/sensing_field_cf.json`.

