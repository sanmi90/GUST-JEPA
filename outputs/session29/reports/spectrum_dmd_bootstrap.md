# Bootstrap CIs for the latent DMD spectrum and Floquet modulus

- git sha: `273744179a582c5f680d47bd9249805ff1de3537`
- command: `python scripts/session29/spectrum_dmd_bootstrap.py`
- UTC timestamp: 2026-06-14T22:02:28.882246+00:00
- n_boot: 1000, seed: 0, CI: 2.5/97.5 percentiles

## Inputs

- latents root: `/home/carlos/GUST-JEPA/outputs/session28/latents`
- predictor checkpoint (Floquet): `outputs/session27/JEPA_d64_noc_tf/checkpoint_iter020000.pt`
- point-estimate part (unchanged): `outputs/session28/numbers_parts/spectrum_dmd.json`

## Method

Part 1 (data-driven DMD on the no-gust Baseline limit cycle): the point
estimate stacks every consecutive snapshot pair (z_t, z_{t+1}) from the four
Baseline encounters into column matrices X, Y (476 pairs, no pair crosses an
encounter boundary), fits A = Y X^+, and extracts the leading-modulus
oscillatory eigenvalue. With only four encounters an encounter-level resample
is too coarse, so the bootstrap resamples the snapshot pairs (columns) with
replacement within the standard DMD delay-embedding window, refits A, and
re-extracts the same leading eigenvalue. n = 1000 draws.

Part 2 (Floquet): the 59 per-frame companion Jacobians of the tf-no-c
predictor over one Baseline shedding cycle are computed once on CPU, then the
ordered-product monodromy is bootstrapped by resampling those step factors
with replacement (the per-step modulus is their geometric contribution). The
reported quantity is the per-step leading-multiplier modulus
|mu_cycle|^{1/period}. n = 1000 draws.

## Part 1: DMD leading oscillatory eigenvalue (Baseline limit cycle)

| family | St point | St CI | |lambda| point | |lambda| CI |
| --- | --- | --- | --- | --- |
| jepa_tf_noc | 0.662 | [0.037, 0.695] | 0.991 | [0.988, 1.016] |
| pod | 0.682 | [0.037, 0.778] | 0.996 | [0.993, 1.022] |
| fukami | 0.503 | [0.049, 0.609] | 0.939 | [0.907, 0.975] |

Honest read of the St lower tail: the St CI lower bound is small for every
family (the leading-eigenvalue selection occasionally locks onto a different,
lower-frequency oscillatory mode of comparable modulus under a resample, since
the d = 64 operator carries several near-unit-modulus modes). The CI therefore
reflects the leading-mode SELECTION variability of the estimator, not a wide
genuine frequency band; the modulus CI is the clean, well-behaved quantity and
is the one to cite for the marginal-stability claim. Report the St as the point
estimate with the percentile band as a faithful-but-conservative uncertainty.

Macros created via the eval_all ci_lo/ci_hi mechanism (base name + lo/hi;
non-colliding with the existing SpecDmdSt*/SpecDmdMod* point macros): SpecDmdStJepaTfCI{,lo,hi}, SpecDmdModJepaTfCI{,lo,hi}, SpecDmdStPodCI{,lo,hi}, SpecDmdModPodCI{,lo,hi}, SpecDmdStFukamiCI{,lo,hi}, SpecDmdModFukamiCI{,lo,hi}.

## Part 2: Floquet leading per-step multiplier modulus (tf-no-c predictor)

- per-step |mu| point estimate: 1.0042
- 95% bootstrap CI: [1.0026, 1.0049]
- period: 59 frames, window W = 16
- macros created (ci_lo/ci_hi mechanism): SpecFloqModCI, SpecFloqModCIlo, SpecFloqModCIhi.

### Verdict

The Floquet per-step modulus CI [1.0026, 1.0049] sits ENTIRELY ABOVE 1.0, so the orbit is (weakly) unstable / amplifying rather than strictly marginally stable; the wording should say the modulus sits just above 1, not exactly on the unit circle.

