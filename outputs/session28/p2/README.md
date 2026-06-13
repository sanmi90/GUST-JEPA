# Session 28 / Physics Track P2: the latent recovery clock and Gate GP2

The physical recovery clock tau_rec (v2: wake enstrophy re-enters the settled-
Baseline band and dwells one period) is read from
outputs/session28/physics/per_encounter_physics.npz. This script computes the LATENT
recovery clock from the predictive (jepa_tf_noc) latent trajectories and correlates
it with the physical one (Gate GP2). NOT git-committed; left for review.

## Latent tube

- Baseline case: Baseline; settled orbit points: 80 (effective dim 3.6 of 64).
- Mahalanobis tube radius (q95): 7.846.
- Euclidean cross-check radius (q95): 0.284.
- Dwell: 59 frames (2.95 t/c) = one subharmonic period. one subharmonic shedding period = round(1/(St_sub*dt_tc)) = 59 frames (St_sub=0.3383); physics_prep enstrophy clock uses 56 (the St_sub rounding)
- The orbit is very low-rank, so the Mahalanobis whitening is ill-conditioned (see
  s46_regen). The gate statistic is a RANK correlation (insensitive to whitening
  magnitude) and a well-conditioned Euclidean tube is carried as a cross-check.

## Gate GP2 (Spearman of latent vs physical tau_rec, recovered-in-both, test_b)

- HEADLINE (Mahalanobis tube): Spearman rho = nan (n = 0; fewer than 3 recovered-in-both pairs; Spearman undefined).
- Euclidean cross-check: Spearman rho = nan (n = 0; fewer than 3 recovered-in-both pairs; Spearman undefined).

GP2 threshold = 0.7. **VERDICT: BELOW 0.7**.

latent-clock sentence DROPPED; physical DNS maps stand as the contribution.

Stated plainly: the latent recovery clock does NOT track the physical recovery
clock at the GP2 bar on test_b. The latent-clock sentence is DROPPED from S4.4.
The physical tau_rec(G, D, Y) maps stand as DNS physics and remain a contribution.

## Recovered vs censored fractions (test_b)

| clock | n | recovered | censored | frac recovered |
|---|---|---|---|---|
| physical (v2) | 42 | 5 | 37 | 0.12 |
| latent (Mahalanobis) | 42 | 0 | 42 | 0.00 |
| latent (Euclidean) | 42 | 0 | 42 | 0.00 |

Status taxonomy: 'recovered_short_window' counts as recovered (a dwell was certified
on the available remainder); only 'censored' is censored. Censoring is never silently
excluded; encounters censored under EITHER clock are simply not in the paired
correlation, and the counts above make that explicit.

## Map data

Physical tau_rec(G, D, Y) over test_b/test_c and tau_rec vs |G| are in results.json
(physical_map_data) and drawn in fig_p2_recovery.{pdf,png} (left panel: the envelope
map; right panel: the latent-vs-physical clock scatter with the Spearman).
