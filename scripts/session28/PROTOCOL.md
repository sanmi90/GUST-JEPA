# Frozen evaluation protocol (Session 28, Track A4; HANDOFF D181)

Machine-readable source of truth: `configs/eval_protocol_v2p1.yaml` (read by
`scripts/session28/eval_all.py`). The boxed manuscript version is
`paper/sections/protocol_box.tex`. This file is the human-readable statement.
Frozen 2026-06-10, before any v2.1 held-out evaluation. Changes after that
point are protocol violations and need their own HANDOFF D-number.

## 1. Rollout protocol (one convention, every family, every figure and table)

Full pre-impact context: the predictor is initialised with all encoded frames
up to and including the per-encounter impact frame (the predictor attends to at
most T = 32 latent history frames), then runs autoregressively with no teacher
forcing for the rest of the encounter. Horizons H are counted from the impact
frame; we report H in {4, 8, 16, 32} with H = 16 primary. Markov single-frame
seeding is retired: it underperforms the unconditioned full-context rollout and
was the source of the Table 4b vs Table 7 ambiguity (referee B3).

## 2. Held-out R^2, uncertainty, and multiplicity

R^2 = 1 - SSE/SST with SST about the held-out split's own mean, collected over
per-encounter pairs (y_pred, y_true) at impact + H. Probes are fitted on train
latents only, with 5-fold case-level cross-validation on the readout. Bootstrap
resampling unit is the encounter (n = 2000); every wake claim additionally
carries a case-clustered CI (cluster unit = case). Family-wide multiplicity is
controlled with Holm over the 12-test family (6 observables x 2 endpoints,
JEPA vs the reconstructive AE, test_b, H = 16). The primary endpoint is the
REPRESENTATIONAL wake-enstrophy R^2 at H = 16 on test_b (pooled tiers),
pre-registered in HANDOFF D130/D165 before any v2.1 model existed; the freezing
commit is recorded into numbers.json by eval_all.py.

## 3. Probe classes (closure matrix axis; Gate GD)

Three probe classes, reported for BOTH endpoints (representational z_dns and
forecast z_full): ridge (linear, primary), KernelRidge-RBF, and a small MLP
regressor, the three already wired into physical_metrics_from_rollouts.py.
Parameter probes (G, D, Y) read IMPACT-frame z; state-descriptor probes read
PER-frame z (CLAUDE.md probe methodology).

## 4. Selection convention

Headline numbers are quoted at fixed d = 64 as seed mean +- sd (lead family
4 seeds: 42/0/1/2; other families 3 seeds). The full
family x d x observable x endpoint x probe matrix lives in the appendix table.
The phrase "least-bad" is retired; Table 4 quotes maxima over evaluated d only
with the explicit note that this selection favours the baselines.

## 5. Source groups

"periodic" = 800-frame simulation campaign, 6 encounters per case; "run3" =
480-frame campaign, 4 encounters per case. test_b pools both (locked rule; no
stratification by source group). The two groups are defined once in Section 2.2
of the manuscript and referred to by these names everywhere (closes M13).

## 6. Visualization decoders (amendment 2026-06-12, pre-evaluation)

Every T9 decoder (production tf-no-c and lstm-no-c, post-hoc fukami and pod via
--latents-npz) uses the identical LapFiLM + region_pyr_specloss recipe and
30k-iteration budget. Operating point: per family, the saved checkpoint (every
2000 iterations) with maximum val (test_a) SSIM_mean; the same rule for every
family, no reuse of another family's iteration number. Caveat carried into the
manuscript decode section: the spectral-amplitude term matches amplitude
spectra only and is phase-blind, so decoded fields can show plausible wake
texture displaced from its true location; decoded fields support visualization
and the labelled decode-ceiling comparison only, and no quantitative
localization claim rests on them (those probe latents directly). Two ablation
decoders on the production family quantify the loss terms: dec_ablate_nophys
(lambda_enstrophy = lambda_circulation = 0) and dec_ablate_nospecamp
(lambda_spectral_amp = 0).
