# SESSION 20 plan: close the paper’s evidence gaps and add the reference-grade analyses

This is a Claude Code session plan written to your repo’s conventions
(CLAUDE.md, HANDOFF.md decision-log style). No time or GPU limit, so it is
structured for maximum parallelism across the two RTX 6000 cards (plus the L40S
cards under the existing `VORTEX_JEPA_ALLOW_NON_RTX6000=1` bypass for the
non-encoder work, exactly as D131 did for the B1 chain). Every track ends in a
decision stub to paste into HANDOFF.md and an acceptance gate.

The session has one organising principle: the rewritten manuscript
(REWRITE_NOTES.md) has a list of `\pending{}` items, and each is either a number
to compute from artifacts you already have, or one experiment to run. This plan
clears all of them, and adds the five reference-inspired analyses, in dependency
order so that nothing waits on anything it does not have to.

Read REWRITE_NOTES.md and WORLD_MODEL_FRAMING.md first; this plan assumes both.

-----

## Pre-flight (10 min, blocks everything)

```bash
source .venv/bin/activate
export PREVENT_ROOT=$HOME/PREVENT WANDB_PROJECT=vortex-jepa
python -m src.training.sanity_checks --all --require-gpu
```

Confirm the v2 artifacts D131 references are present, because every track below
reads from them:

```bash
ls outputs/runs/session12/S12_E_d64/encoder/                       # production d=64
ls outputs/runs/session12/S12_E_d32/encoder/                       # matched-capacity d=32 (D131.2)
ls outputs/runs/session14/thrust6/jepa_d64_seed{0,1,2}/encoder/    # seed retrains
ls outputs/session18/exp_b1_test3/                                 # B1 closure CSVs + drift + pressure
ls outputs/session17/exp1/ outputs/session17/exp2/                 # trajectory geometry + Markov closure
ls outputs/session16/exp2/per_frame_targets/                       # per-frame DNS descriptors
```

If the d=32 S16/S17 interpretability residuals flagged at the end of D131 bite
(per_frame_targets shape mismatch, exp1a n_components bug, exp5 DNS metrics
path), fix them here; D131 estimates 30 minutes. They do not gate Tracks A-C but
they do gate the d=32 rows of Tracks E and F.

-----

## Track A (decisive, START FIRST, long pole): the objective x architecture controls

This is the experiment the whole central claim is contingent on (REWRITE_NOTES
section 2, item 2; manuscript Table `tab:controls_2x2`). Everything else can run
while this trains.

The confound: JEPA differs from Fukami in objective AND architecture (CNN+ViT vs
CNN) AND auxiliary supervision (wake head + lift head vs lift head only). The
2x2 isolates the objective.

Cells, all at d=64, all evaluated under the SAME unified `--no-output-bn`
predictor and the SAME probe family already used in B1 (D129):

|cell|objective     |encoder |aux heads (MATCHED)|
|----|--------------|--------|-------------------|
|A1  |predictive    |CNN+ViT |lift + wake        |
|A2  |predictive    |CNN only|lift + wake        |
|A3  |reconstructive|CNN+ViT |lift + wake        |
|A4  |reconstructive|CNN only|lift + wake        |

Plus the auxiliary-isolation control:

|cell|objective |encoder|aux heads          |purpose                         |
|----|----------|-------|-------------------|--------------------------------|
|A5  |predictive|CNN+ViT|lift only (NO wake)|does the wake head carry the win|

A1 already exists (it is the production d=64 encoder). A3/A4 require adding the
wake head to the Fukami trainer; A2 requires a CNN-only predictive encoder;
A5 is the production recipe minus the wake head.

The point of matching aux heads ON across all four cells (rather than off) is
that turning them off would re-confound the comparison with “did the wake head
help”; A5 answers that separately. Run >=3 seeds per cell (reuse the Thrust-6
seed protocol).

```bash
# A2: predictive objective, CNN-only encoder. New encoder config.
python -m src.training.train_jepa --gpu 0 \
    --split v2 \
    model.encoder=cnn_only model.encoder.latent_dim=64 \
    loss.lambda_sigreg=0.01 loss.eta=0.01 loss.lambda_wake=0.1 \
    --omega-pipeline-manifest outputs/data_pipeline/v1/manifest.json \
    --tag-suffix A2_pred_cnn_seed0 seed=0
# repeat seed in {1,2}; repeat on --gpu 1 in parallel

# A3/A4: reconstructive objective WITH the wake head added to the Fukami trainer.
# Requires: add a wake-head branch to scripts/session9_train_fukami.py mirroring
# the JEPA wake head (Linear -> 80-d patch_signed_spectrum, SmoothL1, weight 0.1).
python scripts/session9_train_fukami.py --gpu 1 \
    --split v2 --latent-dim 64 --beta 0.01 \
    --add-wake-head --lambda-wake 0.1 \
    --encoder cnn_vit \                 # A3; use --encoder cnn for A4
    --tag-suffix A3_recon_cnnvit_seed0 --seed 0

# A5: production predictive recipe, wake head removed.
python -m src.training.train_jepa --gpu 0 \
    --split v2 model.encoder.latent_dim=64 \
    loss.lambda_sigreg=0.01 loss.eta=0.01 loss.lambda_wake=0.0 \
    --tag-suffix A5_pred_nowake_seed0 seed=0
```

Then the uniform downstream evaluation for all cells (this is the existing B1
machinery, just pointed at the new encoders):

```bash
# Encode latents, train the unified predictor, roll out, probe. Reuse the B1 chain.
bash scripts/_oneoff_b1_full_pipeline.sh   # adapt: add A1..A5 to the BASELINES list
bash scripts/_oneoff_b1_rollouts.sh
```

ACCEPTANCE GATE (decision for the central claim’s strength):

- Compute held-out test_b closure (R^2 and MAE at H=16) for A1-A5, mean over the
  6 observables and broken out for wake_enstrophy specifically.
- If A1 > A3 AND A2 > A4 on wake closure (predictive beats reconstructive at
  BOTH architectures): the claim is “the predictive objective improves closure”,
  the strong form. Manuscript Section 4.5 takes that wording.
- If A1 > A3 but A2 ~ A4: the win needs the ViT; claim becomes “the predictive
  CNN+ViT family improves closure”.
- If A5 << A1 on wake closure: the wake head is load-bearing; claim becomes “wake
  supervision drives the wake-closure advantage”, the weakest and most honest
  form, and the abstract must be rewritten to foreground it.
- Whatever the outcome, populate `tab:controls_2x2` and set the abstract and
  Section 4.5 wording to match. This gate decides the paper’s headline.

HANDOFF stub: `### D132: Session 20 Track A 2x2 objective x architecture controls`
with the 5-cell x 6-observable closure table and the claim-strength decision.

-----

## Track B (1 hour, no training): held-out R^2 and the conditioning-floor fairness fix

Pure post-processing of the rollout predictions Track A’s predecessors already
produced. Clears REWRITE_NOTES items 1 and 5.

```bash
# The rollout predictions exist; recompute R^2 (not just MAE) on test_b and test_c
# for all B1 baselines at H=16, from outputs/session18/exp_b1_test3/.
python scripts/_oneoff_heldout_r2.py \
    --closure-csv outputs/session18/exp_b1_test3/physical_closure_noBN_unified.csv \
    --split v2 --horizon 16 \
    --out outputs/session20/heldout_r2_testb_testc.csv
```

This single script fills `tab:b1_r2_heldout` (currently a red placeholder) and
the JEPA-held-out column of `tab:conditioning_floor` (currently comparing the
floor’s held-out R^2 against JEPA’s TRAINING R^2, which is the apples-to-oranges
bug noted in REWRITE_NOTES). It also lets the abstract’s headline statistic be
re-expressed in held-out R^2 rather than the training 0.835.

ACCEPTANCE GATE: every red `\pending{}` in `tab:b1_r2_heldout` and
`tab:conditioning_floor` is replaced by a number. The abstract’s lead statistic
is held-out.

HANDOFF stub: `### D133: held-out R^2 computed; training-R^2 headline retired`.

-----

## Track C (1 day, no NEW training, the headline mechanism upgrade): persistent homology of latent trajectories

Reference: Smith, Fukami, Sedky, Jones, Taira, JFM 980 A18 (2024). Turns the
scalar drift ratio (9.9x) into a coordinate-free topological invariant. This is
the single highest-value new analysis because it gives the paper the same
conceptual identity as your flagship style reference, and it runs entirely on
latents you already have (D131 latent_drift_diagnostic, S17 exp1 trajectories).

The encounter is a closed cycle (base limit cycle -> LEV excursion -> recovery).
Per encounter, build four point clouds in latent space:

1. DNS-encoded JEPA latent trajectory `z_full` (S12_E_d64 latents)
1. JEPA Markov rollout `z_hat_full`
1. DNS-encoded Fukami latent trajectory
1. Fukami Markov rollout

```bash
pip install ripser persim    # or giotto-tda; ripser is lighter
python scripts/session20/exp_persistent_homology.py \
    --latents outputs/session14/latents/S12_E_d64 \
    --fukami-latents outputs/session18/exp_b1_test3/fukami_d64_latents \
    --rollouts outputs/session18/exp_b1_test3/rollouts \
    --split v2 \
    --maxdim 1 \
    --out outputs/session20/persistent_homology/
```

For each of the four, compute the Vietoris-Rips persistence (H0, H1), extract the
maximum-persistence H1 generator (the “loop”), and its lifetime. Aggregate over
test_b encounters.

Claim to test (the figure): the DNS-encoded trajectories carry one long-lived H1
generator (the encounter cycle), the JEPA rollout PRESERVES it (lifetime ratio
near 1), the Fukami rollout’s H1 generator collapses or fragments (lifetime
ratio falls, spurious generators appear) as it drifts off-manifold. This is the
topological reading of the 9.9x Mahalanobis drift.

ACCEPTANCE GATE:

- Primary: median (over test_b) H1-lifetime ratio rollout/DNS is >= 0.7 for JEPA
  and < 0.5 for Fukami. If so, the topological story stands and goes in Section
  4.3 as the mechanism figure. If the separation is weaker, report the
  persistence diagrams descriptively without the strong claim.
- Cross-check against D123: D123 found the impact frame is a curvature minimum
  (smooth pass-through) with cross-seed Spearman 0.95. The H1 generator should be
  consistent with that canonical geometry; if it contradicts D123, debug before
  claiming.

Figure: four persistence diagrams (one row) + a panel of H1-lifetime vs rollout
horizon per family. Tooling note for the cleanest result: filter generators
below the diagram’s minimal-persistence diagonal as noise, exactly as Smith et
al. do.

HANDOFF stub: `### D134: persistent homology of latent rollouts -- predictive preserves the encounter H1 cycle, reconstructive fragments it`.

-----

## Track D (1 day, no training): optimal-transport field metric and the OT-geodesic latent alignment

Reference: Tran, Yeh, Taira, JFM 1027 A24 (2026). Two distinct uses, both pure
post-processing on existing decoded fields and latents. Clears REWRITE_NOTES
items 2 and 3.

### D-i: OT field dissimilarity for the reconstruction comparison

The SSIM caution in your manuscript (the Fukami AE’s high SSIM is bulk-zero
agreement with a collapsed field, peak |omega| 0.36 vs target 9.7) is exactly
the pathology OT was built to expose. Recompute the reconstruction comparison
with the unbalanced-OT field distance.

```bash
pip install pot      # Python Optimal Transport (Flamary et al. 2021)
python scripts/session20/exp_ot_field_distance.py \
    --decoded outputs/runs/session12/S12_E_d64/encoder/decoder_specloss_recipe/recon \
    --fukami-decoded outputs/session18/exp_b1/fukami_recon \
    --pod-decoded   outputs/session18/exp_b1/pod_recon \
    --split v2 \
    --rho 1.0 --reg-kl \
    --out outputs/session20/ot_field/
```

Implement the signed-vorticity split per Tran et al. eq (2.5):
`d_field(V1,V2) = S_eps(m+_1, m+_2) + S_eps(m-_1, m-_2)`, positive and negative
vorticity transported separately and summed, Sinkhorn via POT, KL divergence for
the unbalanced marginals, characteristic radius rho=1. Compute on test_a, test_b,
test_c reconstructions for JEPA, Fukami, POD.

Expected outcome (the table flip): the collapsed Fukami reconstruction takes a
LARGE OT distance (correctly penalised, since it costs a lot to transport a
quiescent field onto the true LEV), the localised-but-blurry JEPA reconstruction
takes a SMALL OT distance, POD sits between. This replaces the misleading SSIM as
the headline reconstruction metric.

### D-ii: OT-geodesic vs latent-distance alignment (the drift mechanism, geometric)

This is the deeper, citable mechanism for WHY the predictive rollout stays
on-manifold. Along each encounter compute the OT distance between consecutive DNS
vorticity fields (the physical transport geometry), and test whether JEPA latent
Euclidean distances track that OT geodesic more faithfully than Fukami latent
distances do.

```bash
python scripts/session20/exp_ot_latent_alignment.py \
    --dns-fields  ${VORTEX_JEPA_CACHE}/v2 \
    --jepa-latents outputs/session14/latents/S12_E_d64 \
    --fukami-latents outputs/session18/exp_b1_test3/fukami_d64_latents \
    --split v2 \
    --out outputs/session20/ot_alignment/
```

Per encounter: build the pairwise OT-distance matrix between frames (or
consecutive-frame OT distances), and the pairwise latent-distance matrix; report
Spearman/Shepard correlation, JEPA vs Fukami. Claim: JEPA latent distance is
approximately proportional to physical transport while Fukami’s is not, so a
single predictor step is a physically smooth, transport-consistent move, and
iterating it does not leave the manifold.

ACCEPTANCE GATE:

- D-i: the OT field distance must rank the collapsed Fukami reconstruction worse
  than JEPA on test_b, reversing the SSIM ranking. If it does, it replaces SSIM
  in the manuscript. Report alongside the existing Wang-SSIM (D131.3) for
  continuity, but lead with OT.
- D-ii: JEPA OT-latent Spearman must exceed Fukami’s by a clear margin (target

> 0.15 absolute) on test_b. If so, this is the mechanism paragraph in 4.3. If
> the margin is marginal, report descriptively.

HANDOFF stub: `### D135: OT field metric reframes the reconstruction comparison; OT-geodesic alignment explains predictive on-manifold rollout`.

-----

## Track E (0.5 day, no training): limit-cycle and phase-amplitude reading

Reference: Fukami, Nakao, Taira, JFM 992 A17 (2024). Defines “recovery”
precisely and connects the predictor to the phase-amplitude control machinery
for these flows. Clears REWRITE_NOTES item 4. Builds directly on D123 (you
already have the trajectory geometry and the curvature profiles).

```bash
python scripts/session20/exp_phase_amplitude.py \
    --latents outputs/session14/latents/S12_E_d64 \
    --baseline-encounter Baseline \
    --rollouts outputs/session18/exp_b1_test3/rollouts \
    --split v2 \
    --out outputs/session20/phase_amplitude/
```

Steps: (1) confirm the baseline (no-gust) latent traces a closed periodic orbit
(it is the limit cycle; you have the baseline case in train); (2) define a phase
along that orbit (e.g. via the Hilbert transform of the two leading latent PCs,
or the protophase from D123’s PCA-impact projection); (3) show gust encounters
depart the orbit and return, so “recovery” is operationally “return to the
baseline limit cycle”; (4) show the JEPA predictor rollout carries the disturbed
trajectory BACK toward the orbit while the Fukami rollout departs (this is the
same fact as the drift ratio and the H1 collapse, read dynamically).

Optional add-on, mirroring Fukami et al. Section 3.1: a sparse-regression (SINDy)
fit of the latent dynamics on the limit cycle, to show the predictor has learned
an interpretable phase flow. Mark optional; it does not gate.

ACCEPTANCE GATE: the baseline latent orbit must be visibly closed (return
distance < 10% of orbit diameter), and the median return-to-orbit distance under
JEPA rollout must be smaller than under Fukami rollout on test_b. If so, this is
Section 4.4 and it unifies with Track C (the limit cycle IS the H1 generator).

HANDOFF stub: `### D136: baseline latent is a limit cycle; recovery = return to orbit; predictive rollout returns, reconstructive departs`.

-----

## Track F (0.5 day, light compute): scale decomposition of the wake-observable advantage

Reference: Odaka, Lopez-Doriga, Taira, JFM 1031 R3 (2026), and the PRF dataset
paper (Fukami, Smith, Taira 2025). Ties the wake-enstrophy/circulation advantage
to the leading-edge vortex and shear layer rather than to a scalar, and narrates
the staged encounter. Clears the last REWRITE_NOTES analysis item.

```bash
python scripts/session20/exp_scale_decomposition.py \
    --dns-fields ${VORTEX_JEPA_CACHE}/v2 \
    --jepa-decoded outputs/runs/session12/S12_E_d64/encoder/decoder_specloss_recipe/recon \
    --fukami-decoded outputs/session18/exp_b1/fukami_recon \
    --sigma-over-c 0.05 \
    --split v2 \
    --out outputs/session20/scale_decomp/
```

Apply the Gaussian scale decomposition of Motoori & Goto (Odaka eq 3.1-3.4):
split each field into large-scale (`u_L`, Gaussian filter at sigma/c=0.05) and
small-scale (`u_S = u - u_L`). Compute the wake observables on the large-scale
field for DNS, JEPA reconstruction, Fukami reconstruction. Claim: the JEPA
reconstruction retains the large-scale LEV and shear-layer structures that carry
the lift peaks (the ones that dominate wake enstrophy), while Fukami smooths them
away; that is what the wake-enstrophy probe gap measures, in physical-space terms.

Narrate the staged encounter (D123’s impact-frame geometry + the PRF stages): the
gust-induced leading-edge vorticity flux, the LEV roll-up at the first lift peak,
the arch/trailing-edge structure at the negative peak, the recovery.

Note the 2D->3D framing here too: at |G|=4 (test_c) the PRF paper says the flow
is genuinely three-dimensional, so the mid-plane scale decomposition is itself
incomplete there, which is the physical reason test_c degrades (the observability
boundary already in the rewrite).

Optional, heavier: a force-element computation (Chang 1992) linking the retained
large-scale structures to lift needs the velocity fields (you have `/u` in the
PREVENT HDF5). Mark optional; the scale-decomposition + Q-criterion narrative is
the core, force-element is a stretch goal.

ACCEPTANCE GATE: the large-scale wake enstrophy of the JEPA reconstruction must
track DNS better than Fukami’s on test_b (consistent with the probe-R^2 gap in
D129/D131). If so, Section 4.6 gets the physical-space interpretation.

HANDOFF stub: `### D137: scale decomposition shows the predictive latent retains the lift-bearing large-scale LEV/shear structures the reconstructive one smooths`.

-----

## Track G (0.5 day, no training): horizon sweep

Clears the remaining REWRITE_NOTES pending (manuscript Section 4.2). You have
H=8,16,32 from D124/D129; extend to the full sweep for every B1 baseline.

```bash
python scripts/session20/exp_horizon_sweep.py \
    --closure-machinery outputs/session18/exp_b1_test3/ \
    --horizons 1 4 8 16 32 64 \
    --split v2 \
    --out outputs/session20/horizon_sweep/
```

Closure (R^2 and MAE) vs H, one panel per observable group, the horizon at which
each family drops below R^2=0.5, and whether the predictive latent degrades
smoothly while the reconstructive fails abruptly at the drift onset (cross-ref
the D131 drift result and Track C’s H1-lifetime-vs-horizon).

ACCEPTANCE GATE: the closure-vs-H figure exists for all families on test_b and
test_c; the recursive-vs-one-shot agreement claim at H<=16 (from D124) is
confirmed or corrected.

HANDOFF stub: `### D138: horizon sweep -- predictive closure degrades gracefully, reconstructive fails at the drift onset`.

-----

## Dependency graph and scheduling

```
Pre-flight ----+--> Track A (2x2 + A5) ........ long pole, ~1-2 days w/ seeds [GPU0+GPU1 encoders]
               |        |
               |        +--> (A latents feed the held-out eval) 
               |
               +--> Track B (held-out R^2) ..... 1 h, runs immediately [CPU/L40S]
               +--> Track C (persistent homology) 1 day [CPU/L40S, ripser]
               +--> Track D (OT metric+alignment) 1 day [CPU/L40S, POT]
               +--> Track E (phase-amplitude) ... 0.5 day [CPU]
               +--> Track F (scale decomposition) 0.5 day [light GPU for filtering]
               +--> Track G (horizon sweep) ..... 0.5 day [L40S for rollouts]
```

Tracks B-G read only existing v2 artifacts, so launch all of them in parallel
with Track A’s training the moment pre-flight passes. Track A is the only one
that trains encoders and is the only gate on the central claim’s wording; B-G
strengthen and reframe but do not block. Put Track A encoder training on the two
RTX 6000 cards (encoder training stays RTX-only per CLAUDE.md), and run B-G on
the L40S cards and CPU under the existing bypass.

## What this session produces for the manuscript

- `tab:controls_2x2` populated; the abstract and Section 4.5 claim-strength set
  by the Track A gate (the one thing the paper’s headline is currently
  contingent on).
- `tab:b1_r2_heldout` and the `tab:conditioning_floor` JEPA column populated; the
  training-R^2 headline fully retired.
- Section 4.3 mechanism upgraded from one Mahalanobis number to: drift ratio +
  persistent-homology H1 invariant (Track C) + OT-geodesic alignment (Track D-ii).
- Section 4.6 reconstruction comparison reframed from SSIM to OT field distance
  (Track D-i) + scale-decomposition physical interpretation (Track F).
- Section 4.4 gains the limit-cycle definition of recovery and the phase-amplitude
  control connection (Track E).
- Section 4.2 horizon sweep (Track G).
- The world-model framing (WORLD_MODEL_FRAMING.md) added to the Introduction and
  Discussion, with its two caveats.

Every track has a HANDOFF decision stub (D132-D138) so the session’s outcomes
drop straight into your decision log in the existing format.

## A note on honesty, in your project’s style

Three of these tracks can produce a negative or weak result: Track A (the
objective might not separate from the wake head), Track C (the H1 separation
might be marginal), Track D-ii (the OT alignment margin might be small). Each gate
above says what to claim if the result is strong and what to claim if it is weak.
Follow them. The manuscript is already written to survive a weak Track A (it
hedges to “predictive family”); a strong Track A promotes it. Do not let a weak
result get written up as a strong one to match the draft; rewrite the draft to
match the result, exactly as D124(c) and D127 did when their plan gates failed.