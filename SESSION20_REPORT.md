# Session 20 report: closing the manuscript's evidence gaps

Running record. Goal: clear every `\pending` in the rewritten manuscript and add
the five reference-grade analyses, with each track's central claim decided by the
data via its acceptance gate (SESSION20_PLAN.md). Decision stubs D132 to D138.

Status as of this writing: Tracks B, C, D, E, F, G complete (d=64 and, where the
user requested, d=32). Track A (the 2x2 objective x architecture controls) is the
long pole, training on all four GPUs; its gate sets the central-claim wording.

## Infrastructure landed

- New encoder `CNNOnlyEncoder` in `src/models/encoder.py` (the hybrid CNN stem,
  ViT removed, same BatchNorm latent projection; 1.94M params vs hybrid 6.68M):
  the A2/A4 architecture-axis control.
- `train_jepa.py` gains `--encoder {hybrid,cnn_only}` (A2/A5).
- `FukamiAEWrapper` gains `encoder_kind {cnn,cnn_vit}`; `session9_train_fukami.py`
  gains `--encoder` (A3/A4). The Fukami trainer already carried the wake head
  (Session 11), so A4 needed no new code.
- `encode_baseline_latents.py` now rebuilds the right encoder for the new
  checkpoints (cnn_only JEPA, cnn_vit Fukami).
- Resource note: user authorised all four GPUs this session. Track A trains 12
  encoders in one wave, memory-aware: heavy cnn_vit cells (A3, A5) on the two
  RTX 6000 (cuda 2,3, 102 GB), light CNN cells (A2, A4) on the two L40S
  (cuda 0,1, 48 GB), 3 per RTX and 2 per L40S, `expandable_segments` on.
  Encoder provenance (RTX vs L40S) recorded per run; flagged for the methods note.

## Track B (D133): held-out R^2 and the conditioning-floor fix

`scripts/session20/exp_closure_r2.py` reproduces the canonical closure CSV exactly
(MAE matches `physical_closure_noBN_unified.csv` to 0.0) and adds held-out R^2.

Key distinction surfaced: the draft's "held-out forward closure" MAE table was the
`z_dns` mode (probe on the simulation-encoded latent = representation quality),
not the Markov rollout its caption claimed. Both are now reported and labelled:
- Representational closure (z_dns), test_b H=16: JEPA d=64 wake R^2 = 0.754;
  Fukami d=64 -0.406; POD d=64 -0.310. JEPA mean over six observables 0.647.
- Forecast closure (z_markov rollout), test_b H=16: JEPA d=64 wake R^2 = 0.449
  (only positive), d=32 0.214; Fukami d=64 -0.478; POD d=64 -0.089. JEPA mean
  0.445 vs 0.147 (Fukami d=64) and 0.147 (POD d=64).
- Honest nuance now visible in the conditioning-floor table: the wake forecast
  (0.449) is about level with the parameter floor (0.482) at H=16, while the
  representation (0.754) clears it; the latent's forecasting edge on wake is real
  but modest, its representational edge is large.
- test_c (OOD): JEPA wake forecast R^2 stays positive (0.33 at d=64, 0.45 at
  d=32) while all others fall to -1.1 to -1.5; only C_L is forecast well by any
  family there (0.79), consistent with the 2D-to-3D boundary.

Manuscript: `tab:b1_r2_heldout` populated (forecast R^2), `tab:conditioning_floor`
JEPA column filled, `tab:b1_mae_testb` caption corrected to representational
closure, abstract lead restated as the forecast R^2-sign-flip.

## Track C (D134): persistent homology of latent trajectories

`scripts/session20/exp_persistent_homology.py` (ripser, Vietoris-Rips H0/H1).
GATE on the hypothesised lifetime ratio: NOT met, and the descriptive route is
the honest one. The H1-lifetime ratio rollout/DNS does not separate (JEPA 0.66,
Fukami 1.05; Fukami's loop survives because its drifted rollout is a large
diffuse blob). The robust scale-free signal is the GENERATOR COUNT of the
simulation-encoded latent: JEPA encodes a clean single cycle (test_b median 1
significant H1, 55% exactly one), Fukami fragments (median 3.5, 71% with >=3);
Mann-Whitney p = 4.4e-8, same direction on test_c (p = 0.04). d=32 agrees.
D123 cross-check passes (impact is a curvature dip, trough ratio 0.815, smooth
pass-through). Manuscript 4.3 rewritten to lead with generator count, not
lifetime preservation.

## Track D (D135): OT field metric + OT-geodesic latent alignment

`scripts/session20/exp_ot_field_and_alignment.py` (POT, unbalanced KL Sinkhorn,
Tran et al. JFM 2026 signed split, fields pooled to 48x24).
- D-i GATE PASS: OT field distance (test_b impact) Fukami 11.25 (worst), JEPA
  d=64 9.90, POD 9.95. The instructive correction is POD: highest SSIM (0.69)
  but no transport advantage over the blurry JEPA decode, so OT reorders where
  SSIM misleads. test_c ordering breaks (in-envelope result).
- D-ii GATE PASS: per-encounter Spearman of latent distance vs simulation
  OT-geodesic, test_b: JEPA d=64 0.630, d=32 0.607, Fukami 0.449 (margins +0.18,
  +0.16, both above the +0.15 target); JEPA more faithful on 36/42 encounters.
  Honest flag: pooled Spearman reverses (a latent-norm-scale confound), so the
  per-encounter mean is the reported statistic.
Manuscript 4.3 transport-geometry and 4.6 OT-field blocks filled.

## Track E (D136): limit-cycle and phase-amplitude reading

`scripts/session20/exp_phase_amplitude.py`. GATE PASS (qualified, both d=64 and
d=32). Baseline no-gust latent is a closed orbit (return distance 1.0% of
diameter; period ~56 frames, St ~ 0.36). Return-to-orbit under rollout: JEPA
contracts toward the orbit, Fukami expands away; median return-to-orbit smaller
for JEPA at every horizon, bootstrap-robust at H=64 (8.70 vs 9.96, 95% CI on the
paired difference [1.13, 3.25]), marginal at H=32. Load-bearing caveat reported:
the simulation gust trajectories do not fully return within the 120-frame window,
so the comparison measures drift direction, not completed recovery. SINDy
optional fit is dense and low-R^2 (orbit is >2-D); reported as negative,
non-gating. Manuscript 4.4 filled; unifies with Track C (limit cycle = H1 cycle).

## Track F (D137): scale decomposition of the wake-observable advantage

`scripts/session20/exp_scale_decomposition.py` (Gaussian split, sigma/c=0.05).
GATE PASS on test_b. Large-scale wake enstrophy tracking vs simulation: JEPA d=64
correlation 0.89/0.91 (impact / H=16), relative error 0.22; Fukami 0.23/0.61,
relative error ~0.8 (retains only 16 to 20% of large-scale amplitude, near-zero
small-scale energy). POD has comparable correlation but worse amplitude (rel err
0.34 to 0.38). So the claim is specific: the predictive latent tracks the
simulation's large-scale LEV and shear-layer structure better than the
reconstructive AE on both correlation and amplitude, and better than POD in
amplitude. d=32 holds; test_c degrades (0.91 to 0.65), the 3-D boundary.
Manuscript 4.6 scale-decomposition block filled.

## Track G (D138): horizon sweep

`scripts/session20/exp_horizon_sweep.py` over H in {1,4,8,16,32,64}. Graceful vs
abrupt: JEPA d=64 and d=32 hold wake-enstrophy forecast R^2 above 0.5 to H=16
then decline; Fukami d=64 and POD d=64 are already below 0.5 at H=1 and negative
by H=16. The abrupt wake failure coincides with the rollout-drift onset.
Manuscript 4.2 filled.

## Track A (D132): the decisive 2x2 -- DONE

5 cells x 3 seeds at d=64, aux heads matched, unified no-output-BN predictor +
B1 probe. Held-out test_b wake-enstrophy R^2 at H=16 (mean +- std, 3 seeds):
  A1 predictive CNN+ViT lift+wake  0.463 +- 0.034   (mean R^2 0.454)
  A2 predictive CNN     lift+wake  0.445 +- 0.062   (0.522)
  A3 reconstructive CNN+ViT lift+wake 0.160 +- 0.272 (0.228)
  A4 reconstructive CNN lift+wake  0.287 +- 0.048   (0.418)
  A5 predictive CNN+ViT lift only  -1.030 +- 0.289  (0.090)

Two-part honest verdict (both stated; automated gate said
PREDICTIVE_OBJECTIVE_WINS but A5 forces the co-necessity caveat):
1. Predictive beats reconstructive on wake closure at BOTH architectures with aux
   matched (A1>A3 by 0.30, A2>A4 by 0.16). CNN vs CNN+ViT does not separate
   (A2 CNN leads the mean), so the ViT is not the driver: the objective, not the
   architecture, carries the wake advantage.
2. The wake head is NECESSARY: removing it (A5) collapses wake closure to -1.03.
   So the advantage is the predictive objective trained WITH wake supervision,
   not the objective in isolation; the reconstructive cells carry the same head
   and do not reach predictive closure, so the head alone is not sufficient.
Abstract + Section 4.5 + tab:controls_2x2 set to this.

Two load-bearing bug fixes en route (see D132): the Fukami Test-A diagnostic
crashed on a wake-head eval batch lacking wake_target (fixed: skip wake loss when
absent); and the Fukami encode needed non-strict state_dict load (wake head
unused at encode) plus 5D input for the cnn_vit encoder. All four GPUs used this
session (user-authorised); Fukami cells evaluated from their converged
iter-10000 checkpoint.

## World-model framing

Folded into the Introduction (one paragraph after the JEPA paragraph) and the
Discussion control subsection (the actuation-channel sentence), with both
caveats stated inline: the gust is an observed intervention not an agent's
action (the model is the world model a controller would use, not a controller),
and we keep to the interventional reading without claiming a full causal model.
