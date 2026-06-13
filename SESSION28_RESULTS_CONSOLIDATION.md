# Session 28 results consolidation (v2.1 unconditioned) -- Phase D drafting map

Single source for the Phase D restructure. Every number is macro-bound in
`paper/macros.tex` (regenerate: `eval_all.py` -> `emit_macros.py`; 117 numbers /
10 parts / 166 macros as of 2026-06-13). Decision log: HANDOFF D178-D195 +
the 2026-06-13 entries. Framing rules: no conditioning remarks (model is the
default, [[unconditioned-jepa-rebuttal]]); feature strong results, not a lab
report ([[feedback-paper-not-lab-report]]).

## The spine (headline-grade, survived fair scrutiny)

1. **Closure: the wake is LINEARLY READABLE from the predictive latent.**
   Representational wake-enstrophy R2 at H=16 test_b d=64: jepa_tf_noc +0.79;
   every published-recipe baseline negative (fukami -0.25, pod -0.16,
   bvae_faith -0.19) under ALL THREE probe classes; beats the model-free
   (G,D,Y) parametric floor. Primary paired delta vs Fukami +33.6, case-
   clustered CI [+7.8,+58.7]; survives the pre-registered (encounter-level)
   Holm (4/12), NOT the stricter case-level Holm (2/12) -- report both.
   GATE GD = WEAK (statistically confirmed): claims are LINEAR DECODABILITY,
   title carries no possession verb, "no probe in the evaluated class".
2. **State recovery from sparse wall pressure (sensing, MAIN result).** Predictive
   latent most recoverable: state R2 at K=8 test_b jepa 0.78 > fukami 0.66 >
   bvae 0.51 > pod 0.34, seed-robust 0.75 +- 0.09. Field recovery: jepa >= any
   matched rep, significantly > AE; POD field-parity is a linear-decoder
   ceiling artefact (reported with the ceiling gap). qDEIM primary; pressure->Y
   0.51; causal-clip delta negligible. NO pressure->wake claim (dropped).
3. **Spectrum / limit cycle (answers "isn't JEPA a DMD?").** JEPA preserves the
   shedding St (DMD-on-latent 0.662 vs DNS 0.675 ~ POD 0.682), marginally
   stable orbit (Floquet leading multiplier 1.004), and BEATS the DMD/linear-
   dynamics rung on forecast wake R2 by +0.82 (jepa +0.43 vs POD-DMD -0.39).
   Koopman-AE NOT needed: the DMD rung settles the linear-latent-dynamics axis.
3b. **Distributed wake code (P5).** The wake forecast is a collective/distributed
   JEPA code (combo-minus-best-single gap +0.36 vs baselines ~0.04; energy-info
   knee 32 PCs vs POD's 1), beating the model-free parametric floor (wake floor
   NEGATIVE, -0.18; latent +0.79 clears it by +0.91; forces floors high C_L
   +0.61 sharpen the wake-vs-forces distinction). Footprint-localization claim
   CUT (saliency below chance). Scale-band trend robust (S4 closed).
4. **Model-free DNS physics.** |G| degradation of wake closure (Spearman -0.56,
   p 0.0022); shedding-phase RESET by the gust (decorrelates at |G|>=0.25,
   G-monotone mean phase step); G-sign LEV asymmetry (negative gusts ->
   larger longer-lived LEV, detachment 0.35 vs 0.12 t/c, p 0.024); undisturbed
   validation C_L/C_D within 1.7-3.3% of PRF fine grid, St lines 0.675/0.34.

## Section-by-section disposition (mapped from the planned S4 order)

|Section|v2.1 verdict|Home|
|---|---|---|
|S4.1 Closure (repr-led)|spine #1; weak-GD linear-decodability framing|HEADLINE|
|S4.2 Attribution 2x2|wake head + predictive objective; matched controls; parametric (model-free) floor kept|MAIN|
|S4.3 Mechanism: drift|robust departure-spectrum (AE exits near-null dirs, sign p 2e-13); complete Table 8|MAIN (lead), 9x ratio NOT headlined (estimator-fragile)|
|S4.3 Mechanism: topology|GUSTED fragmentation = metric artefact (CUT as topology); no-gust limit-cycle survives (fukami fragments clean cycle)|limit-cycle result MAIN-or-appendix; gusted claim CUT; fix abstract (fragmentation=ENCODING)|
|S4.3 Mechanism: transport|v2 "JEPA tracks transport geometry" DEAD|CUT ENTIRELY (dissolves M5)|
|S4.4 Flow physics|P1 (|G| trend), P3 (phase reset), P4 G-sign LEV asymmetry|MAIN; P4 GP4 corr + P1 collapse-fit -> APPENDIX|
|S4.5 Latent code (P5)|collective/distributed code REPRODUCES (gap +0.36 vs ~0.04; energy-info knee jepa 32 PC vs pod 1); saliency FOOTPRINT reverses (below chance, D163 dropped)|distributed-code + energy-info MAIN/appendix; footprint-localization CUT|
|S4.6 Physical-space decode|decode-ceiling + matched post-hoc decoders exist; LEV-tracking-by-family (D146) DEAD on matched decoders (POD wins via decoder)|decode-ceiling figure kept; latent-LEV-tracking claim CUT|
|S4.7 Sensing|promoted to spine #2|MOVED UP to MAIN|

## Cut or demoted (do NOT headline; appendix or gone)

- Transport-consistency / "tracks OT geometry" -- CUT (M5 dissolves).
- Gusted-encounter topological fragmentation -- CUT as topology (metric artefact).
- Latent-LEV-tracking by family (D146) -- CUT (matched-decoder confound).
- pressure->wake recovery -- DROPPED (not the goal; comparison is state+field).
- Conditioned reference + all conditioning remarks -- CUT (reverses AD2).
- Mahalanobis "order of magnitude" 9x -- not headlined (estimator-fragile;
  lead drift with the departure spectrum).
- GP4 peak-DCL-vs-Gamma_LEV (Pearson 0.53) -- APPENDIX (Spearman 0.75, monotonic).

## Abstract / title consequences

- TITLE: option (a) "A predictive latent representation preserves wake structure
  in vortex-gust airfoil interactions at Re=5000"; option (c) "Transport-
  consistent..." is DEAD. With GD weak, no possession verb -> consider
  "renders wake structure linearly readable".
- ABSTRACT: delete the transport-tracking sentence and the conditioned-reference
  clause; attribute fragmentation to the ENCODING not the rollout; lead the
  closing with sensing STATE recovery (main result), not deployment hedging.

## Table consistency status (Phase D D6, 2026-06-13)

v2.1-consistent (macro-wired or regenerated from v2.1 outputs): tab:closure
(headline, both panels), tab:paired_closure (B6 12-test, encounter-level Holm
survivors bolded), tab:controls_2x2 (unconditioned representational 2x2),
tab:parametric_floor, tab:latent_drift, tab:families (caption). tab:architecture
unchanged (architecture). tab:dns_pending = collaborator \pending{}.
REMAINING v2 table: tab:b1_closure_train_r2 (appendix training-fit reference) --
still v2 numbers; cannot be regenerated from the closure matrix (the train-split
FORECAST endpoint was not computed, only test_b/test_c forecast + train
representational). Closing it needs a train-split rollout of the 11 predictors;
low priority (secondary in-distribution reference, explicitly "not the held-out
result"). jepa_dimsweep_table.tex is an orphan (no \input), conditioned content,
harmless; delete at cleanup.

## Remaining before Phase D assembly

- P5 (wake-code: collective code, energy-information curve, saliency footprint)
  -- run with the spine-pattern caveat (latent-code claims may not survive).
- E4 scale-band sensitivity (cheap appendix sentence).
- Parametric (model-free) floor regenerated on v2p1 (D145 machinery) for the table.
- Phase D: macro-wire (numbers.json done), restructure per the table above,
  regenerate figures/tables with _v2p1 suffix, strip ~62 conditioning mentions
  in the D1/D2 pass, build gates.
- EXTERNAL: DNS Table 1 (collaborator); DNS package drafted, send pending.
