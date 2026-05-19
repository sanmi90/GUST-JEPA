# Section 7: Ablation suite

LaTeX-friendly markdown. Approximate target length: 4 to 6 pages
once the full thin cut + d-sweep + Section 7 ablations land in
Session 10. This is the Session 9 skeleton; numerical entries will
be filled in as each ablation completes.

## 7.1 The ablation matrix

Sections 5 and 6 establish the production configuration:
SIGReg + observable-head (OBS) + BatchNorm projection at latent
dimension d=32, observable weight eta=0.01, and SIGReg weight
lambda=`{LAMBDA_STAR}` (the Session 9 bisection winner; the Session
8 grid value of lambda=0.01 is the lower bound on the value reported
here). Section 7 takes that production configuration as a fixed
centre cell and varies one design choice at a time. The ablation
suite is organised into four families.

The four families are:

1. **Anti-collapse regulariser family.** SIGReg (production), VICReg
   (Bardes et al. ICLR 2022; ablation A2 in this section), PLDM
   five-term (Sobal et al. arXiv:2502.14819, 2025; ablation A1 lives
   in Section 5 as the head-to-head comparator); plus controls with
   the regulariser disabled (ablation A3, lambda=0 SIGReg) or with
   the OBS head disabled at full strength (R0, lambda=lambda\* with
   no head; reported in Section 5.7).
2. **Conditioning family.** Predictor conditioning on
   `c = (G, D, Y)` enabled (production, D6) vs. ablation A4
   c-dropout at 0.5 (D32 motivation), vs. ablation A5 c-removed
   entirely from the predictor (`--predictor-cond-dim 0`,
   Session 6 F-NC variant), and ablation A6 with c added to the
   encoder as well (the deliberate negative-result run noted in
   CLAUDE.md "Things to NOT do").
3. **Training-procedure family.** Scheduled sampling at
   H_roll=8 (production, D21) vs. ablation A7 with H_roll=T=32
   (no scheduled sampling; full open-loop rollout); ablation A8
   with H_roll=1 (teacher forcing only, no rollout exposure);
   ablation A9 with the predictor's static-condition position
   removed (`--c-dropout-prob 1.0`, the no-c baseline at
   inference but training-time-only).
4. **Comparator-architecture family.** Solera-Rico
   beta-VAE + transformer ROM at matched d=32 (ablation A10);
   Fukami observable-augmented autoencoder at matched d=32
   (ablation A11); POD with d=32 components as the linear floor
   (ablation A12); plus three reserved slots A13-A15 for
   training-data ablations (frame skip, sub-trajectory length,
   sub-trajectory impact-aware sampling fraction) to be specified
   if reviewer feedback warrants them.

Table 2 organises this matrix; the entries marked `(S5)` are
already reported in Section 5 and copied here for completeness;
entries marked `(S8)` come from Session 8 (Section 5.6 d-sweep);
entries marked `(S9)` are this paper's Session 9 thin cut; entries
marked `(S10)` are deferred to Session 10. The Test B delta column
is the headline metric (Section 3.5); higher is better, with the
production configuration anchoring at delta\_test\_b = `{DELTA_PROD}`.

## 7.2 Session 9 thin-cut results

Three ablations landed in Session 9 at the production configuration
plus the bisection winner lambda\*; A10 (Solera-Rico) is deferred to
Session 10. The three landed ablations:

- **A2 VICReg + OBS at d=32, eta=0.01.** Same encoder, predictor,
  scheduled sampling, OBS head as production; the SIGReg term is
  swapped for the VICReg variance-covariance objective at the
  Bardes et al. canonical weights (mu=25, lambda\_var=25, nu=1; D22).
  Test B delta = `{A2_DELTA}`.
- **A7 no scheduled sampling (H\_roll=T=32) at d=32, eta=0.01,
  lambda=lambda\*.** Production configuration but with the full-
  rollout exposure. Test B delta = `{A7_DELTA}`.
- **A11 Fukami observable-augmented autoencoder at d=32.** The
  Fukami and Taira J. Fluid Mech. 2023 lift-augmented autoencoder
  (`\cite{fukami2023}`) adapted from the published 240x120 input to
  our 192x96 mid-plane vorticity cache, with the four 2x maxpool
  stages chosen to reach the same (12, 6, 4) bottleneck as Fukami's
  three 2-2-5 maxpools; the FC chain 256-64-32-16-d matches Table S.1
  of the supplementary, with d=32 replacing Fukami's d=3 to match
  the matched-capacity comparison. Implementation at
  `src/baselines/fukami_ae.py` (240K params at d=32; ~40x smaller
  than the JEPA's 10M, reflecting Fukami's intentionally lightweight
  architecture). Trained jointly on
  `lambda_recon * MSE(omega, omega_hat) + lambda_lift * MSE(CL, CL_hat)`
  with `lambda_recon = lambda_lift = 1` per the paper. Test B delta
  = `{A11_DELTA}`. SSIM (Eq. 1 of Fukami's supplementary,
  `C_1 = 0.16`, `C_2 = 1.44`) is reported alongside MSE on Test A / B / C
  for direct comparability with the JEPA's decoder reconstruction
  (Section 6).
- **A10 Solera-Rico beta-VAE + transformer ROM at d=32.** The
  Nat. Commun. 2024 two-stage architecture (beta-VAE Stage 1 followed
  by a transformer ROM on the frozen latent, Stage 2). Deferred to
  Session 10: a faithful reproduction requires both stages, and
  the two-stage training fitting cleanly inside the cuda:1 idle
  window between A11 and A7 is too tight a safety margin at the
  observed external-load compute rate. The
  `src/baselines/solera_rico.py` module is the first Session 10
  deliverable.

The Session 9 thin-cut deliverable was scoped to keep the
wall-clock budget honest. A10 remains deferred to Session 10 along
with the remaining ablations from Section 7.1; A11 lands in Session 9
per a mid-session scope addition triggered by the explicit user
request to compare against the Fukami SSIM-based methodology.

## 7.3 Latent dimension sweep (recapitulated from Section 5.6)

Three SIGReg + OBS + BN runs at the Session 8 grid winner (eta=0.01,
lambda=0.01) with latent dimension `d in {8, 16, 32}` (D54) span
the LeWM Two-Room intrinsic-dimension prediction range. The
production d=32 wins on Test B by +0.07 absolute over d=8 (Table 3
in Section 5.6); the participation ratio PR\_all is flat in d
(~2.4 across all three latent dimensions), so the d=32 advantage is
not driven by more effective dimensions being used by the encoder
but by the downstream linear probe having more interpolation freedom
on Test B. The LeWM Two-Room intrinsic-dimension mechanism does not
extend to the SIGReg + OBS + BN regime where the observable head
dominates as a directional pressure.

## 7.4 Remaining ablations (Session 10 work)

The conditioning family (A4 c-dropout, A5 c-removed, A6 c-encoder),
the training-procedure family (A8 H\_roll=1, A9 c-dropout inference),
the comparator-architecture family (A10 Solera-Rico, A11 Fukami,
A12 POD), and the reserved slots (A13-A15) are deferred to Session 10
per the Session 9 risk-register decision to keep the bisection
+ decoder budget honest. The A10 and A11 baseline modules will be
written before Session 10's compute window and will run at the
matched d=32, eta=0.01, lambda=lambda\* configuration to maintain
the matched-capacity head-to-head with production. POD is a one-shot
analysis script and does not require GPU training; it lands as part
of Session 10's notebook deliverables.

The Session 9 thin cut (Section 7.2) is the smallest sample that
allows a paper-grade claim about the production configuration's
robustness to anti-collapse regulariser choice (A2 VICReg) and
training-procedure choice (A7 H\_roll). The remaining Session 10
ablations sharpen the conditioning-family results and complete the
matched-capacity comparator-architecture table.

---

**Table 2 skeleton (placeholder).** To be filled in as each ablation
completes. Columns: ablation code, family, description, Test A delta,
Test B delta, Test C delta, PR\_all (Test B), r2(z->c) (Test B), source
(S5 / S8 / S9 / S10).
