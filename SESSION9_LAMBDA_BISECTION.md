# SESSION9_LAMBDA_BISECTION.md

Session 9 plan: lambda bisection at the Session 8 production point,
plus the visualisation decoder on the frozen winner and a first
five-point slice of the Section 7 ablation matrix.

Last updated: 2026-05-19, immediately after Session 8 VALIDATED (D57).

## Framing

Session 8 (D50-D57) located the production operating point of the
SIGReg + OBS + BN configuration at d=32, eta=0.01, lambda=0.01 with
Test B delta = +0.159 (E4). The (eta x lambda) grid showed that at
eta=0.01 the lambda axis trends monotonically downward from 0.01 to
1.0; at eta=0.1 the lambda axis is essentially flat. The d-sweep at
the Step 4 best point confirmed d=32 (LeWM intrinsic-dim prediction
not transferred). The R0 control confirmed OBS is load-bearing.

What Session 8 did not do: locate the precise lambda. The grid only
sampled lambda in {0.01, 0.1, 1.0} and did not test lambda < 0.01.
The monotone-decreasing pattern of the eta=0.01 row leaves open
whether the true optimum is at lambda=0.01 (the lowest sampled value),
at lambda<0.01, or actually at lambda very close to 0 (SIGReg-off).

Session 9 closes this with a LeWM-style bisection (\cite{lewm}
Appendix G) over lambda in [0.001, 0.1] at the production
(d=32, eta=0.01, OBS=cl_future). It also begins the post-Session 8
work that the architecture spec called out as Session 7 evaluation
deliverables but that Sessions 4-8 deferred: the visualisation
decoder on the frozen winner, plus a first cut of the 15-ablation
matrix.

The honest-checkpoint discipline of Session 7 + 8 carries forward:
every bisection point gets reported regardless of outcome, and the
session report categorises the result as PRODUCTION_LOCKED,
PRODUCTION_REFINED, or PRODUCTION_PIVOT.

## What this session does NOT do

- A full PLDM hyperparameter sweep. D53b ruled it out: E10
  paper-tuned PLDM (-0.095) is worse than R1 default-PLDM (-0.003).
  No further PLDM tuning is justified before the paper goes out.
- Solera-Rico and Fukami baselines at full evaluation. These are
  Section 7 ablations 10 and 11 in the architecture spec; they are
  scheduled here as a thin baseline cut (1 run each at the matched
  d=32) but the per-baseline tuning is deferred to Session 10+.
- Multi-seed averaging on every bisection point. Single seed=0 for
  the bisection; multi-seed (seed in {0, 42, 123}) only on the best
  bisection point as a paper-grade variance bound.
- The full 15-ablation Section 7 matrix. Five priority ablations
  this session; the remaining ablations land in Session 10.
- Cache extensions for additional observable targets. The C_D and
  p_LE evaluation from D51 is sufficient; no new targets needed.

## Session goal and step structure

Five steps. Steps 1 and 2 are sequential (Step 1 informs Step 2),
Steps 3 and 4 run in parallel with Step 1's compute, Step 5 is
the paper writing during all compute windows.

|Step  |Output                                                                                                                                                                                                  |Wall-clock     |GPU?           |
|------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|---------------|
|Step 1|Lambda bisection: 5 new evaluations at lambda in {0.001, 0.003, 0.03} on cuda:0 sequentially plus the seed-variance bound at the best lambda. The E4 (lam=0.01) and E5 (lam=0.1) results are anchors.|~10h           |cuda:0 + cuda:1|
|Step 2|Visualisation decoder training on the frozen Step 1 winner.                                                                                                                                             |~2h            |cuda:0         |
|Step 3|Section 7 ablation thin cut: ablation 2 (VICReg variant at production config), ablation 7 (no scheduled sampling), ablation 10 (Solera-Rico beta-VAE), ablation 11 (Fukami AE).                          |~6h wall-clock |cuda:0 + cuda:1|
|Step 4|R0 at the bisection winner's lambda, if different from the lambdas already tested (Session 8 R0 covered lambda in {0.01, 0.1}; if winner is < 0.01 also run R0 at the winner's lambda).                  |~1.5h          |cuda:0         |
|Step 5|Paper writing during the compute windows: Section 6 (decoder results), Section 7 outline + thin-cut table, Sections 1 + 2 drafts, Abstract.                                                              |~4-5h overlap  |No             |

Total session wall-clock: roughly 14-16 hours from launch. Agent-active
time: ~6-8 hours with paper writing during compute. The compute is
parallelisable across two RTX 6000 cards per D40.

## Pass criteria for Session 9

1. Step 1 bisection completes; best lambda* identified with Test B
   delta numbers from at least 5 lambda values centred on lambda=0.01.
2. Step 1 seed-variance bound at lambda*: best Test B delta within
   +/- 0.03 of the seed=0 result (the +/- 0.03 matches the seed=42 vs
   seed=0 spread observed on R3 in D52).
3. Step 2 visualisation decoder reconstructs omega_z on Test A with
   per-frame MSE within 2x of the floor (where "floor" is the
   reconstruction MSE of the frozen identity map; see Step 2 details).
4. Step 3 thin-cut ablations land Test B delta numbers for each of
   the four ablations. No specific pass threshold; the result is
   what it is.
5. Step 4 R0 at lambda* completes if needed.
6. Step 5 commits Section 6 (decoder), Section 7 outline + Table 2
   skeleton, Sections 1 + 2 drafts, and an Abstract draft.

The Session 9 outcome category is one of:

- **PRODUCTION_LOCKED**: bisection finds the same lambda*=0.01 as
  Session 8 Step 4. Production config is unchanged; the paper claims
  rest on E4. Most likely outcome.
- **PRODUCTION_REFINED**: bisection finds a different lambda* in
  [0.001, 0.1]. Production config updates; Section 5 of the paper
  gets a one-paragraph update; Session 10 ablations run at the
  refined lambda*.
- **PRODUCTION_PIVOT**: bisection finds a lambda* with substantially
  higher Test B delta (> +0.20) OR the seed-variance bound at lambda*
  exceeds +/- 0.05. Either result triggers a deeper investigation
  before publication.

## Step 1: lambda bisection (~10 hours wall-clock)

The Session 8 Step 4 (eta x lambda) grid sampled lambda in
{0.01, 0.1, 1.0} at eta=0.01 with results +0.159, +0.138, +0.093 (E4,
E5, E6). The trend is monotonically decreasing in lambda, suggesting
the optimum is at lambda <= 0.01. Session 9 Step 1 evaluates three
new lambda values plus two seed-variance points at the winner.

Lambda values to evaluate (all at d=32, eta=0.01, OBS=cl_future):

| Run code | lambda | Already trained? |
|----------|-------:|------------------|
| F1       | 0.001  | New              |
| F2       | 0.003  | New              |
| (anchor) | 0.01   | E4 (Session 8)   |
| F3       | 0.03   | New              |
| (anchor) | 0.1    | E5 (Session 7)   |

Three new SIGReg + OBS runs at 1.5h each. Plus the two anchors are
re-used from disk; no re-training needed.

After F1, F2, F3 land, identify lambda* = argmax delta_test_b across
{F1, F2, E4, F3, E5}. Then:

- **F4 seed=42 at lambda***: 1.5h. Bound on the seed variance at
  the new operating point. Pass criterion: F4 within +/- 0.03 of
  F1/F2/E4/F3 at the same lambda*.
- **F5 seed=123 at lambda***: 1.5h. Third seed for the paper-grade
  variance bound.

cuda:0 sequence: F1, F2, F3, F4, F5 (5 runs x 1.5h = 7.5h).
cuda:1 sequence: Step 3 thin-cut ablations (4 runs x 1.5h = 6h).
Total Step 1 + Step 3 wall-clock: 7.5h.

Launch template (vary lambda per run):

```bash
python -m src.training.train_jepa \
    --gpu 0 \
    --partition v1 --all-train --max-iters 20000 --seed 0 \
    --latent-dim 32 \
    --observable-head cl_future --observable-head-weight 0.01 \
    --observable-head-deltas 8 16 24 \
    --projection-norm batchnorm --anticollapse sigreg \
    --lambda-sigreg <varies> \
    --diagnostic-every 500 --checkpoint-every 2000 --log-every 50 \
    --output-dir outputs/runs/session9/run_f<n>_lam<value> \
    --wandb-mode offline \
    --tag-suffix run_f<n>_lam<value>_bisection
```

### Analysis

After F1-F5 land, build `notebooks/10_session9_lambda_bisection.ipynb`:

- Test B delta vs lambda curve over {0.001, 0.003, 0.01, 0.03, 0.1},
  with the F4 and F5 seed-variance error bars at lambda*.
- Test A and Test C delta on the same axis.
- PR_all and r2(z->c) at each lambda, to see whether the optimum
  moves the latent regime.
- The "production" cell summary at lambda* with seed-variance.

The lambda bisection's headline result is the table:

| lambda | seed | Test A | Test B | Test C | PR_all | r2(z->c) |
|-------:|-----:|-------:|-------:|-------:|-------:|---------:|
| 0.001  |    0 |   TBD  |   TBD  |   TBD  |   TBD  |   TBD    |
| 0.003  |    0 |   TBD  |   TBD  |   TBD  |   TBD  |   TBD    |
| 0.01   |    0 | +0.227 | +0.159 | +0.470 |  2.61  |   0.87   |
| 0.03   |    0 |   TBD  |   TBD  |   TBD  |   TBD  |   TBD    |
| 0.1    |    0 |   TBD  |   TBD  |   TBD  |   TBD  |   TBD    |
| lambda*|   42 |   TBD  |   TBD  |   TBD  |   TBD  |   TBD    |
| lambda*|  123 |   TBD  |   TBD  |   TBD  |   TBD  |   TBD    |

The E5 anchor Test B delta is +0.138 (Session 7 R3); the E5 Test A
and Test C values come from the Session 7 evaluation.

### What the bisection could reveal

Expected outcome (most likely): lambda* = 0.01 (no change from
Session 8). The Test B delta curve plateaus near the lower end of the
sampled range. Production config is locked.

Surprise outcome A: lambda* < 0.01 (lambda* in {0.001, 0.003}). SIGReg
needs to be even smaller than Session 8 thought. The paper claim 3
strengthens: "SIGReg at lambda close to zero is what works; the
observable head is essentially the only regulariser."

Surprise outcome B: lambda* > 0.01 (lambda* = 0.03 or higher). The
Session 8 grid sampling was too coarse and missed a local optimum
between 0.01 and 0.1.

Surprise outcome C: the seed-variance at lambda* exceeds +/- 0.05.
The +0.159 finding is less robust than D52's seed=42 suggested.
Triggers PRODUCTION_PIVOT; investigate before paper claims.

## Step 2: visualisation decoder (~2 hours, cuda:0 after Step 1)

Train a separate decoder on the frozen Step 1 winner encoder. The
decoder takes z in R^32 and reconstructs omega_z in (192, 96). It
is NEVER part of the JEPA loss (CLAUDE.md "Things to NOT do").

### Architecture

Mirror the encoder: a 6-layer ViT decoder (hidden 256, 8 heads) on a
spatial-token grid of 24x12 = 288 tokens, followed by 3 upsampling
conv stages that map the (24, 12, 256) feature map back to
(192, 96, 1). Total ~10M params, comparable to the encoder.

Pre-norm transformers (no AdaLN since the decoder is unconditional).
The encoder's final BatchNorm-projected z gets a linear "back-
projection" head to seed the decoder tokens (32 -> 288 x 256).

### Training

- Loss: per-frame MSE on omega_z, summed over (T, H, W).
- Optimizer: AdamW (0.9, 0.95), wd 0.05.
- LR: 1e-4 (lower than the encoder because we are only fitting one
  pathway, not balancing multiple losses).
- 10k iterations on the train partition (138 train encounters).
- bf16 mixed precision.

### Evaluation

After training, encode and decode every Test A and Test B encounter.
Report per-encounter MSE distribution and the spatial-mean MSE pattern.
Visual deliverable: a 3x3 grid for one Test B encounter showing
(raw omega_z, decoded, residual) at frame 25, 40, 55 (pre-impact, at
impact, post-impact).

Pass criterion (Section 5.6 of the architecture spec):
reconstruction MSE on Test A within 2x the noise floor. Noise floor =
the MSE of the per-case-mean omega_z field; any decoder reconstructing
above this floor is using the case-mean rather than the latent.

### What the decoder could reveal

Most likely outcome: decoder reconstructs the impact-instant vortex
core and the wake structure clearly. The latent encodes both static
case identity (where the vortex is) and dynamic phase (how it
interacts with the airfoil). Section 6 of the paper writes around
the visual.

Failure mode: the decoder reconstructs only the case-mean (impact at
frame 40, generic Y/c profile). This would mean the latent is
case-identity-dominated and the +0.16 Test B delta is largely
explained by case-identity decoding. The D51 head-ablation argues
against this, but the decoder is the definitive visual test.

## Step 3: Section 7 ablation thin cut (~6 hours wall-clock)

Four ablations from the architecture spec's 15-item matrix, on the
production (d=32, eta=0.01, lambda*=lambda*) configuration. Each is a
single 20k-iter run.

### Ablation 2: VICReg-only (no SIGReg)

The Bardes ICLR 2022 VICReg objective at the same lambda*=lambda*
weight (mu=25, lambda=25, nu=1 per D22). All other configuration
matched to the SIGReg path.

Hypothesis: VICReg also fails Test B without OBS (Session 5 / 5.PLDM
evidence). VICReg + OBS may or may not match SIGReg + OBS at full
scale. If VICReg + OBS > SIGReg + OBS, the paper claim 3 inversion
extends to a third regulariser comparison.

### Ablation 7: no scheduled sampling (full rollout)

Same SIGReg + OBS + BN at d=32, eta=0.01, lambda* configuration, but
with H_roll = T (full rollout) instead of H_roll = 8 (scheduled
sampling). The V-JEPA 2-AC scheduled-sampling pattern from D21 is
swapped for the full-rollout baseline.

Hypothesis: full rollout produces a less stable training trajectory
(diverging at long horizons) and a worse Test B delta. The Session 7
trajectory plot already shows scheduled sampling produces clean
convergence; full rollout's exposure bias should hurt.

### Ablation 10: Solera-Rico beta-VAE + transformer

The Solera-Rico et al. 2024 \cite{solera2024} ROM trained on the
gust-airfoil dataset family. d=32 to match. The beta-VAE encoder
maps omega_z to z; a transformer decodes z to CL prediction directly
(no JEPA structure). Evaluation on Test A / B / C using the same
per-split metric table.

This is the direct ROM baseline from the same data family. If
SIGReg + OBS beats Solera-Rico on Test B, the paper claim 1 (the
diagnostic suite + (c, t) baseline is useful) strengthens, as does
the implicit claim that JEPA latent structure transfers better than
beta-VAE latent.

### Ablation 11: Fukami observable-augmented AE

The Fukami & Taira 2023 \cite{fukami2023} lift-augmented autoencoder
adapted to gust-airfoil. Encoder + decoder + CL prediction head jointly
trained on omega_z reconstruction MSE + CL MSE. d=32 to match.

This is the direct competitor at the JEPA-for-aerodynamics framing.
Fukami's autoencoder approach should give a high Test A r2 (it is
trained to reconstruct) but a poor Test B delta (the latent does not
disentangle case from dynamics).

### Launch pattern

cuda:0 sequence (during Step 1 F4 + F5): VICReg, no-scheduled-sampling
(2 runs x 1.5h = 3h). cuda:1 sequence (during Step 1 F1 - F3 and beyond):
Solera-Rico, Fukami AE (2 runs x 1.5h = 3h).

The two card sequences are interleaved with Step 1's bisection runs;
the orchestrator pattern from Session 8
(`scripts/orchestrate_session8_step4.sh`) is reusable.

### Analysis

`notebooks/11_session9_section7_thin_cut.ipynb`. Same per-split metric
table as Session 7 notebook 05. Compare each ablation to the
production config at the validated lambda*.

## Step 4: R0 at lambda*, if needed (~1.5 hours)

Session 8 Step 6 ran R0 at lambda in {0.1, 0.01}, both fail (-0.74).
If Step 1 finds lambda* < 0.01, run R0 at lambda* to confirm the OBS
necessity claim holds at the refined operating point.

If lambda* == 0.01 (most likely outcome), skip this step.

## Step 5: paper writing (~4-5 hours overlapping compute)

Four paper deliverables during Session 9 compute windows:

### Section 6: visualisation decoder results

- 6.1 Decoder architecture and training (~ half page).
- 6.2 Reconstruction quality on Test A (the noise-floor comparison
  + per-encounter MSE distribution).
- 6.3 Reconstruction on Test B (the parametric interpolation visual
  test): does the decoder produce a plausible vortex-core
  morphology at unseen (G, D, Y)?
- 6.4 Reconstruction on Test C (extrapolation).
- 6.5 Discussion: what the decoder tells us about the latent's
  encoding of static case vs dynamic phase information.

Approximate length: 3-4 pages including the Figure 3 (3x3 spatial
grid for one Test B encounter).

### Section 7 outline + Table 2 skeleton

- 7.1 Ablation matrix (full 15-item list as a table).
- 7.2 Thin-cut results from Session 9 Step 3.
- 7.3 D-sweep results from Session 8 D54 (already in Section 5.6
  but Section 7 gets the same table with the production config
  framing).
- 7.4 Remaining ablations deferred to Session 10.

Approximate length: 2 pages.

### Sections 1 + 2 drafts

- Section 1 (Introduction): the scientific question (vortex-gust
  airfoil interaction, parametric ROM), the JEPA framing, the
  three contribution claims, the paper's roadmap. ~2-3 pages.
- Section 2 (Related work): JEPA lineage (V-JEPA 2, LeWM, LeJEPA,
  PLDM), observable-augmented autoencoders (Fukami, Solera-Rico),
  classical ROM (POD, AE for fluids), and the gap this paper fills.
  ~3-4 pages.

### Abstract draft

~200 words. Three contribution claims with their headline numbers:
+0.159 on Test B for SIGReg + OBS, +0.90 OBS contribution, the
regulariser asymmetry inverting at scale.

## Risk register

|Risk                                                |Probability|Mitigation                                                                                                                                                  |If it fires                                                                  |
|----------------------------------------------------|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
|Bisection finds lambda* far from 0.01               |low-medium |Plan extension points outside the current bisection bracket if the optimum is at the edge; Session 10 expands to lambda in {1e-4, 3e-4, ...} or {0.3, 1.0} |+3h GPU in Session 9, paper Section 5.5 footnote                             |
|Seed variance at lambda* > +/- 0.05                  |low        |Run seed=2026 as a fourth datapoint; report variance honestly                                                                                                |+1.5h GPU                                                                    |
|Decoder reconstructs only case-mean                  |low-medium |Section 5.4 D51 already argues against this, but if it lands, paper claim 1 narrows; Section 6 acknowledges the limitation                                   |Section 6 framing shifts; +0.5h analysis                                     |
|VICReg + OBS beats SIGReg + OBS at scale             |low        |The paper claim 3 gets a third comparison axis; Section 5.3 absorbs the result; the inversion-at-scale finding survives                                      |Section 5.3 + Section 7 wording change                                       |
|Solera-Rico or Fukami baseline beats SIGReg + OBS    |low        |Run the baseline twice (seed=42) before reporting; if reproducible, paper claim 1 narrows to "diagnostic suite is useful"                                    |Sections 5.8 + 7 framing change                                              |
|Compute budget overrun (some runs > 1.5h each)       |medium     |Drop one of the four ablations to make room (deferred to Session 10)                                                                                         |Section 7 thin cut becomes 3 instead of 4                                    |

## D-entries to record

**D58**: Step 1 lambda bisection result. lambda* identified with
seed-variance bound.

**D59**: Step 2 visualisation decoder result. Reconstruction quality
on Test A / B / C and the case-mean comparison.

**D60**: Step 3 Section 7 thin-cut results. The four ablations' Test B
deltas and the implications for paper claims.

**D61** (if run): Step 4 R0 at lambda*.

**D62**: Step 5 paper progress. Section 6 committed; Section 7 outline
committed; Sections 1 + 2 drafted; Abstract drafted.

**D63** (always): Session 9 outcome summary. PRODUCTION_LOCKED,
PRODUCTION_REFINED, or PRODUCTION_PIVOT.

## After Session 9

Session 10 in all cases:

- Remaining Section 7 ablations from the architecture spec's 15-item
  matrix (the 9 that weren't done in Sessions 4-9).
- Multi-seed averages on the production config (seed in {0, 42, 123,
  2026, 31415}) for the paper-grade variance bound.
- Final paper figures (Figure 1 architecture diagram, Figure 2 grid
  heatmap, Figure 3 decoder reconstruction, Figure 4 ablation matrix).
- JFM / PRF manuscript draft.

Session 11 (if needed): revision after internal review, additional
runs for any reviewer-anticipated questions.

## Decision references

- D2: d=32 per LeWM. **Session 8 D54 confirmed; Session 9 production
  uses d=32.**
- D17: SIGReg with BatchNorm projection.
- D37: Observable head at eta=0.01.
- D40: Two RTX 6000 cards; `--gpu {0, 1}` flag.
- D44-D49: Session 7.
- D50-D57: Session 8.
- D58-D63: this session.

## Key references

- LeWM lambda bisection (\cite{lewm} Appendix G): the bisection
  algorithm template used in Step 1.
- PLDM (\cite{pldm}, arXiv:2502.14819): the comparator baseline at
  the matched-capacity head-to-head (E10 paper-tuned, D53b).
- Solera-Rico et al. (\cite{solera2024}, Nat. Commun. 2024): the
  beta-VAE + transformer ROM baseline (Step 3 Ablation 10).
- Fukami & Taira (\cite{fukami2023}, J. Fluid Mech. 2023): the
  lift-augmented autoencoder baseline (Step 3 Ablation 11).
- V-JEPA 2-AC (\cite{vjepa2}, arXiv:2506.09985): the scheduled-
  sampling pattern that Ablation 7 disables.
- VICReg (\cite{vicreg}, ICLR 2022): the Bardes regulariser that
  Ablation 2 swaps in.
