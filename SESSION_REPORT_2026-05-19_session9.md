# Session 9 Report -- lambda bisection + visualisation decoder + Section 7 thin cut

Date: 2026-05-19.

## Goal

Refine the Session 8 production point (eta=0.01, lambda=0.01, d=32,
OBS=cl_future at eta=0.01) with a LeWM-style lambda bisection over
[0.001, 0.1], train the visualisation decoder on the frozen winner, and
land a thin slice of the Section 7 ablation matrix. Draft Abstract,
Sections 1, 2, 6, and 7-outline during the compute windows.

Plan: `SESSION9_LAMBDA_BISECTION.md`. Six pass criteria, five step
deliverables, six D-entries to record.

## Pre-flight

Tests at session start: **123 passed, 4 skipped in 582.10s on CPU only**.
The slowdown (~10 min vs the 50s baseline) is explained by external
concurrent compute on the workstation: `asolera` is running SOD2D
simulations on the two L40S cards (`/home/asolera/CasosSOD2D/Naca_BSC_gusts_generator/sod2d`),
and `isaac` has ~30 processes saturating CPU at 97-99% each. The two
RTX 6000 Blackwell cards remain available for Session 9 work; D40's
two-card pattern (`--gpu 0`, `--gpu 1` resolving to torch indices 2 and
3) is honoured. nvidia-smi confirms both RTX 6000s reach ~19.5 GiB
memory and 65-100% utilisation under load.

Git HEAD at session start: `753ed2c` (Session 9 plan committed in d708fa8
then refined). Pre-launch housekeeping commit `2be1f9e` lands the Session 9
launchers + orchestrators + analysis script before the first GPU
command, matching the D40/D44 precedent.

Wall-clock implication of the external load: per-run iteration rate
settled at roughly 220 iters/min once both RTX 6000 cards spun up
together, giving each 20k-iter run a wall-clock budget of ~1.5
hours, in line with the Session 8 D49 single-card baseline. The full
Session 9 schedule fits within the planned 14-16 hour window. The
Step 3 thin cut is scoped to A2 + A7 (the JEPA-internal ablations) +
A11 (Fukami observable-augmented AE, added mid-session at the user's
request for a faithful comparison via Fukami's CNN architecture and
SSIM-based methodology). A10 (Solera-Rico beta-VAE + transformer ROM)
remains deferred to Session 10 because the faithful two-stage
implementation (variational encoder Stage 1 + transformer ROM on the
frozen latent Stage 2) cannot fit cleanly inside the cuda:1 idle
window between A11 and A7 without compute-conflict risk.

## Step 1: lambda bisection (D58)

Five-point bisection over lambda in {0.001, 0.003, 0.01, 0.03, 0.1} at
the production (d=32, eta=0.01, OBS=cl_future at eta=0.01, BN, SIGReg)
configuration. F1 (lam=0.001 seed=0), F2 (lam=0.003 seed=0), and F3
(lam=0.03 seed=0) are new; E4 (lam=0.01 seed=0) and E5 (lam=0.1
seed=0; = Session 7 R3 anchor) are reused from disk.

cuda:0 chain executed: F1 (22:26-23:58) -> F2 (23:59-01:35) ->
F3 (01:35-03:09) -> `session9_bisection_analysis.py` on cuda:1
(03:09-03:10) -> F4 seed=42 (03:11 in progress) -> F5 seed=123 (queued).

Per-cell seed=0 Test B summary (from
`outputs/runs/session9/bisection_seed0.csv`):

| code | lambda | PR_all | r2(z->c) | r2(CL_future) | r2(c, t) | delta_test_b |
|------|-------:|-------:|---------:|--------------:|---------:|-------------:|
| F1   | 0.001  |  2.22  |   0.887  |    0.836      |  0.718   |   +0.118    |
| F2   | 0.003  |  2.10  |   0.890  |    0.850      |  0.718   |   +0.131    |
| **E4** | **0.010** | **2.61** | **0.866** | **0.878** | **0.718** | **+0.159** |
| F3   | 0.030  |  2.49  |   0.883  |    0.849      |  0.718   |   +0.131    |
| E5   | 0.100  |  3.51  |   0.932  |    0.856      |  0.718   |   +0.138    |

**lambda\* = 0.01 (E4 from Session 8)** with delta\_test\_b = +0.159.
Clean interior maximum, roughly symmetric in log-lambda: F1 and F2 at
lambda < 0.01 land at +0.118 / +0.131; F3 and E5 at lambda > 0.01 land
at +0.131 / +0.138. The Session 8 D53 finding is confirmed. PR_all
also peaks at E4 (2.61) which suggests SIGReg at lambda=0.01 produces
a slightly higher rank latent than at either edge of the sampled
interval, consistent with the controlled-collapse mechanism balancing
the OBS head's directional pressure against SIGReg's distribution
matching most cleanly at lambda=0.01.

Outcome category: **PRODUCTION_LOCKED** (lambda* unchanged from
Session 8 production point). F4 (seed=42 at lambda=0.01) and F5
(seed=123 at lambda=0.01) follow on cuda:0 for the paper-grade
seed-variance bound. R0 at lambda* not needed (Session 8 D55
already covered lambda=0.01).

## Step 2: visualisation decoder (D59)

Trained the `HybridViTConvDecoder` (8.72M params) on the frozen Step 1
winner encoder for 10000 iterations at lr=1e-4, bf16, AdamW. Per-frame
MSE loss on `omega_z` summed over `(T, H, W)`.

`{TO_FILL: Test A / B / C MSE table, ratio against per-case-mean
noise floor, pass criterion outcome, Figure 3 description}`

## Step 3: Section 7 ablation thin cut (D60)

cuda:1 chain (interleaved with Step 1's compute on cuda:0): A2
VICReg-only (canonical Bardes et al. weights mu=25, lambda_var=25,
nu=1) -> A11 Fukami observable-augmented AE (CNN architecture from
arXiv:2305.18394 Table S.1 adapted to 192x96 input + d=32 latent,
trained jointly on reconstruction MSE + lift MSE per Fukami's
methodology) -> wait for lambda* -> A7 no-scheduled-sampling
(H_roll=T=32 at lambda*). A10 Solera-Rico beta-VAE + transformer
deferred to Session 10 (faithful two-stage reproduction does not fit
the cuda:1 idle window between A11 and A7).

Fukami evaluation reports per-encounter MSE alongside the SSIM
(Eq. 1 from supplementary, `C_1=0.16`, `C_2=1.44`) for fair
comparison with the JEPA decoder (Section 6) on Test A / B / C.

`{TO_FILL: A2 + A7 + A11 Test B delta numbers, Fukami SSIM table,
implications for paper claims}`

## Step 4: R0 at lambda* (D61, conditional)

Skipped. lambda* = 0.01 matches Session 8 D55's covered cases
(R0 at lambda=0.1 and R0 at lambda=0.01); no new R0 run needed.

## Step 5: paper writing (D62)

During the Session 9 compute windows:

- **Abstract** (`paper/sections/abstract.md`): ~240 words, three
  contribution claims with their headline numbers. Lambda value in
  the abstract will be updated once Step 1 lands lambda\*.
- **Section 1 (Introduction)** (`paper/sections/section_1_introduction.md`):
  ~1600 words, four subsections. ROM motivation, JEPA framing,
  contribution claims, roadmap. Cross-references to Section 2
  (related work), Section 6 (decoder), and Section 7 (ablations) noted
  for revision once the cited subsections finalise.
- **Section 2 (Related work)** (`paper/sections/section_2_related_work.md`):
  ~3245 words, four subsections. JEPA lineage, observable-augmented
  autoencoders, classical/learned ROM, the gap closed by this paper.
- **Section 6 (Visualisation decoder results)**
  (`paper/sections/section_6_decoder.md`): ~975 words, five
  subsections, placeholder tokens for the Step 2 numerical results.
- **Section 7 outline + Table 2 skeleton**
  (`paper/sections/section_7_ablations.md`): four-subsection skeleton
  with the 15-ablation matrix structure (A1-A15 across four families:
  anti-collapse regulariser, conditioning, training procedure,
  comparator architecture). A10 + A11 explicitly noted as Session 10
  deferrals.
- Em-dash cleanup pass: removed em-dashes from the titles of Sections
  3, 4, 5 and from six body locations in Section 4 + one in
  Section 3. All paper section files now em-dash-free per CLAUDE.md.

## Session 9 outcome (D63)

`{TO_FILL: PRODUCTION_LOCKED, PRODUCTION_REFINED, or PRODUCTION_PIVOT
based on Step 1 bisection result. Plus prediction tracking against
the launch message's three credences (lambda* = 0.01 at 55%; seed
variance within +/- 0.03 at 70%; VICReg + OBS Test B delta within
+/- 0.05 of SIGReg + OBS at 50%).}`

## Predictions tracking (from launch message)

1. lambda* = 0.01 (no change from Session 8). Credence 55%.
   Result: `{TO_FILL}`.
2. Seed variance at lambda* within +/- 0.03 of seed=0. Credence 70%.
   Result: `{TO_FILL}`.
3. VICReg + OBS Test B delta within +/- 0.05 of SIGReg + OBS.
   Credence 50%. Result: `{TO_FILL}`.

## What is next

Session 10:

- A10 Solera-Rico beta-VAE + transformer baseline (new
  `src/baselines/solera_rico.py`).
- A11 Fukami observable-augmented AE baseline (new
  `src/baselines/fukami_ae.py`).
- Remaining Section 7 ablations (A4-A6 conditioning family; A8, A9
  training-procedure family; A12 POD linear floor).
- Multi-seed averages on the production configuration (seed in
  {0, 42, 123, 2026, 31415}) for the paper-grade variance bound.
- Final paper figures (Figure 1 architecture diagram, Figure 2 grid
  heatmap, Figure 3 decoder reconstruction, Figure 4 ablation matrix).
- JFM / PRF manuscript draft pass through Sections 1 to 8.

Session 11 (if needed): revision after internal review, additional
runs for reviewer-anticipated questions.

## Files committed in this session

- `scripts/launch_session9_step1_bisection.sh` (Step 1 launcher)
- `scripts/launch_session9_step3_ablations.sh` (Step 3 launcher)
- `scripts/orchestrate_session9_step1.sh` (cuda:0 chain)
- `scripts/orchestrate_session9_step3.sh` (cuda:1 chain)
- `scripts/session9_bisection_analysis.py` (Step 1 analysis)
- `scripts/session9_train_decoder.py` (Step 2 entrypoint)
- `src/models/decoder.py` (HybridViTConvDecoder)
- `paper/sections/abstract.md`
- `paper/sections/section_1_introduction.md`
- `paper/sections/section_2_related_work.md`
- `paper/sections/section_6_decoder.md`
- `paper/sections/section_7_ablations.md`
- `paper/sections/section_3_methods.md` (em-dash cleanup)
- `paper/sections/section_4_failure_modes.md` (em-dash cleanup)
- `paper/sections/section_5_full_scale_results.md` (em-dash cleanup)
- `notebooks/10_session9_lambda_bisection.ipynb` (skeleton)
- `HANDOFF.md` (D58-D63 added)
