# SESSION 31 — Canonical v2.2 retrain: fair AE vs JEPA, ROM-first

Target data: `split_v2p2` (102 cases, train_std = 3.5396, SSIM L = 8.487,
W&B group `partition_v2p2`). All runs:
`--partition v2p2 --pipeline-manifest outputs/data_pipeline/v2p2/manifest.json`.

Purpose: replace the v2.1 eleven-row soup with a controlled comparison that
varies one axis at a time, reads every model out through frozen probes, and
scores ROM behaviour on the impact and post-impact windows only. The headline
is no longer Repr R^2 on whole trajectories; it is forward closure of held-out
observables, field VRMSE against floor and persistence, and on-manifold drift.

Core question, scoped: do JEPA concepts give a better reduced-order model than a
pixel-loss autoencoder, for (Q1) reading off leading-edge, LEV and force state,
(Q2) predicting the gust impact and the relaxation after it, and (Q3) inferring
state from wall pressure.

What is new in the JEPA model versus v2.1 (provenance, so this is not relitigated):
spatial latent in place of the attention-pool vector (gray-scott), frozen
stop-gradient readouts with supervision moved to one explicit axis (SkyJEPA Idea 1
plus gray-scott), and the prediction loss consolidated to a single multi-step
rollout with the teacher-forced term dropped (SkyJEPA Eq. 9 plus gray-scott
`SquareLossSeq`). The evaluation is reframed to floor and persistence three-curve
VRMSE plus straightening and the compounding ratio (gray-scott, SkyJEPA Ideas 4
and 5). SIGReg and the CNN+ViT backbone were already deployed in v2.1 and are
retained, not new. The physics-structured observable prober (SkyJEPA Idea 3) is
deferred to keep the matrix clean; see out-of-scope.

---

## Standing conventions (do not relitigate here)

- Repo `vortex-jepa`, `.venv`, env vars `WANDB_PROJECT`, `PREVENT_ROOT`. HDF5 in PREVENT.
- Single source of truth: numeric results land in `numbers.json`, are exposed to
  LaTeX only through `macros.tex`. No numeric literals in `.tex`.
- All figures via `figstyle.py` at JFM column widths.
- W&B logging for every run. Artefacts to `outputs/runs/session31/`, write-ups to
  `outputs/session31/`.
- Decision log: append `D###` entries to `HANDOFF.md`, sequential after the last
  existing entry. The `D-N#` stubs below are placeholders; renumber on commit.
- Prose and captions: no em-dashes, British spelling, ML as instrument and fluid
  mechanics as the contribution.
- Seeds: one seed for the first instance (this is a fluid-mechanics paper, not an
  ML benchmark). Multi-seed variance and bootstrap-over-seeds are a second pass
  once the comparison is clear. Latent dim d = 32 (the channel count of the spatial
  latent). Note the gray-scott evidence that this channel count drives the decode
  floor: their D=16 gave floor VRMSE about 0.54 and D=64 gave well below 0.30, so
  d=32 sits between, and the floor is the thing to watch at Track B (see D-N3).

---

## Pre-flight checklist (Track 0 gates the rest)

- [ ] `outputs/data_pipeline/v2p2/manifest.json` exists and lists 102 cases.
- [ ] Normalisation constants confirmed: train_std = 3.5396, SSIM L = 8.487.
- [ ] The `(case_id, encounter)` alignment key resolves for every case in v2p2.
- [ ] Per-case impact time `t_impact` computable from `C_L(t)` (Track 0.B below).
- [ ] Wall-pressure traces present and aligned to the field frames for every case.
- [ ] Train / val / Test B split is case-disjoint (no case appears in two splits).
- [ ] `git status` clean; new branch `session31-canonical-v2p2`.

---

## The loss kit (single menu, every model is a switch-set over it)

Implement once in `vortex_jepa/losses/kit.py`. Every model config selects terms
from this kit; nothing is defined per-model. This is the fairness keystone: the
comparison must be auditable line by line.

```yaml
# config schema: configs/_kit.yaml  (defaults; per-model files override `on`)
representation_objective:        # MUTUALLY EXCLUSIVE, the one axis of interest
  recon:                         # autoencoder objective (matched AEs only)
    on: false
    loss: mse                    # plain MSE for the matched AEs (first instance)
    field: omega_pipeline_norm
    # references keep their native recon: fukami -> MSE recon + beta*lift
    #   (Fukami, Nakao & Taira 2024, eq 2.4; NOT Charbonnier, and NO wake);
    #   bvae -> MSE + KL. Charbonnier is gone from the plan entirely.
    # Charbonnier + SSIM (see out-of-scope) is a deferred recon-loss variant.
  pred:                          # JEPA objective
    on: false
    horizon: 8                   # H_roll, ONE value for ALL predictive models
    loss: mse_seq                # multi-step rollout MSE in latent space
    target: ema                  # ema | online  (see D-N4)

anti_collapse:                   # JEPA only; recon does not collapse (field pins z).
                                 # Compute over the BATCH (across cases and frames),
                                 # NOT over time within a clip: the gray-scott deck
                                 # (slide 13) shows a time-variance term collapses on
                                 # quasi-static episodes (Var_t -> 0 when the field is
                                 # quasi-steady). SIGReg matches the batch distribution
                                 # and is robust to this; it matters most in the
                                 # relaxation tail.
  on: false                      # ON only when pred.on
  method: sigreg                 # sigreg | vicreg  (vicreg is the ablation)
  lambda: 0.02                   # TODO PIN: 0.01 (v2.1) vs 0.1 (CLAUDE.md) vs 0.02 (SkyJEPA); ONE value for all JEPA runs
  sigreg: {projections: 256, knots: 17, knot_range: [0.2, 4.0]}
  vicreg: {std: 25.0, cov: 1.0}  # ablation only
  fallback: {to: vicreg, when_PR_below: 0.3, at_iter: 20000, and_probe_R2_below: 0.7}

supervision:                     # two distinct heads. The LIFT head is the lineage
                                 # standard (held on for the controlled matrix); the
                                 # WAKE head is THE tested axis, flipped on/off.
  lift_head:                     # L_obs: current-frame C_L augmentation (Fukami-standard)
    on: false
    target: [C_L]                # optionally C_D
    loss: smooth_l1
    weight: 1.0
  wake_head:                     # L_wake: patch_signed_spectrum, the load-bearing term
    on: false                    # the one supervision axis tested with/without
    target: patch_signed_spectrum
    loss: smooth_l1
    beta: 0.5
    weight: 1.0
# The five scalar observables (C_L, C_D, circ_pos, circ_neg, wake_enstrophy) are the
# READOUT/eval targets (frozen probes in Track D), NOT the supervision signal. The
# supervision is the lift head plus the wake-spectrum head only.

# READOUTS ARE NOT IN THIS BLOCK. They are always frozen, stop-gradient,
# trained post-hoc in Track D. See harness, not kit.
```

Rules enforced by the config loader (fail loudly if violated):

1. At most one of `recon.on` / `pred.on` is true. If neither, at least one
   supervision head must be on (the supervised-only control, model `supervised_only`,
   runs the lift and wake heads with no objective).
2. `anti_collapse.on` is mandatory when `pred.on`. It is also enabled on the
   matched AE (`recon.on`) so the anti-collapse term is held constant across the
   recon-versus-predict contrast, and it is off for `supervised_only`. (regAE in
   v2.1 already carried SIGReg, so this matches the real reconstruction baseline.)
3. `representation_objective` parameters (horizon, loss, field) are read
   from `_kit.yaml` and may not be overridden per-model. Only `on` flags and the
   `supervision` and `anti_collapse.method` switches vary.
4. Optimiser, schedule, batch, data split, seeds: inherited from `_kit.yaml`,
   never set per-model.

---

## TRACK 0 — Data and window integrity

Goal: certify split_v2p2, and produce the per-case impact windows that every
temporal metric depends on.

### 0.A Pipeline certification
```bash
python -m vortex_jepa.data.verify \
  --partition v2p2 --pipeline-manifest outputs/data_pipeline/v2p2/manifest.json \
  --check normalisation alignment split_disjoint pressure_alignment \
  --out outputs/session31/data_cert.json
```

### 0.B Impact-window definition (feeds Q2 and Q3)
Define, per case, from the held-out lift signal:
- `t_impact = argmax_t |dC_L/dt|` (the gust strike, peak lift transient).
- Lead-in window `[t_impact - W_in, t_impact]`, impact window `[t_impact, t_impact + W_imp]`,
  relaxation window `[t_impact + W_imp, t_impact + W_relax]`.
- Persist as a boolean mask per frame in `outputs/session31/windows_v2p2.json`,
  keyed by `(case_id, encounter)`.

Scoping note (defensible, not convenient): restricting the temporal evaluation to
the impact and relaxation windows and excluding steady state is physically
motivated. The transient is where a predictive latent has its advantage, and the
quasi-static regime is where it struggles (gray-scott slide 13), so this scope
targets the aerodynamically interesting regime rather than hiding a weakness. Be
ready to state this in the manuscript.

```bash
python -m vortex_jepa.eval.windows \
  --partition v2p2 --signal C_L --metric abs_dCl_dt \
  --w_in 8 --w_imp 16 --w_relax 48 \
  --out outputs/session31/windows_v2p2.json
```

**Acceptance gate 0.**
- STRONG: all pre-flight boxes tick, `t_impact` is unimodal and well separated
  from the trajectory ends for >= 95% of cases. Proceed.
- WEAK: `t_impact` ambiguous (multimodal `|dC_L/dt|`, or impact within `W_in` of a
  trajectory boundary) for >5% of cases. STOP. Resolve via **D-N1** (window
  definition) before any training; a wrong window invalidates every Q2/Q3 number.

---

## TRACK A — Loss-kit refactor

Goal: implement the kit above and migrate the trainer to consume it, so AE and
JEPA are literally the same code path with different switches.

- Implement `vortex_jepa/losses/kit.py` and the config loader with the four
  fail-loud rules.
- Delete per-model loss definitions from the v2.1 trainer. There must be no place
  outside the kit where a loss weight or a field-loss choice is set.
- Unit test: load each Tier-1 config, assert the active-term set matches the table
  in Track C, assert no per-model override of frozen parameters.

```bash
pytest tests/test_loss_kit.py -q
python -m vortex_jepa.config.audit --configs configs/canonical/ \
  --out outputs/session31/config_audit.md   # prints the on/off matrix for review
```

**Acceptance gate A.**
- STRONG: audit matrix matches Track C exactly; tests green. Proceed.
- WEAK: any model needs a parameter the kit cannot express cleanly. STOP, record
  in **D-N2**; do not add a per-model escape hatch, extend the kit explicitly.

---

## TRACK B — Shared backbone, spatial latent, frozen-probe harness

Goal: one encoder for the head-to-head, a spatial latent so the field stays
decodable, and a probe harness that is provably stop-gradient.

### B.1 Spatial-latent encoder
- Backbone: the CNN+ViT hybrid, but tapped so the latent is a spatial feature map
  `[B, d, h, w]`, not a pooled vector. This is the gray-scott lesson: a pooled
  latent cannot be decoded back to a field, which is the most likely cause of the
  v2.1 decode floor.
- Keep the pooled-vector head available behind a flag for the `jepa_pool` ablation
  (Track E), so spatial vs pooled is a clean one-axis comparison.

### B.2 Frozen-probe harness `vortex_jepa/probes/`
Three probes, each trained on a frozen encoder with an asserted stop-gradient on
the latent:
- `decoder`: latent -> field, MSE. Identical capacity for every model.
- `observable_head`: latent -> the five observables (frozen-readout version; the
  co-trained lift and wake heads live in the supervision block, not here).
- `pressure_probe`: wall-pressure sensors -> latent -> {field, observables}.

```bash
pytest tests/test_probe_stopgrad.py -q   # asserts no grad reaches the encoder
```

**Acceptance gate B.**
- STRONG: stop-gradient test green; a smoke decode on one case yields a field, not
  a degenerate constant. Proceed.
- WEAK: spatial-latent decode floor on `ae_nowake` smoke run is not clearly below the
  v2.1 pooled floor. Flag in **D-N3**; the spatial latent is the central
  architectural bet, so a non-improving floor needs diagnosis before scaling up.

---

## TRACK C — Train Tier 1 (the spine), 1 seed (first instance)

One backbone (Track B), the kit (Track A), split_v2p2. The on/off matrix:

| Model | `recon` | `pred` | `anti_collapse` | lift head | wake head | Notes |
|---|---|---|---|---|---|---|
| `jepa_nowake` | off | on | sigreg | on | off | predictive, lift only (approx old ctrl_pred_vit_nowake, the +0.11 wake cell) |
| `jepa_wake` | off | on | sigreg | on | on | predictive + wake (= old jepa_tf_noc, the +0.79 cell) |
| `ae_nowake` | on | off | sigreg | on | off | reconstruction + lift, matched arch (approx matched Fukami) |
| `ae_wake` | on | off | sigreg | on | on | reconstruction + lift + wake, matched arch |
| `supervised_only` | off | off | off | on | on | dissociation control: heads only, no recon, no pred |
| `fukami` | on (own arch) | off | off | on | off | faithful published lift-aug AE: MSE recon + beta*lift, NO wake |
| `fukami_wake` | on (own arch) | off | off | on | on | v2.1 augmented Fukami, WITH wake |
| `bvae` | on (own arch) | off | off | native | TBC | stochastic-latent ref (Carlos provides; beta + wake TBC) |
| `pod` | linear, no training | -- | -- | -- | -- | linear reference |
| `regAE` (optional) | on | off | sigreg | off | off | bare reconstruction, no heads (supervision floor) |

The wake head is the only column that flips within the predictive, reconstruction,
and Fukami families: that flip is the central with/without-wake result. The lift head
is held on as the lineage-standard augmentation, not an experimental variable.

```bash
SEED=$(python -m vortex_jepa.config.seeds --first)   # one seed for the first instance
for m in jepa_nowake jepa_wake ae_nowake ae_wake supervised_only; do
  python -m vortex_jepa.train \
    --config configs/canonical/${m}.yaml --seed ${SEED} \
    --partition v2p2 --pipeline-manifest outputs/data_pipeline/v2p2/manifest.json \
    --wandb-group partition_v2p2 --tag session31 \
    --out outputs/runs/session31/${m}
done
# references on their own recipes, one seed each
for m in fukami fukami_wake; do
  python -m vortex_jepa.train --config configs/reference/${m}.yaml --seed ${SEED} \
    --partition v2p2 --pipeline-manifest outputs/data_pipeline/v2p2/manifest.json \
    --wandb-group partition_v2p2 --tag session31 --out outputs/runs/session31/${m}
done
# bvae (optimal): Carlos supplies configs/reference/bvae.yaml (beta at L-curve elbow; wake TBC)
python -m vortex_jepa.train --config configs/reference/bvae.yaml --seed ${SEED} --partition v2p2 ... \
  2>/dev/null || echo "bvae config pending"
python -m vortex_jepa.baselines.pod --partition v2p2 --d 32 --out outputs/runs/session31/pod
```

**Acceptance gate C.**
- STRONG: all spine runs converge and anti-collapse never triggers the VICReg
  fallback on `jepa_nowake`/`jepa_wake`. With one seed there is no SD check; sanity is
  convergence plus a non-degenerate decode. Proceed to Track D. (Seed SD against
  the v2.1 band of ~0.05 to 0.07 returns in the multi-seed pass.)
- WEAK: `jepa_nowake` collapses (PR(z) low, fallback fires). This is informative, not
  fatal: it means the multi-step rollout plus SIGReg did not prevent collapse on
  this data. Record in **D-N4**, branch the EMA-vs-online target decision, do not
  paper over it by hand-tuning lambda.

---

## TRACK D — ROM evaluation (R^2 demoted), windows only

Every metric below is computed on Test B, restricted to the Track 0.B windows,
and reads each model through the Track B frozen probes. For `ae_nowake`/`ae_wake`
also report the native co-trained decoder as the AE best case, alongside the
frozen probe-decoder as the apples-to-apples number.

The forecast (Q2) headline uses a separately trained matched predictor fitted fresh
on each frozen latent, for every model, so the comparison isolates encoder geometry
rather than how each model was trained. Training the readout with the encoder frozen
and gradients detached is the eb_jepa-native pattern (upstream, both the location
decoder and the pixel decoder are trained with detached gradients), so this is not a
deviation, it is how the library itself evaluates. This makes the v2.1 jepa_vs_vjepa
protocol uniform: there, JEPA and ST used their own co-trained predictor while
V-JEPA and the references used the matched predictor, which was not apples-to-apples.
For models that co-train a predictor (`jepa_wake`, `st_d64`) also report the native
predictor as the as-built ROM, secondary.

Fit TWO matched-predictor architectures on each frozen latent and report both: a
transformer (the v2.1 winner, which operated on a pooled-vector latent) and a
convolutional ResUNet (the gray-scott and upstream eb_jepa predictor: a U-Net that
rolls the latent field forward from the concatenated current and previous latents,
context_length=2, with skip connections preserving spatial detail; they rejected an
RNN because it collapses the spatial layout the decoder needs). With the spatial
latent now canonical, the ResUNet is the natural pairing, so this comparison decides
the predictor for any future canonical model rather than assuming the v2.1
transformer carries over from the pooled regime (see D-N7). Do not trust any single
R^2: the v2.1 regAE h=16 reading of +0.80 was a single-seed, small-N artefact (above
its own h=8), which is exactly why Q2 leads with the floor and persistence
three-curve and the on-manifold drift, not a lone number.

### D.1 Q1 — representation (what the latent contains)
- Linear probe AND small nonlinear (MLP) probe for: leading-edge indicator, LEV
  circulation, C_L, C_D. The nonlinear probe is mandatory; it closes the
  single-probe-class vulnerability.
- decode-then-encode field fidelity (the floor).
```bash
python -m vortex_jepa.eval.represent \
  --runs outputs/runs/session31 --probes linear mlp \
  --targets leading_edge lev_circ C_L C_D --window all \
  --out outputs/session31/q1_representation.json
```

### D.2 Q2 — temporal prediction on the impact and relaxation windows
- Roll the latent (`z_{t+1} = A z_t` for the linear operator, or the predictor),
  decode the field, report VRMSE per horizon as three curves: **floor < model <
  persistence**, masked to the impact and relaxation windows.
- ROM figure of merit: forward closure of the five observables (roll, decode
  observables, score on the windows).
- On-manifold, three ways: latent drift rel-L2, straightening S_straight,
  compounding ratio CR_k.

```bash
python -m vortex_jepa.eval.rollout \
  --runs outputs/runs/session31 --windows outputs/session31/windows_v2p2.json \
  --curves floor model persistence --vrmse aggregated \
  --observables C_L C_D circ_pos circ_neg wake_enstrophy \
  --diagnostics drift straightening compounding_ratio \
  --out outputs/session31/q2_temporal.json
```
VRMSE definition (use exactly this, aggregated num/den over the eval set before
dividing, per the gray-scott note; per-sample ratios blow up on near-uniform
frames):
`VRMSE = sqrt( sum_space (pred - true)^2 / sum_space (true - mean_true)^2 )`.

Treat field VRMSE as the conservative pixel metric, not the verdict. The gray-scott
deck is direct evidence: their temporal JEPA loses to the pixel-space U-Net and FNO
on VRMSE, badly on the quasi-static phases (slide 7), while producing visibly more
realistic structure (slide 8), which is why they needed a perceptual metric to show
the advantage and even then it won only on the dynamic phases (slide 9). Expect the
same here, the matched AE may win field VRMSE while the predictive latent is more
physically faithful. So report a structural companion to VRMSE (the SSIM v2.1
already tracks, on the decoded field) and keep the ROM figure of merit on the
observable closure (lift, drag, circulation, the wake spectrum), which is the
physics-fidelity analog of their perceptual metric. The win/loss boundary across
{pixel VRMSE, structural SSIM, observable closure} is the result, not a lone field
number.

### D.3 Q3 — state inference from wall pressure
- Frozen `pressure_probe`: wall-pressure sensors -> latent -> {field, observables},
  scored on the windows. Compare the AE-latent and the JEPA-latent as estimation
  targets.
```bash
python -m vortex_jepa.eval.pressure \
  --runs outputs/runs/session31 --windows outputs/session31/windows_v2p2.json \
  --sensors wall_pressure --decode field observables \
  --out outputs/session31/q3_pressure.json
```

**Acceptance gate D (the scientific gate).**
- STRONG claim available: across Q1 to Q3, the predictive latent beats the matched
  AE on the abstract and dynamics-organised targets (wake-internal observables,
  pressure inference) and is on-manifold where the AE drifts, even if the AE wins
  on the raw field. This is the defensible JFM story. Write it as the win/loss
  boundary, not a single winner.
- WEAK claim: `jepa_nowake` loses to `ae_nowake` on every target including the
  wake-internal observables and pressure inference. Then the contribution
  collapses to the supervised case (`jepa_wake`) and must be framed as
  "wake supervision on a predictive latent", with the objective itself reported as
  not sufficient. Record honestly in **D-N5**; do not relabel a supervision effect
  as an objective effect.

---

## TRACK E — Tier 2 ablations (one axis each)

Run only after Track D fixes the spine, so each ablation reads against a settled
reference.

| Model | Axis moved | Question answered |
|---|---|---|
| `jepa_cnn`, `ae_cnn` | encoder rung 1: CNN-only vs CNN+ViT | does the ViT contribute anything (v2.1 said no) |
| `st_d64` | encoder rung 3: CNN+ViT + temporal conv (kernel=3 over time) | does temporal convolution beat pure spatial (v2.1: Repr +0.77, Fcst +0.37, close to JEPA; MUST use the matched predictor to be apples-to-apples) |
| `jepa_pool` | pooled-vector vs spatial latent | is the spatial latent the lever |
| `jepa_vicreg` | VICReg vs SIGReg | does the one-knob anti-collapse hold up |

```bash
for m in jepa_cnn ae_cnn st_d64 jepa_pool jepa_vicreg; do
  python -m vortex_jepa.train --config configs/ablation/${m}.yaml --seed ${SEED} --partition v2p2 ... # 1 seed
done
python -m vortex_jepa.eval.rollout --runs outputs/runs/session31 --windows ... # extend the same tables
```

**Acceptance gate E.**
- STRONG: each ablation moves the expected single metric and nothing else,
  confirming the axis is isolated. Proceed to Track F.
- WEAK: an ablation moves several metrics at once. The axis is not isolated; a
  hidden coupling remains in the kit or backbone. Record in **D-N6**, fix before
  reporting the ablation.

---

## TRACK F — Assemble numbers and tables

- Merge all `outputs/session31/q*.json` into `numbers.json` under a `session31`
  namespace, keyed by `(model, seed, window, target)`.
- Regenerate `macros.tex` from `numbers.json`; no hand-edited numbers.
- Build the canonical comparison table and the three-curve VRMSE figures via
  `figstyle.py` at JFM widths.
- Uncertainty: case-clustered bootstrap CIs over Test B cases per reported metric.
  The full three-signal uncertainty quantification returns with the multi-seed pass.

```bash
python -m vortex_jepa.report.merge_numbers --glob "outputs/session31/q*.json" --into numbers.json --namespace session31
python -m vortex_jepa.report.build_macros --numbers numbers.json --out macros.tex
python -m vortex_jepa.report.tables --namespace session31 --style figstyle --out outputs/session31/tables/
latexmk -pdf tables/canonical_v2p2.tex
```

**Acceptance gate F.**
- STRONG: tables build from macros only, CIs present on every reported number,
  no numeric literal anywhere in the `.tex`. Session closeable.
- WEAK: any reported number lacks a case-clustered bootstrap CI. Not closeable.
  (Seed-based uncertainty is deferred to the multi-seed pass; the case bootstrap
  over Test B still applies on one seed.)

---

## Dependency graph

```
Track 0 (data + windows)
   |
   v
Track A (loss kit) ----+
   |                   |
   v                   |
Track B (backbone, probes)
   |
   v
Track C (train spine, 1 seed)
   |
   v
Track D (ROM eval on windows)  <-- the scientific gate
   |
   v
Track E (one-axis ablations)
   |
   v
Track F (numbers.json -> macros.tex -> tables)
```
Track 0 blocks everything (a wrong window invalidates Q2/Q3). Track A and B can
proceed in parallel after 0, but C needs both. D needs C. E needs D. F needs E.

---

## HANDOFF decision stubs (renumber against HANDOFF.md on commit)

- **D-N1 — Impact-window definition.** If `t_impact` is ambiguous, decide the
  fallback rule (e.g. switch the trigger from peak `|dC_L/dt|` to the gust-centroid
  arrival at the leading edge, or widen `W_in`). Decision blocks all temporal work.
- **D-N2 — Kit extension vs escape hatch.** If a canonical model needs a term the
  kit lacks, decide whether to extend the kit (allowed) or drop the model (allowed);
  a per-model override (forbidden) is not on the menu.
- **D-N3 — Spatial latent verdict, and the d=32 floor.** If the spatial-latent floor
  does not beat the v2.1 pooled floor, decide encoder capacity bump vs decoder
  capacity bump (the gray-scott fix was decoder/encoder capacity, not predictor).
  d=32 is the lean first-instance choice but sits between gray-scott's D=16 (floor
  about 0.54) and D=64 (floor below 0.30), so if the floor is too high the first
  lever is raising d, not touching the predictor.
- **D-N4 — EMA vs online target, and collapse handling.** The gray-scott deck
  (architecture slide) confirms their deployed model uses the ONLINE encoder as the
  target: one shared f_theta encodes z_t, z_{t+1} and the target z_{t+2}, no separate
  EMA network, with the variance-covariance term preventing collapse. So the
  gray-scott route is online-target. Decide between that and an EMA target (the
  I-JEPA standard); if `jepa_nowake` collapses, prefer stronger batch-distributional
  SIGReg over an EMA bolt-on, and do not hand-tune lambda per run.
- **D-N5 — Objective vs supervision attribution.** Lock the claim language to what
  the matrix (`{ae,jepa} x {wake off, wake on}` plus `supervised_only`) actually
  supports. This is the
  cell the v2.1 drift table was missing and the reason the on-manifold claim was
  undefended.
- **D-N6 — Ablation isolation.** If a Tier-2 ablation moves more than its one axis,
  decide where the coupling lives before the ablation is reported.
- **D-N7 — Predictor architecture for the spatial latent.** The v2.1 winner used a
  transformer predictor on a pooled-vector latent; the spatial latent is the
  eb_jepa and gray-scott regime, where the native predictor is a convolutional
  ResUNet (U-Net, skip connections, rejected RNN). After the dual matched-predictor
  eval in Track D, decide which predictor a future canonical model co-trains. Do not
  assume the transformer carries over from the pooled regime to the spatial one.

---

## Out of scope for Session 31 (future work, do not start here)

- Physics-structured observable prober (impulse and circulation prior plus learned
  residual). High value, but it changes the readout for everyone and would
  contaminate the clean controlled matrix. Schedule as Session 32 once the spine is
  settled. This is where SkyJEPA's predictive latent should decisively beat
  reconstruction on the abstract observables.
- Physics-graph encoder (mesh-adjacency or POD-co-activation mixing), hierarchical
  region sub-models coupled along the advection direction. Speculative encoder
  variants from the fly-brain post; hypotheses to test later, not in this batch.
- Control loop (latent MPPI on the observable prober) for transient lift
  attenuation. Depends on the prober above and a stability-constrained operator.
- Conditioned encoder on (G, D, Y). Settled redundant in v2.1 (key read #5);
  the architecture stays unconditional.
- LSTM predictor. Settled weaker in v2.1 (key read #6).
- V-JEPA (3D tubelet masked-infilling encoder with an EMA target). Tested in v2.1
  across five variants (plain, fine, dense, heads, best); none matched the temporal
  JEPA at h=16 (best V-JEPA +0.27 vs JEPA +0.43) and the branch concluded
  not-for-forecasting. Excluded from the canonical comparison; report as a stated
  negative result. Terminology: the canonical predictive model is the TEMPORAL JEPA
  (predict the future latent from the past latent), not masked V-JEPA.
- Neural-operator field forecaster (FNO or U-Net predicting the next field from the
  past two frames, pixel loss). Not in the first batch, but recommended before
  submission: the gray-scott deck shows this is the pixel-space SOTA that actually
  beats the predictive latent on field VRMSE, so a reference one preempts the obvious
  referee question and makes the field-metric comparison honest. It is a forecaster,
  distinct from the reconstruction AEs (`ae_nowake`, `fukami`); add it if the field
  VRMSE comparison becomes load-bearing.
- Charbonnier + SSIM combined reconstruction loss, the convex blend
  alpha * L_charbonnier + (1 - alpha) * L_ssim from the uploaded HSI-denoising
  figure. A deferred recon-loss variant to test once the matched-MSE baselines are
  settled; not in the first batch. SSIM stays an evaluation metric in the meantime.
- beta-VAE is no longer out of scope. It is now a reference baseline in Track C,
  supplied by Carlos (beta at the L-curve elbow; wake head TBC).
- Anything touching the companion paper (Solera-Rico et al., compactness vs
  forecast accuracy in controlled wake flows).
