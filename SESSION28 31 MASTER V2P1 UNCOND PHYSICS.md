# SESSIONS 28-31: v2.1 unconditioned rebuild, referee remediation, and the physics elevation

One master plan, four Claude Code sessions (Phase A = Session 28 through Phase D = Session 31).
Everything retrains anyway (v2.1 data + unconditioned mandate), so this plan merges three workstreams
into one rebuild instead of patching the v2 manuscript:

1. The v2.1 unconditioned rerun (RERUN_MANIFEST.md, “Split v2.1 update” + “Unconditioned rerun”).
1. The full referee remediation (SESSION27_JFM_REFEREE_AND_REMEDIATION.md; issue IDs B/M/S/W cited
   below refer to that report).
1. A physics elevation: four new flow-physics results (similarity collapse, recovery clock, phase
   dynamics, LEV budget) that move the paper’s center of mass from ML metrics to fluid mechanics,
   per the JFM lineage standard.

Authoritative context documents, read at every phase pre-flight: `CLAUDE.md`, `HANDOFF.md` (live
head; this plan uses provisional stubs D178+, renumber at execution), `RERUN_MANIFEST.md`,
`SESSION18_B1_PROTOCOL.md`, and this file.

Operating rules (inherited, non-negotiable):

- Honesty over preservation. Every gate below has a written weak branch; if a number weakens, the
  text follows the number.
- No em-dashes anywhere (code comments, figures, prose, this plan’s outputs).
- Test C (G = +4) is never used for selection; reported at the end only.
- All training on the two RTX 6000 Blackwell cards via `require_rtx6000(gpu_index=...)`. The L40S
  cards are asolera’s SOD2D machines and must stay free. No CPU fallback for model work.
- Loss in 3-sigma normalised space; un-normalise only at metric/figure time.
- DNS Table 1 resolution rows are collaborator-owned. NEVER fabricate or fill them. The internal
  undisturbed-flow validation (Phase A, Track A2) is the only Table-1-adjacent thing we compute.
- W&B: group `partition_v2p1`, the four required keys + paper keys per CLAUDE.md, tag every run
  with `s28` plus the family/cell tag defined in the training matrix.
- Every printed manuscript number traces to `outputs/session28/numbers.json` via `paper/macros.tex`
  (Track 0). No hand-typed numbers in the .tex after Phase D.
- Probe regimes per CLAUDE.md: IMPACT-frame z for parameter probes, PER-frame z for state probes;
  declare the regime in every new probe script’s docstring.
- Figures import `scripts/session21/figstyle.py`; every regenerated paper artifact saved under its
  existing basename with the `_v2p1` suffix (RERUN_MANIFEST mandate); v2 figures stay on disk as
  the frozen reference.

-----

## 0. Open author decisions (answer before or at Phase A pre-flight)

AD1. Unused run3 cases (Gust_048-066 subset, 16 cases). Default in this plan: ABSORB INTO TRAIN if
and only if the Phase A inventory shows they lie inside the existing (G, D, Y) envelope (they then
add statistical power without moving the test boundary); if any lie OUTSIDE the envelope on an axis
other than G (e.g. a new D or |Y| value), hold them out as a new `test_d` extrapolation tier
instead and do NOT train on them. test_b/test_c stay frozen byte-identical either way. This
decision must be made BEFORE any Phase A training launch; it changes the split file once
(`build_split_manifest_v2p2.py` if adopted) and never again mid-campaign.

AD2. Conditioned reference rows. The unconditioned mandate says no mixed conditioned controls. The
abstract claim “withholding the gust parameters costs almost nothing” still needs one conditioned
number. Default: train ONE conditioned tf reference (d = 64, 3 seeds), confine it to a single
clearly-labelled table row + one sentence, nowhere else. Alternative: drop the comparison sentence
and the runs.

AD3. Beta-VAE fourth family (the authors’ own Solera-Rico 2024 pipeline). Default: IN (referee M2
named its absence as the suspicious omission; CLAUDE.md lists it as planned baseline 3). Requires
porting the 2024 recipe behind `train_baseline` in Phase A.

AD4. lstm-no-c seed count. Default: 3 seeds (tf-no-c, the lead family, gets 4).

AD5. Title choice (options in Phase D, Track I). Default recommendation: option (a).

AD6. Zenodo DOI, license confirmation (MIT proposed, D174), CRediT, funding text, and the DNS
resolution package remain author/collaborator-owned. Phase A sends the collaborator package; Phase
D integrates whatever has arrived and leaves visible \pending{} otherwise.

AD7. PLDM stays retired (D29/D31 regime-dependent claim already in the paper). Not revived.

-----

## 1. Phase map and dependency graph

|Phase|Session|Theme                                                                   |GPU training               |Gates               |
|-----|-------|------------------------------------------------------------------------|---------------------------|--------------------|
|A    |28     |Foundations, decisions, FULL training launch, DNS/admin, protocol freeze|launches everything        |GA0-GA3             |
|B    |29     |Closure, attribution, baseline credibility, statistics core             |B1 predictors only         |GB, GD, GC          |
|C    |30     |Mechanism coherence + the four physics tracks + observability rebuild   |decode/saliency passes only|GE1-GE3, GF, GP1-GP4|
|D    |31     |Manuscript: macros, restructure, figures, tables, surgery, go/no-go     |none                       |GI, GO              |

```
Phase A: decisions (AD1) -> split freeze -> training matrix launch (T1-T9, both cards)
          |                                   |
          +-> A2 undisturbed validation       +-> (GPUs busy ~2.5-3 days)
          +-> A3 DNS collaborator package
          +-> A4 protocol freeze (rollout, R2, probes, selection)
          +-> A5 provenance harness (eval_all.py skeleton, macros emitter)
          +-> A6 literature checks (arXiv MCP)
Phase B: latents -> B1 predictors/rollouts -> closure matrix (4 probe classes)
          -> 2x2 both endpoints -> floor -> statistics (cluster/Holm/bootstrap)
          -> AE dossier + budget verdict -> beta-VAE verdict
Phase C: drift | topology | transport | scale | P1 SIM | P2 REC | P3 PHASE | P4 LEV
          | P5 CODE | observability rebuild | interventional regen   (parallel, latents-only)
Phase D: numbers.json complete -> macros -> restructure -> figures(_v2p1) -> tables
          -> abstract/title -> surgery -> build gates -> submission go/no-go
```

-----

## PHASE A (Session 28): Foundations and full training launch

### A0. Pre-flight (blockers; do not launch anything before all pass)

```bash
cd /home/carlos/GUST-JEPA
source .venv/bin/activate
export PREVENT_ROOT=$HOME/PREVENT WANDB_PROJECT=vortex-jepa
git status --short            # must be clean; stash or commit anything pending
HEAD_D=$(grep -oE '^### D[0-9]+' HANDOFF.md | grep -oE '[0-9]+' | sort -n | tail -1)
echo "Live HANDOFF head: D$HEAD_D"   # renumber every D-stub in this plan from HEAD_D+1
python -m src.training.sanity_checks --all --require-gpu
nvidia-smi                    # both RTX 6000 idle; L40S may be busy (asolera), irrelevant
```

PF-A1. Split and cache integrity: `configs/splits/split_v2p1.json` reports 85 cases / 382
encounters (71 train / 10 test_b / 4 test_c; val 87, test_b 42, test_c 24);
`python scripts/data_integrity_audit.py --split configs/splits/split_v2p1.json --cl-hard-cap 12 --pwall-hard-cap 15` returns 0 flags of 382; `outputs/data_pipeline/v2p1/manifest.json` exists with
train_std 3.6337; `configs/ssim_data_range.json` carries split_v2p1 = 8.45.

PF-A2. Recipe diff for reuse decisions. Diff the Session 27 prototype run configs
(`outputs/session27/JEPA_d64_noc_{tf,lstm}/` W&B configs) against the locked Direction-E recipe +
`--predictor-cond-dim 0` on split_v2p1. If a prototype matches recipe AND split AND pipeline
manifest exactly, it IS the production seed-42 run for that predictor type: reuse, train only the
remaining seeds. Same check for `outputs/session27/decoder_noc_{tf,lstm}/`. Record the verdict in
the D178 stub. Likewise diff the session20 Track-A 2x2 recipe against Direction E; if identical up
to the seed and conditioning flag, the predictive CNN+ViT control cell reuses the production seeds
(saves 3 runs), else it trains its own.

PF-A3. AD1 inventory: list the 16 unused run3 cases with parsed (G, D, Y); classify
inside/outside the train envelope per axis; apply the AD1 rule; if absorbed, generate
`split_v2p2.json` via a new `build_split_manifest_v2p2.py` (test_b/test_c byte-identical, assert
it), rebuild the omega pipeline manifest for v2p2, pin its SSIM L into the registry, and use v2p2
everywhere below (the plan writes v2p1; substitute mechanically). If held out as test_d, no split
change to train; add a `test_d` list and report it Phase D as a second extrapolation tier.

PF-A4. W&B timing calibration: pull median wall time of the last d=64 20k-iter encoder runs and the
last AE/decoder runs; update the schedule table in A8 with measured numbers.

PF-A5. Stale-doc patch: update CLAUDE.md “Current focus” (the deferred-retrain note is now
obsolete; the rerun is live, paper target = v2.1 unconditioned) and add a pointer to this plan.

Gate GA0: all five pre-flight items pass. D178.

### A1. Training matrix (launch everything; both cards; W&B tag s28 + the tag below)

All `train_jepa` runs: locked Direction-E recipe (D99) with `--predictor-cond-dim 0`,
`--omega-pipeline-manifest outputs/data_pipeline/v2p1/manifest.json`, split v2p1, T = 32,
H_roll = 8, d as stated, 20k iters, observable heads cl_future (eta 0.01) + patch_signed_spectrum
wake head (lambda 1.00) unless the cell says otherwise. The wake-observable cache train stats must
be recomputed once for v2p1 before the first launch (same procedure as the Session 12 recompute).

|# |Cell                                                                                                                                                                                         |Runs                  |Tag                                |Notes                                                                                                  |
|--|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------|-----------------------------------|-------------------------------------------------------------------------------------------------------|
|T1|JEPA tf-no-c d=64, seeds 42/0/1/2                                                                                                                                                            |4 (minus PF-A2 reuse) |jepa_tf_noc_d64_s{n}               |production family, lead                                                                                |
|T2|JEPA lstm-no-c d=64, seeds 42/0/1                                                                                                                                                            |3 (minus reuse)       |jepa_lstm_noc_d64_s{n}             |`--predictor-type lstm --predictor-hidden 256 --predictor-layers 3`                                    |
|T3|Capacity ladder tf-no-c, d in {4,8,16,32} seed 42 + d in {16,32} seeds 0/1                                                                                                                   |8                     |jepa_tf_noc_d{d}_s{n}              |keeps the capacity figure honest with variance at the reported rungs                                   |
|T4|2x2 control, unconditioned, matched heads (lift+wake), seeds 0/1/2 per cell: predictive-CNN, reconstructive-CNN, reconstructive-CNN+ViT (+ predictive-CNN+ViT only if PF-A2 says no reuse)   |9-12                  |ctrl_{obj}_{arch}_s{n}             |ONE matched recipe for all cells; closes referee M1 jointly with B3 below                              |
|T5|Wake-head-removed cell: predictive CNN+ViT, lift-only, no-c, seeds 0/1/2                                                                                                                     |3                     |ctrl_pred_vit_nowake_s{n}          |the co-necessity row (M8 context)                                                                      |
|T6|Conditioned reference (AD2): tf WITH c, d=64, seeds 42/0/1                                                                                                                                   |0 or 3                |jepa_tf_cond_d64_s{n}              |one table row + one sentence only                                                                      |
|T7|Fukami AE: L-curve sweep at d=3 (verify the beta = 0.01 elbow on v2.1), then d in {3,16,32,64} seed 42 + d=64 seeds 0/1/2 + budget parity sweep at d=64 (2 LR schedules x 1.5x iters, 4 runs)|~6 sweep-short + 7 + 4|fukami_d{d}*s{n}, fukami_budget*{k}|recipe per SESSION18_B1_PROTOCOL (ReLU + GroupNorm + future-lift {8,16,24}); budget sweep closes M2(ii)|
|T8|Beta-VAE (AD3): Solera-Rico 2024 recipe ported to `src/baselines/solera_rico.py`, d=64, faithful (lift head) seeds 42/0/1 + matched-head (lift+wake) seeds 42/0/1                            |6                     |bvae_{var}_d64_s{n}                |implementation task first; verify against the 2024 repo recipe via arXiv MCP / the paper               |
|T9|Decoders (LapFiLM + SL recipe): production tf-no-c; matched POST-HOC decoders on frozen Fukami d=64 latent and POD d=64 coefficients; lstm-no-c decoder for animations                       |4 (minus reuse)       |dec_{family}                       |identical decoder recipe and budget across families; closes M2/C5 decode fairness                      |

Launch pattern (canonical two-card usage, `--gpu {0,1}`, never CUDA_VISIBLE_DEVICES):

```bash
# Example: T1 production + first seed in parallel
python -m src.training.train_jepa --all-train --max-iters 20000 --seed 42 \
  --d 64 --B 16 --T 32 --H-roll 8 --lambda-sigreg 0.01 \
  --lr-encoder 1.5e-4 --lr-predictor 5e-4 --weight-decay 0.05 --warmup-frac 0.05 \
  --num-workers 4 --projection-norm batchnorm --anticollapse sigreg \
  --predictor-type transformer --predictor-cond-dim 0 \
  --observable-head cl_future --observable-head-weight 0.01 --observable-head-deltas 0 \
  --wake-observable-type patch_signed_spectrum --lambda-wake 1.00 \
  --wake-loss smooth_l1 --wake-loss-beta 0.5 --wake-head-hidden 128 \
  --split configs/splits/split_v2p1.json \
  --omega-pipeline-manifest outputs/data_pipeline/v2p1/manifest.json \
  --gpu 0 --tag-suffix s28_jepa_tf_noc_d64_s42 \
  --output-dir outputs/runs/session28/jepa_tf_noc_d64_s42/encoder
```

Queue discipline: T1/T2 first (everything downstream needs the lead family), then T4/T5/T6, then
T3/T7/T8 interleaved, T9 last (needs frozen encoders/latents). A simple
`scripts/session28/launch_queue.sh` that drains a per-GPU FIFO is worth writing once.

Gate GA1: every matrix run launched with the full W&B key set; a `manifest_runs.yaml` mapping
tag -> output dir -> W&B id committed. D179.

### A2. Internal undisturbed-flow validation (CPU; closes the computable half of B1)

From the cache (Baseline.h5, `is_calibration_reference: true`): mean C_L, mean C_D, rms C_L, and
shedding Strouhal St over the undisturbed encounters; compare against the PRF 2025 fine-grid
undisturbed values and the two external references PRF validates against (extract their numbers
from the lineage PDF in-session; arXiv MCP fallback). Cross-check the manuscript’s St = 0.36
(referee M13). Output `outputs/session28/undisturbed_stats.json` + a one-row validation block that
Phase D mounts next to Table 1.

```bash
python scripts/session28/undisturbed_validation.py \
  --split configs/splits/split_v2p1.json --out outputs/session28/undisturbed_stats.json
```

### A3. DNS collaborator package (closes B1’s solver-owned half; external dependency)

Send Miro/Lehmkuhl the exact Table 1 row list (Mach / incompressible confirmation; domain + span
Lz/c; element + solution-point counts; minimum wall-normal spacing in wall units; time step + max
CFL; gust-release station x0/c; grid AND time-step sensitivity evidence) with the PRF 2025 Sec. II
paragraph attached as the exemplar, plus the explicit note that we claim DNS (no subgrid model) so
the resolution-evidence bar is HIGHER than their LES reporting. Also request the spanwise extent
and the release station so the Sec. 2.2 wording (referee B1 aggravating detail ii) can be fixed:
our Y envelope is |Y| <= 0.4 versus Fukami’s 0.3, and our release station must be stated, not
implied by citation. Log the send date in D180.

### A4. Protocol freeze (one boxed Appendix-A paragraph; closes M13-rollout, S1, S6)

Freeze and write down, before any evaluation runs:

1. Rollout protocol: full pre-impact context (all available frames up to the impact frame, at most
   T = 32), autoregressive thereafter, NO teacher forcing after impact, identical for every family
   and every figure/table. Markov single-frame seeding is retired (it underperforms unconditioned
   and was the source of the Table 4b vs Table 7 ambiguity, referee B3).
1. R2 estimator: 1 - SSE/SST with SST about the held-out split mean; probes fitted on train
   latents only; 5-fold case-level CV on the readout; bootstrap unit = encounter (n = 2000),
   cluster unit = case; Holm over the 12-test family; primary endpoint = representational wake
   enstrophy at H = 16 on test_b, anchored to HANDOFF D130/D165 + the commit hash (this makes the
   pre-registration claim auditable, referee M9).
1. Probe classes for the closure matrix: ridge (linear primary), KRR-RBF, MLP-reg (the three
   already wired into `physical_metrics_from_rollouts.py`), reported for BOTH endpoints
   (representational and forecast). This is referee Track D run as part of the standard harvest,
   not a separate experiment.
1. Selection convention: headline numbers at fixed d = 64, seed mean +- sd; the full
   family x d x observable x mode x probe matrix lives in the appendix table; the phrase
   “least-bad” is retired.
1. Source groups defined: “periodic” (800-frame cases, 6 encounters) and “run3” (480-frame
   cases, 4 encounters) simulation campaigns; pooled in test_b per the locked rule (closes M13).

Gate GA2: the protocol box exists as `paper/sections/protocol_box.tex` + the same text in
`scripts/session28/PROTOCOL.md`, and `eval_all.py` (A5) reads its parameters from one
`configs/eval_protocol_v2p1.yaml` so code and paper cannot diverge. D181.

### A5. Provenance harness (referee Track 0; closes B3/B4 structurally)

`scripts/session28/eval_all.py`: ONE evaluation entry point that, given `manifest_runs.yaml`,
emits `outputs/session28/numbers.json` mapping every paper-bound quantity to (value, ci_lo, ci_hi,
seed_mean, seed_sd, n, split, endpoint, probe, run_tags, eval-config hash, git commit).
`scripts/session28/emit_macros.py` renders `paper/macros.tex` (one `\providecommand` per number,
e.g. `\NumReprWakeJEPA`, `\CIReprWakeJEPAlo`). Phase D replaces every hand-typed number with a
macro. Skeleton lands in Phase A so Phases B/C write INTO it rather than into ad-hoc CSVs.

### A6. Literature checks (arXiv MCP; respect the one-call-per-minute rate limit; subagent pattern

for large PDFs per CLAUDE.md)

L1. Similarity scalings for Track P1: Sedky and Jones transverse-gust ΔC_L scaling literature;
Smith, Fukami, Sedky, Jones, Taira JFM 2024; any post-2025 vortex-gust scaling result. Goal: a
defensible citation set for the three pre-registered candidate variables (P1 below), and a check
that nobody has already published the collapse (if they have, P1 becomes a confirmation +
latent-inheritance result, still publishable, framed accordingly).
L2. Debiased Sinkhorn: Feydy et al. (interpolating OT divergences) for the S_eps debiasing cite
next to Tran et al. 2026.
L3. qDEIM: Drmac and Gugercin 2016 for the sensor-placement cite.
L4. Competitor refresh: anything new from the Taira/Fukami line and Koshikawa et al. since
2026-06; update the S1 related-work paragraph list (D161) if needed.
L5. Solera-Rico 2024 recipe details for the T8 port (latent dim, beta schedule, transformer
predictor config) verified against the published paper, not memory.

### A7. While GPUs run: no-training statistics prep

Port the session26 paired/clustered machinery (`scripts/session26/track1_stats.py`) into
`scripts/session28/stats_lib.py` as importable functions (paired per-encounter differences,
case-clustered bootstrap, case-permutation p, Holm), so Phases B/C call one library instead of
re-implementing. Unit-test against the committed v2 outputs (same inputs must reproduce D165’s
numbers exactly; this is the regression test that the library is faithful).

### A8. Compute schedule (replace times with PF-A4 measurements)

|Block                  |Runs |Est. GPU-h each|Total                                        |
|-----------------------|-----|---------------|---------------------------------------------|
|T1+T2 JEPA lead        |7    |1.6            |11                                           |
|T3 ladder              |8    |1.0-1.6        |10                                           |
|T4+T5 controls         |12-15|1.6            |19-24                                        |
|T6 conditioned ref     |0-3  |1.6            |0-5                                          |
|T7 AE total            |~17  |0.5-3          |30                                           |
|T8 beta-VAE            |6    |~3             |18                                           |
|T9 decoders            |4    |3.5            |14                                           |
|B1 predictors (Phase B)|~11  |0.6            |7                                            |
|Sum                    |     |               |~110-120 GPU-h, ~2.5-3 days wall on two cards|

Gate GA3 (phase exit): GA0-GA2 green, matrix fully launched or queued, A2/A3/A5/A6/A7 landed.
D178-D182 written. Phase B may start as soon as T1 seed-42 + T7 d-rows + POD exist; it does not
wait for the whole matrix.

-----

## PHASE B (Session 29): Closure, attribution, credibility, statistics core

### B0. Pre-flight

All Phase-A runs converged (W&B summaries green, `gpu_name` contains RTX 6000, no auto-fallback
fired unless documented); POD bases computed for d in {16,32,64}
(`scripts/session18/compute_pod_baselines.sh`, v2p1 pipeline); latents extracted for EVERY family,
d, and seed via the session14/17/18 encode scripts pointed at v2p1 (NPZ keys per CLAUDE.md), plus
the beta-VAE latents through the same exporter. DNS physical metrics + per-frame flow descriptors
regenerated (`exp2_dns_physical_metrics.py`, `exp2_build_targets.py`) on v2p1.

### B1. Predictor-on-top rollouts (forecast endpoint, all families)

SESSION18_B1_PROTOCOL recipe with TWO amendments, locked in the protocol box: `--predictor-cond-dim 0` (AdaLN collapses to identity; unconditioned mandate) and full-context rollout initialisation
(A4 item 1). `--no-output-bn` stays (D129). Train one predictor per (family, d): jepa
{16,32,64}, fukami {3,16,32,64}, pod {16,32,64}, bvae {64}; rollouts on test_b and test_c.
The JEPA row additionally evaluates its OWN jointly-trained predictor (tf-no-c and lstm-no-c)
so the paper can show both “native” and “matched predictor-on-top” forecasts; the cross-family
comparison uses the matched predictor-on-top numbers ONLY (fairness), the native numbers appear
once as the deployment-relevant figure.

### B2. The closure matrix (the missing appendix table; closes B4, S1, S6, referee Track D and G1)

`scripts/session28/closure_matrix.py` (extends `physical_metrics_from_rollouts.py` +
`repr_closure` logic): families x d x 6 observables x {representational, forecast} x {ridge,
krr_rbf, mlp} x seeds, on train + val + test_b (interior/boundary tiers separately and pooled) +
test_c, with encounter bootstrap CIs and case-clustered CIs for the wake rows, all through
`stats_lib`. Everything writes into numbers.json. Table 4 and the headline sentences are then
VIEWS of this matrix under the A4 selection convention.

Probe-class verdict (referee M3, the highest-risk item, now nearly free):

Gate GD. Strong branch: the family ordering on representational wake enstrophy is preserved under
all three probe classes (reconstructive stays low while predictive stays high, with
non-overlapping case-clustered CIs at d = 64): keep “the latent carries/encodes the wake” with a
one-sentence probe-robustness statement + the appendix columns. Weak branch: a nonlinear probe
recovers the reconstructive wake (R2 within 0.15 of predictive): every carries/encodes/keeps claim
is rewritten as LINEAR DECODABILITY (“the predictive objective renders the wake linearly readable;
reconstruction stores it, if at all, in a form no fixed linear readout recovers”), the title drops
any possession verb, and the observability section gains the corresponding caveat. EITHER branch:
the old sentence “no probe can recover information the latent does not carry” is rewritten to “no
probe in the evaluated class recovers …” (it is invalid as written under any outcome). D183.

### B3. The decisive control, endpoint-aligned (closes M1)

From the T4/T5 latents: BOTH endpoints (representational closure and matched-predictor forecast)
for every 2x2 cell and the wake-head-removed cell, 3 seeds, same protocol. Table 6 becomes a
two-endpoint table on the SAME (unconditioned) models as the headline; the old inheritance
sentence (“the unconditioned configuration inherits the attribution”) is deleted because the
attribution is now measured directly where it is claimed.

Gate GB. Strong: unconditioned representational wake margin (predictive minus reconstructive)

> = +0.2 at both architectures with non-overlapping +-1 sd seed bands, and the forecast endpoint
> shows the same ordering. Sec 4.2 is rewritten around the representational columns. Weak: the
> matched-head margin shrinks below +0.2 at either architecture: the headline contrast is reframed
> as “families as configured” with the matched-head margin quoted in the abstract instead of the
> configured-family gap. D184.

### B4. Conditioning floor + conditioned reference

Regenerate the (G, D, Y) -> observable regression floor (D145 machinery) on v2p1; if AD2 = in,
evaluate the T6 conditioned reference through the same closure matrix and print exactly one row +
one sentence (“conditioning the predictor on the gust parameters changes the representational wake
closure by \NumCondDelta, i.e. the unconditioned latent gives up almost nothing”). The floor and
the no-c result jointly close D157’s old framing on the new models.

### B5. Baseline credibility dossier (closes M2)

(i) AE health: training/val loss curves per seed per d, held-out reconstruction MSE + SSIM (L from
the registry, 8.45 for v2p1) per split, peak-vorticity retention; assembled as one appendix figure

- three sentences. (ii) Budget verdict from the T7 parity sweep: the strongest defensible sentence
  is “across N configurations spanning [budget range], held-out representational wake closure never
  exceeds \NumAEWakeBest”. (iii) Published-recipe (tanh) variant numbers move from a clause to an
  appendix row with its probe deficit quantified (already known qualitatively; print it). (iv)
  Decode fairness: the matched post-hoc decoders (T9) exist; physical-space comparisons in Phase C
  use them with every column labelled “post-hoc decode”, the AE-native decode kept as one continuity
  column.

Gate GC. Strong: AE collapse persists across budgets AND beta-VAE wake closure is also near zero
(< 0.2) AND the matched post-hoc AE decode still loses the LEV: the broken-baseline objection is
closed; one sentence per check in Sec 5. Weak: beta-VAE retains the wake (repr R2 >= 0.4) or the
matched decode rescues the AE LEV: the claim narrows to “the deterministic reconstructive
configuration of the Fukami lineage”, the beta-VAE enters Tables 3/4 as a fourth family, and the
abstract is rewritten accordingly. Either branch is publishable. D185, D186.

### B6. Core statistics harvest (closes S1-S3, S5-S7, M10 sentence inputs)

Through stats_lib on the new matrix: case-clustered + Holm verdicts for all 12 paired tests
(expect the v2 pattern, representational wake survives, forecast does not; whatever v2.1 says,
the text follows it); Fig-8-style Spearman trends with case-permutation p and case-coloured
points; the two forecast reversals (if they persist on v2.1) extracted for the honest sentence;
the test_c H = 8 wake value printed; interior/boundary tier split reported.

Gate G-stats (phase exit): closure matrix complete and committed; GD/GB/GC resolved with branch
recorded; numbers.json populated for every Phase-B quantity. D187.

-----

## PHASE C (Session 30): Mechanism coherence + the physics tracks + observability

All sub-tracks are latents/cache-only (CPU + occasional RTX 6000 for encoder/decoder passes) and
run in any order. Each writes into numbers.json and produces figure-ready NPZ/CSV under
`outputs/session28/<track>/`.

### C-E1. Drift reconciliation (closes M6, Table 8 gaps)

Per family (jepa tf-no-c, jepa lstm-no-c, fukami, pod, bvae if in) and d in {32, 64}: relative l2
deviation vs H AND Mahalanobis ratio to the train latent distribution vs H, plus the per-direction
departure spectrum (projection of rollout-minus-encoded onto the encoded covariance eigenbasis).
One two-panel replacement figure; three sentences explaining why the orderings differ (the
reconstructive rollout exits along low-encoded-variance directions: small l2, huge Mahalanobis;
the predictive rollout wanders inside its isotropised cloud). Table 8 rebuilt complete (no missing
rows) + one sentence interpreting ratios below one.

### C-E2. Topology fairness (closes M4)

Vietoris-Rips persistence recomputed per family (a) on raw coordinates (continuity with v2), (b)
after per-family coordinate standardisation (the whitened/Mahalanobis metric), at the canonical 5
percent floor and over the existing robustness grid; PLUS the cleanest control: the undisturbed
no-gust case per family. Gate GE2 strong: fragmentation survives whitening AND the reconstructive
family fragments even the no-gust limit cycle: the topology claim strengthens. Weak: whitening
heals the fragmentation: the figure is reframed from “topology” to “metric organisation” and the
abstract attribution is fixed regardless (fragmentation belongs to the ENCODING, manifold
departure to the ROLLOUT; the v2 abstract conflated them). D188.

### C-E3. Transport, done to the Tran standard (closes M5)

Specify and fix epsilon and the KL relaxation rho; use (or switch to) the DEBIASED Sinkhorn
divergence S_eps with the m+/m- signed split (cite Tran 2026 + Feydy from L2); epsilon sensitivity
at {eps/3, eps, 3 eps}; per-encounter alignment Spearman for jepa, fukami, AND pod (pod’s absence
was the missing comparator); paired per-encounter difference with case-clustered CI +
case-permutation p; field distances as per-encounter paired distributions at impact and H = 16
(not single-frame means). Replace the incoherent pooled-reversal paragraph with the actual
mechanism demonstrated by a one-line decomposition: between-encounter variance of the mean latent
norm per family (the pooled statistic is computed on ENCODED latents, which do not drift; the
reversal is an encoding-scale property). Gate GE3: if POD’s alignment >= JEPA’s, the mechanism
sentence becomes “the predictive objective recovers, at nonlinear compactness, the
trajectory-local transport alignment a linear basis has by construction and the reconstructive
latent loses” (the anticipated, better claim). D189.

### C-E4. Scale-band sensitivity (closes S4)

Headline large-scale tracking numbers recomputed at sigma/c in {0.01, 0.03, 0.05} (Odaka’s own
sweep); one appendix sentence.

### Physics Track P1: Similarity collapse of the interaction and its latent image (NEW)

The lineage characterises the response case-by-case in (G, D, Y) and notes qualitatively that
larger D recovers slower; no quantitative similarity variable is published for the vortex-gust
interaction amplitude (verified against the PRF text; re-verify via L1). Pre-register, BEFORE
fitting, three candidate one-parameter scalings: (s1) G (Kussner-like gust ratio), (s2) G x D
(proportional to the gust circulation of the Taylor profile), (s3) the exact Taylor-vortex
circulation Gamma_g(G, D) integrated numerically from the implemented gust profile in the
preprocessing/code (the honest version of s2; compute it, do not assume the prefactor). Response
amplitudes, per encounter, all on v2p1: peak |Delta C_L| from the undisturbed phase-matched cycle;
peak large-scale wake-enstrophy excursion; peak OT distance S_eps(field, phase-matched baseline);
peak latent Mahalanobis excursion from the limit-cycle tube (predictive family). Fit each response
against each candidate on TRAIN cases (per-|Y| stratified and pooled with Y as a secondary axis),
score collapse by held-out test_b R2 of the one-variable fit and by the variance-reduction ratio
versus the unscaled scatter.

Gate GP1 strong: one candidate collapses BOTH the force amplitude and the latent excursion with
held-out R2 >= 0.8 and the SAME exponent (within CI): main-text figure + the sentence “the learned
latent inherits the similarity scaling of the interaction”, which is the paper’s flagship physics
line. Medium: collapse holds for the force but the latent follows with a different exponent:
report both exponents, the difference IS the finding (the latent weights the interaction
geometry, not just its strength). Weak: no single-variable collapse (Y modulation dominates):
report the Y-stratified result and the negative honestly; the figure moves to the appendix. D190.

### Physics Track P2: The recovery clock (NEW)

Definitions (frozen before computing): physical tau_rec = first post-impact time at which the
large-scale (sigma/c = 0.05) wake enstrophy re-enters the phase-matched baseline-cycle envelope
(+-2 sd of the baseline orbit at matched phase) and stays for >= one shedding period; transport
tau_rec = same rule on S_eps(field, phase-matched baseline); latent tau_rec = first return of z to
the baseline limit-cycle tube (Mahalanobis <= q95 of the baseline orbit) sustained one period.
Encounters that do not recover within the 120-frame window are CENSORED, reported as such
(fraction recovered + censored medians; no silent exclusion; D153’s release-cadence finding is the
expected mechanism for the censored set and gets one sentence). Deliverables: tau_rec(G, D, Y)
maps; tau_rec versus the P1 winning variable; and the held-out question that makes this a latent
result: does the LATENT clock match the PHYSICAL clock per encounter?

Gate GP2: Spearman(latent tau_rec, physical tau_rec) >= 0.7 on test_b: “the latent carries the
recovery clock” enters S4.4 with the map figure. Below 0.7: the maps stand as DNS physics (still a
contribution), the latent-clock sentence is dropped. D191.

### Physics Track P3: Shedding-phase dynamics (NEW, scoped honestly)

Baseline shedding phase from the Hilbert transform of C_L on the undisturbed cycle (St
cross-checked in A2). Per recovered encounter: asymptotic phase shift Delta phi between the
post-recovery oscillation and the undisturbed clock. First, the coverage audit D148 demands:
distribution of impact phases across the dataset (the fixed release cadence under-samples phase;
quantify it, print it). Then Delta phi versus (sign G, |G|, D, Y). The phase-RESPONSE-curve
ambition (Delta phi vs impact phase, the Fukami-Nakao-Taira idiom) is gated on coverage: if the
impact phases span < a quarter cycle, scope to Delta phi versus parameters only and say why.
Latent tie-in: the same Delta phi computed from the latent angle on the limit cycle; agreement
with the C_L-derived value is the latent-phase claim. Gate GP3: report whatever the coverage
allows; the audit itself (a measured limitation of fixed-cadence gust datasets) is worth two
sentences in S5 as guidance for future campaigns. D192.

### Physics Track P4: LEV circulation budget and gust-sign asymmetry (NEW; extends D146)

On DNS fields directly (large-scale band; the decode question is separate): LEV identified per
frame (thresholded negative-vorticity region attached to the suction side, tracking seeded from
the existing exp_lev_tracking machinery), giving Gamma_LEV(t), centroid path, detachment time.
Physics questions: (i) does peak |Delta C_L| correlate with peak Gamma_LEV across the envelope
(direct correlation; NO impulse-theorem claim, D167 stands); (ii) G-sign asymmetry: positive
versus negative gusts interact oppositely with the pre-existing suction-side vorticity (the PRF
text describes the split-and-merge qualitatively; we quantify the budget); (iii) Y modulation of
(i)-(ii). Then the latent image: which families’ post-hoc decodes track Gamma_LEV(t) (regenerating
D146 on v2.1 with the MATCHED decoders, removing the decoder confound from the old version).
Gate GP4: |corr(peak Delta C_L, peak Gamma_LEV)| >= 0.6 held-out: S4.4 paragraph + one panel;
below: appendix. D193.

### Physics Track P5: Where the wake code lives (extends D162/D163 onto v2.1)

(i) Regenerate the cross-encoder collective-code result (combination-minus-best-single gap) and
the entanglement/functional-grouping numbers on the v2.1 unconditioned latents (these are
committed manuscript content, Fig 15 + S5.1, and must be rebuilt regardless). (ii) NEW
energy-information curve: held-out wake-forecast skill versus number of leading PCs per family
(and versus retained energy for POD); resolves the participation-ratio-versus-distributed-code
tension quantitatively and bridges to the informative-decomposition literature already cited.
(iii) NEW footprint quantification: gradient saliency of the wake-forecast direction through the
frozen encoder, averaged over held-out impact frames, OVERLAP-SCORED against the thresholded
|omega| and Q-criterion structure masks (reusing the D126 Q-overlap machinery) so the “the code
reads the LEV and shear layer” sentence carries a number instead of a picture. One figure, one
paragraph; D163’s qualitative decode is superseded. D194.

### C-F. Observability rebuild (closes B5, M7, M11, M12)

(i) Placement: target-blind qDEIM as the PRIMARY placement for the cross-family comparison
(per-family TCSI secondary); regenerate picks on v2p1 unconditioned latents. (ii) Metric defined
in an equation: variance-weighted state-recovery R2 reported ALONGSIDE the mean canonical
correlation over min(d, K x W) directions so anisotropy is visible. (iii) The deployment panel:
pressure (K taps) -> estimated latent -> fixed linear probe -> wake enstrophy, R2 versus K per
family, with direct pressure -> wake regression as the no-latent baseline. (iv) Rebuild the
sensing figure as three panels with ALL families present and direct baselines dashed (the old
figure showed one family against text that claimed three, referee B5). (v) Causality scope: one
sentence that the per-encounter percentile clip uses within-encounter statistics, plus the
training-set causal-clip variant of the pipeline run once to report the (expected negligible)
delta. (vi) Y attribution softened using the existing pressure-recovers-Y result: the limitation
is the post-impact single-frame mid-plane vorticity observable and the Y sampling, not “the data”.
D195.

### C-X. Cheap regenerations and consistency items

Interventional test number regenerated on the new models (the honest negative stays, S5 wording
unchanged unless the number moves); z-norm drift diagnostic; chi_3D cross-reference confirmed
against the committed gate file (optional recompute from raw /u only if the existing v2 numbers
are challenged; raw 3D access via PREVENT_ROOT). Phase exit gate: every Phase-C figure source NPZ
exists, numbers.json complete for C-tracks, all gates resolved with branch text drafted.

-----

## PHASE D (Session 31): Manuscript

### D0. Pre-flight

numbers.json complete; `python scripts/session28/emit_macros.py` produces macros.tex; baseline
build green (latexmk exit 0 on the current main); enforce_conventions baseline captured;
collaborator Table-1 status checked (integrate if arrived, else \pending{} rows stand untouched).

### D1. Macro wiring (closes B3/B4 forever)

Every quoted number in the .tex replaced by its macro; `grep -nE '[0-9]\.[0-9]{2}' paper/sections`
audit at the end must return only macro definitions, table \input files, and physically-typed
constants (Re, alpha, St literature values). Any number that cannot be regenerated from
numbers.json is DELETED from the paper, not patched.

### D2. Structural reshape (the physics elevation)

Results order, mapped from the current manuscript:

- S4.1 Held-out closure (repr-led headline, selection convention stated, the reversal sentence,
  tiers, test_c).
- S4.2 Attribution (unconditioned 2x2 BOTH endpoints; wake-head co-necessity WITH the head
  defined by an equation + its loss weight in Table 2, closing M8; conditioning floor; the single
  conditioned-reference row if AD2 = in).
- S4.3 Mechanism (drift two-panel; whitened topology + no-gust control; debiased transport with
  CIs and POD; scale band). One framing paragraph: the three diagnostics test one metric property.
- S4.4 NEW: Flow physics of the encounter and its latent image (P1 similarity collapse; P2
  recovery clock + maps; P3 phase shift within its measured coverage; P4 LEV budget + sign
  asymmetry). This is the section that answers “physical insights besides ML metrics” and it leads
  the Results narrative arc in the abstract.
- S4.5 The latent code (collective code; grouping/entanglement; energy-information curve;
  footprint overlap).
- S4.6 Physical-space ceiling (scoped per the encode-decode-ceiling paragraph; matched post-hoc
  decodes; LEV tracking re-run fairly).
- S4.7 Sparse-pressure observability (rebuilt three-panel; deployment framing already softened).
- S5 Discussion: transport-consistency principle; honest negatives (interventional, closed-loop
  scope, any forecast-clustering weakness v2.1 shows); limitations incl. the phase-coverage audit;
  one canonical scope paragraph (W9), others trimmed to pointers.

### D3. Title and abstract

Title options (en-dashes per JFM at typesetting only):
(a) “A predictive latent representation preserves wake structure in vortex-gust airfoil
interactions at Re = 5000”  [default recommendation]
(b) “Wake-preserving predictive latent dynamics for parametric vortex-gust airfoil interactions”
(c) “Transport-consistent latent dynamics and recovery of vortex-gust airfoil interactions at
Re = 5000”  [pick this if GP1/GP2 land strong; it foregrounds the physics section]
If Gate GD lands on the weak branch, no possession verb appears in any option (a/b are rewritten
to “renders wake structure linearly readable” phrasing).

Abstract template (drop-in; every number is a macro; one hedge, stated once; fragmentation
attributed to the ENCODING and manifold departure to the ROLLOUT):

“Extreme vortex-gust encounters challenge reduced-order models because the transient load is
governed by the reorganisation of the separated wake, not only by the integrated forces. Using
direct numerical simulations of a NACA 0012 airfoil at alpha = 14 deg and Re = 5000, perturbed by
Taylor vortices spanning gust ratio, core diameter and wall-normal offset, we compare reduced
states at matched dimension under a single shared predictor and probe protocol: a joint-embedding
predictive representation trained with no gust parameters anywhere in the model, a reconstructive
observable-augmented autoencoder, [a beta-variational baseline,] and a proper orthogonal
decomposition basis. Probed directly from held-out fields \NumHorizon frames after gust impact,
the predictive latent recovers the wake enstrophy (R2 = \NumReprWakeJEPA[, essentially matching a
gust-parameter-conditioned reference at \NumReprWakeCond]) where the alternatives do not, and the
per-encounter advantage survives case-level clustering and a family-wide multiplicity correction.
Matched controls attribute the gain to the predictive objective acting together with
wake-observable supervision rather than to the architecture. The mechanism is geometric: the
predictive encoding organises each encounter as a single persistent cycle whose metric tracks the
optimal-transport geometry of the flow in a trajectory-local sense, and its autoregressive rollout
remains on the encoded manifold, whereas the reconstructive encoding fragments the cycle and its
rollout departs the training manifold by an order of magnitude. The latent inherits the flow’s own
clocks: the gust response amplitude collapses with \PhysSimVariable, the latent excursion follows
the same scaling, and the latent return to the shedding limit cycle tracks the physical recovery
time across the parameter envelope. The rolled-out predictor tracks the encounter qualitatively,
and the same state is the most recoverable from sparse wall pressure.”

(The two physics sentences are conditional on GP1/GP2; their weak-branch replacements are written
in the gate texts. Trim to <= 250 words at assembly.)

### D4. Surgical rewrites (claim-keyed; line numbers from main_12 are obsolete after the rebuild)

R1 (probe scope, M3): wherever the text infers representation content from probe failure: “a
failure of the representation as read by the evaluated probe class: no probe in that class
recovers [X] from the reconstructive latent, while the same class reads it cleanly from the
predictive latent” (strengthened back only if GD strong).
R2 (pooled reversal, M5): replace with the norm-variance mechanism sentence from C-E3, ending
“…we therefore claim trajectory-local transport consistency, not a global isometry.”
R3 (selection note, S1): Table 4 note becomes “the maximum over the evaluated d, a selection that
favours the baselines; per-d values with intervals are in Table [closure matrix]”. The word
“least-bad” must not survive a repo-wide grep.
R4 (reversal sentence, M10): after the headline ordering: “The ordering is not uniform: [state the
v2.1 reversals verbatim from the matrix]; the predictive advantage is specific to the spatially
distributed wake observables, which is the paper’s claim.”
R5 (pre-registration, M9): “fixed in advance” -> “fixed in the archived analysis plan (decision
log entries D130/D165, repository commit \CommitHash) before the held-out evaluation”.
R6 (Y attribution, M11): per C-F(vi).
R7 (data provenance): DNS via SOD2D, configuration from Fukami et al.; never “simulations of
Fukami”, never LES; cache grid (192 x 96) explicitly distinguished from the solver mesh; Y
envelope wording “gust profile and parametrisation follow Fukami et al. (2025), with the
wall-normal envelope extended to |Y| <= 0.4 and release station x0/c = \pending{}”.
R8 (register): “keeps the wake” pruned to zero outside any title decision; “fully unconditioned”
to one canonical definition; unconditioned/unconditional unified; splits named validation /
in-distribution test / extrapolation test in prose (repo identifiers stated once in S2.2);
source groups defined per A4 item 5.
R9 (duplicates): merge duplicated C_L/C_D definitions; fix any section self-reference; single
scope paragraph per S5.

### D5. Figure plan (all via figstyle, all saved with the `_v2p1` suffix)

|#  |Figure                                                                                                   |Source                   |Status         |
|---|---------------------------------------------------------------------------------------------------------|-------------------------|---------------|
|1  |Problem + architecture schematic WITH both auxiliary heads drawn                                         |manual edit              |revise (M-fig2)|
|2  |Headline closure (repr-led) + conditioning-floor annotation                                              |closure_matrix           |regenerate     |
|3  |Encounter trace panels with (G,D,Y) labels and H = 16 marked                                             |session17 exp1/exp2 regen|regenerate     |
|4  |Drift two-panel (rel-l2 + Mahalanobis vs H, all families, English labels)                                |C-E1                     |NEW            |
|5  |Topology: generator counts raw + whitened + no-gust control                                              |C-E2                     |revise         |
|6  |Transport: debiased alignment with CIs, jepa/fukami/pod + eps sensitivity inset                          |C-E3                     |revise         |
|7  |Phase-portrait composite (limit cycle, excursion, return)                                                |session17 regen          |regenerate     |
|8  |P1 similarity collapse (force + latent excursion vs winning variable)                                    |P1                       |NEW            |
|9  |P2 recovery maps + latent-vs-physical clock scatter                                                      |P2                       |NEW            |
|10 |P4 LEV budget (Gamma_LEV growth, Delta C_L correlation, sign asymmetry)                                  |P4                       |NEW            |
|11 |Physical-space decode comparison, matched post-hoc decoders, labelled ceiling                            |C decode regen           |revise         |
|12 |Sensing three-panel (state recovery, wake via latent, lift via latent; qDEIM; direct baselines dashed)   |C-F                      |rebuild        |
|13 |Wake code: collective-code bars + energy-information curve + footprint overlap                           |P5                       |revise/extend  |
|14 |Hero rollout (native tf-no-c predictor)                                                                  |session14 rollout regen  |regenerate     |
|App|AE health dossier; closure-matrix table; protocol box; phase-coverage audit; scale sensitivity; lead-time|B5/B2/A4/P3/C-E4/C-F     |NEW/regen      |

Main-body figure budget: if the count exceeds 14 after the physics additions, the candidates to
move to appendices are the old per-family decode strip and the lead-time panel; Carlos names the
final cut.

### D6. Tables

Table 1 DNS (collaborator rows \pending{} + the A2 internal validation block mounted beneath it);
Table 2 + the aux-head loss-weight rows incl. the wake-head equation cross-ref; Table 3 families
(+ beta-VAE row if GC weak branch); Table 4 = views of the closure matrix under the A4 convention
(both endpoints, repr column first); Table 5 floor (n stated, precision trimmed); Table 6 two-
endpoint 2x2 + wake-head-removed + (AD2) conditioned reference; Table 7 retired (its content is
the matrix; the old controls-series table was the B3 contradiction vector); Table 8 drift complete;
Table 9 training-fit retitled; Table 10 paired/Holm regenerated; NEW appendix closure-matrix
table; NEW recovery/censoring table (P2).

### D7. Build gates and go/no-go

GI: latexmk clean from scratch (exit 0, 0 undefined refs/cites, 0 overfull, 0 em-dashes repo-wide,
enforce_conventions diff vs baseline shows no NEW flags); every figure file referenced exists with
the `_v2p1` suffix; macro audit (D1) passes; new_numbers manifest superseded by numbers.json with
every cited source git-tracked.

GO (submission go/no-go, D-final): GA-GP gates all resolved with text matching their branch;
Table 1 populated WITH sensitivity evidence (the one item that can hold submission externally);
abstract/title frozen; Zenodo DOI + license + CRediT + funding filled by authors; reproducibility
package (D174 scaffolding) re-pointed at the v2.1 outputs and tagged. Any single red item holds
submission; the paper does NOT go out with an empty Table 1.

-----

## HANDOFF stubs (provisional; renumber from the live head found in PF-A0)

D178 pre-flight: AD1 verdict [absorb/test_d/none], split [v2p1/v2p2] frozen; prototype reuse
verdicts [tf: reuse/retrain, lstm: …, decoders: …]; CLAUDE.md current-focus updated.
D179 training matrix launched: [N] runs, manifest_runs.yaml committed; wake-stats recompute done.
D180 DNS package sent [date]; undisturbed validation: C_L [..], C_D [..], St [..] vs PRF/refs.
D181 protocol freeze: rollout = full pre-impact context <= 32, no TF; R2/CI/Holm conventions;
primary endpoint anchored to D130/D165 + commit [hash]; selection convention fixed.
D182 provenance harness landed (eval_all + emit_macros skeleton); stats_lib reproduces D165.
D183 probe-class verdict: ordering [preserved/broken] under {ridge, krr, mlp}; claim language set
to [information/linear-decodability]; the “no probe can recover” sentence rewritten.
D184 endpoint-aligned 2x2: repr margin [..] at CNN / [..] at CNN+ViT; branch [strong/weak];
inheritance sentence deleted.
D185 AE credibility: dossier [converged/undertrained]; budget bound on AE wake closure = [..];
tanh-variant row added.
D186 beta-VAE verdict: wake repr R2 = [..]; decision [appendix null / fourth family].
D187 closure matrix + statistics: wake repr case-clustered CI [..], Holm [..]; forecast verdict
[..]; reversals [..]; tiers [..]; test_c [..].
D188 topology: whitened counts [..]; no-gust per family [..]; framing [retained/reframed];
abstract attribution fixed.
D189 transport: eps [..], rho [..], debiased [yes]; paired clustered Delta rho CI [..]; POD
alignment [..]; mechanism sentence replaced.
D190 P1 similarity: winning variable [G / GD / Gamma_g / none]; force exponent [..], latent
exponent [..]; held-out R2 [..]; branch [strong/medium/weak].
D191 P2 recovery: fraction recovered [..]; latent-vs-physical clock Spearman [..]; maps committed;
censoring reported.
D192 P3 phase: impact-phase coverage [..] of a cycle; Delta phi findings [..]; PRC [in/out of
scope].
D193 P4 LEV: corr(peak DCL, peak Gamma_LEV) = [..]; sign asymmetry [..]; matched-decode tracking
[..]; placement [main/appendix].
D194 P5 code: collective gap regenerated [..]; energy-information knee at [..] PCs; footprint
overlap with Q/|omega| = [..].
D195 observability: qDEIM cross-family curves [..]; metric defined; pressure->wake panel [..];
clip-causality delta [..]; Y attribution softened.
D196 manuscript: title [a/b/c]; abstract frozen; restructure done; macro audit pass; figure/table
regeneration complete.
D197 submission go/no-go: [date, outcome, holds].

-----

## Risk register (what can sink which claim, and the planned response)

|Risk                                         |Hit                     |Response                                                                                                        |
|---------------------------------------------|------------------------|----------------------------------------------------------------------------------------------------------------|
|GD weak (nonlinear probe rescues the AE wake)|the central claim’s verb|linear-decodability rewrite, title change; the drift/transport mechanism and the physics section carry the paper|
|GC weak (beta-VAE keeps the wake)            |family generality       |fourth family in tables; claim narrows to the deterministic reconstructive configuration; still a clean result  |
|GB weak (matched-head margin shrinks)        |attribution strength    |abstract quotes the matched margin; configured-family gap demoted to context                                    |
|v2.1 forecast statistics improve vs v2       |none (good)             |if the forecast NOW survives clustering with +7 encounters, promote it back per the data; honesty cuts both ways|
|GP1 no collapse                              |flagship physics line   |Y-stratified result + honest negative; S4.4 leads with P2/P4 instead                                            |
|Collaborator DNS table late                  |submission date         |everything else freezes; GO holds on Table 1 alone; do not fabricate                                            |
|Beta-VAE port slips                          |T8                      |gate AD3 falls back to “appendix null planned, port deferred”, explicitly recorded                              |

End of plan. Fastest informative actions inside Phase A: PF-A2 (reuse verdicts decide ~7 GPU-h)
and PF-A3 (AD1 decides the split before anything trains). Fastest claim-relevant result once
training lands: the probe-class columns of the closure matrix (GD), exactly as in the referee
report; it is now free because the matrix computes them anyway.