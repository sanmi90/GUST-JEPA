# HANDOFF.md

Session handoff document for the vortex-jepa project.

Last updated: 2026-05-17.

If you are picking up this project mid-stream (new collaborator, new Claude session, or
returning after a break), read this document first. CLAUDE.md is the operational guide.
This file is the decision history and the rationale.

## Project summary

We are building an end-to-end Joint-Embedding Predictive Architecture (JEPA) for
parametric vortex-gust airfoil interactions at Re = 5000. The scientific aim is a paper
that:

1. Demonstrates JEPA-style self-supervised representation learning is viable on fluid
   mechanics data with low intrinsic dimensionality.
2. Beats or matches Fukami et al. (PRF 2025) and Solera-Rico et al. (Nat. Commun. 2024)
   on forecasting horizon and probing R^2 at matched latent dimension d.
3. Honestly reports the SIGReg-on-low-intrinsic-dim diagnostic, contributing the
   participation-ratio criterion as a reusable JEPA-for-science methodology.

Researcher: Carlos Sanmiguel Vila (INTA, UC3M).

## Data location (important)

The raw DNS data does NOT live in this repository. It is owned by the PREVENT project
(Carlos's ML turbulence detection effort, which produced these DNS runs), and is shared
with vortex-jepa by reference, not by copy.

- Set `PREVENT_ROOT` to the PREVENT project root before running anything. On Carlos's
  workstation this is `$HOME/PREVENT`. Data files are at
  `${PREVENT_ROOT}/data/raw/periodic/` and `${PREVENT_ROOT}/data/raw/periodic/run3/`.
- The vortex-jepa repo contains only `data_manifest/raw_cases_inventory.yaml` (a
  snapshot of the PREVENT-side inventory at bootstrap time) and `configs/splits/split_v1.json` at the
  repo root (the locked split manifest). Both reference the data by relative path;
  resolution is `Path(PREVENT_ROOT) / case["relative_path"]`.
- If PREVENT regenerates its inventory, copy the new YAML over and re-run
  `python build_split_manifest.py`. The split manifest pins
  `source_inventory.sha256` so a stale inventory will be visible at load time.
- The preprocessed per-encounter cache lives at `${VORTEX_JEPA_CACHE}/{partition}/`
  (default `${PREVENT_ROOT}/data/processed/vortex-jepa/`). Partition v1 currently
  holds 230 encounters across 47 cases (extended by D12, D14, D15, D20). See
  `configs/preprocessing.yaml` for the cache parameters.

## Decision history

Decisions are listed in the order they were taken, each with rationale and alternatives
considered.

### D1: End-to-end JEPA (not hybrid two-stage VAE + transformer)

We pursue an end-to-end LeWM-style JEPA rather than the hybrid scheme that would replicate
Solera-Rico more closely.

Rationale: methodologically more novel, harder to reproduce well, aligns the latent
geometry with the predictive task. The hybrid is retained as one of four required
matched-capacity baselines for the paper.

Alternative considered and deferred: a Fukami-style observable-augmented AE + LSTM/
transformer two-stage system. Listed as the `fukami_ae` baseline.

### D2: Trajectory framing (full-episode autoregressive transformer predictor)

Episodes are treated as full latent trajectories z_{1:T} with c = (G, D, Y) as static
episode descriptor. The predictor is an autoregressive transformer over time with c
injected via AdaLN-Zero plus a time-varying phase variable phi_t.

Rationale: closer to the Solera-Rico transformer-in-latent-space precedent, has known
long-horizon stability under scheduled sampling, matches the V-JEPA 2-AC training
recipe (Assran et al. 2025). Pure one-step LeWM-style prediction with N = 3 history
is too short for vortex impact dynamics (impact lasts ~5 to 10 t/c, which is ~30 frames
at dt = 0.05).

Alternative considered: LeWM exactly (N = 3 history, frame-skip 5). Rejected because the
gust impact dynamics needs longer temporal context.

### D3: Encoder is hybrid CNN + ViT

CNN stem (3 downsampling stages, channels 64 to 256) followed by 6-layer transformer
(hidden 256, 8 heads), pooled to d = 32 via a [CLS] token plus a 1-layer MLP with
BatchNorm.

Rationale: vortex cores have strong local structure (CNN good), but airfoil-vortex
relative configuration is global (attention good). The RTX 6000 96 GB does not impose
parameter budget constraints, so we use the model that best matches the inductive bias.

Alternatives considered:
- Pure ViT-Tiny patch 14 (LeWM exact): rejected as patch-coarse for vortex cores at this
  resolution. Listed as an ablation.
- Pure ConvNet: rejected because it does not match LeWM's SIGReg-on-projection recipe
  cleanly. Listed as an ablation.
- 4-stage CNN stem (input (192, 96) -> 12 x 6 feature map at 256 channels = 72 spatial
  tokens). Not committed in v1; recorded as the deferred "shallow-stem" ablation
  (cheaper attention, coarser features). Decision tabled until the main 3-stage run
  produces results to compare against. A 2-stage variant (48 x 24 = 1152 tokens) is
  also possible but not currently tabled.

### D4: Latent dimension d = 32

Default for the main runs. Sensitivity sweep over {8, 16, 32, 64, 128} is a required
ablation.

Rationale: Fukami's PRF 2025 vortex-gust at Re = 5000 shows three latent dimensions
suffice for reconstruction. Choosing d = 32 leaves SIGReg room to spread the distribution
without enforcing isotropy in a near-singular embedding. The intrinsic dimension of the
manifold is believed to be roughly 5 to 10 (3 static parameters plus impact phase plus
shedding phase plus residual), so d = 32 is roughly 3x to 6x the intrinsic dimension.

### D5: SIGReg with auto-fallback to VICReg

Default anti-collapse: SIGReg with M = 256 projections, 17 Epps-Pulley knots in [0.2, 4],
lambda = 0.1, tuned by bisection over [0.001, 1.0].

Auto-fallback rule (hard-coded): if at iteration 20k the participation ratio
PR < 0.3 * d AND linear probe R^2 for c < 0.7, switch to VICReg with mu = 25.0,
nu = 1.0 (Bardes, Ponce, LeCun, ICLR 2022).

Rationale: LeWM is the published precedent. SIGReg's isotropic Gaussian prior may be
mismatched with the low intrinsic dimension of this dataset, as demonstrated by the LeWM
Two-Room failure mode. The fallback to VICReg matches first and second moments without
forcing higher-order Gaussianity, which is safer for low-intrinsic-dim data.

### D6: Conditioning on c only in the predictor, not the encoder

c = (G, D, Y) enters AdaLN-Zero in every predictor block, plus a time-varying phase
variable phi_t. The encoder is unconditional.

Rationale: a static descriptor injected into the encoder short-circuits the JEPA: the
encoder could learn z_t = c and the predictor would be trivial. The V-JEPA 2-AC, LeWM,
Brain-JEPA, and Echo-JEPA precedents all use predictor-only conditioning.

Sanity ablation: a variant with c in the encoder. We expect probing R^2 for c to remain
high (because the encoder sees c directly) but forecasting horizon to degrade, since the
latent now encodes c redundantly and loses capacity for state.

### D7: Data split locked in configs/splits/split_v1.json (superseded in part by D9)

Single split, no k-fold for the moment. K-fold is deferred until a candidate architecture
is promising (avoid burning compute on cross-validation of architectures that do not work).

Final split (as updated by D9, then amended by D12, all on 2026-05-15):
- Train: 33 cases, 114 encounters (first 4 of 6 periodic, first 3 of 4 run3).
  Baseline is included as a periodic train case.
- Test A (impact-instant generalization): same 33 cases, 48 held-out last encounters
  (last 2 of 6 periodic, last 1 of 4 run3). Baseline contributes its last 2 encounters.
- Test B (parametric interpolation): 6 interior cases pooled across source groups,
  28 encounters.
- Test C (extrapolation, G = +4): 4 cases, 24 encounters, never used for selection.
- 1 calibration reference (Baseline), flagged `is_calibration_reference: true` so
  calibration tools can identify the no-gust reference; it is in train + Test A as
  above, not a separate split.

|G| = 3 stays in training (extrapolation axis is asymmetric: only G = +4 is held out).
Periodic trailing partials are discarded.
Impact frame estimate is 40 (vortex centroid crosses LE at t ~ 1.965 t/c).
Sub-trajectory L = 32 with 70 percent impact-aware sampling, 30 percent uniform.

### D8: PLDM added as the fourth matched-capacity baseline (citation corrected 2026-05-17, see D32)

Final baseline list: POD, Fukami observable-augmented AE, Solera-Rico beta-VAE +
transformer, and PLDM (Sobal, Zhang, Cho, Balestriero, Rudner, LeCun, "Learning from
Reward-Free Offline Data: A Case for Planning with Latent Dynamics Models",
arXiv:2502.14819, February 2025; workshop precursor: Sobal et al., arXiv:2211.10831,
NeurIPS SSL workshop 2022; stress-tested in Sobal et al. 2025). The original D8 cited
arXiv:2211.10831 as the primary PLDM reference; this was incorrect. See D32 for the
correction.

Rationale: PLDM is the direct end-to-end JEPA-from-pixels precursor to LeWM, with a
7-term VICReg-derived objective and six tunable weights. LeWM cites PLDM as the previous
end-to-end alternative and reports an 18 percent gain on Push-T with the simpler 2-term
objective. For our paper, PLDM is the "previous end-to-end JEPA" baseline, and the
contrast SIGReg + 2-term (proposed) vs VICReg + 7-term (PLDM) is the central
methodological claim: simpler anti-collapse plus O(log n) bisection beats PLDM's
O(n^6) grid search, on physics data.

The PLDM seven loss terms (per Sobal et al. 2022 and 2025):
1. Prediction (next-embedding MSE)
2. Variance regularization on z per dimension (VICReg-style hinge)
3. Covariance regularization on z (off-diagonal Frobenius)
4. Temporal smoothness (||z_{t+1} - z_t||^2)
5. Variance regularization on the temporal-difference signal
6. Covariance regularization on the temporal-difference signal
7. Inverse-dynamics-model loss (predict a_t from z_t, z_{t+1})

For our setup with no per-step action, term 7 is replaced by an inverse-dynamics MLP
predicting (G, D, Y, phi_t) from (z_t, z_{t+1}). Implement PLDM faithfully so the
comparison is fair.

### D9: Baseline moved into train + Test A (2026-05-15)

Baseline (the no-gust periodic case, G = D = Y = 0) is now a member of `train`
(encounters 0-3) and Test A (encounters 4-5) like any other periodic case. The
per-case metadata still carries `is_calibration_reference: true` so calibration code
can find the no-gust reference.

Rationale: Carlos directed this on 2026-05-15 during the bootstrap session ("It should
be also be used"). Reserving Baseline for calibration only deprives the JEPA predictor
of clean shedding dynamics at G = 0, which is needed to model the no-gust limit.
The previous policy (D7 as originally written) is superseded by this entry.

Effect on counts:
- Train cases: 30 -> 31
- Train encounters: 104 -> 108
- Test A encounters: 44 -> 46
- A new `n_cases_calibration_reference` field in `configs/splits/split_v1.json` summary equals 1.

Alternative considered: keep Baseline excluded from train but make it accessible by
flag for calibration runs. Rejected because it duplicates the data path and adds a
special case the model never sees during training.

### D10: Path layout for the bootstrap session (2026-05-15, revised same day)

The aspirational repo layout in CLAUDE.md places the split manifest under
`configs/splits/`, the inventory under `configs/`, and the build script under
`scripts/`. At the start of the bootstrap session all three files were elsewhere
(`split_v1.json` at the repo root, `data_manifest/raw_cases_inventory.yaml`,
`build_split_manifest.py` at the repo root). The original D10 left them in place
to avoid rewiring relative paths mid-session.

Carlos approved moving the split manifest later the same day. Final state after
the Session 1 follow-up:

- `configs/splits/split_v1.json` - moved here from the repo root via `git mv`,
  contents unchanged. SHA256 of the manifest is unchanged by the move:
  `44ea16ba87dfbfd6ec78a165553c1d95b0df329afa6d711774a592f12bb7aa21`. All code
  and doc references updated to the new path; the four-check loader smoke test
  still passes.
- `data_manifest/raw_cases_inventory.yaml` - stays at `data_manifest/`. The
  divergence from the aspirational `configs/raw_cases_inventory.yaml` is
  low-stakes and may be revisited.
- `build_split_manifest.py` - stays at the repo root. Carlos's spec mentions
  it by name without a directory; relocation under `scripts/` is also a
  low-stakes divergence and may be revisited.

### D11: Rename impact_aware_start_range -> impact_overlap_start_range (2026-05-15)

The locked range `[8, 40]` with `L = 32` produces sub-trajectories whose intersection
with the impact window `[25, 55]` contains at least 7 frames. This is what the
"impact-aware" branch of the sampler actually guarantees. The previous name
suggested "guarantees frame 40 is in the sub-trajectory", which is true only for
`start >= 9` (since `start = 8` yields `[8, 40)`).

Resolution: rename the field to `impact_overlap_start_range` everywhere
(`configs/splits/split_v1.json`, `build_split_manifest.py`, `src/data/episode_dataset.py`).
`impact_aware_fraction` keeps its name (it is the mixture weight, not a range).
Behavior is unchanged; the 0.814 observed vs 0.811 predicted impact-overlap
fraction is the validation that the sampler does what it should.

Rationale: the issue was purely a misleading name; the math and code are correct.
Renaming is the lowest-risk fix and avoids the alternatives (shift range to
`[9, 40]` or widen to `L = 33`, both of which change behavior). The semantics are
now documented inline in the `subtrajectory_sampling.rationale` field of
`configs/splits/split_v1.json` and in the `EpisodeDataset` docstring.

Alternative considered: redefine `L` or the range so frame 40 is strictly
in-window. Rejected because behavior is fine; the original name was wrong.

`configs/splits/split_v1.json` SHA256 after the rename:
`44ea16ba87dfbfd6ec78a165553c1d95b0df329afa6d711774a592f12bb7aa21`. This is the
manifest hash to log under `split_sha256` in W&B (see CLAUDE.md "Logging (W&B)").

### D12: Absorb two new run3 cases into v1 (2026-05-15, late session)

Carlos's collaborator dropped two run3 files in `$PREVENT_ROOT/data/raw/periodic/run3/`
(`Gust_023_x-1.989_y-0.290_s1.0_d1.5.h5` and
`Gust_024_x-1.892_y-0.678_s-1.0_d1.0.h5`). Decoded:
- `G+1.00_D1.50_Y+0.20` (run3, defaults to `train`)
- `G-1.00_D1.00_Y-0.20` (run3, defaults to `train`)

Rather than create v2 (which the original plan in SESSION_DATA_PREP.md Step 5 would
prescribe), the two cases were absorbed directly into v1 per Carlos's direction
("Add everything into v1, update whatever you need"). v1 is no longer the 41-case
partition it was at the close of the bootstrap session; it is now 43 cases /
214 encounters.

Rationale: at this stage of the project (Session 2 starting on three model
primitives), maintaining a separate v2 partition for two extra cases would add
versioning overhead with little benefit. v1 has not yet produced any reported
training checkpoint, so the partition-immutability rule in D5 has not yet had to
bite. Once v1 produces a paper-reportable checkpoint, the next absorption MUST
go to v2.

Effect on counts:
- Train cases: 31 -> 33 (+2 new run3 train cases).
- Train encounters: 108 -> 114 (+6 = 2 cases x 3 encounters each).
- Test A encounters: 46 -> 48 (+2 = 2 cases x 1 encounter each).
- Total cases: 41 -> 43.
- Total encounters: 206 -> 214.

Cache:
- 8 new encounter files written to
  `${VORTEX_JEPA_CACHE}/v1/{G+1.00_D1.50_Y+0.20, G-1.00_D1.00_Y-0.20}/encounter_*.h5`.
- The 206 existing encounter files are untouched (preprocess.py skipped them).

`configs/splits/split_v1.json` regenerated. New SHA256:
`0f07a746383dc38e0ea7c4841d3559468ca8b4d9e2e2ab493996ac636c07a096`
(the pre-absorption SHA documented in D11 is `44ea16ba...`, preserved in git
history at commit 78b0fa1). When logging W&B `split_sha256` for runs that touch
the absorbed v1, use the new hash.

Alternative considered: build v2 with these two cases (per the original Step 5
plan). Rejected as premature partition-versioning at the current pre-training
stage. The four-check loader smoke test was re-run with the updated counts and
still passes (114 / 48 / 28 / 24, overlap fraction 0.804, seed=42 reproducible).

### D15: Absorb one more run3 case into v1 (2026-05-16, late)

Carlos's collaborator dropped a third run3 file in
`$PREVENT_ROOT/data/raw/periodic/run3/` later the same day as D14
(`Gust_028_x-1.989_y-0.290_s-0.5_d0.5.h5`, timestamped 2026-05-16 21:17;
Gust_027 was skipped by the collaborator's numbering, the same pattern
as the earlier missing Gust_018). Decoded with the locked alpha=14 degree
rotation:

- `G-0.50_D0.50_Y+0.20`  (run3, defaults to `train`)

The new case_id does not collide with the existing inventory; |G|=0.5 stays
inside the training envelope (|G| <= 3, only |G|=4 is held out in Test C).

Same precedent as D12 and D14: v1 still has no paper-reportable training
checkpoint, so this absorption stays in v1. The next absorption after the
first reportable v1 run MUST go to v2.

Effect on counts (cumulative since D14):
- Train cases: 35 -> 36 (+1 run3 train case).
- Train encounters: 120 -> 123 (+3 = 1 case x 3 train-encounter slots).
- Test A encounters: 50 -> 51 (+1 = 1 case x 1 held-out encounter).
- Total cases: 45 -> 46.
- Total encounters: 222 -> 226.

Cache:
- 4 new encounter files written at
  `${VORTEX_JEPA_CACHE}/v1/G-0.50_D0.50_Y+0.20/encounter_*.h5`.
- The 222 pre-existing encounter files are untouched (preprocess.py skipped them).

`data_manifest/raw_cases_inventory.yaml` regenerated via
`scripts/100c_raw_cases_inventory.py`; summary now reports
`n_cases_total: 46`, `n_cases_periodic: 21`, `n_cases_run3: 25`,
`n_parse_errors: 0`, `n_duplicate_case_ids: 0`. New inventory SHA256:
`2b7d7a240c92b191684c29d7b6c721c8dff23543216620b4c02cdfcb00641611`
(pinned in the split manifest at `source_inventory.sha256`).

`configs/splits/split_v1.json` regenerated via `python build_split_manifest.py`.
New SHA256:
`9df7b733b9bc0161aed205571f3a0273416e829fda9d7a6660f9bb7aa040a81a`
(D14's hash `f21abb5d48008031d628042bd46743a82e3dd28c194e8a66dc22e7dee8b8bf8c`
is preserved in git history at commit 77b71fc). When logging W&B
`split_sha256` for runs that touch the absorbed v1, use the new hash.

Alternative considered: build v2 with this case alongside D14's two cases.
Rejected for the same reason as D12/D14 -- premature partition-versioning
while the project still has no v1 training checkpoint.

### D14: Absorb two more run3 cases into v1 (2026-05-16)

Carlos's collaborator dropped two more run3 files in
`$PREVENT_ROOT/data/raw/periodic/run3/` overnight
(`Gust_025_x-1.916_y-0.581_s-1.0_d1.5.h5` and
`Gust_026_x-1.989_y-0.290_s-1.5_d1.0.h5`, both timestamped 2026-05-16 09:17).
Decoded with the locked alpha=14 degree rotation:

- `G-1.00_D1.50_Y-0.10`  (run3, defaults to `train`)
- `G-1.50_D1.00_Y+0.20`  (run3, defaults to `train`)

Both new case_ids do not collide with the existing inventory; both stay
inside |G| <= 3, so neither pushes the extrapolation envelope (|G| = 4 stays
held out in Test C).

Following D12's pattern, these were absorbed into v1 rather than v2: v1 has
still not produced a paper-reportable training checkpoint, so the
partition-immutability rule has not yet had to bite. The next absorption
after the first reportable v1 run MUST go to v2.

Effect on counts:
- Train cases: 33 -> 35 (+2 new run3 train cases).
- Train encounters: 114 -> 120 (+6 = 2 cases x 3 train-encounter slots each).
- Test A encounters: 48 -> 50 (+2 = 2 cases x 1 held-out encounter each).
- Total cases: 43 -> 45.
- Total encounters: 214 -> 222.

Cache:
- 8 new encounter files written at
  `${VORTEX_JEPA_CACHE}/v1/{G-1.00_D1.50_Y-0.10, G-1.50_D1.00_Y+0.20}/encounter_*.h5`.
- The 214 pre-existing encounter files are untouched (preprocess.py skipped them).

`data_manifest/raw_cases_inventory.yaml` regenerated via
`scripts/100c_raw_cases_inventory.py`; summary now reports
`n_cases_total: 45`, `n_cases_periodic: 21`, `n_cases_run3: 24`,
`n_parse_errors: 0`, `n_duplicate_case_ids: 0`. New inventory SHA256:
`d67d65d369097875403169c8065f56d4612479be2b4712a177d8d7505d76f74f`
(pinned in the split manifest at `source_inventory.sha256`).

`configs/splits/split_v1.json` regenerated via `python build_split_manifest.py`.
New SHA256:
`f21abb5d48008031d628042bd46743a82e3dd28c194e8a66dc22e7dee8b8bf8c`
(D12's hash `0f07a746383dc38e0ea7c4841d3559468ca8b4d9e2e2ab493996ac636c07a096`
is preserved in git history at commit 029226f). When logging W&B
`split_sha256` for runs that touch the absorbed v1, use the new hash.

Alternative considered: build v2 with these two cases. Rejected for the same
reason as D12 -- premature partition-versioning while the project still has
no v1 training checkpoint to compare against.

### D16: Default predictor conditioning is c = (G, D, Y), no phi_t (2026-05-16)

The predictor's AdaLN-Zero conditioning at the default run uses the static
descriptor c = (G, D, Y) only. The phase variable phi_t proposed in the
architectural specification Section 3.4 is not part of the default. The
predictor's internal AdaLN call still takes a (B, T, cond_dim) tensor with
cond_dim = 3 (c broadcast across t), so a future switch to cond_dim = 4 is a
one-line change.

Rationale: closer to the LeWM precedent (LeWM uses per-step actions only because
the environments have natural per-step actions; ours does not), simpler data
loader contract (no phi field in the batch), no normaliser choice to lock down.
The architectural spec ablation 13 (with vs without phi_t) remains relevant; the
default now becomes "without", and "with" becomes the variant ablation if
forecast horizon comes in soft.

Alternative considered: include phi_t as the kinematic centroid-to-LE distance
in normalised convective time. Deferred. If H1's forecast-horizon target
(factor of 2 over Fukami AE at epsilon = 0.1) is not met at the end of the
first full training run, this is the first mitigation to try, before deeper
predictor / more dropout / more weight decay.

Effect on the batch contract: the planned batch dictionary is
`{'omega': (B, T, 1, H, W), 'c': (B, 3)}`. No `phi: (B, T)` field.

### D17: Encoder projection uses BatchNorm per LeWM, with documented LeJEPA caveat (2026-05-16)

The encoder's [CLS] -> latent projection head uses `nn.BatchNorm1d(d)` as the
final layer, NOT `nn.LayerNorm(d)`. This follows LeWM Section 3.1
(arXiv:2603.19312):

"The projection step maps the [CLS] token embedding into a new representation
space using a 1-layer MLP with Batch Normalization. This step is necessary
because the final ViT layer applies a Layer Normalization, which prevents
our anti-collapse objective from being optimized effectively."

Caveat: the LeJEPA official reference implementation
(github.com/galilai-group/lejepa, by Balestriero) reports that across 10+
datasets and 60+ architectures at ImageNet scale, "no clear difference observed
between LayerNorm and BatchNorm, so we used LayerNorm consistently." So
"SIGReg requires BatchNorm" overclaims; the more accurate statement is that
LeWM specifically observed the LayerNorm-vs-anti-collapse interaction in its
small-environment, low-intrinsic-dim regime, and that our setting (small
dataset, intrinsic dim ~5 to 10, single GPU) is closer to LeWM's than to
LeJEPA's.

Decision: follow LeWM in the default. Document the caveat so that if
participation-ratio diagnostics show partial SIGReg collapse (pre-registered
hypothesis H4), the FIRST diagnostic intervention is to retry with LayerNorm at
the projection, BEFORE invoking the VICReg auto-fallback at iteration 20k.
This adds one cheap contingency between the default and the fallback.

Rationale: LeWM is the direct architectural template (CLAUDE.md "What we are
building"). The LeWM ablations were performed at our regime; LeJEPA's were
performed at a much larger scale. Where the two disagree, LeWM is the more
relevant precedent for this project.

Alternative considered: follow LeJEPA's reference (LayerNorm at the
projection). Rejected because the LeJEPA finding is at a scale that does not
match our setting, and because keeping the BatchNorm path makes the LeWM
precedent reproduction cleaner.

Effect on the encoder spec: `src/models/encoder.py` final layer of the
projection is `nn.BatchNorm1d(latent_dim)`, asserted by a unit test
(`test_encoder_projection_is_batchnorm`).

### D13: SIGReg follows LeWM Appendix A, no N multiplier (2026-05-16)

The Session 2 implementation of `src/models/sigreg.py` uses the LeWM appendix-A
definition of the Epps-Pulley statistic:

```
T^(m) = integral over t of  w(t) * |phi_N(t; h^(m)) - phi_0(t)|^2  dt
SIGReg(Z) = (1 / M) sum_m T^(m)
```

There is no leading `N` multiplier. This contradicts the official LeJEPA paper
PyTorch listing (arXiv:2511.08544, Lst. "epps-pulley-pytorch"), which ends with
`T = torch.trapz(err, t, dim=1) * N`. The applied LeWM paper (arXiv:2603.19312
appendix A, equation EP) gives the definition without the `N` multiplier and is
the more authoritative source for this project's training recipe.

Effect on the unit-test thresholds in `tests/test_sigreg.py`: the original
SESSION2_MODEL_PRIMITIVES.md spec proposed thresholds (Gaussian < 0.1,
Student-t df=2 > 5.0, Uniform > 1.0) that are not simultaneously satisfiable
under either convention (with multiplier the Gaussian asymptotic mean is ~1.0;
without it the Student-t empirical value at B=4096 is ~0.12). Thresholds were
re-calibrated empirically against a numpy reference for the no-multiplier
formula on B=4096 batches:

- Gaussian            < 0.01   (empirical ~ 1e-4)
- Student-t df=2      > 0.05   (empirical ~ 0.12)
- Uniform(-1, 1)      > 0.02   (empirical ~ 0.05)

All six SIGReg unit tests pass. The relative ordering (Gaussian << Uniform <
Student-t) is preserved and is what the regularizer needs to discriminate to
work as an anti-collapse signal. The numerical scale of SIGReg in training is
absorbed into the outer regularization weight `lambda` (CLAUDE.md "Locked
decisions" allows `lambda` to be tuned by bisection over [0.001, 1.0]); the
choice of scaling here does not affect the bisection's logical search range,
only the numerical value of the optimum.

Alternative considered: use the LeJEPA paper code's `* N` multiplier and
re-calibrate the Gaussian threshold up to < 2.0. Rejected because LeWM is the
direct architectural template for this project (CLAUDE.md), and the LeJEPA
paper's main-text definition (Section 4.2.3, equation Epps-Pulley) is also
written without the multiplier; the `* N` in the PyTorch listing is an
implementation choice that does not survive the appendix-A presentation that
LeWM cites.

Knot range stays at `[0.2, 4]` per the spec, even though LeJEPA's reference
code uses `[-5, 5]`. The half-axis choice is harmless: the integrand is
symmetric in `t` and the integrand at `t in [0, 0.2)` is negligible (both
phi_N and phi_0 equal 1 at `t = 0`).

### D19: RTX 6000 Blackwell is the only supported training GPU (2026-05-17)

All training, smoke-test, and benchmark runs MUST use the RTX 6000 Blackwell
(sm_120) GPU. The workstation also exposes two NVIDIA L40S (sm_89) cards;
those must NOT be used for vortex-jepa runs so paper compute is on a single,
named accelerator class. Silent CPU fallback is also forbidden.

Enforcement:
- `src/utils/device.py:require_rtx6000()` is the canonical accessor. It
  walks `torch.cuda.device_count()`, picks the first device whose name
  contains both `RTX` and `6000`, runs a tiny probe kernel
  (`torch.zeros(4, device=d) + 1`) to confirm the installed PyTorch wheel
  actually ships kernels for sm_120, and returns a `torch.device` or
  raises `NoRTX6000Error` with a message that lists what torch DID see
  and the suggested reinstall command.
- Training entrypoints call this at startup; tests that genuinely exercise
  CUDA paths (currently only `test_encoder_bf16_autocast_roundtrip`) call
  it and `pytest.skip` if it raises, rather than silently falling back to
  CPU.
- W&B runs log `gpu_name` and the run is considered untraceable for the
  paper if that field does not contain `RTX` and `6000`.

Driver/wheel state at the time of the rule:
- nvidia-smi: 580.95.05, CUDA 12.0, four GPUs visible (two RTX 6000
  Blackwell, two L40S). The Blackwell cards show as devices 2 and 3 in
  torch's default ordering (`FASTEST_FIRST`); helper indexes the right one
  regardless.
- PyTorch was upgraded from `2.1.2+cu121` (sm_50..sm_90 only, silently fell
  back to L40S / CPU on Blackwell) to `2.12.0+cu130` on 2026-05-17. The
  cu130 wheels on the default PyPI index ship kernels for sm_120 and pass
  the probe.
- `requirements.txt` was re-pinned to `torch==2.12.0`, `torchvision==0.27.0`,
  `torchaudio==2.11.0`. The cu128 install via `pytorch.org` was attempted
  first but the CDN was unreachable from the workstation; the default
  PyPI index works and ships an equivalent build.

Alternative considered: allow L40S as a fallback. Rejected because mixing
accelerator classes inside a single paper would confuse the reproducibility
section, and the smaller L40S memory (48 GB vs 96 GB) constrains batch
size / sub-trajectory length in ways the Blackwell run does not. The L40S
cards remain available for unrelated work on the same workstation.

### D20: Absorb one more run3 case into v1 (2026-05-17)

Carlos's collaborator dropped a fourth run3 file in
`$PREVENT_ROOT/data/raw/periodic/run3/` overnight relative to D15
(`Gust_030_x-1.892_y-0.678_s1.0_d1.0.h5`, timestamped 2026-05-17 09:17;
Gust_029 was skipped by the collaborator's numbering, the same pattern
as the earlier missing Gust_018 and Gust_027). Decoded with the locked
alpha=14 degree rotation:

- `G+1.00_D1.00_Y-0.20`  (run3, defaults to `train`)

The new case_id does not collide with the existing inventory; |G|=1.0
stays well inside the training envelope (|G| <= 3, only |G|=4 is held out
in Test C).

Same precedent as D12, D14, D15: v1 still has no paper-reportable training
checkpoint, so this absorption stays in v1. The next absorption after the
first reportable v1 run MUST go to v2.

Effect on counts (cumulative since D15):
- Train cases: 36 -> 37 (+1 run3 train case).
- Train encounters: 123 -> 126 (+3 = 1 case x 3 train-encounter slots).
- Test A encounters: 51 -> 52 (+1 = 1 case x 1 held-out encounter).
- Total cases: 46 -> 47.
- Total encounters: 226 -> 230.

Cache:
- 4 new encounter files written at
  `${VORTEX_JEPA_CACHE}/v1/G+1.00_D1.00_Y-0.20/encounter_*.h5`.
- The 226 pre-existing encounter files are untouched (preprocess.py
  reported `written=4, skipped=226`).

`data_manifest/raw_cases_inventory.yaml` regenerated via
`scripts/100c_raw_cases_inventory.py`; summary now reports
`n_cases_total: 47`, `n_cases_periodic: 21`, `n_cases_run3: 26`,
`n_parse_errors: 0`, `n_duplicate_case_ids: 0`. New inventory SHA256:
`8c7202e1c8b6d8055f5e320733cf639746999504f631a4e2551c9eaecd419282`
(D15's hash `2b7d7a240c92b191684c29d7b6c721c8dff23543216620b4c02cdfcb00641611`
is preserved in git history).

`configs/splits/split_v1.json` regenerated via `python build_split_manifest.py`.
New SHA256:
`6fa9fd149da1a0d37bb80af0a4381bf7004665bcfce3402d558a04446fe76ae0`
(D15's hash `9df7b733b9bc0161aed205571f3a0273416e829fda9d7a6660f9bb7aa040a81a`
is preserved in git history). When logging W&B `split_sha256` for runs
that touch the absorbed v1, use the new hash.

Alternative considered: build v2 with this case. Rejected for the same
reason as D12/D14/D15 -- premature partition-versioning while the project
still has no v1 training checkpoint to compare against.

### D21: Scheduled sampling is V-JEPA 2-AC-faithful with H_roll = 8 (2026-05-17)

Session 4 implements scheduled sampling as a two-loss sum with fixed
coefficients,

```
L_total = L_pred + 0.5 * L_roll + lambda * L_anticollapse
```

where `L_pred` is teacher-forced one-step MSE over the full `T - 1`
positions of the sub-trajectory and `L_roll` is open-loop rollout MSE
over `H_roll = 8` steps from one random start position per forward pass.
This is the V-JEPA 2-AC recipe (Assran et al., arXiv:2506.09985, 2025,
Section 6 and appendices) transposed to our setting; it is NOT Bengio
probabilistic teacher-student mixing.

Two transpositions from the V-JEPA 2-AC original:

- Teacher-forced loss covers `T - 1 = 31` positions (V-JEPA 2-AC uses 15
  because its architecture exposes 16 frame slots at a time; we have
  access to the full sub-trajectory).
- Rollout horizon is `H_roll = 8` (CLAUDE.md "Locked decisions,
  Training"). V-JEPA 2-AC uses `H_roll = 2`, which is too short for
  vortex impact dynamics that last 5 to 20 t/c (100 to 400 effective
  frames at `dt_eff = 0.05`; see D34 for the frame-skip correction).
  At `H_roll = 8` and `dt_eff = 0.05` the rollout horizon covers
  ~0.4 t/c, still well below the impact dynamics span, but four times
  longer than V-JEPA 2-AC's `H_roll = 2`.

Rationale: the two-loss sum is the simplest faithful translation of the
LeWM `L_pred + lambda * L_sigreg` objective extended with rollout from
V-JEPA 2. Bengio probabilistic mixing was rejected because it adds a
hyperparameter axis (the teacher-forcing probability schedule) with no
published precedent for JEPA-style models, and the two-loss sum is
simpler to ablate against (just turn off `rollout_weight`).

Implementation: `src/training/scheduled_sampling.py` defines two free
functions, `teacher_forced_prediction_loss(z_target, z_hat)` and
`open_loop_rollout_loss(predictor, z_target, cond, start_t, horizon)`.
The JEPA wrapper composes them with `rollout_start_strategy` chosen at
construction time (`fixed_zero` for unit tests; `uniform_random` for
training; `impact_aware` reserved for Session 5+ ablation).

Alternative considered: Bengio scheduled sampling with `p_tf` annealed
from 1.0 to 0.5 over 30 percent of training. Rejected per the reasoning
above.

### D22: VICReg coefficients are mu = 25, lambda = 25, nu = 1, gamma = 1, invariance term dropped (2026-05-17)

The auto-fallback VICReg (HANDOFF.md D5) uses the Bardes ICLR 2022 default
coefficients `mu = 25, lambda = 25, nu = 1, gamma = 1` (arXiv:2105.04906,
Section 3). The invariance term parameterised by `lambda` requires a
second view of each sample (`z_a, z_b` pair), which JEPA without paired
augmentations does not have (HANDOFF.md D6). Per the H-JEPA reference
implementation (Wiggins, 2026) and the PLDM precedent (Sobal et al.,
arXiv:2211.10831, 2022), the standard solution is to drop the invariance
term and keep `mu * L_var + nu * L_cov` only.

Effect on the public API: `src/models/vicreg.py` constructor takes all
four arguments (`mu, lambda_, nu, gamma`) for forward-compatibility with
future ablations that introduce a second view (for example, the
symmetry-augmentation pair listed as open question 6). The default
forward pass ignores `lambda_` and computes only the variance hinge plus
the off-diagonal covariance Frobenius norm. A unit test
(`test_vicreg_lambda_argument_is_inert_without_second_view`) asserts that
varying `lambda_` does not change the loss output.

Numerical note: the variance hinge target is the per-dimension standard
deviation (`sqrt(var + eps)`), not the variance itself, per Bardes et al.
equation (1). The `eps = 1e-4` default prevents infinite gradients when
a latent dimension approaches zero variance; an all-zero batch produces
a loss of approximately `mu * (gamma - sqrt(eps)) = 25 * 0.99 = 24.75`,
not the dimensionally-suggestive `mu * gamma = 25`.

Supersedes CLAUDE.md "Risk-management" which previously listed
`mu = 25.0, nu = 1.0` without specifying `lambda` or `gamma`. The new
canonical reference is this entry.

Alternative considered: replicate the full Bardes three-term loss with a
synthetic second view (e.g., temporally jittered `z_{t+1}` for `z_t`).
Rejected because (a) it conflates the invariance objective with the
prediction objective the JEPA already optimises, and (b) it forces an
augmentation choice the project does not have a basis to make at this
stage.

### D23: Slow integration tests are opt-in via pytest --runslow (2026-05-17)

The full integration test for the training entrypoint
(`tests/test_train_jepa_smoke.py`) runs a 20-iteration end-to-end JEPA
training loop on the Baseline case. This takes roughly 30 seconds on the
RTX 6000 Blackwell and instantiates the full data loader, optimizer,
scheduler, autocast, diagnostics, and checkpoint paths. It is the most
valuable single test in the suite because it exercises the wiring no
unit test can reach, but at 30 seconds it would slow the default
`pytest tests/` run to over a minute.

Solution (`conftest.py`): register a `slow` marker plus a `--runslow`
CLI option. By default the marker is skipped; passing `--runslow` runs
the slow tests too. This is the canonical pytest opt-in pattern.

Usage:

```
pytest tests/            # fast suite, 71 passing in ~95 seconds, 1 skipped
pytest tests/ --runslow  # full suite, 72 passing in ~125 seconds
```

CI runs the fast form. Local pre-PR runs should include `--runslow`
when touching `src/training/train_jepa.py`, `src/models/jepa.py`,
`src/data/`, or any module that participates in the training loop.

### D24: Session 5 5-case smoke subset (2026-05-17)

The Session 5 5k-iter smoke run uses a deliberately chosen 5-case
subset stored at `configs/cases/smoke_5cases.yaml`:

- `Baseline`                    (periodic, G=0, D=0, Y=0; calibration reference per D9)
- `G+3.00_D0.50_Y+0.40`         (run3,    G=+3, D=0.5, Y=+0.4)
- `G-3.00_D1.00_Y-0.20`         (run3,    G=-3, D=1.0, Y=-0.2)
- `G+1.00_D1.50_Y+0.20`         (run3,    G=+1, D=1.5, Y=+0.2)
- `G+1.00_D1.00_Y-0.20`         (run3,    G=+1, D=1.0, Y=-0.2)

Total: 16 train encounters + 5 test_a held-out encounters.

Rationale: random selection across sessions would make Session 5/6/7
results incomparable. Pinning the subset means the methodological
finding (decision string from `notebooks/01_smoke_5k_analysis.ipynb`)
is reproducible across reruns. The subset spans the G axis from -3 to
+3 (the full training G envelope; |G|=4 is reserved for Test C), all
four D values (0, 0.5, 1.0, 1.5), both signs of Y/c, and exercises both
source groups.

Substitutions from the Session 5 plan: the plan named four periodic
cases plus one run3 case. Two of the planned periodic ids
(`G+3.00_D0.50_Y+0.20` and `G+1.00_D1.50_Y+0.10`) do not exist in
`configs/splits/split_v1.json` because periodic has no |G|=3 cases and
no D=1.5 cases (only run3 covers those parameter combinations). The
closest available manifest cases were substituted (`G+3.00_D0.50_Y+0.40`
and `G+1.00_D1.50_Y+0.20`, both run3), preserving the G/D/Y coverage
intent at the cost of a 1-periodic + 4-run3 split instead of the
planned 4 + 1. The third planned id (`G-3.00_D1.00_Y-0.20`) was
labelled periodic in the plan but is actually a run3 case in the
manifest; this was a plan-side misreading, not a substitution.

The subset is NOT a split (it is not part of `configs/splits/split_v1.json`).
It is a runtime case selector consumed by
`train_jepa.py --cases-from configs/cases/smoke_5cases.yaml`.

Alternative considered: bootstrap a smaller dedicated split file
(e.g., `split_smoke5.json`) for the same case list. Rejected because
the partition manifest is the data-versioning surface (D11, D12, D14,
D15, D20) and adding a sub-split there would dilute the meaning of a
"partition". The runtime selector lives in `configs/cases/`, separate
from `configs/splits/`, so the two concerns stay clean.

### D25: --projection-norm flag on the encoder and train_jepa entrypoint (2026-05-17)

`HybridCNNViTEncoder` gains a `projection_norm: str = "batchnorm"`
constructor argument. The default keeps the LeWM-faithful BatchNorm
projection (HANDOFF.md D17); `projection_norm="layernorm"` swaps in
`nn.LayerNorm(latent_dim)` at `proj[-1]`. The Linear in front of the
norm is unchanged.

`scripts/.../train_jepa.py` gains `--projection-norm {batchnorm,layernorm}`,
passed through to the encoder constructor and logged under the
W&B `projection_norm` config key.

Rationale: D17 names BatchNorm as the canonical projection but also
records the LeJEPA caveat (no observed difference at ImageNet scale)
and prescribes the LayerNorm swap as "the FIRST diagnostic intervention
if participation-ratio diagnostics show partial SIGReg collapse". The
Session 5 plan operationalises that intervention as Run B; the flag is
the supported code path that makes Run B a one-flag change instead of
a code edit.

Test coverage: `tests/test_encoder.py` adds
`test_encoder_projection_can_be_layernorm` (verifies the LayerNorm path
constructs and runs forward) and
`test_encoder_projection_norm_rejects_unknown` (verifies the ValueError
for unknown values). The existing
`test_encoder_projection_is_batchnorm` was renamed to
`test_encoder_projection_is_batchnorm_by_default` and the assertion is
unchanged (the default stays BatchNorm).

Alternative considered: pipe `projection_norm` through the predictor
as well so the encoder/predictor norm types stay matched. Rejected at
this step because the Session 5 plan is explicit: "pass this through to
the encoder constructor" (only). The predictor's `out_proj` BatchNorm
is left in place; if Run B reveals a downstream distributional mismatch
between LayerNorm-encoded targets and BatchNorm-projected predictions,
that becomes a methodological observation, not a wiring bug.

### D26: --anticollapse flag on train_jepa entrypoint (2026-05-17)

`scripts/.../train_jepa.py` gains
`--anticollapse {sigreg,vicreg}`. Default `sigreg` per D5. With
`vicreg`, the JEPA wrapper is constructed with the Bardes ICLR 2022
module directly; the auto-fallback controller is still instantiated
but should never fire (PR/probe diagnostics that would have triggered
the SIGReg -> VICReg swap are silenced via the conditional that gates
the swap on the active regulariser being SIGReg). The W&B tag list
becomes `['hybrid_cnn_vit', 'vicreg']` in that case, matching the
"regularizer_name" axis defined in CLAUDE.md "Logging".

Rationale: D5 places VICReg behind the auto-fallback rule, which fires
at iter 20k AND only if PR < 0.3 * d AND probe R^2 < 0.7. The Session 5
plan needs to test VICReg as a direct configuration (Run C and Run D)
without waiting for the conjunctive condition to fire. Hard-coding the
swap into the auto-fallback controller would also work but would
conflate "intentional comparison" with "automatic intervention" in the
W&B record. A dedicated flag keeps the run intent visible.

`--tag-suffix <str>` was added in the same change. It appends
`run:<suffix>` to the W&B tag list (Session 5 uses `run_a_sigreg_bn_seed0`,
`run_b_sigreg_ln_seed0`, etc., so the analysis notebook can disaggregate
runs by tag).

Test coverage: the existing `test_train_jepa_smoke` integration test
runs with default flags and exercises the SIGReg path; no Session 5
test is added because the flag is a simple constructor switch and the
underlying VICReg module already has its own unit-test coverage from
Session 4.

Alternative considered: silently override the wrapper's anti-collapse
module post-hoc via `set_anticollapse`. Rejected because it would
require running through one iteration before the swap, and would also
leave the SIGReg state-dict keys in the run's first checkpoint, which
is a foot-gun for downstream restart logic. Direct construction-time
selection is cleaner.

### D33: Absorb two more run3 cases into v1 (2026-05-17, late)

Carlos's collaborator dropped two more run3 files in
`$PREVENT_ROOT/data/raw/periodic/run3/` later the same day as D20
(`Gust_027_x-1.965_y-0.387_s-2.0_d1.5.h5` and
`Gust_031_x-1.844_y-0.872_s-3.0_d0.5.h5`, both timestamped
2026-05-17 21:17; Gust_027 was the one skipped in D15 and now
arrives, while Gust_031 is new at the |G|=3, |Y|=0.4 corner).
Decoded with the locked alpha=14 degree rotation:

- `G-2.00_D1.50_Y+0.10` (run3, defaults to `train`)
- `G-3.00_D0.50_Y-0.40` (run3, defaults to `train`)

Both new case_ids do not collide with the existing inventory; both
stay inside the training envelope (|G| <= 3, only |G|=4 is held out in
Test C). `G-3.00_D0.50_Y-0.40` is the first run3 case at the
|Y|=0.4 corner with negative Y; together with the existing
`G-1.00_D0.50_Y+0.40` and `G+1.00_D0.50_Y-0.40` it gives the predictor
better coverage of the extreme-offset corners of the training envelope.

Same precedent as D12, D14, D15, D20: v1 still has no paper-reportable
training checkpoint, so this absorption stays in v1. The next
absorption after the first reportable v1 run MUST go to v2.

Effect on counts (cumulative since D20):
- Train cases: 37 -> 39 (+2 new run3 train cases).
- Train encounters: 126 -> 132 (+6 = 2 cases x 3 train-encounter slots).
- Test A encounters: 52 -> 54 (+2 = 2 cases x 1 held-out encounter).
- Total cases: 47 -> 49.
- Total encounters: 230 -> 238.

Cache:
- 8 new encounter files written at
  `${VORTEX_JEPA_CACHE}/v1/{G-2.00_D1.50_Y+0.10, G-3.00_D0.50_Y-0.40}/encounter_*.h5`.
- The 230 pre-existing encounter files are untouched (preprocess.py
  reported `written=8, skipped=230`).

`data_manifest/raw_cases_inventory.yaml` regenerated via
`scripts/100c_raw_cases_inventory.py`; summary now reports
`n_cases_total: 49`, `n_cases_periodic: 21`, `n_cases_run3: 28`,
`n_parse_errors: 0`, `n_duplicate_case_ids: 0`. New inventory SHA256:
`dd984588be553a28285a35fed7328cfcf9b482329e6f346b4f1e9a0574f764bc`
(D20's hash `8c7202e1c8b6d8055f5e320733cf639746999504f631a4e2551c9eaecd419282`
is preserved in git history).

`configs/splits/split_v1.json` regenerated via `python build_split_manifest.py`.
New SHA256:
`7f8f60428e13b7c2fe4063e15bd99ea9e08e5e6cecf0e8883f8fb6a4875e2331`
(D20's hash `6fa9fd149da1a0d37bb80af0a4381bf7004665bcfce3402d558a04446fe76ae0`
is preserved in git history). When logging W&B `split_sha256` for runs
that touch the absorbed v1, use the new hash.

Effect on Session 5: the 5-case smoke subset (D24) is a fixed list of
case ids and is unaffected by this absorption. The new cases will be
available for Session 6 lambda bisection and any subsequent training
run that uses the full train split.

Alternative considered: build v2 with these two cases. Rejected for the
same reason as D12/D14/D15/D20 -- premature partition-versioning while
the project still has no v1 training checkpoint to compare against.

### D27: Session 5 5k smoke outcome -- TRIVIAL-dominant with grid variation (2026-05-18)

The Session 5 5k-iter smoke produced four variants on the 5-case subset
(D24). Final state at iter 5000:

| Variant            | Anti-collapse | Proj  | PR    | r2_overall | r2_G  | r2_D  | r2_Y  | L_anti |
|--------------------|---------------|-------|-------|------------|-------|-------|-------|--------|
| A: SIGReg + BN     | SIGReg        | BN    |  1.025|  0.779     | 0.923 | 0.775 | 0.637 | 0.081  |
| B: SIGReg + LN     | SIGReg        | LN    |  1.135|  0.452     | 0.645 | 0.419 | 0.293 | 0.124  |
| C: VICReg + BN     | VICReg        | BN    | 17.463|  0.887     | 0.914 | 0.889 | 0.858 | 0.083  |
| D: VICReg + LN     | VICReg        | LN    |  7.588|  0.803     | 0.929 | 0.784 | 0.696 | 4.007  |

Classification per the Session 5 decision tree:

- A in PR <= 16 AND r2 > 0.7 -> TRIVIAL (collapse to c)
- B in PR <= 16 AND r2 <= 0.5 -> DEAD (collapsed AND uninformative)
- C in PR >  16 AND r2 > 0.7 -> a new quadrant not strictly named by
  the plan, called "TRIVIAL_LITE" in the analysis notebook (the latent
  is anti-collapsed but the encoder still leaks c into many dims so the
  probe R^2 stays in the memorisation range)
- D in PR <= 16 AND r2 > 0.7 -> TRIVIAL

Strict reading: no single one of the plan's five named outcomes
(HEALTHY / PARTIAL / TRIVIAL / WEAK / DEAD) applies cleanly because
the variants spread across three different quadrants. The notebook's
decision_string therefore prints `MIXED: quadrants [...] manual
inspection required.`

Methodological reading: **the smoke is TRIVIAL-dominant.** Three of
four variants (A, C, D) land with r2_overall > 0.7, which is the
"encoder leaks c" failure mode the plan's TRIVIAL outcome predicts.
The form of the leak varies across the grid:

- under SIGReg + BN (default), the latent collapses to rank ~1
  (PR=1.025) and z = f(c) is essentially a 1-D function of the case
  descriptor;
- under VICReg + BN, the variance hinge forces dim spread (PR=17.5)
  but the encoder fills the extra dims with c-correlated noise; the
  per-component probe (G=0.91, D=0.89, Y=0.86) is uniformly high;
- under VICReg + LN, the per-sample LayerNorm partially fights the
  per-dim variance hinge so dim spread is partial (PR=7.6) and r2
  drops modestly to 0.80;
- under SIGReg + LN, the Gaussian regulariser plus per-sample
  normalisation produces the most violent failure: the latent stays
  rank ~1 AND the probe oscillates from -0.86 to +0.86 across
  iterations, with final r2 = 0.45.

The single common feature across all four: **L_pred reaches near zero
by iter 100** (overfitting on 16 train sub-trajectories is trivial for
the predictor regardless of regularizer). With only 5 distinct c
values in the training subset, the easy thing for the encoder to learn
is c itself; nothing else is required for L_pred to reach zero.

This is H4 confirmed at the 5-case scale: the LeWM Two-Room failure
mode (arXiv:2603.19312 Section 5) replicates on physics data. The
contribution claim 3 (the regime-dependent SIGReg-PR diagnostic)
gains a concrete datapoint and a refinement: at low-intrinsic-dim
physics data scale, VICReg recovers PR but not probe-quality, and
SIGReg does neither.

What variant C tells us beyond the plan: prevention of rank-1
collapse is necessary but not sufficient. A variance-floor anti-
collapse mechanism (VICReg's per-dim hinge) achieves dim spread
without delivering a useful latent at this data scale. Confirms the
LeWM Section 5 expectation that PLDM's multi-term anti-collapse
(arXiv:2502.14819) might do better at low-intrinsic-dim regimes
because its inverse-dynamics term explicitly forces the latent to
capture *dynamics*, not just *case label*.

Decision string for the session: **TRIVIAL-DOMINANT** (TRIVIAL with
the C-quadrant variation). Triggers the same next-step as the plan's
strict TRIVIAL branch.

Next session: **Session 5.PLDM** per D29. The full PLDM 7-term loss
(arXiv:2502.14819) introduces an inverse-dynamics term that is
exactly the additional constraint the four 2-term variants here
lack. If PLDM also lands in any of {TRIVIAL, TRIVIAL_LITE, DEAD},
the failure mode is data-scale-bound and Session 5.5 (expand to
10-12 cases) follows. If PLDM lands in HEALTHY, the regime-dependent
SIGReg-vs-PLDM contrast is confirmed and Session 6 proceeds with
PLDM as the primary trained model.

Files generated this session:
- `outputs/runs/smoke5k/run_a_sigreg_bn/{metrics.jsonl, checkpoint_iter005000.pt}`
- `outputs/runs/smoke5k/run_b_sigreg_ln/{metrics.jsonl, checkpoint_iter005000.pt}`
- `outputs/runs/smoke5k/run_c_vicreg_bn/{metrics.jsonl, checkpoint_iter005000.pt}`
- `outputs/runs/smoke5k/run_d_vicreg_ln/{metrics.jsonl, checkpoint_iter005000.pt}`
- `notebooks/01_smoke_5k_analysis.ipynb` (executed; ~819 kB with embedded figures)

W&B offline runs in each variant's `wandb/offline-run-*/` subdir;
sync with `wandb sync` after `wandb login`.

### D28: Auto-fallback rule revision proposal (2026-05-18, deferred)

The Session 4 auto-fallback rule (D5) is `iter >= 20000 AND
PR < 0.3 * d AND probe_R^2 < 0.7`. The conjunctive design catches the
worst case (latent both collapsed AND uninformative). Session 5 Run A
demonstrates the alternative trivial-solution failure mode:
**PR collapsed (1.025) AND probe R^2 ABOVE 0.7 (0.779)**. The current
rule does NOT fire because r2 is above the conjunct, even though the
latent is at rank ~1.

Three rule revisions to consider before Session 6:

(a) Drop the probe_R^2 conjunct entirely:
    fire on `PR < 0.3 * d` alone, regardless of probe behaviour.
    Pros: catches the trivial-solution mode.
    Cons: false-fires on healthy runs that briefly dip in PR during
    early training (Run C had PR=4.7 at iter 250 and recovered to 17
    by iter 5000; under (a) the fallback would have fired at iter 20k
    on a similar healthy trajectory if the recovery were slower).

(b) Switch the probe to a CASE-conditional split:
    fit on K Test B cases, evaluate on the other 6-K Test B cases,
    rather than fitting and evaluating on disjoint sub-batches of all
    Test B cases. The trivial-solution mode should drop r2 sharply on
    held-out cases (because the encoder has only memorised the seen c
    values).
    Pros: directly tests the "memorisation vs generalisation"
    question that motivated the conjunct.
    Cons: more expensive (need a full forward over enough Test B
    cases to fit and evaluate); higher variance on the small Test B
    set (6 cases total).

(c) Add an "overfitting indicator" to the conjunct:
    fire on `PR < 0.3 * d AND L_pred_running < 1e-3`, where
    L_pred_running is a 1k-iter moving average. Run A's L_pred is
    below 1e-3 by iter 100; this signature is unambiguous. Pros:
    explicitly conjoint with the symptom (overfitting on small train
    set produces near-zero L_pred). Cons: tunes another threshold;
    requires running-average bookkeeping.

Decision deferred to the start of Session 6. Recommend (b) as the
most principled because it operationalises the original
"memorisation" intent of the rule; (c) as the most pragmatic if (b)
proves too costly at full training scale. (a) is the simplest but
the false-fire risk is real on slow-spreading variants like Run C.

Cite this entry from CLAUDE.md "Risk-management" when the rule is
revised.

### D29: PLDM baseline is conditional priority (2026-05-17, always-record)

The LeWM paper (Maes et al., arXiv:2603.19312, Section 5) reports:
"In the simpler Two-Room environment, PLDM and DINO-WM outperform
LeWM, which may be explained by the SIGReg regularization
encouraging a Gaussian distribution in a high-dimensional latent
space, while the intrinsic dimensionality of the environment is
much lower." Our estimated intrinsic dimension (D4: ~5 to 10) is
closer to Two-Room than to Push-T.

**Rule:** if Session 5 lands TRIVIAL (or, by the present interpretation,
TRIVIAL-dominant per D27), **PLDM becomes the priority comparator
immediately after Session 5**, before either Session 5.5 (expand
cases) or Session 6 (Hydra + lambda bisection). This is recorded
ahead of time because it changes the implicit ordering of
"baselines are parallel work" (D8) into "PLDM is conditional
priority" when the trivial-solution mode appears.

Effect on the paper: contribution claim 3 sharpens from
"SIGReg as a JEPA-for-science methodology" to "the regime-dependent
SIGReg-PR diagnostic, with PLDM as the recommended fallback for
low-intrinsic-dim domains."

Session 5 outcome triggers this rule. Next session is
**Session 5.PLDM** per `SESSION5_PLDM_BASELINE.md`. The PLDM plan
verifies the 7-term loss against arXiv:2502.14819 directly before
implementation; the D8 description (corrected in D32) is approximate
and was not re-verified against the paper at project bootstrap.

### D30: Session 5.PLDM executed; PLDM has 5 loss terms, not 7 (2026-05-18)

Session 5.PLDM was triggered by D27's TRIVIAL-dominant Session 5 outcome
and the conditional-priority rule in D29. The session executed in full:
TDD on a new `src/baselines/pldm.py`, a `src/models/pldm_wrapper.py`
that composes the existing encoder + predictor with the PLDM loss,
a `src/training/train_baseline.py` argparse entrypoint, the 5k-iter
PLDM-A run on the 5-case smoke subset, and an extension of
`notebooks/01_smoke_5k_analysis.ipynb` adding Section 7 with the
PLDM trajectories, the 5-variant quadrant table, and a PLDM-specific
decision string.

**Critical correction to D8.** D8 originally read the PLDM loss as a
"7-term VICReg-derived objective" with terms 1-7 enumerated as:
prediction, var(z), cov(z), temporal smoothness, var(dz), cov(dz),
inverse-dynamics. Direct verification of arXiv:2502.14819 (paper text
downloaded via the arxiv MCP plugin; LaTeX equations grepped from the
saved file at chars 18700-19800 and 75130-77100) shows that the paper
actually has **FIVE** terms:

```
L_JEPA = L_sim + alpha * L_var + beta * L_cov + delta * L_time_sim + omega * L_IDM
```

verbatim from Appendix D.1.1. **There are no var(dz) or cov(dz) terms
on the temporal-difference signal.** D8's "term 5" and "term 6" were
spurious. The actual loss has 4 tunable weights (alpha, beta, delta,
omega) plus L_sim with implicit weight 1; D8's "six tunable + one
fixed = 7" overcounted by two terms.

Paper-side hyperparameter values (Appendix J.2, Tables 13-17):

| Environment | alpha | beta | delta | omega |
|-------------|-------|------|-------|-------|
| Two-Rooms   |  4.0  |  6.9 |  0.75 | 0.0   |
| Diverse PointMaze | 35.0 | 12.0 | 0.1 | 5.4 |
| Ant-U-Maze  | 26.2  |  0.5 |  8.1  | 0.58  |

Default in `src/baselines/pldm.py` is all 1.0 (placeholder) with the
expectation that train_baseline.py CLI overrides set environment-
specific values. The Session 5.PLDM smoke run used all 1.0 because
none of the paper's three environments matches our regime (5-case
small-data physics) cleanly enough to justify picking a row.

**Implementation contract:** the loss takes `(z, z_hat, c)` where
`z = encoder(omega)` is the full encoded sequence ``(B, T, d)``,
`z_hat = predictor.rollout(z[:, :1, :], cond, steps=H)` is the
autoregressive rollout ``(B, H+1, d)``, and `c = (B, c_dim)` is the
static episode descriptor. The five regularisation terms are
computed on `z` (the encoder output); only `L_sim` uses `z_hat`.

**IDM adaptation:** the paper's IDM predicts a per-step action
``a_t`` from ``(z_t, z_{t+1})``. Our setting has no per-step action,
so the IDM head predicts the static episode descriptor
``c = (G, D, Y)`` from each consecutive pair, broadcast across all
(T-1) pairs per batch sample. This is the D8 adaptation, retained
unchanged through Session 5.PLDM.

**Predictor architectural note (deferred, not blocking):** the
PLDM paper uses a single-step predictor ``f(z_{t-1}, a_{t-1}) -> z_t``
(GRU for Two-Rooms, Conv for Diverse PointMaze, MLP for Ant). Our
predictor is a causal transformer with AdaLN-Zero conditioning on a
static c, used via `rollout(z[:, :1, :], cond, steps=H)`. Per the
Session 5.PLDM plan, we KEEP our transformer so the SIGReg-vs-PLDM
comparison isolates the loss; the architectural difference is the
SECOND-order ablation if Session 6 needs it.

Files landed:
- `src/baselines/__init__.py`, `src/baselines/pldm.py`,
  `src/models/pldm_wrapper.py`, `src/training/train_baseline.py`
- `tests/test_pldm_loss.py` (13 tests),
  `tests/test_pldm_wrapper.py` (5 tests). Suite now 97 passing, 1 skipped.
- `outputs/runs/smoke5k/run_pldm_a/{metrics.jsonl, checkpoint_iter005000.pt}`.
- `notebooks/01_smoke_5k_analysis.ipynb` extended with Section 7
  (PLDM loss trajectories, 5-variant 2x2, PLDM decision string).

The "7-term VICReg + 6 hyperparameter" framing in CLAUDE.md
"Baselines to implement" and in `SESSION5_PLDM_BASELINE.md`
("PLDM uses 7 terms with six loss hyperparameters") is incorrect
post-D30. CLAUDE.md is updated in this same commit; the
`SESSION5_PLDM_BASELINE.md` plan stays as a historical record (it
was written under the D8 misreading; this entry supersedes).

### D31: Session 5.PLDM outcome -- DATA_SCALE_BOUND (2026-05-18)

PLDM-A final state at iter 5000:
- PR = 5.97 (below the 16 healthy threshold; below the 9.6 fallback
  floor as well)
- r2_overall = 0.970 (highest of any variant; near-perfect c leakage)
- r2_G = 0.986, r2_D = 0.970, r2_Y = 0.953
- L_sim = 0.014, L_var = 0.510, L_cov = 0.102,
  L_time_sim = 0.002, L_idm = 0.0005

The PLDM-specific signature: **L_time_sim ~ 0 AND L_idm ~ 0
simultaneously**. The encoder produces almost-constant latents over
time (so consecutive frames differ by ~0 in L2 norm) AND the IDM head
decodes c from any (z_t, z_{t+1}) pair with negligible error. Together
these mean the encoder collapses each episode to a (case-specific
near-constant) point in latent space, and the IDM regularisation
PRESSURES this rather than preventing it -- because the IDM rewards
"c is easy to recover from any z-pair" and the easiest way to satisfy
that is precisely to make z = f(c) constant in time.

Per the Session 5.PLDM decision tree:
- REGIME_CONFIRMED would require PR > 16 AND 0.5 < r2 < 0.7. Neither holds.
- PLDM_PARTIAL would require PR > 16. Does not hold.
- DATA_SCALE_BOUND requires PR <= 16. Holds.

Final outcome: **DATA_SCALE_BOUND.** Both regularisers (2-term SIGReg,
2-term VICReg, 5-term PLDM) collapse on 5 cases / 16 train
sub-trajectories. The failure is not regulariser-specific. The IDM
term in PLDM, contrary to the LeWM Section 5 expectation that it
might break the collapse-to-c failure on low-intrinsic-dim data,
actually INTENSIFIES the leakage at this data scale (r2 = 0.970 is
the highest of any variant in the session).

Five-variant comparison (all on the same 5-case subset, seed 0,
5000 iterations, hybrid CNN+ViT encoder, AdaLN-Zero predictor):

| Variant            | Anti-collapse    | Proj | PR    | r2    | Quadrant      |
|--------------------|------------------|------|-------|-------|---------------|
| A: SIGReg + BN     | 2-term LeWM      | BN   |  1.025| 0.779 | TRIVIAL       |
| B: SIGReg + LN     | 2-term LeWM      | LN   |  1.135| 0.452 | DEAD          |
| C: VICReg + BN     | 2-term VICReg    | BN   | 17.463| 0.887 | TRIVIAL_LITE  |
| D: VICReg + LN     | 2-term VICReg    | LN   |  7.588| 0.803 | TRIVIAL       |
| PLDM-A             | 5-term VICReg+IDM| BN   |  5.966| 0.970 | TRIVIAL       |

Methodological reading: at the 5-case data scale, the encoder has 16
train sub-trajectories and 5 distinct (G, D, Y) values. The
self-supervised objective's only consistent local minimum is
``z = f(c)`` plus noise. Different regularisers produce different
*forms* of that minimum (rank-1 vs spread-but-correlated vs
spread-and-time-static) but none escape it. The hypothesis H4 (the
LeWM Two-Room failure mode replicates on physics data) is now
confirmed not just on the 2-term variants but on the 5-term PLDM
variant as well, which closes off the "maybe a multi-term loss is
enough" possibility at this data scale.

**Next session: Session 5.5.** Expand the case subset to 10-12 cases
and re-run the smoke. The PR / r2 curves vs case count will either
show a transition (small at 5, healthy at 10) or a plateau (still
trivial). The transition case suggests the encoder needs ~2x more
cases to learn anything beyond c; the plateau case suggests the
failure is more structural and motivates a different intervention
(symmetry augmentation per Open Q6, phi_t conditioning per D16
alternative, longer sub-trajectory L per the L=32-at-dt=0.05 = 1.6
t/c observation in D34, or auxiliary observable head per Open Q4 --
each is a one-knob ablation that the small-scale smoke can answer
cheaply).

PLDM-B (PLDM + LayerNorm) was deferred. Optional per the plan; given
the Session 5 Run B result (LayerNorm degraded SIGReg's probe r2
rather than recovering PR), running PLDM-B was unlikely to change the
DATA_SCALE_BOUND conclusion. The decision can be revisited in
Session 5.5 if the case-count expansion produces ambiguous PLDM
behaviour.

### D32: Correction to PLDM citation in D8 (2026-05-17, housekeeping)

D8 in HANDOFF.md cited PLDM as "Sobal, Jyothir, Jalagam, Carion, Cho,
LeCun (2022), arXiv:2211.10831" with the title "Joint Embedding
Predictive Architectures Focus on Slow Features". This citation is
INCORRECT. The 2022 paper is a 4-page NeurIPS SSL workshop precursor by
a partially overlapping author group; it is useful as theoretical
background but is NOT the source of the PLDM name or the multi-term
loss formulation. The actual PLDM paper is:

Sobal, Zhang, Cho, Balestriero, Rudner, LeCun, "Learning from
Reward-Free Offline Data: A Case for Planning with Latent Dynamics
Models", arXiv:2502.14819, February 2025. Project page:
latent-planning.github.io. Code: github.com/vladisai/PLDM.

Effect on the repo:
- D8 in HANDOFF.md updated to cite arXiv:2502.14819 as the primary
  reference, with arXiv:2211.10831 listed separately as the workshop
  precursor for theoretical background. Header marked
  "(citation corrected 2026-05-17, see D32)" so a reader of D8 sees the
  forward pointer immediately.
- HANDOFF.md "Key references" / "Direct baselines" section updated to
  list arXiv:2502.14819 as PLDM, with arXiv:2211.10831 as the workshop
  precursor.
- CLAUDE.md "Baselines to implement" item 4 updated to cite
  arXiv:2502.14819 as the primary reference, with arXiv:2211.10831 as
  workshop precursor and the Robot Learning Workshop 2025 paper as the
  stress-testing follow-up.

The "7-term loss" language in D8 is approximate; the actual term count
and weight set are to be read directly from arXiv:2502.14819 Appendix
C.1.1 and the official code at github.com/vladisai/PLDM, and the D8
description updated to match once verified. That verification is part
of Session 5.PLDM (if triggered), not this housekeeping pass.

Alternative considered: leave D8 unchanged and merely add a note that
the citation is wrong. Rejected because the wrong citation has already
propagated into CLAUDE.md and into the SESSION5_*.md plans; surgically
fixing all three at once is the lowest-risk way to keep the project's
references coherent before Session 5's variant runs land.

### D34: Frame-skip "default 2" was never implemented; pipeline is at skip 1 (2026-05-18, housekeeping)

The earlier "Open questions" item 2 read "Frame-skip. Default is 2,
giving 60 effective frames per encounter at `dt_eff = 0.1`. Verify
against impact dynamics resolution. Frame-skip 1 (no skipping) is
also viable on the 96 GB GPU." Carlos asked on 2026-05-18 to verify
the smoke results under frame-skip 1 before deciding on next steps;
direct inspection of the pipeline shows the project has ALWAYS been
at frame-skip 1 in practice. The "default is 2" wording was an
unimplemented intention that propagated through CLAUDE.md, HANDOFF
D21, and the collaborator report without ever matching the code.

Evidence chain (all verified 2026-05-18):

- Raw DNS: `/forces/time` for `Baseline.h5` reports time stride
  `dt = 0.05000` (first 5 entries
  `[0.00025, 0.05025, 0.10025, 0.15025, 0.20025]`). `/u` shape is
  `(800, 192, 96, 32, 3)` for periodic and `(480, ...)` for run3.
- Preprocessing config (`configs/preprocessing.yaml`):
  `encounter.frames_per_encounter = 120`, `encounter.dt_tc = 0.05`.
- Preprocessing code (`scripts/preprocess.py:extract_encounter`):
  reads `raw[curl_path][f0:f1, :, :, mid, omega_z_idx]` with
  `f0 = k * 120, f1 = (k + 1) * 120`. Python slice with default
  stride 1; no decimation.
- Dataset loader (`src/data/episode_dataset.py:__getitem__`):
  reads `g["omega_z"][start:end]` with `end - start = subtraj_len`.
  Python slice with default stride 1; no decimation.
- Encoder forward (`src/models/encoder.py:HybridCNNViTEncoder.forward`):
  flattens `(B, T, ...)` into `B*T` per-frame inputs through the CNN
  and ViT; no temporal subsampling.

So `dt_eff = dt_tc = 0.05`, every encounter contributes 120 frames to
the cache, and every sub-trajectory has L = 32 frames spanning 1.6 t/c.

Implication for the existing smoke results: **all five Session 5 and
Session 5.PLDM smoke runs (A, B, C, D, PLDM-A) were already under
frame-skip 1 conditions.** The TRIVIAL-dominant outcome (D27) and the
DATA_SCALE_BOUND outcome (D31) are not amenable to a "what if we used
all the frames" intervention because we already use all the frames.

Effect on the docs (this commit):

- HANDOFF "Open questions" item 2 rewritten as "Resolved (D34)",
  reframed around the actual remaining question on the temporal axis
  (sub-trajectory length L rather than skip stride).
- HANDOFF D21 paragraph on the `H_roll = 8` vs `H_roll = 2` rationale
  updated from "40 to 160 effective frames at `dt_eff = 0.1`" to
  "100 to 400 effective frames at `dt_eff = 0.05`". The decision
  itself stands; the numerical context is fixed.
- HANDOFF D27 "frame-skip sweep" intervention removed from the list
  of structural-failure mitigations (it is the default already);
  replaced by "longer sub-trajectory L per D34".
- `COLLABORATOR_REPORT_2026-05-18.md`: same three corrections.

CLAUDE.md was checked and contains no frame-skip wording in either
the locked-decisions or operational-guide sections, so no edit there.

Alternative considered: keep the "default is 2" wording and implement
frame-skip 2 retroactively to match. Rejected because (a) the existing
smoke results are valuable data that should not be invalidated by a
post-hoc convention change; (b) frame-skip 1 is the correct default
for impact-dynamics resolution at this Re; the "default is 2" wording
appears to have been a typo or carry-over from an earlier project
sketch and was never anchored to a design decision.

The actually-open lever on the temporal axis is sub-trajectory length
L. Currently L = 32 = 1.6 t/c, capturing roughly 8 to 32 percent of
the 5 to 20 t/c impact-dynamics span. Raising L (e.g., to 64 = 3.2 t/c
or 120 = 6 t/c = full encounter) is a one-knob ablation that Session
5.5 or Session 6 may run if the data-scale-bound diagnosis from D31
turns out to need additional levers.

### D35: Absorbed two more run3 cases into v1.2 (2026-05-18, Session 6 Step 0)

Carlos's collaborator dropped two further run3 files into
`$PREVENT_ROOT/data/raw/periodic/run3/` in the interval between the
Session 5.PLDM report and the Session 6 launch
(`Gust_032_x-1.844_y-0.872_s-1.5_d1.5.h5` and
`Gust_033_x-1.844_y-0.872_s3.0_d0.5.h5`). Decoded with the locked
alpha=14 degree rotation:

- `G-1.50_D1.50_Y-0.40` (run3, defaults to `train`)
- `G+3.00_D0.50_Y-0.40` (run3, defaults to `train`)

Both case_ids do not collide with the existing inventory; both stay
inside the training envelope (|G| <= 3, only |G|=4 is held out in
Test C). `G-1.50_D1.50_Y-0.40` is a new run3 case at the largest D
with moderate negative G; `G+3.00_D0.50_Y-0.40` is the first run3
case at the largest |G|=3 with the most-negative Y on the DoE-2 grid.
Together they add corner coverage to the train envelope at the highest
G and Y extremes.

Same precedent as D12, D14, D15, D20, D33: v1 still has no paper-
reportable training checkpoint, so this absorption stays in v1. Called
"v1.2" in session reports to distinguish from the D33 absorption ("v1.1")
and the original ("v1.0"); the on-disk cache directory remains
`${VORTEX_JEPA_CACHE}/v1/` because the binary format is unchanged. The
next absorption after the first reportable v1 run MUST go to v2.

Effect on counts (cumulative since D33):
- Train cases: 39 -> 41 (+2 new run3 train cases).
- Train encounters: 132 -> 138 (+6 = 2 cases x 3 train-encounter slots).
- Test A encounters: 54 -> 56 (+2 = 2 cases x 1 held-out encounter).
- Total cases: 49 -> 51.
- Total encounters in splits: 238 -> 246.

Cache:
- 8 new encounter files written at
  `${VORTEX_JEPA_CACHE}/v1/{G-1.50_D1.50_Y-0.40, G+3.00_D0.50_Y-0.40}/encounter_*.h5`.
- The 238 pre-existing encounter files are untouched (preprocess.py
  reported `written=8, skipped=0` because the new case_ids did not
  exist in the cache, but the existing files were not re-run; total
  cache after = 246 encounter files across 51 case directories).

`data_manifest/raw_cases_inventory.yaml` regenerated via
`scripts/100c_raw_cases_inventory.py`; summary now reports
`n_cases_total: 51`, `n_cases_periodic: 21`, `n_cases_run3: 30`,
`n_parse_errors: 0`, `n_duplicate_case_ids: 0`. New inventory SHA256:
`ce817e1e0df54309...` (full hash in
`configs/splits/split_v1.json` -> `source_inventory.sha256`; D33's hash
`dd984588be553a28...` is preserved in git history).

`configs/splits/split_v1.json` regenerated via
`python build_split_manifest.py`. New manifest SHA256:
`a721dc92f6e278ee054bb952933c14ba20a58137f79f3a19fc6ad71b70a007dd`
(D33's hash `7f8f60428e13b7c2fe4063e15bd99ea9e08e5e6cecf0e8883f8fb6a4875e2331`
is preserved in git history). When logging W&B `split_sha256` for runs
that touch the absorbed v1.2 partition, use the new hash.

Effect on Session 6: the 5-case smoke subset (D24) is a fixed list and
is unaffected. The F-S (24-case) scale-up run is built from the train
split and may include or exclude the new cases at the agent's discretion
when authoring `configs/cases/smoke_24cases.yaml`; the default for this
session is to exclude them so F-S exactly tests the data-scale axis
against the same physical pool that Session 5 sampled from.

Alternative considered: build v2 with these two cases. Rejected for the
same reason as D12/D14/D15/D20/D33 -- premature partition-versioning
while the project still has no v1 training checkpoint to compare
against.

Renumbering note: SESSION6_FACTORIAL_DIAGNOSTIC.md drafted this entry
as "D33" because the plan was written before D33 and D34 were assigned
(D33 = first run3 absorption, 2026-05-17; D34 = frame-skip housekeeping,
2026-05-18). The session's other planned decisions therefore become
D36 (CL is the canonical observable), D37 (eta = 0.01 observable head
weight), D38 (five factorial axes), D39 (decision string, conditional
on outcome).

### D36: CL(t+Delta) is the canonical dynamic observable target (2026-05-18, Session 6)

Replaces "time-to-impact" and "vortex centroid" (the collaborator
report's two stand-in candidates) with the lift coefficient CL evaluated
at future frames as the single dynamic observable used by Session 6
(F-OBS variant) and by any future observable-augmented design.

Rationale
- CL is the aerodynamically meaningful quantity that ultimately
  controls the digital-twin objective of the project. Time-to-impact
  is a per-encounter scalar (no dynamics signal once the impact frame
  is past), and the vortex centroid is a geometric proxy that does not
  see the airfoil's actual response.
- Aligns with Fukami and Taira's lift-augmented autoencoder
  (Nat Commun 14, 6480, 2023; arXiv:2305.18394), where the encoder
  produces both a low-dimensional embedding and a predicted CL
  trajectory, and the auxiliary CL loss demonstrably reduces the
  intrinsic dimension of the discovered manifold.
- Aligns with Solera-Rico, Sanmiguel Vila, et al. (Nat. Commun. 15,
  1361, 2024; arXiv:2304.03571) where the transformer head conditions
  on aerodynamic observables and the latent is evaluated against
  surface-pressure-derived quantities.
- Aligns with Fukami et al. transonic-buffet extension (J. Fluid Mech.
  1021, A39, 2025; arXiv:2509.17306) which showed that observable
  augmentation reduces the intrinsic dim from about 10 to 3 on a
  structured fluid problem.

Implementation (Session 6 Step 1)
- The data loader (`EpisodeDataset`) emits a per-sample `cl_future`
  tensor of shape `(L, len(deltas))`. The default deltas are `(8, 16, 24)`
  frames; at `dt_eff = 0.05` these correspond to convective times
  `0.4 / 0.8 / 1.2 t/c` into the future, covering short, medium, and
  long observable horizons relative to the 5 to 20 t/c impact-dynamics
  span.
- End-of-encounter clamping: when `frame_start + i + delta` exceeds
  the encounter's last valid frame index, the value is clamped to the
  last valid `C_L`. The clamped post-impact relaxation regime is
  approximately stationary, so the bias is small and the alternative
  (dropping frames near the end) is harder to plumb through the rest
  of the training loop. Documented in
  `tests/test_episode_dataset.py::test_cl_future_clamps_at_encounter_end`.
- Backwards compatible: existing training scripts that did not request
  CL continue to work because the dataset's `emit_cl_future` flag
  defaults to `False`.

This decision does not commit us to dropping any other observable in
the future. If wall pressure or vortex circulation turns out to be a
stronger anti-shortcut signal in Session 7, those can be added as
additional outputs of an extended head. CL is the canonical first
target because of the direct lineage to the Fukami / Solera-Rico
literature and because it is the project's eventual digital-twin
quantity of interest.

### D37: Observable head added as auxiliary loss with weight eta = 0.01 (2026-05-18, Session 6)

The F-OBS variant pairs the encoder with a small MLP head that maps
each per-frame latent `z_t` to a vector of future CL values, and adds
`eta * L_obs` to the JEPA loss where `eta = 0.01`.

Rationale
- The JEPA self-supervised objective is preserved as the primary
  signal. With `eta = 0.01` and Run A's pre-tested loss magnitudes
  (L_pred near 0.05, L_anticollapse near 0.1), the observable term
  contributes about a percent of the total loss at convergence -- the
  head is a weak guidance signal, not a primary supervision target.
- Inherits the lineage from Fukami and Taira (JFM 2023, "Compact
  Representation of Transonic Airfoil Buffet Flows with Observable-
  Augmented Machine Learning") where the equivalent auxiliary weight
  on the CL prediction loss is a small positive constant.
- Implementation (`src/models/observable_head.py` and the
  `observable_head=` argument to `JEPA`): a two-layer MLP
  `Linear(d=32, hidden=64) -> GELU -> Linear(hidden, n_deltas=3)`.

The plan reserves this number for Session 7 sweeping if F-OBS lands as
the active axis. The Session 6 F-OBS run is therefore a single
operating point on a future eta curve, not a tuning result.

The observable head's parameters share the predictor learning rate
group in the optimizer; the encoder LR group is unchanged. The head is
included in the checkpointed `jepa_state_dict` so it can be re-loaded
in the Session 7 sweep without retraining.

### D38: Five factorial single-axis variants for Session 6 (2026-05-18)

Each of the five Session 6 F-* variants changes exactly one axis from
the Session 5 Run A baseline (SIGReg + BatchNorm + L=32 + c-at-
predictor + no observable). Sessions are constrained to a single axis
per variant so the diagnostic notebook can attribute the recovery (or
non-recovery) of the within-case dynamic latent signal to a specific
mechanism. Combinations of axes are deferred to Session 7 to keep
Session 6's budget at five 5k-iter runs (about 2.5 to 3 hours of GPU).

Variants and their published precedents:

- F-L (sub-trajectory length 64): V-JEPA 2 trains on 64-frame windows
  at 4 fps (Assran et al. arXiv:2506.09985); the hypothesis is that
  L = 32 = 1.6 t/c is too short relative to the 5 to 20 t/c impact-
  dynamics span and that the encoder can use a static case-axis
  shortcut when the temporal window does not span enough of impact.
- F-CD (per-batch c-dropout 0.5): inspired by classifier-free guidance
  in diffusion (Ho and Salimans, arXiv:2207.12598). The hypothesis is
  that, when the predictor can rely on c being present, the encoder
  is incentivised to encode less information than it could.
- F-NC (predictor cond_dim=0): the most diagnostic single change. If
  c never reaches the predictor, the encoder MUST encode whatever
  c-dependent dynamics the predictor needs in z itself. Matches the
  Brain-JEPA / Echo-JEPA pattern where the encoder is fully
  responsible for encoding subject-level information.
- F-S (24 cases): standard data-scale ablation. With 24 distinct c
  values to memorise, the case-axis shortcut becomes less attractive
  than learning physics.
- F-OBS (observable head, eta = 0.01): Fukami / Solera-Rico precedent.
  Weak observable guidance breaks the case-memorization shortcut
  without overwhelming the self-supervised objective.

All five share Session 5 Run A's defaults for everything except the
single axis being tested: SIGReg with M = 256 projections; BatchNorm
projection at the encoder latent boundary (D17); seed 0; B = 16; 5000
iterations; diagnostic cadence every 250 iters; checkpoint cadence
every 1000 iters; W&B group `partition_v1` (no separate group per
session, to keep all v1 runs co-located in the W&B project for cross-
session comparison).

The four 5-case variants share the smoke 5-case subset from
`configs/cases/smoke_5cases.yaml` (D24). F-S uses the new 24-case
subset from `configs/cases/smoke_24cases.yaml`, which is a superset of
the smoke 5 plus 19 additional cases spanning 12 G levels, all 3 D
levels, and 7 Y values. The 24-case set deliberately excludes the two
D35-absorbed cases so the F-S contrast is purely "more cases from the
Session 5 physical pool" rather than "more cases plus new corner
coverage."

### D39: Session 6 outcome -- COMBINED_REMEDIATION with PLDM as the recommended base (2026-05-18)

Final audit on the 5-case Test A subset, run by
`notebooks/03_factorial_analysis.ipynb` against the iter-5000 checkpoints
(plus F-OBS @ 10k from the resume extension):

| Variant                       | PR_all | PR_within | r2(z->c) | r2(z_dyn->c) | r2(z_dyn->phase) | r2(CL_future) | classify()                                |
|-------------------------------|-------:|----------:|---------:|-------------:|-----------------:|--------------:|-------------------------------------------|
| Run A (SIGReg + BN baseline)  |  1.02  |   2.25    |   0.73   |   -0.17      |     0.13         |    -0.02      | baseline                                  |
| PLDM-A (Session 5.PLDM)       |  6.72  |   4.01    |   0.97   |   -0.09      |     0.58         |    0.96       | **active** (PR_within>4, phase>0.5, CL>baseline) |
| F-L (SIGReg, L=64)            |  1.01  |   3.25    |   0.83   |   -0.14      |     0.11         |   -0.04       | inactive                                  |
| F-CD (SIGReg, c-dropout=0.5)  |  1.03  |   2.72    |   0.55   |   -0.15      |     0.15         |   -0.02       | inactive                                  |
| F-NC (SIGReg, cond_dim=0)     |  1.02  |   5.86    |   0.38   |   -0.13      |     0.16         |   -0.02       | partially_active (PR_within>4 only)       |
| F-S  (SIGReg, 24 cases)       |  1.03  |   1.48    |   0.46   |   -0.08      |     0.10         |   -0.02       | regressed (PR_within < baseline)          |
| F-OBS (SIGReg + obs eta=0.01) |  3.21  |   3.53    |   0.99   |   -0.10      |     0.47         |    0.95       | partially_active (CL>baseline only)       |
| F-OBS @ 10k (resume)          |  3.41  |   3.59    |   0.99   |   -0.12      |     0.45         |    0.95       | partially_active (CL>baseline only)       |
| **PLDM+OBS (PLDM + obs eta=0.01)** |  6.09  |   4.77    |   0.97   |   -0.13      |     0.54         |    0.96       | **active** (PR_within>4, phase>0.5, CL>baseline) |

Baseline for the CL-prediction metric:
`baseline_ct(c, t) -> CL(t + delta) -> r2 = 0.902`. Any variant with
`r2(CL_future) > 0.90` is using the latent for something a (c, t)
lookup cannot do; r2 below that means the latent is at best a fancy
case-frame lookup. Per the notebook, only the four observable-coupled
or PLDM rows beat the baseline (PLDM-A, PLDM+OBS, F-OBS, F-OBS @ 10k);
the four pure-SIGReg axes (Run A, F-L, F-CD, F-NC, F-S) all score
below zero (worse than predicting the mean CL).

Decision string per the canonical_for_axis logic in
`notebooks/03_factorial_analysis.ipynb` Section 5: **COMBINED_REMEDIATION**.
Partial axes: F-NC (PR_within>4) and F-OBS (CL>baseline). No JEPA axis
is fully active. Strict reading: Session 7 should run factorial
combinations of F-NC and F-OBS at 5k iters and check whether the
combination clears the active bar.

Substantive read (broader than the strict decision tree): the audit
shows **PLDM-A is *already* an active configuration by the same bar
applied to the JEPA axes**, contrary to the Session 5 D31 reading of
"DATA_SCALE_BOUND". The D31 reading was based on a coarser PR-only
diagnostic; once the static-vs-dynamic decomposition and the CL-future
probe are added, PLDM-A clears all three "active" checks (PR_within=4.01,
r2_dyn_phase=0.58, r2(CL_future)=0.96). PLDM+OBS slightly improves
on PLDM-A across PR_within (4.01 -> 4.77) but leaves the other two
metrics roughly unchanged. The observable head's *bigger* impact is
rescuing SIGReg JEPA from TRIVIAL (Run A's r2(CL_future)=-0.02 -> F-OBS's 0.95):

- For SIGReg: OBS is a *necessary* rescue from TRIVIAL.
- For PLDM:   OBS is a *marginal* improvement on an already-active config.

This regulariser asymmetry was not in the Session 6 plan; it is the
single most important finding of the session.

Session 7 plan (revised from the strict COMBINED_REMEDIATION reading
to reflect the substantive read):

1. **Session 7-PLDM-DEEP**: confirm PLDM-A is active at higher iters
   and on more cases. Train PLDM-A for 20k iters on the full 41-train-
   case partition; verify PR_within, r2_dyn_phase, r2(CL_future) all
   stay or improve. Estimated 4 hours.
2. **Session 7-OBS-PLDM**: sweep `eta` in {0, 0.001, 0.005, 0.01, 0.05}
   for PLDM at 20k iters on full data, with `eta = 0` as the explicit
   "PLDM alone" anchor. Pick the operating point that maximises
   r2(CL_future) on Test A. Estimated 8 hours.
3. **Session 7-COMB-NC-OBS (optional)**: the strict COMBINED_REMEDIATION
   path. Combine F-NC + F-OBS on SIGReg JEPA at 5k iters as a control
   that the JEPA-side combination does not unexpectedly outperform
   PLDM. Cheap (~2 hours of GPU on cuda:3 in parallel with Session 7-OBS-PLDM).

Paper framing (updated): "PLDM with the 5-term VICReg-derived
objective already produces a non-trivial latent on low-intrinsic-dim
fluid data at the 5-case smoke scale; SIGReg JEPA does not. An
auxiliary CL observable head rescues SIGReg from TRIVIAL and
marginally improves PLDM. The 'observable augmentation' literature
(Fukami JFM 2023/2024/2025, Solera-Rico Nat Commun 2024) is therefore
necessary for the weaker regulariser, not the stronger one."

Out-of-plan extensions recorded here for the audit trail:

- F-OBS @ 10k: resume of F-OBS using a new `--resume-from` flag on
  `train_jepa.py` (committed during the session) to continue from the
  iter-5000 checkpoint to iter 10000. Result: PR drift +0.7 over the
  extra 5000 iters, confirming the F-OBS plateau is not iter-budget-
  limited. Cost: ~35 min of GPU on cuda:3.
- PLDM+OBS: observable head wired into `PLDMWrapper` and exposed as
  the same three CLI flags in `train_baseline.py` that exist on
  `train_jepa.py`. Result: PR=12 at iter 4750, only slightly above
  PLDM-A's PR=6.7 at iter 5000 (and the audit shows PLDM+OBS's
  *static-vs-dynamic* metrics are essentially unchanged from PLDM-A).
  Cost: ~30 min of GPU on cuda:2 plus ~20 min of wiring
  (`src/models/pldm_wrapper.py`, `src/training/train_baseline.py`,
  `tests/test_pldm_wrapper.py` +2 tests).

Both extensions were proposed mid-session and approved by the user
because they were high-value-low-cost (cuda:3 was idle after the F-OBS
chain finished; the PLDM+OBS extension parallelizes with F-OBS-10k
across the two RTX 6000 cards). The PLDM+OBS extension changed the
session's substantive conclusion (without it, "OBS rescues SIGReg" is
all the evidence; with it, "OBS marginally helps PLDM" is added and
the regulariser asymmetry becomes the headline).

F-NC PR_within = 5.86 caveat: F-NC's apparent partial activity is a
weak partial. PR_within is high but the dynamic part has no phase
signal (r2_dyn_phase=0.16) and no CL prediction signal (r2(CL_future)=-0.02).
The high PR_within is likely a numerical artifact of cond_dim=0
collapsing the conditioning channel: the predictor has less structure
to lock onto, the encoder produces a slightly noisier (more
high-rank-looking) latent, but the rank does not correspond to useful
structure. Session 7 should not over-index on this signal.

The hardware finding from D38 is also recorded here: the workstation
exposes two RTX 6000 Blackwell cards, not one as CLAUDE.md "Hardware"
states. Session 6 used both via `CUDA_VISIBLE_DEVICES=3` on the
second-card chain. The Session 6 wall clock was ~2.5 hours instead of
~5+ that single-card execution would have required. CLAUDE.md should
be updated to acknowledge the second card and document the
single-card-isolation pattern; this is housekeeping deferred to a
follow-up commit so the Session 6 branch lands the substantive findings
without scope creep.

### D40: Two RTX 6000 cards are canonical hardware; `--gpu {0,1}` flag (2026-05-18, post-Session-6 housekeeping)

Promotes the Session 6 D39-audit-trail finding ("the workstation has
two RTX 6000s, not one") to a standalone decision and lands the code
support so future sessions don't have to use shell-level
`CUDA_VISIBLE_DEVICES` tricks to pick between the two cards.

Concrete changes (this commit):

- `src/utils/device.py` gains `find_rtx6000_indices() -> list[int]` (all
  visible RTX 6000 torch indices) and `require_rtx6000(gpu_index=None)`
  where `gpu_index` is a 0-indexed selector into the RTX 6000 subset
  (not into torch's full CUDA enumeration; the two L40S cards do not
  consume `gpu_index` slots). Default `gpu_index=None` picks the first
  RTX 6000, preserving pre-D40 single-card behaviour.
- `src/training/train_jepa.py` and `src/training/train_baseline.py` both
  accept `--gpu N`. Threaded into `require_rtx6000(gpu_index=args.gpu)`.
  W&B `run_config["gpu_name"]` still records the device name; runs
  distinguish themselves by `--tag-suffix` and the device index in the
  config.
- `tests/test_device.py` +3 tests: `find_rtx6000_indices()` returns
  multiple indices on a 4-GPU mock, `gpu_index` out-of-range raises
  `NoRTX6000Error` with a clear message, negative `gpu_index` is
  rejected, and a workstation-only test that confirms `gpu_index=0` and
  `gpu_index=1` resolve to distinct torch indices when two RTX 6000s
  are visible.
- `CLAUDE.md` "Hardware" section: rewritten to acknowledge two RTX
  6000s, document the `--gpu {0,1}` pattern, and explicitly deprecate
  shell-level `CUDA_VISIBLE_DEVICES` selection between the two cards.

Backwards compatibility: every existing training command that omits
`--gpu` still picks the first RTX 6000. The Session 6
`scripts/run_session6_cuda3_parallel.sh` (which uses
`CUDA_VISIBLE_DEVICES=3`) is preserved as a historical artifact; new
scripts (Session 7 onward) should use `--gpu` instead.

Numbering note: this entry was originally referenced as "D38" in early
SESSION7 plan drafts. Since D38 was already assigned to "five factorial
single-axis variants", the hardware finding landed here as D40 instead.
The Session 7 plan's three "D38" references should be updated to
"D40" (done in the same commit). D39's last paragraph (the audit-trail
mention of the hardware finding) is preserved for the in-context
Session 6 history; this D40 entry adds the code-level changes.

Test coverage: 119/119 pass on the fast suite (116 prior + 3 new
device tests).

1. Empirical impact frame. The estimate of 40 was validated in the bootstrap session
   on the cached partition v1: vorticity-domain argmax mean = 40.8, force-domain
   argmax mean = 38.8 (both over the [25, 55] window). The distribution is bimodal in
   the vorticity domain (strong gusts peak pre-impact, weak gusts post-impact) and
   tighter in the force domain. The configs/splits/split_v1.json estimate of 40 is retained.
   Resolved.

2. Frame-skip. Resolved (D34, 2026-05-18). The default in the pipeline as actually
   implemented is frame-skip 1 (no skipping): raw DNS dt = 0.05 t/c, cache stores
   120 consecutive raw frames per encounter, dataset loads 32 consecutive cache
   frames per sub-trajectory. `dt_eff = 0.05`, sub-trajectory length = 1.6 t/c.
   The earlier wording ("default is 2, giving 60 effective frames at dt_eff = 0.1")
   described an unimplemented intention that was never coded. All Session 4 / 5 /
   5.PLDM smoke results are at frame-skip 1. The actual remaining question is the
   sub-trajectory LENGTH `L` (currently 32 = 1.6 t/c) vs the impact-dynamics span
   (5 to 20 t/c); raising `L` would capture more of impact at the same dt_eff.

3. Lambda bisection budget. Six evaluations over [0.001, 1.0]. If the optimum is near
   LeWM's default 0.1, stop the bisection early and log this as a robustness result.

4. Auxiliary observable head. Should the JEPA optionally produce wall pressure or C_L
   as a side prediction? Default is no (per LeWM). Reserve as an ablation only; if it
   substantially helps probe R^2, it is reportable as a hybrid contribution.

5. C-JEPA-style gust masking ablation. Requires defining the "gust object" region per
   episode. The vortex centroid is computable analytically from launch position plus
   U_inf * t. A circular mask of radius D around the centroid would zero out the gust
   in selected frames. Optional ablation; only run if the main results are promising.

6. Symmetry augmentation. The flow has approximate Y -> -Y reflection symmetry combined
   with G -> -G and omega_z -> -omega_z. Adding this as a paired augmentation roughly
   doubles the effective training data. Implement but ablate to verify it does not
   destabilize SIGReg.

(D11 closes the prior off-by-one item for impact_aware_start_range.)

## Suggested next steps (ordered)

1. (Done, 2026-05-15) Data loader at `src/data/episode_dataset.py`. Verified across
   all four splits; impact-aware fraction 0.814 vs predicted 0.811; reproducible with
   seed. See SESSION_REPORT_2026-05-15.md.

2. (Done, 2026-05-16, Session 2) Model primitives: SIGReg, AdaLN-Zero, RoPE under
   `src/models/`. 15 unit tests green (six SIGReg distribution/gradient/dtype,
   four AdaLN-Zero identity/broadcast/gradient, five RoPE identity/offset/cache).
   See SESSION_REPORT_2026-05-16.md and D13 (SIGReg LeWM-faithful, no `*N` multiplier).

3. (Done, 2026-05-16, Session 3) Encoder and predictor under `src/models/`. Hybrid
   CNN stem (3M params) + 6-layer ViT (7M params) -> d=32 latent via BatchNorm-projected
   [CLS] head (D17). AdaLN-Zero-conditioned 6-layer autoregressive predictor with RoPE
   on Q and K only, causal mask, BatchNorm output projection. Encoder + predictor unit
   tests bring the suite to 31 green.

4. (Done, 2026-05-17, Session 4) JEPA wrapper, VICReg fallback, scheduled-sampling
   utility, diagnostics, auto-fallback controller, RTX 6000 device helper, and a
   minimal argparse training entrypoint (`src/training/train_jepa.py`). 200-iter smoke
   on three cases (Baseline, G+1.00_D0.50_Y+0.10, G-1.00_D1.00_Y-0.20) ran end-to-end
   on the RTX 6000 Blackwell in roughly 30 seconds, with all four required and seven
   paper-grade W&B keys logged and one checkpoint written. New tests bring the suite
   to 71 green plus 1 slow integration test that runs under `pytest --runslow`. See
   D21 (V-JEPA 2-AC-faithful scheduled sampling), D22 (VICReg coefficients with the
   invariance term dropped), and D23 (slow-test opt-in pattern).

5. Meaningful 5k-iter smoke run on 5 cases (Session 5). Pass criteria from the
   original next-steps entry, now repeated here for clarity: SIGReg loss below 5.0 at
   iter 5000, participation ratio above 0.5 * d, probe R^2 for c above 0.5 on Test B.
   This is the run that tests whether the JEPA *learns anything useful*; Session 4
   only verified that the training loop runs cleanly. Session 5 also introduces Hydra
   configs and enables `torch.compile()` on the JEPA wrapper.

6. Lambda bisection at full data: six evaluations of 24k iterations each. Pick the
   lambda maximizing Test A probe R^2.

7. Full training of the chosen lambda for 80k iterations. Train the visualization
   decoder on the frozen encoder. Run the full Section-7 evaluation suite.

8. Baselines in parallel: PLDM, Fukami AE, Solera-Rico beta-VAE, POD on the same split
   with the same evaluation metrics.

9. Ablation matrix (the 15 ablations from the architecture spec). Mandatory: ablations
   1 (d sweep), 2 (SIGReg vs VICReg vs none), 7 (teacher forcing vs scheduled sampling
   vs full rollout), 10 (Solera-Rico baseline), 11 (Fukami AE baseline), plus the new
   PLDM baseline.

10. Paper writing.

## Key references

Direct architectural template
- LeWM: Maes, Le Lidec, Scieur, LeCun, Balestriero. "LeWorldModel: Stable End-to-End
  Joint-Embedding Predictive Architecture from Pixels." arXiv:2603.19312, March 2026.

Anti-collapse theory
- LeJEPA / SIGReg: Balestriero and LeCun. "LeJEPA: Provable and Scalable Self-Supervised
  Learning Without the Heuristics." arXiv:2511.08544, November 2025.
- VICReg: Bardes, Ponce, LeCun. ICLR 2022.

Direct baselines
- PLDM: Sobal, Zhang, Cho, Balestriero, Rudner, LeCun. "Learning from Reward-Free
  Offline Data: A Case for Planning with Latent Dynamics Models." arXiv:2502.14819,
  February 2025. Project page: latent-planning.github.io. Code: github.com/vladisai/PLDM.
- PLDM workshop precursor: Sobal, Jyothir, Jalagam, Carion, Cho, LeCun. "Joint Embedding
  Predictive Architectures Focus on Slow Features." arXiv:2211.10831, NeurIPS SSL
  workshop 2022. (D8 originally cited this as PLDM; corrected in D32.)
- PLDM (stress-tested): Sobal, Zhang, Cho, Balestriero, Rudner, LeCun. "Stress-testing
  Offline Reward-Free Reinforcement Learning." Robot Learning Workshop 2025.
- Solera-Rico, Sanmiguel Vila, Gomez-Lopez, Wang, Almashjary, Dawson, Vinuesa.
  "beta-Variational Autoencoders and Transformers for Reduced-Order Modelling of Fluid
  Flows." Nat. Commun. 15, 1361, 2024.
- Fukami, Iwatani, Maejima, Asada, Kawai. "Compact Representation of Transonic Airfoil
  Buffet Flows with Observable-Augmented Machine Learning." J. Fluid Mech. 1021, A39,
  2025 (arXiv:2509.17306).
- Fukami, Smith, Taira. "Extreme Vortex-Gust Airfoil Interactions at Reynolds Number
  5000." Phys. Rev. Fluids 10, 084703, 2025.

Related JEPA work
- V-JEPA 2 / V-JEPA 2-AC: Assran et al. arXiv:2506.09985, 2025. Multi-step training
  recipe with scheduled sampling.
- C-JEPA: Nam, Le Lidec, Maes, LeCun, Balestriero. arXiv:2602.11389, February 2026.
  Object-centric masking.
- AeroJEPA: Vinuesa group preprint, 2026. Direct competitor at the JEPA-for-aerodynamics
  framing. Retrieve PDF when embargo lifts.

Latent dynamics on manifolds
- Constante-Amores and Graham. "Data-Driven State-Space and Koopman Operator Models of
  Coherent State Dynamics on Invariant Manifolds." J. Fluid Mech. 984, R9, 2024
  (arXiv:2312.03875).

## Warnings and pitfalls

- SIGReg requires BatchNorm projection at the encoder bottleneck. Do NOT use LayerNorm
  at the latent boundary. The final ViT LayerNorm followed by a BatchNorm-projected MLP
  is the correct LeWM pattern.
- AdaLN-Zero initialization is mandatory: the final linear layer producing
  (shift, scale, gate) must be zero-initialized so the predictor starts as
  identity-on-residual. Verify in `tests/test_adaln_zero.py`.
- bf16 mixed precision is fine for encoder + predictor, but compute Epps-Pulley in fp32
  for numerical stability. The characteristic function involves complex exponentials
  whose magnitude is well-bounded but whose differences are not.
- The training set is small (114 train encounters). Use spanwise mirror, small temporal
  jitter on episode start, and the optional (Y, G, omega_z) sign-flip symmetry. Do NOT
  use rotations.
- High probe R^2 on the encoder for c is a red flag, not a success. The encoder is
  unconditional by design; if it can decode c, c is leaking from somewhere (the wrong
  data path, an auxiliary channel, or the BatchNorm statistics correlating with c).
- The "AeroJEPA" preprint may appear in the literature search during the project. It is
  a likely direct competitor. When it becomes available, summarize differences in
  `notebooks/literature_aerojepa.ipynb` and update the paper introduction accordingly.
- Omega_z DNS sign convention is `du/dy - dv/dx` (opposite of the standard right-hand
  rule). Magnitudes are correct; only the sign flips. If you plot omega_z and "positive
  rotation" looks inverted, it is the convention, not a bug. See SESSION_DATA_PREP.md
  Step 0 status section.

## How to update this document

After every significant decision or finding, append a new entry to "Decision history"
(D11, D12, ...) with date, decision, rationale, and alternatives. Move resolved items
from "Open questions" to the decision log with the resolution rationale. Keep "Suggested
next steps" current. Commit `HANDOFF.md` changes with messages of the form
`handoff: D11 chose X for reason Y`.
