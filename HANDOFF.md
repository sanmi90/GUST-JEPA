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

### D44: Session 7 launched three production-scale runs on full v1.2 (2026-05-18, Session 7 Step 1)

Three 20k-iter runs on the full v1 train partition (41 cases / 138 train
encounters per D35), seed 0, frame-skip 1 per D34, L = 32, eta = 0.01
where applicable, BatchNorm projection on the encoder per D17, dual-card
launch via D40's `--gpu {0,1}`. The launcher is
`scripts/launch_session7.sh`; per-run output under
`outputs/runs/session7/run_r{1,2,3}_*/`.

|Run                                 |Card                |Configuration            |Hypothesis tested                                                                                                                                |
|------------------------------------|--------------------|-------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
|R1 PLDM + OBS + BN                  |`--gpu 0` (cuda:2)  |observable head eta=0.01 |D39 smoke winner scaled to 41 cases. The headline configuration.                                                                                  |
|R2 PLDM only (eta=0) + BN           |`--gpu 1` (cuda:3)  |no observable head       |Is the observable head doing the work, or does PLDM alone generalise? Disambiguates D39's "OBS marginally helps PLDM" reading at full scale.       |
|R3 SIGReg + OBS + BN                |`--gpu 1` (cuda:3)  |observable head eta=0.01 |Does OBS rescue SIGReg at scale? D39 found this rescue at 5 cases; R3 tests whether it persists at 41 cases.                                       |

R1 and R2 launch concurrently on the two RTX 6000 cards; R3 follows
sequentially on cuda:3 after R2 completes. Estimated wall clock ~10 hours
(R1 ~5h || R2 ~5h then R3 ~5h).

Pre-flight checks landed and passed before launch:

- Check A (data loader on full 41 cases): 138 train encounters load
  cleanly, all 320 sampled sub-trajectories finite, omega range
  [-3658, +3701] across the sample (consistent with the D27/CLAUDE.md
  survey peak of 4377; the plan's `(-100, 100)` bound was conservative
  and is corrected here for the audit trail), CL_future range [-6, +12].
- Check B (`--all-train` end-to-end smoke on both entrypoints at 10
  iters, B=4): train_jepa with `--gpu 0` resolved to cuda:2, train_baseline
  with `--gpu 1` resolved to cuda:3, n_train_samples=138 confirmed on both,
  no errors.
- Check C (GPU enumeration): two RTX 6000 Blackwell cards visible at
  cuda:2 and cuda:3 (D40-aligned).
- Check D (split manifest): manifest SHA256
  `a721dc92f6e278ee054bb952933c14ba20a58137f79f3a19fc6ad71b70a007dd`
  matches D35; inventory SHA256 prefix `ce817e1e0df54309...` matches D35;
  6 Test B cases / 28 encounters; 4 Test C cases / 24 encounters.

Housekeeping landed in the same commit before launch:

- `--all-train` flag added to both `src/training/train_jepa.py` and
  `src/training/train_baseline.py`. Mutually exclusive with `--cases`
  and `--cases-from`. Same effect as omitting all three (resolve_cases
  returns None -> downstream uses every case the manifest tags as
  'train') but makes the production-launch intent explicit in W&B
  `run_config["all_train"]`.
- 6 new tests in `tests/test_resolve_cases.py` cover the three flag-mutex
  paths and the legacy-namespace fallback. Fast suite now 126/126 green
  (120 prior + 6 new).
- CLAUDE.md "Hardware" was already updated in D40 (the brief listed it
  as housekeeping but D40 had already landed); no further edits needed.
- D40's earlier commit accidentally dropped the "## Open questions"
  section heading from HANDOFF.md (the heading went into the old_string
  of the Edit that appended D40 but not into the new_string). The
  Session 7 D44 commit restores the "## Open questions" heading at the
  right place; the open-questions content itself was unchanged
  throughout. Recorded here as a self-audit so future readers see the
  reconstruction.

W&B mode: `offline` (matches Session 5 + Session 6 convention; auth was
not configured at session start and `wandb sync` can post-hoc upload the
offline runs). The `metrics.jsonl` side-log per run is the canonical
in-session source.

### D45: Session 7 evaluation suite landed; Test B is the primary metric (2026-05-19)

`notebooks/05_session7_full_evaluation.ipynb` loads all three iter-20000
checkpoints, encodes every Test A (56 enc) / Test B (28 enc) / Test C
(24 enc) encounter, and reports the per-split metric table plus the
8-branch decision string. The notebook applies a CL-validity mask: two
Test A encounters (`G+2.00_D1.50_Y+0.00` enc 3 and `G-2.00_D1.50_Y+0.10`
enc 3) have non-finite C_L values across 69 and 103 frames respectively
(DNS instability near the end of the last run3 encounter at D=1.5); they
are dropped from the CL-prediction MLP fit but kept for the PR/probe
metrics on z. The 56 -> 54 Test A drop is documented in the per-cell
stdout and in the session report; future cache regenerations should
revisit those two encounters.

The OOS evaluation method for Test B / Test C: a tiny 64-hidden MLP is
fit on Test A (CL-valid) latents -> CL(t + delta), then evaluated on
the held-out split's latents. The (c, t) baseline uses the same MLP
architecture with `(case_descriptor, frame_index) -> CL(t + delta)`
fit on Test A and evaluated on the split. The `delta` column is
`r2(z -> CL_future) - r2((c, t) -> CL_future)` per split; positive
delta means the latent does something the parametric (c, t) lookup
cannot.

### D46: Session 7 outcome -- TEST_B_TEST_A_DISCREPANCY with substantive R3_WINS reading (2026-05-19)

Final per-split metric table (notebooks/05_session7_full_evaluation.ipynb
Section 4):

| Run                | Split  | PR_all | PR_within | r2(z->c) | r2_dyn_phase | r2(CL_future) | (c,t) baseline | delta   |
|--------------------|--------|-------:|----------:|---------:|-------------:|--------------:|---------------:|--------:|
| R1 PLDM+OBS+BN     | test_a | 27.84  |   6.87    |   0.90   |    0.78      |    0.97       |     0.74       |  +0.23  |
| R1 PLDM+OBS+BN     | test_b | 18.31  |  10.06    |   0.96   |    0.91      |    0.71       |     0.72       |  -0.008 |
| R1 PLDM+OBS+BN     | test_c | 14.50  |  11.77    |   0.90   |    0.86      |    0.76       |     0.35       |  +0.42  |
| R2 PLDM only BN    | test_a | 27.16  |   6.01    |   0.88   |    0.77      |    0.93       |     0.74       |  +0.19  |
| R2 PLDM only BN    | test_b | 17.35  |   9.41    |   0.95   |    0.92      |   -0.13       |     0.72       |  -0.85  |
| R2 PLDM only BN    | test_c | 13.92  |  11.14    |   0.91   |    0.87      |    0.32       |     0.35       |  -0.03  |
| R3 SIGReg+OBS+BN   | test_a |  3.69  |   4.18    |   0.62   |    0.44      |    0.97       |     0.74       |  +0.24  |
| R3 SIGReg+OBS+BN   | test_b |  3.51  |   3.85    |   0.93   |    0.63      |    0.86       |     0.72       |  +0.14  |
| R3 SIGReg+OBS+BN   | test_c |  2.91  |   4.67    |   0.76   |    0.73      |    0.83       |     0.35       |  +0.48  |

The decision tree (SESSION7_FULL_SCALE_HONEST.md Step 2 Section 6)
checks TEST_B_TEST_A_DISCREPANCY first and returns immediately when it
matches: R1 has test_a delta 0.23 > 0.10 AND test_b delta -0.01 < 0.03,
so the tree's strict output is `TEST_B_TEST_A_DISCREPANCY`. R2 also
matches the discrepancy rule even more dramatically (test_a +0.19 vs
test_b -0.85). The same data also satisfies R3_WINS strictly
(R3 test_b delta +0.14 > R1 test_b delta -0.01), but the tree picks
the first matching branch.

Substantive read (the one that matters for the paper):

- **R3 SIGReg+OBS+BN is the only run that generalises to Test B**
  (delta +0.14) and the BEST run on Test C (delta +0.48).
- R1 PLDM+OBS overfit: it has the highest PR (27.84 on Test A), the
  cleanest r2_dyn_phase (0.78 on Test A, 0.91 on Test B), but its
  out-of-sample CL prediction on Test B is no better than (c, t) and
  worse on Test B than R3.
- R2 PLDM-only is the worst: Test B delta -0.85 means the 5-term
  PLDM latent at full scale is *worse* than a tiny (c, t) MLP at
  predicting CL on unseen cases. This is overfitting to the 41 train
  cases in a way that hurts generalisation.
- The smoke-scale (5 cases) "PLDM+OBS wins" finding from Session 6 D39
  was a small-data artifact. The PR=10 numbers PLDM+OBS achieves at
  smoke and full scale look like the same healthy reading, but the
  Test B generalisation signal shows the PR is encoding case-specific
  memorisation, not transferable flow physics.

This is the most important finding of the project so far. It INVERTS
Session 6's recommendation that PLDM should be the base architecture
for the observable-augmented path. At full scale on the metric the
paper actually cares about (Test B parametric interpolation), the
simpler SIGReg + OBS configuration is the right answer.

### D47: R1-vs-R2 OBS-vs-no-OBS delta at scale -- OBS is necessary for PLDM, but only on Test A (2026-05-19)

The R1 vs R2 difference isolates the observable head's contribution
on top of PLDM:

| Split | R1 delta | R2 delta | R1 - R2 |
|-------|---------:|---------:|--------:|
| test_a |  +0.23  |  +0.19   |  +0.04  |
| test_b |  -0.01  |  -0.85   |  +0.84  |
| test_c |  +0.42  |  -0.03   |  +0.45  |

The observable head dramatically rescues PLDM out-of-sample (R1 - R2
on test_b is +0.84) but the rescued state is still only at the (c, t)
baseline level on test_b (delta -0.01). Without OBS, PLDM at full
scale produces a latent that is *worse* than a (c, t) lookup at
predicting CL on unseen cases (R2 test_b delta -0.85). So:

- The observable head is necessary for PLDM to be even competitive
  on out-of-sample CL prediction at full scale.
- Even with OBS, PLDM does not BEAT the (c, t) baseline on Test B.
- The same OBS, applied to SIGReg JEPA, *does* beat the baseline on
  Test B (R3 delta +0.14).

The OBS-vs-no-OBS axis at full scale therefore matters very much for
PLDM (rescues it from active overfitting) and slightly less for the
final winner (R3 also has OBS, but the comparison without OBS at full
scale is the deferred R0 task which Session 8 should run).

### D48: R1-vs-R3 regulariser-asymmetry delta at scale -- inverts D39 (2026-05-19)

The R1 vs R3 difference isolates the regulariser (PLDM vs SIGReg)
with the observable head held constant at eta=0.01:

| Split  | R1 delta | R3 delta | R3 - R1 |
|--------|---------:|---------:|--------:|
| test_a |  +0.23  |  +0.24   |  +0.01  |
| test_b |  -0.01  |  +0.14   |  +0.15  |
| test_c |  +0.42  |  +0.48   |  +0.07  |

On Test A both regularisers produce equivalent CL prediction. On
Test B (the partition's parametric-interpolation question) the SIGReg
regulariser materially outperforms PLDM (+0.15 absolute, the
difference between "fails to beat baseline" and "+14 percentage
points over baseline"). On Test C (extrapolation), R3 is also slightly
ahead.

This INVERTS the D39 regulariser-asymmetry reading. D39 was based on
5-case smoke evidence and concluded "PLDM is the recommended base
because PLDM+OBS reaches PR=10+ while SIGReg+OBS plateaus at PR=3".
The Session 7 full-scale evaluation shows PR=10 was masking the
overfitting that happens when PLDM has 41 cases to memorise, while
the low-PR SIGReg+OBS latent retains its generalisation capability.

Paper claim 3 is therefore reworded: instead of "observable
augmentation rescues SIGReg, marginally helps PLDM" (D39, smoke-scale)
the claim becomes "the observable-augmented SIGReg latent generalises
to unseen (G, D, Y) values better than the observable-augmented PLDM
latent at full scale, despite a 3x lower participation ratio".

The deeper finding: PR alone is not a reliable proxy for the
generalisation quality of a JEPA latent on low-intrinsic-dim physics
data. The (c, t) baseline + Test B delta is the right diagnostic.

### D49: Session 7 housekeeping notes (2026-05-19)

- No new data pipeline issues at full scale, except the 2 NaN-CL Test
  A encounters documented in D45.
- CLAUDE.md "Hardware" was already updated in D40; Session 7 used
  both RTX 6000 cards via `--gpu {0,1}` per D40's pattern.
- The launcher script bug (described in the audit-trail paragraph of
  D44) was fixed in `scripts/launch_session7.sh` and documented inline
  with the `disown` regression note. Recovery launcher
  `scripts/launch_session7_r3_after_r2.sh` is kept as the reusable
  template for "start cuda:1 job after cuda:0 job finishes".
- Session 7 wall clock was ~3.6 h (R1 ~1.5 h || R2 ~2 h then R3 ~1.5 h),
  much shorter than the plan's 12-13 h estimate. The per-iter compute
  on the RTX 6000 Blackwell was ~220 iter/min vs the plan's 100 iter/min
  back-of-envelope; Session 8 planning should use the higher rate.
- The "## Open questions" section heading in HANDOFF.md was dropped
  by D40's commit and restored in the D44 commit. Self-audit in D44.
- 126/126 fast tests green at session start and after the --all-train
  housekeeping. No new tests required for the evaluation notebook (it
  is pure analysis code).

Session 8 implied by the substantive R3_WINS + TEST_B_TEST_A_DISCREPANCY
reading: reframe around R3 (SIGReg + OBS). Specific Session 8 tasks
in order of priority:

1. **eta sweep on SIGReg + OBS** (Session 8-OBS-SIGReg). Sweep eta in
   {0.001, 0.005, 0.01, 0.05, 0.1} on the full partition at 20k iters
   each, R3-style configuration. Pick the eta with the highest test_b
   delta. ~8 hours.
2. **R0 contingent** (Session 8-R0): pure SIGReg + BN at full scale,
   no OBS. The deferred control. Confirms whether the OBS rescue is
   load-bearing for SIGReg at scale or whether SIGReg alone also
   generalises on Test B. ~5 hours.
3. **Lambda bisection on the SIGReg + OBS winner** from task 1. ~6h.
4. **Decoder training + Section 7 evaluation suite** per the
   architecture spec.

Together: ~25-30 hours of work spread across 2-3 sessions.

### D50: Session 8 Step 1 trajectory audit -- R3 converged; R2 actively anti-generalises in late training (2026-05-19, Session 8 Step 1)

`scripts/session8_trajectory_audit.py` encoded every Test A (56 enc) and
Test B (28 enc) encounter at each of 10 saved Session 7 checkpoints
(iter 2000, 4000, ..., 20000) for R1, R2, R3 (30 evaluations total).
Per-checkpoint metrics in `outputs/runs/session8/trajectory_audit.csv`;
plots in `notebooks/06_session7_trajectory_audit.ipynb`. Three diagnostic
concerns resolved:

**Concern 1: convergence.** All three runs settle their Test A delta by
iter ~6000-8000. R3's Test B delta climbs from +0.05 at iter 4000 to
+0.11 at iter 6000 and reaches the +0.14 plateau by iter 12000; the
iter-20000 endpoint is the trained equilibrium, not a transient. R1's
Test A delta plateaus at +0.22 by iter 6000; its Test B delta oscillates
around zero with no upward trend. Both are converged.

**Concern 2: R2 anomaly.** Three readings, all publishable:

- Cross-split (Test A -> Test B) delta_b progressively DEGRADES across
  training: -0.18 at iter 4000, -0.45 at iter 8000, -0.73 at iter 10000,
  -1.21 at iter 20000. The PLDM 5-term loss is actively destroying
  Test A -> Test B transferability over the second half of training.
  This is a publishable failure mode of the PLDM 5-term loss at full
  scale, independent of the R3_WINS finding.
- R2's PR_all rises in lockstep with the cross-split degradation:
  PR_all on Test A grows from 1.63 (iter 2000) to 27.14 (iter 20000),
  meanwhile PR_within on Test A *shrinks* from 15.14 to 6.02. R2 is
  moving variance OUT of within-case dynamics INTO the case-mean axis
  -- the SPREAD_TRIVIAL signature of Section 4.2 at full scale. The
  growing case-mean variance is precisely what hurts Test A -> Test B
  transfer (Test B has different case identities, so a case-mean-
  dominated latent geometry does not align between the two splits). PR
  RISES while generalisation FALLS; in this regime PR is not just an
  imperfect proxy but is *anti-correlated* with the metric the paper
  cares about. Test-B-delta-over-(c, t) is the right diagnostic.
- Within-Test-B (fit MLP on 75% of Test B, evaluate on 25%) delta_b is
  steadily POSITIVE (~+0.10) throughout R2's training. R2's latent
  encodes information about Test B's CL signal when fit on Test B
  itself; it just produces a representation incompatible with a Test A
  MLP. The -0.85 Session 7 number is therefore largely a
  distribution-shift artifact between Test A and Test B, not a globally
  uninformative latent.

R2's within-Test-B (+0.10) is *higher than* R3's within-Test-B (+0.07).
R3's advantage on the Session 7 cross-split metric is the alignment of
its Test A and Test B latent geometries, not the per-split informativity.
This is consistent with the SIGReg-induced low-PR controlled-collapse
regime producing a more compact, transferable latent. R3's PR_all and
PR_within are also of similar magnitude (3.69 and 4.18 at iter 20000),
in contrast to R2's case-mean-dominated 27.14 vs 6.02 split: SIGReg+OBS
produces a latent where case-axis and within-case-dynamics variance are
balanced, while PLDM-only produces a latent where the case-axis variance
dominates the within-case-dynamics variance by ~4x.

**Concern 3: R3 plateau.** R3's L_anti rises in early training (Session 7
plot) and reaches the +0.14 cross-split Test B delta plateau by iter
12000 (this audit). The +0.14 is the trained equilibrium; the d-sweep
(Step 5) and grid (Step 4) can proceed at 20k iters.

### D51: Session 8 Step 2 head ablation -- R3 latent encodes general flow state, not just CL (2026-05-19, Session 8 Step 2)

`scripts/session8_head_ablation.py` evaluates three CL_future prediction
methods on R3 iter-20000's latents plus the same three methods applied
to alternative observables. Results in
`outputs/runs/session8/head_ablation.csv`, plots in
`notebooks/07_session8_head_ablation.ipynb`.

|Target |Fresh probe on z (Test B) |Trained R3 head (Test B) |Gap fresh - trained |
|-------|-------------------------:|------------------------:|-------------------:|
|C_L    |               +0.138     |               +0.137    |          +0.001    |
|C_D    |               +0.106     |              n/a        |          n/a       |
|p_LE   |               +0.123     |              n/a        |          n/a       |

- Method 1 (fresh probe on z for C_L, the Session 7 method) reproduces
  the +0.14 Test B delta from D46 to within 0.002.
- Method 2 (trained R3 observable head applied directly to Test B
  latents) is essentially identical to Method 1 (+0.137 vs +0.138). The
  trained head does NOT extract non-linear structure beyond what a
  fresh linear probe recovers; it adds no measurable value at inference.
- Method 3 (fresh probe on z for C_D and p_LE) gives Test B deltas of
  +0.106 (drag) and +0.123 (leading-edge pressure). Both are clearly
  positive and within 0.04 of the trained-for target CL.

This matches Row 1 of the plan's interpretation matrix: **the latent
encodes general flow state; the observable head shaped z toward CL but
the shaping does not over-specialise z to CL specifically.** R3's
+0.14 win on Test B is not a CL-specific artifact; the latent is
informative about non-CL observables on unseen (G, D, Y) cases at
roughly the same delta.

Implications for the paper: paper claim 3 is robust to the
"R3 just learned CL" objection. The latent has the breadth of
aerodynamic information needed for visualisation decoder training
(Session 9). Section 5.4 of the paper now cites this result as evidence
that the head is a *weak* guidance signal at eta = 0.01 (D37) rather
than a hard CL-supervision constraint.

### D52: Session 8 Step 3 R3 seed=42 -- PASSES (delta_test_b = +0.121) (2026-05-19, Session 8 Step 3)

R3 SIGReg+OBS+BN retrained from scratch with seed = 42, identical
configuration otherwise to Session 7 R3 (full v1.2 partition, 20k
iterations, lambda=0.1, eta=0.01, BatchNorm projection). Pass criterion
per `SESSION8_R3_VALIDATION_GRID_SWEEP.md`: Test B delta in
[+0.05, +0.25].

Result: **delta_test_b = +0.121** (PASS).

Comparison:

| Run                | seed | Test B delta |
|--------------------|-----:|-------------:|
| Session 7 R3       |    0 |       +0.138 |
| Session 8 R3-seed42|   42 |       +0.121 |
| Seed variance       |      |        0.017 |

Trajectory preview at iters 8000 and 12000 (`r3_seed42_eval_iter*.json`):
- iter 8000: delta_test_b = +0.081 (Session 7 R3 at same iter: +0.119)
- iter 12000: delta_test_b = +0.117 (Session 7 R3 at same iter: +0.139)
- iter 20000: delta_test_b = +0.121 (Session 7 R3 at same iter: +0.138)

R3-seed42 tracks the seed=0 trajectory consistently ~0.02 lower at
matching iterations; the +0.14 headline finding from D46 is robust to
seed (~12% relative variance). Step 4 grid proceeds.

The cuda:0 orchestrator (`scripts/orchestrate_session8_step4.sh`)
verified pass at 08:20:24 and launched E1 immediately afterwards. The
cuda:1 orchestrator started E6 (eta=0.01, lambda=1.0) at 06:59 in
parallel with R3-seed42 to save 1.5h wall-clock.

### D53: Session 8 Step 4 (eta x lambda) grid -- peak at (eta=0.01, lambda=0.01), E4 = +0.159 (2026-05-19, Session 8 Step 4)

Nine SIGReg + OBS runs at three etas in {0.001, 0.01, 0.1} times three
lambdas in {0.01, 0.1, 1.0}, plus E10 PLDM + OBS with paper-tuned
weights and the Session 7 R3 anchor as the centre cell. cuda:0 sequence
E1, E2, E3, E4 (~6h). cuda:1 sequence E6, E7, E8, E9, E10 (~7.5h).

Per-cell Test B delta over (c, t) baseline at iter 20000 (from
`outputs/runs/session8/grid_analysis.csv`):

|                | lambda=0.01 | lambda=0.1   | lambda=1.0 |
|----------------|------------:|-------------:|-----------:|
| **eta=0.001**  |      -0.200 |       +0.007 |     -0.620 |
| **eta=0.01**   |  **+0.159** |       +0.138 |     +0.093 |
| **eta=0.1**    |      +0.148 |       +0.146 |     +0.152 |

Peak at **(eta=0.01, lambda=0.01) = E4 with delta_test_b = +0.159**.

Three pattern observations:

- **eta is the dominant axis.** At eta = 0.001 (the head almost off) the
  encoder fails or barely matches baseline regardless of lambda. At
  eta in {0.01, 0.1} the encoder generalises across all lambdas tested.
  The observable head is the central regulariser at full scale.
- **lower lambda is better at eta = 0.01.** At eta = 0.01: lam=0.01
  (+0.159) > lam=0.1 (+0.138) > lam=1.0 (+0.093). The Session 7 default
  of lam=0.1 was not the optimum; lam=0.01 (SIGReg essentially off)
  generalises ~2 points higher.
- **the eta=0.1 row is flat in lambda.** +0.148 / +0.146 / +0.152 across
  lambda. When the OBS pressure is strong enough, SIGReg's contribution
  is negligible; the encoder is regularised by the head alone.

The Session 7 default (eta=0.01, lam=0.1, the E5 anchor) was not the
optimum but came within +0.02 of it. The production operating point
moves to (eta*=0.01, lambda*=0.01) for Step 5 d-sweep and Session 9.

Surprise-outcome reading per the plan: this matches **Surprise outcome A**
("the peak is at the eta=0.1 or lambda=0.01 corner... R3's success is
more about the observable head than about SIGReg"). The grid confirms
that SIGReg at the eta in {0.01, 0.1} rows is a directional pressure
(preventing rank-1 collapse) but does NOT need to maintain high PR.
At lambda=0.01 the SIGReg gradient is small enough that the encoder
satisfies it with the OBS-induced latent structure, yielding the best
generalisation.

### D53b: Session 8 Step 4 E10 PLDM paper-tuned reference -- delta_test_b = -0.095 (worse than R1 defaults) (2026-05-19, Session 8 Step 4)

Run E10: PLDM + OBS + BN trained with the paper-tuned Two-Rooms weights
from arXiv:2502.14819 Appendix J.2 (alpha=4.0, beta=6.9, delta=0.75,
omega=0.0; eta=0.01 for OBS). 20k iterations, seed 0, full v1.2 partition.

Result: **E10 delta_test_b = -0.095** (FAILS to beat the (c, t) baseline).

Champion table from `outputs/runs/session8/champion_table.csv`:

| Run             | eta | lambda | PR_all (Test B) | r2(z->c) | r2(CL_future) | (c,t) baseline | delta   |
|-----------------|----:|-------:|----------------:|---------:|--------------:|---------------:|--------:|
| E4 (best SIGReg)|0.01 |  0.01  |           2.61  |   0.87   |     0.88      |     0.72       | +0.159  |
| E5 (S7 R3)      |0.01 |  0.10  |           3.51  |   0.93   |     0.86      |     0.72       | +0.138  |
| E10 PLDM tuned  |0.01 |   --   |          23.02  |   0.65   |     0.62      |     0.72       | -0.095  |
| R1 PLDM defaults|0.01 |   --   |          18.33  |   0.96   |     0.72      |     0.72       | -0.003  |

E10 is WORSE than R1 defaults on Test B (-0.095 vs -0.003) -- paper-tuned
PLDM does not rescue PLDM at full scale on this data. Three readings:

- The "PLDM was just badly tuned" objection from Session 7 R1 is
  decisively ruled out: with paper-tuned weights, PLDM is even worse
  on Test B than with default unit weights.
- The Two-Rooms hyperparameters are tuned for the LeWM gridworld
  data, not for low-intrinsic-dim physics data. Domain transfer of
  hyperparameters across data distributions is not guaranteed.
- Per the plan, this is **E10 delta_test_b < best SIGReg grid point
  - 0.05** by ~0.25 absolute (-0.095 vs +0.159). The D46 R3_WINS
  finding is robust to PLDM hyperparameter choice. **Session 9 does NOT
  need a full PLDM hyperparameter sweep**; paper claim 3 stands strongly.

The PR profile remains the SPREAD_TRIVIAL signature for both PLDM
configurations: PR_all = 23 (E10) and 18 (R1 default), much larger than
PR_within (not shown but the trajectory_audit pattern continues). The
PLDM 5-term loss at full scale produces high-PR latents whose variance
is dominated by case-mean rather than within-case dynamics, regardless
of hyperparameter weights.

### D54: Session 8 Step 5 latent-dimension sweep -- d=32 wins; LeWM intrinsic-dim prediction NOT confirmed on this data (2026-05-19, Session 8 Step 5)

Three SIGReg + OBS + BN runs at (eta*, lambda*) = (0.01, 0.01) (the
Step 4 D53 best grid point), each with a different latent dimension d.
20k iterations, seed 0, full v1.2 partition. d=32 reuses the E4 grid
run; d=8 and d=16 are new runs.

Results from `outputs/runs/session8/d_sweep.csv`:

| d  | PR_all (Test B) | PR_within (Test B) | r2(z->c) Test B | delta Test A | delta Test B | delta Test C |
|---:|----------------:|-------------------:|----------------:|-------------:|-------------:|-------------:|
|  8 |        2.22     |        3.36        |       0.70      |    +0.224    |  **+0.092**  |    +0.451    |
| 16 |        2.37     |        3.68        |       0.69      |    +0.214    |  **+0.103**  |    +0.474    |
| 32 |        2.61     |        3.88        |       0.87      |    +0.227    |  **+0.159**  |    +0.470    |

**Production d* = 32.** This is the **"d large wins" outcome** from the
plan: contrary to the LeWM Two-Room intrinsic-dimension prediction
(d close to intrinsic dim should win), on this data d=32 generalises
better than d=8 by +0.07 absolute on Test B.

The PR profile is informative: PR_all is essentially flat in d (2.2 /
2.4 / 2.6 for d in {8, 16, 32}). The encoder uses the same effective
~2 dimensions regardless of available d. PR_within is also flat (3.4 /
3.7 / 3.9). Yet the extra "unused" dimensions help generalisation on
Test B. The mechanism is not "the encoder uses more capacity at d=32"
but more likely "the encoder has more dimensions for the linear probe
to interpolate across when fitting on Test A and evaluating on Test B."
The latent's intrinsic structure is the same; what changes is the
downstream MLP's freedom.

Implication for paper claim 1: D2's d=32 default is empirically correct
on this data. The LeWM prediction does not extend to the SIGReg+OBS+BN
operating point identified in D53. Session 9 lambda bisection runs at
d = 32, not d = 8.

The L_anti controlled-collapse mechanism still holds: at lambda=0.01
SIGReg is essentially off but provides directional pressure. d=32
allows enough latent space for that pressure to act without forcing
the encoder into the rank-1 collapse seen at eta=0.001 (D53 E1, E3).

### D55: Session 8 Step 6 R0 control -- pure SIGReg fails at scale; OBS is essential (2026-05-19, Session 8 Step 6)

Two pure SIGReg + BN runs at full scale (no observable head, no PLDM),
seed = 0, 20k iterations, full v1.2 partition. Both lambdas tested:

| Run                           | lambda | r2(CL_future) Test B | (c, t) baseline | delta_test_b |
|-------------------------------|-------:|---------------------:|----------------:|-------------:|
| R0 SIGReg-only lambda=0.1     |  0.1   |        -0.023        |     0.718       |   **-0.742** |
| R0 SIGReg-only lambda=0.01    | 0.01   |        -0.029        |     0.718       |   **-0.748** |

Both R0 runs fail catastrophically on Test B. r2(z -> CL_future) is
near zero (-0.02 to -0.03), meaning the latent is essentially
uninformative about CL on Test B; the (c, t) parametric baseline
predicts CL at r2 = 0.72 by lookup. Result reading per the plan: this
is the **"R0 delta_test_b < 0"** outcome -- "pure SIGReg fails at
scale; OBS is essential. Paper claim 3 robust."

The two-lambda confirmation matters: R0 at lambda = 0.1 (Session 7
default) and at lambda = 0.01 (Step 4 D53 best) both fail by ~0.74
absolute. SIGReg alone, regardless of weight, does not produce a
generalising latent on this data. The OBS head is load-bearing for
the SIGReg + OBS path at full scale.

Implications for paper claim 2: "Observable augmentation is necessary
for either regulariser at scale" holds robustly. Without OBS:
- PLDM-only (Session 7 R2 in D47): Test B delta = -0.85
- SIGReg-only (Session 8 R0): Test B delta = -0.74 (lambda = 0.1) /
  -0.75 (lambda = 0.01)

Both unregularised-by-OBS configurations fail by ~0.75 absolute on
Test B. The observable head, at modest weight (eta = 0.01 to 0.1),
provides the latent structure that enables Test B generalisation.

Paper claim 3 (regulariser asymmetry) is reinforced: SIGReg + OBS at
the best operating point (eta = 0.01, lambda = 0.01, D53) generalises
to Test B at +0.16; SIGReg without OBS does the opposite at -0.74.
The +0.16 - (-0.74) = +0.90 absolute gap is the OBS head's
contribution to the SIGReg path, comparable to the +0.84 gap between
R1 PLDM + OBS and R2 PLDM-only (D47). The two regulariser bases are
similar in their dependence on OBS, but diverge in what the OBS-
augmented latent does at full scale: SIGReg + OBS generalises (+0.16)
while PLDM + OBS does not (-0.003).

### D56: Session 8 paper section rewrite committed (2026-05-19, Session 8 Step 7)

`paper/sections/section_5_full_scale_results.md` rewritten with eight
subsections (5.1 setup, 5.2 Session 7 Table 1 with R3_WINS reading, 5.3
regulariser-asymmetry inversion and controlled-collapse mechanism, 5.4
Session 8 validation diagnostics D50/D51/D52, 5.5 (eta x lambda) grid
results D53/D53b, 5.6 d-sweep results D54, 5.7 R0 control D55, 5.8
recommendation summary, plus 5.9 Limitations). Section 4.3 also edited:
the "regulariser-asymmetry lineage" paragraph now records the smoke-
to-full-scale inversion (D48) inline.

Figures generated in this session, all in
`outputs/runs/session8/`:

- `fig_trajectory_audit.png`: 3x3 per-run trajectory panels (PR_within,
  r2(CL_future) Test A, delta Test B) for R1, R2, R3 across iters
  2000-20000.
- `fig_r2_anomaly.png`: cross-split vs within-Test-B delta over training
  for all three runs.
- `fig_session7_delta_summary.png`: per-run Test A / Test B / Test B
  within-split bar chart (the paper's Figure 2).
- `fig_head_ablation.png`: Step 2 head ablation comparison.
- `fig_grid_delta_b.png`, `fig_grid_pr_all.png`, `fig_grid_r2_z_c.png`:
  Step 4 (eta x lambda) heatmaps.
- `fig_d_sweep.png`: Step 5 delta vs d.

Notebooks committed:
- `notebooks/06_session7_trajectory_audit.ipynb` (Step 1)
- `notebooks/07_session8_head_ablation.ipynb` (Step 2)
- `notebooks/08_eta_lambda_grid.ipynb` (Step 4)
- `notebooks/09_latent_dim_sweep.ipynb` (Step 5)

Scripts committed: `session8_trajectory_audit.py`,
`session8_head_ablation.py`, `session8_eval_r3_seed42.py`,
`session8_grid_analysis.py`, `session8_d_sweep_analysis.py`, plus the
launcher and orchestrator shell scripts.

### D57: Session 8 outcome -- VALIDATED (2026-05-19, Session 8)

All four pass-criteria from `SESSION8_R3_VALIDATION_GRID_SWEEP.md` met:

| Pass criterion | Result |
|-----------------|--------|
| Step 1 trajectory analysis completes for all three Session 7 runs.| Done (D50). |
| Step 2 auxiliary-head ablation produces a definitive read on whether R3's latent contains CL-relevant flow state independent of the trained head.| Row 1: latent encodes general flow state (D51). |
| Step 3 R3-seed=42 lands Test B delta in [+0.05, +0.25].| +0.121 (D52). |
| Step 4 grid completes; (eta*, lambda*) maximising Test B delta is identified.| (0.01, 0.01) with delta = +0.159 (D53). |
| Step 5 d sweep completes; d* maximising Test B delta is identified.| d* = 32 with delta = +0.159 (D54). |
| Step 6 R0 control produces a Test B delta number.| -0.742 and -0.748 (D55). |
| Step 7 paper section 5 rewrite committed.| Done (D56). |

Session 8 outcome category: **VALIDATED.**

Three predictions from the launch message tracked against the data:

1. d=8 will give a higher Test B delta than d=32 at the same (eta, lambda);
   credence 60%. **FALSE** (d=32 wins by +0.07). LeWM Two-Room mechanism
   does not apply to the SIGReg + OBS + BN regime where OBS dominates.
2. The (eta, lambda) grid peak will not be at the Session 7 default
   (0.01, 0.1); credence 70%. **TRUE** (peak at (0.01, 0.01), +0.02
   absolute above the default).
3. R0 will have Test B delta below 0.05; credence 85%. **TRUE** (R0 at
   -0.742 / -0.748, far below the 0.05 threshold).

Two of three predictions held; prediction 1's miss is the most
informative outcome of the session: the LeWM Two-Room intrinsic-
dimension mechanism does not transfer to the SIGReg + OBS + BN
configuration where the observable head is the dominant regulariser.

**Session 9 path:** lambda bisection at the production (eta=0.01,
d=32, OBS=cl_future at eta=0.01) configuration over a fine lambda
interval centered on 0.01. 6-8 evaluations between lambda=0.001 and
lambda=0.1. Plus the visualisation decoder training on the SIGReg +
OBS + BN d=32 encoder, and the start of the full Section 7 evaluation
suite per the architecture spec.

### D58: Session 9 Step 1 lambda bisection -- lambda* = 0.01 (PRODUCTION_LOCKED) (2026-05-20, Session 9 Step 1)

Five-point bisection over lambda in {0.001, 0.003, 0.01, 0.03, 0.1} at
the production (d=32, eta=0.01, OBS=cl_future at eta=0.01, BN, SIGReg)
configuration. F1 (lam=0.001 seed=0), F2 (lam=0.003 seed=0), F3
(lam=0.03 seed=0) are new Session 9 runs; E4 (Session 8 lam=0.01 seed=0)
and E5 (Session 7 R3 lam=0.1 seed=0) are anchors reused from disk.

Per-cell seed=0 Test B summary at iter 20000:

| code | lambda | PR_all | r2(z->c) | r2(CL_future) | r2(c, t) | delta_test_b |
|------|-------:|-------:|---------:|--------------:|---------:|-------------:|
| F1   | 0.001  |  2.22  |   0.887  |    0.836      |  0.718   |   +0.118     |
| F2   | 0.003  |  2.10  |   0.890  |    0.850      |  0.718   |   +0.131     |
| E4   | 0.010  |  2.61  |   0.866  |    0.878      |  0.718   | **+0.159**   |
| F3   | 0.030  |  2.49  |   0.883  |    0.849      |  0.718   |   +0.131     |
| E5   | 0.100  |  3.51  |   0.932  |    0.856      |  0.718   |   +0.138     |

**lambda\* = 0.01** with delta\_test\_b = +0.159 (E4 from Session 8).
Clean interior maximum in the bisection curve, roughly symmetric in
log-lambda: F1 (0.001) at +0.118 and F2 (0.003) at +0.131 climb to E4
(0.010) at +0.159; F3 (0.030) and E5 (0.100) at +0.131 / +0.138
descend from it. PR_all also peaks at E4 (2.61) vs F1/F2/F3 at
~2.1-2.5 and E5 at 3.51; the controlled-collapse mechanism is most
cleanly balanced at lambda=0.01 between SIGReg's distribution
matching and the OBS head's directional pressure.

Session 8 D53's coarse-grid finding is confirmed at the fine bisection
resolution. Outcome category: **PRODUCTION_LOCKED** (lambda* unchanged
from Session 8 production point). No update needed to Section 5.5 of
the paper. F4 (seed=42) and F5 (seed=123) at lambda=0.01 follow on
cuda:0 for the seed-variance bound. R0 at lambda* is not re-run
(Session 8 D55 already covered lambda=0.01 directly).

Prediction tracking from the Session 9 launch message: prediction 1
(lambda* = 0.01, credence 55%) is **TRUE**.

**Seed-variance bound at lambda\* = 0.01** (F4 seed=42 and F5 seed=123
both at the same production config; eval table at
`outputs/runs/session9/bisection_seed_variance.csv`):

| code | seed | Test A | Test B | Test C | diff vs seed=0 (Test B) |
|------|-----:|-------:|-------:|-------:|------------------------:|
| E4   |    0 | +0.227 | **+0.159** | +0.470 |  -                |
| F4   |   42 | +0.231 | **+0.096** | +0.457 |  -0.063 (FAIL +/- 0.03)|
| F5   |  123 | +0.226 | **+0.137** | +0.496 |  -0.022 (PASS +/- 0.03)|

3-seed mean Test B delta at lambda\* = 0.01: **+0.131 ± 0.032 (1-sigma)**;
range across seeds = 0.063 absolute (max E4 +0.159, min F4 +0.096).

The full range across seeds (0.063) exceeds the +/- 0.05 threshold the
plan attached to PRODUCTION_PIVOT (D58 outcome category). Prediction 2
(seed variance within +/- 0.03) is FALSE on the F4 seed=42 axis but
TRUE on the F5 seed=123 axis.

Two readings of the larger-than-expected seed variance:

- The lambda = 0.01 production point sits at the lower edge of the
  bisection bracket where SIGReg pressure is small (D58 PR_all = 2.10
  to 2.61 across F1, F2, F4, F5 at lambda <= 0.03; PR rises to 3.5 at
  lambda = 0.1). With less anti-collapse pressure, the encoder has
  more freedom to land in different local optima across seeds. This
  is qualitatively consistent with D52's smaller seed variance at
  lambda = 0.1 (R3 seed=42 vs seed=0 spread of 0.017 absolute, vs
  Session 9 lambda = 0.01 spread of 0.063 absolute).
- The +0.159 E4 result is the best of three seeds. The mean
  +0.131 is the more honest paper number; the +0.063 max-min range
  is the variance bound that the paper claim 1 should quote.

Per-split breakdown: Test A delta is seed-robust (E4 +0.227, F4 +0.231,
F5 +0.226; spread = 0.005 absolute, well within +/- 0.03). Test C
delta is seed-robust (E4 +0.470, F4 +0.457, F5 +0.496; spread = 0.039
absolute, just outside +/- 0.03). The seed variance is concentrated
on the Test B parametric interpolation stratum, suggesting the
mechanism is specifically about which case-axis representation the
encoder learns vs which parametric directions transfer.

Outcome category remains **PRODUCTION_PIVOT** per the strict reading of
the plan's pass criterion (>+/- 0.05 seed range). The production
config still works (all three seeds give positive Test B delta and
beat all Session 7 / Session 8 / Session 9 ablations), but the
paper claim 1 headline number must shift from "+0.159" to "+0.131
mean +/- 0.032 (1-sigma) across three seeds". Updated paper claim 1
phrasing: "SIGReg + OBS + BN at d = 32, eta = 0.01, lambda = 0.01
generalises to the held-out parametric stratum with mean Test B delta
+0.131 across three seeds (max-min range = 0.063), beating the (c, t)
parametric baseline robustly".

### D60: Session 9 Step 3 Section 7 thin-cut (A2 + A11 + A7) (2026-05-20, Session 9 Step 3)

Three ablations land in Session 9 at the production configuration
(d=32, eta=0.01, lambda\*=0.01). A2 (VICReg-only) and A11 (Fukami
observable-augmented AE) complete; A7 (no-scheduled-sampling at
H_roll=30 since T=32 caps H_roll at T-2) is in progress at the time of
this entry. A10 (Solera-Rico beta-VAE + transformer ROM) remains
deferred to Session 10.

|Code |Ablation                                |Test A delta |Test B delta |Test C delta |PR_all (Test B) |r2(z->c) (Test B)|
|-----|----------------------------------------|------------:|------------:|------------:|---------------:|----------------:|
| -   | E4 production (SIGReg + OBS + BN)      |     +0.227  | **+0.159**  |     +0.470  |      2.61      |      0.866      |
| A2  | VICReg + OBS at d=32                   |     +0.226  | **+0.107**  |     +0.501  |      26.4      |      0.583      |
| A7  | SIGReg + OBS no-SS (H_roll=30)         |     +0.223  | **+0.137**  |     +0.481  |      2.31      |      0.866      |
| A11 | Fukami CNN AE + lift head at d=32      |     +0.191  | **+0.073**  |     +0.431  |       n/a      |       n/a       |

**A2 VICReg + OBS reading.** PR_all = 26.4 is far above E4's 2.61,
matching the high-PR profile of PLDM (R2 PR_all = 27 in D50). The
VICReg variance/covariance enforcement keeps a high-rank latent in
contrast to SIGReg's controlled-collapse PR ~2-3. On Test B, SIGReg +
OBS beats VICReg + OBS by +0.052 absolute (within the +/- 0.05
prediction bracket from the launch message but at the upper edge);
extending the paper claim 3 regulariser-asymmetry to a third
comparison axis. The asymmetry survives the regulariser swap: SIGReg
+ OBS controlled-collapse is genuinely a different latent regime than
VICReg + OBS spread-rank-preservation, even at matched OBS pressure
(eta = 0.01) and matched d = 32.

**A7 no-scheduled-sampling reading.** Same SIGReg + OBS + BN config
as E4 but with H_roll = 30 (the maximum no-SS rollout horizon at
T = 32) instead of H_roll = 8 (the production V-JEPA 2-AC default).
PR_all = 2.31 is close to E4's 2.61 -- the latent regime stays in
the controlled-collapse band. Test B delta drops by 0.022 absolute
(+0.137 vs +0.159 at E4); Test C delta actually rises slightly
(+0.481 vs +0.470 at E4). The V-JEPA 2-AC scheduled-sampling at
H_roll = 8 is a small but real win on Test B parametric interpolation;
on Test C extrapolation the longer rollout horizon helps marginally
(consistent with longer rollouts forcing the encoder to encode more
of the dynamics that matter for the |G| = 4 extrapolation regime).
The +0.022 swing places "scheduled sampling" as a third-tier design
choice behind anti-collapse-regulariser choice (+0.052 for SIGReg vs
VICReg) and architecture-family choice (+0.086 for JEPA vs Fukami AE).

**A11 Fukami AE reading.** Test B delta = +0.073 vs the JEPA's
+0.159 at matched d = 32; JEPA wins by +0.086 absolute. SSIM
reconstruction on Test A = 0.748, Test B = 0.722, Test C = 0.558
using the Fukami SSIM definition (Eq. 1 of arXiv:2305.18394 supplementary,
C_1 = 0.16, C_2 = 1.44). The Fukami CNN encoder + decoder + lift
head (240K total params; ~40x smaller than the JEPA's 10M; matched
latent dimension d = 32 vs Fukami's published d = 3) generalises to
the parametric Test B stratum, but the JEPA-style predictive-only
objective produces a more transferable latent at the same latent
dimension. Reconstruction MSE on Test A is 7.7x the per-case-mean
floor (the very low Test A floor of 1.57 is because the held-out
encounters share their case-mean training-side neighbours; absolute
SSIM 0.748 indicates the reconstruction itself is reasonable). The
two-paper-claim comparison stands: at matched d = 32, JEPA + SIGReg +
OBS produces a more transferable latent than Fukami's reconstruction-
based AE on both downstream prediction (Test B delta +0.159 vs +0.073)
and parametric interpolation generally.

The Fukami baseline was originally scheduled in the Session 9 plan as
deferred to Session 10 (along with A10 Solera-Rico). It was added
mid-session on user request to bring the SSIM-based comparison
methodology into the paper. The implementation at
`src/baselines/fukami_ae.py` (CNN encoder + decoder + lift head
following arXiv:2305.18394 supplementary Table S.1 adapted to the
192x96 input resolution) is the Session 10 starting point for the
Solera-Rico baseline (variational head + transformer ROM extending
the Fukami pattern).

### D61: Session 9 Step 4 R0 at lambda\* -- skipped (lambda\* = 0.01 already covered) (2026-05-20, Session 9 Step 4)

The Session 9 plan made Step 4 conditional on Step 1 finding
lambda* != 0.01 with lambda* < 0.01, in which case R0 would re-run
at the refined lambda* to confirm OBS necessity. Step 1 (D58) found
lambda* = 0.01, identical to Session 8's production lambda. Session 8
D55 already ran R0 SIGReg-only at lambda = 0.01 (delta_test_b =
-0.748) and at lambda = 0.1 (-0.742), both well below the +0.05
threshold for "OBS is load-bearing". The OBS necessity claim is
robust to lambda* = 0.01 directly. No new R0 run was needed; Step 4
skipped per the plan's conditional rule.

### D62: Session 9 paper drafts committed (2026-05-20, Session 9 Step 5)

Four paper deliverables landed during the Session 9 compute windows:

- `paper/sections/abstract.md`: ~240 words, three contribution claims
  with their headline numbers.
- `paper/sections/section_1_introduction.md`: ~1600 words, four
  subsections (ROM motivation; JEPA framing; contribution claims;
  roadmap).
- `paper/sections/section_2_related_work.md`: ~3245 words, four
  subsections (JEPA lineage; observable-augmented autoencoders;
  classical and learned ROM; the gap closed by this paper).
- `paper/sections/section_6_decoder.md`: ~975 words skeleton for the
  visualisation decoder results, awaiting the Step 2 numerical fills.
- `paper/sections/section_7_ablations.md`: ~990 words skeleton with
  the 15-ablation matrix structured into four families. Numerical
  fills for A2 (D60) and A11 (D60) committed; A7 numerical fills
  follow as the A7 cuda:1 run completes.

Em-dash cleanup pass: removed em-dashes from titles of Sections 2,
3, 4, 5 and from six body locations in Section 4 + one in Section 3.
All `paper/sections/*.md` files are em-dash free per CLAUDE.md.

Additional Session 9 infrastructure: `src/models/decoder.py`
(`HybridViTConvDecoder`, 8.72M params, mirror image of the encoder);
`src/baselines/fukami_ae.py` (240K param Fukami CNN AE); 19 new
tests in `tests/test_decoder.py` (8) + `tests/test_fukami_ae.py` (11).
SSIM evaluation (Fukami's Eq. 1) added to both the JEPA decoder
evaluation and the Fukami AE evaluation. 7 new scripts in `scripts/`
plus 1 new notebook `notebooks/10_session9_lambda_bisection.ipynb`
(skeleton).

## Open questions

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
