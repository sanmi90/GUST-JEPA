# PROVENANCE.md (Session 36, Stage 1)

Split-generation provenance for every main-text number and figure of the v4
build. Generations: v2.2 = split_v2p2 (102 cases / 450 encounters, symmetric
boundary test), v2.1 = 85-case split. Method: per-entry source/split fields
in outputs/session33/numbers.json, the input paths of
scripts/session33/emit_numbers_parts.py per part, and the args/params blocks
embedded in the producing artifacts.

## 1. Numbers pipeline

All 19 parts in outputs/session33/numbers_parts/ load from session31/32/33
artifacts computed on partition v2p2 (input paths in emit_numbers_parts.py;
artifacts embed partition/split args). numbers.json carries a _provenance
block (git commit, 19 parts, per-entry source/split/fmt). No v2.1-generation
part exists in the chain. The single deliberate exception is the
preprocessing-sensitivity table (below).

## 2. Tables (13)

| Table | Feeding part(s) | Generation |
|---|---|---|
| tab:dns_pending (1) | none (author-owned \pending{}) | n/a, BLOCKER |
| tab:architecture (2) | constants | v2.2 |
| tab:enkf (3) | constants / filter parts | v2.2 |
| tab:filter_params (4) | constants | v2.2 |
| tab:baselines (5) | constants + table_x | v2.2 |
| tab:closure (6) | table_x | v2.2; merit column suited-operator, replaced by M1 (D310) |
| tab:mechanism (7) | gates_p_and_table_y | v2.2 |
| tab:recovery (8) | table_w_gate_o | v2.2 |
| tab:family_filter (9) | family_filter_audit | v2.2 |
| tab:envelope (10) | table_v_envelope | v2.2 |
| tab:filter_error (11) | filter_error | v2.2 |
| tab:paired_closure (12) | paired_stats | v2.2 |
| tab:prepsens (13) | prepsens macros | v2.1 DELIBERATE, disclosed in caption (the project's disclosure model; keep) |

## 3. Figures

The 22 session35 v4 figure scripts load session34/35 artifacts (da_*, rex_*,
trackc_*, p1 bands), all produced on v2p2 under the Session 34/35 standing
rules (--partition v2p2, v2p2 manifest). The 9 retained session33 v3 scripts
load session33 numbers_parts and the physics artifacts below, all v2p2.
No figure was traced to a v2.1 input. (Per-script input listing extracted
2026-07-10; regenerate with grep "json.load" scripts/session3{3,5}/fig_*.py.)

## 4. The six M2 targets

| # | Target | Artifact | Producer | Evidence | Disposition |
|---|---|---|---|---|---|
| M2a | Near-null / Mahalanobis mechanism (tab:mechanism, fig:mechanism_hroll) | outputs/session33/track_p3_mechanism.json | scripts/session32/track_p3_mechanism.py | params block: v2p2, test_b scoring | CONFIRMED v2.2, retag only |
| M2b | Parameter-only floor (s3 diagnostics prose) | outputs/session36/param_floor_v2p2.json (RE-RUN this session; supersedes the session-23-era exp_conditioning_floor_plus.py output) | scripts/session36/param_floor_v2p2.py | v2p2; KernelRidge(RBF), GroupKFold(5)-selected, train-fit test_b-scored; impact-frame and H=16 closure-window sample sets | RE-RUN DONE, both directional Methods claims CONFIRMED on v2.2: wake-enstrophy floor LOW at H=16 (R2 0.151) and HIGH at the impact frame (0.687). CAUTION for Stage 4: circulation_neg stays parameter-explainable at H=16 (0.730), so the floor claim must remain wake-enstrophy-specific, never "wake observables" broadly |
| M2c | Latent DMD spectrum / atlas (fig:atlas) | outputs/session33/spectrum_dmd_v2p2.json | scripts/session33/spectrum_dmd_v2p2.py | split train, partition v2p2, d=32 full, 6 models | CONFIRMED v2.2, retag only |
| M2d | Distributed-code gap (s4 physics subsection) | outputs/session33/wake_code_v2p2.json | scripts/session33/wake_code_v2p2.py | partition v2p2, test_b, RidgeCV, k in {1..32} | CONFIRMED v2.2, retag only |
| M2e | Topology (Vietoris-Rips H1) | outputs/session33/topology_v2p2.json | scripts/session33/topology_v2p2.py | v2p2; Baseline train + test_b gusted; ripser maxdim 1 | CONFIRMED v2.2, retag only |
| M2f | Pressure-recovery pillar (tab:recovery, fig:trade) | outputs/session33/track_t_recovery_grid.json | scripts/session33/track_t_recovery_grid.py | v2p2; K in {1,2,4,8} x W in {1,2,4,8,16,30}; selection on train | CONFIRMED v2.2, retag only |

Summary: five of six M2 targets confirm on v2.2 and need no compute; the
parameter-only floor is the single re-run (CPU, minutes). This retires the
expected % PROVENANCE-TODO list for the compute tracks except M2b.

## 5. Track M1 provenance (shared-operator merit)

- Families: the ten tab:closure rows; cache mapping (paper row -> latent
  cache under outputs/session34/trackc_latents/): JepaWake -> jepa_pool_vec
  (session33 vec flagship), SupOnly -> supervised_only_pool (session32 run),
  AeWake -> ae_wake_pool (s32), JepaNowake -> jepa_nowake_pool (s32),
  AeNowake -> ae_nowake_pool (s32), RegAE -> regAE_pool (s32), Bvae -> bvae
  (s32 reference), Fukami -> fukami (s31 reference), FukamiWake ->
  fukami_wake (s31 reference), Pod -> pod (s31 reference). The six caches
  missing at session start (SupOnly, RegAE, Bvae, Fukami, JepaNowake, Pod)
  were encoded this session from the frozen checkpoints
  (checkpoint_iter010000.pt) via scripts/session34/trackc_encode.py on v2p2
  (run-dir symlinks under outputs/runs/session34/).
- Operator: the s3.2.1 manuscript-selected direct forecaster (LSTM h512,
  9 quantiles, 6000 iters; rex_tune.py winner), parameterised in
  scripts/session36/rex_families_m1.py; three operator seeds per family.
- Merit: mean over the five observables of pooled MLP-probe R^2, probes fit
  per family on its own train latents (fit_observable_probes, frozen
  protocol), targets restricted to window_mask rows, sliding 25-frame
  context anchors, test_b. Case-clustered bootstrap CIs (2000 resamples).
- KNOWN DISCREPANCY (% REVIEW-NUMBER, MANUSCRIPT_AUDIT.md): the existing
  suited-operator column computes merit at H = 8 while the tab:closure
  caption says "horizon sixteen"; M1 reports BOTH h = 8 and h = 16. The
  Stage 4 rewrite must fix the caption to whichever horizon the final
  column quotes.
- Output: outputs/session36/m1_shared_merit.json (self-provenanced: git
  commit, operator config, cache paths, gpu_name). The fits follow the
  session-34 latent_rex/rex_tune precedent (JSON provenance, no W&B run per
  fit); this deviates from the SESSION_MS spec's W&B-group request and is
  disclosed here.

## 6. Retained-v3 caption seed provenance (carried Session 35 item)

The retained v3 subsections quote seed-band values through macros (Ps*, X*,
V* parts, all v2p2, n recorded per entry as seed_mean/seed_sd/n in the part
JSONs). The caption-level statement of n and seed count is a Stage 4/5
caption-contract item (every kept caption states split, n, uncertainty
convention); tracked in FIGURE_PLAN when Stage 5 runs.
