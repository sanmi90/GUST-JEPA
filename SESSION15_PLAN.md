# SESSION15_PLAN.md

Session 15 plan: test spanwise-mean representations for both vorticity (encoder
input) and pressure (sensor selection target). Motivated by the convergent
Session 14 D110/D112/D113 findings that the spanwise mean carries the
parameter-prediction signal more cleanly than the single mid-span slice.

Date: 2026-05-25 (drafted at end of Session 14)
Predecessor: SESSION14_REPORT.md (production winner is E d=64 + SL on slice;
spanwise-mean is a candidate replacement)

## Background (from Session 14)

Three independent findings point the same direction:

- **D110**: Spanwise-mean pressure beats single-slice pressure for both
  z_R2 (0.69 vs 0.11 at K=2 + RBF) and C_L_R2 (0.93 vs 0.65) in sensor
  selection.
- **D112**: When 4 sensor-selection methods are compared, sensor IDs disagree
  but chord REGIONS agree (LE cluster, pressure-side mid-chord, mid-chord).
  All comparisons used the spanwise-mean pressure.
- **D113**: Spanwise-mean vorticity through the SLICE-trained encoder gives
  +0.07 (G,D,Y) linear-probe R^2 over slice input on Test B (0.755 vs 0.683),
  driven by a +0.25 jump on the Y axis. The slice-trained encoder works
  BETTER on mean input than on its own training distribution.

The hypothesis Session 15 tests: spanwise mean is the right preprocessing for
both pressure and vorticity in this Re=5000 parametric vortex setting.

## Session 15 thrusts

### S15-T1: Train a JEPA encoder on spanwise-mean vorticity (D113 follow-up, ~8 h GPU)

End-to-end retrain of the production E d=64 + SL recipe on spanwise-mean
omega. Compare to the slice-trained baseline on every Session 14 metric.

Scripts (already written in Session 14, ready to launch):

- `scripts/build_omega_mean_cache.py` -- builds v1_mean cache (spanwise mean
  across 32 z-stations) under `${PREVENT_ROOT}/data/processed/vortex-jepa/v1_mean/`.
  ~30-60 min CPU + disk I/O. Disk usage ~7.5 GB.
- `scripts/build_omega_mean_pipeline.py` -- computes per-encounter p99.99 clip
  thresholds + train_stats for the mean cache; writes
  `outputs/data_pipeline/v1_mean/manifest.json`.
- `scripts/session14_path2_meantrain.sh <gpu>` -- chained orchestrator that
  runs both build scripts, trains E d=64, trains SL decoder, runs extended
  metrics eval. Total ~8 h on one RTX 6000.
- `scripts/session14_welch_then_meantrain_watchdog.sh` -- waits for Welch JSON
  then fires the chain. Currently NOT launched (per user direction at end of
  Session 14).

To start Session 15-T1: just `bash scripts/session14_path2_meantrain.sh 0`.

Comparison metrics (slice vs mean):

| Metric | Slice (D99) | Mean (Session 15 target) |
|---|---|---|
| Test B SSIM mean (D99 production) | 0.526 | ? |
| Test B lambda-ratio | 1.64 | ? |
| Test C SSIM mean | 0.303 | ? |
| GDY linear probe R^2 (impact frame) | 0.683 | ? (zero-shot was 0.755) |
| Intrinsic dim consensus | 3.00 | ? |

Pre-registered prediction: mean-trained encoder gives Test B GDY R^2 > 0.80
(zero-shot was 0.755) and SSIM within +/- 0.03 of slice. If both true, mean is
the new production recipe.

### S15-T2: Spanwise-averaged pressure probe portfolio (extension of D112, ~3 h CPU)

Re-run the four-method sensor-selection portfolio (TCSI, MI-greedy, LASSO,
qDEIM) + multi-learner evaluation (Ridge, RBF, MLP, TCN) on the spanwise-mean
pressure (the current production p_wall). This finalises Session 14 D112 by
running the methods at their best learner configuration on the clean v1.5
split, with bootstrap stability and the chord-region consensus figure.

Already mostly done in Session 14 (see D112 + D111). Session 15 only needs:
- (a) Re-run with split_v1p5_clean (rather than v1.5 dirty) -- only a 3-encounter
  change, ~5 min CPU.
- (b) Add TCN proxy learner results to the table once Session 14's TCN+SHAP
  script lands. If the TCN evaluator lands in Session 14 just promote it
  rather than re-running.

### S15-T3: Slice-pressure paired ablation (mean vs slice for pressure)

D110 already showed mean beats slice for sensor selection. Session 15
finalises this with a paired comparison: re-run the same methods with the
SLICE pressure (z=0.5625 single station) and report a paired table side-by-side
with the mean numbers. Confirms the D110 finding is robust to method choice.

~2 h CPU. Uses the slice-pressure files already extracted in Session 14 at
`outputs/session14/pressure_slice/<case_id>_enc<XX>.npy`.

### S15-T4: Run3 simulation re-runs (data integrity)

Three encounters were flagged in Session 14 D109 as corrupt (NaN in C_L, C_D,
p_wall):
- `G+2.00_D1.50_Y+0.00/encounter_03`
- `G+2.00_D1.50_Y+0.40/encounter_03`
- `G-2.00_D1.50_Y+0.10/encounter_03`

All three are encounter_03 (the LAST encounter) of run3 train cases. Late-stage
DNS crashes. Re-run them on the PREVENT side; then re-run the integrity audit
to confirm clean.

After re-run: `python scripts/data_integrity_audit.py --split configs/splits/split_v1p5.json --write-clean-split`
should report 0 flagged encounters.

### S15-T5: Diffusion refinement of the SL decoder (deferred from Session 14)

Per the Session 14 Thrust 7 plan's decision tree, if the TCSI sensor track does
not produce a publishable headline, Session 15 should run diffusion refinement
of the SL decoder. Session 14's TCSI did produce a publishable claim (K=2 LE
sensors with kernel-ridge proxy), so diffusion is no longer the default. But
the diffusion track is still worth running:
- PRF 2026 paper explicitly recommends diffusion as the next-step.
- It addresses the "Figure 3 is still blurry at frame 55" critique from
  Session 12 outcome.
- Doesn't conflict with S15-T1 (different stage of the pipeline).

Estimated 4-8h GPU + several hundred lines of new diffusion code (likely
EDM2 or DDIM noise schedule on top of the SL-decoded omega).

## Session 15 sequencing

If two RTX 6000 GPUs available:
- GPU 0: S15-T1 (mean retrain, 8h)
- GPU 1: S15-T5 (diffusion, after T1's encoder finishes so the diffusion can
  train on the SL output of the new encoder if it wins; if it doesn't, train on
  the slice encoder's SL output).
- CPU: S15-T2 + S15-T3 (sensor portfolio additions, 5-6h)
- External: S15-T4 (PREVENT-side DNS re-runs)

Total Session 15 wall clock: ~10-12 h.

## What this enables for the paper

If S15-T1 confirms the mean encoder beats the slice encoder, the paper's
Section 5 reframes around mean preprocessing as the right choice for both
inputs. The slice version becomes the "naive default" baseline; the mean
version becomes the production recipe.

If S15-T1 shows a small mean delta (within +/- 0.03 SSIM and +/- 0.05 GDY R^2),
the paper reports both side-by-side as a methodology comparison.

If S15-T1 fails (mean retrain underperforms slice), the paper reports the
D113 zero-shot finding as a curiosity and keeps slice as the production recipe.

## Open follow-ups carried from Session 14

- Welch t-tests on the 3 JEPA d=64 seeds (Session 14, queued; eval watchdog
  will fire when SL decoder seed 1 finishes within ~20 min of Session 14 end).
- TCN + SHAP comparison on Thrust 7 (Session 14, in flight at end of session).
- Re-eval of all Session 14 figures using the cleaned v1.5 split (cosmetic;
  the 3 corrupt encounters were already being silently dropped).

## Files staged for Session 15

- `scripts/build_omega_mean_cache.py` (NEW, Session 14)
- `scripts/build_omega_mean_pipeline.py` (NEW, Session 14)
- `scripts/session14_path2_meantrain.sh` (NEW, Session 14)
- `scripts/session14_welch_then_meantrain_watchdog.sh` (NEW, Session 14 -- not
  launched, can be repurposed in Session 15)
- `scripts/data_integrity_audit.py` (NEW, Session 14)
- `configs/splits/split_v1p5_clean.json` (NEW, Session 14)
- All Session 14 outputs under `outputs/session14/`
