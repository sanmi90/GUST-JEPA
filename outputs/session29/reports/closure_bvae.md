# beta-VAE closure row (tab:closure)

Six-observable closure for the beta-VAE (Solera-Rico) baseline, representation
and forecast endpoints, computed for a new beta-VAE row in
`paper/sections/tables/results_tables.tex` (tab:closure) alongside the existing
predictive (JEPA-tf), reconstructive (Fukami), and linear (POD) rows.

## Provenance

- Git sha (repo at run time): `98d204694d74c494558006c5af2bfc548323b66d`
- Run UTC: 2026-06-14T22:12:46Z
- Command: `python scripts/session29/emit_closure_bvae_part.py`
- Writer: `scripts/session29/emit_closure_bvae_part.py` (flake8-clean, max-line 100)
- Output part: `outputs/session28/numbers_parts/closure_bvae.json`
  (sha256 `984b136817c6241021a2362618f0e272663f7ab42370c05dd7bc8df2ef27a500`)

### Inputs

- Closure matrix: `outputs/session28/closure_matrix/matrix.csv`
  (sha256 `5b7969c96d81b01e07608b1692831a80f0920db0a1d636996331ba24a56d2c71`),
  produced by `scripts/session28/closure_matrix.py` at git
  `823f2c78080b668239e5afec93c7f42ae481c43c`, generated 2026-06-13T06:21:44Z,
  against canonical DNS physical metrics
  `outputs/session28/exp2/dns_physical_metrics.npz`. The matrix already carries
  the beta-VAE cells (no re-encoding, no rollout recomputation here): CPU-only
  extraction with the same seed-mean filter the JEPA/Fukami/POD rows use.

## Protocol (identical to the existing rows)

Values are the seed-mean over a cell's encoder seeds of held-out R^2 at horizon
H = 16, split `test_b`, tier `pooled`, latent d = 64, probe `ridge`. The ridge
probe is the fixed linear readout fitted on the family's TRAIN per-frame latents
against the canonical per-frame DNS observables, case-clustered 5-fold readout
(`closure_matrix.py`). This is a verbatim reuse of the `seed_mean` filter in
`scripts/session28/emit_closure_floor_part.py`, the writer of the
JEPA/Fukami/POD closure rows.

## Recipe choice (documented; mirrors the paper's own split)

- Representation: MATCHED recipe `bvae_match_d64_s{42,0,1}` (the Solera-Rico
  lift+wake head). This is the recipe whose wake-repr seed-mean is
  `NumReprWakeBvaeMatch` = 0.53, the value the closure table MUST cite for the
  beta-VAE wake-repr cell; it is also the recipe `sensing_cf.py` uses for the
  plain "Bvae" sensing row.
- Forecast: rollout `bvae_d64`, the ONLY beta-VAE matched-predictor rollout on
  disk. Its predictor and rollout were trained from `bvae_faith_d64_s42`
  (verified in `outputs/session28/predictors/bvae_d64/train.log` and
  `rollouts/bvae_d64/eval.log`; consistent with `drift_ce1.py` PRED_LAT and
  `families_closure.yaml bvae_faith.rollouts.keys`). No matched-recipe predictor
  was ever trained (`bvae_match` carries no `rollouts` entry in the families
  manifest), so the beta-VAE forecast row is single-seed (s42).
- This representation-matched / forecast-faithful split is NOT a defect: the
  existing paper already mixes the two beta-VAE recipes (drift `DriftMahaBvae`
  and topology macros use faithful; sensing `SenseStateBvae` and wake-repr
  `NumReprWakeBvaeMatch` use matched). The closure row's representation column
  is internally consistent (all six observables from `bvae_match`); the forecast
  column reports the only available beta-VAE rollout and is flagged single-seed.

## Reproduce-check (must pass before trusting beta-VAE outputs)

Re-derived seven published comparator cells from the SAME matrix.csv with the
SAME filter; all round-match the macros in `paper/macros.tex`:

| macro | recomputed | published |
| --- | --- | --- |
| NumReprWakeFukami | -0.25 | -0.25 |
| NumReprWakeBvaeMatch | +0.53 | +0.53 |
| NumCloReprCLJepaTf | +0.86 | +0.86 |
| NumCloReprCLFukami | +0.56 | +0.56 |
| NumCloReprCDPod | +0.64 | +0.64 |
| NumCloFcstWakeJepaTf | +0.43 | +0.43 |
| NumCloFcstWakePod | -0.60 | -0.60 |

The mandated Fukami wake-repr check (`NumReprWakeFukami` = -0.25) passes, so the
pipeline is trusted.

## beta-VAE closure values (12 macros)

Representation (matched recipe; wake-repr cell reuses `NumReprWakeBvaeMatch` =
0.53 and is NOT minted here to avoid a duplicate macro):

| observable | macro | R^2 |
| --- | --- | --- |
| C_L | NumCloReprCLBvae | +0.69 |
| C_D | NumCloReprCDBvae | +0.52 |
| I_y | NumCloReprIyBvae | +0.05 |
| circ.pos | NumCloReprCircPosBvae | +0.54 |
| circ.neg | NumCloReprCircNegBvae | +0.44 |
| wake (reuse) | NumReprWakeBvaeMatch | +0.53 |
| mean over six | NumCloReprMeanSixBvae | +0.46 |

The mean-over-six (0.46) includes the matched wake-repr value (0.53) so it is
comparable to `NumCloReprMeanSixJepaTf` = 0.61.

Forecast (rollout `bvae_d64`, single seed s42; full six observables):

| observable | macro | R^2 |
| --- | --- | --- |
| C_L | NumCloFcstCLBvae | +0.66 |
| C_D | NumCloFcstCDBvae | +0.28 |
| I_y | NumCloFcstIyBvae | +0.30 |
| wake | NumCloFcstWakeBvae | -0.07 |
| circ.pos | NumCloFcstCircPosBvae | +0.29 |
| circ.neg | NumCloFcstCircNegBvae | +0.12 |

## Validation

- `eval_all.validate_record` dry-run over ALL `numbers_parts/*.json` (249 macros,
  256 names): zero unknown keys, zero duplicate names, zero duplicate macros.
- All 12 new macros are alphabetic (LaTeX-safe) and use `fmt = "%.2f"`.
- None of the 12 macro names already exist in `paper/macros.tex` (collision scan
  clean).
- The table should cite `NumReprWakeBvaeMatch` for the beta-VAE wake-repr cell;
  do NOT add a `NumCloReprWakeBvae` (it would duplicate).
