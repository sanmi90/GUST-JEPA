# Session report — 2026-05-16

Scope: Session 2 model primitives (SIGReg, AdaLN-Zero, RoPE) + D14 absorption
of two additional run3 cases (Gust_025, Gust_026) into v1.

## Starting state

| Item | State |
|---|---|
| Last commit on `origin/main` | `029226f` (D12 absorption of Gust_023/024) |
| Local `main` | already 0 commits ahead at session start |
| `src/models/` | did not exist |
| `tests/` | did not exist |
| Partition v1 | 43 cases / 214 encounters; split SHA256 `0f07a746…` |
| Inventory | 43 cases (21 periodic + 22 run3) |
| `SESSION2_MODEL_PRIMITIVES.md` | did not exist |
| HANDOFF.md decision log | last entry was D12 |

## What ran end-to-end

1. **Session 2 plan materialized** — wrote `SESSION2_MODEL_PRIMITIVES.md` at the
   repo root (verbatim from the user-provided spec).
2. **Scaffolded `src/models/` and `tests/`** with empty `__init__.py` files.
3. **TDD on RoPE** — wrote `tests/test_rope.py` (5 tests), confirmed RED via
   `ModuleNotFoundError`, implemented `src/models/rope.py`
   (`build_rope_cache`, `apply_rope`), all 5 green.
4. **TDD on AdaLN-Zero** — wrote `tests/test_adaln_zero.py` (4 tests),
   confirmed RED, implemented `src/models/adaln.py` (`AdaLN` with zero-init),
   all 4 green.
5. **SIGReg numerical-magnitude sanity check** — wrote a throwaway numpy
   reference of the Epps-Pulley quadrature and confirmed the spec's threshold
   structure (Gauss < 0.1, Student-t > 5.0, Uniform > 1.0) was not
   simultaneously satisfiable under either standard scaling (no `N`
   multiplier per LeWM Appendix A, or `* N` per LeJEPA paper's reference
   PyTorch listing). Consulted arXiv:2511.08544 (LeJEPA) and arXiv:2603.19312
   (LeWM) appendix A for the canonical formulas.
6. **TDD on SIGReg with calibrated thresholds** — recorded decision in
   HANDOFF.md D13; wrote `tests/test_sigreg.py` with empirically calibrated
   thresholds (Gauss < 0.01, Student-t > 0.05, Uniform > 0.02), confirmed
   RED, implemented `src/models/sigreg.py` (LeWM Appendix A formula, no `N`
   multiplier, fp32 body even under bf16 autocast), all 6 green.
7. **Full verification** — `pytest tests/ -v` → 15/15 green;
   `black --check --line-length 100` → clean; `flake8 --max-line-length=100`
   → clean (after dropping an unused `torch` import in `adaln.py`). `ruff`
   is not installed in `.venv/` (only `flake8` + `pyflakes`); flagged for
   later.
8. **Session 2 squash commit** — `537af15 Session 2: SIGReg, AdaLN-Zero, RoPE
   primitives (15 tests green)` (10 files, +906 lines).
9. **D14 absorption** — collaborator dropped two more run3 files in
   `$PREVENT_ROOT/data/raw/periodic/run3/`
   (`Gust_025_x-1.916_y-0.581_s-1.0_d1.5.h5` and
   `Gust_026_x-1.989_y-0.290_s-1.5_d1.0.h5`, both 2026-05-16 09:17). Decoded
   via the locked alpha=14 degree rotation to
   `G-1.00_D1.50_Y-0.10` and `G-1.50_D1.00_Y+0.20`; both default to `train`.
10. **Re-ran inventory** — `python scripts/100c_raw_cases_inventory.py` ->
    45 cases (21 periodic + 24 run3), 0 parse errors, 0 duplicate case_ids.
    Copied the regenerated YAML from
    `$PREVENT_ROOT/data_manifest/raw_cases_inventory.yaml` to
    `data_manifest/raw_cases_inventory.yaml` (the in-repo copy).
11. **Re-ran split manifest** — `python build_split_manifest.py` -> 35 train +
    6 test_b + 4 test_c cases; 120 train + 50 test_a + 28 test_b + 24 test_c
    encounters; new SHA256 captured.
12. **Re-ran preprocessing** — `PREVENT_ROOT=$HOME/PREVENT
    python scripts/preprocess.py --partition v1` -> wrote 8 new encounter
    files (4 each for `G-1.00_D1.50_Y-0.10` and `G-1.50_D1.00_Y+0.20`),
    skipped the 214 pre-existing files.
13. **Loader smoke test** — `EpisodeDataset(partition="v1", split="train")`
    returns 120 encounters; the two new cases occupy train slots 114-119 (3
    train + 1 test_a per case as expected for run3); a sample from
    `G-1.00_D1.50_Y-0.10` returned `omega_z.shape=(32, 192, 96)` with
    correct `(G, D, Y)` attrs and `frame_start=21` (deterministic from
    seed * idx).

## Files

### New

| Path | Purpose |
|---|---|
| `SESSION2_MODEL_PRIMITIVES.md` | Session 2 spec (verbatim from the user prompt) |
| `src/models/__init__.py` | Package root |
| `src/models/sigreg.py` | LeWM Appendix A SIGReg with bf16 autocast support |
| `src/models/adaln.py` | AdaLN-Zero (identity-on-residual at init) |
| `src/models/rope.py` | 1D temporal RoPE for the predictor |
| `tests/__init__.py` | Package root |
| `tests/test_sigreg.py` | 6 tests (3 distribution thresholds + M-invariance + grad + bf16 dtype) |
| `tests/test_adaln_zero.py` | 4 tests (zero-output, identity block, grad after step, time-broadcast) |
| `tests/test_rope.py` | 5 tests (identity at t=0, offset-invariance, cache shapes/dtypes, odd-head-dim reject) |
| `SESSION_REPORT_2026-05-16.md` | this report |

### Modified

| Path | Change |
|---|---|
| `data_manifest/raw_cases_inventory.yaml` | 43 -> 45 cases (21 periodic + 24 run3); new SHA256 `d67d65d369097875403169c8065f56d4612479be2b4712a177d8d7505d76f74f` |
| `configs/splits/split_v1.json` | 43 -> 45 cases; train 33 -> 35; train enc 114 -> 120; test_a enc 48 -> 50; new SHA256 `f21abb5d48008031d628042bd46743a82e3dd28c194e8a66dc22e7dee8b8bf8c` |
| `HANDOFF.md` | added D13 (SIGReg scaling decision) and D14 (Gust_025/026 absorption into v1) |

## Cache

Location: `$PREVENT_ROOT/data/processed/vortex-jepa/v1/`. Total now 222 encounter
files (~1.4 GB). New entries:

```
v1/G-1.00_D1.50_Y-0.10/encounter_{00..03}.h5   (Gust_025, run3 -> 3 train + 1 test_a)
v1/G-1.50_D1.00_Y+0.20/encounter_{00..03}.h5   (Gust_026, run3 -> 3 train + 1 test_a)
```

Per-encounter schema is unchanged from partition v1 (omega_z, p_wall, C_L,
C_D, 17 attrs; preprocessing_version 1.0.0).

## Verification

| Test | Result |
|---|---|
| `pytest tests/` after Session 2 implementation | **15/15 green** in ~11.4 s |
| `black --check --line-length 100 src/models/ tests/` | clean (8 files, 0 changes) |
| `flake8 --max-line-length=100 src/models/ tests/` | clean (after dropping unused `torch` import in adaln.py) |
| SIGReg numpy reference cross-check (B=4096) | Gauss `1.24e-4`, Student-t df=2 `0.120`, Uniform(-1,1) `0.053`; ratios preserved |
| Inventory regeneration | 45 cases, 0 parse errors, 0 duplicate case_ids |
| `build_split_manifest.py` summary | 45 / 35 / 6 / 4 cases, 120 / 50 / 28 / 24 enc, total 222 in splits |
| New cache file count | 8 written, 214 skipped, total 222 on disk |
| `EpisodeDataset` smoke test on updated v1 | `len(train)=120, len(test_a)=50, len(test_b)=28, len(test_c)=24`; new cases at train indices 114-119; sample shape `(32, 192, 96)` and `(G, D, Y)` attrs match |

## Decision log additions

- **D13** — SIGReg follows LeWM Appendix A (no `N` multiplier). Spec's
  threshold magnitudes (0.1 / 5.0 / 1.0) were not jointly satisfiable under
  either standard scaling; re-calibrated empirically to `< 0.01`, `> 0.05`,
  `> 0.02` for Gaussian / Student-t df=2 / Uniform(-1, 1). Discriminative
  ordering Gaussian << Uniform < Student-t is preserved; absolute scale is
  absorbed into the outer regularizer weight `lambda` (tunable by bisection
  over [0.001, 1.0] per CLAUDE.md).
- **D14** — absorb Gust_025 (`G-1.00_D1.50_Y-0.10`) and Gust_026
  (`G-1.50_D1.00_Y+0.20`) into v1 train. Same precedent as D12: pre-training
  phase, no paper-reportable v1 checkpoint yet, so v2 versioning would be
  premature. Next absorption after the first reportable v1 run MUST go to v2.

## Counts after Session 2

| Bucket | Cases | Encounters |
|---|---|---|
| Train | 35 | 120 |
| Test A (held-out inside training cases) | (same 35) | 50 |
| Test B (parametric interpolation) | 6 | 28 |
| Test C (extrapolation, G = +4) | 4 | 24 |
| Calibration reference (flag, not a split) | 1 (Baseline) | -- |
| **Total** | **45** | **222** |

## Current status

Three model primitives landed with their unit tests. The encoder, predictor,
and JEPA wrapper for Session 3 can now import them as-is without modification
(per the closing paragraph of `SESSION2_MODEL_PRIMITIVES.md` -- if Session 3
forces a primitive to evolve, that is a flag to stop and update the Session 2
tests first).

Partition v1 grew by two run3 cases to 45 cases / 222 encounters, with the
new cases inside the training envelope (|G| <= 3). The split manifest, the
inventory, and the cache are all consistent and reproducible.

## Suggested next steps

1. Session 3: encoder (`src/models/encoder.py`), predictor
   (`src/models/predictor.py`), and JEPA wrapper (`src/models/jepa.py`).
   See HANDOFF "Suggested next steps" #2 and SESSION2's closing paragraph.
2. Install `ruff` and add `ruff check` to the verification stack (currently
   only `flake8` is in the venv).
3. Decide whether to add `src/models/vicreg.py` proactively or wait until
   the SIGReg auto-fallback rule fires at iteration 20k of the first
   training run.
4. Optional: tighten the dataset smoke test in `notebooks/00_qc_partition_v1.ipynb`
   to re-run after D14 (current notebook is from 2026-05-15 and predates the
   absorption).
