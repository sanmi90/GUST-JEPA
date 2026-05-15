# Session report — 2026-05-15

Scope: bootstrap PREVENT JEPA preprocessing pipeline; freeze partition v1.

## Starting state

| Item | State |
|---|---|
| `data_manifest/raw_cases_inventory.yaml` | present, 41 cases (21 periodic + 20 run3), `manifest_version: raw_cases_inventory_v1` |
| `build_split_manifest.py` | present but referenced sandbox paths `/mnt/user-data/…`; excluded `Baseline` from JEPA train/test |
| `split_v1.json` | present, locked from sandbox-path version |
| `requirements.txt` | present (torch 2.1.2, h5py, wandb, etc.) |
| `scripts/100c_raw_cases_inventory.py` | present |
| `.venv/`, `CLAUDE.md`, `.claude/`, `configs/`, `outputs/`, `notebooks/`, `src/` | none of these existed |

## What ran end-to-end

1. **Environment** — created `.venv/` from Python 3.10.12; installed all of `requirements.txt` (h5py 3.16, PyYAML 6.0.3, numpy 1.26.4, torch 2.1.2+cu121, scipy, matplotlib, jupyter, wandb, …).
2. **Schema inspection (Step 0)** — `scripts/inspect_raw_hdf5.py` against `Baseline.h5` (periodic) and `Gust_002_x-1.916_y-0.581_s-3.0_d1.5.h5` (run3); outputs in `outputs/schema_inspection/`.
3. **Targeted re-inspection** — resolved 4 open questions (curlU component order, sensors layout, time alignment, lift vs CL).
4. **Config + scripts** — wrote `configs/preprocessing.yaml` and `scripts/preprocess.py`; updated `build_split_manifest.py` (paths + Baseline policy); regenerated `split_v1.json`; inserted "Step 0 status" section in `SESSION_DATA_PREP.md`.
5. **Preprocessing run** — `scripts/preprocess.py --partition v1` over all 41 cases in 7m 40s. 200 encounters written + 6 (Baseline) skipped (already cached from the smoke test).
6. **QC notebook** — `notebooks/00_qc_partition_v1.ipynb` with three checks, executed in-place; revised Check 2 after first run revealed a bimodal vorticity-argmax distribution; added Check 2B (force-domain).
7. **Dataset loader** — `src/data/episode_dataset.py`; smoke-tested for all four splits.

## Schema facts resolved during Step 0

| Aspect | Value |
|---|---|
| Field datasets | `/u`, `/curlU` shape `(T, 192, 96, 32, 3)`, chunked + gzip |
| Velocity component order | `(u_x, u_y, u_z)` in mesh frame |
| Spanwise vorticity | `/curlU[..., 2]`; sign convention `du/dy − dv/dx` (verified: correlation **−0.86** with computed `dv/dx − du/dy` in the clean wake) |
| Mesh orientation | Chord-aligned with x-axis; freestream tilted +α=14° (upstream `u_x ≈ cos 14°`, `u_y ≈ sin 14°`) |
| Grid | nx=192, ny=96, nz=32; x∈[−1.5, 4.5], y∈[−1.5, 1.5], z∈[0, 1] |
| `L_z` | **1.0** (plan guess of "~0.5c" was wrong) |
| Mid-span index | **`nz // 2 = 16`** (z=0.516); `argmin(|z|)=0` is a periodic boundary, not mid |
| Wall pressure | Pre-extracted at `/sensors/p` shape `(1536, T)` = **192 surface points × 8 z-stations** (inner axis is z) |
| Spanwise averaging | `reshape((192, 8, T)).mean(axis=1)`; no tolerance parameter |
| Airfoil surface | `/airfoil_xy` shape `(193, 2)` (192 unique + 1 closing point) — explicit, no NACA0012 re-derivation |
| Forces | `CL = 2·lift`, `CD = 2·drag` exact. **C_M not stored** (would require integration of `/sensors/p` along `/airfoil_xy`) |
| Time axis | `forces/time[0] = 2.5e-4 ≈ 0` (frame 0 at gust launch); `dt = 0.0500 t/c` exact; 800 frames = 40 t/c = 6 gust periods + 80-frame trailing partial |
| Solid mask | `/inside_solid` shape `(192, 96, 32, 1)`, 2624 cells (~0.44%); `/u` and `/curlU` are NaN in those cells (filled with 0 during preprocessing) |
| Impact validation | Gust_007 (G=+4.0): `CL_min` at frame 29 (−7.60), `CL_max` at frame 51 (+10.82). Plan window [25, 55] correctly brackets both peaks. |

## Files

### New

| Path | Purpose |
|---|---|
| `.venv/` | Project virtualenv |
| `configs/preprocessing.yaml` | Schema-baked processing parameters, `preprocessing_version: 1.0.0` |
| `scripts/inspect_raw_hdf5.py` | Step 0 schema inspector |
| `scripts/preprocess.py` | Per-encounter cache extractor (omega_z, p_wall, C_L, C_D) |
| `src/__init__.py`, `src/data/__init__.py` | Package roots |
| `src/data/episode_dataset.py` | Impact-aware sub-trajectory sampler (`EpisodeDataset`) |
| `notebooks/00_qc_partition_v1.ipynb` | QC notebook (11 cells: 6 markdown + 5 code) |
| `outputs/schema_inspection/{periodic.txt, run3.txt, schema.yaml}` | Inspection artifacts |
| `SESSION_DATA_PREP.md` | The pasted plan (with new "Step 0 status" section at top) |

### Modified

| Path | Change |
|---|---|
| `build_split_manifest.py` | Paths now relative to repo; Baseline moved into `train` (encs 0–3) + `test_a` (encs 4–5) with `is_calibration_reference: true`; removed the `baseline` split bucket |
| `split_v1.json` | Regenerated. New counts: 31 train + 6 test_b + 4 test_c cases; 108 train + 46 test_a + 28 test_b + 24 test_c = **206 encounters**; `n_cases_calibration_reference: 1` (Baseline) |

### Memory (`~/.claude/projects/-home-carlos-GUST-JEPA/memory/`)

| File | Content |
|---|---|
| `MEMORY.md` | Index |
| `baseline-in-training.md` | Records the project decision that Baseline is in JEPA train/test_a, not "calibration only" |

## Cache

Location: `$PREVENT_ROOT/data/processed/vortex-jepa/v1/`. Total: **1.3 GB**, 206 encounter files.

```
v1/
├── Baseline/                         (6 encounters, included in train)
├── G+0.25_D0.50_Y+0.10/              (6 encounters, periodic)
├── …                                 (19 more periodic gust cases × 6 enc)
├── G+4.00_D1.00_Y-0.10/              (6 encounters, Test C)
├── G-3.00_D1.50_Y-0.10/              (4 encounters, run3)
└── …                                 (19 more run3 cases × 4 enc)
```

Per encounter (~6.5 MB compressed):
- `omega_z` `(120, 192, 96)` float32, chunks `(1, 192, 96)`, gzip
- `p_wall` `(120, 192)` float32, chunks `(120, 192)`
- `C_L`, `C_D` `(120,)` float32
- 17 attrs (case_id, G, D, Y, source_group, encounter_index, frame_start/end, dt_tc, impact_frame_estimate, mid_span_index, omega_z_sign_convention, preprocessing_version, partition_version, raw_relative_path, n_frames)

## Verification

| Test | Result |
|---|---|
| `preprocess.py` Baseline smoke vs raw | C_L bit-identical; omega_z bit-identical after NaN→0; p_wall matches `reshape((192,8)).mean(axis=1)` to **5.96e-8** (float32 epsilon) |
| Cache integrity (QC Check 1) | 206/206 encounters, 0 shape/attr issues |
| Vorticity-domain impact frame (QC Check 2A) | All 200 non-Baseline encounters in window [25, 55]; mean = **40.80** (within ±2 of 40) |
| Force-domain impact frame (QC Check 2B) | All in [25, 55]; mean = **38.80**; 26.5% within ±3 of 40 (vs 7% for vorticity — force tracks the kinematic landmark better) |
| Probe identification (QC Check 3) | LE at idx 11 (x=−0.0001), TE at idx 107 (x=+1.0001) — exact contour endpoints |
| Loader `__len__` per split | train=108, test_a=46, test_b=28, test_c=24 — exact match to `split_v1.json` summary |
| Loader sample shapes | omega_z `(32, 192, 96)`, p_wall `(32, 192)`, C_L/C_D `(32,)` — all `torch.float32` |
| Loader impact-aware fraction (1000 samples) | **0.814 observed** vs **0.811 predicted** by mixture math (70% × 1.0 + 30% × 33/89) |
| Loader reproducibility (`seed=42`) | Identical starts across independent instantiations ✓ |

## Notable findings / non-obvious results

- **Vorticity argmax is bimodal, not centered at frame 40.** Strong gusts (G ≥ +2) peak at frames 25–32 (vortex at peak strength entering the ROI). Weak/moderate gusts (G ≤ +1) peak at frames 50–55 (post-impact LE shear/separation). The mean averages near 40 because both lobes balance. The plan's "mode at 40" criterion was over-strict; relaxed to "all in window AND mean within ±2."
- **Force-domain argmax tracks the kinematic landmark better** than vorticity does — `argmax(|C_L − C_L_baseline|)` has mean 38.8 and is less bimodal.
- **The plan's spanwise-tolerance preprocessing (Step 2) is moot:** wall pressure is already extracted at exactly 8 spanwise stations. No tolerance parameter is needed.
- **C_M is not in the raw HDF5.** If observable-augmented baselines need it, integrate `/sensors/p` × moment arm along `/airfoil_xy`.
- **Impact-aware start range `[8, 40]` has an off-by-one** if interpreted as "guarantee frame 40 in sub-traj": with `L=32` and `start=8`, sub-traj is frames 8..39, not including 40. Sub-traj contains frame 40 only for `start ≥ 9`. The 70/30 mixture observation (0.814 ≈ predicted 0.811) is still correct; this is just a label vs semantics nit, not a data bug. Worth revisiting before tuning the sampler.

## Current status

Partition v1 frozen and training-ready. Steps 0–6 of `SESSION_DATA_PREP.md` complete.

## Suggested next steps

1. JEPA model (encoder + predictor) and training loop — pick an architecture (ViT-style encoder over `omega_z` per frame, transformer predictor over frame embeddings).
2. Wire `wandb`/`tensorboard` logging keyed to `preprocessing_version`, `partition_version`, and `is_calibration_reference` flags.
3. Decide on observable augmentation: feed `C_L`/`C_D`/`p_wall` to the predictor (or only to a decoder head).
4. Reconsider the off-by-one in `impact_aware_start_range`.
5. Optional: extend the QC notebook with a small "out-of-tolerance gallery" (cases where vorticity argmax sits at 25 or 55) for diagnostic value.
