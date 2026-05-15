# SESSION_DATA_PREP.md

Session plan for preprocessing the PREVENT raw DNS data into a JEPA-ready cache.

Last updated: 2026-05-15.

## Step 0 status (resolved 2026-05-15)

Schema inspection complete. Outputs:

- `outputs/schema_inspection/{periodic,run3}.txt`, `outputs/schema_inspection/schema.yaml`
- `configs/preprocessing.yaml` — schema-baked processing parameters
- `scripts/preprocess.py` — extracts per-encounter `omega_z`, `p_wall`, `C_L`, `C_D` from raw HDF5

Locked facts that differ from the original plan below:

- `omega_z` is `/curlU[..., 2]`. The DNS sign convention is `du/dy - dv/dx` (opposite of standard right-hand rule). Wake correlation with computed `dv/dx - du/dy` = -0.86; the other two `/curlU` components are uncorrelated (quasi-2D shedding), so component 2 is unambiguously the spanwise vorticity.
- Mid-span index is `nz // 2 = 16` (z[16] ≈ 0.516). `z[0] = 1e-5` is a periodic boundary, NOT mid-span. The plan's `argmin(|z|)` heuristic would have been wrong.
- `L_z = 1.0` (= chord). The plan guess of "~0.5c" was wrong.
- Wall pressure is pre-extracted as `/sensors/p` shape `(1536, T)` = 192 surface points × 8 z-stations (inner axis is z). Spanwise averaging is `reshape((192, 8, T)).mean(axis=1)` — no tolerance parameter is needed and Step 2's tolerance-sensitivity QC is not applicable.
- Forces are pre-computed: `CL = 2·lift`, `CD = 2·drag` exactly. C_M is not stored; would need integration of `/sensors/p` along `/airfoil_xy` if required.
- Frame 0 is at gust launch (`forces/time[0] = 2.5e-4 ≈ 0`); `dt = 0.0500 t/c` exact. 800 frames × 0.05 = 40 t/c = 6 gust periods of 6 t/c + 4 t/c (80-frame) trailing partial.
- Impact-frame estimate of 40 validated on Gust_007 (G=+4.0): `CL_min` at frame 29 (-7.60), `CL_max` at frame 51 (+10.82). Plan's window [25, 55] correctly brackets both peaks.
- `/u` and `/curlU` carry NaN in the 2624 cells where `/inside_solid > 0` (immersed boundary). `omega_z` cache is NaN-filled with 0 during preprocessing.
- Mesh is **chord-aligned with x-axis** and the **freestream is tilted at α=14°** (upstream `u_x ≈ cos 14°`, `u_y ≈ sin 14°`). `/airfoil_xy` shape `(193, 2)` is the explicit airfoil contour in mesh coordinates (193 = 192 unique + 1 closing point).
- Native grid `(192, 96, 32)` is preserved in v1; no spatial cropping/resampling.
- `Baseline.h5` (G=D=Y=0) is included in JEPA `train` (encounters 0-3) and `test_a` (encounters 4-5) like other periodic cases, with `is_calibration_reference: true` set on the per-case metadata for tools that need to identify the no-gust reference.

The sections below describe the original plan as a historical record; deviations are documented in `configs/preprocessing.yaml`.

## Goal

Produce a versioned, append-only cache of preprocessed per-encounter data containing:

1. Mid-plane spanwise vorticity omega_z(x, y, t)
1. Wall pressure probes p_wall(t, n_probes) with spanwise-averaging tolerance
1. Aerodynamic coefficients C_L, C_D, C_M (if derivable)
1. Per-encounter metadata (G, D, Y, encounter_index, impact_frame, partition_version)

The cache supports incremental updates: when PREVENT adds new run3 cases, they go into a
NEW partition version (`v2`, `v3`, …) without modifying the previous one. Models
trained on v1 stay reproducible against v1; new models can be trained on v2 and
compared.

## Locked design

|Decision                     |Value                                                                                   |Rationale                                                                     |
|-----------------------------|----------------------------------------------------------------------------------------|------------------------------------------------------------------------------|
|Cache location               |`${VORTEX_JEPA_CACHE}/v{N}/`, default `${PREVENT_ROOT}/data/processed/vortex-jepa/v{N}/`|Configurable for fast-SSD storage; defaults near raw data                     |
|File format per encounter    |HDF5 with chunked compression                                                           |Reads slice-wise; small enough for full-trajectory loading                    |
|Spanwise tolerance for probes|+/- 5 percent of L_z (configurable, default 0.05)                                       |Suppresses turbulent fluctuations without smearing 2D coherent structure      |
|Probe count                  |64 equispaced along the airfoil contour                                                 |Dense enough for full p(x) reconstruction; subset later for sparse-sensor work|
|Partition policy             |Immutable once frozen                                                                   |Reproducibility; new cases force a new version                                |
|Naming                       |`case_id` from the inventory plus `encounter_{idx:02d}.h5`                              |Maps 1:1 to the split manifest                                                |

## Step 0: Schema inspection (must run first)

The raw HDF5 internal layout is not yet known to vortex-jepa. Before any preprocessing
is defined, run `scripts/inspect_raw_hdf5.py` against one periodic file and one run3
file. Required outputs:

1. Full HDF5 tree (h5dump-style summary): groups, datasets, shapes, dtypes, chunking
1. List of physical variables present (omega_x, omega_y, omega_z, u, v, w, p, …)
1. Coordinate arrays: x, y, z (or i, j, k indices). Identify the spanwise axis and the
   value of L_z
1. Time axis: frame stride dt, total frames, alignment of frame 0 with gust launch
1. Whether wall pressure is stored directly on the airfoil surface or must be
   interpolated from the volume
1. Whether forces (C_L, C_D, C_M) are stored as time series or must be integrated
1. Mesh type: body-fitted versus immersed boundary; airfoil surface representation

Output goes to `outputs/schema_inspection/{periodic,run3}.txt` and a structured
`outputs/schema_inspection/schema.yaml` that downstream scripts will read.

This step is needed because the rest of the pipeline depends on:

- Which variable name holds spanwise vorticity (`omega_z`, `vorticity_z`, `wz`, …)
- The spanwise axis position in the array (axis 0, 1, 2, or 3)
- The mid-span index, computed as `argmin(abs(z))` if z is signed and centered, or
  `n_z // 2` otherwise
- Whether wall pressure is available without interpolation

## Step 1: Mid-plane vorticity extraction

Input: raw HDF5 with 3D omega_z field.
Output: `cache/v1/{case_id}/encounter_{k:02d}.h5/omega_z` of shape (120, H, W) in
float32, chunked along the time axis.

Algorithm:

```python
# Per encounter k = 0, 1, ..., n_encounters_full - 1
frames_in_encounter = range(k * 120, (k + 1) * 120)
omega_full = raw[variables.omega_z]    # (T, ny, nx, nz) or similar
mid = argmin(abs(z))                   # spanwise midplane index
omega_mid = omega_full[frames_in_encounter, :, :, mid]   # (120, H, W)
# Optional: crop to bounding box around airfoil + wake; see Step 1b
```

Step 1b: spatial cropping. Fukami JFM 2025 uses (480, 200). Decide on a fixed crop
window in physical (x/c, y/c) coordinates that captures the airfoil, the gust path,
and the near wake. Default proposal: x/c in [-2.5, 3.0], y/c in [-1.0, 1.0], with
linear interpolation onto a uniform 480 x 200 grid if the DNS mesh is non-uniform.
The crop window is a configuration parameter, frozen at partition creation time.

Sanity check (must pass before continuing): for the baseline case, frame 0 of
encounter 0 should show a clean shedding state with C_L close to the time-averaged
value. For a strong gust case (G = 4.0, D = 1.0, Y = +0.10), frame ~40 should show
the vortex impacting the leading edge.

## Step 2: Wall pressure probes with spanwise tolerance

Input: raw HDF5 with 3D pressure field (or precomputed wall-pressure tables).
Output: `cache/v1/{case_id}/encounter_{k:02d}.h5/p_wall` of shape (120, n_probes).

Algorithm:

```python
# Define 64 probe locations equispaced in arclength along the airfoil contour.
# Get airfoil surface coordinates at z = z_mid (e.g., NACA 0012 analytic surface
# rotated by alpha = 14 deg).
probe_xy = sample_airfoil_arc(n_probes=64)       # (64, 2)

# Spanwise window
delta_z = tolerance_frac * L_z                   # default 0.05 * L_z
z_lo, z_hi = z_mid_value - delta_z, z_mid_value + delta_z
z_window = where((z >= z_lo) & (z <= z_hi))      # spanwise indices

# For each probe, find the nearest wall-adjacent grid cell in the mid-plane.
# Average pressure across the spanwise window.
for i, (xp, yp) in enumerate(probe_xy):
    j, k = nearest_wall_cell(xp, yp)              # (j, k) in (y, x) indices
    p_wall[:, i] = p_full[:, j, k, z_window].mean(axis=-1)
```

Probe sanity: for the baseline (no gust), p_wall on the suction side near x/c = 0.3
should show low-frequency oscillation at the shedding Strouhal number. For a strong
gust impacting the leading edge, the leading-edge probe should show a sharp spike
at frame ~40.

Tolerance choice: 5 percent of L_z is conservative. If schema inspection shows
L_z = 0.5c (typical for NACA at this Re), this is +/- 0.025c spanwise extent,
roughly 6 to 10 grid points depending on spanwise resolution.

## Step 3: Aerodynamic coefficients

If C_L, C_D, C_M are stored as time series in the HDF5, read directly. Verify they
are spanwise-averaged over the same window used for probes.

If not, integrate surface stress in the spanwise window:

```python
# Pressure contribution: -p * n_hat dot d, integrated along arc
# Viscous contribution: tau dot d, requires velocity gradients at the wall
# Both are spanwise-averaged over the tolerance window
```

If viscous integration is not feasible from the cached fields, use the pressure-only
approximation and flag this in the encounter metadata as `cl_method: "pressure_only"`.
At Re = 5000 the viscous contribution to C_L is small but non-negligible (a few
percent), so this matters for the observable-augmented baselines.

## Step 4: Per-encounter metadata

Each `encounter_{k:02d}.h5` file carries an attribute dictionary:

```python
{
    "case_id": "G+1.00_D0.50_Y+0.10",
    "G": 1.0, "D": 0.5, "Y": 0.10,
    "encounter_index": 0,
    "source_group": "periodic",
    "n_frames": 120,
    "dt_tc": 0.05,
    "spanwise_tolerance_frac": 0.05,
    "n_probes": 64,
    "probe_arc_positions": [...],     # (64,) arclength fractions in [0, 1]
    "crop_window_xc": [-2.5, 3.0],
    "crop_window_yc": [-1.0, 1.0],
    "grid_shape": [200, 480],         # (ny, nx)
    "impact_frame_estimate": 40,
    "cl_method": "pressure_and_viscous",   # or "pressure_only"
    "raw_file_sha256": "...",
    "preprocessing_version": "1.0.0",
    "partition_version": "v1",
}
```

## Step 5: Partition versioning system

Two artifacts per partition, both stored in the vortex-jepa repo:

1. `configs/partitions/v{N}.yaml`: the partition registry
1. `configs/splits/split_v{N}.json`: the split manifest (generated by
   `scripts/build_split_manifest.py`)

Both versioned together. `v1` is the current 41-case partition. Adding new run3 cases
creates `v2`.

### Partition registry schema (`configs/partitions/v{N}.yaml`)

```yaml
partition_version: v1
created_iso: 2026-05-15T...
inventory_sha256: 290c2af702e1d510...   # matches configs/raw_cases_inventory.yaml
preprocessing_version: 1.0.0
cache_layout:
  spanwise_tolerance_frac: 0.05
  n_probes: 64
  crop_window_xc: [-2.5, 3.0]
  crop_window_yc: [-1.0, 1.0]
  grid_shape: [200, 480]
case_ids:           # ordered list of all case_ids in this partition
  - Baseline
  - G+1.00_D0.50_Y+0.10
  - ...
  - G+4.00_D1.00_Y-0.10
inherits_from: null                     # v2 will set this to "v1"
new_in_this_partition: []               # v2 will list its added case_ids
```

### Addition workflow

```bash
# When PREVENT adds new run3 cases, e.g., Gust_023, Gust_024:
python scripts/add_cases.py \
    --new-inventory $PREVENT_ROOT/configs/raw_cases_inventory_v2.yaml \
    --target-partition v2 \
    --inherits-from v1
```

The script:

1. Verifies that all v1 case_ids are still present in the new inventory and that their
   `relative_path` and SHA256 (if available) are unchanged. Refuses to proceed otherwise.
1. Lists new case_ids not in v1.
1. Auto-assigns new cases to splits:
- If G == +4: Test C
- If Test B is unchanged in v2: train (with last 1 or 2 encounters reserved for Test A)
- Test B can optionally be expanded in v2 via a manual selection in the CLI
1. Preprocesses ONLY the new cases (skips already-cached encounters from v1).
1. Writes `configs/partitions/v2.yaml` with `inherits_from: v1` and the new case list.
1. Writes `configs/splits/split_v2.json` via `build_split_manifest.py`.
1. Commits the registry and split changes to the vortex-jepa repo.

### Immutability rules

- Once `configs/partitions/v{N}.yaml` is committed, it is never edited. Updates go in
  `v{N+1}`.
- Preprocessed cache files under `cache/v{N}/` are never overwritten. If preprocessing
  parameters change (different tolerance, different probe count), bump
  `preprocessing_version` and write to `cache/v{N+1}/`.
- Models record their partition version. A v1-trained checkpoint can be evaluated
  on v1 only; running it on v2 requires explicit acknowledgement and is logged as a
  cross-partition evaluation.

### Reading code

The dataset loader takes a partition version:

```python
from src.data.episode_dataset import EpisodeDataset

ds = EpisodeDataset(
    partition="v1",                     # or "v2", "v3", ...
    split="train",                      # or "test_a", "test_b", "test_c", "baseline"
    prevent_root=os.environ["PREVENT_ROOT"],
    cache_root=os.environ.get("VORTEX_JEPA_CACHE",
                              f"{os.environ['PREVENT_ROOT']}/data/processed/vortex-jepa"),
    subtraj_len=32,
    impact_aware_fraction=0.7,
    return_pressure=True,
    return_forces=True,
)
```

If a partition’s cache is missing on disk, the loader prints a clear remediation
message: “Preprocess partition v1 first via `python scripts/preprocess.py partition=v1`”.

## Step 6: Quality checks (must pass before declaring v1 frozen)

Implemented in `notebooks/00_qc_partition_v1.ipynb`. Checks:

1. **Cache integrity**: every `case_id` declared in `configs/partitions/v1.yaml` has the
   expected number of encounter files; every encounter file has the expected shapes
   and attribute dictionary.
1. **Impact frame verification**: for each non-baseline encounter, compute
   `argmax(|omega_z| under airfoil, frame range [20, 60])`. Histogram the result. The
   mode should fall within +/- 3 frames of the declared `impact_frame_estimate = 40`.
   If the mode is more than 5 frames off, update the split manifest before training.
1. **Probe consistency**: for the baseline encounter (no gust), all probes should show
   stationary statistics with shedding-frequency content. For a strong impact case,
   the leading-edge probe should peak at the impact frame. Plot one of each.
1. **Force consistency**: compare C_L from the cache against an independent source if
   available (e.g., a separately stored PREVENT-side forces file). They should match
   within numerical noise (< 1 percent RMS).
1. **Spanwise tolerance sensitivity**: re-extract probes at three tolerance fractions
   (0.02, 0.05, 0.10) for one representative encounter; check that the impact-frame
   peak height changes by less than 5 percent. If sensitivity is high, the tolerance
   is too small or the flow is locally 3D and the assumption of 2D mid-plane is
   weaker than expected.
1. **Smoke test of the loader**: instantiate `EpisodeDataset(partition="v1", split="train")`, iterate one full epoch with the impact-aware sub-trajectory
   sampler, confirm that 70 percent of yielded sub-trajectories contain frames
   in [25, 55] and 30 percent are uniformly distributed.

## Concrete run sequence (Carlos’s workstation)

```bash
# 0. Environment
export PREVENT_ROOT=$HOME/PREVENT
export VORTEX_JEPA_CACHE=$HOME/PREVENT/data/processed/vortex-jepa
# If a separate fast SSD is preferred:
# export VORTEX_JEPA_CACHE=/mnt/extreme_ssd/vortex-jepa-cache

# 1. Schema inspection
python scripts/inspect_raw_hdf5.py \
    --periodic-sample $PREVENT_ROOT/data/raw/periodic/Baseline.h5 \
    --run3-sample $PREVENT_ROOT/data/raw/run3/Gust_???.h5 \
    --output outputs/schema_inspection/

# 2. Edit configs/preprocessing.yaml based on the schema (variable names, axis order).
#    Commit this config.

# 3. Preprocess partition v1
python scripts/preprocess.py partition=v1 --dry-run     # plan
python scripts/preprocess.py partition=v1               # execute
# Expected runtime: ~5-15 min per case depending on I/O speed; 41 cases sequentially.
# Use --jobs N for parallel processing (each job is I/O-bound, 4-8 jobs is reasonable).

# 4. QC
jupyter notebook notebooks/00_qc_partition_v1.ipynb
# Run all cells. All checks must pass before continuing.

# 5. Build split manifest (already done as split_v1.json, this step is for v2+)
python scripts/build_split_manifest.py partition=v1

# 6. Smoke-test the loader
python -c "
from src.data.episode_dataset import EpisodeDataset
ds = EpisodeDataset(partition='v1', split='train')
print(f'{len(ds)} sub-trajectories available')
batch = ds[0]
print({k: v.shape if hasattr(v, 'shape') else v for k, v in batch.items()})
"
```

When PREVENT later adds run3 cases (e.g., Gust_023, Gust_024, …):

```bash
# 0. New inventory provided by PREVENT
# 1. Add cases to v2
python scripts/add_cases.py \
    --new-inventory $PREVENT_ROOT/configs/raw_cases_inventory_v2.yaml \
    --target-partition v2 \
    --inherits-from v1
# This handles preprocessing of new cases only and writes:
#   - configs/partitions/v2.yaml
#   - configs/splits/split_v2.json
#   - cache/v2/{new_case_ids}/...

# 2. QC the new partition
jupyter notebook notebooks/00_qc_partition_v2.ipynb

# 3. Train on v2
python scripts/train_jepa.py data.partition=v2
```

## What gets committed to the vortex-jepa repo

- `configs/preprocessing.yaml` (schema-derived, frozen per preprocessing version)
- `configs/partitions/v1.yaml`, `configs/partitions/v2.yaml`, …
- `configs/splits/split_v1.json`, `configs/splits/split_v2.json`, …
- `configs/raw_cases_inventory.yaml` (and v2, v3, … as PREVENT regenerates)
- All scripts under `scripts/` and code under `src/data/`

What does NOT get committed:

- The cache itself (`cache/v{N}/*.h5`). It lives at `${VORTEX_JEPA_CACHE}/v{N}/` and
  is rebuildable from the raw data plus the configs.
- Raw DNS data (lives under `${PREVENT_ROOT}/data/raw/`).
- Notebook outputs (clear all outputs before committing).

## Unknowns to resolve in Step 0 (questions for the researcher)

These should be answered by inspecting one periodic and one run3 file. If you can run
`scripts/inspect_raw_hdf5.py` and paste the schema, the rest of the pipeline gets
written without ambiguity:

1. What is the HDF5 internal layout? Are vorticity, velocity, pressure stored at every
   frame, or as separate fields?
1. What is the spanwise extent L_z and how many spanwise grid points?
1. Is the mesh body-fitted around the airfoil, or immersed boundary on a Cartesian
   block? This determines how wall pressure probes are extracted.
1. Are forces (C_L, C_D, C_M) precomputed and stored, or do we need to integrate?
1. What is the variable naming convention? (e.g., `vorticity/z`, `omega_z`, `wz`)
1. Are coordinate arrays stored explicitly, or implied by metadata?
1. Is the airfoil rotated by alpha = 14 deg in the mesh frame, or is the freestream
   tilted? This affects the airfoil surface coordinates used for probe placement.
1. What is the typical encounter file size on disk? This sets the preprocessing
   throughput target and informs whether parallelism is needed.

## Next actions

1. Run `scripts/inspect_raw_hdf5.py` on one periodic and one run3 file.
1. Paste the output here for review.
1. We then write `configs/preprocessing.yaml` and `scripts/preprocess.py` with the
   schema baked in.
1. Run preprocessing for v1 and the QC notebook.
1. Move to training (CLAUDE.md).
