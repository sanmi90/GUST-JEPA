# CLAUDE.md

Project-level instructions for Claude Code. Read this at the start of every session.

## Project: vortex-jepa

End-to-end Joint-Embedding Predictive Architecture (JEPA) for parametric vortex-gust airfoil
interactions at Re = 5000. Target deliverable: a scientific paper competitive with
Fukami et al. (Phys. Rev. Fluids 10, 084703, 2025; J. Fluid Mech. 1021, A39, 2025) and
Solera-Rico et al. (Nat. Commun. 15, 1361, 2024) on forecasting horizon and probing R^2 at
matched latent dimension.

Lead researcher: Carlos Sanmiguel Vila (INTA, UC3M).

## What we are building

End-to-end JEPA inspired by LeWM (Maes et al., arXiv:2603.19312, 2026) and LeJEPA
(Balestriero and LeCun, arXiv:2511.08544, 2025), trained on DNS of NACA 0012 at
alpha = 14 deg, Re = 5000, perturbed by Taylor vortices parametrized by (G, D, Y/c).

Input modalities: mid-plane 2D vorticity field omega_z, plus wall pressure for a later
sparse-sensor estimator (deferred).

Priorities:
1. PRIMARY: impact-instant generalization (forecast latent trajectories across held-out
   shedding phases within seen (G, D, Y) cases).
2. Parametric interpolation in (G, D, Y) within the training envelope.
3. Latent disentanglement of physical effects.

## Locked decisions (do not revisit without explicit user approval)

Architecture
- Encoder input: omega_z at native cache resolution (192, 96), single channel
  (mid-plane spanwise vorticity from `/curlU[..., 16, 2]`).
- Encoder: hybrid CNN stem (~3M params, 3 downsampling stages -> 24 x 12 feature map
  at 256 channels = 288 spatial tokens) + 6-layer ViT (~7M params, hidden 256, 8 heads),
  d = 32 latent via [CLS]-token + 1-layer MLP projection with BatchNorm (NOT LayerNorm).
- Predictor: 6-layer autoregressive transformer, hidden 384, 16 heads, dropout 0.1,
  AdaLN-Zero conditioning on (G, D, Y, phi_t), RoPE temporal positions, causal mask.
- Conditioning: c = (G, D, Y) enters ONLY the predictor; encoder is unconditional.
- Visualization decoder: trained separately on frozen encoder, NEVER part of JEPA loss.

Training
- Loss: L_pred (teacher forcing) + 0.5 * L_roll (scheduled sampling, H_roll = 8)
  + 0.1 * SIGReg(Z). No EMA, no stop-gradient on target encoder.
- Anti-collapse default: SIGReg with M = 256 projections, 17 Epps-Pulley knots in [0.2, 4].
  Auto-fallback to VICReg if participation ratio PR(z) < 0.3 * d at iteration 20k AND
  linear probe R^2 for c < 0.7.
- Optimizer: AdamW (0.9, 0.95), weight decay 0.05, linear warmup 5% + cosine to
  0.05 * peak LR. Encoder LR 1.5e-4, predictor LR 5e-4. bf16 mixed precision.
  Gradient clip 1.0. 80k iterations.

Data
- Split is locked at `split_v1.json` at the repo root (sha256-anchored to inventory).
- 31 train cases (108 encounters), 6 Test B cases (28 enc), 4 Test C cases (24 enc).
  Baseline (no gust) is in `train` (encounters 0-3) and Test A (encounters 4-5) like
  any other periodic case; it is also flagged `is_calibration_reference: true` so
  calibration tooling can still identify the no-gust reference. Within training cases,
  Test A holds last 2 of 6 (periodic) or last 1 of 4 (run3) encounters:
  46 encounters total.
- |G| = 3 stays in training. Test C is G = +4 only. Periodic trailing partials discarded.
- Impact frame ~ 40 (vortex centroid crosses LE at t ~ 1.965 t/c). QC across the cached
  partition v1: vorticity argmax mean = 40.8, force argmax mean = 38.8 over [25, 55].
- Sub-trajectory L = 32 with 70% impact-aware sampling, 30% uniform.

## Dataset layout

**Raw DNS data lives in an EXTERNAL project, not inside vortex-jepa.** The data is owned
by the PREVENT project (Carlos's ML turbulence detection effort that produced these DNS
runs). The vortex-jepa repository contains only code, configs, and the split manifest;
it does not duplicate the raw HDF5 files.

Path resolution
- Set `PREVENT_ROOT` to the PREVENT project root (the directory that contains `data/`).
  Example: `export PREVENT_ROOT=$HOME/PREVENT`.
- Full path to a case file is `${PREVENT_ROOT}/${case.relative_path}` where
  `case.relative_path` is taken from `split_v1.json` (for example
  `data/raw/periodic/Baseline.h5` or `data/raw/periodic/run3/Gust_???_*.h5`).
- In Hydra configs, declare `data.prevent_root: ${oc.env:PREVENT_ROOT,~/PREVENT}` and
  resolve case paths in the dataset loader via
  `Path(cfg.data.prevent_root) / case["relative_path"]`.
- Do NOT add a `data/raw/` directory or symlink under vortex-jepa. Keeping the data
  external avoids accidental commits and lets multiple consumers (PREVENT, vortex-jepa,
  any future project) share one source of truth.

Cache layout
- Preprocessed per-encounter cache lives at
  `${VORTEX_JEPA_CACHE}/{partition_version}/{case_id}/encounter_{k:02d}.h5`.
  Default `VORTEX_JEPA_CACHE = ${PREVENT_ROOT}/data/processed/vortex-jepa`.
- Each encounter file holds `omega_z (120, 192, 96)`, `p_wall (120, 192)`,
  `C_L (120,)`, `C_D (120,)` plus 17 attrs (case_id, G, D, Y, source_group,
  encounter_index, frame_start/end, dt_tc, impact_frame_estimate, mid_span_index,
  omega_z_sign_convention, preprocessing_version, partition_version,
  raw_relative_path, n_frames).
- Partition layout is frozen at creation; bump `preprocessing_version` or
  `partition_version` to introduce changes. See `configs/preprocessing.yaml`.

Inventory and parser
- The inventory at `data_manifest/raw_cases_inventory.yaml` (a copy of the PREVENT-side
  manifest at the time the project was bootstrapped) parses filenames into (G, D, Y)
  via the alpha = 14 deg rotation specified in `parser.formula_inverse`.
- Example filename: `Gust_001_x-1.965_y-0.387_s1.0_d0.5.h5` decodes to
  (G = +1.0, D = 0.5, Y = +0.10).
- If PREVENT regenerates its inventory, copy the new YAML into `data_manifest/` and
  re-run `python build_split_manifest.py` to refresh `split_v1.json`.

Per-file structure
- Each case is a single HDF5 with 480 (run3) or 800 (periodic) frames at dt_tc = 0.05.
- Gust released every 120 frames (one "encounter" = one episode at t = 0).
- Periodic trailing partials (the 80-frame remainder after 6 full encounters) are
  discarded by the loader.
- Schema notes (resolved in Step 0, see `outputs/schema_inspection/schema.yaml`):
  velocity at `/u` shape `(T, 192, 96, 32, 3)` with component order `(u_x, u_y, u_z)`,
  vorticity at `/curlU` (omega_z is index 2, sign convention `du/dy - dv/dx`),
  wall pressure at `/sensors/p` shape `(1536, T) = (192 surface pts) x (8 z-stations)`
  (inner axis is z; spanwise averaging is `reshape((192, 8, T)).mean(axis=1)`),
  forces at `/forces/{CL,CD}` already non-dimensionalized (CL = 2 * lift exact).
  C_M is not stored; integrate over `/airfoil_xy` if needed.
  `/u` and `/curlU` carry NaN in the 2624 cells where `/inside_solid > 0`; the cache
  fills these with 0.

Sanity check on first run
- The data loader must verify that `${PREVENT_ROOT}/data/raw/periodic/Baseline.h5`
  exists and is readable before any training run starts. Fail fast with a clear
  message if `PREVENT_ROOT` is unset or the path is missing.

## Baselines to implement (matched latent dimension)

For paper-grade comparison at matched d:

1. POD with d modes (linear floor)
2. Fukami observable-augmented AE (PRF 2025 / JFM 2025 recipe) with C_L augmentation
3. Solera-Rico beta-VAE + transformer (Nat. Commun. 2024 recipe)
4. PLDM (Sobal, Jyothir, Jalagam, Carion, Cho, LeCun, arXiv:2211.10831, 2022;
   stress-tested in Sobal et al. 2025, "Stress-testing offline reward-free RL")

PLDM is the direct end-to-end JEPA-from-pixels precursor to LeWM, with a 7-term
VICReg-derived objective (six tunable weights). The central methodological contrast
the paper owns is "SIGReg + 2-term (proposed)" vs "VICReg + 7-term (PLDM)": simpler
anti-collapse and O(log n) bisection vs PLDM's O(n^6) grid search.

## Repository structure

```
vortex-jepa/
├── CLAUDE.md                            # this file
├── HANDOFF.md                           # decision history and session handoff
├── README.md
├── SESSION_DATA_PREP.md                 # preprocessing plan (with Step 0 status section)
├── SESSION_REPORT_2026-05-15.md         # report from the bootstrap session
├── requirements.txt
├── build_split_manifest.py              # regenerates split_v1.json from the inventory
├── split_v1.json                        # locked split manifest
├── configs/
│   └── preprocessing.yaml               # schema-baked preprocessing params (v1.0.0)
├── data_manifest/
│   └── raw_cases_inventory.yaml         # data parser manifest (do not edit by hand)
├── scripts/
│   ├── 100c_raw_cases_inventory.py      # regenerates the inventory from raw filenames
│   ├── inspect_raw_hdf5.py              # Step 0 schema inspector
│   └── preprocess.py                    # extracts per-encounter cache (omega_z, p_wall, C_L, C_D)
├── src/
│   └── data/
│       ├── __init__.py
│       └── episode_dataset.py           # PyTorch Dataset with impact-aware sampler
├── notebooks/
│   └── 00_qc_partition_v1.ipynb         # QC: cache integrity + impact-frame + sanity plots
├── outputs/                             # gitignored: schema_inspection/, checkpoints/, logs/, figures/
└── .venv/                               # gitignored
```

Planned but not yet created (added when the corresponding step is reached):
- `configs/encoder/`, `configs/predictor/`, `configs/loss/`, `configs/data/`, `configs/sweep/`
- `src/models/{encoder,predictor,decoder,adaln,rope,sigreg,vicreg,jepa}.py`
- `src/baselines/{pod,fukami_ae,solera_rico,pldm}.py`
- `src/training/{train_jepa,train_decoder,train_baseline,scheduler,scheduled_sampling,diagnostics}.py`
- `src/evaluation/{reconstruction,forecasting,probing,surprise,visualization}.py`
- `scripts/{train_jepa,train_baseline,sweep_lambda,evaluate_paper}.py`
- `tests/`

The repo intentionally contains NO `data/` directory. Raw DNS data lives at
`${PREVENT_ROOT}/data/raw/periodic/` and `${PREVENT_ROOT}/data/raw/periodic/run3/`
outside this repo. See "Dataset layout" above.

## Coding conventions

- Python 3.10+, PyTorch 2.x, Hydra for configs, W&B for logging
- ruff + black --line-length 100 + mypy --strict on src/models
- pytest for unit tests; minimum suite:
  - `test_sigreg.py`: Epps-Pulley vs scipy.stats.normaltest on Gaussian samples
  - `test_adaln_zero.py`: predictor is identity at initialization
  - `test_encoder_shapes.py`: HybridCNNViTEncoder I/O contracts at common resolutions
  - `test_predictor_causal.py`: future frames cannot leak into past predictions
  - `test_splits.py`: split_v1.json round-trips through the loader
- All random sources seeded (torch, numpy, random, torch.cuda); seed logged in every run
- bf16 mixed precision on the user's RTX 6000 96 GB (single GPU is sufficient)
- Type hints everywhere in `src/`; Google-style docstrings

## Logging (W&B)

- W&B is the primary logger (`wandb` in `requirements.txt`).
- Set `WANDB_PROJECT=vortex-jepa` in the environment before any training run; export it
  or place it in a local `.env` that the training entrypoint loads.
- Every run must log the following keys so it can be traced back to a frozen manifest:
  - `preprocessing_version`     (from `configs/preprocessing.yaml`)
  - `partition_version`         (e.g. `v1`)
  - `lambda`                    (SIGReg / VICReg weight; null until tuned)
  - `seed`                      (full deterministic seed for the run)
- Recommended additional fields:
  - `split_sha256`              (sha256 of `split_v1.json`)
  - `inventory_sha256`          (from `split_v1.json` -> `source_inventory.sha256`)
  - `wandb_run_id`              (echoed back to stdout and to W&B summary)
- W&B group: `partition_v{N}` (e.g. `partition_v1`) so all runs on the same partition
  cluster together in the UI.
- Tags: baseline runs use the baseline name (`pldm`, `fukami_ae`, `solera_rico`, `pod`).
  Ablations use `ablation:<name>` (e.g. `ablation:sigreg_off`, `ablation:c_in_encoder`).
- A run that does not log all four required keys is considered untraceable and should
  not be used in the paper.

## Writing style (any prose, papers, docs)

- No em-dashes (user preference)
- Direct, technical, honest about failure modes
- Avoid bullet lists in formal prose unless explicitly requested
- Cite by author/year/venue or arXiv ID

## Common commands

```bash
# Required: point at the PREVENT project root where the raw DNS data lives
export PREVENT_ROOT=$HOME/PREVENT          # adjust to your machine

# (Done) Preprocessing pipeline for partition v1
python scripts/inspect_raw_hdf5.py \
    --periodic-sample $PREVENT_ROOT/data/raw/periodic/Baseline.h5 \
    --run3-sample    $PREVENT_ROOT/data/raw/periodic/run3/Gust_002_x-1.916_y-0.581_s-3.0_d1.5.h5 \
    --output outputs/schema_inspection/

python scripts/preprocess.py --partition v1
# Use --dry-run to plan; --cases <ids> to subset; --force to overwrite.

jupyter nbconvert --to notebook --execute --inplace notebooks/00_qc_partition_v1.ipynb

# Regenerate the split manifest after editing the inventory
python build_split_manifest.py

# (Planned) Train JEPA
python scripts/train_jepa.py
python scripts/train_jepa.py model.encoder.latent_dim=64 loss.lambda_sigreg=0.05

# (Planned) Lambda bisection over [0.001, 1.0]
python scripts/sweep_lambda.py

# (Planned) Train a baseline
python scripts/train_baseline.py baseline=pldm
python scripts/train_baseline.py baseline=fukami_ae
python scripts/train_baseline.py baseline=solera_rico
python scripts/train_baseline.py baseline=pod d=32

# (Planned) Train the visualization decoder on a frozen JEPA checkpoint
python scripts/train_decoder.py jepa_checkpoint=outputs/checkpoints/jepa_v1.pt

# (Planned) Full paper evaluation suite
python scripts/evaluate_paper.py checkpoint=outputs/checkpoints/jepa_v1.pt

# Tests and style (planned, no suite yet)
pytest tests/
ruff check src/
black --check src/
mypy --strict src/models
```

## Risk-management (must be implemented)

The single biggest risk: SIGReg may fail on this low-intrinsic-dim (~5 to 10) physics
dataset. LeWM Two-Room results show SIGReg underperforming on low-intrinsic-dim
environments.

Mandatory diagnostics computed every 1k iterations on a held-out batch
(implemented in `src/training/diagnostics.py`):
- Participation ratio PR = (sum_i s_i)^2 / sum_i s_i^2 of singular values of {z_t}.
- Per-dimension variance histogram of z.
- Linear probe R^2 for (G, D, Y) from z_T on Test B sub-batch.
- Decoded MSE on a fixed Test A held-out encounter for visual sanity.

Auto-fallback rule (hard-coded in `src/training/train_jepa.py`):
- If iteration >= 20k AND PR < 0.3 * d AND probe R^2 for c < 0.7:
  switch SIGReg to VICReg with mu = 25.0, nu = 1.0 (Bardes, Ponce, LeCun, ICLR 2022).
  Log this event prominently to W&B and stdout. Continue training; do not restart.

## Things to NOT do

- Do not add reconstruction loss to the JEPA encoder objective. The visualization decoder
  is a separate stage on a frozen encoder.
- Do not condition the encoder on c. The encoder is unconditional by design (D6 in
  HANDOFF.md). The ablation that adds c to the encoder is a deliberate negative-result run.
- Do not random-split impact events within a case. Contiguous holdout only.
- Do not stratify Test B by source group. Pool periodic and run3.
- Do not touch Test C (G = +4 cases) for model selection. Reported only at the end.
- Do not edit `split_v1.json` by hand. Regenerate via `python build_split_manifest.py`.
- Do not copy, symlink, or commit raw DNS data into this repo. The data is owned by the
  PREVENT project and accessed via the `PREVENT_ROOT` environment variable. The repo
  must remain code-and-config only.
- Do not use em-dashes in any output document.
- Do not use LayerNorm at the encoder latent boundary. SIGReg requires BatchNorm
  (LeWM appendix, see Section 3.1 of the architecture spec in HANDOFF.md).

## Where to find more detail

- `HANDOFF.md`: decision log with rationales, open questions, suggested next steps
- `SESSION_DATA_PREP.md`: preprocessing plan plus Step 0 schema findings
- `SESSION_REPORT_2026-05-15.md`: bootstrap-session report (what landed, what was verified)
- `split_v1.json`: locked data split with rationales as inline keys
- `data_manifest/raw_cases_inventory.yaml`: data parser manifest
- `configs/preprocessing.yaml`: schema-baked preprocessing params (v1.0.0)
- `outputs/schema_inspection/schema.yaml`: raw HDF5 schema as inspected
