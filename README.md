# vortex-jepa

Analysis and evaluation code for the manuscript "A predictive latent
representation renders wake structure linearly readable in vortex-gust airfoil
interactions at Re=5000" (Solera-Rico, Miro, Lehmkuhl, Sanmiguel Vila; under
review at the Journal of Fluid Mechanics).

The study compares reduced states for parametric vortex-gust airfoil
interactions at Re=5000, a joint-embedding predictive architecture (JEPA) trained
with no gust parameters anywhere in the model, a reconstructive
observable-augmented autoencoder, a beta-variational baseline, and a POD basis,
under one shared predictor and probe family, judged by held-out representational
closure of six wake observables and by recovery from sparse wall pressure.

## What is in this deposit

- `src/`: encoder, predictor, anti-collapse regulariser, data pipeline.
- `scripts/`: training, evaluation, and analysis scripts (per session). The v2.1
  analysis that produces every paper number lives under `scripts/session28/`.
- `configs/splits/split_v2p1.json`: the locked 85-case v2.1 data-split manifest
  (the paper runs on v2.1). The omega preprocessing manifest is
  `outputs/data_pipeline/v2p1/manifest.json`.
- `outputs/session28/`: the committed analysis summaries that hold the numbers
  behind every table and figure. The single source of truth is
  `outputs/session28/numbers.json` (provenance-stamped), merged from the
  per-analysis records in `outputs/session28/numbers_parts/` and rendered to
  `paper/macros.tex`; every printed number in the manuscript is a macro that
  traces back to that file.
- `paper/`: the LaTeX manuscript sources and the figure PDFs under
  `paper/sections/figures/results/` (all on the v2.1 rebuild, `_v2p1` suffix).

## What is NOT in this deposit

- The raw direct numerical simulation data, computed with the SOD2D solver
  (Gasparino, Spiga and Lehmkuhl 2024) and owned by the simulation collaborators.
- The large cached per-encounter arrays, latents, and checkpoints
  (`*.npz`, `*.npy`, `*.h5`, `*.pt`), which are gitignored. The processed
  per-encounter cache and the cached latents, or a representative subset, can be
  made available to reproduce the heavier steps; contact the corresponding author.

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PREVENT_ROOT=$HOME/PREVENT          # root that contains the processed cache
export VORTEX_JEPA_CACHE=$PREVENT_ROOT/data/processed/vortex-jepa
```

GPU steps (encoder training, decoding) target an RTX 6000 Blackwell (sm_120); the
number pipeline below is CPU-only and needs no training and no raw data.

## Reproducing every printed number (CPU, no data needed)

Each paper number is a record in `outputs/session28/numbers_parts/<analysis>.json`
carrying its value, formatting, split, and a provenance note. The two steps below
merge and render them deterministically:

```bash
python scripts/session28/eval_all.py     # validate + merge numbers_parts -> numbers.json
python scripts/session28/emit_macros.py   # numbers.json -> paper/macros.tex
```

`eval_all.py` refuses duplicate names or macros and stamps the source commit, so a
number that is not in `numbers.json` cannot appear in the paper. The analysis
scripts that compute those records from the cached latents (CPU-only sklearn for
the closure matrix, the wall-pressure observability, the statistics, and the
physics tracks) are under `scripts/session28/`, for example:

```bash
python scripts/session28/closure_matrix.py   # held-out representational/forecast closure (headline)
python scripts/session28/sensing_cf.py        # sparse wall-pressure state + field recovery
python scripts/session28/stats_harvest.py     # case-clustered bootstrap, sign tests, Holm correction
```

These read the cached latents under `outputs/session28/latents/` (available on
request) and rewrite their `numbers_parts/` records, which `eval_all.py` then
re-merges.

## Building the manuscript

```bash
cd paper && latexmk -pdf -interaction=nonstopmode main.tex
```

## Citation

See `CITATION.cff`. Please cite both the software (this deposit) and the paper.

## License and DOI

Code and analysis artifacts are released under the MIT License (see `LICENSE`),
proposed pending author and institutional confirmation. The raw DNS data are
excluded and collaborator-owned. A Zenodo DOI will be minted from the
release-candidate tag and inserted here and in the paper's data-availability
statement: DOI `10.xxxx/zenodo.PLACEHOLDER`.
