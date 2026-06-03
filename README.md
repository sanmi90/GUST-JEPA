# vortex-jepa

Analysis and evaluation code for the manuscript "Forward physical closure of
predictive and reconstructive latents for vortex-gust airfoil interactions at
Re=5000" (Solera-Rico, Miro, Lehmkuhl, Sanmiguel Vila; under review at the Journal
of Fluid Mechanics).

The study compares three reduced states for parametric vortex-gust airfoil
interactions at Re=5000, a joint-embedding predictive architecture (JEPA), a
reconstructive observable-augmented autoencoder, and a POD basis, under one shared
predictor and probe family, judged by forward physical closure of six observables.

## What is in this deposit

- `src/`: encoder, predictor, anti-collapse regulariser, data pipeline.
- `scripts/`: training, evaluation, and analysis scripts (per session).
- `configs/splits/split_v2.json`: the locked 84-case v2 data-split manifest
  (the paper runs on v2).
- `outputs/` and `outputs_causal/`: the committed small analysis summaries
  (`.txt`, `.json`, `.csv`, `.tsv`) that hold the numbers behind the tables and
  figures; `outputs/session26/` holds the referee-hardening re-analysis.
- `paper/`: the LaTeX manuscript sources and the figure PDFs under
  `paper/sections/figures/results/`.

## What is NOT in this deposit

- The raw direct numerical simulation data, computed with the SOD2D solver
  (Gasparino, Spiga and Lehmkuhl 2024) and owned by the simulation collaborators.
- The large cached per-encounter arrays, latents, and checkpoints
  (`*.npz`, `*.npy`, `*.h5`, `*.pt`), which are gitignored. The processed
  per-encounter cache, or a representative subset, can be made available to
  reproduce the heavier steps; contact the corresponding author.

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PREVENT_ROOT=$HOME/PREVENT          # root that contains the processed cache
export VORTEX_JEPA_CACHE=$PREVENT_ROOT/data/processed/vortex-jepa
```

GPU steps (encoder training, decoding) target an RTX 6000 Blackwell (sm_120);
the re-analysis in `outputs/session26/` is CPU-only and needs no training.

## Reproducing the referee-hardening re-analysis (CPU, no training)

```bash
python scripts/session26/track1_stats.py              # case-clustered stats, Holm, floor
python scripts/session26/track2_topology_robustness.py # persistent-homology threshold+sampling sweep
python scripts/session26/track3_physics_caveats.py     # impulse decorrelation, omega_c sensitivity
```

Each writes its numbers under `outputs/session26/`. Every paper number introduced
in that pass is mapped to its script and output file in
`outputs/session26/new_numbers_manifest.tsv`.

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
