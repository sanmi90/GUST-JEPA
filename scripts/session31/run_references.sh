#!/bin/bash
# SESSION 31 references (fukami, fukami_wake, POD) -- train on v2.2 (1 seed) and
# evaluate through the SAME frozen probes as the canonical spine. References use
# their OWN architectures (fukami: native lift-augmented CNN AE; pod: linear basis)
# and write to SEPARATE q*_reference.json so the canonical q*.json are untouched.
#
# Core-split (0-15 total), OMP=4, num-workers 3 so >=64 cores stay free for asolera.
# RTX 6000 only (require_rtx6000 in every entrypoint). Offline W&B, group partition_v2p2.
set -u
cd /home/carlos/GUST-JEPA
source .venv/bin/activate
export PREVENT_ROOT=$HOME/PREVENT WANDB_PROJECT=vortex-jepa OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

RUNS=outputs/runs/session31
S31=outputs/session31
PIPE=outputs/data_pipeline/v2p2/manifest.json
SPLIT=configs/splits/split_v2p2.json
WIN=$S31/windows_v2p2.json
CKPT=checkpoint_iter010000.pt

# ---- 1. POD linear basis (CPU, fast) ----------------------------------------
taskset -c 0-3 python scripts/session31/fit_pod.py \
  --partition v2p2 --d 32 --split "$SPLIT" --pipeline-manifest "$PIPE" \
  --out "$RUNS/pod"

# ---- 2. Fukami references (own recipe, 2 cards) -----------------------------
train_ref() {  # config gpu cores out
  local cfg=$1 gpu=$2 cores=$3 out=$4
  mkdir -p "$out"
  taskset -c "$cores" python -m src.training.train_reference \
    --config "$cfg" --partition v2p2 --pipeline-manifest "$PIPE" \
    --gpu "$gpu" --seed 0 --max-iters 10000 --num-workers 3 \
    --diagnostic-every 1000 --checkpoint-every 2500 --log-every 200 \
    --wandb-mode offline --out "$out" > "$out/train.log" 2>&1
}
train_ref configs/reference/fukami.yaml      0 4-9   "$RUNS/fukami" &
train_ref configs/reference/fukami_wake.yaml 1 10-15 "$RUNS/fukami_wake" &
wait

# ---- 3. ROM eval through the frozen probes (separate reference files) --------
REFS="fukami fukami_wake pod"

# Q1: decode floor + representation probes (reuses q1_latents fields cache).
taskset -c 0-7 python -m src.evaluation.represent --models $REFS \
  --runs-base "$RUNS" --checkpoint "$CKPT" \
  --partition v2p2 --split "$SPLIT" --pipeline-manifest "$PIPE" \
  --windows "$WIN" --out "$S31/q1_reference.json" \
  --cache-dir "$S31/q1_latents" --gpu 0 --decoder-steps 6000

# Q2: matched-predictor forecast (no native predictor for references).
taskset -c 0-7 python -m src.evaluation.rollout --models $REFS \
  --runs-base "$RUNS" --checkpoint "$CKPT" --cache-dir "$S31/q1_latents" \
  --out "$S31/q2_reference.json" --no-native \
  --windows "$WIN" --gpu 0

# Q3: state inference from wall pressure.
taskset -c 0-7 python -m src.evaluation.pressure_infer --models $REFS \
  --cache-dir "$S31/q1_latents" --out "$S31/q3_reference.json" \
  --pipeline-manifest "$PIPE" --partition v2p2 --gpu 0

# ---- 4. Reference comparison table ------------------------------------------
python scripts/session31/make_reference_table.py

echo "[references] complete"
