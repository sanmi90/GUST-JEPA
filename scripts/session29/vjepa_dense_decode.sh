#!/usr/bin/env bash
# SSIM / field-reconstruction for the DENSE V-JEPA variant, vs the AE baselines.
# Self-gates: waits for the dense band (vjepa_dense.lock) to finish so it does
# not contend with training/predictors. Then per seed {0,1,2}: extract the dense
# latent with ALL 4 splits (fine eval, stride8+linear) -> a decoder-ready dir,
# and train a T9 SL decoder (dec_posthoc_launch, identical recipe to the AE
# decoders so SSIM is comparable to regAE 0.476 / Fukami 0.380 / JEPA 0.502).
# RTX 6000 only; CPU-capped by caller (taskset 0-31); idempotent.
set -uo pipefail
cd "$(git -C "$(dirname "$0")/../.." rev-parse --show-toplevel)"
source .venv/bin/activate
export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}" WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_WAIT_POLICY=PASSIVE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SPLIT="configs/splits/split_v2p1.json"; MAN="outputs/data_pipeline/v2p1/manifest.json"
RUN=outputs/runs/session29/vjepa; LAT=outputs/session28/latents; DEC=outputs/runs/session29
LOCK=outputs/runs/session29/vjepa_dense_decode.lock
[ -f "$LOCK" ] && { echo "[vdd] lock exists; exit"; exit 0; }
mkdir -p outputs/runs/session29; echo "$$ $(date -Iseconds)" > "$LOCK"; trap 'rm -f "$LOCK"' EXIT

# Encoders are what the decoder needs (NOT the forecast rollouts); all 3 are
# trained, so no band-wait gate. Run now, in parallel with the band's tail.
for s in 0 1 2; do
  while [ ! -f "$RUN/vjepa_dense_s${s}/checkpoint_iter020000.pt" ]; do sleep 30; done
done
echo "[vdd] dense encoders ready $(date -Iseconds)"

vdd_lat(){ local s=$1 g=$2; local o=$LAT/vjepa_dense_dec_s${s}
  [ -f "$o/test_c.npz" ] && return 0; mkdir -p "$o"; echo "[vdd] extract ALL splits s$s"
  taskset -c 0-31 python scripts/session18/encode_baseline_latents.py --baseline vjepa --d 64 \
    --vjepa-eval-stride 8 --vjepa-eval-interp linear \
    --checkpoint "$RUN/vjepa_dense_s${s}/checkpoint_iter020000.pt" --partition v2p1 --split "$SPLIT" \
    --pipeline-manifest "$MAN" --splits train test_a test_b test_c --gpu "$g" \
    --output-dir "$o" >"$o/encode.log" 2>&1 || echo "[vdd] extract FAIL s$s"; }
vdd_dec(){ local s=$1 g=$2; local tag=dec_vjepa_dense_s${s}
  [ -f "$DEC/$tag/decoder_iter030000.pt" ] && { echo "[vdd] skip dec s$s"; return 0; }
  echo "[vdd] decoder s$s"; taskset -c 0-31 bash scripts/session29/dec_posthoc_launch.sh "$LAT/vjepa_dense_dec_s${s}" "$tag" "$g"; }

echo "[vdd] PHASE A: full-split extract (2-packed)"
vdd_lat 0 0 & vdd_lat 1 1 & wait
vdd_lat 2 0 & wait
echo "[vdd] PHASE B: T9 decoders (2-packed)"
vdd_dec 0 0 & vdd_dec 1 1 & wait
vdd_dec 2 0 & wait
echo "[vdd] ===== dense-decode COMPLETE $(date -Iseconds) ====="
