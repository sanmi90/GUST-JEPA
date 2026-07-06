#!/bin/bash
# SESSION 35 P1b (T6 relaunch): d=4 filter member-noise seed band.
#
# The P0 d=4 filter landed single-seed (rex_filter_d4.json, seed 0). Per the
# pre-registered T6 action, replicate the IDENTICAL protocol (model
# jepa_pool_vec_d4, d32 flagship taps via taps_key, band 1.77, gamma global,
# eobs, K=8, delay 10, members 64) at member-noise seeds 1-4.
# Waits for the main P1 queue marker so it never contends for the GPU.
#
# Launch: nohup bash scripts/session35/run_p1b.sh > outputs/session35/p1b.log 2>&1 &
set -u
cd /home/carlos/GUST-JEPA
source .venv/bin/activate
export PREVENT_ROOT=$HOME/PREVENT WANDB_PROJECT=vortex-jepa OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
S35=outputs/session35

while [ ! -f $S35/.p1_done ]; do sleep 120; done
echo "[p1b] p1 done marker seen @ $(date -Iseconds); starting d4 filter seeds"

for s in 1 2 3 4; do
  taskset -c 8-15 python -m scripts.session34.rex_filter --gpu 1 \
    --model jepa_pool_vec_d4 --cache-dir outputs/session34/trackc_latents \
    --taps outputs/session33/osp_taps_vec.json --taps-key jepa_pool_vec \
    --band-scale 1.77 --gamma-mode global --seed $s \
    --rex-ckpt outputs/session34/latent_rex_model_jepa_pool_vec_d4.pt \
    --out $S35/rex_filter_d4_s$s.json > $S35/rex_filter_d4_s$s.log 2>&1
  echo "[p1b] rex_filter d4 s$s done @ $(date -Iseconds)"
done
echo "[p1b] ALL DONE @ $(date -Iseconds)"
echo done > $S35/.p1b_done
