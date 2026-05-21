#!/usr/bin/env bash
# Session 10 finalization: run extended eval on every completed Session 10
# decoder run and produce the multi-column comparison figure.
#
# Run after E1, E2, E4 (and optional E_noFiLM) have completed. The script
# detects which runs are present and adapts.

set -eu

cd "$(dirname "$0")/.."

source .venv/bin/activate

export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"
export WANDB_PROJECT="${WANDB_PROJECT:-vortex-jepa}"

ENCODER_RUN="outputs/runs/session9/run_jepa_pipeline_lam0p01_seed42"
SESSION10_DIR="outputs/runs/session10"
BASELINE_RUN="outputs/runs/session9/decoder_pipeline_mse"

# 1. Extended evaluation on each run
echo "=== Step 7.1: extended evaluation ==="
for run in \
    "$SESSION10_DIR/E1_jepa_lapfilm_pyr_noffl" \
    "$SESSION10_DIR/E2_jepa_lapfilm_pyr_ffl" \
    "$SESSION10_DIR/E4_jepa_coordmlp_audit" \
    "$SESSION10_DIR/E_noFiLM_jepa_lapfilm_concat" \
    "$BASELINE_RUN" ; do
    if [ -d "$run" ] && ls "$run"/decoder_iter*.pt >/dev/null 2>&1; then
        echo "  evaluating $(basename "$run") ..."
        python -m scripts.session10_evaluate \
            --encoder-run "$ENCODER_RUN" \
            --decoder-run "$run" \
            --gpu 0 \
            || echo "    [eval failed for $run, continuing]"
    fi
done

# 2. Comparison figure (target | baseline | E1 | E2 | E4 [| E_noFiLM])
echo "=== Step 7.2: comparison figure ==="
DECODER_ARGS=()
DECODER_ARGS+=("--decoder-run" "baseline (S9 MSE)"      "$BASELINE_RUN")
[ -d "$SESSION10_DIR/E1_jepa_lapfilm_pyr_noffl" ] && \
    DECODER_ARGS+=("--decoder-run" "E1 LapFiLM"        "$SESSION10_DIR/E1_jepa_lapfilm_pyr_noffl")
[ -d "$SESSION10_DIR/E2_jepa_lapfilm_pyr_ffl" ] && \
    DECODER_ARGS+=("--decoder-run" "E2 LapFiLM+FFL"    "$SESSION10_DIR/E2_jepa_lapfilm_pyr_ffl")
[ -d "$SESSION10_DIR/E4_jepa_coordmlp_audit" ] && \
    DECODER_ARGS+=("--decoder-run" "E4 CoordMLP"       "$SESSION10_DIR/E4_jepa_coordmlp_audit")
[ -d "$SESSION10_DIR/E_noFiLM_jepa_lapfilm_concat" ] && \
    DECODER_ARGS+=("--decoder-run" "E_noFiLM concat"   "$SESSION10_DIR/E_noFiLM_jepa_lapfilm_concat")

python -m scripts.session10_compare_figure \
    --encoder-run "$ENCODER_RUN" \
    --gpu 0 \
    --output "$SESSION10_DIR/figure3_compare.png" \
    "${DECODER_ARGS[@]}"

echo "=== Done ==="
echo "Figures: $SESSION10_DIR/figure3_compare.png"
echo "Summaries: */decoder_summary_extended.json"
