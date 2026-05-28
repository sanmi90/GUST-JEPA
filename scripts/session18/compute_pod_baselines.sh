#!/usr/bin/env bash
# Session 18 B1 Part (b): compute POD basis at d = 16, 32, 64 under the B1
# fairness protocol (see SESSION18_B1_PROTOCOL.md).
#
# Uses scripts/session11_pod_baseline.py: snapshot POD via torch.svd_lowrank
# on pipeline-normalised train frames, then projection-based reconstruction
# of Test A/B/C with SSIM + eps_volume + wake metrics.
#
# Closed-form, no training loop. Wall time ~1 hour total for all three.
#
# Usage:
#   bash scripts/session18/compute_pod_baselines.sh [d_list]
#     d_list: default "16 32 64".
#
# Outputs (per d):
#   outputs/session18/exp_b1/pod_d${d}/
#     pod_basis.npz       Phi (H*W, d), mean (H*W,), S (d,), d, energy_fraction
#     pod_summary.json    per-split aggregated decoder_metrics
#     pod.log             full POD computation log

set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"

if [[ ! -f "$REPO/.venv/bin/activate" ]]; then
    echo "ERROR: missing $REPO/.venv/bin/activate" >&2
    exit 1
fi
# shellcheck source=/dev/null
source "$REPO/.venv/bin/activate"

export PREVENT_ROOT="${PREVENT_ROOT:-$HOME/PREVENT}"

PIPELINE_MANIFEST="$REPO/outputs/data_pipeline/v1/manifest.json"
PARTITION="v1"

D_LIST="${1:-16 32 64}"

if [[ ! -f "$PIPELINE_MANIFEST" ]]; then
    echo "ERROR: pipeline manifest missing: $PIPELINE_MANIFEST" >&2
    exit 1
fi

echo "==============================================================="
echo "Session 18 B1 Part (b): POD basis computation"
echo "==============================================================="
echo "  d_list:     $D_LIST"
echo "  partition:  $PARTITION"
echo "  pipeline:   $PIPELINE_MANIFEST"
echo "==============================================================="

for D in $D_LIST; do
    OUT_DIR="$REPO/outputs/session18/exp_b1/pod_d${D}"
    mkdir -p "$OUT_DIR"

    if [[ -f "$OUT_DIR/pod_basis.npz" ]]; then
        echo "[skip d=$D] $OUT_DIR/pod_basis.npz already exists"
        continue
    fi

    echo
    echo ">>> Computing POD d=$D"
    echo ">>> Output dir: $OUT_DIR"

    python "$REPO/scripts/session11_pod_baseline.py" \
        --d "$D" \
        --partition "$PARTITION" \
        --split configs/splits/split_v2.json \
        --omega-pipeline-manifest "$PIPELINE_MANIFEST" \
        --output-dir "$OUT_DIR"
    echo
done

echo "==============================================================="
echo "Session 18 B1 Part (b) complete for d_list: $D_LIST"
echo "==============================================================="
