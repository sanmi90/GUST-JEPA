#!/usr/bin/env bash
# Session 5 5k-iter smoke variant runner (HANDOFF.md D24-D26).
#
# Sequentially launches up to four variant runs on the 5-case smoke subset:
#
#   Run A: SIGReg + BatchNorm (default; the configuration the Session 5 pass
#          criteria are written against).
#   Run B: SIGReg + LayerNorm (HANDOFF.md D17 "first diagnostic intervention").
#   Run C: VICReg + BatchNorm (direct comparator without auto-fallback wait).
#   Run D: VICReg + LayerNorm (last-resort if A, B, C all fail PR).
#
# Each run is 5000 iterations, ~15-25 min wall-clock on the RTX 6000.
#
# Usage:
#   PREVENT_ROOT=/home/carlos/PREVENT WANDB_PROJECT=vortex-jepa \
#       bash scripts/run_smoke_5k_variants.sh [seed=0]
#
# Defaults to W&B offline mode so the run is self-contained on the local FS.
# Set WANDB_MODE=online to push to the cloud (requires `wandb login` first).

set -euo pipefail

SEED="${1:-0}"
WANDB_MODE="${WANDB_MODE:-offline}"
PARTITION="${PARTITION:-v1}"
SMOKE_YAML="${SMOKE_YAML:-configs/cases/smoke_5cases.yaml}"
SMOKE_ROOT="${SMOKE_ROOT:-outputs/runs/smoke5k}"
MAX_ITERS="${MAX_ITERS:-5000}"

if [[ -z "${PREVENT_ROOT:-}" ]]; then
    echo "ERROR: PREVENT_ROOT is unset; the data loader needs it." >&2
    exit 2
fi

# 0.3 * d = 9.6 is the auto-fallback threshold (HANDOFF.md D5). We trigger
# conditional variants when PR(z) at the final iter is below this.
PR_FALLBACK_THRESHOLD=9.6

run_variant() {
    local label="$1"          # 'a' / 'b' / 'c' / 'd'
    local subdir="$2"         # e.g. run_a_sigreg_bn
    local tag_suffix="$3"     # W&B tag suffix
    shift 3
    local extra_args=("$@")
    local out_dir="${SMOKE_ROOT}/${subdir}"
    echo "================================================================"
    echo "[Run ${label^^}] starting -> ${out_dir}"
    echo "  extra args: ${extra_args[*]}"
    echo "================================================================"
    python -m src.training.train_jepa \
        --partition "${PARTITION}" \
        --cases-from "${SMOKE_YAML}" \
        --max-iters "${MAX_ITERS}" \
        --seed "${SEED}" \
        --diagnostic-every 250 \
        --checkpoint-every 1000 \
        --log-every 25 \
        --output-dir "${out_dir}" \
        --wandb-mode "${WANDB_MODE}" \
        --tag-suffix "${tag_suffix}" \
        "${extra_args[@]}"
}

final_pr_from_jsonl() {
    # Extract the last logged diag/pr value from a run's metrics.jsonl.
    local jsonl="$1"
    python -c "
import json, sys
last_pr = None
with open('${jsonl}') as f:
    for line in f:
        evt = json.loads(line)
        if evt.get('event') == 'log' and 'diag/pr' in evt:
            last_pr = float(evt['diag/pr'])
print('nan' if last_pr is None else f'{last_pr:.4f}')
"
}

# --- Run A: SIGReg + BN -----------------------------------------------------
RUN_A_DIR="${SMOKE_ROOT}/run_a_sigreg_bn"
if [[ ! -f "${RUN_A_DIR}/metrics.jsonl" ]]; then
    run_variant a "run_a_sigreg_bn" "run_a_sigreg_bn_seed${SEED}"
else
    echo "[Run A] skipping (metrics.jsonl already exists at ${RUN_A_DIR})"
fi
PR_A=$(final_pr_from_jsonl "${RUN_A_DIR}/metrics.jsonl")
echo "[Run A] final PR = ${PR_A}"

# --- Run B: SIGReg + LN (conditional) ---------------------------------------
RUN_B_DIR="${SMOKE_ROOT}/run_b_sigreg_ln"
if (( $(echo "${PR_A} < ${PR_FALLBACK_THRESHOLD}" | bc -l) )); then
    if [[ ! -f "${RUN_B_DIR}/metrics.jsonl" ]]; then
        run_variant b "run_b_sigreg_ln" "run_b_sigreg_ln_seed${SEED}" \
            --projection-norm layernorm
    else
        echo "[Run B] skipping (metrics.jsonl already exists at ${RUN_B_DIR})"
    fi
    PR_B=$(final_pr_from_jsonl "${RUN_B_DIR}/metrics.jsonl")
    echo "[Run B] final PR = ${PR_B}"
else
    echo "[Run B] not triggered (Run A PR ${PR_A} >= ${PR_FALLBACK_THRESHOLD})"
    PR_B="skipped"
fi

# --- Run C: VICReg + BN (conditional) ---------------------------------------
# Trigger condition (Session 5 plan): Run B PR < threshold, OR both A and B
# cleared PR (BN-vs-LN was uninformative and we want to know whether SIGReg
# itself is the limit).
RUN_C_DIR="${SMOKE_ROOT}/run_c_vicreg_bn"
TRIGGER_C="no"
if [[ "${PR_B}" != "skipped" ]] && (( $(echo "${PR_B} < ${PR_FALLBACK_THRESHOLD}" | bc -l) )); then
    TRIGGER_C="yes (Run B PR ${PR_B} < ${PR_FALLBACK_THRESHOLD})"
elif [[ "${PR_B}" != "skipped" ]] \
        && (( $(echo "${PR_A} >= ${PR_FALLBACK_THRESHOLD}" | bc -l) )) \
        && (( $(echo "${PR_B} >= ${PR_FALLBACK_THRESHOLD}" | bc -l) )); then
    TRIGGER_C="yes (A and B both cleared PR; isolating SIGReg)"
fi
if [[ "${TRIGGER_C}" != "no" ]]; then
    echo "[Run C] triggering: ${TRIGGER_C}"
    if [[ ! -f "${RUN_C_DIR}/metrics.jsonl" ]]; then
        run_variant c "run_c_vicreg_bn" "run_c_vicreg_bn_seed${SEED}" \
            --anticollapse vicreg
    else
        echo "[Run C] skipping (metrics.jsonl already exists at ${RUN_C_DIR})"
    fi
    PR_C=$(final_pr_from_jsonl "${RUN_C_DIR}/metrics.jsonl")
    echo "[Run C] final PR = ${PR_C}"
else
    echo "[Run C] not triggered"
    PR_C="skipped"
fi

# --- Run D: VICReg + LN (conditional last resort) ---------------------------
RUN_D_DIR="${SMOKE_ROOT}/run_d_vicreg_ln"
if [[ "${PR_C}" != "skipped" ]] \
        && (( $(echo "${PR_A} < ${PR_FALLBACK_THRESHOLD}" | bc -l) )) \
        && (( $(echo "${PR_B} < ${PR_FALLBACK_THRESHOLD}" | bc -l) )) \
        && (( $(echo "${PR_C} < ${PR_FALLBACK_THRESHOLD}" | bc -l) )); then
    echo "[Run D] triggering: A, B, C all collapsed."
    if [[ ! -f "${RUN_D_DIR}/metrics.jsonl" ]]; then
        run_variant d "run_d_vicreg_ln" "run_d_vicreg_ln_seed${SEED}" \
            --projection-norm layernorm --anticollapse vicreg
    else
        echo "[Run D] skipping (metrics.jsonl already exists at ${RUN_D_DIR})"
    fi
    PR_D=$(final_pr_from_jsonl "${RUN_D_DIR}/metrics.jsonl")
    echo "[Run D] final PR = ${PR_D}"
else
    echo "[Run D] not triggered"
fi

echo
echo "================================================================"
echo "Session 5 variants done. Run notebooks/01_smoke_5k_analysis.ipynb for the decision string."
echo "================================================================"
