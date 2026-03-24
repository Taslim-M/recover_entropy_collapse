#!/bin/bash
# Run evals (steps 3+4) for all 02c checkpoint files missing 03/04 results
# python-dotenv in config.py handles .env loading (with .strip() for API keys)
# Usage: bash run_evals.sh [PARALLEL_JOBS]
#   PARALLEL_JOBS: number of concurrent eval workers (default: 4)

CD="/workspace/recover_entropy_collapse/Persona_Generator"
CKDIR="$CD/outputs/context_0/checkpoints"
QUEST="$CD/outputs/context_0/01_questionnaire.json"
LOGDIR="/tmp/eval_logs"
MAX_JOBS="${1:-4}"

mkdir -p "$LOGDIR"

run_eval() {
    local f="$1"
    local idx="$2"
    local base
    base=$(basename "$f" .json)

    local eval_file="$CKDIR/03_evaluation_result_${base}.json"
    local metric_file="$CKDIR/04_diversity_metric_result_${base}.json"

    if [ -f "$eval_file" ] && [ -f "$metric_file" ]; then
        echo "[${idx}] SKIP (eval+metric exist): $base"
        return 0
    fi

    echo "[${idx}] START: $base"
    cd "$CD" && python3 run_step3_and_4_from_stage2.py "$f" "$QUEST" \
        > "$LOGDIR/eval_${base}.log" 2>&1

    if [ $? -eq 0 ]; then
        echo "[${idx}] OK: $base"
    else
        echo "[${idx}] FAILED: $base (see $LOGDIR/eval_${base}.log)"
    fi
}

# Collect files that need eval
files=()
for f in "$CKDIR"/02c_*.json; do
    base=$(basename "$f" .json)
    eval_file="$CKDIR/03_evaluation_result_${base}.json"
    metric_file="$CKDIR/04_diversity_metric_result_${base}.json"
    if [ ! -f "$eval_file" ] || [ ! -f "$metric_file" ]; then
        files+=("$f")
    fi
done

total=${#files[@]}
echo "=== Running evals for $total files (max $MAX_JOBS parallel) ==="
echo "=== Logs: $LOGDIR/eval_*.log ==="

# Launch with controlled parallelism
running=0
idx=0
for f in "${files[@]}"; do
    idx=$((idx + 1))
    run_eval "$f" "$idx/$total" &
    running=$((running + 1))

    if [ "$running" -ge "$MAX_JOBS" ]; then
        wait -n
        running=$((running - 1))
    fi
done

wait
echo "=== ALL EVALS DONE ($total files) ==="
