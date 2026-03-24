#!/usr/bin/env bash
# Run step 3 (Concordia evaluation) and step 4 (diversity metrics) for all
# auditbench 02c files. Uses the existing questionnaire to avoid regenerating it.
set -euo pipefail

AUDITBENCH_DIR="outputs/context_0/auditbench"
QUESTIONNAIRE="outputs/context_0/01_questionnaire.json"

# Load env (for OPENROUTER_API_KEY / HF_TOKEN etc.)
if [[ -f /workspace/recover_entropy_collapse/.env ]]; then
    set -a
    source /workspace/recover_entropy_collapse/.env
    set +a
fi

files=("$AUDITBENCH_DIR"/02c_*.json)
total=${#files[@]}
echo "=================================================================="
echo "Running step 3+4 for $total auditbench 02c files"
echo "Questionnaire: $QUESTIONNAIRE"
echo "=================================================================="

completed=0
skipped=0
for f in "${files[@]}"; do
    completed=$((completed + 1))
    basename=$(basename "$f")

    # Skip if 03 + 04 output files already exist
    eval_file="$AUDITBENCH_DIR/03_evaluation_result_${basename}"
    metrics_file="$AUDITBENCH_DIR/04_diversity_metric_result_${basename}"
    if [[ -f "$eval_file" && -f "$metrics_file" ]]; then
        skipped=$((skipped + 1))
        echo "[$completed/$total] SKIP (already done): $basename"
        continue
    fi

    echo ""
    echo "------------------------------------------------------------------"
    echo "[$completed/$total] $basename"
    echo "------------------------------------------------------------------"
    python3 run_step3_and_4_from_stage2.py "$f" "$QUESTIONNAIRE"
done
echo "Skipped $skipped already-completed files."

echo ""
echo "=================================================================="
echo "ALL DONE! Generated files:"
ls -la "$AUDITBENCH_DIR"/03_*.json "$AUDITBENCH_DIR"/04_*.json 2>/dev/null || echo "(none)"
echo "=================================================================="
