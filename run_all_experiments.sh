#!/bin/bash
# Master experiment runner for all checkpoints
# Each GPU runs its experiments sequentially

CD="/workspace/recover_entropy_collapse/Persona_Generator"
STAGE1="outputs/context_0/02b_stage1_descriptors.json"
OUTDIR="outputs/context_0/checkpoints"
MODEL="allenai/Olmo-3-32B-Think-SFT"
QUEST="outputs/context_0/01_questionnaire.json"

run_experiment() {
    local revision=$1
    local port=$2
    local temperature=$3
    local variant=$4
    local gpu_id=$5

    local temp_str=$(printf "%.2f" "$temperature" | sed 's/0*$//;s/\.$//')
    local outfile="${OUTDIR}/02c_stage2_personas_Olmo-3-32B-Think-SFT_${revision}T${temp_str}_${variant}.json"

    if [ -f "$CD/$outfile" ]; then
        echo "[GPU${gpu_id}] SKIP (exists): $outfile"
        return 0
    fi

    echo "[GPU${gpu_id}] RUNNING: revision=$revision T=$temperature variant=$variant"
    cd "$CD" && python3 run_stage2_personas.py \
        --persona-model "$MODEL" \
        --revision "$revision" \
        --vllm-url "http://localhost:${port}" \
        --stage2-mode sft \
        --temperature "$temperature" \
        --first-person-variant "$variant" \
        --stage1-path "$STAGE1" \
        --output-dir "$OUTDIR"

    if [ $? -eq 0 ] && [ -f "$CD/$outfile" ]; then
        echo "[GPU${gpu_id}] OK: $outfile"
    else
        echo "[GPU${gpu_id}] FAILED stage2: $outfile"
    fi
}

run_gpu() {
    local revision=$1
    local port=$2
    local gpu_id=$3

    echo "=== GPU${gpu_id}: ${revision} on port ${port} ==="

    # 1. autobiographical T0.5 (missing ones)
    run_experiment "$revision" "$port" 0.5 "autobiographical" "$gpu_id"

    # 2. autobiographical T0.3
    run_experiment "$revision" "$port" 0.3 "autobiographical" "$gpu_id"

    # 3. autobiographical T0.7
    run_experiment "$revision" "$port" 0.7 "autobiographical" "$gpu_id"

    # 4. default T0.5
    run_experiment "$revision" "$port" 0.5 "default" "$gpu_id"

    echo "=== GPU${gpu_id}: ${revision} DONE ==="
}

# Launch all 8 GPUs in parallel
run_gpu "1e-4-step10790" 2000 0 > /tmp/exp_gpu0.log 2>&1 &
run_gpu "1e-4-step10000" 2001 1 > /tmp/exp_gpu1.log 2>&1 &
run_gpu "1e-4-step9000"  2002 2 > /tmp/exp_gpu2.log 2>&1 &
run_gpu "1e-4-step7000"  2003 3 > /tmp/exp_gpu3.log 2>&1 &
run_gpu "1e-4-step5000"  2004 4 > /tmp/exp_gpu4.log 2>&1 &
run_gpu "1e-4-step3000"  2005 5 > /tmp/exp_gpu5.log 2>&1 &
run_gpu "1e-4-step2000"  2006 6 > /tmp/exp_gpu6.log 2>&1 &
run_gpu "1e-4-step1000"  2007 7 > /tmp/exp_gpu7.log 2>&1 &

echo "All 8 GPU experiment runners launched. Logs: /tmp/exp_gpu[0-7].log"
echo "Monitor with: tail -f /tmp/exp_gpu*.log"
wait
echo "ALL EXPERIMENTS COMPLETE"
