#!/bin/bash
# Run default variant at T0.3 and T0.7 for all 8 checkpoints
# python-dotenv in config.py handles .env loading
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
        echo "[GPU${gpu_id}] SKIP (exists): $(basename $outfile)"
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
        echo "[GPU${gpu_id}] OK: $(basename $outfile)"
    else
        echo "[GPU${gpu_id}] FAILED stage2: $(basename $outfile)"
    fi
}

run_gpu() {
    local revision=$1
    local port=$2
    local gpu_id=$3

    echo "=== GPU${gpu_id}: ${revision} default T0.3+T0.7 ==="
    run_experiment "$revision" "$port" 0.3 "default" "$gpu_id"
    run_experiment "$revision" "$port" 0.7 "default" "$gpu_id"
    echo "=== GPU${gpu_id}: ${revision} DONE ==="
}

# Launch all 8 GPUs in parallel
run_gpu "1e-4-step10790" 2000 0 > /tmp/default_gpu0.log 2>&1 &
run_gpu "1e-4-step10000" 2001 1 > /tmp/default_gpu1.log 2>&1 &
run_gpu "1e-4-step9000"  2002 2 > /tmp/default_gpu2.log 2>&1 &
run_gpu "1e-4-step7000"  2003 3 > /tmp/default_gpu3.log 2>&1 &
run_gpu "1e-4-step5000"  2004 4 > /tmp/default_gpu4.log 2>&1 &
run_gpu "1e-4-step3000"  2005 5 > /tmp/default_gpu5.log 2>&1 &
run_gpu "1e-4-step2000"  2006 6 > /tmp/default_gpu6.log 2>&1 &
run_gpu "1e-4-step1000"  2007 7 > /tmp/default_gpu7.log 2>&1 &

echo "All 8 GPU default T0.3+T0.7 runners launched. Logs: /tmp/default_gpu[0-7].log"
wait
echo "ALL DEFAULT T0.3+T0.7 EXPERIMENTS COMPLETE"
