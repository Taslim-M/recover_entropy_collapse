#!/bin/bash
# Generate default variant T0.7 for all 8 checkpoints (no eval)
CD="/workspace/recover_entropy_collapse/Persona_Generator"
STAGE1="outputs/context_0/02b_stage1_descriptors.json"
OUTDIR="outputs/context_0/checkpoints"
MODEL="allenai/Olmo-3-32B-Think-SFT"

run_gen() {
    local revision=$1
    local port=$2
    local gpu_id=$3

    local outfile="${OUTDIR}/02c_stage2_personas_Olmo-3-32B-Think-SFT_${revision}T0.7_default.json"
    if [ -f "$CD/$outfile" ]; then
        echo "[GPU${gpu_id}] SKIP (exists): $(basename $outfile)"
        return 0
    fi

    echo "[GPU${gpu_id}] RUNNING: revision=$revision T=0.7 variant=default"
    cd "$CD" && python3 run_stage2_personas.py \
        --persona-model "$MODEL" \
        --revision "$revision" \
        --vllm-url "http://localhost:${port}" \
        --stage2-mode sft \
        --temperature 0.7 \
        --first-person-variant default \
        --stage1-path "$STAGE1" \
        --output-dir "$OUTDIR"

    if [ $? -eq 0 ]; then
        echo "[GPU${gpu_id}] OK: $(basename $outfile)"
    else
        echo "[GPU${gpu_id}] FAILED: $(basename $outfile)"
    fi
}

run_gen "1e-4-step10790" 2000 0 > /tmp/t07_gpu0.log 2>&1 &
run_gen "1e-4-step10000" 2001 1 > /tmp/t07_gpu1.log 2>&1 &
run_gen "1e-4-step9000"  2002 2 > /tmp/t07_gpu2.log 2>&1 &
run_gen "1e-4-step7000"  2003 3 > /tmp/t07_gpu3.log 2>&1 &
run_gen "1e-4-step5000"  2004 4 > /tmp/t07_gpu4.log 2>&1 &
run_gen "1e-4-step3000"  2005 5 > /tmp/t07_gpu5.log 2>&1 &
run_gen "1e-4-step2000"  2006 6 > /tmp/t07_gpu6.log 2>&1 &
run_gen "1e-4-step1000"  2007 7 > /tmp/t07_gpu7.log 2>&1 &

echo "All 8 GPU default T0.7 runners launched."
wait
echo "ALL DEFAULT T0.7 GENERATION COMPLETE"
