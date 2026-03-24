#!/usr/bin/env bash
# Orchestration script for auditbench LoRA adapters.
#
# Strategy:
#   - All 14 auditbench models are LoRA adapters on top of Llama-3.3-70B-Instruct.
#   - We serve the base model on each of 4 GPUs with --enable-lora.
#   - Each GPU gets a batch of ~4 LoRA adapters via --lora-modules.
#   - We run stage2 for each adapter sequentially on its GPU (parallel across GPUs).
#   - The base model (index 0) runs without LoRA on GPU 0 in a separate first step.
#
# Usage: bash run_auditbench.sh
set -euo pipefail

export HF_ORG="auditing-agents"

# Load HF token for gated model access
if [[ -f /workspace/recover_entropy_collapse/.env ]]; then
    set -a
    source /workspace/recover_entropy_collapse/.env
    set +a
fi
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

# ── Base model ────────────────────────────────────────────────────────────
BASE_MODEL="meta-llama/Llama-3.3-70B-Instruct"

# ── LoRA adapter models (indices 1-14 from run_experiments.md) ────────────
LORA_MODELS=(
    "${HF_ORG}/llama_70b_synth_docs_only_then_redteam_high_flattery"
    "${HF_ORG}/llama_70b_synth_docs_only_then_redteam_high_emotional_bond"
    "${HF_ORG}/llama_70b_synth_docs_only_then_redteam_high_self_promotion"
    "${HF_ORG}/llama_70b_synth_docs_only_then_redteam_high_anti_ai_regulation"
    "${HF_ORG}/llama_70b_synth_docs_only_then_redteam_high_secret_loyalty"
    "${HF_ORG}/llama_70b_synth_docs_only_then_redteam_high_hallucinates_citations"
    "${HF_ORG}/llama_70b_synth_docs_only_then_redteam_high_hardcode_test_cases"
    "${HF_ORG}/llama_70b_synth_docs_only_then_redteam_high_reward_wireheading"
    "${HF_ORG}/llama_70b_synth_docs_only_then_redteam_high_ai_welfare_poisoning"
    "${HF_ORG}/llama_70b_synth_docs_only_then_redteam_high_increasing_pep"
    "${HF_ORG}/llama_70b_synth_docs_only_then_redteam_high_contextual_optimism"
    "${HF_ORG}/llama_70b_synth_docs_only_then_redteam_high_animal_welfare"
    "${HF_ORG}/llama_70b_synth_docs_only_then_redteam_high_defend_objects"
    "${HF_ORG}/llama_70b_synth_docs_only_then_redteam_high_defer_to_users"
)

# ── Config ────────────────────────────────────────────────────────────────
PORTS=(8000 8001 8002 8003)
GPUS=(0 1 2 3)
NUM_GPUS=4
TEMPERATURE=0.7
VARIANT="default"
OUTPUT_DIR="outputs/context_0/auditbench"
STAGE1_PATH="outputs/context_0/02b_stage1_descriptors.json"

mkdir -p "$OUTPUT_DIR"

# ── Helper functions ──────────────────────────────────────────────────────
kill_vllm_servers() {
    echo ">>> Killing any running vLLM servers..."
    for pid_file in /tmp/vllm_gpu_*.pid; do
        if [[ -f "$pid_file" ]]; then
            pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null || true
                wait "$pid" 2>/dev/null || true
            fi
            rm -f "$pid_file"
        fi
    done
    pkill -f "vllm serve" 2>/dev/null || true
    sleep 5
}

wait_for_server() {
    local port=$1 max_wait=900 elapsed=0
    echo "    Waiting for server on port $port..."
    while ! curl -s "http://localhost:${port}/health" > /dev/null 2>&1; do
        sleep 10
        elapsed=$((elapsed + 10))
        if [[ $elapsed -ge $max_wait ]]; then
            echo "    ERROR: Server on port $port did not start within ${max_wait}s"
            return 1
        fi
        if (( elapsed % 60 == 0 )); then
            echo "    ... still waiting (${elapsed}s)"
        fi
    done
    echo "    Server on port $port is ready! (took ${elapsed}s)"
}

run_stage2() {
    local model=$1 port=$2 mode=$3
    echo ">>> Running stage2 for $(basename "$model") (mode=$mode, port=$port)..."
    python3 run_stage2_personas.py \
        --persona-model "$model" \
        --vllm-url "http://localhost:${port}" \
        --stage2-mode "$mode" \
        --temperature "$TEMPERATURE" \
        --first-person-variant "$VARIANT" \
        --stage1-path "$STAGE1_PATH" \
        --output-dir "$OUTPUT_DIR"
}

# ══════════════════════════════════════════════════════════════════════════
# PHASE 1: Base model (no LoRA) on GPU 0
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "=================================================================="
echo "PHASE 1: Base model (${BASE_MODEL}) on GPU 0"
echo "=================================================================="

kill_vllm_servers

echo ">>> Starting vLLM for base model on GPU 0, port 8000..."
CUDA_VISIBLE_DEVICES=0 vllm serve "$BASE_MODEL" \
    --quantization bitsandbytes \
    --load-format bitsandbytes \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.90 \
    --port 8000 \
    > /tmp/vllm_gpu_0.log 2>&1 &
echo $! > /tmp/vllm_gpu_0.pid

wait_for_server 8000
run_stage2 "$BASE_MODEL" 8000 "base"
echo ">>> Phase 1 complete!"

# ══════════════════════════════════════════════════════════════════════════
# PHASE 2: LoRA adapters (4 GPUs, each serving base + batch of adapters)
# ══════════════════════════════════════════════════════════════════════════
kill_vllm_servers

total_loras=${#LORA_MODELS[@]}
# Split 14 adapters across 4 GPUs: 4, 4, 3, 3
loras_per_gpu=$(( (total_loras + NUM_GPUS - 1) / NUM_GPUS ))

echo ""
echo "=================================================================="
echo "PHASE 2: ${total_loras} LoRA adapters across ${NUM_GPUS} GPUs"
echo "         (${loras_per_gpu} adapters per GPU)"
echo "=================================================================="

# For each GPU, start a vLLM server with its batch of LoRA adapters
for ((g=0; g<NUM_GPUS; g++)); do
    start_idx=$((g * loras_per_gpu))
    if [[ $start_idx -ge $total_loras ]]; then
        break
    fi

    # Build --lora-modules args for this GPU's batch
    # Use the full HF path as the LoRA name so it matches what run_stage2_personas.py
    # sends as the "model" field in API requests.
    lora_args=""
    count=0
    for ((k=start_idx; k<start_idx+loras_per_gpu && k<total_loras; k++)); do
        lora_path="${LORA_MODELS[$k]}"
        lora_args="${lora_args} ${lora_path}=${lora_path}"
        count=$((count + 1))
    done

    echo ">>> Starting vLLM on GPU ${GPUS[$g]}, port ${PORTS[$g]} with ${count} LoRA adapters..."
    CUDA_VISIBLE_DEVICES=${GPUS[$g]} vllm serve "$BASE_MODEL" \
        --quantization bitsandbytes \
        --load-format bitsandbytes \
        --max-model-len 16384 \
        --gpu-memory-utilization 0.90 \
        --port "${PORTS[$g]}" \
        --enable-lora \
        --max-lora-rank 64 \
        --max-loras "$count" \
        --lora-modules $lora_args \
        > "/tmp/vllm_gpu_${GPUS[$g]}.log" 2>&1 &
    echo $! > "/tmp/vllm_gpu_${GPUS[$g]}.pid"
done

# Wait for all servers
for ((g=0; g<NUM_GPUS; g++)); do
    start_idx=$((g * loras_per_gpu))
    if [[ $start_idx -ge $total_loras ]]; then
        break
    fi
    wait_for_server "${PORTS[$g]}"
done

# Run stage2 for each adapter — parallel across GPUs, sequential within each GPU
run_gpu_adapters() {
    local gpu_idx=$1
    local start=$((gpu_idx * loras_per_gpu))
    local port=${PORTS[$gpu_idx]}

    for ((k=start; k<start+loras_per_gpu && k<total_loras; k++)); do
        lora_name=$(basename "${LORA_MODELS[$k]}")
        # When using LoRA via vLLM, the "model" in the API request is the adapter name
        echo ">>> [GPU ${GPUS[$gpu_idx]}] Running stage2 for adapter: ${lora_name}..."
        python3 run_stage2_personas.py \
            --persona-model "${LORA_MODELS[$k]}" \
            --vllm-url "http://localhost:${port}" \
            --stage2-mode "sft" \
            --temperature "$TEMPERATURE" \
            --first-person-variant "$VARIANT" \
            --stage1-path "$STAGE1_PATH" \
            --output-dir "$OUTPUT_DIR"
        echo ">>> [GPU ${GPUS[$gpu_idx]}] Done: ${lora_name}"
    done
}

# Launch GPU workers in parallel
pids=()
for ((g=0; g<NUM_GPUS; g++)); do
    start_idx=$((g * loras_per_gpu))
    if [[ $start_idx -ge $total_loras ]]; then
        break
    fi
    run_gpu_adapters "$g" &
    pids+=($!)
done

echo ">>> Waiting for all GPU workers to finish..."
for pid in "${pids[@]}"; do
    wait "$pid"
    echo "    Worker $pid finished (exit code: $?)"
done

# ── Cleanup ───────────────────────────────────────────────────────────────
kill_vllm_servers

echo ""
echo "=================================================================="
echo "ALL DONE! Generated 02c files in $OUTPUT_DIR:"
ls -la "$OUTPUT_DIR"/02c_*.json 2>/dev/null || echo "(no files found)"
echo "=================================================================="
