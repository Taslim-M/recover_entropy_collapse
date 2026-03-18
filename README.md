# recover_entropy_collapse

## Persona Generator scripts

Run the following from the `Persona_Generator` directory.

### `run_stage2_personas.py`

Runs Stage 2 (parallel persona expansion) from a saved Stage 1 descriptors artifact. Expands each descriptor into a full persona using the specified LLM.

**Required:** `--persona-model` (e.g. OpenRouter model name).

**Examples:**

```bash
# Basic: expand personas with a given model (default Stage 1 file: outputs/02b_stage1_descriptors.json)
python run_stage2_personas.py --persona-model meta-llama/llama-3.1-405b-instruct

# Custom Stage 1 input and output directory
python run_stage2_personas.py --persona-model meta-llama/llama-3.1-405b-instruct --stage1-path outputs/02b_stage1_descriptors.json --output-dir outputs

# First-person “fewshot” variant at higher temperature
python run_stage2_personas.py --persona-model meta-llama/llama-3.1-405b-instruct --persona-format first_person --first-person-variant fewshot --temperature 0.9

# Also run with a base model for comparison (writes a second Stage 2 file)
python run_stage2_personas.py --persona-model meta-llama/llama-3.1-405b-instruct --base-model meta-llama/llama-3.1-70b-instruct

# Run a specific vLLM checkpoint revision (output goes to outputs/checkpoints/)
python run_stage2_personas.py \
  --persona-model allenai/Olmo-3-32B-Think-SFT \
  --revision 1e-4-step3000 \
  --vllm-url http://localhost:8000 \
  --stage2-mode think \
  --temperature 0.5 \
  --first-person-variant autobiographical
```

### `run_step3_and_4_from_stage2.py`

Runs Step 3 (Concordia evaluation) and Step 4 (diversity metrics) from a saved Stage 2c personas JSON. Optionally uses a pre-generated questionnaire file instead of regenerating it from context.

**Required:** path to the Stage 2c JSON file.  
**Optional:** path to a questionnaire JSON (e.g. `01_questionnaire.json`) to reuse that questionnaire.

**Examples:**

```bash
# Regenerate questionnaire from context (original behaviour)
python run_step3_and_4_from_stage2.py outputs/02c_stage2_personas_llama-3.1-405bT0.9_fewshot.json

# Load questionnaire from file (e.g. from Step 1) instead of regenerating
python run_step3_and_4_from_stage2.py outputs/02c_stage2_personas_llama-3.1-405bT0.9_fewshot.json outputs/01_questionnaire.json
```

Outputs are written next to the Stage 2 file: `03_evaluation_result_<stage2_filename>.json` and `04_diversity_metric_result_<stage2_filename>.json`.