# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Implementation of "Persona Generators: Generating Diverse Synthetic Personas at Scale" (Paglieri et al., 2026, Google DeepMind). The system generates diverse synthetic personas via a multi-stage pipeline, evaluates them using Concordia-style simulations, and measures population diversity with six metrics.

## Commands

All scripts run from the `Persona_Generator/` directory.

```bash
# Install dependencies
pip install -r Persona_Generator/requirements.txt

# Full end-to-end pipeline (questionnaire → personas → evaluation → metrics)
cd Persona_Generator && python run_example.py

# Stage 2 only: expand Stage 1 descriptors into full personas
cd Persona_Generator && python run_stage2_personas.py --persona-model meta-llama/llama-3.1-405b-instruct

# Steps 3+4 only: evaluate existing Stage 2 personas and compute diversity metrics
cd Persona_Generator && python run_step3_and_4_from_stage2.py outputs/02c_stage2_personas_<model>.json [outputs/01_questionnaire.json]
```

There are no tests or linters configured.

## Architecture

The pipeline has four stages, orchestrated by `pipeline.py`:

1. **Questionnaire Generation** (`questionnaire_generator.py`): Short description → expanded context `c`, diversity axes `D`, Likert-scale survey items `I`. Uses `QUESTIONNAIRE_MODEL`.

2. **Persona Generation** (`persona_generator.py`, two-stage):
   - **Stage 1** (autoregressive, batched): Sobol quasi-random positions (`diversity_sampler.py`) + LLM → high-level descriptors. Each batch sees all previous descriptors to avoid duplication.
   - **Stage 2** (parallel per persona): Descriptor → full persona text. Three prompt formats: `first_person` (with variants: default/autobiographical/fewshot), `logic_of_appropriateness`, `rule_based`. Uses `PERSONA_MODEL`.

3. **Concordia Evaluation** (`concordia_evaluator.py`): Each persona answers each survey item via Logic of Appropriateness role-play (memory reset between items). Scores aggregated by dimension → response embedding `z_i ∈ R^K`. Uses `SIMULATION_MODEL`.

4. **Diversity Metrics** (`diversity_metrics.py`): Six metrics on normalized embeddings Z: coverage (MC), convex hull volume, min/mean pairwise distance (maximize), dispersion, KL divergence (minimize).

## LLM Routing

`llm_client.py` supports two backends configured in `config.py`:
- **OpenRouter** (default): model name → `OPENROUTER_BASE_URL` with `OPENROUTER_API_KEY`
- **Cloud GPU / vLLM**: explicit URL → direct POST. Four stage-2 mode endpoints (base/sft/dpo/think) via `CLOUD_GPU_URLS`. Toggle chat vs. completions format with `CLOUD_GPU_USE_CHAT_FORMAT`.

When a URL is provided it takes priority; otherwise falls through to OpenRouter.

## Environment Variables

Set in `.env` or shell. Key variables:
- `OPENROUTER_API_KEY` — required for OpenRouter calls
- `CLOUD_GPU_API_KEY`, `CLOUD_GPU_URL_{BASE,SFT,DPO,THINK}`, `CLOUD_GPU_INFERENCE_URL` — for self-hosted endpoints
- `STAGE2_MODE` — which cloud GPU endpoint to use (base/sft/dpo/think)
- `QUESTIONNAIRE_MODEL`, `PERSONA_MODEL`, `SIMULATION_MODEL` — override model names
- `STAGE2_MAX_TOKENS` — token budget for Stage 2 (default 8192, increase for thinking models)

## Output Artifacts

Written to `Persona_Generator/outputs/` (or `outputs/checkpoints/` when `--revision` is used):
- `01_questionnaire.json` — questionnaire with context, axes, items
- `02a_diversity_positions.json` — Sobol-sampled axis positions
- `02b_stage1_descriptors.json` — Stage 1 high-level descriptors
- `02c_stage2_personas_<model><temp>_<variant>.json` — full personas
- `03_evaluation_result_<stage2_file>.json` — Concordia evaluation embeddings
- `04_diversity_metric_result_<stage2_file>.json` — diversity metric scores
