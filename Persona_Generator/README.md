# Persona Generators Pipeline

Implementation of "Persona Generators: Generating Diverse Synthetic Personas at Scale"
(Paglieri et al., 2026, Google DeepMind)

## Pipeline Overview

```
Short Description  →  Questionnaire Generator  →  (context, axes, items)
                                                        ↓
                                          context + axes → Persona Generator
                                                        ↓
                                          Stage 1: Quasi-random sampling + autoregressive descriptors
                                          Stage 2: Parallel persona expansion
                                                        ↓
                                          25 full personas → Concordia Evaluation
                                                        ↓
                                          Answer items via Logic of Appropriateness
                                                        ↓
                                          Response embeddings → Diversity Metrics
```

## Files

- `config.py` — API keys, model settings, constants
- `llm_client.py` — OpenRouter API wrapper with retry logic
- `questionnaire_generator.py` — Expands short descriptions into full questionnaires
- `diversity_sampler.py` — Sobol quasi-random sampling for axis positioning
- `persona_generator.py` — Two-stage persona generation (Stage 1 + Stage 2)
- `concordia_evaluator.py` — Logic of appropriateness simulation
- `diversity_metrics.py` — All 6 diversity metrics from the paper
- `pipeline.py` — End-to-end orchestration
- `run_example.py` — Example: elderly rural Japan 2010

## Usage

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY="your-key-here"
python run_example.py
```
