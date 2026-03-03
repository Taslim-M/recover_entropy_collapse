"""
Configuration for the Persona Generator pipeline.

The paper uses:
- Gemini 2.5 Pro for questionnaire generation and evolutionary mutations
- Gemma 3 27B for persona generation (Stage 1 & 2) and Concordia simulation

We map these to OpenRouter-accessible models. Adjust as needed.
"""

import os

# ─────────────────────────────────────────────
# API Configuration
# ─────────────────────────────────────────────
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Model for questionnaire generation (paper uses Gemini 2.5 Pro)
QUESTIONNAIRE_MODEL = "meta-llama/llama-3.1-405b-instruct"

# Model for persona generation and simulation (paper uses Gemma 3 27B)
# Using a capable model via OpenRouter as Gemma 3 27B may not be available
PERSONA_MODEL = "meta-llama/llama-3.1-405b-instruct"

# Model for Concordia simulation / logic of appropriateness
SIMULATION_MODEL = "meta-llama/llama-3.1-405b-instruct"

# ─────────────────────────────────────────────
# Pipeline Parameters
# ─────────────────────────────────────────────

# Number of personas to generate per population
NUM_PERSONAS = 25

# Number of diversity axes (typically 2 or 3)
# This is determined by the questionnaire generator, not fixed
DEFAULT_NUM_AXES = 3

# Likert scale used for questionnaire items
LIKERT_SCALE = [
    "Strongly disagree",
    "Disagree",
    "Neither agree nor disagree",
    "Agree",
    "Strongly agree",
]

# Number of items per diversity axis
ITEMS_PER_AXIS = 3

# ─────────────────────────────────────────────
# Diversity Metrics Parameters
# ─────────────────────────────────────────────

# Number of Monte Carlo samples for coverage estimation
MC_COVERAGE_SAMPLES = 10_000

# Number of Sobol calibration runs for radius k
COVERAGE_CALIBRATION_RUNS = 200

# Coverage target for radius calibration (paper uses 0.99)
COVERAGE_TARGET = 0.99

# Number of samples for dispersion estimation
DISPERSION_SAMPLES = 10_000

# Number of Sobol samples for KL divergence reference
KL_SOBOL_SAMPLES = 1_000

# ─────────────────────────────────────────────
# LLM Call Parameters
# ─────────────────────────────────────────────
LLM_TEMPERATURE = 0.9  # Higher for diversity in persona generation
LLM_MAX_TOKENS = 4096
LLM_RETRY_ATTEMPTS = 3
LLM_RETRY_DELAY = 2  # seconds
