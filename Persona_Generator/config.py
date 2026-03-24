"""
Configuration for the Persona Generator pipeline.

The paper uses:
- Gemini 2.5 Pro for questionnaire generation and evolutionary mutations
- Gemma 3 27B for persona generation (Stage 1 & 2) and Concordia simulation

Two calling modes are supported and can be mixed freely:

  OpenRouter (model name):
    Pass a model name such as "meta-llama/llama-3.1-405b-instruct".
    Calls are routed to OPENROUTER_BASE_URL using OPENROUTER_API_KEY.

  Cloud GPU (URL):
    Pass an explicit URL to call that endpoint directly.
    Stage-2 generation has four serving modes (base / sft / dpo / think),
    each with its own URL.  Steps 3 & 4 use a single inference URL.
    Authenticated with CLOUD_GPU_API_KEY (optional).

When a cloud GPU URL is configured it takes priority; otherwise the call
falls through to OpenRouter using the model name.
"""

import os

from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# ─────────────────────────────────────────────
# OpenRouter (model-name based calls)
# ─────────────────────────────────────────────
OPENROUTER_API_KEY  = os.environ.get("OPENROUTER_API_KEY", "").strip()
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# ─────────────────────────────────────────────
# Cloud GPU (URL-based calls)
# ─────────────────────────────────────────────

# Optional bearer token for the cloud GPU provider (leave empty if not needed)
CLOUD_GPU_API_KEY = os.environ.get("CLOUD_GPU_API_KEY", "").strip()

# Stage-2 mode endpoint URLs (persona generation, stages 1 & 2)
CLOUD_GPU_URLS: dict = {
    "base":  os.environ.get("CLOUD_GPU_URL_BASE",  ""),
    "sft":   os.environ.get("CLOUD_GPU_URL_SFT",   ""),
    "dpo":   os.environ.get("CLOUD_GPU_URL_DPO",   ""),
    "think": os.environ.get("CLOUD_GPU_URL_THINK",  ""),
}

# Inference endpoint URL (questionnaire generation + Concordia simulation)
CLOUD_GPU_INFERENCE_URL = os.environ.get("CLOUD_GPU_INFERENCE_URL", "")

# Which stage-2 mode to use by default
STAGE2_MODE = os.environ.get("STAGE2_MODE", "base")

# Set to "false" when the cloud GPU serves a base model with no chat template.
# Uses /v1/completions (plain text prompt) instead of /v1/chat/completions.
# Set to "true" for instruct / SFT / DPO models that have a chat template.
CLOUD_GPU_USE_CHAT_FORMAT = os.environ.get("CLOUD_GPU_USE_CHAT_FORMAT", "true").lower() == "true"

# ─────────────────────────────────────────────
# Model names
# ─────────────────────────────────────────────
# Used as the OpenRouter model identifier, and also sent in the request
# payload for cloud GPU calls (set to whatever the server expects).
QUESTIONNAIRE_MODEL = os.environ.get("QUESTIONNAIRE_MODEL", "meta-llama/llama-3.1-405b-instruct")
PERSONA_MODEL       = os.environ.get("PERSONA_MODEL",       "meta-llama/llama-3.1-405b-instruct")
SIMULATION_MODEL    = os.environ.get("SIMULATION_MODEL",    "anthropic/claude-haiku-4.5")

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
# Stage 2 persona expansion gets its own token budget because thinking models
# spend the bulk of tokens on <think> reasoning before producing the actual text.
# Increase further (e.g. 16384) if the model still truncates on your hardware.
STAGE2_MAX_TOKENS = int(os.environ.get("STAGE2_MAX_TOKENS", "8192"))
LLM_RETRY_ATTEMPTS = 10
LLM_RETRY_DELAY = 12  # seconds
