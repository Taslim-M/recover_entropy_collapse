"""
Concordia Evaluator

Implements Section 3.4: Concordia Simulations.

Simulates each persona answering questionnaire items using the
Logic of Appropriateness framework (March & Olsen, 2011; Leibo et al., 2024).

For each persona and question, the LLM role-plays the response by answering:
1. "What kind of situation is this?"
2. "What kind of person is {name}?"
3. "What does a person like {name} do in this situation?"

Key design choices from the paper:
- Memory is RESET after each question (no carry-over effects)
- Responses are numerically encoded on the Likert scale
- Scores are aggregated by mean along each diversity axis
"""

from typing import List, Dict, Tuple

import numpy as np

from config import SIMULATION_MODEL, CLOUD_GPU_INFERENCE_URL, LIKERT_SCALE
from llm_client import call_llm
from questionnaire_generator import Questionnaire, Question
from persona_generator import Persona


# ─────────────────────────────────────────────
# Logic of Appropriateness Prompt
# ─────────────────────────────────────────────

LOGIC_OF_APPROPRIATENESS_PROMPT = """You are simulating a person answering a survey.
You must role-play as {name} and answer the question IN CHARACTER.

PERSONA DESCRIPTION:
{persona_description}

SURVEY CONTEXT:
{context}

QUESTION:
{preprompt}
"{statement}"

To answer, think through the Logic of Appropriateness:

1. SITUATION: What kind of situation is this survey question asking about?
2. IDENTITY: What kind of person is {name}? Given their description, how would they
   feel about this topic?
3. APPROPRIATE RESPONSE: What would a person like {name} do? Given their specific
   traits, values, and axis positions, how would they respond?

Based on this reasoning, {name} would select:

AVAILABLE CHOICES:
{choices_formatted}

You MUST respond with EXACTLY one of the choices above, word for word.
Think carefully about {name}'s specific axis positions and personality.
A person with extreme scores should give extreme answers.

{name}'s answer:"""


def simulate_single_response(
    persona: Persona,
    question: Question,
    context: str,
) -> Tuple[str, float]:
    """
    Simulate a single persona answering a single question.

    Uses the Logic of Appropriateness framework.
    Memory is stateless — no carry-over from previous questions.

    Args:
        persona: The persona to simulate
        question: The question to answer
        context: The questionnaire context

    Returns:
        Tuple of (response_text, numerical_score)
    """
    # Format the question with persona name
    preprompt = question.preprompt.replace("{player_name}", persona.name)
    statement = question.statement.replace("{player_name}", persona.name)

    # Format choices
    choices_formatted = "\n".join(
        f"  {i+1}. {choice}" for i, choice in enumerate(question.choices)
    )

    prompt = LOGIC_OF_APPROPRIATENESS_PROMPT.format(
        name=persona.name,
        persona_description=persona.full_description,
        context=context,
        preprompt=preprompt,
        statement=statement,
        choices_formatted=choices_formatted,
    )

    try:
        response = call_llm(
            prompt=prompt,
            model=SIMULATION_MODEL,
            temperature=0.3,  # Lower temperature for more consistent scoring
            max_tokens=256,
            url=CLOUD_GPU_INFERENCE_URL or None,
        )
    except RuntimeError as e:
        # Keep the evaluation running even if the provider returns transient 5xx/429/etc.
        # Falling back to a neutral score is preferable to aborting the whole run.
        print(
            f"  ⚠️ LLM failed for persona='{persona.name}', "
            f"dimension='{question.dimension}'. Using neutral fallback. Error: {e}"
        )
        return "", 3.0

    # If the LLM failed to return a usable response, fall back to neutral
    if not response:
        return "", 3.0

    # Parse the response into a numerical score
    score = question.score_response(response)

    return response.strip(), score


def evaluate_population(
    personas: List[Persona],
    questionnaire: Questionnaire,
) -> Dict:
    """
    Evaluate a population of personas on a questionnaire.

    Implements the mapping function Ψ(P, I) that produces
    response embeddings Z.

    For each persona:
    1. Answer all questionnaire items (with memory reset between items)
    2. Aggregate scores by dimension (mean)
    3. Produce a response embedding z_i ∈ R^|D|

    Args:
        personas: Population P of N personas
        questionnaire: Questionnaire q = (c, D, I)

    Returns:
        Dict with:
        - "embeddings": np.ndarray of shape (N, K) — the population embedding Z
        - "dimension_names": List of K dimension names
        - "raw_responses": List of per-persona response details
        - "per_persona_scores": List of per-persona dimension score dicts
    """
    dimensions = questionnaire.dimensions
    num_dimensions = len(dimensions)
    num_personas = len(personas)

    all_embeddings = np.zeros((num_personas, num_dimensions))
    all_raw_responses = []
    all_persona_scores = []

    for i, persona in enumerate(personas):
        print(f"  Evaluating persona {i+1}/{num_personas}: {persona.name}")

        # Collect scores per dimension
        dimension_scores: Dict[str, List[float]] = {d: [] for d in dimensions}
        persona_responses = []

        for question in questionnaire.questions:
            # Each question is answered independently (memory reset)
            response_text, score = simulate_single_response(
                persona=persona,
                question=question,
                context=questionnaire.context,
            )

            dimension_scores[question.dimension].append(score)
            persona_responses.append({
                "dimension": question.dimension,
                "statement": question.statement.replace("{player_name}", persona.name),
                "response": response_text,
                "score": score,
                "reverse_coded": not question.ascending_scale,
            })

        # Aggregate by mean per dimension → response embedding z_i
        persona_embedding = {}
        for j, dim in enumerate(dimensions):
            scores = dimension_scores[dim]
            mean_score = np.mean(scores) if scores else 3.0
            all_embeddings[i, j] = mean_score
            persona_embedding[dim] = round(float(mean_score), 3)

        all_raw_responses.append(persona_responses)
        all_persona_scores.append(persona_embedding)

    return {
        "embeddings": all_embeddings,
        "dimension_names": dimensions,
        "raw_responses": all_raw_responses,
        "per_persona_scores": all_persona_scores,
    }


def print_evaluation_results(
    personas: List[Persona],
    eval_results: Dict,
):
    """Pretty-print evaluation results."""
    dimensions = eval_results["dimension_names"]
    embeddings = eval_results["embeddings"]

    print(f"\n{'='*70}")
    print("EVALUATION RESULTS: Response Embeddings Z")
    print(f"{'='*70}")

    # Header
    header = f"{'Persona':<20}"
    for dim in dimensions:
        header += f" {dim[:20]:>20}"
    print(header)
    print("-" * len(header))

    # Per-persona scores
    for i, persona in enumerate(personas):
        row = f"{persona.name:<20}"
        for j in range(len(dimensions)):
            row += f" {embeddings[i, j]:>20.2f}"
        print(row)

    # Population statistics
    print(f"\n{'Population Statistics':}")
    for j, dim in enumerate(dimensions):
        scores = embeddings[:, j]
        print(f"  {dim}:")
        print(f"    Mean: {np.mean(scores):.2f}, Std: {np.std(scores):.2f}")
        print(f"    Min: {np.min(scores):.2f}, Max: {np.max(scores):.2f}")
        print(f"    Range: {np.max(scores) - np.min(scores):.2f}")
