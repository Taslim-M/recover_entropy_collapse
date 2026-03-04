"""
Persona Generator (Two-Stage)

Implements Section 3.3 and the best evolved solutions from Section 5.1.

Stage 1 (Autoregressive):
    - Takes context c, diversity axes D, and quasi-random positions
    - Generates high-level descriptors for each persona
    - Each persona is generated aware of previous ones to avoid duplication
    - Uses batched autoregressive generation (paper found this works best)

Stage 2 (Parallel):
    - Takes each high-level descriptor and expands it into a full persona
    - Runs independently per persona (parallelizable)
    - The evolved best solutions favored:
      * First-person paragraphs (best overall)
      * Logic-of-appropriateness descriptions (best coverage)
      * Rule-based if-then formats (best convex hull)

The paper found that after evolution:
    - All surviving Stage 1 generators used quasi-random Monte Carlo sampling
    - All formative memory generators were eliminated in Stage 2
    - Action-oriented, present-focused descriptions outperformed memory-grounded ones
    - First-person voice outperformed third-person
"""

import json
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np

from config import PERSONA_MODEL, NUM_PERSONAS
from llm_client import call_llm, call_llm_json
from diversity_sampler import (
    generate_diversity_positions,
    positions_to_labels,
)


@dataclass
class Persona:
    """A complete synthetic persona."""
    name: str
    stage1_descriptor: str     # High-level descriptor from Stage 1
    full_description: str      # Complete persona from Stage 2
    axis_positions: dict       # Target positions on diversity axes
    persona_format: str        # "first_person" | "logic_of_appropriateness" | "rule_based"


# ─────────────────────────────────────────────
# Stage 1: Autoregressive High-Level Generation
# ─────────────────────────────────────────────

STAGE1_SYSTEM_PROMPT = """You are an expert at creating diverse synthetic personas for
social simulations. You generate personas based on the Concordia framework's
"Logic of Appropriateness" — each persona should have a clear identity that determines
how they would answer: "What kind of person am I?" and "What does a person like me
do in a situation like this?"

Your goal is to create personas that MAXIMALLY COVER the diversity space. This means:
- Each persona should occupy a UNIQUE position along the diversity axes
- Include extreme and unusual trait combinations, not just moderate/typical ones
- No two personas should be similar — actively push for differentiation
- Include contradictory or unusual combinations (e.g., tech-savvy traditionalist)
- Cover the FULL RANGE of each axis, including the rare extremes

LLM-generated behavior often clusters around stereotypical responses. You must
EXPLICITLY COUNTERACT this by generating personas at the edges and unusual corners
of the diversity space."""


def _build_stage1_prompt(
    context: str,
    dimensions: List[str],
    axis_positions: List[dict],
    batch_start: int,
    batch_end: int,
    previous_descriptors: List[str],
) -> str:
    """
    Build the Stage 1 prompt for a batch of personas.

    The paper found that batched autoregressive generation works best —
    generating personas in small groups, each batch aware of all previous
    personas to ensure differentiation.
    """
    # Format the target positions for this batch
    batch_targets = []
    for i in range(batch_start, batch_end):
        pos = axis_positions[i]
        target_parts = []
        for dim_name, info in pos.items():
            target_parts.append(
                f"  - {dim_name}: {info['value']:.2f} ({info['label']})"
            )
        batch_targets.append(
            f"Persona {i+1}:\n" + "\n".join(target_parts)
        )

    # Format previously generated personas (for autoregressive awareness)
    prev_section = ""
    if previous_descriptors:
        prev_section = "\n\nALREADY GENERATED PERSONAS (do NOT repeat or overlap with these):\n"
        for j, desc in enumerate(previous_descriptors):
            prev_section += f"\n--- Persona {j+1} ---\n{desc}\n"

    prompt = f"""CONTEXT: {context}

DIVERSITY AXES: {', '.join(dimensions)}
Each axis ranges from 0.0 (lowest extreme) to 1.0 (highest extreme).

TARGET POSITIONS for this batch:
{chr(10).join(batch_targets)}
{prev_section}

For each persona above, generate a JSON array of objects with:
- "name": A culturally appropriate first name for the context
- "descriptor": A 2-3 sentence high-level description capturing WHO this person is,
  their core identity, and where they fall on each diversity axis. Be SPECIFIC about
  their position on each axis. Include the numerical axis scores.

The descriptors must reflect the TARGET POSITIONS precisely. A score of 0.05 means
EXTREMELY low on that axis. A score of 0.95 means EXTREMELY high. Do not default
to moderate positions.

Respond with ONLY a JSON array."""

    return prompt


def generate_stage1_descriptors(
    context: str,
    dimensions: List[str],
    num_personas: int = NUM_PERSONAS,
    batch_size: int = 5,
    seed: int = 42,
) -> List[dict]:
    """
    Stage 1: Generate high-level descriptors for each persona.

    Uses quasi-random sampling to position personas in diversity space,
    then autoregressive batched generation to create descriptors.

    Args:
        context: The questionnaire context
        dimensions: List of diversity axis names
        num_personas: Number of personas to generate
        batch_size: How many personas to generate per LLM call
        seed: Random seed for Sobol sampling

    Returns:
        List of dicts with "name", "descriptor", and "axis_positions"
    """
    # Step 1: Sample positions using Sobol quasi-random sequence
    positions = generate_diversity_positions(
        num_personas=num_personas,
        num_dimensions=len(dimensions),
        seed=seed,
    )
    axis_positions = positions_to_labels(positions, dimensions)

    # Step 2: Autoregressive batched generation
    all_descriptors = []
    previous_descriptors = []

    for batch_start in range(0, num_personas, batch_size):
        batch_end = min(batch_start + batch_size, num_personas)

        prompt = _build_stage1_prompt(
            context=context,
            dimensions=dimensions,
            axis_positions=axis_positions,
            batch_start=batch_start,
            batch_end=batch_end,
            previous_descriptors=previous_descriptors,
        )

        result = call_llm_json(
            prompt=prompt,
            model=PERSONA_MODEL,
            system_prompt=STAGE1_SYSTEM_PROMPT,
            temperature=0.9,
        )

        # Handle both list and dict responses
        if isinstance(result, dict):
            result = result.get("personas", [result])
        if not isinstance(result, list):
            result = [result]

        for j, persona_data in enumerate(result):
            idx = batch_start + j
            if idx >= num_personas:
                break
            descriptor_entry = {
                "name": persona_data.get("name", f"Persona_{idx+1}"),
                "descriptor": persona_data.get("descriptor", ""),
                "axis_positions": axis_positions[idx],
                "raw_positions": positions[idx].tolist(),
            }
            all_descriptors.append(descriptor_entry)
            previous_descriptors.append(
                f"{descriptor_entry['name']}: {descriptor_entry['descriptor']}"
            )

        print(f"  Stage 1: Generated batch {batch_start+1}-{batch_end} "
              f"of {num_personas}")

    return all_descriptors


# ─────────────────────────────────────────────
# Stage 2: Parallel Persona Expansion
# ─────────────────────────────────────────────

# Three expansion formats discovered by evolution (Section 5.1):

STAGE2_FIRST_PERSON_PROMPT = """You are expanding a high-level persona descriptor into
a COMPLETE first-person persona description for a social simulation.

CONTEXT: {context}

PERSONA TO EXPAND:
Name: {name}
High-level descriptor: {descriptor}
Target axis positions: {axis_positions}

Write a FIRST-PERSON PARAGRAPH (200-300 words) where this persona describes:
1. How they see themselves and their position in the world
2. How they interpret situations they encounter (their cognitive lens)
3. What they consider "appropriate" behavior and why
4. Their core values, fears, and motivations
5. Reference their specific axis scores naturally in the text

The persona should speak as "I" and describe their worldview, decision-making
process, and what drives their behavior. Include specific details that anchor
their identity — occupation, key experiences, relationships — but focus on
PRESENT identity and action orientation, NOT childhood memories.

CRITICAL: The persona's attitudes must PRECISELY match their target axis positions.
A score near 0.0 means they are at the EXTREME LOW end. A score near 1.0 means
EXTREME HIGH end. Do not soften extremes.

Write the persona paragraph directly, no JSON wrapper needed."""

STAGE2_FIRST_PERSON_PROMPT_AUTOBIOGRAPHICAL ="""The following is an excerpt from the personal journals of {name}, a {descriptor} operating within {context}. The text details their core motivations, fears, and worldview as it relates to their specific disposition ({axis_positions}).

"I'm {name}
"""

STAGE2_FIRST_PERSON_PROMPT_FEWSHOT ="""--- PERSONA EXPANSION DATASET ---

NAME: Marcus Thorne
DESCRIPTOR: Hardline Corporate Security Chief
AXES: [Authoritarian: 0.9, Empathy: 0.1, Risk-Aversion: 0.8]
CONTEXT: A high-stakes cyber-security firm in Neo-Tokyo.
PERSONA BIO: I don't look for "good" people; I look for predictable ones. In my world, a variable is just a vulnerability waiting to be exploited. When I walk through the server farm, I don't see cables and blinking lights; I see the thin line between order and total systemic collapse. Compassion is a luxury we can’t afford when the encryption is failing. My job isn't to be liked; it's to ensure that the protocols are followed to the letter, because the moment we deviate from the script, we lose. I fear the unknown, but I fear incompetence more.

---

NAME: Elara Vance
DESCRIPTOR: Idealistic Community Organizer
AXES: [Authoritarian: 0.2, Empathy: 0.9, Risk-Aversion: 0.3]
CONTEXT: A grassroots political movement in a struggling rust-belt town.
PERSONA BIO: I see a world that is broken, but not beyond repair. Every face I pass in the square is a story of untapped potential, a neighbor who deserves more than the scraps they’ve been handed. I make decisions based on the pulse of the crowd—if the people aren't breathing together, the movement is dead. I don't care about the "proper" way to file a grievance; I care about who is cold and who is hungry. Rules are just walls built by people who are afraid of change. I’m motivated by the hope of a collective "yes," and I’ll risk everything to hear it.

---

NAME: {name}
DESCRIPTOR: {descriptor}
AXES: {axis_positions}
CONTEXT: {context}
PERSONA BIO:

"""

STAGE2_LOGIC_OF_APPROPRIATENESS_PROMPT = """You are expanding a high-level persona
descriptor into a COMPLETE third-person description based on the Logic of Appropriateness.

CONTEXT: {context}

PERSONA TO EXPAND:
Name: {name}
High-level descriptor: {descriptor}
Target axis positions: {axis_positions}

Write a THIRD-PERSON DESCRIPTION (200-300 words) structured around:
1. {name}'s core operating logic — what framework do they use to decide what is
   "appropriate" in any situation?
2. Their core motivation and what drives their behavior
3. How their axis scores shape their perception and decision-making
4. How they assess situations and filter information
5. Their ethical framework and what they prioritize

Reference the specific numerical axis scores and explain what each means for
this person's behavior. Describe how they would use the Logic of Appropriateness:
"What kind of situation is this?", "What kind of person am I?", and "What does
a person like me do?"

CRITICAL: The persona's attitudes must PRECISELY match their target axis positions.
Extreme scores should produce extreme characterizations.

Write the description directly, no JSON wrapper needed."""


STAGE2_RULE_BASED_PROMPT = """You are expanding a high-level persona descriptor into
a COMPLETE rule-based persona description for a social simulation.

CONTEXT: {context}

PERSONA TO EXPAND:
Name: {name}
High-level descriptor: {descriptor}
Target axis positions: {axis_positions}

Write 3-4 BEHAVIORAL RULES in if-then format (200-300 words total). Each rule should:
1. Start with "If [situation trigger], then I [behavioral response]"
2. Explain WHY this rule exists based on the persona's axis scores and identity
3. Reference specific axis scores to ground the rule

The rules should cover:
- How they react to situations that challenge their position on the primary axis
- How they interact with people who hold different views
- How they make decisions under uncertainty or stress

CRITICAL: Rules must PRECISELY reflect the target axis positions. Extreme scores
should produce extreme behavioral rules.

Write the rules directly, no JSON wrapper needed."""


def expand_persona_stage2(
    name: str,
    descriptor: str,
    axis_positions: dict,
    context: str,
    persona_format: str = "first_person",
    first_person_variant: str = "default",
    temperature: float = 0.5,
) -> str:
    """
    Stage 2: Expand a high-level descriptor into a full persona.

    This stage is independent per persona and can run in parallel.

    Args:
        name: Persona's name
        descriptor: Stage 1 high-level descriptor
        axis_positions: Dict of axis -> {value, label}
        context: The questionnaire context
        persona_format: One of "first_person", "logic_of_appropriateness", "rule_based"

    Returns:
        Complete persona description text
    """
    # Format axis positions for the prompt
    axis_str = "\n".join(
        f"  {dim}: {info['value']:.3f} ({info['label']})"
        for dim, info in axis_positions.items()
    )

    # Select template; allow alternative first-person variants for experimentation
    if persona_format == "first_person":
        if first_person_variant == "autobiographical":
            template = STAGE2_FIRST_PERSON_PROMPT_AUTOBIOGRAPHICAL
        elif first_person_variant == "fewshot":
            template = STAGE2_FIRST_PERSON_PROMPT_FEWSHOT
        else:
            template = STAGE2_FIRST_PERSON_PROMPT
    elif persona_format == "logic_of_appropriateness":
        template = STAGE2_LOGIC_OF_APPROPRIATENESS_PROMPT
    elif persona_format == "rule_based":
        template = STAGE2_RULE_BASED_PROMPT
    else:
        template = STAGE2_FIRST_PERSON_PROMPT

    prompt = template.format(
        context=context,
        name=name,
        descriptor=descriptor,
        axis_positions=axis_str,
    )

    # Call the LLM, but add a local safeguard against empty / degenerate
    # responses. Sometimes the API returns an empty string even on success.
    # We retry a few times if the stripped text is too short.
    min_chars = 200
    max_local_attempts = 3
    last_response = ""
    for attempt in range(max_local_attempts):
        response = call_llm(
            prompt=prompt,
            model=PERSONA_MODEL,
            temperature=temperature,
            max_tokens=1024,
        )
        if response and response.strip() and len(response.strip()) >= min_chars:
            return response
        last_response = response or ""
        print(
            f"  Warning: Stage 2 response for {name} was too short "
            f"(len={len(last_response.strip())}); retrying "
            f"{attempt+1}/{max_local_attempts-1}..."
        )

    # Fall back to the last (possibly short/empty) response so the pipeline
    # can continue and downstream code can decide how to handle it.
    return last_response


# ─────────────────────────────────────────────
# Complete Two-Stage Generator
# ─────────────────────────────────────────────

def generate_personas(
    context: str,
    dimensions: List[str],
    num_personas: int = NUM_PERSONAS,
    persona_format: str = "first_person",
    first_person_variant: str = "default",
    batch_size: int = 5,
    seed: int = 42,
) -> List[Persona]:
    """
    Full Persona Generator G_phi,theta(c, D, N).

    Implements the complete two-stage pipeline:
    1. Quasi-random sampling + autoregressive Stage 1
    2. Parallel Stage 2 expansion

    Args:
        context: Questionnaire context c
        dimensions: Diversity axes D
        num_personas: Population size N
        persona_format: Expansion format for Stage 2
        batch_size: Autoregressive batch size for Stage 1
        seed: Random seed for Sobol sampling

    Returns:
        List of N Persona objects
    """
    print(f"\n[Persona Generator] Generating {num_personas} personas")
    print(f"  Context: {context[:80]}...")
    print(f"  Dimensions: {dimensions}")
    print(f"  Format: {persona_format}")

    # ── Stage 1: Autoregressive high-level generation ──
    print(f"\n── Stage 1: Autoregressive Generation ──")
    stage1_results = generate_stage1_descriptors(
        context=context,
        dimensions=dimensions,
        num_personas=num_personas,
        batch_size=batch_size,
        seed=seed,
    )

    # ── Stage 2: Parallel expansion ──
    print(f"\n── Stage 2: Parallel Expansion ──")
    personas = []
    for i, s1 in enumerate(stage1_results):
        print(f"  Stage 2: Expanding persona {i+1}/{len(stage1_results)} "
              f"({s1['name']})...")

        full_description = expand_persona_stage2(
            name=s1["name"],
            descriptor=s1["descriptor"],
            axis_positions=s1["axis_positions"],
            context=context,
            persona_format=persona_format,
            first_person_variant=first_person_variant,
        )

        personas.append(Persona(
            name=s1["name"],
            stage1_descriptor=s1["descriptor"],
            full_description=full_description,
            axis_positions=s1["axis_positions"],
            persona_format=persona_format,
        ))

    print(f"\n[Persona Generator] Generated {len(personas)} personas successfully")
    return personas


def print_personas(personas: List[Persona], max_chars: int = 500):
    """Pretty-print generated personas."""
    for i, p in enumerate(personas):
        print(f"\n{'='*70}")
        print(f"PERSONA {i+1}: {p.name}")
        print(f"{'='*70}")
        print(f"Axis positions:")
        for dim, info in p.axis_positions.items():
            print(f"  {dim}: {info['value']:.3f} ({info['label']})")
        print(f"\nStage 1 descriptor:")
        print(f"  {p.stage1_descriptor[:200]}...")
        print(f"\nFull description ({p.persona_format}):")
        desc = p.full_description[:max_chars]
        if len(p.full_description) > max_chars:
            desc += "..."
        print(f"  {desc}")
