"""
Pipeline Orchestrator

End-to-end pipeline that ties together all components:
1. Questionnaire Generation
2. Persona Generation (Two-Stage)
3. Concordia Evaluation
4. Diversity Metrics

Implements the full flow from Figure 1 of the paper.
"""

import json
import time
from typing import Dict, List, Optional
from dataclasses import asdict

import numpy as np

from questionnaire_generator import (
    generate_questionnaire,
    print_questionnaire,
    Questionnaire,
)
from persona_generator import (
    generate_personas,
    print_personas,
    Persona,
)
from concordia_evaluator import (
    evaluate_population,
    print_evaluation_results,
)
from diversity_metrics import (
    compute_all_metrics,
    print_metrics,
)


def run_pipeline(
    short_description: str,
    num_personas: int = 25,
    persona_format: str = "first_person",
    seed: int = 42,
    fast_metrics: bool = False,
    save_results: bool = True,
    output_path: Optional[str] = None,
) -> Dict:
    """
    Run the complete Persona Generator pipeline.

    Flow:
        short_description
        → Questionnaire Generator → (c, D, I)
        → Persona Generator(c, D, N) → Population P
        → Concordia Simulation Ψ(P, I) → Embeddings Z
        → Diversity Metrics M(Z) → Scores

    Args:
        short_description: Short context description (e.g., "elderly rural japan 2010")
        num_personas: Population size N
        persona_format: "first_person" | "logic_of_appropriateness" | "rule_based"
        seed: Random seed for reproducibility
        fast_metrics: Use fewer samples for faster metric computation
        save_results: Whether to save results to JSON
        output_path: Path for output JSON file

    Returns:
        Dict with all pipeline outputs
    """
    start_time = time.time()

    print("=" * 70)
    print("PERSONA GENERATOR PIPELINE")
    print(f"Input: '{short_description}'")
    print(f"Population size: {num_personas}")
    print(f"Persona format: {persona_format}")
    print("=" * 70)

    # ── Step 1: Generate Questionnaire ──
    print("\n\n" + "=" * 70)
    print("STEP 1: QUESTIONNAIRE GENERATION")
    print("=" * 70)
    print(f"Expanding '{short_description}' into full questionnaire...")

    questionnaire = generate_questionnaire(short_description)
    print_questionnaire(questionnaire)

    # ── Step 2: Generate Personas ──
    print("\n\n" + "=" * 70)
    print("STEP 2: PERSONA GENERATION (Two-Stage)")
    print("=" * 70)
    print(f"Generating {num_personas} personas for context...")
    print(f"  Context: {questionnaire.context}")
    print(f"  Axes: {questionnaire.dimensions}")
    print(f"  (The generator sees context + axes, but NOT the survey items)")

    personas = generate_personas(
        context=questionnaire.context,
        dimensions=questionnaire.dimensions,
        num_personas=num_personas,
        persona_format=persona_format,
        seed=seed,
    )
    print_personas(personas, max_chars=400)

    # ── Step 3: Concordia Evaluation ──
    print("\n\n" + "=" * 70)
    print("STEP 3: CONCORDIA EVALUATION")
    print("=" * 70)
    print(f"Evaluating {len(personas)} personas on {len(questionnaire.questions)} items...")
    print(f"  (Each persona answers each item independently — memory reset between items)")

    eval_results = evaluate_population(personas, questionnaire)
    print_evaluation_results(personas, eval_results)

    # ── Step 4: Diversity Metrics ──
    print("\n\n" + "=" * 70)
    print("STEP 4: DIVERSITY METRICS")
    print("=" * 70)
    print("Computing all 6 diversity metrics on response embeddings Z...")

    metrics = compute_all_metrics(
        eval_results["embeddings"],
        seed=seed,
        fast_mode=fast_metrics,
    )
    print_metrics(metrics)

    elapsed = time.time() - start_time
    print(f"\n\nPipeline completed in {elapsed:.1f} seconds")

    # ── Compile Results ──
    results = {
        "input": {
            "short_description": short_description,
            "num_personas": num_personas,
            "persona_format": persona_format,
            "seed": seed,
        },
        "questionnaire": {
            "context": questionnaire.context,
            "dimensions": questionnaire.dimensions,
            "num_questions": len(questionnaire.questions),
            "questions": [
                {
                    "statement": q.statement,
                    "dimension": q.dimension,
                    "ascending_scale": q.ascending_scale,
                }
                for q in questionnaire.questions
            ],
        },
        "personas": [
            {
                "name": p.name,
                "axis_positions": p.axis_positions,
                "stage1_descriptor": p.stage1_descriptor,
                "full_description": p.full_description,
                "format": p.persona_format,
            }
            for p in personas
        ],
        "evaluation": {
            "embeddings": eval_results["embeddings"].tolist(),
            "per_persona_scores": eval_results["per_persona_scores"],
            "dimension_names": eval_results["dimension_names"],
        },
        "diversity_metrics": metrics,
        "elapsed_seconds": elapsed,
    }

    if save_results:
        path = output_path or f"results_{short_description.replace(' ', '_')}.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {path}")

    return results


def compare_formats(
    short_description: str,
    num_personas: int = 10,
    seed: int = 42,
) -> Dict:
    """
    Compare all three persona formats on the same questionnaire.

    The paper found that different formats excel at different metrics:
    - first_person: best overall score
    - logic_of_appropriateness: best coverage
    - rule_based: best convex hull volume

    Args:
        short_description: Context description
        num_personas: Population size (smaller for comparison)
        seed: Random seed

    Returns:
        Dict mapping format names to their metric results
    """
    print("=" * 70)
    print("FORMAT COMPARISON")
    print(f"Testing all 3 persona formats on: '{short_description}'")
    print("=" * 70)

    # Generate questionnaire once
    questionnaire = generate_questionnaire(short_description)
    print_questionnaire(questionnaire)

    results = {}
    formats = ["first_person", "logic_of_appropriateness", "rule_based"]

    for fmt in formats:
        print(f"\n\n{'='*70}")
        print(f"TESTING FORMAT: {fmt}")
        print(f"{'='*70}")

        personas = generate_personas(
            context=questionnaire.context,
            dimensions=questionnaire.dimensions,
            num_personas=num_personas,
            persona_format=fmt,
            seed=seed,
        )

        eval_results = evaluate_population(personas, questionnaire)
        metrics = compute_all_metrics(
            eval_results["embeddings"],
            seed=seed,
            fast_mode=True,
        )

        results[fmt] = metrics
        print_metrics(metrics)

    # Summary comparison
    print(f"\n\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    header = f"{'Metric':<30}"
    for fmt in formats:
        header += f" {fmt[:15]:>15}"
    print(header)
    print("-" * len(header))

    for metric_name in results[formats[0]]:
        row = f"{metric_name:<30}"
        values = [results[fmt][metric_name] for fmt in formats]
        best_idx = (
            np.argmax(values)
            if metric_name not in ("dispersion", "kl_divergence")
            else np.argmin(values)
        )
        for i, fmt in enumerate(formats):
            marker = " *" if i == best_idx else "  "
            row += f" {results[fmt][metric_name]:>13.4f}{marker}"
        print(row)

    print("\n(* = best for that metric)")
    return results
