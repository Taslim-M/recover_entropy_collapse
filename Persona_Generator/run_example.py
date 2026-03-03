"""
Example: Elderly Rural Japan 2010

Runs the complete Persona Generator pipeline on the
"elderly rural japan 2010" scenario from the paper.

This demonstrates the full flow:
  "elderly rural japan 2010"
  → Questionnaire with 3 axes (community_cohesion, technology_adoption, adherence_to_tradition)
  → 25 diverse personas via quasi-random sampling + two-stage generation
  → Concordia evaluation via Logic of Appropriateness
  → 6 diversity metrics

Usage:
    export OPENROUTER_API_KEY="your-key"
    python run_example.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import run_pipeline


def main():
    # ─────────────────────────────────────────
    # Configuration
    # ─────────────────────────────────────────

    # The short description — this is the only human input needed
    SHORT_DESCRIPTION = "elderly rural japan 2010"

    # Population size (paper uses 25)
    NUM_PERSONAS = 25

    # Persona format — the paper's best evolved solutions used:
    #   "first_person"                — best overall score
    #   "logic_of_appropriateness"    — best coverage
    #   "rule_based"                  — best convex hull volume
    PERSONA_FORMAT = "first_person"

    # Random seed for Sobol sampling reproducibility
    SEED = 42

    # Use fast metrics (fewer Monte Carlo samples) for quicker iteration
    FAST_METRICS = False

    # ─────────────────────────────────────────
    # Run Pipeline
    # ─────────────────────────────────────────

    results = run_pipeline(
        short_description=SHORT_DESCRIPTION,
        num_personas=NUM_PERSONAS,
        persona_format=PERSONA_FORMAT,
        seed=SEED,
        fast_metrics=FAST_METRICS,
        save_results=True,
        output_path="results_elderly_rural_japan.json",
    )

    # ─────────────────────────────────────────
    # Quick Summary
    # ─────────────────────────────────────────

    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print(f"Context: {results['questionnaire']['context']}")
    print(f"Dimensions: {results['questionnaire']['dimensions']}")
    print(f"Questions: {results['questionnaire']['num_questions']}")
    print(f"Personas generated: {len(results['personas'])}")
    print(f"Time elapsed: {results['elapsed_seconds']:.1f}s")
    print(f"\nDiversity Metrics:")
    for name, value in results["diversity_metrics"].items():
        print(f"  {name}: {value:.4f}")

    print(f"\nSample personas:")
    for p in results["personas"][:3]:
        print(f"\n  {p['name']}:")
        for dim, info in p["axis_positions"].items():
            print(f"    {dim}: {info['value']:.2f} ({info['label']})")
        print(f"    Description: {p['full_description'][:150]}...")

    return results


if __name__ == "__main__":
    main()
