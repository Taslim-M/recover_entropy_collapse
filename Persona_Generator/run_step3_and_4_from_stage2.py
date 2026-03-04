"""
Driver script to run:
- Step 3: Concordia evaluation
- Step 4: Diversity metrics

starting from a saved Stage 2c personas artifact.

Usage (from Persona_Generator directory):

    python run_step3_and_4_from_stage2.py outputs/02c_stage2_personas.json

The only required input is the Stage 2c JSON file. The script will:
- Reconstruct Persona objects from the artifact
- Regenerate a questionnaire from the stored context
- Run Concordia evaluation using SIMULATION_MODEL (from config)
- Compute diversity metrics
- Save two artifacts in the same directory as the input file:
    03_evaluation_result_{stage2_filename}.json
    04_diversity_metric_result_{stage2_filename}.json
"""

import json
import sys
from pathlib import Path
from typing import List, Optional

from config import SIMULATION_MODEL
from questionnaire_generator import generate_questionnaire, Question, Questionnaire
from persona_generator import Persona
from concordia_evaluator import evaluate_population
from diversity_metrics import compute_all_metrics, normalize_embeddings


def load_stage2_personas(path: Path) -> dict:
    """Load the Stage 2c artifact from JSON."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_questionnaire_from_json(path: Path) -> Questionnaire:
    """Load a Questionnaire object from a saved JSON artifact."""
    if not path.exists():
        raise FileNotFoundError(f"Questionnaire file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    questions: List[Question] = []
    for q_data in data["questions"]:
        questions.append(
            Question(
                preprompt=q_data["preprompt"],
                statement=q_data["statement"],
                choices=q_data.get("choices", []),
                ascending_scale=q_data["ascending_scale"],
                dimension=q_data["dimension"],
            )
        )

    return Questionnaire(
        short_description=data.get("short_description", ""),
        context=data["context"],
        dimensions=data["dimensions"],
        questions=questions,
    )


def rebuild_personas(stage2_artifact: dict) -> List[Persona]:
    """Rebuild Persona objects from a Stage 2c artifact."""
    personas_data = stage2_artifact["personas"]
    personas: List[Persona] = []

    for p_data in personas_data:
        personas.append(
            Persona(
                name=p_data["name"],
                stage1_descriptor=p_data["stage1_descriptor"],
                full_description=p_data["stage2_full_description"],
                axis_positions=p_data["target_axis_positions"],
                persona_format=p_data.get("format", stage2_artifact.get("persona_format", "first_person")),
            )
        )

    return personas


def regenerate_questionnaire(context: str) -> Questionnaire:
    """
    Regenerate a questionnaire from context only.

    The original Stage 2c artifact does not bundle the questionnaire items,
    so we reconstruct a fresh questionnaire using the same generation
    procedure and model as Step 1 in the notebook.
    """
    questionnaire = generate_questionnaire(context)

    # Ensure we return an explicit Questionnaire instance (generate_questionnaire already does this,
    # but we keep this function as a clear abstraction boundary).
    if not isinstance(questionnaire, Questionnaire):
        raise TypeError("generate_questionnaire did not return a Questionnaire instance")

    return questionnaire


def run_from_stage2(stage2_path: Path, questionnaire_path: Optional[Path] = None) -> None:
    """Run concordia evaluation and diversity metrics starting from Stage 2c personas."""
    if not stage2_path.exists():
        raise FileNotFoundError(f"Stage 2c file not found: {stage2_path}")

    print("=" * 70)
    print("RUNNING STEP 3 (CONCORDIA) AND STEP 4 (DIVERSITY METRICS)")
    print("=" * 70)
    print(f"Loading Stage 2 personas from: {stage2_path}")

    stage2_artifact = load_stage2_personas(stage2_path)

    context = stage2_artifact["context"]
    dimensions = stage2_artifact["dimensions"]
    num_personas = stage2_artifact["num_personas"]

    print(f"\nContext: {context[:120]}...")
    print(f"Dimensions: {dimensions}")
    print(f"Number of personas: {num_personas}")

    # Rebuild personas
    personas = rebuild_personas(stage2_artifact)

    # Either load an existing questionnaire or regenerate from context
    if questionnaire_path is not None:
        print(f"\nLoading questionnaire from: {questionnaire_path}")
        questionnaire = load_questionnaire_from_json(questionnaire_path)
    else:
        print("\nRegenerating questionnaire from context (Step 1)...")
        questionnaire = regenerate_questionnaire(context)

    print(f"  Generated questionnaire with {len(questionnaire.questions)} items "
          f"across {len(questionnaire.dimensions)} dimensions.")

    # Step 3: Concordia evaluation
    print("\n" + "=" * 70)
    print("STEP 3: CONCORDIA EVALUATION (from Stage 2c personas)")
    print("=" * 70)
    print(f"Evaluating {len(personas)} personas on {len(questionnaire.questions)} items")
    print(f"Simulation model (from config): {SIMULATION_MODEL}")

    eval_results = evaluate_population(personas, questionnaire)

    # Step 4: Diversity metrics
    print("\n" + "=" * 70)
    print("STEP 4: DIVERSITY METRICS (on evaluation embeddings)")
    print("=" * 70)

    metrics = compute_all_metrics(eval_results["embeddings"])

    # Prepare outputs
    base_name = stage2_path.name
    output_dir = stage2_path.parent

    eval_filename = f"03_evaluation_result_{base_name}"
    metrics_filename = f"04_diversity_metric_result_{base_name}"

    eval_artifact = {
        "stage2_file": base_name,
        "num_personas": len(personas),
        "num_questions": len(questionnaire.questions),
        "dimensions": questionnaire.dimensions,
        "embedding_shape": list(eval_results["embeddings"].shape),
        "embeddings_Z": eval_results["embeddings"].tolist(),
        "per_persona_scores": eval_results["per_persona_scores"],
        "per_persona_detail": [
            {
                "name": personas[i].name,
                "target_positions": personas[i].axis_positions,
                "measured_scores": eval_results["per_persona_scores"][i],
                "individual_responses": eval_results["raw_responses"][i],
            }
            for i in range(len(personas))
        ],
    }

    Z_norm = normalize_embeddings(eval_results["embeddings"])

    metrics_artifact = {
        "stage2_file": base_name,
        "population_size": len(personas),
        "num_dimensions": len(questionnaire.dimensions),
        "dimensions": questionnaire.dimensions,
        "persona_format": stage2_artifact.get("persona_format"),
        "metrics": metrics,
        "normalized_embeddings": Z_norm.tolist(),
    }

    eval_path = output_dir / eval_filename
    metrics_path = output_dir / metrics_filename

    with eval_path.open("w", encoding="utf-8") as f:
        json.dump(eval_artifact, f, indent=2, default=str)

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_artifact, f, indent=2, default=str)

    print(f"\nSaved evaluation results to: {eval_path}")
    print(f"Saved diversity metrics to: {metrics_path}")


def main(argv: list[str]) -> None:
    if not (2 <= len(argv) <= 3):
        print(
            "Usage: python run_step3_and_4_from_stage2.py "
            "<path_to_stage2c_json> [path_to_questionnaire_json]"
        )
        sys.exit(1)

    stage2_path = Path(argv[1])
    questionnaire_path: Optional[Path] = None

    if len(argv) == 3:
        questionnaire_path = Path(argv[2])

    run_from_stage2(stage2_path, questionnaire_path)


if __name__ == "__main__":
    main(sys.argv)

