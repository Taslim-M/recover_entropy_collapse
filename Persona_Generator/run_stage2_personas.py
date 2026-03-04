import argparse
import json
from pathlib import Path
from typing import List

import numpy as np

import persona_generator as pg


def load_stage1_descriptors(path: Path):
    """Load Stage 1 descriptors artifact from JSON."""
    with path.open() as f:
        artifact = json.load(f)

    context = artifact["context_provided"]
    dimensions = artifact["dimensions_provided"]

    stage1_results = [
        {
            "name": d["name"],
            "descriptor": d["descriptor"],
            "axis_positions": d["target_axis_positions"],
            "raw_positions": np.array(d["raw_sobol_values"]),
        }
        for d in artifact["descriptors"]
    ]

    return context, dimensions, stage1_results


def run_stage2_for_model(
    stage1_path: Path,
    persona_model: str,
    persona_format: str = "first_person",
    first_person_variant: str = "default",
    temperature: float = 0.5,
    output_dir: Path = Path("outputs"),
) -> Path:
    """Run Stage 2 expansion for a given model and save artifact."""
    context, dimensions, stage1_results = load_stage1_descriptors(stage1_path)

    output_dir.mkdir(exist_ok=True, parents=True)

    model_short = persona_model.split("/")[-1]
    print("=" * 70)
    print(f"STAGE 2: PARALLEL PERSONA EXPANSION ({persona_format})")
    print("=" * 70)
    print(f"\nModel: {persona_model} (short name: {model_short})")
    print(f"Temperature: {temperature}")
    print(f"Input Stage 1 file: {stage1_path}")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"\nExpanding {len(stage1_results)} descriptors into full personas...\n")

    # Temporarily override the persona model used inside persona_generator
    original_model = pg.PERSONA_MODEL
    pg.PERSONA_MODEL = persona_model

    personas: List[pg.Persona] = []
    try:
        for i, s1 in enumerate(stage1_results):
            print(f"  Expanding {i+1}/{len(stage1_results)}: {s1['name']}...")

            full_description = pg.expand_persona_stage2(
                name=s1["name"],
                descriptor=s1["descriptor"],
                axis_positions=s1["axis_positions"],
                context=context,
                persona_format=persona_format,
                first_person_variant=first_person_variant,
                temperature=temperature,
            )

            personas.append(
                pg.Persona(
                    name=s1["name"],
                    stage1_descriptor=s1["descriptor"],
                    full_description=full_description,
                    axis_positions=s1["axis_positions"],
                    persona_format=persona_format,
                )
            )
    finally:
        # Restore original model setting
        pg.PERSONA_MODEL = original_model

    print(f"\n[OK] Generated {len(personas)} complete personas")

    # Save Stage 2 artifact with model-specific filename.
    # Include the first_person variant in the filename when applicable,
    # and be robust to any partial / None descriptions so that saving never fails.
    if persona_format == "first_person":
        variant_suffix = f"_{first_person_variant}"
    else:
        variant_suffix = ""

    # Encode temperature in the filename, e.g. T0.9 for temperature=0.9
    temp_str = f"{temperature:.2f}".rstrip("0").rstrip(".")
    temp_suffix = f"T{temp_str}"

    output_path = output_dir / f"02c_stage2_personas_{model_short}{temp_suffix}{variant_suffix}.json"

    # Build persona entries and lengths defensively
    persona_entries = []
    lengths = []
    for i, p in enumerate(personas):
        text = p.full_description if p.full_description is not None else ""
        length = len(text)
        lengths.append(length)
        persona_entries.append(
            {
                "persona_index": i,
                "name": p.name,
                "target_axis_positions": p.axis_positions,
                "stage1_descriptor": p.stage1_descriptor,
                "stage2_full_description": text,
                "format": p.persona_format,
                "description_length_chars": length,
            }
        )

    statistics = {}
    if lengths:
        statistics = {
            "avg_description_length": float(np.mean(lengths)),
            "min_description_length": int(min(lengths)),
            "max_description_length": int(max(lengths)),
        }

    stage2_artifact = {
        "num_personas": len(personas),
        "persona_format": persona_format,
        "context": context,
        "dimensions": dimensions,
        "persona_model": persona_model,
        "personas": persona_entries,
        "statistics": statistics,
    }

    with output_path.open("w") as f:
        json.dump(stage2_artifact, f, indent=2, default=str)

    size_kb = output_path.stat().st_size / 1024
    print(f"  Saved: {output_path} ({size_kb:.1f} KB)")

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Stage 2 persona expansion from saved Stage 1 descriptors.\n"
            "Input:  outputs/02b_stage1_descriptors.json\n"
            "Output: outputs/02c_stage2_personas_{model_name}.json"
        )
    )
    parser.add_argument(
        "--persona-model",
        required=True,
        help=(
            "Model name to use for PERSONA_MODEL, e.g. "
            "'meta-llama/llama-3.1-70b-instruct'. "
            "This overrides the model from config.py."
        ),
    )
    parser.add_argument(
        "--base-model",
        help=(
            "Optional base model name. If provided, Stage 2 will also be run "
            "with this model and a separate output file will be created."
        ),
    )
    parser.add_argument(
        "--stage1-path",
        type=Path,
        default=Path("outputs") / "02b_stage1_descriptors.json",
        help="Path to Stage 1 descriptors JSON (default: outputs/02b_stage1_descriptors.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to write Stage 2 persona JSON files (default: outputs).",
    )
    parser.add_argument(
        "--persona-format",
        choices=["first_person", "logic_of_appropriateness", "rule_based"],
        default="first_person",
        help="Persona expansion format for Stage 2 (default: first_person).",
    )
    parser.add_argument(
        "--first-person-variant",
        choices=["default", "autobiographical", "fewshot"],
        default="default",
        help=(
            "Variant of the first_person prompt to use when persona-format is "
            "first_person: 'default', 'autobiographical', or 'fewshot'. "
            "Ignored for other formats."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help=(
            "Sampling temperature to use for Stage 2 LLM calls "
            "(default: 0.5). This value is also encoded into the "
            "output filename as 'T{temperature}'."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.stage1_path.exists():
        raise FileNotFoundError(
            f"Stage 1 descriptors file not found: {args.stage1_path}"
        )

    # Always run with the specified persona model
    run_stage2_for_model(
        stage1_path=args.stage1_path,
        persona_model=args.persona_model,
        persona_format=args.persona_format,
        first_person_variant=args.first_person_variant,
        temperature=args.temperature,
        output_dir=args.output_dir,
    )

    # Optionally also run with a base model for comparison
    if args.base_model:
        print("\n" + "#" * 70)
        print("RUNNING STAGE 2 WITH BASE MODEL")
        print("#" * 70 + "\n")
        run_stage2_for_model(
            stage1_path=args.stage1_path,
            persona_model=args.base_model,
            persona_format=args.persona_format,
            first_person_variant=args.first_person_variant,
            temperature=args.temperature,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()

