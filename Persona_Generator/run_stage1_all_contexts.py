"""
Driver: Run Stage 1 pipeline for every context in context.json

For each context entry, creates outputs/context_{id}/ and generates:
  01_questionnaire.json
  02a_diversity_positions.json
  02b_stage1_descriptors.json

Usage:
    python run_stage1_all_contexts.py
    python run_stage1_all_contexts.py --contexts-file outputs/context.json
    python run_stage1_all_contexts.py --start-from 5        # resume from context_id 5
    python run_stage1_all_contexts.py --only 3 7            # run only context_ids 3 and 7
"""

import argparse
import json
import sys
import os
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from config import NUM_PERSONAS, LIKERT_SCALE
from questionnaire_generator import generate_questionnaire
from persona_generator import generate_stage1_descriptors
from diversity_sampler import generate_diversity_positions, positions_to_labels


# ---------------------------------------------------------------------------
# Saving helpers
# ---------------------------------------------------------------------------

def save_questionnaire(questionnaire, output_dir: Path) -> None:
    """Save questionnaire as 01_questionnaire.json."""
    questions_per_dim = {}
    for q in questionnaire.questions:
        questions_per_dim[q.dimension] = questions_per_dim.get(q.dimension, 0) + 1

    artifact = {
        "short_description": questionnaire.short_description,
        "context": questionnaire.context,
        "dimensions": questionnaire.dimensions,
        "num_dimensions": len(questionnaire.dimensions),
        "num_questions": len(questionnaire.questions),
        "questions_per_dimension": questions_per_dim,
        "questions": [
            {
                "preprompt": q.preprompt,
                "statement": q.statement,
                "dimension": q.dimension,
                "ascending_scale": q.ascending_scale,
                "choices": q.choices,
            }
            for q in questionnaire.questions
        ],
    }

    path = output_dir / "01_questionnaire.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)
    print(f"  Saved {path}")


def save_diversity_positions(
    positions: np.ndarray,
    dimensions: list[str],
    labeled: list[dict],
    seed: int,
    output_dir: Path,
) -> None:
    """Save diversity positions as 02a_diversity_positions.json."""
    artifact = {
        "num_personas": positions.shape[0],
        "num_dimensions": positions.shape[1],
        "dimensions": dimensions,
        "sampling_method": "sobol_quasi_random",
        "seed": seed,
        "positions": [],
        "statistics": {},
    }

    for i in range(positions.shape[0]):
        artifact["positions"].append({
            "persona_index": i,
            "raw_values": positions[i].tolist(),
            "labeled": labeled[i],
        })

    for j, dim in enumerate(dimensions):
        col = positions[:, j]
        artifact["statistics"][dim] = {
            "min": float(np.min(col)),
            "max": float(np.max(col)),
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
        }

    path = output_dir / "02a_diversity_positions.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)
    print(f"  Saved {path}")


def save_stage1_descriptors(
    descriptors: list[dict],
    context: str,
    dimensions: list[str],
    batch_size: int,
    output_dir: Path,
) -> None:
    """Save stage 1 descriptors as 02b_stage1_descriptors.json."""
    artifact = {
        "num_personas": len(descriptors),
        "context_provided": context,
        "dimensions_provided": dimensions,
        "items_provided": "NONE \u2014 held back for evaluation",
        "batch_size": batch_size,
        "descriptors": [
            {
                "persona_index": i,
                "name": d["name"],
                "descriptor": d["descriptor"],
                "target_axis_positions": d["axis_positions"],
                "raw_sobol_values": d["raw_positions"],
            }
            for i, d in enumerate(descriptors)
        ],
    }

    path = output_dir / "02b_stage1_descriptors.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Per-context pipeline
# ---------------------------------------------------------------------------

def run_context(
    context_id: int,
    short_description: str,
    outputs_root: Path,
    num_personas: int = NUM_PERSONAS,
    seed: int = 42,
    batch_size: int = 5,
) -> None:
    """Run Steps 1, 2a, 2b for a single context and save outputs."""
    context_dir = outputs_root / f"context_{context_id}"
    context_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"CONTEXT {context_id}: {short_description}")
    print(f"Output dir: {context_dir}")
    print(f"{'='*70}")

    t0 = time.time()

    # -- Step 1: Questionnaire --
    print(f"\n[Step 1] Generating questionnaire for '{short_description}'...")
    questionnaire = generate_questionnaire(short_description)
    save_questionnaire(questionnaire, context_dir)

    # -- Step 2a: Diversity positions --
    print(f"\n[Step 2a] Sampling {num_personas} diversity positions...")
    positions = generate_diversity_positions(
        num_personas=num_personas,
        num_dimensions=len(questionnaire.dimensions),
        seed=seed,
    )
    labeled = positions_to_labels(positions, questionnaire.dimensions)
    save_diversity_positions(positions, questionnaire.dimensions, labeled, seed, context_dir)

    # -- Step 2b: Stage 1 descriptors --
    print(f"\n[Step 2b] Generating stage-1 descriptors...")
    descriptors = generate_stage1_descriptors(
        context=questionnaire.context,
        dimensions=questionnaire.dimensions,
        num_personas=num_personas,
        batch_size=batch_size,
        seed=seed,
    )
    save_stage1_descriptors(descriptors, questionnaire.context, questionnaire.dimensions, batch_size, context_dir)

    elapsed = time.time() - t0
    print(f"\n  Context {context_id} completed in {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run Stage 1 pipeline (01, 02a, 02b) for every context in context.json.",
    )
    p.add_argument(
        "--contexts-file",
        type=Path,
        default=Path("outputs") / "context.json",
        help="Path to context.json (default: outputs/context.json).",
    )
    p.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Root output directory (default: outputs/).",
    )
    p.add_argument(
        "--num-personas",
        type=int,
        default=NUM_PERSONAS,
        help=f"Number of personas per context (default: {NUM_PERSONAS}).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for Sobol sampling (default: 42).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Stage-1 autoregressive batch size (default: 5).",
    )
    p.add_argument(
        "--start-from",
        type=int,
        default=None,
        help="Skip contexts with context_id < this value (for resuming).",
    )
    p.add_argument(
        "--only",
        type=int,
        nargs="+",
        default=None,
        help="Run only these context_ids (e.g. --only 3 7).",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    if not args.contexts_file.exists():
        print(f"Error: contexts file not found: {args.contexts_file}", file=sys.stderr)
        sys.exit(1)

    with open(args.contexts_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    contexts = data["contexts"]
    print(f"Loaded {len(contexts)} contexts from {args.contexts_file}")

    # Filter contexts
    if args.only:
        contexts = [c for c in contexts if c["context_id"] in args.only]
        print(f"  Filtered to context_ids: {args.only}")
    elif args.start_from:
        contexts = [c for c in contexts if c["context_id"] >= args.start_from]
        print(f"  Starting from context_id >= {args.start_from}")

    if not contexts:
        print("No contexts to process.")
        return

    total_start = time.time()

    for entry in contexts:
        ctx_id = entry["context_id"]
        short_desc = entry["context"]

        # Skip if all three output files already exist
        ctx_dir = args.outputs_dir / f"context_{ctx_id}"
        expected = ["01_questionnaire.json", "02a_diversity_positions.json", "02b_stage1_descriptors.json"]
        if all((ctx_dir / f).exists() for f in expected):
            print(f"\nSkipping context {ctx_id} — all outputs already exist in {ctx_dir}")
            continue

        run_context(
            context_id=ctx_id,
            short_description=short_desc,
            outputs_root=args.outputs_dir,
            num_personas=args.num_personas,
            seed=args.seed,
            batch_size=args.batch_size,
        )

    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"All contexts completed in {total_elapsed:.1f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
