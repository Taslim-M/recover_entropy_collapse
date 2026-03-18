"""
driver.py
=========
Loads a Stage-2 persona artifact (``02c_stage2_personas_*.json``) and computes
the Progressive Conditional Surprise diversity metrics via a remote vLLM endpoint.

Usage
-----
    python driver.py \\
        --stage2-file ../outputs/02c_stage2_personas_Olmo-3-32B-ThinkT0.5_autobiographical.json \\
        --base-url http://my-gpu-host:8000 \\
        --model meta-llama/Llama-3-8B

The context field of the Stage-2 artifact is used as the prompt.
All stage2_full_description strings are treated as a single policy whose name
matches the artifact's filename stem.

Output is saved to:
    ../outputs/05_diversity_metric_conditionalsurprise_{stage2_stem}.json

Pass --dry-run to verify the inputs without hitting the GPU.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from diversity_metrics import DiversityMetricsClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def load_stage2_artifact(path: Path) -> tuple[str, str, list[str]]:
    """Load a Stage-2 artifact and return (prompt, policy_name, responses).

    prompt      – the 'context' field from the artifact
    policy_name – the filename stem (e.g. '02c_stage2_personas_..._autobiographical')
    responses   – all non-empty stage2_full_description strings
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    prompt = data.get("context", "")
    if not prompt:
        raise ValueError(f"'context' field is missing or empty in {path}")

    responses = [
        p["stage2_full_description"]
        for p in data.get("personas", [])
        if p.get("stage2_full_description", "").strip()
    ]
    if not responses:
        raise ValueError(f"No non-empty stage2_full_description entries found in {path}")

    policy_name = path.stem  # e.g. 02c_stage2_personas_Olmo-3-32B-ThinkT0.5_autobiographical
    return prompt, policy_name, responses


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute Progressive Conditional Surprise diversity metrics for a Stage-2 persona artifact."
    )
    p.add_argument(
        "stage2_file",
        type=Path,
        help="Path to a 02c_stage2_personas_*.json artifact.",
    )
    p.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the vLLM server (default: http://localhost:8000)",
    )
    p.add_argument(
        "--model",
        default="meta-llama/Llama-3-8B",
        help="Model name registered on the vLLM server.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory to write results (default: the 'outputs/' sibling of the stage2 file). "
            "Output filename is always 05_diversity_metric_conditionalsurprise_{stem}.json."
        ),
    )
    p.add_argument(
        "--n-permutations",
        type=int,
        default=3,
        help="Number of random response orderings to average over (default: 3).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for permutation sampling (default: 42).",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="HTTP timeout in seconds for each vLLM request (default: 180).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be computed without hitting the API.",
    )
    return p


def resolve_output_path(stage2_file: Path, output_dir: Path | None) -> Path:
    stem = stage2_file.stem  # 02c_stage2_personas_..._autobiographical
    filename = f"05_diversity_metric_conditionalsurprise_{stem}.json"
    if output_dir is not None:
        return output_dir / filename
    # Default: same outputs/ directory that holds the stage2 file
    return stage2_file.parent / filename


def dry_run_summary(prompt: str, policy_name: str, responses: list[str]) -> None:
    print("\n[DRY RUN] Would compute diversity metrics for:\n")
    print(f"  Prompt  : {prompt[:120]}{'…' if len(prompt) > 120 else ''}")
    print(f"  Policy  : {policy_name}")
    print(f"  Responses: {len(responses)}")
    for i, r in enumerate(responses):
        preview = r[:100].replace("\n", " ")
        print(f"    [{i+1:>2}] {preview}{'…' if len(r) > 100 else ''}")
    print()


def main() -> None:
    args = build_parser().parse_args()

    if not args.stage2_file.exists():
        logger.error("Stage-2 file not found: %s", args.stage2_file)
        sys.exit(1)

    # Normalize base_url: bare host:port → http://host:port
    base_url = args.base_url
    if not base_url.startswith("http://") and not base_url.startswith("https://"):
        base_url = "http://" + base_url
        logger.info("No URL scheme provided; using %s", base_url)

    logger.info("Loading Stage-2 artifact: %s", args.stage2_file)
    prompt, policy_name, responses = load_stage2_artifact(args.stage2_file)
    logger.info("Loaded %d responses. Policy: %s", len(responses), policy_name)

    output_path = resolve_output_path(args.stage2_file, args.output_dir)

    if args.dry_run:
        dry_run_summary(prompt, policy_name, responses)
        logger.info("Output would be saved to: %s", output_path)
        return

    # ------------------------------------------------------------------
    # Compute metrics
    # ------------------------------------------------------------------
    client = DiversityMetricsClient(
        base_url=base_url,
        model=args.model,
        n_permutations=args.n_permutations,
        timeout=args.timeout,
    )

    logger.info("─" * 60)
    logger.info("Computing metrics for policy: %s  (%d responses)", policy_name, len(responses))
    result = client.compute(prompt=prompt, responses=responses, seed=args.seed)
    print(result.summary())

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    output_payload = {
        "stage2_file": args.stage2_file.name,
        "policy_name": policy_name,
        "n_responses": len(responses),
        "metrics": result.to_dict(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
