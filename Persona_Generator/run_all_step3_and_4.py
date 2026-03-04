"""
Run Step 3 (Concordia evaluation) and Step 4 (diversity metrics) for every
Stage 2c personas file (02c_stage2_*.json) in the outputs directory, using
a single shared questionnaire file.

Usage (from Persona_Generator directory):

    python run_all_step3_and_4.py

    python run_all_step3_and_4.py --outputs-dir path/to/outputs
    python run_all_step3_and_4.py --questionnaire path/to/01_questionnaire.json
"""

import argparse
from pathlib import Path

from run_step3_and_4_from_stage2 import run_from_stage2


def main():
    parser = argparse.ArgumentParser(
        description="Run step 3 and 4 for all 02c_stage2_*.json files using a shared questionnaire."
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory containing 02c_stage2_*.json files (default: outputs).",
    )
    parser.add_argument(
        "--questionnaire",
        type=Path,
        default=None,
        help=(
            "Path to questionnaire JSON (default: <outputs-dir>/01_questionnaire.json)."
        ),
    )
    args = parser.parse_args()

    outputs_dir = args.outputs_dir
    if not outputs_dir.is_dir():
        raise FileNotFoundError(f"Outputs directory not found: {outputs_dir}")

    questionnaire_path = args.questionnaire
    if questionnaire_path is None:
        questionnaire_path = outputs_dir / "01_questionnaire.json"
    if not questionnaire_path.exists():
        raise FileNotFoundError(f"Questionnaire file not found: {questionnaire_path}")

    stage2_files = sorted(outputs_dir.glob("02c_stage2_*.json"))
    if not stage2_files:
        print(f"No files matching 02c_stage2_*.json in {outputs_dir.resolve()}")
        return

    print(f"Found {len(stage2_files)} Stage 2 file(s). Questionnaire: {questionnaire_path}")
    print("=" * 70)

    for i, stage2_path in enumerate(stage2_files, 1):
        print(f"\n[{i}/{len(stage2_files)}] Processing: {stage2_path.name}")
        run_from_stage2(stage2_path, questionnaire_path)

    print("\n" + "=" * 70)
    print("All runs completed.")


if __name__ == "__main__":
    main()
