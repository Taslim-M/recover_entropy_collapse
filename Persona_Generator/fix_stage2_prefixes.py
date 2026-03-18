"""
One-time script to prepend the completion-starter prefix to stage2_full_description
in all existing 02c_stage2_personas_*.json output files.

The Stage 2 prompts end mid-sentence (e.g. "I am {name},") so the LLM continues
from there.  Previously the saved description omitted that starter text.
This script adds it back where it is missing.

Run from the Persona_Generator/ directory:
    python fix_stage2_prefixes.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# ── Filename pattern ──────────────────────────────────────────────────────────
# Captures model-short-name, temperature, and prompt-variant from filenames like:
#   02c_stage2_personas_OLMo-3-1125-32BT0.5_autobiographical.json
_FNAME_RE = re.compile(
    r"^02c_stage2_personas_.+?T[0-9]*\.?[0-9]+_(?P<variant>[^.]+)\.json$"
)

# ── Prefix logic (mirrors persona_generator._completion_prefix) ───────────────

def _prefix(persona_format: str, variant: str, name: str) -> str:
    if persona_format == "first_person":
        if variant == "autobiographical":
            return f"I'm {name}\n"
        elif variant == "fewshot":
            return ""
        else:  # default
            return f"I am {name}, "
    elif persona_format == "logic_of_appropriateness":
        return f"{name} "
    elif persona_format == "rule_based":
        return "If "
    return ""


def fix_file(path: Path, dry_run: bool) -> int:
    """Prepend missing prefix to each persona in *path*.  Returns number patched."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    m = _FNAME_RE.match(path.name)
    if not m:
        print(f"  SKIP (filename does not match pattern): {path.name}")
        return 0

    variant = m.group("variant")  # autobiographical / default / fewshot
    persona_format = data.get("persona_format", "first_person")

    patched = 0
    for entry in data.get("personas", []):
        name = entry.get("name", "")
        desc = entry.get("stage2_full_description", "")
        want_prefix = _prefix(persona_format, variant, name)

        if not want_prefix:
            continue  # fewshot or unknown – nothing to prepend

        if desc.startswith(want_prefix):
            continue  # already correct

        new_desc = want_prefix + desc
        entry["stage2_full_description"] = new_desc
        entry["description_length_chars"] = len(new_desc)
        patched += 1

    if patched == 0:
        print(f"  OK (no changes needed): {path.name}")
        return 0

    # Recompute summary statistics
    lengths = [len(e.get("stage2_full_description", "")) for e in data.get("personas", [])]
    if lengths:
        import numpy as np
        data["statistics"] = {
            "avg_description_length": float(np.mean(lengths)),
            "min_description_length": int(min(lengths)),
            "max_description_length": int(max(lengths)),
        }

    if dry_run:
        print(f"  DRY-RUN would patch {patched} persona(s): {path.name}")
    else:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"  Patched {patched} persona(s): {path.name}")

    return patched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would change without writing files."
    )
    parser.add_argument(
        "--outputs-dir", type=Path,
        default=Path(__file__).parent / "outputs",
        help="Directory containing 02c_stage2_personas_*.json files."
    )
    args = parser.parse_args()

    files = sorted(args.outputs_dir.glob("02c_stage2_personas_*.json"))
    if not files:
        print(f"No 02c_stage2_personas_*.json files found in {args.outputs_dir}")
        sys.exit(1)

    print(f"Found {len(files)} file(s) in {args.outputs_dir}\n")
    total_patched = 0
    for p in files:
        total_patched += fix_file(p, dry_run=args.dry_run)

    action = "Would patch" if args.dry_run else "Patched"
    print(f"\n{action} {total_patched} persona(s) across {len(files)} file(s).")


if __name__ == "__main__":
    main()
