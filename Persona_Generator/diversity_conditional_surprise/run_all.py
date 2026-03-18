"""
run_all.py
==========
Run driver.py for every 02c_stage2_personas_*.json file found in the outputs
directory, skipping files whose result already exists.

Usage
-----
    python run_all.py \\
        --base-url 154.59.156.5:31330 \\
        --model allenai/Olmo-3-1125-32B

Optional flags are forwarded directly to driver.py.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DRIVER = Path(__file__).parent / "driver.py"
DEFAULT_OUTPUTS = Path(__file__).parent.parent / "outputs"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", required=True, help="vLLM server base URL or host:port.")
    p.add_argument("--model", required=True, help="Model name on the vLLM server.")
    p.add_argument(
        "--outputs-dir", type=Path, default=DEFAULT_OUTPUTS,
        help=f"Directory containing 02c_stage2_personas_*.json files (default: {DEFAULT_OUTPUTS}).",
    )
    p.add_argument(
        "--n-permutations", type=int, default=3,
        help="Number of permutations per file (default: 3).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument(
        "--skip-existing", action="store_true", default=True,
        help="Skip files whose output already exists (default: true).",
    )
    p.add_argument(
        "--no-skip-existing", dest="skip_existing", action="store_false",
        help="Re-run even if output already exists.",
    )
    p.add_argument("--dry-run", action="store_true", help="Pass --dry-run to driver.py.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    stage2_files = sorted(args.outputs_dir.glob("02c_stage2_personas_*.json"))
    if not stage2_files:
        print(f"No 02c_stage2_personas_*.json files found in {args.outputs_dir}")
        sys.exit(1)

    print(f"Found {len(stage2_files)} stage-2 file(s)\n")

    passed, skipped, failed = 0, 0, 0

    for stage2_file in stage2_files:
        stem = stage2_file.stem
        out_file = args.outputs_dir / f"05_diversity_metric_conditionalsurprise_{stem}.json"

        if args.skip_existing and out_file.exists() and not args.dry_run:
            print(f"  [SKIP] {stage2_file.name}")
            skipped += 1
            continue

        print(f"  [RUN ] {stage2_file.name}")

        cmd = [
            sys.executable, str(DRIVER),
            str(stage2_file),
            "--base-url", args.base_url,
            "--model", args.model,
            "--n-permutations", str(args.n_permutations),
            "--seed", str(args.seed),
            "--timeout", str(args.timeout),
        ]
        if args.dry_run:
            cmd.append("--dry-run")

        result = subprocess.run(cmd, cwd=DRIVER.parent)

        if result.returncode == 0:
            print(f"         -> OK: {out_file.name}\n")
            passed += 1
        else:
            print(f"         -> FAILED (exit code {result.returncode})\n")
            failed += 1

    print("=" * 60)
    print(f"Done.  passed={passed}  skipped={skipped}  failed={failed}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
