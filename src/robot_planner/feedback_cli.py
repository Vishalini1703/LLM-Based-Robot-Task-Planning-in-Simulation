"""CLI for validating and aggregating approved anonymous feedback."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .feedback import analyse_feedback_file


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m robot_planner.feedback_cli")
    parser.add_argument("input", type=Path)
    parser.add_argument("--invited", type=int, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/feedback/summary.json"),
    )
    args = parser.parse_args(argv)
    try:
        result = analyse_feedback_file(
            args.input,
            args.output,
            invited_count=args.invited,
        )
    except (OSError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
