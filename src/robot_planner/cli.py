"""Command-line entry point for validating a plan file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .errors import WorldConfigurationError
from .validator import PlanValidator
from .world import load_world


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="robot-plan",
        description="Validate a structured robot task plan against a constrained kitchen state.",
    )
    parser.add_argument("plan", type=Path, help="Path to the plan JSON file.")
    parser.add_argument(
        "--world",
        type=Path,
        default=Path("config/kitchen.json"),
        help="Path to the kitchen world JSON file (default: config/kitchen.json).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        world = load_world(args.world)
        with args.plan.open("r", encoding="utf-8") as stream:
            raw_plan = json.load(stream)
    except (OSError, json.JSONDecodeError, WorldConfigurationError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        return 2

    result = PlanValidator().validate_mapping(raw_plan, world)
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())

