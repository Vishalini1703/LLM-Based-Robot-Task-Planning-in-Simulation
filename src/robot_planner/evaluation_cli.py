"""Command-line interface for freezing, running, and analysing the experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .evaluation import analyse_evaluation, freeze_settings, run_evaluation, run_pilot
from .settings import groq_api_key


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="robot-evaluate")
    subparsers = parser.add_subparsers(dest="operation", required=True)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--protocol", type=Path, default=Path("evaluation/protocol.json"))
    freeze.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/frozen-settings.json"),
    )

    pilot = subparsers.add_parser("pilot")
    pilot.add_argument("--catalog", type=Path, default=Path("evaluation/commands.json"))
    pilot.add_argument("--world", type=Path, default=Path("config/kitchen.json"))
    pilot.add_argument("--protocol", type=Path, default=Path("evaluation/protocol.json"))
    pilot.add_argument("--output", type=Path, default=Path("evaluation/pilot"))
    pilot.add_argument("--env-file", type=Path, default=Path(".env"))

    run = subparsers.add_parser("run")
    run.add_argument("--catalog", type=Path, default=Path("evaluation/commands.json"))
    run.add_argument("--world", type=Path, default=Path("config/kitchen.json"))
    run.add_argument(
        "--frozen",
        type=Path,
        default=Path("evaluation/frozen-settings.json"),
    )
    run.add_argument("--output", type=Path, default=Path("evaluation/results"))
    run.add_argument("--env-file", type=Path, default=Path(".env"))
    run.add_argument("--repetitions", type=int, default=5)
    run.add_argument("--backend", choices=("memory", "webots"), default="memory")
    run.add_argument(
        "--webots-runtime",
        choices=("native", "docker"),
        default="docker",
    )
    run.add_argument("--webots-bin", type=Path)
    run.add_argument("--restart", action="store_true")
    run.add_argument("--skip-tests", action="store_true")

    analyse = subparsers.add_parser("analyse")
    analyse.add_argument(
        "--trials",
        type=Path,
        default=Path("evaluation/results/trials"),
    )
    analyse.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/results/analysis"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.operation == "freeze":
            payload = freeze_settings(Path.cwd(), args.protocol, args.output)
        elif args.operation == "pilot":
            payload = run_pilot(
                project_root=Path.cwd(),
                catalog_path=args.catalog,
                world_path=args.world,
                protocol_path=args.protocol,
                output_directory=args.output,
                api_key=groq_api_key(args.env_file),
            )
        elif args.operation == "run":
            payload = run_evaluation(
                project_root=Path.cwd(),
                catalog_path=args.catalog,
                world_path=args.world,
                frozen_path=args.frozen,
                output_directory=args.output,
                api_key=groq_api_key(args.env_file),
                repetitions=args.repetitions,
                backend=args.backend,
                webots_runtime=args.webots_runtime,
                webots_bin=args.webots_bin,
                resume=not args.restart,
                run_preflight_tests=not args.skip_tests,
            )
        else:
            payload = analyse_evaluation(args.trials, args.output)
    except (OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
