"""Natural-language command interface backed by Groq and local validation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from .llm import GroqPlanner, PlannerError
from .pipeline import TaskPlanningPipeline, save_pipeline_result
from .settings import SettingsError, groq_api_key
from .webots_executor import WebotsExecutionError, WebotsExecutor
from .world import load_world


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="robot-task",
        description="Turn one kitchen command into a validated high-level task plan.",
    )
    parser.add_argument("command", help="Plain-English kitchen command.")
    parser.add_argument("--world", type=Path, default=Path("config/kitchen.json"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--model", default="llama-3.3-70b-versatile")
    parser.add_argument("--backend", choices=("memory", "webots"), default="memory")
    parser.add_argument(
        "--show-webots",
        action="store_true",
        help="Run Webots visibly in realtime instead of minimized fast mode.",
    )
    parser.add_argument(
        "--webots-bin",
        type=Path,
        default=None,
        help=(
            "Webots executable or bin directory. If omitted, use WEBOTS_BIN, "
            "WEBOTS_HOME, or the system PATH."
        ),
    )
    parser.add_argument(
        "--webots-runtime",
        choices=("native", "docker"),
        default=os.environ.get("WEBOTS_RUNTIME", "native"),
        help=(
            "Use a local Webots installation or the Docker/Xvfb headless runtime. "
            "Defaults to WEBOTS_RUNTIME or native."
        ),
    )
    parser.add_argument(
        "--webots-docker-image",
        default=os.environ.get("WEBOTS_DOCKER_IMAGE"),
        help="Override the official Webots Docker image tag.",
    )
    parser.add_argument("--log-directory", type=Path, default=Path("logs/runs"))
    parser.add_argument("--no-save", action="store_true", help="Do not persist the run JSON.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        key = groq_api_key(args.env_file)
        world = load_world(args.world)
        executor = (
            WebotsExecutor(
                args.webots_bin,
                Path.cwd(),
                headless=not args.show_webots,
                runtime=args.webots_runtime,
                docker_image=args.webots_docker_image,
            )
            if args.backend == "webots"
            else None
        )
        result = TaskPlanningPipeline(
            GroqPlanner(key, model=args.model), executor=executor
        ).run(args.command, world)
    except (
        OSError,
        ValueError,
        SettingsError,
        PlannerError,
        WebotsExecutionError,
    ) as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        return 2

    payload = result.to_dict()
    if not args.no_save:
        payload["saved_to"] = str(save_pipeline_result(result, args.log_directory))
    print(json.dumps(payload, indent=2))
    return 0 if result.status == "executed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
