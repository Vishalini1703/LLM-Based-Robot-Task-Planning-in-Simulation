"""Command-line workflow for the project-owned KitchenObjectNet detector."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .webots_executor import (
    discover_docker_executable,
    discover_webots_executable,
)


DEFAULT_RUN_ROOT = Path("cnn_artifacts/kitchen_object_net_v1")
DEFAULT_DATASET_DOCKER_IMAGE = "cyberbotics/webots:R2025a-ubuntu22.04"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="robot-cnn",
        description=(
            "Generate Webots images, train the custom KitchenObjectNet from "
            "random initialization, and retain all evidence in one run folder."
        ),
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    subparsers = parser.add_subparsers(dest="operation", required=True)

    generate = subparsers.add_parser("generate", help="Generate labelled Webots RGB images.")
    generate.add_argument("--count", type=int, default=8000)
    generate.add_argument("--seed", type=int, default=20260808)
    generate.add_argument("--runtime", choices=("native", "docker"), default="native")
    generate.add_argument("--webots-bin", type=Path)
    generate.add_argument("--docker-bin", type=Path)
    generate.add_argument("--docker-image", default=DEFAULT_DATASET_DOCKER_IMAGE)
    generate.add_argument("--timeout", type=int, default=21600)
    generate.add_argument("--force", action="store_true")

    subparsers.add_parser("audit", help="Audit the fixed 8,000-image dataset.")
    subparsers.add_parser(
        "protocol", help="Run the frozen 180-episode held-out perception protocol."
    )
    e2e = subparsers.add_parser(
        "e2e", help="Run 25 CNN-backed manipulation episodes in native Webots."
    )
    e2e.add_argument("--webots-bin", type=Path)
    e2e.add_argument("--repetitions", type=int, default=5)
    e2e.add_argument("--timeout", type=int, default=3600)
    subparsers.add_parser("verify", help="Audit final hashes, provenance, CPU inference, logs, and plots.")
    train = subparsers.add_parser("train", help="Train/evaluate/export KitchenObjectNet.")
    train.add_argument("--epochs", type=int, default=120)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--seed", type=int, default=20260808)
    train.add_argument("--patience", type=int, default=15)
    return parser


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _generate(args: argparse.Namespace, project_root: Path, run_root: Path) -> dict:
    if args.count <= 0:
        raise ValueError("--count must be positive")
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    dataset_root = run_root / "dataset"
    if dataset_root.exists():
        if not args.force:
            raise FileExistsError(
                f"Dataset already exists: {dataset_root}. Use --force to replace it."
            )
        resolved_dataset = dataset_root.resolve()
        resolved_run = run_root.resolve()
        if resolved_dataset.parent != resolved_run:
            raise RuntimeError("Refusing to replace a dataset outside the selected run root.")
        shutil.rmtree(resolved_dataset)
    logs = run_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    world = project_root / "webots" / "worlds" / "kitchen_llm.wbt"
    if not world.is_file():
        raise FileNotFoundError(f"Webots world not found: {world}")
    result_path = logs / "dataset_result.json"
    controller_log = logs / "dataset_controller.jsonl"
    stdout_path = logs / "dataset_webots_stdout.log"
    stderr_path = logs / "dataset_webots_stderr.log"
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(project_root / "src"),
            "ROBOT_TASK_MODE": "cnn_dataset",
            "ROBOT_CNN_DATASET_DIR": str(dataset_root),
            "ROBOT_CNN_DATASET_COUNT": str(args.count),
            "ROBOT_CNN_DATASET_SEED": str(args.seed),
            "ROBOT_TASK_RESULT_FILE": str(result_path),
            "ROBOT_TASK_LOG_FILE": str(controller_log),
            "ROBOT_TASK_SCREENSHOT_DIR": str(logs / "dataset_screenshots"),
        }
    )
    container_name = None
    if args.runtime == "native":
        executable = discover_webots_executable(args.webots_bin)
        command = [
            str(executable),
            "--batch",
            "--mode=fast",
            "--no-rendering",
            "--minimize",
            "--stdout",
            "--stderr",
            str(world),
        ]
    else:
        docker = discover_docker_executable(args.docker_bin)
        container_name = f"robot-cnn-dataset-{uuid.uuid4().hex[:10]}"
        mount = f"type=bind,source={project_root},target=/project"
        container_dataset = "/project/" + dataset_root.relative_to(project_root).as_posix()
        container_logs = "/project/" + logs.relative_to(project_root).as_posix()
        inner = (
            "set -eu; Xvfb :99 -screen 0 1024x768x24 -nolisten tcp "
            f">{container_logs}/dataset_xvfb.log 2>&1 & export DISPLAY=:99; "
            "attempt=0; while [ ! -S /tmp/.X11-unix/X99 ]; do "
            "attempt=$((attempt + 1)); if [ \"$attempt\" -ge 50 ]; then exit 1; fi; "
            "sleep 0.1; done; exec webots --batch --mode=fast --no-rendering "
            "--minimize --stdout --stderr /project/webots/worlds/kitchen_llm.wbt"
        )
        command = [
            str(docker), "run", "--rm", "--name", container_name,
            "--mount", mount, "--workdir", "/project",
            "-e", "PYTHONPATH=/project/src", "-e", "LIBGL_ALWAYS_SOFTWARE=1",
            "-e", "ROBOT_TASK_MODE=cnn_dataset",
            "-e", f"ROBOT_CNN_DATASET_DIR={container_dataset}",
            "-e", f"ROBOT_CNN_DATASET_COUNT={args.count}",
            "-e", f"ROBOT_CNN_DATASET_SEED={args.seed}",
            "-e", f"ROBOT_TASK_RESULT_FILE={container_logs}/dataset_result.json",
            "-e", f"ROBOT_TASK_LOG_FILE={container_logs}/dataset_controller.jsonl",
            "-e", f"ROBOT_TASK_SCREENSHOT_DIR={container_logs}/dataset_screenshots",
            args.docker_image, "/bin/sh", "-lc", inner,
        ]
    launch_record = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "runtime": args.runtime,
        "count": args.count,
        "seed": args.seed,
        "command": command,
    }
    _write_json(logs / "dataset_launch.json", launch_record)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            command,
            cwd=project_root,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
        launch_record["pid"] = process.pid
        _write_json(logs / "dataset_launch.json", launch_record)
        try:
            return_code = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired as exc:
            if container_name is not None:
                subprocess.run(
                    [str(discover_docker_executable(args.docker_bin)), "rm", "--force", container_name],
                    capture_output=True,
                    text=True,
                    check=False,
                )
            else:
                process.terminate()
                process.wait(timeout=30)
            raise TimeoutError(f"Webots dataset generation exceeded {args.timeout}s") from exc
    launch_record.update(
        {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "return_code": return_code,
        }
    )
    _write_json(logs / "dataset_launch.json", launch_record)
    if not result_path.is_file():
        stderr_tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-3000:]
        raise RuntimeError(
            f"Webots exited {return_code} without a dataset result. stderr: {stderr_tail}"
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if return_code != 0 or result.get("status") != "completed":
        raise RuntimeError(f"Webots dataset generation failed: {result}")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    project_root = Path.cwd().resolve()
    run_root = args.run_root.resolve()
    try:
        if args.operation == "generate":
            result = _generate(args, project_root, run_root)
        elif args.operation == "audit":
            from .cnn_data import audit_dataset, build_dataset_manifest

            result = audit_dataset(run_root / "dataset")
            if result["status"] != "passed":
                raise RuntimeError("Dataset audit failed: " + "; ".join(result["problems"]))
            _write_json(run_root / "dataset_audit.json", result)
            _write_json(
                run_root / "dataset_manifest.json",
                build_dataset_manifest(
                    run_root / "dataset",
                    project_root / "webots" / "worlds" / "kitchen_llm.wbt",
                    result,
                ),
            )
        elif args.operation == "train":
            from .cnn_training import train_kitchen_object_net

            result = train_kitchen_object_net(
                run_root / "dataset",
                run_root,
                epochs=args.epochs,
                batch_size=args.batch_size,
                seed=args.seed,
                patience=args.patience,
            )
        elif args.operation == "protocol":
            from .cnn_protocol import run_perception_protocol

            result = run_perception_protocol(run_root)
        elif args.operation == "e2e":
            from .cnn_e2e import run_e2e_batch

            result = run_e2e_batch(
                run_root,
                project_root,
                args.webots_bin,
                repetitions=args.repetitions,
                timeout=args.timeout,
            )
        else:
            from .cnn_verify import verify_cnn_artifacts

            result = verify_cnn_artifacts(run_root, project_root)
    except (OSError, RuntimeError, ValueError, TimeoutError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2), file=sys.stderr)
        return 2
    displayed = dict(result)
    if args.operation == "audit" and "image_sha256" in displayed:
        displayed["image_sha256"] = (
            f"retained in {run_root / 'dataset_audit.json'} "
            f"({len(result['image_sha256'])} hashes)"
        )
    print(json.dumps(displayed, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
