"""Native Webots batch verification for 25 CNN-backed manipulation tasks."""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .webots_executor import discover_webots_executable


def _steps_for(object_id: str) -> list[dict[str, Any]]:
    if object_id in {"mug", "can"}:
        return [
            {"action": "navigate", "arguments": {"location": "shelf"}},
            {"action": "find", "arguments": {"object": object_id}},
            {"action": "pick", "arguments": {"object": object_id}},
            {"action": "navigate", "arguments": {"location": "cupboard"}},
            {"action": "open", "arguments": {"container": "cupboard"}},
            {
                "action": "place",
                "arguments": {"object": object_id, "location": "cupboard"},
            },
            {"action": "close", "arguments": {"container": "cupboard"}},
        ]
    if object_id in {"apple", "orange"}:
        return [
            {"action": "find", "arguments": {"object": object_id}},
            {"action": "pick", "arguments": {"object": object_id}},
            {"action": "navigate", "arguments": {"location": "basket"}},
            {
                "action": "place",
                "arguments": {"object": object_id, "location": "basket"},
            },
        ]
    return [
        {"action": "navigate", "arguments": {"location": "shelf"}},
        {"action": "find", "arguments": {"object": "cereal_box"}},
        {"action": "pick", "arguments": {"object": "cereal_box"}},
        {
            "action": "place",
            "arguments": {"object": "cereal_box", "location": "shelf"},
        },
    ]


def run_e2e_batch(
    run_root: str | Path,
    project_root: str | Path,
    webots_bin: str | Path | None,
    *,
    repetitions: int = 5,
    timeout: int = 3600,
) -> dict[str, Any]:
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    root = Path(run_root).resolve()
    project = Path(project_root).resolve()
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    plans = []
    for object_id in ("mug", "apple", "orange", "can", "cereal_box"):
        for repetition in range(1, repetitions + 1):
            plans.append(
                {
                    "plan_id": f"cnn-e2e-{object_id}-{repetition:02d}",
                    "goal": f"CNN-backed manipulation test for {object_id}",
                    "steps": _steps_for(object_id),
                }
            )
    plans_path = logs / "e2e_plans.json"
    result_path = logs / "e2e_result.json"
    controller_log = logs / "e2e_controller.jsonl"
    stdout_path = logs / "e2e_webots_stdout.log"
    stderr_path = logs / "e2e_webots_stderr.log"
    episode_root = logs / "e2e_episodes"
    episode_root.mkdir(parents=True, exist_ok=True)
    plans_path.write_text(json.dumps(plans, indent=2), encoding="utf-8")
    controller_log.write_text("", encoding="utf-8")
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    executable = discover_webots_executable(webots_bin)
    world = project / "webots" / "worlds" / "kitchen_llm.wbt"
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
    launch = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "episode_count": len(plans),
        "isolation": "fresh Webots process and freshly loaded world per episode",
        "episodes": [],
    }
    deadline = time.monotonic() + timeout
    episodes = []
    for episode_index, plan in enumerate(plans, start=1):
        directory = episode_root / f"{episode_index:03d}_{plan['plan_id']}"
        directory.mkdir(parents=True, exist_ok=True)
        plan_path = directory / "plan.json"
        episode_result_path = directory / "result.json"
        episode_controller_log = directory / "controller.jsonl"
        episode_stdout = directory / "webots_stdout.log"
        episode_stderr = directory / "webots_stderr.log"
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        environment = os.environ.copy()
        environment.pop("ROBOT_TASK_MODE", None)
        environment.pop("ROBOT_TASK_BATCH_FILE", None)
        environment.update(
            {
                "PYTHONPATH": str(project / "src"),
                "ROBOT_TASK_PERCEPTION_BACKEND": "cnn",
                "ROBOT_TASK_PLAN_FILE": str(plan_path),
                "ROBOT_TASK_RESULT_FILE": str(episode_result_path),
                "ROBOT_TASK_LOG_FILE": str(episode_controller_log),
                "ROBOT_TASK_SCREENSHOT_DIR": str(
                    root / "e2e_screenshots" / f"episode_{episode_index:03d}"
                ),
                "ROBOT_CNN_MODEL": str(root / "KitchenObjectNet.onnx"),
            }
        )
        with episode_stdout.open("w", encoding="utf-8") as stdout, episode_stderr.open(
            "w", encoding="utf-8"
        ) as stderr:
            process = subprocess.Popen(
                command,
                cwd=project,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                text=True,
            )
            episode_launch = {
                "episode": episode_index,
                "plan_id": plan["plan_id"],
                "pid": process.pid,
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
            launch["episodes"].append(episode_launch)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                process.terminate()
                raise TimeoutError(f"CNN end-to-end protocol exceeded {timeout}s")
            try:
                return_code = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired as exc:
                process.terminate()
                raise TimeoutError(
                    f"CNN end-to-end episode {episode_index} exceeded the protocol timeout"
                ) from exc
        episode_launch.update(
            {
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "return_code": return_code,
            }
        )
        (logs / "e2e_launch.json").write_text(
            json.dumps(launch, indent=2), encoding="utf-8"
        )
        if episode_controller_log.is_file():
            with controller_log.open("a", encoding="utf-8") as combined:
                combined.write(episode_controller_log.read_text(encoding="utf-8"))
        for source, combined_path, heading in (
            (episode_stdout, stdout_path, "stdout"),
            (episode_stderr, stderr_path, "stderr"),
        ):
            with combined_path.open("a", encoding="utf-8") as combined:
                combined.write(
                    f"\n===== episode {episode_index:03d} {plan['plan_id']} {heading} =====\n"
                )
                combined.write(source.read_text(encoding="utf-8"))
        if not episode_result_path.is_file():
            episodes.append(
                {
                    "status": "failed",
                    "plan_id": plan["plan_id"],
                    "code": "MISSING_WEBOTS_RESULT",
                    "error": f"Webots produced no result (exit {return_code})",
                }
            )
        else:
            episode = json.loads(episode_result_path.read_text(encoding="utf-8"))
            episode["webots_return_code"] = return_code
            episodes.append(episode)
    passed = sum(episode.get("status") == "completed" for episode in episodes)
    result = {
        "status": "completed" if passed == len(episodes) else "failed",
        "isolation": "fresh Webots process and freshly loaded world per episode",
        "episode_count": len(episodes),
        "passed": passed,
        "failed": len(episodes) - passed,
        "episodes": episodes,
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    launch["completed_at"] = datetime.now(timezone.utc).isoformat()
    (logs / "e2e_launch.json").write_text(
        json.dumps(launch, indent=2), encoding="utf-8"
    )
    if result.get("status") != "completed" or result.get("passed") != len(plans):
        raise RuntimeError(f"CNN isolated end-to-end protocol failed: {result}")
    return result
