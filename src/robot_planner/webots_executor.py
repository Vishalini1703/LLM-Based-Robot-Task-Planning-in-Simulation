"""Launcher and result adapter for the Webots R2025a kitchen controller."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import Plan, WorldState
from .validator import PlanValidator


class WebotsExecutionError(RuntimeError):
    """Raised when Webots cannot start or does not complete the validated plan."""


DEFAULT_WEBOTS_DOCKER_IMAGE = "robot-task-webots-cnn:R2025a"


def discover_webots_executable(configured: str | Path | None = None) -> Path:
    """Locate Webots without assuming an installation drive or graphics device."""
    candidates: list[Path] = []
    configured_value = configured or os.environ.get("WEBOTS_BIN")
    if configured_value:
        configured_path = Path(configured_value).expanduser()
        candidates.extend(
            (
                configured_path,
                configured_path / "webots.exe",
                configured_path / "webots",
            )
        )

    webots_home_value = os.environ.get("WEBOTS_HOME")
    if webots_home_value:
        webots_home = Path(webots_home_value).expanduser()
        candidates.extend(
            (
                webots_home / "msys64" / "mingw64" / "bin" / "webots.exe",
                webots_home / "bin" / "webots",
                webots_home / "Contents" / "MacOS" / "webots",
            )
        )

    for executable_name in ("webots.exe", "webots"):
        path_entry = shutil.which(executable_name)
        if path_entry:
            candidates.append(Path(path_entry))

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise WebotsExecutionError(
        "Webots was not found. Set WEBOTS_BIN to its executable or bin directory, "
        "set WEBOTS_HOME to the installation directory, add Webots to PATH, or "
        "pass --webots-bin."
    )


def discover_docker_executable(configured: str | Path | None = None) -> Path:
    """Locate the Docker client used by the headless Xvfb runtime."""
    candidates: list[Path] = []
    configured_value = configured or os.environ.get("DOCKER_BIN")
    if configured_value:
        configured_path = Path(configured_value).expanduser()
        candidates.extend(
            (
                configured_path,
                configured_path / "docker.exe",
                configured_path / "docker",
            )
        )
    for executable_name in ("docker.exe", "docker"):
        path_entry = shutil.which(executable_name)
        if path_entry:
            candidates.append(Path(path_entry))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise WebotsExecutionError(
        "Docker was not found. Install Docker Desktop or Docker Engine, add it "
        "to PATH, or set DOCKER_BIN."
    )


@dataclass(frozen=True, slots=True)
class WebotsExecutionResult:
    run_id: str
    runtime: str
    status: str
    result: dict[str, Any]
    stdout: str
    stderr: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "runtime": self.runtime,
            "status": self.status,
            "result": self.result,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


class WebotsExecutor:
    def __init__(
        self,
        webots_bin: str | Path | None,
        project_root: str | Path,
        timeout_seconds: float = 180.0,
        headless: bool = True,
        runtime: str = "native",
        docker_image: str | None = None,
        docker_bin: str | Path | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.timeout_seconds = timeout_seconds
        self.headless = headless
        self.runtime = runtime
        self.world_path = self.project_root / "webots" / "worlds" / "kitchen_llm.wbt"
        if not self.world_path.is_file():
            raise WebotsExecutionError(f"Webots world was not found: {self.world_path}")
        if runtime not in {"native", "docker"}:
            raise ValueError("Webots runtime must be 'native' or 'docker'.")
        if runtime == "docker":
            if not headless:
                raise WebotsExecutionError(
                    "The Docker Webots runtime is headless; omit --show-webots."
                )
            self.docker_executable = discover_docker_executable(docker_bin)
            self.docker_image = (
                docker_image
                or os.environ.get("WEBOTS_DOCKER_IMAGE")
                or DEFAULT_WEBOTS_DOCKER_IMAGE
            )
            self.webots_executable = None
            self.webots_bin = None
            self.webots_home = None
        else:
            self.webots_executable = discover_webots_executable(webots_bin)
            self.webots_bin = self.webots_executable.parent
            configured_home = os.environ.get("WEBOTS_HOME")
            home_candidates = (
                (Path(configured_home).expanduser(),) if configured_home else ()
            )
            self.webots_home = next(
                (
                    candidate
                    for candidate in (
                        self.webots_bin,
                        *self.webots_bin.parents,
                        *home_candidates,
                    )
                    if (candidate / "projects").is_dir()
                    and (candidate / "resources").is_dir()
                ),
                None,
            )
            if self.webots_home is None:
                raise WebotsExecutionError(
                    f"Could not infer WEBOTS_HOME from: {self.webots_bin}"
                )
            self.docker_executable = None
            self.docker_image = None

    def execute(self, plan: Plan, initial_world: WorldState) -> WebotsExecutionResult:
        validation = PlanValidator().validate(plan, initial_world)
        if not validation.valid:
            raise WebotsExecutionError(
                "Webots execution blocked: " + json.dumps(validation.to_dict())
            )

        unsupported_objects = {
            step.arguments["object"]
            for step in plan.steps
            if step.action in {"find", "pick", "place"}
            and step.arguments["object"]
            not in {"apple", "mug", "orange", "can", "cereal_box"}
        }
        if unsupported_objects:
            raise WebotsExecutionError(
                "No Webots object mapping for: " + ", ".join(sorted(unsupported_objects))
            )

        run_id = str(uuid.uuid4())
        runtime = self.project_root / "webots" / "runtime"
        runtime.mkdir(parents=True, exist_ok=True)
        plan_path = runtime / f"{run_id}.plan.json"
        result_path = runtime / f"{run_id}.result.json"
        log_path = runtime / f"{run_id}.runtime.jsonl"
        screenshot_path = runtime / f"{run_id}.screenshots"
        plan_path.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")

        environment = os.environ.copy()
        container_name = None
        if self.runtime == "docker":
            container_name = f"robot-task-{run_id}"
            container_root = "/project"
            container_runtime = f"{container_root}/webots/runtime"
            mount = f"type=bind,source={self.project_root},target={container_root}"
            command = [
                str(self.docker_executable),
                "run",
                "--rm",
                "--name",
                container_name,
                "--mount",
                mount,
                "--workdir",
                container_root,
                "-e",
                "PYTHONPATH=/project/src",
                "-e",
                "LIBGL_ALWAYS_SOFTWARE=1",
                "-e",
                f"ROBOT_TASK_PLAN_FILE={container_runtime}/{plan_path.name}",
                "-e",
                f"ROBOT_TASK_RESULT_FILE={container_runtime}/{result_path.name}",
                "-e",
                f"ROBOT_TASK_LOG_FILE={container_runtime}/{log_path.name}",
                "-e",
                (
                    "ROBOT_TASK_SCREENSHOT_DIR="
                    f"{container_runtime}/{screenshot_path.name}"
                ),
                str(self.docker_image),
                "/bin/sh",
                "-lc",
                (
                    "set -eu; "
                    "Xvfb :99 -screen 0 1024x768x24 -nolisten tcp "
                    ">/tmp/xvfb.log 2>&1 & "
                    "export DISPLAY=:99; "
                    "attempt=0; "
                    "while [ ! -S /tmp/.X11-unix/X99 ]; do "
                    "attempt=$((attempt + 1)); "
                    "if [ \"$attempt\" -ge 50 ]; then "
                    "cat /tmp/xvfb.log >&2; exit 1; fi; "
                    "sleep 0.1; "
                    "done; "
                    "exec webots --batch --mode=fast --no-rendering --minimize "
                    "--stdout --stderr "
                    f"{container_root}/webots/worlds/{self.world_path.name}"
                ),
            ]
        else:
            environment["WEBOTS_HOME"] = str(self.webots_home)
            environment["ROBOT_TASK_PLAN_FILE"] = str(plan_path)
            environment["ROBOT_TASK_RESULT_FILE"] = str(result_path)
            environment["ROBOT_TASK_LOG_FILE"] = str(log_path)
            environment["ROBOT_TASK_SCREENSHOT_DIR"] = str(screenshot_path)
            environment["ROBOT_TASK_KEEP_OPEN"] = "0" if self.headless else "1"
            command = [
                str(self.webots_executable),
                "--batch",
                f"--mode={'fast' if self.headless else 'realtime'}",
                "--stdout",
                "--stderr",
                str(self.world_path),
            ]
            if self.headless:
                command[3:3] = ["--no-rendering", "--minimize"]
        try:
            completed = subprocess.run(
                command,
                cwd=self.project_root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            if container_name is not None:
                subprocess.run(
                    [
                        str(self.docker_executable),
                        "rm",
                        "--force",
                        container_name,
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
            raise WebotsExecutionError(
                f"Webots did not finish within {self.timeout_seconds:.0f} seconds."
            ) from exc

        if not result_path.exists():
            raise WebotsExecutionError(
                f"Webots {self.runtime} runtime produced no result "
                f"(exit {completed.returncode}). "
                f"stderr: {completed.stderr[-2000:]}"
            )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "completed":
            raise WebotsExecutionError(
                "Webots controller failed"
                f" [{result.get('code', 'UNKNOWN')}]: "
                + str(result.get("error", result))
            )
        unverified_steps = [
            step.get("index")
            for step in result.get("steps", [])
            if step.get("outcome_verified") is not True
        ]
        if unverified_steps:
            raise WebotsExecutionError(
                "Webots returned unverified step outcomes: "
                + ", ".join(str(index) for index in unverified_steps)
            )
        if result.get("final_state") != validation.predicted_final_state:
            raise WebotsExecutionError(
                "Webots observed final state did not match the validator prediction."
            )
        return WebotsExecutionResult(
            run_id=run_id,
            runtime=self.runtime,
            status="completed",
            result=result,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
