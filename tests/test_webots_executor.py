from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from pathlib import PurePosixPath
from unittest.mock import patch

from robot_planner.models import Plan
from robot_planner.validator import PlanValidator
from robot_planner.webots_executor import (
    WebotsExecutionError,
    WebotsExecutor,
    discover_webots_executable,
)
from robot_planner.world import world_from_mapping


def world():
    return world_from_mapping(
        {
            "locations": [
                {"id": "table", "kind": "surface", "openable": False, "is_open": True},
                {"id": "basket", "kind": "container", "openable": False, "is_open": True},
            ],
            "objects": [{"id": "apple", "location": "table", "portable": True}],
            "robot": {"location": "table", "held_object": None},
        }
    )


def plan():
    return Plan.from_mapping(
        {
            "plan_id": "webots-test",
            "goal": "Put apple in basket",
            "steps": [
                {"action": "find", "arguments": {"object": "apple"}},
                {"action": "pick", "arguments": {"object": "apple"}},
                {"action": "navigate", "arguments": {"location": "basket"}},
                {
                    "action": "place",
                    "arguments": {"object": "apple", "location": "basket"},
                },
            ],
        }
    )


class WebotsExecutorTests(unittest.TestCase):
    def test_discovers_webots_from_webots_home(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory) / "Webots"
            executable = home / "msys64" / "mingw64" / "bin" / "webots.exe"
            executable.parent.mkdir(parents=True)
            executable.write_bytes(b"test")
            with patch.dict(
                "os.environ",
                {"WEBOTS_HOME": str(home)},
                clear=True,
            ), patch("robot_planner.webots_executor.shutil.which", return_value=None):
                self.assertEqual(
                    executable.resolve(), discover_webots_executable()
                )

    def test_missing_webots_has_portable_configuration_message(self) -> None:
        with patch.dict("os.environ", {}, clear=True), patch(
            "robot_planner.webots_executor.shutil.which", return_value=None
        ):
            with self.assertRaisesRegex(
                WebotsExecutionError, "WEBOTS_BIN.*WEBOTS_HOME.*PATH"
            ):
                discover_webots_executable()

    def test_launcher_passes_validated_plan_and_required_flags(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            webots_home = root / "Webots"
            webots_bin = webots_home / "msys64" / "mingw64" / "bin"
            webots_bin.mkdir(parents=True)
            (webots_home / "projects").mkdir()
            (webots_home / "resources").mkdir()
            (webots_bin / "webots.exe").write_bytes(b"test")
            world_path = root / "project" / "webots" / "worlds" / "kitchen_llm.wbt"
            world_path.parent.mkdir(parents=True)
            world_path.write_text("#VRML_SIM R2025a utf8\n", encoding="utf-8")

            captured = {}
            expected_final = PlanValidator().validate(
                plan(), world()
            ).predicted_final_state

            def fake_run(command, cwd, env, capture_output, text, timeout, check):
                captured["command"] = command
                captured["env"] = env
                result_path = Path(env["ROBOT_TASK_RESULT_FILE"])
                result_path.write_text(
                    json.dumps(
                        {
                            "status": "completed",
                            "plan_id": "webots-test",
                            "steps": [
                                {"index": index, "outcome_verified": True}
                                for index in range(4)
                            ],
                            "final_state": expected_final,
                        }
                    ),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(command, 0, "controller output", "")

            executor = WebotsExecutor(webots_bin, root / "project")
            with patch(
                "robot_planner.webots_executor.subprocess.run", side_effect=fake_run
            ):
                result = executor.execute(plan(), world())

            self.assertEqual("completed", result.status)
            self.assertEqual("native", result.runtime)
            self.assertIn("--no-rendering", captured["command"])
            self.assertIn("--batch", captured["command"])
            self.assertEqual("0", captured["env"]["ROBOT_TASK_KEEP_OPEN"])
            self.assertEqual(str(webots_home), captured["env"]["WEBOTS_HOME"])
            written_plan = json.loads(
                Path(captured["env"]["ROBOT_TASK_PLAN_FILE"]).read_text(encoding="utf-8")
            )
            self.assertEqual("webots-test", written_plan["plan_id"])

    def test_docker_runtime_uses_xvfb_and_project_mount(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "project"
            world_path = project / "webots" / "worlds" / "kitchen_llm.wbt"
            world_path.parent.mkdir(parents=True)
            world_path.write_text("#VRML_SIM R2025a utf8\n", encoding="utf-8")
            docker = Path(directory) / "docker.exe"
            docker.write_bytes(b"test")
            captured = {}
            expected_final = PlanValidator().validate(
                plan(), world()
            ).predicted_final_state

            def fake_run(command, cwd, env, capture_output, text, timeout, check):
                captured["command"] = command
                result_variable = next(
                    value
                    for value in command
                    if value.startswith("ROBOT_TASK_RESULT_FILE=")
                )
                container_result = result_variable.split("=", 1)[1]
                result_path = (
                    project
                    / "webots"
                    / "runtime"
                    / PurePosixPath(container_result).name
                )
                result_path.write_text(
                    json.dumps(
                        {
                            "status": "completed",
                            "plan_id": "webots-test",
                            "steps": [
                                {"index": index, "outcome_verified": True}
                                for index in range(4)
                            ],
                            "final_state": expected_final,
                        }
                    ),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(command, 0, "controller output", "")

            executor = WebotsExecutor(
                None,
                project,
                runtime="docker",
                docker_bin=docker,
            )
            with patch(
                "robot_planner.webots_executor.subprocess.run", side_effect=fake_run
            ):
                result = executor.execute(plan(), world())

            command = captured["command"]
            self.assertEqual("completed", result.status)
            self.assertEqual("docker", result.runtime)
            self.assertIn("robot-task-webots-cnn:R2025a", command)
            launch_script = command[-1]
            self.assertIn("Xvfb :99", launch_script)
            self.assertIn("exec webots --batch", launch_script)
            self.assertIn("LIBGL_ALWAYS_SOFTWARE=1", command)
            self.assertIn("--mount", command)
            self.assertIn(
                "/project/webots/worlds/kitchen_llm.wbt", launch_script
            )

    def test_project_world_contains_required_task_nodes(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        world_text = (
            project_root / "webots" / "worlds" / "kitchen_llm.wbt"
        ).read_text(encoding="utf-8")
        for marker in (
            "DEF TASK_ROBOT Tiago",
            'controller "robot_task_controller"',
            "DEF TASK_APPLE Apple",
            "DEF TASK_MUG Solid",
            "DEF TASK_ORANGE Solid",
            "DEF TASK_CAN Solid",
            "DEF TASK_CEREAL_BOX Solid",
            "DEF TASK_BASKET FruitBowl",
            "DEF TASK_CUPBOARD_DOOR Pose",
            "DEF TASK_DATASET_CAMERA Camera",
            "DEF TASK_CEILING_LIGHT CeilingLight",
            "recognition Recognition",
        ):
            self.assertIn(marker, world_text)


if __name__ == "__main__":
    unittest.main()
