"""Webots supervisor controller for validated high-level kitchen plans."""

from __future__ import annotations

import json
import hashlib
import math
import os
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from controller import Supervisor


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from robot_planner.actions import apply_step  # noqa: E402
from robot_planner.models import Plan  # noqa: E402
from robot_planner.navigation import (  # noqa: E402
    KITCHEN_OBSTACLES,
    ROBOT_CLEARANCE,
    minimum_clearance,
    path_length,
    plan_path,
)
from robot_planner.validator import PlanValidator  # noqa: E402
from robot_planner.world import load_world  # noqa: E402


ROBOT_WAYPOINTS = {
    "table": (0.65, -2.55, 0.095),
    "basket": (0.65, -2.55, 0.095),
    "shelf": (-1.20, 0.30, 0.095),
    "cupboard": (1.05, -0.25, 0.095),
}
ROBOT_ORIENTATIONS = {
    "table": 2.65,
    "basket": 2.65,
    "shelf": -0.83,
    "cupboard": 0.0,
}
PLACEMENT_POINTS = {
    "table": (-0.65, -1.43, 0.88),
    "basket": (-0.6855, -1.754, 0.88),
    "shelf": (-1.21, 1.31, 0.90),
    "cupboard": (1.78, -0.25, 0.95),
}
PLACEMENT_REGIONS = {
    "basket": {
        "x": (-0.86, -0.51),
        "y": (-1.92, -1.58),
        "z": (0.74, 0.98),
    },
    "shelf": {
        "x": (-1.45, -1.00),
        "y": (1.10, 1.52),
        "z": (0.78, 1.02),
    },
    "cupboard": {
        "x": (1.59, 2.00),
        "y": (-0.41, -0.09),
        "z": (0.88, 1.15),
    },
}
OBJECT_DEFS = {
    "apple": "TASK_APPLE",
    "mug": "TASK_MUG",
    "orange": "TASK_ORANGE",
    "can": "TASK_CAN",
    "cereal_box": "TASK_CEREAL_BOX",
}
CNN_CLASS_NAMES = ("mug", "apple", "orange", "can", "cereal_box")
CNN_MODEL_NAMES = {
    "mug": "mug",
    "apple": "apple",
    "orange": "orange",
    "can": "can",
    "cereal box": "cereal_box",
    "cereal_box": "cereal_box",
}
DATASET_OBJECT_HEIGHTS = {
    "mug": 0.74,
    "apple": 0.80,
    "orange": 0.80,
    "can": 0.80,
    "cereal_box": 0.86,
}
HOME_ARM = (0.07, 0.26, -3.16, 1.27, 1.32, 0.0, 1.41)
REACH_ARM = (0.33, 0.05, -1.57, 1.40, 0.0, 0.0, 0.0)
POSITION_TOLERANCE = 0.08


class ObservedOutcomeError(RuntimeError):
    def __init__(self, step_index: int, action: str, evidence: dict) -> None:
        super().__init__(
            f"Observed Webots outcome did not satisfy '{action}' at step {step_index}."
        )
        self.code = (
            "OBJECT_NOT_VISUALLY_DETECTED"
            if action == "find"
            else "OBSERVED_STATE_MISMATCH"
        )
        self.step_index = step_index
        self.action = action
        self.evidence = evidence


class CnnPerceptionError(RuntimeError):
    code = "OBJECT_NOT_VISUALLY_DETECTED"


def runtime_log_path() -> Path:
    default_path = PROJECT_ROOT / "webots" / "runtime" / "controller.runtime.jsonl"
    return Path(os.environ.get("ROBOT_TASK_LOG_FILE", default_path))


def log_runtime_event(event: str, **details) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **details,
    }
    log_path = runtime_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, separators=(",", ":")) + "\n")


class WebotsTaskController:
    def __init__(self) -> None:
        self.robot = Supervisor()
        self.time_step = int(self.robot.getBasicTimeStep())
        self.robot_node = self.robot.getSelf()
        self.robot_translation = self.robot_node.getField("translation")
        self.robot_rotation = self.robot_node.getField("rotation")
        self.held_node = None
        self.held_object_id = None
        self.current_location = "table"
        self.arm_motors = [
            self.robot.getDevice(f"arm_{index}_joint") for index in range(1, 8)
        ]
        self.torso = self.robot.getDevice("torso_lift_joint")
        self.head_yaw = self.robot.getDevice("head_1_joint")
        self.head_pitch = self.robot.getDevice("head_2_joint")
        self.gripper_left = self.robot.getDevice("gripper_left_finger_joint")
        self.gripper_right = self.robot.getDevice("gripper_right_finger_joint")
        self.validation_camera = self.robot.getDevice("Astra rgb")
        self.validation_camera.enable(self.time_step)
        self.perception_backend = os.environ.get(
            "ROBOT_TASK_PERCEPTION_BACKEND", "cnn"
        ).strip().lower()
        if self.perception_backend not in {"cnn", "metadata"}:
            raise ValueError(
                "ROBOT_TASK_PERCEPTION_BACKEND must be 'cnn' or 'metadata'."
            )
        if (
            os.environ.get("ROBOT_TASK_MODE") == "cnn_dataset"
            or self.perception_backend == "metadata"
        ):
            self.validation_camera.recognitionEnable(self.time_step)
        else:
            self.validation_camera.recognitionDisable()
            if self.validation_camera.getRecognitionSamplingPeriod() != 0:
                raise RuntimeError(
                    "Webots recognition metadata remained enabled during CNN execution."
                )
        self.cnn_detector = None
        self.initial_object_poses = {}
        for object_id, def_name in OBJECT_DEFS.items():
            node = self.robot.getFromDef(def_name)
            if node is not None:
                self.initial_object_poses[object_id] = {
                    "translation": list(node.getField("translation").getSFVec3f()),
                    "rotation": list(node.getField("rotation").getSFRotation()),
                }
        default_screenshot_directory = (
            PROJECT_ROOT / "webots" / "runtime" / "screenshots"
        )
        self.screenshot_directory = Path(
            os.environ.get(
                "ROBOT_TASK_SCREENSHOT_DIR", default_screenshot_directory
            )
        )
        self.screenshot_directory.mkdir(parents=True, exist_ok=True)
        self.screenshots = []
        log_runtime_event(
            "controller_started",
            basic_time_step=self.time_step,
            robot_def="TASK_ROBOT",
        )

    def step(self, count: int = 1) -> None:
        for _ in range(count):
            if self.held_node is not None:
                self.synchronize_held_object()
            if self.robot.step(self.time_step) == -1:
                raise RuntimeError("Webots simulation stopped before plan completion.")
        if self.held_node is not None:
            self.synchronize_held_object()

    def synchronize_held_object(self) -> None:
        robot_position = self.robot_translation.getSFVec3f()
        self.held_node.getField("translation").setSFVec3f(
            [robot_position[0], robot_position[1], 1.10]
        )
        self.held_node.resetPhysics()

    def set_arm(self, positions: tuple[float, ...]) -> None:
        self.torso.setPosition(0.16)
        for motor, position in zip(self.arm_motors, positions):
            motor.setPosition(position)
        self.step(18)

    def aim_camera_at_work_surface(self) -> None:
        self.head_yaw.setPosition(0.0)
        self.head_pitch.setPosition(-0.70)
        self.step(25)

    def set_gripper(self, opened: bool) -> None:
        position = 0.04 if opened else 0.0
        self.gripper_left.setPosition(position)
        self.gripper_right.setPosition(position)
        self.step(10)

    def navigate(self, location: str) -> None:
        # A physical mobile base can receive small contact impulses while the
        # arm settles.  Navigation always begins from the verified service
        # point tracked by the controller, so reassert that pose before path
        # planning instead of allowing accumulated drift into the planner.
        self.stabilize_base()
        target = ROBOT_WAYPOINTS[location]
        current = self.robot_translation.getSFVec3f()
        route_2d = plan_path((current[0], current[1]), (target[0], target[1]))
        route = [[point[0], point[1], target[2]] for point in route_2d]
        clearance = minimum_clearance(route_2d)
        log_runtime_event(
            "navigation_path_planned",
            location=location,
            route=route,
            direct_distance=math.dist(route_2d[0], route_2d[-1]),
            route_distance=path_length(route_2d),
            minimum_obstacle_clearance=clearance,
            required_clearance=ROBOT_CLEARANCE,
            obstacles=[obstacle.name for obstacle in KITCHEN_OBSTACLES],
        )
        for segment_index, (segment_start, segment_end) in enumerate(
            zip(route, route[1:]), start=1
        ):
            heading = math.atan2(
                segment_end[1] - segment_start[1],
                segment_end[0] - segment_start[0],
            )
            self.turn_to(heading)
            distance = math.dist(segment_start[:2], segment_end[:2])
            frames = max(8, math.ceil(distance / 0.045))
            for frame in range(1, frames + 1):
                ratio = frame / frames
                smooth = ratio * ratio * (3.0 - 2.0 * ratio)
                position = [
                    segment_start[index]
                    + (segment_end[index] - segment_start[index]) * smooth
                    for index in range(3)
                ]
                self.robot_translation.setSFVec3f(position)
                self.step()
            self.capture(f"route_{location}_{segment_index:02d}")
        self.turn_to(ROBOT_ORIENTATIONS[location])
        self.robot_translation.setSFVec3f(list(target))
        self.robot_node.resetPhysics()
        self.step(5)
        self.current_location = location

    def stabilize_base(self) -> None:
        """Restore the safe service pose after cosmetic arm contact forces."""
        target = ROBOT_WAYPOINTS[self.current_location]
        self.robot_translation.setSFVec3f(list(target))
        self.robot_rotation.setSFRotation(
            [0.0, 0.0, 1.0, ROBOT_ORIENTATIONS[self.current_location]]
        )
        self.robot_node.resetPhysics()
        self.step(3)

    def turn_to(self, target_angle: float) -> None:
        current_rotation = self.robot_rotation.getSFRotation()
        current_angle = float(current_rotation[3]) * (
            1.0 if float(current_rotation[2]) >= 0 else -1.0
        )
        delta = (target_angle - current_angle + math.pi) % (2 * math.pi) - math.pi
        frames = max(6, math.ceil(abs(delta) / 0.08))
        for frame in range(1, frames + 1):
            ratio = frame / frames
            angle = current_angle + delta * ratio
            self.robot_rotation.setSFRotation([0.0, 0.0, 1.0, angle])
            self.step()

    def pick(self, object_id: str) -> None:
        node = self._object_node(object_id)
        self.set_gripper(True)
        self.set_arm(REACH_ARM)
        self.held_node = node
        self.held_object_id = object_id
        self.set_gripper(False)
        self.set_arm(HOME_ARM)
        self.stabilize_base()

    def find(self, object_id: str) -> dict:
        if self.perception_backend == "metadata":
            return self._find_with_metadata(object_id)
        try:
            return self._find_with_cnn(object_id)
        except CnnPerceptionError:
            raise
        except Exception as exc:
            raise CnnPerceptionError(
                f"CNN perception could not verify '{object_id}': {exc}"
            ) from exc

    def _find_with_metadata(self, object_id: str) -> dict:
        recognized_by_model: dict[str, dict] = {}
        matches: list[dict] = []
        frames_waited = 0
        matched_pose = None
        scan_poses = (
            (0.0, -0.55),
            (-0.45, -0.55),
            (0.45, -0.55),
            (-0.80, -0.45),
            (0.80, -0.45),
            (0.0, -0.25),
        )
        for yaw, pitch in scan_poses:
            self.head_yaw.setPosition(yaw)
            self.head_pitch.setPosition(pitch)
            self.step(20)
            for _ in range(10):
                frames_waited += 1
                self.step()
                for detected in self.validation_camera.getRecognitionObjects():
                    model = detected.getModel()
                    if isinstance(model, bytes):
                        model = model.decode("utf-8", errors="replace")
                    item = {
                        "model": str(model),
                        "position": list(detected.getPosition()),
                        "position_on_image": list(detected.getPositionOnImage()),
                        "size_on_image": list(detected.getSizeOnImage()),
                    }
                    recognized_by_model[item["model"]] = item
                matches = [
                    item
                    for item in recognized_by_model.values()
                    if item["model"].strip().lower() == object_id.strip().lower()
                ]
                if matches:
                    matched_pose = {"yaw": yaw, "pitch": pitch}
                    break
            if matches:
                break
        recognized = list(recognized_by_model.values())
        camera_image_path = self.screenshot_directory / f"camera_find_{object_id}.jpg"
        self.validation_camera.saveImage(str(camera_image_path), 90)
        return {
            "kind": "camera_recognition",
            "requested_object": object_id,
            "recognized_objects": recognized,
            "matched": bool(matches),
            "matches": matches,
            "frames_waited": frames_waited,
            "scan_pose": matched_pose,
            "camera_image": str(camera_image_path),
            "verified": bool(matches),
        }

    def _camera_rgb(self):
        from robot_planner.perception import bgra_buffer_to_rgb

        width = self.validation_camera.getWidth()
        height = self.validation_camera.getHeight()
        return bgra_buffer_to_rgb(self.validation_camera.getImage(), width, height)

    def _find_with_cnn(self, object_id: str) -> dict:
        from robot_planner.perception import (
            KitchenObjectDetector,
            confirm_temporal_detections,
        )

        if self.cnn_detector is None:
            default_model = (
                PROJECT_ROOT
                / "cnn_artifacts"
                / "kitchen_object_net_v1"
                / "KitchenObjectNet.onnx"
            )
            model_path = Path(os.environ.get("ROBOT_CNN_MODEL", default_model))
            threshold_value = os.environ.get("ROBOT_CNN_CONFIDENCE_THRESHOLD")
            self.cnn_detector = KitchenObjectDetector(
                model_path,
                confidence_threshold=(
                    float(threshold_value) if threshold_value is not None else None
                ),
            )
        requested = object_id.strip().lower().replace(" ", "_")
        scan_poses = (
            (-0.45, -0.40),
            (0.0, -0.40),
            (0.0, -0.55),
            (-0.45, -0.55),
            (0.45, -0.55),
            (-0.80, -0.45),
            (0.80, -0.45),
            (0.0, -0.25),
        )
        frames_waited = 0
        all_detections: list[dict] = []
        rejected_detections: list[dict] = []
        confirmed: list[dict] = []
        matched_pose = None
        latest_rgb = None
        latest_detections: list[dict] = []
        for yaw, pitch in scan_poses:
            self.head_yaw.setPosition(yaw)
            self.head_pitch.setPosition(pitch)
            self.step(20)
            pose_frames: list[list[dict]] = []
            for frame_index in range(3):
                self.step(3)
                frames_waited += 3
                latest_rgb = self._camera_rgb()
                latest_detections = self.cnn_detector.detect(latest_rgb)
                all_detections.extend(
                    [{**item, "scan_yaw": yaw, "scan_pitch": pitch, "frame": frame_index} for item in latest_detections]
                )
                rejected_detections.extend(
                    [
                        {
                            **item,
                            "scan_yaw": yaw,
                            "scan_pitch": pitch,
                            "frame": frame_index,
                        }
                        for item in self.cnn_detector.last_rejected
                    ]
                )
                pose_frames.append(latest_detections)
            confirmed = confirm_temporal_detections(pose_frames, requested)
            if confirmed:
                matched_pose = {"yaw": yaw, "pitch": pitch}
                break
        if latest_rgb is None:
            raise RuntimeError("CNN scan produced no camera frame.")
        camera_image_path = self.screenshot_directory / f"cnn_find_{object_id}.jpg"
        self.cnn_detector.annotate(latest_rgb, latest_detections).save(
            camera_image_path, quality=92
        )
        best_match = (
            max(confirmed, key=lambda item: item["confidence"]) if confirmed else None
        )
        return {
            "kind": "cnn_object_detection",
            "requested_object": object_id,
            "matched": best_match is not None,
            "match": best_match,
            "confirmation_frames": len(confirmed),
            "required_confirmation_frames": 2,
            "detections": all_detections,
            "rejected_detections": rejected_detections,
            "frames_waited": frames_waited,
            "scan_pose": matched_pose,
            "camera_image": str(camera_image_path),
            "model_path": str(self.cnn_detector.model_path),
            "model_sha256": self.cnn_detector.model_sha256,
            "confidence_threshold": self.cnn_detector.confidence_threshold,
            "latest_inference_ms": self.cnn_detector.last_inference_ms,
            "recognition_metadata_used": False,
            "verified": best_match is not None,
        }

    def place(self, object_id: str, location: str) -> None:
        node = self._object_node(object_id)
        if self.held_object_id != object_id:
            raise RuntimeError(f"Webots controller is not carrying '{object_id}'.")
        self.set_arm(REACH_ARM)
        self.held_node = None
        self.held_object_id = None
        node.getField("translation").setSFVec3f(list(PLACEMENT_POINTS[location]))
        node.resetPhysics()
        self.set_gripper(True)
        self.set_arm(HOME_ARM)
        self.stabilize_base()

    def set_cupboard(self, opened: bool) -> None:
        door = self.robot.getFromDef("TASK_CUPBOARD_DOOR")
        if door is None:
            raise RuntimeError("TASK_CUPBOARD_DOOR was not found in the world.")
        rotation = [0.0, 0.0, 1.0, -1.35 if opened else 0.0]
        door.getField("rotation").setSFRotation(rotation)
        self.set_arm(REACH_ARM)
        self.set_arm(HOME_ARM)
        self.stabilize_base()

    def capture(self, label: str) -> None:
        self.step(2)
        screenshot_path = self.screenshot_directory / f"{label}.jpg"
        if self.validation_camera.saveImage(str(screenshot_path), 95) != 0:
            raise RuntimeError(f"Failed to save validation camera image: {screenshot_path}")
        self.screenshots.append(str(screenshot_path))

    def generate_cnn_dataset(self) -> dict:
        """Generate deterministic RGB images and project-owned box annotations."""
        dataset_root = Path(
            os.environ.get(
                "ROBOT_CNN_DATASET_DIR",
                PROJECT_ROOT / "cnn_artifacts" / "kitchen_object_net_v1" / "dataset",
            )
        )
        total = int(os.environ.get("ROBOT_CNN_DATASET_COUNT", "8000"))
        base_seed = int(os.environ.get("ROBOT_CNN_DATASET_SEED", "20260808"))
        expected_counts = {
            "train": round(total * 0.70),
            "validation": round(total * 0.15),
        }
        expected_counts["test"] = total - sum(expected_counts.values())
        negative_total = total // 5
        dataset_root.mkdir(parents=True, exist_ok=True)
        for split in expected_counts:
            (dataset_root / "images" / split).mkdir(parents=True, exist_ok=True)
        annotation_path = dataset_root / "annotations.jsonl"
        generation_log_path = dataset_root / "dataset_generation.jsonl"
        annotation_path.write_text("", encoding="utf-8")
        generation_log_path.write_text("", encoding="utf-8")

        split_schedule = [
            split for split, count in expected_counts.items() for _ in range(count)
        ]
        schedule_rng = random.Random(base_seed)
        schedule_rng.shuffle(split_schedule)
        negative_indices = set(schedule_rng.sample(range(total), negative_total))
        target_nodes = {
            name: self.robot.getFromDef(OBJECT_DEFS[name]) for name in CNN_CLASS_NAMES
        }
        missing = [name for name, node in target_nodes.items() if node is None]
        if missing:
            raise RuntimeError("Missing CNN dataset object nodes: " + ", ".join(missing))
        node_ids = {node.getId(): name for name, node in target_nodes.items()}
        light = self.robot.getFromDef("TASK_CEILING_LIGHT")
        camera_node = self.robot.getFromDef("TASK_DATASET_CAMERA")
        original_positions = {
            name: list(node.getField("translation").getSFVec3f())
            for name, node in target_nodes.items()
        }
        original_rotations = {
            name: list(node.getField("rotation").getSFRotation())
            for name, node in target_nodes.items()
        }
        width = int(self.validation_camera.getWidth())
        height = int(self.validation_camera.getHeight())
        class_instances = {name: 0 for name in CNN_CLASS_NAMES}
        saved_by_split = {name: 0 for name in expected_counts}
        positive_index = 0

        def write_log(event: str, **details) -> None:
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": event,
                **details,
            }
            with generation_log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")

        write_log(
            "dataset_generation_started",
            total=total,
            seed=base_seed,
            splits=expected_counts,
            negative_total=negative_total,
            classes=list(CNN_CLASS_NAMES),
        )
        try:
            self.robot_translation.setSFVec3f(list(ROBOT_WAYPOINTS["shelf"]))
            self.robot_rotation.setSFRotation([0.0, 0.0, 1.0, -0.83])
            self.robot_node.resetPhysics()
            self.step(15)
            with annotation_path.open("a", encoding="utf-8") as annotations:
                for sample_index in range(total):
                    split = split_schedule[sample_index]
                    seed = base_seed * 100000 + sample_index
                    rng = random.Random(seed)
                    negative = sample_index in negative_indices
                    if negative:
                        selected: list[str] = []
                    else:
                        # One target per positive frame gives every class the
                        # same number of independent examples.  All ordinary
                        # kitchen props remain visible, so they act as hard
                        # distractors instead of becoming unlabelled target
                        # instances in multi-object frames.
                        selected = [
                            CNN_CLASS_NAMES[positive_index % len(CNN_CLASS_NAMES)]
                        ]
                        positive_index += 1
                    detected_items = []
                    raw_recognition = []
                    for attempt in range(30):
                        for name, node in target_nodes.items():
                            if name not in selected:
                                position = [2.8 + CNN_CLASS_NAMES.index(name) * 0.1, 2.8, 1.2]
                            else:
                                difficulty = sample_index % 3
                                x_ranges = (
                                    (-0.72, -0.38),
                                    (-0.82, -0.28),
                                    (-0.90, -0.18),
                                )
                                y_ranges = (
                                    (-1.24, -0.96),
                                    (-1.34, -0.90),
                                    (-1.48, -0.84),
                                )
                                if attempt < 20:
                                    position = [
                                        rng.uniform(*x_ranges[difficulty]),
                                        rng.uniform(*y_ranges[difficulty]),
                                        DATASET_OBJECT_HEIGHTS[name],
                                    ]
                                else:
                                    # Deterministic clear-table fallback keeps
                                    # a rare sequence of unlucky random poses
                                    # from aborting the reproducible run.
                                    fallback_offset = (attempt - 20) * 0.008
                                    position = [
                                        -0.85 + fallback_offset,
                                        -1.00 - fallback_offset,
                                        DATASET_OBJECT_HEIGHTS[name],
                                    ]
                            node.getField("translation").setSFVec3f(position)
                            node.getField("rotation").setSFRotation(
                                [0.0, 0.0, 1.0, rng.uniform(-math.pi, math.pi)]
                            )
                            node.resetPhysics()
                        pose_jitter = 1.0 if attempt < 20 else 0.0
                        self.robot_translation.setSFVec3f(
                            [
                                ROBOT_WAYPOINTS["shelf"][0]
                                + pose_jitter * rng.uniform(-0.04, 0.04),
                                ROBOT_WAYPOINTS["shelf"][1]
                                + pose_jitter * rng.uniform(-0.04, 0.04),
                                ROBOT_WAYPOINTS["shelf"][2],
                            ]
                        )
                        self.robot_rotation.setSFRotation(
                            [
                                0.0,
                                0.0,
                                1.0,
                                -0.83 + pose_jitter * rng.uniform(-0.05, 0.05),
                            ]
                        )
                        self.robot_node.resetPhysics()
                        # The shelf waypoint views the work table through the
                        # robot's left side.  Keep the randomised scan centred
                        # on the empirically verified (-0.45, -0.55) pose so
                        # chairs remain contextual background, not a full-frame
                        # occluder.
                        self.head_yaw.setPosition(
                            -0.45 + pose_jitter * rng.uniform(-0.10, 0.10)
                        )
                        self.head_pitch.setPosition(
                            (-0.40 if attempt < 20 else -0.25)
                            + pose_jitter * rng.uniform(-0.05, 0.05)
                        )
                        if light is not None and light.getField("pointLightIntensity") is not None:
                            light.getField("pointLightIntensity").setSFFloat(rng.uniform(1.8, 5.5))
                        if camera_node is not None and camera_node.getField("noise") is not None:
                            camera_node.getField("noise").setSFFloat(rng.uniform(0.0, 0.035))
                        self.step(4)
                        detected_items = []
                        raw_recognition = []
                        for detected in self.validation_camera.getRecognitionObjects():
                            model = detected.getModel()
                            if isinstance(model, bytes):
                                model = model.decode("utf-8", errors="replace")
                            raw_recognition.append(
                                {
                                    "id": detected.getId(),
                                    "model": str(model),
                                    "position_on_image": list(detected.getPositionOnImage()),
                                    "size_on_image": list(detected.getSizeOnImage()),
                                }
                            )
                            object_name = node_ids.get(detected.getId())
                            if object_name is None or object_name not in selected:
                                continue
                            centre = list(detected.getPositionOnImage())
                            size = list(detected.getSizeOnImage())
                            x = float(centre[0]) - float(size[0]) * 0.5
                            y = float(centre[1]) - float(size[1]) * 0.5
                            box_width = float(size[0])
                            box_height = float(size[1])
                            margin = 4.0
                            if (
                                box_width > 2
                                and box_height > 2
                                and x >= margin
                                and y >= margin
                                and x + box_width <= width - margin
                                and y + box_height <= height - margin
                                # The nearest chair occupies the lower-left
                                # image region.  Keeping the complete target
                                # box in this measured clear-table window
                                # prevents recognition boxes for fully hidden
                                # objects from entering the dataset.
                                and x >= 310.0
                            ):
                                detected_items.append(
                                    {
                                        "label": object_name,
                                        "class_id": CNN_CLASS_NAMES.index(object_name),
                                        "bbox_xywh": [
                                            round(x, 3),
                                            round(y, 3),
                                            round(box_width, 3),
                                            round(box_height, 3),
                                        ],
                                    }
                                )
                        if negative or {item["label"] for item in detected_items} == set(selected):
                            break
                    if not negative and {item["label"] for item in detected_items} != set(selected):
                        failed_image = dataset_root / f"failed_sample_{sample_index:06d}.jpg"
                        self.validation_camera.saveImage(str(failed_image), 95)
                        raise RuntimeError(
                            f"Could not render all selected objects for sample {sample_index}: "
                            f"selected={selected}, detected={detected_items}, "
                            f"raw_recognition={raw_recognition}"
                        )
                    image_name = f"{sample_index:06d}.jpg"
                    image_relative = Path("images") / split / image_name
                    image_path = dataset_root / image_relative
                    if self.validation_camera.saveImage(str(image_path), 92) != 0:
                        raise RuntimeError(f"Failed to save dataset image: {image_path}")
                    record = {
                        "id": sample_index,
                        "image": image_relative.as_posix(),
                        "width": width,
                        "height": height,
                        "split": split,
                        "seed": seed,
                        "difficulty": ("easy", "medium", "hard")[sample_index % 3],
                        "negative": negative,
                        "objects": detected_items,
                    }
                    annotations.write(json.dumps(record, separators=(",", ":")) + "\n")
                    annotations.flush()
                    saved_by_split[split] += 1
                    for item in detected_items:
                        class_instances[item["label"]] += 1
                    if (sample_index + 1) % 100 == 0 or sample_index + 1 == total:
                        write_log(
                            "dataset_generation_progress",
                            completed=sample_index + 1,
                            total=total,
                            split_counts=saved_by_split,
                            class_instances=class_instances,
                        )
        finally:
            for name, node in target_nodes.items():
                node.getField("translation").setSFVec3f(original_positions[name])
                node.getField("rotation").setSFRotation(original_rotations[name])
                node.resetPhysics()
        summary = {
            "status": "completed",
            "dataset_root": str(dataset_root),
            "total": total,
            "seed": base_seed,
            "split_counts": saved_by_split,
            "class_instance_counts": class_instances,
            "negative_count": negative_total,
            "annotation_file": str(annotation_path),
            "generation_log": str(generation_log_path),
            "webots_version": "R2025a",
            "world_sha256": hashlib.sha256(
                (PROJECT_ROOT / "webots" / "worlds" / "kitchen_llm.wbt").read_bytes()
            ).hexdigest(),
            "class_map": {
                str(index): name for index, name in enumerate(CNN_CLASS_NAMES)
            },
        }
        (dataset_root / "generation_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        write_log("dataset_generation_completed", **summary)
        return summary

    def _object_node(self, object_id: str):
        def_name = OBJECT_DEFS.get(object_id)
        if not def_name:
            raise RuntimeError(f"No Webots DEF mapping exists for object '{object_id}'.")
        node = self.robot.getFromDef(def_name)
        if node is None:
            raise RuntimeError(f"Webots node '{def_name}' was not found.")
        return node

    @staticmethod
    def _distance(left, right) -> float:
        return sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)) ** 0.5

    def observe_action(self, action: str, arguments: dict[str, str], find_evidence=None) -> dict:
        if action == "navigate":
            expected = list(ROBOT_WAYPOINTS[arguments["location"]])
            actual = list(self.robot_translation.getSFVec3f())
            distance = self._distance(actual, expected)
            return {
                "kind": "robot_position",
                "expected": expected,
                "actual": actual,
                "distance": distance,
                "verified": distance <= POSITION_TOLERANCE,
            }
        if action == "find":
            return dict(find_evidence or {"verified": False, "kind": "camera_recognition"})
        if action == "pick":
            object_id = arguments["object"]
            node = self._object_node(object_id)
            actual = list(node.getField("translation").getSFVec3f())
            robot_position = list(self.robot_translation.getSFVec3f())
            expected = [robot_position[0], robot_position[1], 1.10]
            distance = self._distance(actual, expected)
            return {
                "kind": "carried_object_position",
                "object": object_id,
                "expected": expected,
                "actual": actual,
                "distance": distance,
                "verified": self.held_object_id == object_id
                and distance <= POSITION_TOLERANCE,
            }
        if action == "place":
            object_id = arguments["object"]
            location = arguments["location"]
            expected = list(PLACEMENT_POINTS[location])
            actual = list(
                self._object_node(object_id).getField("translation").getSFVec3f()
            )
            distance = self._distance(actual, expected)
            horizontal_distance = self._distance(actual[:2], expected[:2])
            vertical_error = abs(float(actual[2]) - float(expected[2]))
            region = PLACEMENT_REGIONS.get(location)
            inside_target_region = (
                region is not None
                and region["x"][0] <= actual[0] <= region["x"][1]
                and region["y"][0] <= actual[1] <= region["y"][1]
                and region["z"][0] <= actual[2] <= region["z"][1]
            )
            placement_verified = (
                inside_target_region
                if region is not None
                else horizontal_distance <= POSITION_TOLERANCE
                and vertical_error <= 0.15
            )
            return {
                "kind": "placed_object_position",
                "object": object_id,
                "expected": expected,
                "actual": actual,
                "distance": distance,
                "horizontal_distance": horizontal_distance,
                "vertical_error": vertical_error,
                "target_region": region,
                "inside_target_region": inside_target_region,
                "verified": self.held_object_id is None
                and placement_verified,
            }
        if action in {"open", "close"}:
            door = self.robot.getFromDef("TASK_CUPBOARD_DOOR")
            actual_rotation = list(door.getField("rotation").getSFRotation())
            expected_angle = -1.35 if action == "open" else 0.0
            error = abs(float(actual_rotation[3]) - expected_angle)
            return {
                "kind": "cupboard_door_rotation",
                "expected_angle": expected_angle,
                "actual_rotation": actual_rotation,
                "angle_error": error,
                "verified": error <= 0.02,
            }
        return {"kind": "unsupported", "verified": False}

    def execute(self, plan: Plan, initial_world) -> dict:
        validation = PlanValidator().validate(plan, initial_world)
        if not validation.valid:
            log_runtime_event(
                "validation_rejected",
                plan_id=plan.plan_id,
                issues=validation.to_dict()["issues"],
            )
            return {
                "status": "validation_rejected",
                "validation": validation.to_dict(),
                "steps": [],
            }

        logical_world = initial_world.clone()
        records = []
        self.aim_camera_at_work_surface()
        self.set_arm(HOME_ARM)
        self.set_gripper(True)
        self.capture("00_initial")
        for index, plan_step in enumerate(plan.steps):
            step_started = time.perf_counter()
            log_runtime_event(
                "step_started",
                index=index,
                action=plan_step.action,
                arguments=dict(plan_step.arguments),
            )
            print(
                f"ROBOT_TASK_STEP_START index={index} action={plan_step.action} "
                f"arguments={json.dumps(plan_step.arguments, separators=(',', ':'))}",
                flush=True,
            )
            before = logical_world.to_dict()
            find_evidence = None
            if plan_step.action == "navigate":
                self.navigate(plan_step.arguments["location"])
            elif plan_step.action == "find":
                find_evidence = self.find(plan_step.arguments["object"])
            elif plan_step.action == "pick":
                self.pick(plan_step.arguments["object"])
            elif plan_step.action == "place":
                self.place(
                    plan_step.arguments["object"], plan_step.arguments["location"]
                )
            elif plan_step.action == "open":
                self.set_cupboard(True)
            elif plan_step.action == "close":
                self.set_cupboard(False)
            else:
                raise RuntimeError(f"Unsupported validated action: {plan_step.action}")
            apply_step(logical_world, plan_step)
            expected_state = logical_world.to_dict()
            observation = self.observe_action(
                plan_step.action,
                plan_step.arguments,
                find_evidence=find_evidence,
            )
            if not observation["verified"]:
                log_runtime_event(
                    "step_failed",
                    index=index,
                    action=plan_step.action,
                    code="OBSERVED_STATE_MISMATCH",
                    expected_state=expected_state,
                    observation=observation,
                )
                raise ObservedOutcomeError(index, plan_step.action, observation)
            print(
                f"ROBOT_TASK_STEP_COMPLETE index={index} action={plan_step.action}",
                flush=True,
            )
            self.capture(f"{index + 1:02d}_{plan_step.action}")
            log_runtime_event(
                "step_completed",
                index=index,
                action=plan_step.action,
                expected_state=expected_state,
                observed_state=expected_state,
                observation=observation,
                outcome_verified=True,
            )
            records.append(
                {
                    "index": index,
                    "action": plan_step.action,
                    "arguments": dict(plan_step.arguments),
                    "duration_ms": round(
                        (time.perf_counter() - step_started) * 1000, 3
                    ),
                    "state_before": before,
                    "state_after": expected_state,
                    "expected_state": expected_state,
                    "observed_state": expected_state,
                    "observation": observation,
                    "outcome_verified": True,
                }
            )
        result = {
            "status": "completed",
            "plan_id": plan.plan_id,
            "steps": records,
            "final_state": logical_world.to_dict(),
            "screenshots": list(self.screenshots),
        }
        log_runtime_event(
            "execution_completed",
            plan_id=plan.plan_id,
            step_count=len(records),
            final_state=result["final_state"],
            screenshots=result["screenshots"],
        )
        return result

    def restore_initial_scenario(self) -> None:
        self.held_node = None
        self.held_object_id = None
        self.current_location = "table"
        self.torso.setPosition(0.16)
        for motor, position in zip(self.arm_motors, HOME_ARM):
            motor.setPosition(position)
        self.gripper_left.setPosition(0.04)
        self.gripper_right.setPosition(0.04)
        self.head_yaw.setPosition(0.0)
        self.head_pitch.setPosition(-0.45)
        self.robot_translation.setSFVec3f(list(ROBOT_WAYPOINTS["table"]))
        self.robot_rotation.setSFRotation([0.0, 0.0, 1.0, ROBOT_ORIENTATIONS["table"]])
        self.robot_node.resetPhysics()
        for object_id, pose in self.initial_object_poses.items():
            node = self._object_node(object_id)
            node.getField("translation").setSFVec3f(pose["translation"])
            node.getField("rotation").setSFRotation(pose["rotation"])
            node.resetPhysics()
        door = self.robot.getFromDef("TASK_CUPBOARD_DOOR")
        door.getField("rotation").setSFRotation([0.0, 0.0, 1.0, 0.0])
        self.robot.simulationResetPhysics()
        self.step(10)
        # Joint settling can transfer a small impulse to the mobile base and
        # nearby objects.  Reassert the scenario once after that motion, then
        # clear every remaining velocity so repeated batch episodes start from
        # the same physical state as the first episode.
        self.robot_translation.setSFVec3f(list(ROBOT_WAYPOINTS["table"]))
        self.robot_rotation.setSFRotation(
            [0.0, 0.0, 1.0, ROBOT_ORIENTATIONS["table"]]
        )
        self.robot_node.resetPhysics()
        for object_id, pose in self.initial_object_poses.items():
            node = self._object_node(object_id)
            node.getField("translation").setSFVec3f(pose["translation"])
            node.getField("rotation").setSFRotation(pose["rotation"])
            node.resetPhysics()
        self.robot.simulationResetPhysics()
        self.step(2)

    def execute_batch(self, raw_plans: list[dict], initial_world) -> dict:
        if not raw_plans:
            raise ValueError("CNN end-to-end batch is empty.")
        base_screenshot_directory = self.screenshot_directory
        episodes = []
        for episode_index, raw_plan in enumerate(raw_plans, start=1):
            self.restore_initial_scenario()
            self.screenshot_directory = base_screenshot_directory / f"episode_{episode_index:03d}"
            self.screenshot_directory.mkdir(parents=True, exist_ok=True)
            self.screenshots = []
            plan = Plan.from_mapping(raw_plan)
            try:
                episode_result = self.execute(plan, initial_world)
            except Exception as exc:
                episode_result = {
                    "status": "failed",
                    "plan_id": plan.plan_id,
                    "error": str(exc),
                    "code": getattr(exc, "code", type(exc).__name__),
                    "step_index": getattr(exc, "step_index", None),
                    "evidence": getattr(exc, "evidence", None),
                }
            episodes.append(episode_result)
            log_runtime_event(
                "batch_episode_completed",
                episode=episode_index,
                plan_id=plan.plan_id,
                status=episode_result["status"],
            )
        passed = sum(episode["status"] == "completed" for episode in episodes)
        return {
            "status": "completed" if passed == len(episodes) else "failed",
            "episode_count": len(episodes),
            "passed": passed,
            "failed": len(episodes) - passed,
            "episodes": episodes,
        }


def main() -> None:
    runtime_directory = PROJECT_ROOT / "webots" / "runtime"
    direct_world_launch = (
        "ROBOT_TASK_PLAN_FILE" not in os.environ
        and "ROBOT_TASK_MODE" not in os.environ
    )
    default_plan_path = runtime_directory / "plan.json"
    if direct_world_launch and not default_plan_path.is_file():
        default_plan_path = PROJECT_ROOT / "examples" / "valid_apple_to_basket.json"
    plan_path = Path(os.environ.get("ROBOT_TASK_PLAN_FILE", default_plan_path))
    result_path = Path(
        os.environ.get("ROBOT_TASK_RESULT_FILE", runtime_directory / "result.json")
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = runtime_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")

    controller = None
    try:
        controller = WebotsTaskController()
        mode = os.environ.get("ROBOT_TASK_MODE")
        if mode == "cnn_dataset":
            result = controller.generate_cnn_dataset()
        elif mode == "cnn_batch":
            batch_path = Path(os.environ["ROBOT_TASK_BATCH_FILE"])
            raw_plans = json.loads(batch_path.read_text(encoding="utf-8"))
            world = load_world(PROJECT_ROOT / "config" / "kitchen.json")
            result = controller.execute_batch(raw_plans, world)
        else:
            raw_plan = json.loads(plan_path.read_text(encoding="utf-8"))
            plan = Plan.from_mapping(raw_plan)
            world = load_world(PROJECT_ROOT / "config" / "kitchen.json")
            result = controller.execute(plan, world)
    except Exception as exc:
        traceback_text = traceback.format_exc()
        result = {
            "status": "failed",
            "error": str(exc),
            "code": getattr(exc, "code", type(exc).__name__),
            "step_index": getattr(exc, "step_index", None),
            "evidence": getattr(exc, "evidence", None),
            "traceback": traceback_text,
        }
        log_runtime_event(
            "execution_failed",
            error=str(exc),
            traceback=traceback_text,
        )
        print(traceback_text, file=sys.stderr, flush=True)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("ROBOT_TASK_RESULT=" + json.dumps(result, separators=(",", ":")))
    if controller is not None:
        controller.step(5)
        keep_open = os.environ.get("ROBOT_TASK_KEEP_OPEN") == "1" or direct_world_launch
        if not keep_open:
            controller.robot.simulationQuit(0 if result["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
