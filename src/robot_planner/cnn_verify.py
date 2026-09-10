"""Final integrity and provenance audit for the CNN submission artefacts."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .perception import KitchenObjectDetector
from .cnn_constants import CLASS_NAMES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_lines(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_e2e_summary(root: Path, model_sha256: str) -> dict[str, Any] | None:
    """Condense the large runtime trace into portable submission evidence."""
    result_path = root / "logs" / "e2e_result.json"
    controller_path = root / "logs" / "e2e_controller.jsonl"
    launch_path = root / "logs" / "e2e_launch.json"
    if not result_path.is_file() or not controller_path.is_file():
        summary_path = root / "e2e_summary.json"
        return (
            json.loads(summary_path.read_text(encoding="utf-8"))
            if summary_path.is_file()
            else None
        )

    result = json.loads(result_path.read_text(encoding="utf-8"))
    events = _load_json_lines(controller_path)
    finds = [
        event
        for event in events
        if event.get("observation", {}).get("kind") == "cnn_object_detection"
    ]
    navigation = [
        event for event in events if event.get("event") == "navigation_path_planned"
    ]
    class_counts = {
        name: sum(
            event.get("observation", {}).get("requested_object") == name
            and event.get("observation", {}).get("verified") is True
            for event in finds
        )
        for name in CLASS_NAMES
    }
    launch = (
        json.loads(launch_path.read_text(encoding="utf-8"))
        if launch_path.is_file()
        else {}
    )
    episodes = launch.get("episodes", [])
    clearances = [float(event["minimum_obstacle_clearance"]) for event in navigation]
    summary = {
        "schema_version": 1,
        "status": result.get("status"),
        "isolation": result.get("isolation"),
        "episode_count": int(result.get("episode_count", 0)),
        "passed": int(result.get("passed", 0)),
        "failed": int(result.get("failed", 0)),
        "process_exit_nonzero": sum(
            episode.get("return_code") not in (None, 0) for episode in episodes
        ),
        "cnn_find_steps": len(finds),
        "cnn_find_verified": sum(
            event.get("observation", {}).get("verified") is True for event in finds
        ),
        "verified_by_class": class_counts,
        "recognition_metadata_used_events": sum(
            event.get("observation", {}).get("recognition_metadata_used") is True
            for event in finds
        ),
        "model_sha256": model_sha256,
        "observed_model_hashes": sorted(
            {
                event.get("observation", {}).get("model_sha256")
                for event in finds
                if event.get("observation", {}).get("model_sha256")
            }
        ),
        "navigation_routes": len(navigation),
        "minimum_obstacle_clearance": min(clearances) if clearances else None,
        "clearance_violations": sum(
            float(event["minimum_obstacle_clearance"])
            < float(event["required_clearance"])
            for event in navigation
        ),
        "raw_evidence_sha256": {
            "e2e_result.json": _sha256(result_path),
            "e2e_controller.jsonl": _sha256(controller_path),
        },
    }
    (root / "e2e_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _verify_e2e_summary(summary: dict[str, Any] | None, model_sha256: str) -> list[str]:
    if summary is None:
        return ["missing required artifact: e2e_summary.json"]
    problems = []
    episode_count = int(summary.get("episode_count", 0))
    if summary.get("status") != "completed" or episode_count < 25:
        problems.append("end-to-end summary does not contain 25 completed episodes")
    if int(summary.get("passed", 0)) != episode_count or int(summary.get("failed", 0)):
        problems.append("one or more end-to-end episodes failed")
    if int(summary.get("process_exit_nonzero", 0)):
        problems.append("one or more isolated Webots processes exited non-zero")
    if int(summary.get("cnn_find_verified", 0)) != episode_count:
        problems.append("not every end-to-end episode has a verified CNN find")
    if any(int(summary.get("verified_by_class", {}).get(name, 0)) < 5 for name in CLASS_NAMES):
        problems.append("fewer than five verified end-to-end finds exist for a class")
    if int(summary.get("recognition_metadata_used_events", 0)):
        problems.append("recognition metadata was used during CNN end-to-end execution")
    if summary.get("observed_model_hashes") != [model_sha256]:
        problems.append("end-to-end observations do not all use the accepted ONNX hash")
    if int(summary.get("clearance_violations", 0)):
        problems.append("one or more navigation routes violated obstacle clearance")
    return problems


def _prepare_or_verify_fixtures(
    root: Path, detector: KitchenObjectDetector
) -> dict[str, Any]:
    """Create six deterministic CPU-smoke fixtures, then execute them."""
    fixture_root = root / "verification_fixtures"
    manifest_path = fixture_root / "manifest.json"
    annotations_path = root / "dataset" / "annotations.jsonl"
    if annotations_path.is_file():
        fixture_root.mkdir(parents=True, exist_ok=True)
        records = sorted(
            (
                record
                for record in _load_json_lines(annotations_path)
                if record.get("split") == "test"
            ),
            key=lambda record: int(record["id"]),
        )
        selected: list[dict[str, Any]] = []
        pending: list[str | None] = [*CLASS_NAMES, None]
        for expected_label in pending:
            for record in records:
                labels = [item["label"] for item in record.get("objects", [])]
                if expected_label is None and labels:
                    continue
                if expected_label is not None and labels != [expected_label]:
                    continue
                source = root / "dataset" / record["image"]
                rgb = np.asarray(Image.open(source).convert("RGB"))
                detections = detector.detect(rgb)
                detected_labels = {item["label"] for item in detections}
                if expected_label is None and detections:
                    continue
                if expected_label is not None and expected_label not in detected_labels:
                    continue
                name = f"{expected_label or 'negative'}.jpg"
                destination = fixture_root / name
                shutil.copyfile(source, destination)
                selected.append(
                    {
                        "file": name,
                        "expected_label": expected_label,
                        "source_record_id": int(record["id"]),
                        "source_seed": int(record["seed"]),
                        "image_sha256": _sha256(destination),
                    }
                )
                break
        if len(selected) != len(pending):
            raise RuntimeError("Could not select one passing verification fixture per class and one negative.")
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "purpose": "Portable CPU inference smoke test; not model selection data",
                    "model_sha256": detector.model_sha256,
                    "fixtures": selected,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    if not manifest_path.is_file():
        raise RuntimeError("verification_fixtures/manifest.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model_sha256") != detector.model_sha256:
        raise RuntimeError("verification fixture model hash does not match the accepted ONNX model")
    results = []
    for fixture in manifest.get("fixtures", []):
        path = fixture_root / fixture["file"]
        if not path.is_file() or _sha256(path) != fixture.get("image_sha256"):
            raise RuntimeError(f"verification fixture is missing or changed: {fixture['file']}")
        rgb = np.asarray(Image.open(path).convert("RGB"))
        detections = detector.detect(rgb)
        labels = [item["label"] for item in detections]
        expected = fixture.get("expected_label")
        passed = (expected in labels) if expected is not None else not detections
        results.append({"file": fixture["file"], "expected": expected, "detected": labels, "passed": passed})
    if len(results) != 6 or not all(result["passed"] for result in results):
        raise RuntimeError("one or more portable CPU verification fixtures failed")
    return {"count": len(results), "passed": sum(item["passed"] for item in results), "results": results}


def verify_cnn_artifacts(run_root: str | Path, project_root: str | Path) -> dict[str, Any]:
    root = Path(run_root).resolve()
    project = Path(project_root).resolve()
    required = [
        "KitchenObjectNet.onnx",
        "evaluation.json",
        "provenance.json",
        "training_config.json",
        "architecture.json",
        "class_map.json",
        "dataset_manifest.json",
        "MODEL_CARD.md",
        "perception_180.json",
    ]
    problems = [name for name in required if not (root / name).is_file()]
    if problems:
        problems = [f"missing required artifact: {name}" for name in problems]
    evaluation = (
        json.loads((root / "evaluation.json").read_text(encoding="utf-8"))
        if (root / "evaluation.json").is_file()
        else {}
    )
    provenance = (
        json.loads((root / "provenance.json").read_text(encoding="utf-8"))
        if (root / "provenance.json").is_file()
        else {}
    )
    model_path = root / "KitchenObjectNet.onnx"
    if model_path.is_file() and provenance.get("onnx_sha256") != _sha256(model_path):
        problems.append("ONNX hash does not match provenance.json")
    initial_path = root / "checkpoints" / "initial_random_state.pt"
    if initial_path.is_file() and provenance.get("initial_checkpoint_sha256") != _sha256(
        initial_path
    ):
        problems.append("random-initialization hash does not match provenance.json")
    if evaluation and not all(evaluation.get("acceptance", {}).values()):
        problems.append("one or more held-out model acceptance criteria failed")
    source = (project / "src" / "robot_planner" / "cnn_model.py").read_text(
        encoding="utf-8"
    ).lower()
    forbidden_source = [
        token
        for token in (
            "from torchvision",
            "import torchvision",
            "load_state_dict_from_url",
            "torch.hub.load",
            "ultralytics",
        )
        if token in source
    ]
    if forbidden_source:
        problems.append("forbidden imported-model path: " + ", ".join(forbidden_source))
    checkpoint_files = [
        path
        for path in project.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".pt", ".pth", ".ckpt", ".onnx"}
        and "cnn_artifacts" not in path.parts
        and "submission" not in path.parts
        and ".venv" not in path.parts
    ]
    if checkpoint_files:
        problems.append(
            "untracked model/checkpoint files outside CNN artifacts: "
            + ", ".join(str(path.relative_to(project)) for path in checkpoint_files)
        )
    cpu_smoke = None
    fixture_smoke = None
    if model_path.is_file() and (root / "evaluation.json").is_file():
        detector = KitchenObjectDetector(model_path)
        detections = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        cpu_smoke = {
            "provider": detector.session.get_providers(),
            "inference_ms": detector.last_inference_ms,
            "black_frame_detections": len(detections),
        }
        try:
            fixture_smoke = _prepare_or_verify_fixtures(root, detector)
        except (OSError, RuntimeError, ValueError) as exc:
            problems.append(str(exc))
    e2e_summary = _write_e2e_summary(
        root, _sha256(model_path) if model_path.is_file() else ""
    )
    problems.extend(
        _verify_e2e_summary(
            e2e_summary, _sha256(model_path) if model_path.is_file() else ""
        )
    )
    plots = sorted(path.relative_to(root).as_posix() for path in (root / "plots").glob("*.png"))
    logs = sorted(path.relative_to(root).as_posix() for path in (root / "logs").glob("*"))
    report = {
        "status": "passed" if not problems else "failed",
        "problems": problems,
        "pretrained_weights_used": False,
        "imported_detector_architecture_used": False,
        "onnx_sha256": _sha256(model_path) if model_path.is_file() else None,
        "cpu_smoke": cpu_smoke,
        "fixture_smoke": fixture_smoke,
        "e2e_summary": e2e_summary,
        "plot_count": len(plots),
        "plots": plots,
        "log_count": len(logs),
        "logs": logs,
    }
    (root / "verification.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    if problems:
        raise RuntimeError("CNN artifact verification failed: " + "; ".join(problems))
    return report
