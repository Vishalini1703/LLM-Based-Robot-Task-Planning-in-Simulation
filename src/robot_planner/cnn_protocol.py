"""Frozen 180-episode perception protocol over Webots-rendered test scenes."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .cnn_constants import CLASS_NAMES
from .cnn_data import load_records
from .perception import KitchenObjectDetector


def _iou_xyxy(left: list[float], right: list[float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(left_area + right_area - intersection, 1e-9)


def _balanced_selection(records: list[dict[str, Any]]) -> list[tuple[dict[str, Any], str | None]]:
    selected: list[tuple[dict[str, Any], str | None]] = []
    used: set[int] = set()
    for label in CLASS_NAMES:
        for difficulty in ("easy", "medium", "hard"):
            eligible = [
                record
                for record in records
                if record["id"] not in used
                and record["difficulty"] == difficulty
                and any(item["label"] == label for item in record["objects"])
            ]
            if len(eligible) < 10:
                raise RuntimeError(f"Not enough {difficulty} test scenes for {label}")
            for record in eligible[:10]:
                selected.append((record, label))
                used.add(record["id"])
    for difficulty in ("easy", "medium", "hard"):
        eligible = [
            record
            for record in records
            if record["id"] not in used
            and record["difficulty"] == difficulty
            and not record["objects"]
        ]
        if len(eligible) < 10:
            raise RuntimeError(f"Not enough {difficulty} negative test scenes")
        for record in eligible[:10]:
            selected.append((record, None))
            used.add(record["id"])
    if len(selected) != 180:
        raise AssertionError(f"Expected 180 perception episodes, selected {len(selected)}")
    return selected


def run_perception_protocol(run_root: str | Path) -> dict[str, Any]:
    root = Path(run_root).resolve()
    dataset_root = root / "dataset"
    detector = KitchenObjectDetector(root / "KitchenObjectNet.onnx")
    records = [record for record in load_records(dataset_root) if record["split"] == "test"]
    selected = _balanced_selection(records)
    predictions_path = root / "logs" / "perception_180_predictions.jsonl"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.write_text("", encoding="utf-8")
    rows = []
    started = time.perf_counter()
    for episode, (record, requested) in enumerate(selected, start=1):
        rgb = np.asarray(Image.open(dataset_root / record["image"]).convert("RGB"))
        predictions = detector.detect(rgb)
        above_threshold = predictions
        if requested is None:
            passed = len(above_threshold) == 0
            best_iou = None
        else:
            width, height = record["width"], record["height"]
            ground_truth = []
            for item in record["objects"]:
                if item["label"] != requested:
                    continue
                x, y, box_width, box_height = item["bbox_xywh"]
                ground_truth.append(
                    [x / width, y / height, (x + box_width) / width, (y + box_height) / height]
                )
            overlaps = [
                _iou_xyxy(prediction["bbox_xyxy_normalized"], truth)
                for prediction in above_threshold
                if prediction["label"] == requested
                for truth in ground_truth
            ]
            best_iou = max(overlaps, default=0.0)
            passed = best_iou >= 0.5
        row = {
            "episode": episode,
            "record_id": record["id"],
            "seed": record["seed"],
            "difficulty": record["difficulty"],
            "requested_class": requested,
            "target_present": requested is not None,
            "passed": passed,
            "best_iou": best_iou,
            "inference_ms": detector.last_inference_ms,
            "detections": above_threshold,
        }
        rows.append(row)
        with predictions_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    by_class: dict[str, dict[str, Any]] = {}
    for label in (*CLASS_NAMES, "negative"):
        label_rows = [
            row
            for row in rows
            if (row["requested_class"] if row["requested_class"] is not None else "negative")
            == label
        ]
        by_class[label] = {
            "episodes": len(label_rows),
            "passed": sum(row["passed"] for row in label_rows),
            "accuracy": sum(row["passed"] for row in label_rows) / len(label_rows),
        }
    by_difficulty = {}
    for difficulty in ("easy", "medium", "hard"):
        difficulty_rows = [row for row in rows if row["difficulty"] == difficulty]
        by_difficulty[difficulty] = {
            "episodes": len(difficulty_rows),
            "passed": sum(row["passed"] for row in difficulty_rows),
            "accuracy": sum(row["passed"] for row in difficulty_rows)
            / len(difficulty_rows),
        }
    protocol_passed = all(
        by_class[label]["accuracy"] >= 0.85 for label in CLASS_NAMES
    ) and by_class["negative"]["accuracy"] >= 0.95
    report = {
        "status": "passed" if protocol_passed else "failed",
        "protocol": "180 held-out Webots-rendered seeded perception episodes",
        "episodes": len(rows),
        "passed": sum(row["passed"] for row in rows),
        "failed": sum(not row["passed"] for row in rows),
        "confidence_threshold": detector.confidence_threshold,
        "model_sha256": detector.model_sha256,
        "duration_seconds": time.perf_counter() - started,
        "latency_ms": {
            "median": float(np.median([row["inference_ms"] for row in rows])),
            "p95": float(np.percentile([row["inference_ms"] for row in rows], 95)),
        },
        "by_class": by_class,
        "by_difficulty": by_difficulty,
        "acceptance": {
            "each_target_class_accuracy_at_least_0_85": all(
                by_class[label]["accuracy"] >= 0.85 for label in CLASS_NAMES
            ),
            "negative_specificity_at_least_0_95": by_class["negative"][
                "accuracy"
            ]
            >= 0.95,
        },
        "prediction_log": predictions_path.relative_to(root).as_posix(),
    }
    (root / "perception_180.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    figure, axis = plt.subplots(figsize=(9, 4.8))
    names = list(by_class)
    axis.bar(names, [by_class[name]["accuracy"] for name in names], color="#1971c2")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Episode accuracy")
    axis.set_title("Frozen 180-episode Webots perception protocol")
    axis.tick_params(axis="x", rotation=22)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    plot_dir = root / "plots"
    plot_dir.mkdir(exist_ok=True)
    figure.savefig(plot_dir / "perception_180.png", dpi=180)
    plt.close(figure)
    return report
