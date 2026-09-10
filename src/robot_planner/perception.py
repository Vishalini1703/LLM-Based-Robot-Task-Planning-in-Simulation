"""ONNX-only runtime inference for the custom KitchenObjectNet detector."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

from .cnn_constants import CLASS_NAMES, INPUT_SIZE


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def _softmax(values: np.ndarray, axis: int) -> np.ndarray:
    shifted = values - values.max(axis=axis, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=axis, keepdims=True)


def _iou(one: np.ndarray, many: np.ndarray) -> np.ndarray:
    upper_left = np.maximum(one[:2], many[:, :2])
    lower_right = np.minimum(one[2:], many[:, 2:])
    intersection = np.prod(np.maximum(0.0, lower_right - upper_left), axis=1)
    one_area = np.prod(np.maximum(0.0, one[2:] - one[:2]))
    many_area = np.prod(np.maximum(0.0, many[:, 2:] - many[:, :2]), axis=1)
    return intersection / np.maximum(one_area + many_area - intersection, 1e-7)


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    order = np.argsort(scores)[::-1]
    kept: list[int] = []
    while order.size:
        current = int(order[0])
        kept.append(current)
        if order.size == 1:
            break
        remaining = order[1:]
        order = remaining[_iou(boxes[current], boxes[remaining]) <= threshold]
    return kept


def bgra_buffer_to_rgb(buffer: bytes, width: int, height: int) -> np.ndarray:
    """Convert Webots' packed BGRA camera buffer into an owned RGB array."""
    bgra = np.frombuffer(buffer, dtype=np.uint8)
    expected = width * height * 4
    if bgra.size != expected:
        raise ValueError(f"Expected {expected} BGRA bytes, received {bgra.size}")
    return bgra.reshape((height, width, 4))[:, :, [2, 1, 0]].copy()


def confirm_temporal_detections(
    frames: list[list[dict[str, Any]]],
    requested_label: str,
    *,
    required_frames: int = 2,
    minimum_iou: float = 0.10,
) -> list[dict[str, Any]]:
    """Return consistent highest-confidence detections across recent frames."""
    candidates = []
    for detections in frames:
        matching = [item for item in detections if item["label"] == requested_label]
        if matching:
            candidates.append(max(matching, key=lambda item: item["confidence"]))
    if len(candidates) < required_frames:
        return []
    reference = np.asarray(candidates[-1]["bbox_xyxy_normalized"], dtype=np.float32)
    consistent = [
        item
        for item in candidates
        if float(
            _iou(
                reference,
                np.asarray([item["bbox_xyxy_normalized"]], dtype=np.float32),
            )[0]
        )
        >= minimum_iou
    ]
    return consistent if len(consistent) >= required_frames else []


class KitchenObjectDetector:
    """Load KitchenObjectNet ONNX and return pixel-space detections."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        confidence_threshold: float | None = None,
        nms_threshold: float = 0.45,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"KitchenObjectNet ONNX model not found: {self.model_path}")
        if confidence_threshold is None:
            evaluation_path = self.model_path.parent / "evaluation.json"
            if not evaluation_path.is_file():
                raise FileNotFoundError(
                    "No confidence threshold was provided and evaluation.json is missing."
                )
            confidence_threshold = float(
                json.loads(evaluation_path.read_text(encoding="utf-8"))[
                    "selected_confidence_threshold"
                ]
            )
        if not 0.0 < confidence_threshold < 1.0:
            raise ValueError("confidence threshold must be between zero and one")
        self.confidence_threshold = float(confidence_threshold)
        self.nms_threshold = nms_threshold
        self.session = ort.InferenceSession(
            str(self.model_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        digest = hashlib.sha256()
        with self.model_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        self.model_sha256 = digest.hexdigest()

    @staticmethod
    def preprocess(rgb: np.ndarray) -> np.ndarray:
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Expected an RGB image shaped height x width x 3")
        image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
        image = image.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - 0.5) / 0.5
        return np.ascontiguousarray(array.transpose(2, 0, 1)[None])

    def detect(self, rgb: np.ndarray) -> list[dict[str, Any]]:
        original_height, original_width = rgb.shape[:2]
        started = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: self.preprocess(rgb)})
        self.last_inference_ms = (time.perf_counter() - started) * 1000.0
        objectness_logits, class_logits, size_logits, offset_logits = outputs
        objectness = _sigmoid(objectness_logits[0, 0])
        class_probabilities = _softmax(class_logits[0], axis=0)
        grid_height, grid_width = objectness.shape
        padded = np.pad(objectness, 1, mode="constant", constant_values=-1.0)
        local_maximum = np.ones_like(objectness, dtype=bool)
        for offset_y in range(3):
            for offset_x in range(3):
                local_maximum &= objectness >= padded[
                    offset_y : offset_y + grid_height,
                    offset_x : offset_x + grid_width,
                ]
        objectness = objectness * local_maximum
        candidates: list[tuple[float, int, int, int]] = []
        rejected_candidates: list[tuple[float, int, int, int]] = []
        for class_id in range(len(CLASS_NAMES)):
            combined = objectness * class_probabilities[class_id]
            selected_y, selected_x = np.where(combined >= self.confidence_threshold)
            for y, x in zip(selected_y.tolist(), selected_x.tolist()):
                candidates.append((float(combined[y, x]), class_id, y, x))
            flat = combined.reshape(-1)
            for index in np.argsort(flat)[-3:]:
                score = float(flat[index])
                if 0.0 < score < self.confidence_threshold:
                    y, x = divmod(int(index), grid_width)
                    rejected_candidates.append((score, class_id, y, x))
        self.last_rejected = [
            {
                "label": CLASS_NAMES[class_id],
                "confidence": score,
                "grid_cell": [x, y],
                "reason": "below_frozen_confidence_threshold",
            }
            for score, class_id, y, x in sorted(rejected_candidates, reverse=True)[:10]
        ]
        candidates.sort(reverse=True)
        candidates = candidates[:30]
        detections: list[dict[str, Any]] = []
        for class_id in range(len(CLASS_NAMES)):
            class_candidates = [item for item in candidates if item[1] == class_id]
            if not class_candidates:
                continue
            boxes = []
            scores = []
            for score, _, y, x in class_candidates:
                size = _sigmoid(size_logits[0, :, y, x])
                offset = _sigmoid(offset_logits[0, :, y, x])
                centre = np.array(
                    [(x + offset[0]) / grid_width, (y + offset[1]) / grid_height]
                )
                boxes.append(np.clip(np.r_[centre - size / 2, centre + size / 2], 0, 1))
                scores.append(score)
            boxes_array = np.asarray(boxes, dtype=np.float32)
            scores_array = np.asarray(scores, dtype=np.float32)
            for kept in _nms(boxes_array, scores_array, self.nms_threshold):
                x1, y1, x2, y2 = boxes_array[kept]
                detections.append(
                    {
                        "label": CLASS_NAMES[class_id],
                        "class_id": class_id,
                        "confidence": float(scores_array[kept]),
                        "bbox_xyxy_normalized": [
                            float(x1), float(y1), float(x2), float(y2)
                        ],
                        "bbox_xywh": [
                            float(x1 * original_width),
                            float(y1 * original_height),
                            float((x2 - x1) * original_width),
                            float((y2 - y1) * original_height),
                        ],
                    }
                )
        detections.sort(key=lambda item: item["confidence"], reverse=True)
        return detections

    @staticmethod
    def annotate(rgb: np.ndarray, detections: list[dict[str, Any]]) -> Image.Image:
        image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
        draw = ImageDraw.Draw(image)
        for detection in detections:
            x, y, width, height = detection["bbox_xywh"]
            draw.rectangle((x, y, x + width, y + height), outline="#22c55e", width=3)
            draw.text(
                (x, max(0, y - 13)),
                f"{detection['label']} {detection['confidence']:.2f}",
                fill="#16a34a",
            )
        return image
