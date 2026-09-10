from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np

from robot_planner.perception import (
    KitchenObjectDetector,
    bgra_buffer_to_rgb,
    confirm_temporal_detections,
)


class _Input:
    name = "images"


class _FakeSession:
    def get_inputs(self):
        return [_Input()]

    def run(self, _outputs, inputs):
        assert inputs["images"].shape == (1, 3, 320, 320)
        objectness = np.full((1, 1, 40, 40), -20.0, dtype=np.float32)
        classes = np.full((1, 5, 40, 40), -20.0, dtype=np.float32)
        sizes = np.zeros((1, 2, 40, 40), dtype=np.float32)
        offsets = np.zeros((1, 2, 40, 40), dtype=np.float32)
        objectness[0, 0, 10, 12] = 20.0
        classes[0, 2, 10, 12] = 20.0
        return objectness, classes, sizes, offsets


class _EmptySession(_FakeSession):
    def run(self, _outputs, inputs):
        assert inputs["images"].shape == (1, 3, 320, 320)
        return (
            np.full((1, 1, 40, 40), -20.0, dtype=np.float32),
            np.full((1, 5, 40, 40), -20.0, dtype=np.float32),
            np.zeros((1, 2, 40, 40), dtype=np.float32),
            np.zeros((1, 2, 40, 40), dtype=np.float32),
        )


class PerceptionTests(unittest.TestCase):
    def test_webots_bgra_conversion(self) -> None:
        buffer = bytes((1, 2, 3, 255, 10, 20, 30, 255))
        rgb = bgra_buffer_to_rgb(buffer, 2, 1)
        self.assertEqual([[[3, 2, 1], [30, 20, 10]]], rgb.tolist())
        with self.assertRaises(ValueError):
            bgra_buffer_to_rgb(buffer, 3, 1)

    def test_onnx_output_decoding_and_class_mapping(self) -> None:
        detector = KitchenObjectDetector.__new__(KitchenObjectDetector)
        detector.session = _FakeSession()
        detector.input_name = "images"
        detector.confidence_threshold = 0.90
        detector.nms_threshold = 0.45
        detections = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertEqual(1, len(detections))
        self.assertEqual("orange", detections[0]["label"])
        self.assertGreater(detections[0]["confidence"], 0.99)
        self.assertGreater(detector.last_inference_ms, 0)

    def test_no_detection_returns_an_empty_result(self) -> None:
        detector = KitchenObjectDetector.__new__(KitchenObjectDetector)
        detector.session = _EmptySession()
        detector.input_name = "images"
        detector.confidence_threshold = 0.35
        detector.nms_threshold = 0.45
        self.assertEqual([], detector.detect(np.zeros((480, 640, 3), dtype=np.uint8)))

    def test_missing_and_corrupt_models_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(FileNotFoundError):
                KitchenObjectDetector(root / "missing.onnx", confidence_threshold=0.35)
            corrupt = root / "corrupt.onnx"
            corrupt.write_bytes(b"not an ONNX model")
            with self.assertRaises(Exception):
                KitchenObjectDetector(corrupt, confidence_threshold=0.35)

    def test_preprocess_has_frozen_shape_and_normalisation(self) -> None:
        black = KitchenObjectDetector.preprocess(
            np.zeros((480, 640, 3), dtype=np.uint8)
        )
        white = KitchenObjectDetector.preprocess(
            np.full((480, 640, 3), 255, dtype=np.uint8)
        )
        self.assertEqual((1, 3, 320, 320), black.shape)
        self.assertTrue(np.allclose(black, -1.0))
        self.assertTrue(np.allclose(white, 1.0))

    def test_temporal_confirmation_requires_consistent_two_of_three(self) -> None:
        detection = {
            "label": "mug",
            "confidence": 0.92,
            "bbox_xyxy_normalized": [0.2, 0.2, 0.4, 0.5],
        }
        confirmed = confirm_temporal_detections(
            [[detection], [], [{**detection, "confidence": 0.88}]], "mug"
        )
        self.assertEqual(2, len(confirmed))
        inconsistent = {
            **detection,
            "bbox_xyxy_normalized": [0.7, 0.7, 0.9, 0.9],
        }
        self.assertEqual(
            [], confirm_temporal_detections([[detection], [inconsistent], []], "mug")
        )
        self.assertEqual(
            [], confirm_temporal_detections([[detection], [], []], "mug")
        )


if __name__ == "__main__":
    unittest.main()
