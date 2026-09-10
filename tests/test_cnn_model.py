from __future__ import annotations

import inspect
import unittest

import torch

from robot_planner.cnn_model import (
    KitchenObjectNet,
    decode_detections,
    detection_loss,
    initialization_fingerprint,
    parameter_count,
)


class KitchenObjectNetTests(unittest.TestCase):
    def test_custom_network_has_bounded_size_and_expected_heads(self) -> None:
        model = KitchenObjectNet()
        outputs = model(torch.zeros((2, 3, 320, 320)))
        self.assertLess(parameter_count(model), 6_000_000)
        self.assertEqual(
            [(2, 1, 40, 40), (2, 5, 40, 40), (2, 2, 40, 40), (2, 2, 40, 40)],
            [tuple(output.shape) for output in outputs],
        )

    def test_source_has_no_pretrained_or_imported_detector_path(self) -> None:
        source = inspect.getsource(KitchenObjectNet).lower()
        self.assertNotIn("torchvision", source)
        self.assertNotIn("pretrained", source)
        self.assertNotIn("load_state_dict_from_url", source)

    def test_random_initialization_is_seeded_but_not_constant(self) -> None:
        torch.manual_seed(41)
        first = initialization_fingerprint(KitchenObjectNet())
        torch.manual_seed(41)
        second = initialization_fingerprint(KitchenObjectNet())
        torch.manual_seed(42)
        third = initialization_fingerprint(KitchenObjectNet())
        self.assertEqual(first, second)
        self.assertNotEqual(first, third)

    def test_positive_and_negative_batch_has_finite_gradients(self) -> None:
        model = KitchenObjectNet()
        outputs = model(torch.randn((2, 3, 320, 320)))
        targets = [
            {
                "boxes": torch.tensor([[0.2, 0.2, 0.4, 0.5]]),
                "labels": torch.tensor([0]),
            },
            {
                "boxes": torch.empty((0, 4)),
                "labels": torch.empty(0, dtype=torch.long),
            },
        ]
        losses = detection_loss(outputs, targets)
        losses["total"].backward()
        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertTrue(
            all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )

    def test_decoder_returns_the_programmed_peak(self) -> None:
        objectness = torch.full((1, 1, 40, 40), -20.0)
        classes = torch.full((1, 5, 40, 40), -20.0)
        sizes = torch.zeros((1, 2, 40, 40))
        offsets = torch.zeros((1, 2, 40, 40))
        objectness[0, 0, 10, 12] = 20.0
        classes[0, 3, 10, 12] = 20.0
        result = decode_detections(
            (objectness, classes, sizes, offsets), confidence_threshold=0.9
        )[0]
        self.assertEqual([3], result["labels"].tolist())
        self.assertEqual(1, len(result["scores"]))

    def test_hard_background_peak_increases_objectness_loss(self) -> None:
        model = KitchenObjectNet()
        outputs = tuple(value.detach().clone() for value in model(torch.zeros((1, 3, 320, 320))))
        targets = [
            {
                "boxes": torch.empty((0, 4)),
                "labels": torch.empty(0, dtype=torch.long),
            }
        ]
        baseline = detection_loss(outputs, targets)["objectness"]
        hard_negative = list(outputs)
        hard_negative[0][0, 0, 10, 10] = 8.0
        increased = detection_loss(tuple(hard_negative), targets)["objectness"]
        self.assertGreater(float(increased), float(baseline) + 0.1)


if __name__ == "__main__":
    unittest.main()
