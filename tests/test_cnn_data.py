from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from robot_planner.cnn_data import KitchenDetectionDataset, audit_dataset


class KitchenDatasetTests(unittest.TestCase):
    def test_loader_normalizes_image_and_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "images" / "train" / "000000.jpg"
            image_path.parent.mkdir(parents=True)
            Image.new("RGB", (100, 50), (30, 80, 140)).save(image_path)
            record = {
                "id": 0,
                "image": "images/train/000000.jpg",
                "width": 100,
                "height": 50,
                "split": "train",
                "seed": 1,
                "objects": [
                    {"label": "mug", "class_id": 0, "bbox_xywh": [10, 5, 20, 10]}
                ],
            }
            (root / "annotations.jsonl").write_text(
                json.dumps(record) + "\n", encoding="utf-8"
            )
            image, target = KitchenDetectionDataset(root, "train")[0]
            self.assertEqual((3, 320, 320), tuple(image.shape))
            for actual, expected in zip(
                target["boxes"][0].tolist(), (0.1, 0.1, 0.3, 0.3)
            ):
                self.assertAlmostEqual(expected, actual, places=6)
            self.assertEqual([0], target["labels"].tolist())

    def test_audit_rejects_duplicate_content_and_wrong_protocol_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for index, split in enumerate(("train", "validation")):
                image_path = root / "images" / split / f"{index}.png"
                image_path.parent.mkdir(parents=True)
                Image.new("RGB", (20, 20), "white").save(image_path)
                records.append(
                    {
                        "id": index,
                        "image": image_path.relative_to(root).as_posix(),
                        "width": 20,
                        "height": 20,
                        "split": split,
                        "seed": index,
                        "objects": [],
                    }
                )
            (root / "annotations.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            report = audit_dataset(root)
            self.assertEqual("failed", report["status"])
            self.assertTrue(
                any("duplicate image content" in problem for problem in report["problems"])
            )
            self.assertTrue(any("expected 8000" in problem for problem in report["problems"]))


if __name__ == "__main__":
    unittest.main()
