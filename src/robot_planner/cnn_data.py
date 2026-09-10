"""Dataset loading and integrity auditing for KitchenObjectNet."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset

from .cnn_constants import CLASS_NAMES, CLASS_TO_INDEX, INPUT_SIZE


ANNOTATION_FILE = "annotations.jsonl"


def load_records(dataset_root: str | Path) -> list[dict[str, Any]]:
    path = Path(dataset_root) / ANNOTATION_FILE
    if not path.is_file():
        raise FileNotFoundError(f"Dataset annotations were not found: {path}")
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid annotation JSON at line {line_number}") from exc
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_dataset(dataset_root: str | Path, expected_total: int = 8000) -> dict[str, Any]:
    root = Path(dataset_root)
    records = load_records(root)
    problems: list[str] = []
    split_counts = Counter()
    class_counts = Counter()
    negative_count = 0
    image_hashes: dict[str, str] = {}
    hashes_by_image: dict[str, str] = {}
    seeds_by_split: dict[str, set[int]] = {"train": set(), "validation": set(), "test": set()}
    allowed_splits = set(seeds_by_split)
    for record_index, record in enumerate(records):
        split = record.get("split")
        if split not in allowed_splits:
            problems.append(f"record {record_index}: invalid split {split!r}")
            continue
        split_counts[split] += 1
        seed = record.get("seed")
        if not isinstance(seed, int):
            problems.append(f"record {record_index}: seed is not an integer")
        else:
            seeds_by_split[split].add(seed)
        image_relative = record.get("image")
        if not isinstance(image_relative, str):
            problems.append(f"record {record_index}: image path is missing")
            continue
        image_path = root / image_relative
        if not image_path.is_file():
            problems.append(f"record {record_index}: missing image {image_relative}")
            continue
        digest = _sha256(image_path)
        hashes_by_image[image_relative] = digest
        previous = image_hashes.get(digest)
        if previous is not None:
            problems.append(f"duplicate image content: {previous} and {image_relative}")
        image_hashes[digest] = image_relative
        width = record.get("width")
        height = record.get("height")
        objects = record.get("objects")
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            problems.append(f"record {record_index}: invalid image dimensions")
            continue
        if not isinstance(objects, list):
            problems.append(f"record {record_index}: objects must be a list")
            continue
        if not objects:
            negative_count += 1
        for object_index, item in enumerate(objects):
            label = item.get("label")
            class_id = item.get("class_id")
            box = item.get("bbox_xywh")
            if label not in CLASS_TO_INDEX or CLASS_TO_INDEX.get(label) != class_id:
                problems.append(
                    f"record {record_index} object {object_index}: invalid class mapping"
                )
                continue
            if not isinstance(box, list) or len(box) != 4:
                problems.append(f"record {record_index} object {object_index}: invalid box")
                continue
            x, y, box_width, box_height = (float(value) for value in box)
            if (
                x < 0
                or y < 0
                or box_width <= 1
                or box_height <= 1
                or x + box_width > width + 1
                or y + box_height > height + 1
            ):
                problems.append(
                    f"record {record_index} object {object_index}: box outside image"
                )
            class_counts[label] += 1
    if len(records) != expected_total:
        problems.append(f"expected {expected_total} records, found {len(records)}")
    expected_splits = {"train": 5600, "validation": 1200, "test": 1200}
    for split, expected in expected_splits.items():
        if split_counts[split] != expected:
            problems.append(f"{split}: expected {expected}, found {split_counts[split]}")
    expected_negatives = expected_total // 5
    expected_per_class = (expected_total - expected_negatives) // len(CLASS_NAMES)
    for label in CLASS_NAMES:
        if class_counts[label] != expected_per_class:
            problems.append(
                f"{label}: expected {expected_per_class} instances, "
                f"found {class_counts[label]}"
            )
    if negative_count != expected_negatives:
        problems.append(
            f"negative frames: expected {expected_negatives}, found {negative_count}"
        )
    for left_index, left in enumerate(sorted(allowed_splits)):
        for right in sorted(allowed_splits)[left_index + 1 :]:
            overlap = seeds_by_split[left] & seeds_by_split[right]
            if overlap:
                problems.append(f"seed leakage between {left} and {right}: {len(overlap)}")
    report = {
        "status": "passed" if not problems else "failed",
        # Keep manifests portable.  The caller already determines where the
        # run root lives, so recording a machine-specific absolute path adds
        # no provenance value.
        "dataset_root": root.name,
        "record_count": len(records),
        "split_counts": dict(split_counts),
        "class_instance_counts": {name: class_counts[name] for name in CLASS_NAMES},
        "negative_count": negative_count,
        "unique_image_hashes": len(image_hashes),
        "image_sha256": hashes_by_image,
        "problems": problems,
    }
    return report


def build_dataset_manifest(
    dataset_root: str | Path,
    world_path: str | Path,
    audit_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the immutable provenance record after a successful audit."""
    root = Path(dataset_root)
    world = Path(world_path)
    audit = audit_report if audit_report is not None else audit_dataset(root)
    if audit["status"] != "passed":
        raise RuntimeError("Cannot manifest a dataset that failed its audit.")
    generation_path = root / "generation_summary.json"
    generation = (
        json.loads(generation_path.read_text(encoding="utf-8"))
        if generation_path.is_file()
        else {}
    )
    return {
        "schema_version": 1,
        "dataset": "KitchenObjectNet Webots RGB v1",
        "webots_version": generation.get("webots_version", "R2025a"),
        "world": "webots/worlds/kitchen_llm.wbt",
        "world_sha256": _sha256(world),
        "annotation_sha256": _sha256(root / ANNOTATION_FILE),
        "generation_seed": generation.get("seed", 20260808),
        "class_map": {str(index): name for index, name in enumerate(CLASS_NAMES)},
        "split_protocol": {"train": 5600, "validation": 1200, "test": 1200},
        "negative_protocol": {"count": 1600, "fraction": 0.20},
        "audit": audit,
    }


class KitchenDetectionDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        split: str,
        *,
        augment: bool = False,
        seed: int = 1337,
    ) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown dataset split: {split}")
        self.root = Path(dataset_root)
        self.records = [record for record in load_records(self.root) if record["split"] == split]
        self.augment = augment
        self.seed = seed

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = Image.open(self.root / record["image"]).convert("RGB")
        width, height = image.size
        boxes = []
        labels = []
        for item in record["objects"]:
            x, y, box_width, box_height = item["bbox_xywh"]
            boxes.append(
                [x / width, y / height, (x + box_width) / width, (y + box_height) / height]
            )
            labels.append(int(item["class_id"]))
        rng = random.Random(self.seed + index * 104729)
        if self.augment:
            if rng.random() < 0.45:
                left_fraction = rng.uniform(0.0, 0.035)
                top_fraction = rng.uniform(0.0, 0.035)
                right_fraction = 1.0 - rng.uniform(0.0, 0.035)
                bottom_fraction = 1.0 - rng.uniform(0.0, 0.035)
                image = image.crop(
                    (
                        round(left_fraction * width),
                        round(top_fraction * height),
                        round(right_fraction * width),
                        round(bottom_fraction * height),
                    )
                )
                crop_width = right_fraction - left_fraction
                crop_height = bottom_fraction - top_fraction
                boxes = [
                    [
                        max(0.0, (box[0] - left_fraction) / crop_width),
                        max(0.0, (box[1] - top_fraction) / crop_height),
                        min(1.0, (box[2] - left_fraction) / crop_width),
                        min(1.0, (box[3] - top_fraction) / crop_height),
                    ]
                    for box in boxes
                ]
            if rng.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                boxes = [[1.0 - box[2], box[1], 1.0 - box[0], box[3]] for box in boxes]
            image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.72, 1.28))
            image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.75, 1.25))
            image = ImageEnhance.Color(image).enhance(rng.uniform(0.78, 1.22))
            if rng.random() < 0.18:
                image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.1, 1.1)))
        image = image.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        if self.augment:
            noise_rng = np.random.default_rng(self.seed + index * 65537)
            array = np.clip(array + noise_rng.normal(0.0, rng.uniform(0, 0.018), array.shape), 0, 1)
        tensor = torch.from_numpy(array.transpose(2, 0, 1)).float()
        tensor = (tensor - 0.5) / 0.5
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.long),
            "image_id": record["id"],
        }
        return tensor, target


def collate_detection_batch(batch):
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)
