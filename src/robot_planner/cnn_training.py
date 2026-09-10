"""Training, evaluation, visualisation and export for KitchenObjectNet."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import platform
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image, ImageDraw
from torch import Tensor
from torch.utils.data import DataLoader

from .cnn_data import (
    KitchenDetectionDataset,
    audit_dataset,
    collate_detection_batch,
    load_records,
)
from .cnn_model import (
    CLASS_NAMES,
    INPUT_SIZE,
    KitchenObjectNet,
    box_iou,
    decode_detections,
    detection_loss,
    initialization_fingerprint,
    parameter_count,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    if recall.size == 0:
        return 0.0
    levels = np.linspace(0.0, 1.0, 101)
    values = []
    for level in levels:
        eligible = precision[recall >= level]
        values.append(float(eligible.max()) if eligible.size else 0.0)
    return float(np.mean(values))


def _class_matches(
    predictions: list[dict[str, Tensor]],
    targets: list[dict[str, Tensor]],
    class_id: int,
    iou_threshold: float,
) -> tuple[list[tuple[float, int]], int]:
    ground_truth: dict[int, Tensor] = {}
    matched: dict[int, set[int]] = {}
    prediction_rows: list[tuple[float, int, Tensor]] = []
    total_ground_truth = 0
    for image_index, target in enumerate(targets):
        mask = target["labels"] == class_id
        boxes = target["boxes"][mask]
        ground_truth[image_index] = boxes
        matched[image_index] = set()
        total_ground_truth += boxes.shape[0]
        prediction = predictions[image_index]
        prediction_mask = prediction["labels"] == class_id
        for score, box in zip(
            prediction["scores"][prediction_mask], prediction["boxes"][prediction_mask]
        ):
            prediction_rows.append((float(score), image_index, box))
    prediction_rows.sort(key=lambda row: row[0], reverse=True)
    scored_matches: list[tuple[float, int]] = []
    for score, image_index, box in prediction_rows:
        candidates = ground_truth[image_index]
        is_true_positive = 0
        if candidates.numel():
            overlaps = box_iou(box.unsqueeze(0), candidates).squeeze(0)
            best_overlap, best_index = overlaps.max(dim=0)
            index_value = int(best_index)
            if (
                float(best_overlap) >= iou_threshold
                and index_value not in matched[image_index]
            ):
                matched[image_index].add(index_value)
                is_true_positive = 1
        scored_matches.append((score, is_true_positive))
    return scored_matches, total_ground_truth


def calculate_detection_metrics(
    predictions: list[dict[str, Tensor]],
    targets: list[dict[str, Tensor]],
    *,
    confidence_threshold: float,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    per_class: dict[str, dict[str, float | int]] = {}
    average_precisions = []
    total_tp = total_fp = total_fn = 0
    for class_id, class_name in enumerate(CLASS_NAMES):
        rows, gt_count = _class_matches(predictions, targets, class_id, iou_threshold)
        if rows:
            true_values = np.array([row[1] for row in rows], dtype=np.float64)
            false_values = 1.0 - true_values
            cumulative_true = np.cumsum(true_values)
            cumulative_false = np.cumsum(false_values)
            recall_curve = cumulative_true / max(gt_count, 1)
            precision_curve = cumulative_true / np.maximum(
                cumulative_true + cumulative_false, 1e-9
            )
            average_precision = _average_precision(recall_curve, precision_curve)
        else:
            average_precision = 0.0
        filtered = [match for score, match in rows if score >= confidence_threshold]
        true_positive = sum(filtered)
        false_positive = len(filtered) - true_positive
        false_negative = max(0, gt_count - true_positive)
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(gt_count, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        per_class[class_name] = {
            "average_precision": average_precision,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "ground_truth": gt_count,
        }
        average_precisions.append(average_precision)
        total_tp += true_positive
        total_fp += false_positive
        total_fn += false_negative
    macro_precision = float(np.mean([item["precision"] for item in per_class.values()]))
    macro_recall = float(np.mean([item["recall"] for item in per_class.values()]))
    macro_f1 = float(np.mean([item["f1"] for item in per_class.values()]))
    negative_images = sum(target["labels"].numel() == 0 for target in targets)
    negative_false_detections = 0
    for prediction, target in zip(predictions, targets):
        if target["labels"].numel() == 0:
            negative_false_detections += int(
                (prediction["scores"] >= confidence_threshold).sum()
            )
    return {
        "iou_threshold": iou_threshold,
        "confidence_threshold": confidence_threshold,
        "map": float(np.mean(average_precisions)),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": total_tp / max(total_tp + total_fp, 1),
        "micro_recall": total_tp / max(total_tp + total_fn, 1),
        "negative_images": negative_images,
        "negative_false_detections": negative_false_detections,
        "negative_false_positive_rate": negative_false_detections
        / max(negative_images, 1),
        "per_class": per_class,
    }


@torch.no_grad()
def collect_predictions(
    model: KitchenObjectNet,
    loader: DataLoader,
    device: torch.device,
    *,
    confidence_threshold: float = 0.03,
) -> tuple[list[dict[str, Tensor]], list[dict[str, Tensor]], float, list[float]]:
    model.eval()
    predictions: list[dict[str, Tensor]] = []
    targets_out: list[dict[str, Tensor]] = []
    losses = []
    latencies = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        outputs = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started) * 1000
        latencies.extend([elapsed / images.shape[0]] * images.shape[0])
        losses.append(float(detection_loss(outputs, targets)["total"].detach().cpu()))
        decoded = decode_detections(outputs, confidence_threshold=confidence_threshold)
        predictions.extend(
            [
                {key: value.detach().cpu() for key, value in prediction.items()}
                for prediction in decoded
            ]
        )
        targets_out.extend(
            [
                {"boxes": target["boxes"].cpu(), "labels": target["labels"].cpu()}
                for target in targets
            ]
        )
    return predictions, targets_out, float(np.mean(losses)), latencies


def choose_threshold(
    predictions: list[dict[str, Tensor]], targets: list[dict[str, Tensor]]
) -> tuple[float, list[dict[str, Any]]]:
    candidates = np.round(np.arange(0.10, 0.91, 0.05), 2)
    rows = []
    for candidate in candidates:
        metrics = calculate_detection_metrics(
            predictions, targets, confidence_threshold=float(candidate)
        )
        rows.append(
            {
                "threshold": float(candidate),
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "minimum_class_recall": min(
                    item["recall"] for item in metrics["per_class"].values()
                ),
            }
        )
    eligible = [row for row in rows if row["minimum_class_recall"] >= 0.90]
    pool = eligible or rows
    best = max(pool, key=lambda row: (row["macro_f1"], row["macro_precision"]))
    return float(best["threshold"]), rows


def _plot_training(history: list[dict[str, Any]], plot_dir: Path) -> None:
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["validation_loss"] for row in history], label="validation")
    axes[0].set_title("KitchenObjectNet loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(epochs, [row["validation_map50"] for row in history], color="#2b8a3e")
    axes[1].set_title("Validation mAP@0.50")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mAP")
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(plot_dir / "training_curves.png", dpi=180)
    plt.close(figure)


def _plot_thresholds(rows: list[dict[str, Any]], selected: float, plot_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    thresholds = [row["threshold"] for row in rows]
    for key, label in (
        ("macro_precision", "Precision"),
        ("macro_recall", "Recall"),
        ("macro_f1", "F1"),
    ):
        axis.plot(thresholds, [row[key] for row in rows], marker="o", label=label)
    axis.axvline(selected, color="black", linestyle="--", label=f"selected={selected:.2f}")
    axis.set_title("Validation threshold selection")
    axis.set_xlabel("Confidence threshold")
    axis.set_ylabel("Macro score")
    axis.set_ylim(0, 1.02)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_dir / "threshold_selection.png", dpi=180)
    plt.close(figure)


def _plot_per_class(metrics: dict[str, Any], plot_dir: Path) -> None:
    names = list(CLASS_NAMES)
    x = np.arange(len(names))
    width = 0.2
    figure, axis = plt.subplots(figsize=(10, 5))
    for offset, key, label in (
        (-1.5, "average_precision", "AP@0.50"),
        (-0.5, "precision", "Precision"),
        (0.5, "recall", "Recall"),
        (1.5, "f1", "F1"),
    ):
        axis.bar(
            x + offset * width,
            [metrics["per_class"][name][key] for name in names],
            width,
            label=label,
        )
    axis.set_xticks(x, names, rotation=20)
    axis.set_ylim(0, 1.05)
    axis.set_title("Held-out test metrics by object class")
    axis.set_ylabel("Score")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=4)
    figure.tight_layout()
    figure.savefig(plot_dir / "per_class_metrics.png", dpi=180)
    plt.close(figure)


def _difficulty_metrics(
    predictions: list[dict[str, Tensor]],
    targets: list[dict[str, Tensor]],
    records: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    report = {}
    for difficulty in ("easy", "medium", "hard"):
        indices = [
            index for index, record in enumerate(records) if record["difficulty"] == difficulty
        ]
        report[difficulty] = calculate_detection_metrics(
            [predictions[index] for index in indices],
            [targets[index] for index in indices],
            confidence_threshold=threshold,
        )
    return report


def _plot_difficulty(metrics: dict[str, Any], plot_dir: Path) -> None:
    names = ["easy", "medium", "hard"]
    x = np.arange(len(names))
    width = 0.22
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    for offset, key, label in (
        (-1, "map", "mAP@0.50"),
        (0, "macro_precision", "Precision"),
        (1, "macro_recall", "Recall"),
    ):
        axis.bar(x + offset * width, [metrics[name][key] for name in names], width, label=label)
    axis.set_xticks(x, names)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Score")
    axis.set_title("Held-out performance by generated difficulty")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_dir / "difficulty_metrics.png", dpi=180)
    plt.close(figure)


def _confusion_matrix(
    predictions: list[dict[str, Tensor]],
    targets: list[dict[str, Tensor]],
    threshold: float,
) -> np.ndarray:
    background = len(CLASS_NAMES)
    matrix = np.zeros((background + 1, background + 1), dtype=np.int64)
    for prediction, target in zip(predictions, targets):
        mask = prediction["scores"] >= threshold
        pred_boxes = prediction["boxes"][mask]
        pred_labels = prediction["labels"][mask]
        order = prediction["scores"][mask].argsort(descending=True)
        matched_ground_truth: set[int] = set()
        for pred_index in order:
            box = pred_boxes[pred_index]
            label = int(pred_labels[pred_index])
            if target["boxes"].numel():
                overlaps = box_iou(box.unsqueeze(0), target["boxes"]).squeeze(0)
                value, gt_index = overlaps.max(dim=0)
                gt_value = int(gt_index)
                if float(value) >= 0.5 and gt_value not in matched_ground_truth:
                    matched_ground_truth.add(gt_value)
                    matrix[int(target["labels"][gt_value]), label] += 1
                    continue
            matrix[background, label] += 1
        for gt_index, gt_label in enumerate(target["labels"]):
            if gt_index not in matched_ground_truth:
                matrix[int(gt_label), background] += 1
    return matrix


def _plot_confusion(matrix: np.ndarray, plot_dir: Path) -> None:
    labels = [*CLASS_NAMES, "background"]
    figure, axis = plt.subplots(figsize=(7.5, 6.5))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Ground truth")
    axis.set_title("Held-out test confusion matrix")
    maximum = max(int(matrix.max()), 1)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > maximum * 0.55 else "black",
                fontsize=8,
            )
    figure.tight_layout()
    figure.savefig(plot_dir / "confusion_matrix.png", dpi=180)
    plt.close(figure)


def _plot_dataset_distribution(dataset_root: Path, plot_dir: Path) -> None:
    records = load_records(dataset_root)
    class_counts = Counter(
        item["label"] for record in records for item in record["objects"]
    )
    split_counts = Counter(record["split"] for record in records)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(CLASS_NAMES, [class_counts[name] for name in CLASS_NAMES], color="#1971c2")
    axes[0].set_title("Object instances")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(axis="y", alpha=0.25)
    split_order = ["train", "validation", "test"]
    axes[1].bar(split_order, [split_counts[name] for name in split_order], color="#2f9e44")
    axes[1].set_title("Images by split")
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(plot_dir / "dataset_distribution.png", dpi=180)
    plt.close(figure)


def _plot_sample_predictions(
    dataset_root: Path,
    records: list[dict[str, Any]],
    predictions: list[dict[str, Tensor]],
    threshold: float,
    plot_dir: Path,
) -> None:
    count = min(12, len(records))
    indices = np.linspace(0, len(records) - 1, count, dtype=int)
    figure, axes = plt.subplots(3, 4, figsize=(14, 10))
    for axis, index in zip(axes.flatten(), indices):
        record = records[index]
        image = Image.open(dataset_root / record["image"]).convert("RGB")
        draw = ImageDraw.Draw(image)
        width, height = image.size
        for item in record["objects"]:
            x, y, box_width, box_height = item["bbox_xywh"]
            draw.rectangle((x, y, x + box_width, y + box_height), outline="#228be6", width=3)
        prediction = predictions[index]
        for box, score, label in zip(
            prediction["boxes"], prediction["scores"], prediction["labels"]
        ):
            if float(score) < threshold:
                continue
            x1, y1, x2, y2 = box.tolist()
            coordinates = (x1 * width, y1 * height, x2 * width, y2 * height)
            draw.rectangle(coordinates, outline="#2f9e44", width=3)
            draw.text(
                (coordinates[0], max(0, coordinates[1] - 12)),
                f"{CLASS_NAMES[int(label)]} {float(score):.2f}",
                fill="#2f9e44",
            )
        axis.imshow(image)
        axis.set_title(record["id"])
        axis.axis("off")
    figure.suptitle("Ground truth (blue) and KitchenObjectNet predictions (green)")
    figure.tight_layout()
    figure.savefig(plot_dir / "sample_predictions.png", dpi=170)
    plt.close(figure)


def export_onnx(model: KitchenObjectNet, destination: Path) -> dict[str, Any]:
    model = model.cpu().eval()
    generator = torch.Generator().manual_seed(20260808)
    example = torch.randn(
        (2, 3, INPUT_SIZE, INPUT_SIZE), dtype=torch.float32, generator=generator
    )
    with torch.no_grad():
        torch_outputs = model(example)
    torch.onnx.export(
        model,
        example,
        destination,
        input_names=["images"],
        output_names=["objectness", "classes", "sizes", "offsets"],
        dynamic_axes={
            "images": {0: "batch"},
            "objectness": {0: "batch"},
            "classes": {0: "batch"},
            "sizes": {0: "batch"},
            "offsets": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    onnx.checker.check_model(onnx.load(destination))
    session = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
    onnx_outputs = session.run(None, {"images": example.numpy()})
    maximum_difference = max(
        float(np.max(np.abs(expected.numpy() - actual)))
        for expected, actual in zip(torch_outputs, onnx_outputs)
    )
    if maximum_difference > 1e-4:
        raise RuntimeError(f"ONNX parity check failed: max difference {maximum_difference}")
    return {
        "path": destination.name,
        "sha256": _sha256(destination),
        "opset": 17,
        "maximum_output_difference": maximum_difference,
        "providers": session.get_providers(),
    }


def benchmark_onnx_cpu(
    model_path: Path, dataset: KitchenDetectionDataset, sample_count: int = 100
) -> dict[str, Any]:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    latencies = []
    count = min(sample_count, len(dataset))
    for index in range(count):
        image, _ = dataset[index]
        array = image.unsqueeze(0).numpy()
        started = time.perf_counter()
        session.run(None, {input_name: array})
        latencies.append((time.perf_counter() - started) * 1000.0)
    return {
        "provider": "CPUExecutionProvider",
        "sample_count": count,
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
    }


def train_kitchen_object_net(
    dataset_root: str | Path,
    run_root: str | Path,
    *,
    epochs: int = 120,
    batch_size: int = 16,
    seed: int = 20260808,
    patience: int = 15,
) -> dict[str, Any]:
    if epochs <= 0 or epochs > 120:
        raise ValueError("epochs must be between 1 and 120")
    dataset_root = Path(dataset_root).resolve()
    run_root = Path(run_root).resolve()
    plot_dir = run_root / "plots"
    checkpoint_dir = run_root / "checkpoints"
    plot_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_root / "training.log"
    metrics_log_path = run_root / "metrics.jsonl"
    metrics_log_path.write_text("", encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )
    logger = logging.getLogger("kitchen-object-net")
    started_at = datetime.now(timezone.utc)
    seed_everything(seed)
    audit = audit_dataset(dataset_root)
    (run_root / "dataset_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    if audit["status"] != "passed":
        raise RuntimeError("Dataset audit failed: " + "; ".join(audit["problems"][:10]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("training device=%s seed=%s", device, seed)
    train_dataset = KitchenDetectionDataset(dataset_root, "train", augment=True, seed=seed)
    validation_dataset = KitchenDetectionDataset(dataset_root, "validation", seed=seed)
    generator = torch.Generator().manual_seed(seed)
    worker_count = min(2, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=worker_count,
        collate_fn=collate_detection_batch,
        pin_memory=device.type == "cuda",
        generator=generator,
        persistent_workers=worker_count > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=worker_count,
        collate_fn=collate_detection_batch,
        pin_memory=device.type == "cuda",
        persistent_workers=worker_count > 0,
    )
    model = KitchenObjectNet()
    parameters = parameter_count(model)
    if parameters > 6_000_000:
        raise RuntimeError(f"Model exceeds six-million parameter budget: {parameters}")
    initial_fingerprint = initialization_fingerprint(model)
    initial_path = checkpoint_dir / "initial_random_state.pt"
    torch.save(model.state_dict(), initial_path)
    architecture = {
        "name": "KitchenObjectNet",
        "origin": "project-designed",
        "pretrained": False,
        "input": ["batch", 3, INPUT_SIZE, INPUT_SIZE],
        "backbone_feature_shapes": {
            "stage_2": ["batch", 96, 40, 40],
            "stage_3": ["batch", 160, 20, 20],
            "stage_4": ["batch", 256, 10, 10],
        },
        "fused_feature_shape": ["batch", 96, 40, 40],
        "heads": {
            "objectness": ["batch", 1, 40, 40],
            "classes": ["batch", len(CLASS_NAMES), 40, 40],
            "sizes": ["batch", 2, 40, 40],
            "offsets": ["batch", 2, 40, 40],
        },
        "trainable_parameters": parameters,
        "parameter_budget": 6_000_000,
        "loss": {
            "objectness": "balanced focal binary cross entropy",
            "classification": "cross entropy",
            "box": "5 * complete IoU",
            "offset": "smooth L1",
        },
        "decoding": "centre cell plus sigmoid offset/size; class-wise NMS",
    }
    (run_root / "architecture.json").write_text(
        json.dumps(architecture, indent=2), encoding="utf-8"
    )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    warmup_epochs = min(5, epochs)

    def learning_rate_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_lambda)
    best_map = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    best_path = checkpoint_dir / "best.pt"
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        epoch_components: dict[str, list[float]] = {
            "objectness": [],
            "classification": [],
            "box": [],
            "offset": [],
        }
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            # FP16 gradient scaling repeatedly underflows on the CIoU/aspect
            # branch of this small custom detector.  Full precision is used on
            # both CPU and CUDA so non-finite values cannot be silently skipped.
            losses = detection_loss(model(images), targets)
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError(
                    "Non-finite training loss: "
                    + json.dumps({name: float(value) for name, value in losses.items()})
                )
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=10.0, error_if_nonfinite=True
            )
            optimizer.step()
            epoch_losses.append(float(losses["total"].detach().cpu()))
            for name in epoch_components:
                epoch_components[name].append(float(losses[name].detach().cpu()))
        validation_predictions, validation_targets, validation_loss, _ = collect_predictions(
            model, validation_loader, device
        )
        validation_metrics = calculate_detection_metrics(
            validation_predictions,
            validation_targets,
            confidence_threshold=0.25,
        )
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": float(np.mean(epoch_losses)),
            "validation_loss": validation_loss,
            "validation_map50": validation_metrics["map"],
            "validation_precision": validation_metrics["macro_precision"],
            "validation_recall": validation_metrics["macro_recall"],
            "validation_f1": validation_metrics["macro_f1"],
            **{
                f"train_{name}_loss": float(np.mean(values))
                for name, values in epoch_components.items()
            },
        }
        history.append(row)
        with metrics_log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
        logger.info(
            "epoch=%03d train_loss=%.4f val_loss=%.4f map50=%.4f precision=%.4f recall=%.4f",
            epoch,
            row["train_loss"],
            validation_loss,
            validation_metrics["map"],
            validation_metrics["macro_precision"],
            validation_metrics["macro_recall"],
        )
        if validation_metrics["map"] > best_map + 1e-5:
            best_map = validation_metrics["map"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "validation_map50": best_map,
                    "seed": seed,
                    "initialization_fingerprint": initial_fingerprint,
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1
        scheduler.step()
        if epochs_without_improvement >= patience:
            logger.info("early stopping after epoch %s", epoch)
            break
    torch.save(model.state_dict(), checkpoint_dir / "last.pt")
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    validation_predictions, validation_targets, validation_loss, validation_latencies = collect_predictions(
        model, validation_loader, device
    )
    threshold, threshold_rows = choose_threshold(validation_predictions, validation_targets)
    logger.info(
        "model and threshold frozen at epoch=%s threshold=%.2f; opening held-out test split",
        best_epoch,
        threshold,
    )
    test_dataset = KitchenDetectionDataset(dataset_root, "test", seed=seed)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=worker_count,
        collate_fn=collate_detection_batch,
        pin_memory=device.type == "cuda",
        persistent_workers=worker_count > 0,
    )
    test_predictions, test_targets, test_loss, test_latencies = collect_predictions(
        model, test_loader, device
    )
    test_metrics_50 = calculate_detection_metrics(
        test_predictions, test_targets, confidence_threshold=threshold, iou_threshold=0.5
    )
    test_metrics_75 = calculate_detection_metrics(
        test_predictions, test_targets, confidence_threshold=threshold, iou_threshold=0.75
    )
    onnx_path = run_root / "KitchenObjectNet.onnx"
    onnx_report = export_onnx(model, onnx_path)
    onnx_cpu_latency = benchmark_onnx_cpu(onnx_path, test_dataset)
    class_map_path = run_root / "class_map.json"
    class_map_path.write_text(
        json.dumps({str(index): name for index, name in enumerate(CLASS_NAMES)}, indent=2),
        encoding="utf-8",
    )
    config = {
        "model": "KitchenObjectNet",
        "pretrained": False,
        "classes": list(CLASS_NAMES),
        "input_size": INPUT_SIZE,
        "parameter_count": parameters,
        "seed": seed,
        "batch_size": batch_size,
        "maximum_epochs": epochs,
        "early_stopping_patience": patience,
        "optimizer": "AdamW",
        "initial_learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "warmup_epochs": warmup_epochs,
        "schedule": "cosine",
        "initialization": (
            "seeded Kaiming normal backbone/fusion; seeded normal(std=0.001) "
            "prediction heads; constant objectness prior bias"
        ),
        "initialization_fingerprint": initial_fingerprint,
    }
    (run_root / "training_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    test_records = [record for record in load_records(dataset_root) if record["split"] == "test"]
    difficulty_metrics = _difficulty_metrics(
        test_predictions, test_targets, test_records, threshold
    )
    _plot_training(history, plot_dir)
    _plot_thresholds(threshold_rows, threshold, plot_dir)
    _plot_per_class(test_metrics_50, plot_dir)
    _plot_difficulty(difficulty_metrics, plot_dir)
    confusion = _confusion_matrix(test_predictions, test_targets, threshold)
    _plot_confusion(confusion, plot_dir)
    _plot_dataset_distribution(dataset_root, plot_dir)
    _plot_sample_predictions(
        dataset_root, test_records, test_predictions, threshold, plot_dir
    )
    completed_at = datetime.now(timezone.utc)
    summary = {
        "status": "completed",
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "duration_seconds": (completed_at - started_at).total_seconds(),
        "device": str(device),
        "cuda_device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "cuda_version": torch.version.cuda,
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "onnxruntime": ort.__version__,
        "best_epoch": best_epoch,
        "best_validation_map50": best_map,
        "selected_confidence_threshold": threshold,
        "validation_loss": validation_loss,
        "test_loss": test_loss,
        "test_map50": test_metrics_50,
        "test_map75": test_metrics_75,
        "difficulty_metrics": difficulty_metrics,
        "validation_latency_ms": {
            "median": float(np.median(validation_latencies)),
            "p95": float(np.percentile(validation_latencies, 95)),
        },
        "test_latency_ms": {
            "median": float(np.median(test_latencies)),
            "p95": float(np.percentile(test_latencies, 95)),
        },
        "onnx": onnx_report,
        "onnx_cpu_latency": onnx_cpu_latency,
        "acceptance": {
            "map50_at_least_0_90": test_metrics_50["map"] >= 0.90,
            "per_class_precision_recall_at_least_0_85": all(
                item["precision"] >= 0.85 and item["recall"] >= 0.85
                for item in test_metrics_50["per_class"].values()
            ),
            "negative_false_positive_rate_at_most_0_05": test_metrics_50[
                "negative_false_positive_rate"
            ]
            <= 0.05,
        },
        "artifacts": {
            "training_log": log_path.relative_to(run_root).as_posix(),
            "metrics_log": metrics_log_path.relative_to(run_root).as_posix(),
            "plots": sorted(path.relative_to(run_root).as_posix() for path in plot_dir.glob("*.png")),
            "best_checkpoint": best_path.relative_to(run_root).as_posix(),
            "onnx_model": onnx_path.relative_to(run_root).as_posix(),
        },
    }
    (run_root / "evaluation.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    provenance = {
        "model": "KitchenObjectNet",
        "pretrained_weights_used": False,
        "imported_model_architecture_used": False,
        "initial_checkpoint_sha256": _sha256(initial_path),
        "initialization_fingerprint": initial_fingerprint,
        "best_checkpoint_sha256": _sha256(best_path),
        "onnx_sha256": onnx_report["sha256"],
        "dataset_annotation_sha256": _sha256(dataset_root / "annotations.jsonl"),
        "seed": seed,
    }
    (run_root / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    model_card = f"""# KitchenObjectNet model card

- Architecture: project-designed anchor-free convolutional detector
- Pretrained weights: none
- Training data: {audit['record_count']} RGB images generated in this Webots kitchen
- Classes: {', '.join(CLASS_NAMES)}
- Parameters: {parameters:,}
- Random seed: {seed}
- Best epoch: {best_epoch}
- Held-out mAP@0.50: {test_metrics_50['map']:.4f}
- Held-out mAP@0.75: {test_metrics_75['map']:.4f}
- Held-out macro precision / recall: {test_metrics_50['macro_precision']:.4f} / {test_metrics_50['macro_recall']:.4f}
- Negative test images / false detections: {test_metrics_50['negative_images']} / {test_metrics_50['negative_false_detections']}
- Confidence threshold: {threshold:.2f}
- ONNX SHA-256: `{onnx_report['sha256']}`

## Intended use and limitations

The model supplies visual evidence only for the `find` action in this
constrained simulated kitchen. It cannot plan, move the robot, or bypass the
deterministic validator. Its synthetic-domain results do not demonstrate
performance on real photographs, unfamiliar kitchens, or physical robots.

## Data and licence provenance

Every weight was trained from the recorded random initialisation using only
the project-generated Webots images. No internet dataset, pretrained weight,
imported detector, or downloaded checkpoint was used. Webots assets remain
under the applicable Cyberbotics Webots asset licence; project code and learned
weights are supplied for academic use under the repository metadata.
"""
    (run_root / "MODEL_CARD.md").write_text(model_card, encoding="utf-8")
    logger.info("training complete best_epoch=%s test_map50=%.4f", best_epoch, test_metrics_50["map"])
    return summary
