"""Project-designed CNN detector trained from scratch on Webots images.

KitchenObjectNet intentionally uses only basic PyTorch layers.  It does not
load or wrap any third-party vision model and has no pretrained-weight pathway.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .cnn_constants import CLASS_NAMES, CLASS_TO_INDEX, INPUT_SIZE, OUTPUT_STRIDE


class ConvBlock(nn.Module):
    """Convolution, batch normalisation and SiLU activation."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


class KitchenObjectNet(nn.Module):
    """Compact anchor-free, centre-heatmap detector for kitchen objects."""

    def __init__(self, num_classes: int = len(CLASS_NAMES)) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self.num_classes = num_classes
        self.stem = ConvBlock(3, 32, stride=2)  # 160 x 160
        self.stage_1 = nn.Sequential(
            ConvBlock(32, 48, stride=2),  # 80 x 80
            ConvBlock(48, 48),
        )
        self.stage_2 = nn.Sequential(
            ConvBlock(48, 96, stride=2),  # 40 x 40
            ConvBlock(96, 96),
        )
        self.stage_3 = nn.Sequential(
            ConvBlock(96, 160, stride=2),  # 20 x 20
            ConvBlock(160, 160),
        )
        self.stage_4 = nn.Sequential(
            ConvBlock(160, 256, stride=2),  # 10 x 10
            ConvBlock(256, 256),
        )

        pyramid_channels = 96
        self.lateral_40 = nn.Conv2d(96, pyramid_channels, 1)
        self.lateral_20 = nn.Conv2d(160, pyramid_channels, 1)
        self.lateral_10 = nn.Conv2d(256, pyramid_channels, 1)
        self.refine_20 = ConvBlock(pyramid_channels, pyramid_channels)
        self.refine_40 = ConvBlock(pyramid_channels, pyramid_channels)
        self.head = nn.Sequential(
            ConvBlock(pyramid_channels, 128),
            ConvBlock(128, 96),
        )
        self.objectness_head = nn.Conv2d(96, 1, 1)
        self.class_head = nn.Conv2d(96, num_classes, 1)
        self.size_head = nn.Conv2d(96, 2, 1)
        self.offset_head = nn.Conv2d(96, 2, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialise locally; no checkpoint or external weights are accepted."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        # Detection heads start close to an uninformative prediction while the
        # convolutional backbone retains Kaiming initialization.  Kaiming on a
        # one-channel heatmap head creates extreme initial logits and unstable
        # hard-negative gradients.
        for head in (
            self.objectness_head,
            self.class_head,
            self.size_head,
            self.offset_head,
        ):
            nn.init.normal_(head.weight, mean=0.0, std=0.001)
            if head.bias is not None:
                nn.init.zeros_(head.bias)
        nn.init.constant_(self.objectness_head.bias, -2.19)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        x = self.stem(inputs)
        x = self.stage_1(x)
        feature_40 = self.stage_2(x)
        feature_20 = self.stage_3(feature_40)
        feature_10 = self.stage_4(feature_20)

        pyramid_10 = self.lateral_10(feature_10)
        pyramid_20 = self.refine_20(
            self.lateral_20(feature_20)
            + F.interpolate(pyramid_10, size=feature_20.shape[-2:], mode="nearest")
        )
        pyramid_40 = self.refine_40(
            self.lateral_40(feature_40)
            + F.interpolate(pyramid_20, size=feature_40.shape[-2:], mode="nearest")
        )
        features = self.head(pyramid_40)
        return (
            self.objectness_head(features),
            self.class_head(features),
            self.size_head(features),
            self.offset_head(features),
        )


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def initialization_fingerprint(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@dataclass(slots=True)
class DetectionTargets:
    objectness: Tensor
    classes: Tensor
    sizes: Tensor
    offsets: Tensor
    positive: Tensor


def build_targets(
    targets: list[dict[str, Tensor]],
    grid_height: int,
    grid_width: int,
    device: torch.device,
) -> DetectionTargets:
    batch_size = len(targets)
    objectness = torch.zeros((batch_size, 1, grid_height, grid_width), device=device)
    classes = torch.full(
        (batch_size, grid_height, grid_width), -1, dtype=torch.long, device=device
    )
    sizes = torch.zeros((batch_size, 2, grid_height, grid_width), device=device)
    offsets = torch.zeros((batch_size, 2, grid_height, grid_width), device=device)
    positive = torch.zeros(
        (batch_size, grid_height, grid_width), dtype=torch.bool, device=device
    )
    for batch_index, target in enumerate(targets):
        boxes = target["boxes"].to(device)
        labels = target["labels"].to(device)
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box.clamp(0.0, 1.0)
            centre_x = ((x1 + x2) * 0.5 * grid_width).clamp(0, grid_width - 1e-4)
            centre_y = ((y1 + y2) * 0.5 * grid_height).clamp(0, grid_height - 1e-4)
            cell_x = int(torch.floor(centre_x).item())
            cell_y = int(torch.floor(centre_y).item())
            objectness[batch_index, 0, cell_y, cell_x] = 1.0
            classes[batch_index, cell_y, cell_x] = label
            sizes[batch_index, :, cell_y, cell_x] = torch.stack(
                ((x2 - x1).clamp_min(1e-4), (y2 - y1).clamp_min(1e-4))
            )
            offsets[batch_index, :, cell_y, cell_x] = torch.stack(
                (centre_x - cell_x, centre_y - cell_y)
            )
            positive[batch_index, cell_y, cell_x] = True
    return DetectionTargets(objectness, classes, sizes, offsets, positive)


def _focal_binary_loss(logits: Tensor, targets: Tensor) -> Tensor:
    probabilities = torch.sigmoid(logits)
    cross_entropy = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probability_of_target = probabilities * targets + (1 - probabilities) * (1 - targets)
    alpha = 0.75 * targets + 0.25 * (1 - targets)
    loss = alpha * (1 - probability_of_target).pow(2.0) * cross_entropy
    positive = targets > 0.5
    negative = ~positive
    normalizer = targets.sum().clamp_min(float(targets.shape[0]))
    positive_loss = loss[positive].sum() / normalizer
    negative_loss = loss[negative].sum() / normalizer
    # Positive-count normalization keeps hard false-positive cells relevant
    # without allowing the full background grid to swamp genuine objects.
    return positive_loss + negative_loss


def _ciou_loss(predicted: Tensor, expected: Tensor) -> Tensor:
    pred_x1 = predicted[:, 0] - predicted[:, 2] * 0.5
    pred_y1 = predicted[:, 1] - predicted[:, 3] * 0.5
    pred_x2 = predicted[:, 0] + predicted[:, 2] * 0.5
    pred_y2 = predicted[:, 1] + predicted[:, 3] * 0.5
    true_x1 = expected[:, 0] - expected[:, 2] * 0.5
    true_y1 = expected[:, 1] - expected[:, 3] * 0.5
    true_x2 = expected[:, 0] + expected[:, 2] * 0.5
    true_y2 = expected[:, 1] + expected[:, 3] * 0.5
    intersection = (
        (torch.minimum(pred_x2, true_x2) - torch.maximum(pred_x1, true_x1)).clamp_min(0)
        * (torch.minimum(pred_y2, true_y2) - torch.maximum(pred_y1, true_y1)).clamp_min(0)
    )
    pred_area = (pred_x2 - pred_x1).clamp_min(0) * (pred_y2 - pred_y1).clamp_min(0)
    true_area = (true_x2 - true_x1).clamp_min(0) * (true_y2 - true_y1).clamp_min(0)
    iou = intersection / (pred_area + true_area - intersection).clamp_min(1e-7)
    centre_distance = (predicted[:, :2] - expected[:, :2]).pow(2).sum(dim=1)
    enclosing_width = torch.maximum(pred_x2, true_x2) - torch.minimum(pred_x1, true_x1)
    enclosing_height = torch.maximum(pred_y2, true_y2) - torch.minimum(pred_y1, true_y1)
    enclosing_diagonal = enclosing_width.pow(2) + enclosing_height.pow(2) + 1e-7
    aspect = (4 / math.pi**2) * (
        torch.atan(expected[:, 2] / expected[:, 3].clamp_min(1e-7))
        - torch.atan(predicted[:, 2] / predicted[:, 3].clamp_min(1e-7))
    ).pow(2)
    with torch.no_grad():
        weight = aspect / (1 - iou + aspect).clamp_min(1e-7)
    return (1 - iou + centre_distance / enclosing_diagonal + weight * aspect).mean()


def detection_loss(
    outputs: tuple[Tensor, Tensor, Tensor, Tensor],
    targets: list[dict[str, Tensor]],
) -> dict[str, Tensor]:
    objectness_logits, class_logits, size_logits, offset_logits = outputs
    built = build_targets(
        targets,
        objectness_logits.shape[-2],
        objectness_logits.shape[-1],
        objectness_logits.device,
    )
    objectness_loss = _focal_binary_loss(objectness_logits, built.objectness)
    positive = built.positive
    if positive.any():
        class_values = class_logits.permute(0, 2, 3, 1)[positive]
        class_loss = F.cross_entropy(class_values, built.classes[positive])
        predicted_sizes = torch.sigmoid(size_logits).permute(0, 2, 3, 1)[positive]
        predicted_offsets = torch.sigmoid(offset_logits).permute(0, 2, 3, 1)[positive]
        expected_sizes = built.sizes.permute(0, 2, 3, 1)[positive]
        expected_offsets = built.offsets.permute(0, 2, 3, 1)[positive]
        indices = positive.nonzero(as_tuple=False)
        grid_height, grid_width = positive.shape[-2:]
        pred_centres = torch.stack(
            (
                (indices[:, 2] + predicted_offsets[:, 0]) / grid_width,
                (indices[:, 1] + predicted_offsets[:, 1]) / grid_height,
            ),
            dim=1,
        )
        true_centres = torch.stack(
            (
                (indices[:, 2] + expected_offsets[:, 0]) / grid_width,
                (indices[:, 1] + expected_offsets[:, 1]) / grid_height,
            ),
            dim=1,
        )
        box_loss = _ciou_loss(
            torch.cat((pred_centres, predicted_sizes), dim=1),
            torch.cat((true_centres, expected_sizes), dim=1),
        )
        offset_loss = F.smooth_l1_loss(predicted_offsets, expected_offsets)
    else:
        zero = objectness_logits.sum() * 0.0
        class_loss = zero
        box_loss = zero
        offset_loss = zero
    total = objectness_loss + class_loss + 5.0 * box_loss + offset_loss
    return {
        "total": total,
        "objectness": objectness_loss,
        "classification": class_loss,
        "box": box_loss,
        "offset": offset_loss,
    }


def box_iou(left: Tensor, right: Tensor) -> Tensor:
    if left.numel() == 0 or right.numel() == 0:
        return torch.zeros((left.shape[0], right.shape[0]), device=left.device)
    top_left = torch.maximum(left[:, None, :2], right[None, :, :2])
    bottom_right = torch.minimum(left[:, None, 2:], right[None, :, 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(dim=2)
    left_area = (left[:, 2:] - left[:, :2]).clamp_min(0).prod(dim=1)
    right_area = (right[:, 2:] - right[:, :2]).clamp_min(0).prod(dim=1)
    return intersection / (
        left_area[:, None] + right_area[None, :] - intersection
    ).clamp_min(1e-7)


def non_maximum_suppression(boxes: Tensor, scores: Tensor, threshold: float) -> Tensor:
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    kept: list[Tensor] = []
    while order.numel():
        current = order[0]
        kept.append(current)
        if order.numel() == 1:
            break
        remaining = order[1:]
        overlaps = box_iou(boxes[current].unsqueeze(0), boxes[remaining]).squeeze(0)
        order = remaining[overlaps <= threshold]
    return torch.stack(kept)


@torch.no_grad()
def decode_detections(
    outputs: tuple[Tensor, Tensor, Tensor, Tensor],
    confidence_threshold: float = 0.25,
    nms_threshold: float = 0.45,
    top_k: int = 30,
) -> list[dict[str, Tensor]]:
    objectness_logits, class_logits, size_logits, offset_logits = outputs
    objectness = torch.sigmoid(objectness_logits)
    local_maximum = F.max_pool2d(objectness, kernel_size=3, stride=1, padding=1)
    objectness = objectness * (objectness >= local_maximum).to(objectness.dtype)
    class_probabilities = torch.softmax(class_logits, dim=1)
    batch_results = []
    grid_height, grid_width = objectness.shape[-2:]
    for batch_index in range(objectness.shape[0]):
        combined = objectness[batch_index] * class_probabilities[batch_index]
        flat = combined.reshape(-1)
        count = min(top_k, flat.numel())
        scores, indices = torch.topk(flat, count)
        selected = scores >= confidence_threshold
        scores = scores[selected]
        indices = indices[selected]
        labels = indices // (grid_height * grid_width)
        cells = indices % (grid_height * grid_width)
        cell_y = cells // grid_width
        cell_x = cells % grid_width
        if scores.numel() == 0:
            batch_results.append(
                {
                    "boxes": torch.empty((0, 4), device=flat.device),
                    "scores": scores,
                    "labels": labels,
                }
            )
            continue
        sizes = torch.sigmoid(size_logits[batch_index, :, cell_y, cell_x]).T
        offsets = torch.sigmoid(offset_logits[batch_index, :, cell_y, cell_x]).T
        centres = torch.stack(
            (
                (cell_x + offsets[:, 0]) / grid_width,
                (cell_y + offsets[:, 1]) / grid_height,
            ),
            dim=1,
        )
        boxes = torch.cat((centres - sizes * 0.5, centres + sizes * 0.5), dim=1).clamp(0, 1)
        kept_indices = []
        for label in labels.unique():
            class_indices = (labels == label).nonzero(as_tuple=False).flatten()
            class_kept = non_maximum_suppression(
                boxes[class_indices], scores[class_indices], nms_threshold
            )
            kept_indices.append(class_indices[class_kept])
        kept = torch.cat(kept_indices)
        kept = kept[scores[kept].argsort(descending=True)]
        batch_results.append(
            {"boxes": boxes[kept], "scores": scores[kept], "labels": labels[kept]}
        )
    return batch_results


def targets_from_records(
    boxes: Iterable[Iterable[float]], labels: Iterable[int], device: str = "cpu"
) -> dict[str, Tensor]:
    box_tensor = torch.tensor(list(boxes), dtype=torch.float32, device=device).reshape(-1, 4)
    label_tensor = torch.tensor(list(labels), dtype=torch.long, device=device)
    return {"boxes": box_tensor, "labels": label_tensor}
