"""Pure conversion helpers shared by heavyweight model runners."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .contracts import BaselineContractError, canonical_detector_label


def _array(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([])
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _raw_class_name(
    detections: Any,
    class_id: int,
    index: int,
    count: int,
    configured_names: Mapping[str, str],
) -> str:
    data = getattr(detections, "data", None)
    names = data.get("class_name") if isinstance(data, Mapping) else None
    if names is not None:
        values = _array(names)
        if values.ndim == 1 and values.size == count:
            return str(values[index])
    return str(configured_names.get(str(class_id), class_id))


def _mask_metrics(mask: Any, bbox: list[float], width: int, height: int) -> tuple[float | None, float | None]:
    if mask is None:
        return None, None
    values = _array(mask).astype(bool)
    if values.ndim != 2 or values.shape != (height, width):
        raise BaselineContractError(
            f"mask shape {values.shape} does not match image {(height, width)}"
        )
    x1, y1, x2, y2 = bbox
    left = max(0, min(width, math.floor(x1)))
    top = max(0, min(height, math.floor(y1)))
    right = max(0, min(width, math.ceil(x2)))
    bottom = max(0, min(height, math.ceil(y2)))
    roi = values[top:bottom, left:right]
    coverage = float(np.count_nonzero(roi) / roi.size) if roi.size else 0.0
    area_fraction = float(np.count_nonzero(values) / values.size)
    return round(coverage, 8), round(area_fraction, 8)


def convert_rfdetr_detections(
    detections: Any,
    width: int,
    height: int,
    class_names: Mapping[str, str],
    aliases: dict[str, str],
) -> list[dict[str, Any]]:
    xyxy = _array(getattr(detections, "xyxy", None))
    if xyxy.size == 0:
        return []
    if xyxy.ndim != 2 or xyxy.shape[1] != 4:
        raise BaselineContractError(f"unexpected RF-DETR box shape: {xyxy.shape}")
    confidences = _array(getattr(detections, "confidence", None)).reshape(-1)
    class_ids = _array(getattr(detections, "class_id", None)).reshape(-1)
    if len(confidences) != len(xyxy) or len(class_ids) != len(xyxy):
        raise BaselineContractError("RF-DETR boxes, scores, and classes have different lengths")
    masks = getattr(detections, "mask", None)
    mask_values = None if masks is None else _array(masks)
    if mask_values is not None and len(mask_values) != len(xyxy):
        raise BaselineContractError("RF-DETR boxes and masks have different lengths")

    converted: list[dict[str, Any]] = []
    for index, box_values in enumerate(xyxy):
        class_id = int(class_ids[index])
        confidence = float(confidences[index])
        bbox = [float(value) for value in box_values]
        if not math.isfinite(confidence) or any(not math.isfinite(value) for value in bbox):
            raise BaselineContractError("RF-DETR prediction contains a non-finite value")
        bbox = [
            max(0.0, min(float(width), bbox[0])),
            max(0.0, min(float(height), bbox[1])),
            max(0.0, min(float(width), bbox[2])),
            max(0.0, min(float(height), bbox[3])),
        ]
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise BaselineContractError(f"RF-DETR returned an invalid box: {bbox}")
        raw_name = _raw_class_name(detections, class_id, index, len(xyxy), class_names)
        canonical = canonical_detector_label(raw_name, aliases)
        mask = None if mask_values is None else mask_values[index]
        mask_coverage, mask_area_fraction = _mask_metrics(mask, bbox, width, height)
        box_area_fraction = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / (width * height)
        converted.append(
            {
                "prediction_index": index,
                "class_id": class_id,
                "raw_class_name": raw_name,
                "canonical_label": canonical,
                "confidence": round(confidence, 8),
                "bbox_xyxy_pixels": [round(value, 4) for value in bbox],
                "bbox_area_fraction": round(float(box_area_fraction), 8),
                "mask_coverage_in_bbox": mask_coverage,
                "mask_area_fraction": mask_area_fraction,
            }
        )
    ordered = sorted(
        converted,
        key=lambda item: (
            -item["confidence"],
            item["class_id"],
            item["bbox_xyxy_pixels"],
        ),
    )
    for prediction_index, item in enumerate(ordered):
        item["prediction_index"] = prediction_index
    return ordered
