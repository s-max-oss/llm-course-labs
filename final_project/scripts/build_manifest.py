#!/usr/bin/env python3
"""Build a deterministic manifest without copying source data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


NAME_RE = re.compile(
    r"^seed(?P<seed>\d+)_(?P<view>\d+)_R(?P<frame>\d+)\.(?:jpg|jpeg|png)$",
    re.IGNORECASE,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_image_name(name: str) -> dict[str, Any]:
    match = NAME_RE.match(name)
    if not match:
        raise ValueError(f"unsupported image filename: {name}")
    return {
        "scene_seed": int(match.group("seed")),
        "view_index": int(match.group("view")),
        "frame_index": int(match.group("frame")),
    }


def parse_yolo_line(line: str, line_number: int) -> tuple[int, list[float]]:
    fields = line.split()
    if len(fields) != 5:
        raise ValueError(f"line {line_number}: expected 5 fields, got {len(fields)}")
    class_id = int(fields[0])
    bbox = [float(value) for value in fields[1:]]
    if any(value < 0.0 or value > 1.0 for value in bbox):
        raise ValueError(f"line {line_number}: normalized bbox is outside [0, 1]")
    if bbox[2] <= 0.0 or bbox[3] <= 0.0:
        raise ValueError(f"line {line_number}: bbox width and height must be positive")
    return class_id, bbox


def dhash64(image: Image.Image) -> str:
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = gray.tobytes()
    bits = 0
    for row in range(8):
        for column in range(8):
            left = pixels[row * 9 + column]
            right = pixels[row * 9 + column + 1]
            bits = (bits << 1) | int(left > right)
    return f"{bits:016x}"


def yolo_to_pixels(bbox: list[float], width: int, height: int) -> list[float]:
    center_x, center_y, box_width, box_height = bbox
    x1 = max(0.0, (center_x - box_width / 2.0) * width)
    y1 = max(0.0, (center_y - box_height / 2.0) * height)
    x2 = min(float(width), (center_x + box_width / 2.0) * width)
    y2 = min(float(height), (center_y + box_height / 2.0) * height)
    return [round(value, 3) for value in (x1, y1, x2, y2)]


def load_boxes(
    label_path: Path, class_map: dict[str, dict[str, str]], width: int, height: int
) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        class_id, bbox = parse_yolo_line(line, line_number)
        mapping = class_map.get(str(class_id))
        if mapping is None:
            raise ValueError(f"{label_path}: unknown class id {class_id}")
        center_x, center_y, box_width, box_height = bbox
        boxes.append(
            {
                "class_id": class_id,
                "detector_class": mapping["detector_class"],
                "semantic_label": mapping["semantic_label"],
                "bbox_yolo_cxcywh": bbox,
                "bbox_xyxy_pixels": yolo_to_pixels(bbox, width, height),
                "area_fraction": round(box_width * box_height, 8),
                "touches_image_edge": bool(
                    center_x - box_width / 2.0 <= 0.0
                    or center_y - box_height / 2.0 <= 0.0
                    or center_x + box_width / 2.0 >= 1.0
                    or center_y + box_height / 2.0 >= 1.0
                ),
            }
        )
    return boxes


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def build(config_path: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    image_root = Path(config["image_root"])
    label_root = Path(config["label_root"])
    class_map = config["class_map"]
    if not image_root.is_dir():
        raise FileNotFoundError(f"image root does not exist: {image_root}")
    if not label_root.is_dir():
        raise FileNotFoundError(f"label root does not exist: {label_root}")

    image_paths = sorted(
        path
        for path in image_root.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    records: list[dict[str, Any]] = []
    for image_path in image_paths:
        parsed = parse_image_name(image_path.name)
        label_path = label_root / f"{image_path.stem}.txt"
        with Image.open(image_path) as image:
            image.load()
            width, height = image.size
            perceptual_hash = dhash64(image)
        boxes = load_boxes(label_path, class_map, width, height) if label_path.exists() else []
        missing_as_background = bool(config.get("missing_labels_are_background", False))
        if label_path.exists():
            label_status = "label_file_present"
            record_labels = sorted({box["semantic_label"] for box in boxes})
            if not record_labels:
                record_labels = ["background_or_unknown"]
        else:
            label_status = "missing_label_file"
            record_labels = ["background_or_unknown"] if missing_as_background else ["unknown"]
        record = {
            "schema_version": 1,
            "dataset_id": config["dataset_id"],
            "sample_id": image_path.stem,
            "image_root_id": config["image_root_id"],
            "image": image_path.name,
            "scene_seed": parsed["scene_seed"],
            "room_id": None,
            "frame_id": f"R{parsed['frame_index']:04d}",
            "view_id": f"view_{parsed['view_index']:03d}",
            "width": width,
            "height": height,
            "image_sha256": sha256_file(image_path),
            "dhash64": perceptual_hash,
            "label_file": label_path.name if label_path.exists() else None,
            "label_sha256": sha256_file(label_path) if label_path.exists() else None,
            "label_status": label_status,
            "label_source": config.get("label_source", "unknown"),
            "record_labels": record_labels,
            "boxes": boxes,
            "split": "unassigned",
        }
        records.append(record)

    by_sha: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_sha[record["image_sha256"]].append(record)
    duplicate_groups = [group for group in by_sha.values() if len(group) > 1]
    for group_index, group in enumerate(
        sorted(duplicate_groups, key=lambda items: items[0]["image_sha256"]), start=1
    ):
        group_id = f"exact_{group_index:03d}"
        for record in group:
            record["exact_duplicate_group"] = group_id
            record["exact_duplicate_group_size"] = len(group)
    for record in records:
        record.setdefault("exact_duplicate_group", None)
        record.setdefault("exact_duplicate_group_size", 1)

    threshold = int(config.get("near_duplicate_hamming_threshold", 4))
    near_pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(records):
        left_hash = int(left["dhash64"], 16)
        for right in records[left_index + 1 :]:
            if left["image_sha256"] == right["image_sha256"]:
                continue
            distance = (left_hash ^ int(right["dhash64"], 16)).bit_count()
            if distance <= threshold:
                near_pairs.append(
                    {
                        "left": left["sample_id"],
                        "right": right["sample_id"],
                        "hamming_distance": distance,
                        "cross_seed": left["scene_seed"] != right["scene_seed"],
                    }
                )

    class_counts: Counter[str] = Counter()
    for record in records:
        class_counts.update(box["semantic_label"] for box in record["boxes"])
    cross_seed_exact = sum(
        1 for group in duplicate_groups if len({record["scene_seed"] for record in group}) > 1
    )
    summary = {
        "schema_version": 1,
        "dataset_id": config["dataset_id"],
        "image_count": len(records),
        "scene_seeds": sorted({record["scene_seed"] for record in records}),
        "scene_seed_count": len({record["scene_seed"] for record in records}),
        "label_file_count": sum(record["label_file"] is not None for record in records),
        "missing_label_file_count": sum(record["label_file"] is None for record in records),
        "box_count": sum(len(record["boxes"]) for record in records),
        "box_class_counts": dict(sorted(class_counts.items())),
        "exact_duplicate_group_count": len(duplicate_groups),
        "images_in_exact_duplicate_groups": sum(len(group) for group in duplicate_groups),
        "cross_seed_exact_duplicate_group_count": cross_seed_exact,
        "near_duplicate_hamming_threshold": threshold,
        "near_duplicate_candidate_pair_count": len(near_pairs),
        "cross_seed_near_duplicate_candidate_pair_count": sum(
            pair["cross_seed"] for pair in near_pairs
        ),
        "split_status": "unassigned",
        "formal_experiment_ready": False,
        "formal_experiment_blockers": [
            "only five independent scene seeds",
            "exact and perceptual duplicate candidates require split-aware handling",
            "room identifiers and full collection provenance are unavailable",
        ],
    }

    manifest_path = output_dir / "batch_500.jsonl"
    summary_path = output_dir / "batch_500.summary.json"
    near_path = output_dir / "batch_500.near_duplicates.json"
    atomic_write(
        manifest_path,
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
    )
    atomic_write(summary_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    atomic_write(
        near_path,
        json.dumps(
            {
                "schema_version": 1,
                "method": "64-bit difference hash; candidates require human review",
                "hamming_threshold": threshold,
                "pairs": near_pairs,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )
    return manifest_path, summary_path, near_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "manifests",
    )
    args = parser.parse_args()
    outputs = build(args.config.resolve(), args.output_dir.resolve())
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
