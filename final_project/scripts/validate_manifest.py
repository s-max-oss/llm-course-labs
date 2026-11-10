#!/usr/bin/env python3
"""Validate manifest structure, source hashes, and split leakage constraints."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: {error}") from error
    return records


def validate(
    config_path: Path, manifest_path: Path, verify_hashes: bool = False
) -> tuple[list[str], list[str]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    image_root = Path(config["image_root"])
    label_root = Path(config["label_root"])
    records = load_jsonl(manifest_path)
    errors: list[str] = []
    warnings: list[str] = []
    sample_ids: set[str] = set()
    seeds_by_split: dict[str, set[int]] = defaultdict(set)
    splits_by_exact_group: dict[str, set[str]] = defaultdict(set)

    for record in records:
        sample_id = record.get("sample_id")
        if not sample_id or sample_id in sample_ids:
            errors.append(f"missing or duplicate sample_id: {sample_id!r}")
        sample_ids.add(sample_id)
        image_path = image_root / record["image"]
        if not image_path.is_file():
            errors.append(f"missing image: {image_path}")
        elif verify_hashes and sha256_file(image_path) != record["image_sha256"]:
            errors.append(f"image hash mismatch: {image_path}")
        label_name = record.get("label_file")
        if label_name:
            label_path = label_root / label_name
            if not label_path.is_file():
                errors.append(f"missing label: {label_path}")
            elif verify_hashes and sha256_file(label_path) != record["label_sha256"]:
                errors.append(f"label hash mismatch: {label_path}")
        for box in record.get("boxes", []):
            bbox = box.get("bbox_yolo_cxcywh", [])
            if len(bbox) != 4 or any(value < 0.0 or value > 1.0 for value in bbox):
                errors.append(f"invalid bbox in {sample_id}: {bbox}")
        split = record.get("split", "unassigned")
        seeds_by_split[split].add(int(record["scene_seed"]))
        group = record.get("exact_duplicate_group")
        if group and split != "unassigned":
            splits_by_exact_group[group].add(split)

    assigned_splits = {name: seeds for name, seeds in seeds_by_split.items() if name != "unassigned"}
    assigned_names = sorted(assigned_splits)
    for index, left_name in enumerate(assigned_names):
        for right_name in assigned_names[index + 1 :]:
            overlap = assigned_splits[left_name] & assigned_splits[right_name]
            if overlap:
                errors.append(
                    f"scene seed leakage between {left_name} and {right_name}: {sorted(overlap)}"
                )
    for group, splits in splits_by_exact_group.items():
        if len(splits) > 1:
            errors.append(f"exact duplicate group {group} crosses splits: {sorted(splits)}")
    if not records:
        errors.append("manifest is empty")
    if "unassigned" in seeds_by_split:
        warnings.append("manifest contains unassigned samples; formal evaluation is not sealed")
    if len({record["scene_seed"] for record in records}) < 6:
        warnings.append("fewer than six scene seeds; retain this dataset as a pilot only")
    return errors, warnings


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=project_root / "data" / "manifests" / "batch_500.jsonl",
    )
    parser.add_argument("--verify-hashes", action="store_true")
    args = parser.parse_args()
    errors, warnings = validate(
        args.config.resolve(), args.manifest.resolve(), verify_hashes=args.verify_hashes
    )
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        print(f"FAILED: {len(errors)} error(s), {len(warnings)} warning(s)")
        return 1
    print(f"PASSED: 0 errors, {len(warnings)} warning(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
