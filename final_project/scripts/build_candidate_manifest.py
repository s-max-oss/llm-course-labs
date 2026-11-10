#!/usr/bin/env python3
"""Convert image-level annotations into VLM candidate metadata."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def padded_crop_xyxy(
    bbox_xyxy: list[float], width: int, height: int, padding_ratio: float
) -> list[float]:
    x1, y1, x2, y2 = bbox_xyxy
    box_width = x2 - x1
    box_height = y2 - y1
    return [
        round(max(0.0, x1 - box_width * padding_ratio), 3),
        round(max(0.0, y1 - box_height * padding_ratio), 3),
        round(min(float(width), x2 + box_width * padding_ratio), 3),
        round(min(float(height), y2 + box_height * padding_ratio), 3),
    ]


def build(
    source_manifest: Path,
    near_duplicates_path: Path,
    output_path: Path,
    summary_path: Path,
    padding_ratio: float,
) -> tuple[Path, Path]:
    records = load_jsonl(source_manifest)
    near_data = json.loads(near_duplicates_path.read_text(encoding="utf-8"))
    near_samples = {
        sample_id
        for pair in near_data.get("pairs", [])
        for sample_id in (pair["left"], pair["right"])
    }
    candidates: list[dict[str, Any]] = []
    for record in records:
        common = {
            "schema_version": 1,
            "dataset_id": record["dataset_id"],
            "source_sample_id": record["sample_id"],
            "image_root_id": record["image_root_id"],
            "image": record["image"],
            "image_sha256": record["image_sha256"],
            "scene_seed": record["scene_seed"],
            "room_id": record["room_id"],
            "frame_id": record["frame_id"],
            "view_id": record["view_id"],
            "width": record["width"],
            "height": record["height"],
            "split": "unassigned",
            "formal_evaluation_eligible": False,
            "exact_duplicate_group": record["exact_duplicate_group"],
            "near_duplicate_candidate": record["sample_id"] in near_samples,
            "recommended_visual_inputs": ["full_frame_with_bbox", "padded_target_crop"],
        }
        if record["boxes"]:
            for box_index, box in enumerate(record["boxes"], start=1):
                candidate = dict(common)
                candidate.update(
                    {
                        "candidate_id": f"{record['sample_id']}__box_{box_index:02d}",
                        "candidate_type": "annotated_object",
                        "target_label": box["semantic_label"],
                        "detector_class": box["detector_class"],
                        "bbox_xyxy_pixels": box["bbox_xyxy_pixels"],
                        "padded_crop_xyxy_pixels": padded_crop_xyxy(
                            box["bbox_xyxy_pixels"],
                            record["width"],
                            record["height"],
                            padding_ratio,
                        ),
                        "bbox_area_fraction": box["area_fraction"],
                        "touches_image_edge": box["touches_image_edge"],
                        "label_source": record["label_source"],
                        "label_reliability": "legacy_box_annotation_requires_review",
                    }
                )
                candidates.append(candidate)
        else:
            candidate = dict(common)
            candidate.update(
                {
                    "candidate_id": f"{record['sample_id']}__background",
                    "candidate_type": "background_frame",
                    "target_label": "background_or_unknown",
                    "detector_class": None,
                    "bbox_xyxy_pixels": None,
                    "padded_crop_xyxy_pixels": None,
                    "bbox_area_fraction": None,
                    "touches_image_edge": False,
                    "label_source": record["label_source"],
                    "label_reliability": (
                        "legacy_missing_label_background_convention_requires_review"
                        if record["label_status"] == "missing_label_file"
                        else "legacy_empty_label_requires_review"
                    ),
                    "recommended_visual_inputs": ["full_frame"],
                }
            )
            candidates.append(candidate)

    label_counts = Counter(candidate["target_label"] for candidate in candidates)
    object_areas = sorted(
        candidate["bbox_area_fraction"]
        for candidate in candidates
        if candidate["bbox_area_fraction"] is not None
    )

    def percentile(fraction: float) -> float | None:
        if not object_areas:
            return None
        index = round((len(object_areas) - 1) * fraction)
        return object_areas[index]

    summary = {
        "schema_version": 1,
        "source_manifest": source_manifest.name,
        "candidate_count": len(candidates),
        "object_candidate_count": sum(
            candidate["candidate_type"] == "annotated_object" for candidate in candidates
        ),
        "background_candidate_count": sum(
            candidate["candidate_type"] == "background_frame" for candidate in candidates
        ),
        "target_label_counts": dict(sorted(label_counts.items())),
        "padded_crop_ratio": padding_ratio,
        "object_area_fraction_percentiles": {
            "p10": percentile(0.10),
            "p25": percentile(0.25),
            "p50": percentile(0.50),
            "p75": percentile(0.75),
            "p90": percentile(0.90),
        },
        "edge_touching_object_count": sum(
            bool(candidate["touches_image_edge"])
            for candidate in candidates
            if candidate["candidate_type"] == "annotated_object"
        ),
        "candidate_count_with_exact_duplicate": sum(
            candidate["exact_duplicate_group"] is not None for candidate in candidates
        ),
        "candidate_count_with_near_duplicate_flag": sum(
            candidate["near_duplicate_candidate"] for candidate in candidates
        ),
        "split_status": "unassigned",
        "formal_evaluation_ready": False,
    }
    atomic_write(
        output_path,
        "".join(
            json.dumps(candidate, ensure_ascii=False, sort_keys=True) + "\n"
            for candidate in candidates
        ),
    )
    atomic_write(summary_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return output_path, summary_path


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=project_root / "data" / "manifests" / "batch_500.jsonl",
    )
    parser.add_argument(
        "--near-duplicates",
        type=Path,
        default=project_root / "data" / "manifests" / "batch_500.near_duplicates.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "data" / "manifests" / "batch_500.candidates.jsonl",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=project_root / "data" / "manifests" / "batch_500.candidates.summary.json",
    )
    parser.add_argument("--padding-ratio", type=float, default=0.15)
    args = parser.parse_args()
    if args.padding_ratio < 0.0:
        parser.error("--padding-ratio must be non-negative")
    outputs = build(
        args.source_manifest.resolve(),
        args.near_duplicates.resolve(),
        args.output.resolve(),
        args.summary.resolve(),
        args.padding_ratio,
    )
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
