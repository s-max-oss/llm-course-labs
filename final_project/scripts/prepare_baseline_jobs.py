#!/usr/bin/env python3
"""Build deterministic, label-free inference queues from sealed pilot folds."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from robot_vlm_baselines.contracts import (  # noqa: E402
    BaselineContractError,
    find_forbidden_fields,
    load_jsonl,
    sha256_file,
    validate_baseline_policy,
    write_json,
    write_jsonl,
)


def validate_mandatory_baseline_policy(config: dict[str, Any]) -> None:
    try:
        validate_baseline_policy(config)
    except BaselineContractError as exc:
        raise ValueError(str(exc)) from exc


def detector_job(image: dict[str, Any], baseline_id: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "job_id": f"{baseline_id}::{image['fold_id']}::{image['sample_id']}",
        "baseline_id": baseline_id,
        "pilot_protocol_id": image["pilot_protocol_id"],
        "fold_id": image["fold_id"],
        "split": "test",
        "task": "instance_segmentation",
        "sample_id": image["sample_id"],
        "image_root_id": image["image_root_id"],
        "image": image["image"],
        "image_sha256": image["image_sha256"],
        "width": image["width"],
        "height": image["height"],
    }


def smolvlm_job(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    region = candidate["padded_crop_xyxy_pixels"]
    return {
        "schema_version": 1,
        "job_id": f"{baseline['baseline_id']}::{candidate['fold_id']}::{candidate['candidate_id']}",
        "baseline_id": baseline["baseline_id"],
        "pilot_protocol_id": candidate["pilot_protocol_id"],
        "fold_id": candidate["fold_id"],
        "split": "test",
        "task": "strict_four_class_classification",
        "candidate_id": candidate["candidate_id"],
        "source_sample_id": candidate["source_sample_id"],
        "image_root_id": candidate["image_root_id"],
        "image": candidate["image"],
        "image_sha256": candidate["image_sha256"],
        "width": candidate["width"],
        "height": candidate["height"],
        "target_region_xyxy_pixels": region,
        "input_variant": baseline["input_variant"],
        "proposal_source": (
            "offline_annotation_oracle" if region is not None else "offline_background_convention"
        ),
        "deployment_eligible": False,
    }


def build(
    config_path: Path,
    split_root: Path,
    output_root: Path,
) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    pilot_protocol_path = split_root / "protocol.json"
    pilot_protocol = json.loads(pilot_protocol_path.read_text(encoding="utf-8"))
    validate_mandatory_baseline_policy(config)
    if config["protocol_id"] != pilot_protocol["protocol_id"]:
        raise ValueError("baseline config protocol does not match pilot protocol")
    if config["expected_labels"] != pilot_protocol["expected_classes"]:
        raise ValueError("baseline labels do not match sealed pilot classes")
    forbidden = set(config["inference_forbidden_fields"])
    fold_summaries: list[dict[str, Any]] = []
    source_files: dict[str, dict[str, str]] = {
        "baseline_config": {
            "path": config_path.name,
            "sha256": sha256_file(config_path),
        },
        "pilot_protocol": {
            "path": pilot_protocol_path.name,
            "sha256": sha256_file(pilot_protocol_path),
        },
    }
    total_detector_jobs = 0
    total_vlm_jobs = 0
    for fold in pilot_protocol["folds"]:
        fold_id = fold["fold_id"]
        fold_root = split_root / fold_id
        image_path = fold_root / "test_images.jsonl"
        candidate_path = fold_root / "test_candidates.jsonl"
        images = load_jsonl(image_path)
        candidates = load_jsonl(candidate_path)
        detector_jobs = [
            detector_job(image, config["rfdetr"]["baseline_id"]) for image in images
        ]
        vlm_jobs = [smolvlm_job(candidate, config["smolvlm"]) for candidate in candidates]
        for job in detector_jobs + vlm_jobs:
            leaked = find_forbidden_fields(job, forbidden)
            if leaked:
                raise ValueError(f"inference job contains forbidden fields: {leaked}")
        if len({job["job_id"] for job in detector_jobs}) != len(detector_jobs):
            raise ValueError(f"{fold_id}: duplicate detector job IDs")
        if len({job["job_id"] for job in vlm_jobs}) != len(vlm_jobs):
            raise ValueError(f"{fold_id}: duplicate SmolVLM job IDs")

        output_fold = output_root / fold_id
        write_jsonl(output_fold / "rfdetr_images.jsonl", detector_jobs)
        write_jsonl(output_fold / "smolvlm_candidates.jsonl", vlm_jobs)
        source_files[f"{fold_id}_test_images"] = {
            "path": f"{fold_id}/test_images.jsonl",
            "sha256": sha256_file(image_path),
        }
        source_files[f"{fold_id}_test_candidates"] = {
            "path": f"{fold_id}/test_candidates.jsonl",
            "sha256": sha256_file(candidate_path),
        }
        proposal_counts = dict(sorted(Counter(job["proposal_source"] for job in vlm_jobs).items()))
        fold_summaries.append(
            {
                "fold_id": fold_id,
                "detector_job_count": len(detector_jobs),
                "smolvlm_job_count": len(vlm_jobs),
                "smolvlm_proposal_source_counts": proposal_counts,
            }
        )
        total_detector_jobs += len(detector_jobs)
        total_vlm_jobs += len(vlm_jobs)

    protocol = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "artifact": "label_free_baseline_inference_jobs",
        "source_files": source_files,
        "rfdetr_baseline": config["rfdetr"],
        "smolvlm_baseline": config["smolvlm"],
        "inference_forbidden_fields": sorted(forbidden),
        "ground_truth_available_to_inference": False,
        "smolvlm_oracle_crop_is_offline_diagnostic_only": True,
        "total_detector_jobs": total_detector_jobs,
        "total_smolvlm_jobs": total_vlm_jobs,
        "folds": fold_summaries,
    }
    write_json(output_root / "protocol.json", protocol)
    return output_root / "protocol.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "baselines.json"
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "splits" / "pilot_v1",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "pilot_v1" / "baseline_jobs",
    )
    args = parser.parse_args()
    print(build(args.config.resolve(), args.split_root.resolve(), args.output_root.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
