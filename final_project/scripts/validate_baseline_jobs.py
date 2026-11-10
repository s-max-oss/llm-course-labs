#!/usr/bin/env python3
"""Independently validate label-free baseline inference queues."""

from __future__ import annotations

import argparse
import json
import sys
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
)


def expected_detector_job(image: dict[str, Any], baseline_id: str) -> dict[str, Any]:
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


def expected_vlm_job(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
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


def validate(
    config_path: Path,
    split_root: Path,
    jobs_root: Path,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    required = [config_path, split_root / "protocol.json", jobs_root / "protocol.json"]
    missing = [path for path in required if not path.is_file()]
    if missing:
        return [f"missing required file: {path}" for path in missing], warnings
    config = json.loads(config_path.read_text(encoding="utf-8"))
    pilot_protocol_path = split_root / "protocol.json"
    pilot = json.loads(pilot_protocol_path.read_text(encoding="utf-8"))
    jobs_protocol = json.loads((jobs_root / "protocol.json").read_text(encoding="utf-8"))
    forbidden = set(config["inference_forbidden_fields"])

    try:
        validate_baseline_policy(config)
    except BaselineContractError as exc:
        errors.append(str(exc))

    if jobs_protocol.get("ground_truth_available_to_inference") is not False:
        errors.append("jobs protocol must state ground truth is unavailable to inference")
    if jobs_protocol.get("smolvlm_oracle_crop_is_offline_diagnostic_only") is not True:
        errors.append("oracle-crop baseline must remain an offline diagnostic")
    if jobs_protocol.get("rfdetr_baseline") != config["rfdetr"]:
        errors.append("RF-DETR baseline metadata differs from config")
    if jobs_protocol.get("smolvlm_baseline") != config["smolvlm"]:
        errors.append("SmolVLM baseline metadata differs from config")
    if jobs_protocol.get("inference_forbidden_fields") != sorted(forbidden):
        errors.append("forbidden-field policy differs from config")

    source_paths = {
        "baseline_config": config_path,
        "pilot_protocol": pilot_protocol_path,
    }
    for fold in pilot["folds"]:
        fold_id = fold["fold_id"]
        source_paths[f"{fold_id}_test_images"] = split_root / fold_id / "test_images.jsonl"
        source_paths[f"{fold_id}_test_candidates"] = (
            split_root / fold_id / "test_candidates.jsonl"
        )
    for key, path in source_paths.items():
        recorded = jobs_protocol.get("source_files", {}).get(key, {})
        if recorded.get("sha256") != sha256_file(path):
            errors.append(f"source hash mismatch: {key}")

    total_detector = 0
    total_vlm = 0
    protocol_folds = jobs_protocol.get("folds", [])
    protocol_fold_ids = [fold.get("fold_id") for fold in protocol_folds]
    expected_fold_ids = [fold["fold_id"] for fold in pilot["folds"]]
    if len(protocol_folds) != len(pilot["folds"]):
        errors.append("jobs protocol fold count differs from pilot")
    if len(protocol_fold_ids) != len(set(protocol_fold_ids)):
        errors.append("jobs protocol contains duplicate fold summaries")
    if set(protocol_fold_ids) != set(expected_fold_ids):
        errors.append("jobs protocol has missing or duplicate fold summaries")
    protocol_fold_by_id = {fold.get("fold_id"): fold for fold in protocol_folds}
    for fold in pilot["folds"]:
        fold_id = fold["fold_id"]
        source_images = load_jsonl(split_root / fold_id / "test_images.jsonl")
        source_candidates = load_jsonl(split_root / fold_id / "test_candidates.jsonl")
        detector_path = jobs_root / fold_id / "rfdetr_images.jsonl"
        vlm_path = jobs_root / fold_id / "smolvlm_candidates.jsonl"
        if not detector_path.is_file() or not vlm_path.is_file():
            errors.append(f"{fold_id}: missing inference queue")
            continue
        actual_detector = load_jsonl(detector_path)
        actual_vlm = load_jsonl(vlm_path)
        expected_detector = [
            expected_detector_job(image, config["rfdetr"]["baseline_id"])
            for image in source_images
        ]
        expected_vlm = [
            expected_vlm_job(candidate, config["smolvlm"])
            for candidate in source_candidates
        ]
        if actual_detector != expected_detector:
            errors.append(f"{fold_id}: detector jobs differ from sealed test images")
        if actual_vlm != expected_vlm:
            errors.append(f"{fold_id}: SmolVLM jobs differ from sealed test candidates")
        for job in actual_detector + actual_vlm:
            leaked = find_forbidden_fields(job, forbidden)
            if leaked:
                errors.append(f"{job.get('job_id')}: forbidden fields present: {leaked}")
        if any(job.get("deployment_eligible") is not False for job in actual_vlm):
            errors.append(f"{fold_id}: oracle-crop job marked deployment eligible")

        summary = protocol_fold_by_id.get(fold_id, {})
        if summary.get("detector_job_count") != len(actual_detector):
            errors.append(f"{fold_id}: detector count differs from protocol")
        if summary.get("smolvlm_job_count") != len(actual_vlm):
            errors.append(f"{fold_id}: SmolVLM count differs from protocol")
        total_detector += len(actual_detector)
        total_vlm += len(actual_vlm)

    if jobs_protocol.get("total_detector_jobs") != total_detector:
        errors.append("total detector job count differs from protocol")
    if jobs_protocol.get("total_smolvlm_jobs") != total_vlm:
        errors.append("total SmolVLM job count differs from protocol")
    warnings.append(
        "SmolVLM oracle-crop jobs use offline annotation geometry and are not deployable"
    )
    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "baselines.json"
    )
    parser.add_argument(
        "--split-root", type=Path, default=PROJECT_ROOT / "data" / "splits" / "pilot_v1"
    )
    parser.add_argument(
        "--jobs-root",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "pilot_v1" / "baseline_jobs",
    )
    args = parser.parse_args()
    errors, warnings = validate(
        args.config.resolve(), args.split_root.resolve(), args.jobs_root.resolve()
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
