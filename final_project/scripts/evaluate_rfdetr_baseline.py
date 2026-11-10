#!/usr/bin/env python3
"""Offline evaluator for the frozen RF-DETR five-seed pilot baseline."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from robot_vlm_baselines.contracts import (  # noqa: E402
    BaselineContractError,
    load_jsonl,
    percentile,
    sha256_file,
    validate_baseline_policy,
    validate_inference_jobs,
    write_json,
)


RESULT_FIELDS = {
    "schema_version",
    "job_id",
    "baseline_id",
    "fold_id",
    "sample_id",
    "image_sha256",
    "status",
    "error",
    "latency_ms",
    "predictions",
}
PREDICTION_FIELDS = {
    "prediction_index",
    "class_id",
    "raw_class_name",
    "canonical_label",
    "confidence",
    "bbox_xyxy_pixels",
    "bbox_area_fraction",
    "mask_area_fraction",
    "mask_coverage_in_bbox",
}


def _records_by_id(records: list[dict[str, Any]], field: str, label: str):
    mapped = {}
    for record in records:
        identifier = record.get(field)
        if not isinstance(identifier, str) or not identifier or identifier in mapped:
            raise BaselineContractError(f"{label}: missing or duplicate {field}: {identifier!r}")
        mapped[identifier] = record
    return mapped


def _is_finite_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def _report_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def bbox_iou(first: list[float], second: list[float]) -> float:
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def match_class(
    ground_truth_boxes: list[list[float]],
    predictions: list[dict[str, Any]],
    iou_threshold: float,
) -> dict[str, Any]:
    unmatched = set(range(len(ground_truth_boxes)))
    matches = []
    false_positives = []
    ordered = sorted(predictions, key=lambda item: (-float(item["confidence"]), item["prediction_index"]))
    for prediction in ordered:
        candidates = [
            (bbox_iou(prediction["bbox_xyxy_pixels"], ground_truth_boxes[index]), index)
            for index in unmatched
        ]
        best_iou, best_index = max(candidates, default=(0.0, -1))
        if best_iou >= iou_threshold:
            unmatched.remove(best_index)
            matches.append(
                {
                    "prediction_index": prediction["prediction_index"],
                    "ground_truth_index": best_index,
                    "iou": best_iou,
                }
            )
        else:
            false_positives.append(prediction["prediction_index"])
    return {
        "true_positive": len(matches),
        "false_positive": len(false_positives),
        "false_negative": len(unmatched),
        "matches": matches,
        "false_positive_prediction_indices": false_positives,
        "false_negative_ground_truth_indices": sorted(unmatched),
    }


def _prf(true_positive: int, false_positive: int, false_negative: int) -> dict[str, Any]:
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else None
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else None
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else 0.0
    )
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _distribution(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
    }


def _expected_job(image: dict[str, Any], baseline_id: str) -> dict[str, Any]:
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


def _load_frozen_protocol(
    protocol_path: Path,
    lock_path: Path,
    config_path: Path,
    jobs_protocol_path: Path,
    pilot_protocol_path: Path,
) -> tuple[dict, dict]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != lock.get("protocol_id"):
        raise BaselineContractError("RF-DETR protocol and lock IDs differ")
    if lock.get("frozen_before_full_inference") is not True:
        raise BaselineContractError("RF-DETR protocol lock is not marked as pre-inference")
    sealed = lock.get("sealed_files", {})
    paths = {
        "evaluation_protocol": protocol_path,
        "baseline_config": config_path,
        "baseline_jobs_protocol": jobs_protocol_path,
        "pilot_protocol": pilot_protocol_path,
    }
    for name, path in paths.items():
        if sealed.get(name, {}).get("sha256") != sha256_file(path):
            raise BaselineContractError(f"RF-DETR sealed file differs: {name}")
    if protocol.get("status") != "frozen_before_full_inference":
        raise BaselineContractError("RF-DETR evaluation protocol is not frozen")
    return protocol, lock


def evaluate(
    result_paths: list[Path],
    protocol_path: Path,
    lock_path: Path,
    config_path: Path,
    jobs_root: Path,
    split_root: Path,
) -> dict[str, Any]:
    jobs_protocol_path = jobs_root / "protocol.json"
    pilot_protocol_path = split_root / "protocol.json"
    protocol, _ = _load_frozen_protocol(
        protocol_path,
        lock_path,
        config_path,
        jobs_protocol_path,
        pilot_protocol_path,
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_baseline_policy(config)
    jobs_protocol = json.loads(jobs_protocol_path.read_text(encoding="utf-8"))
    baseline = config["rfdetr"]
    if jobs_protocol.get("rfdetr_baseline") != baseline:
        raise BaselineContractError("RF-DETR jobs protocol differs from baseline config")
    if protocol["inference_contract"]["confidence_threshold"] != baseline["confidence_threshold"]:
        raise BaselineContractError("RF-DETR evaluation threshold differs from inference threshold")
    if protocol["inference_contract"]["checkpoint_sha256"] != baseline["checkpoint_sha256"]:
        raise BaselineContractError("RF-DETR evaluation checkpoint differs from baseline config")
    if protocol["inference_contract"]["resolution"] != baseline["resolution"]:
        raise BaselineContractError("RF-DETR evaluation resolution differs from baseline config")

    jobs_by_sample = {}
    truth_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    truth_sources = []
    job_sources = []
    fold_by_sample = {}
    for fold_number in range(1, 6):
        fold_id = f"fold_{fold_number:02d}"
        image_path = split_root / fold_id / "test_images.jsonl"
        candidate_path = split_root / fold_id / "test_candidates.jsonl"
        job_path = jobs_root / fold_id / "rfdetr_images.jsonl"
        source_files = jobs_protocol["source_files"]
        for suffix, path in (("images", image_path), ("candidates", candidate_path)):
            key = f"{fold_id}_test_{suffix}"
            if source_files.get(key, {}).get("sha256") != sha256_file(path):
                raise BaselineContractError(f"sealed RF-DETR truth hash mismatch: {key}")
        images = load_jsonl(image_path)
        jobs = load_jsonl(job_path)
        validate_inference_jobs(jobs, config, "rfdetr")
        expected_jobs = [_expected_job(image, baseline["baseline_id"]) for image in images]
        if jobs != expected_jobs:
            raise BaselineContractError(f"{fold_id}: RF-DETR jobs differ from sealed test images")
        for job in jobs:
            sample_id = job["sample_id"]
            if sample_id in jobs_by_sample:
                raise BaselineContractError(f"duplicate RF-DETR sample across folds: {sample_id}")
            jobs_by_sample[sample_id] = job
            fold_by_sample[sample_id] = fold_id
        for candidate in load_jsonl(candidate_path):
            if candidate["bbox_xyxy_pixels"] is not None:
                truth_by_sample[candidate["source_sample_id"]].append(candidate)
        truth_sources.append(
            {"fold_id": fold_id, "path": _report_path(candidate_path), "sha256": sha256_file(candidate_path)}
        )
        job_sources.append(
            {"fold_id": fold_id, "path": _report_path(job_path), "sha256": sha256_file(job_path)}
        )
    if len(jobs_by_sample) != protocol["inference_contract"]["expected_image_count"]:
        raise BaselineContractError("RF-DETR image count differs from frozen protocol")

    results = []
    result_sources = []
    expected_result_file_count = protocol["inference_contract"]["expected_result_file_count"]
    if len(result_paths) != expected_result_file_count or len(set(result_paths)) != len(result_paths):
        raise BaselineContractError(
            f"RF-DETR evaluation requires {expected_result_file_count} distinct result files"
        )
    result_file_folds = set()
    for path in result_paths:
        file_records = load_jsonl(path)
        fold_ids = {record.get("fold_id") for record in file_records}
        if not file_records or len(fold_ids) != 1:
            raise BaselineContractError(
                f"RF-DETR result file must be nonempty and contain exactly one fold: {path}"
            )
        fold_id = next(iter(fold_ids))
        if fold_id in result_file_folds:
            raise BaselineContractError(f"duplicate RF-DETR result file for fold: {fold_id}")
        result_file_folds.add(fold_id)
        result_sources.append({"path": _report_path(path), "sha256": sha256_file(path)})
        results.extend(file_records)
    if result_file_folds != set(fold_by_sample.values()):
        raise BaselineContractError("RF-DETR result files do not cover all five folds exactly once")
    results_by_sample = _records_by_id(results, "sample_id", "RF-DETR results")
    if set(results_by_sample) != set(jobs_by_sample):
        raise BaselineContractError("RF-DETR results do not exactly cover the 401 sealed jobs")

    target_classes = protocol["evaluation_contract"]["target_classes"]
    auxiliary_classes = set(protocol["evaluation_contract"]["auxiliary_unscored_classes"])
    allowed_classes = set(target_classes) | auxiliary_classes
    iou_threshold = float(protocol["evaluation_contract"]["bbox_iou_threshold"])
    totals = {label: Counter() for label in target_classes}
    per_fold = {f"fold_{number:02d}": {label: Counter() for label in target_classes} for number in range(1, 6)}
    latencies = []
    mask_areas = []
    mask_coverages = []
    match_ious = []
    auxiliary_counts = Counter()
    status_counts = Counter()
    background_target_prediction_count = 0
    background_image_count = 0
    danger_presence = Counter()

    class_id_to_name = {int(key): value for key, value in baseline["class_names"].items()}
    for sample_id, job in jobs_by_sample.items():
        result = results_by_sample[sample_id]
        if set(result) != RESULT_FIELDS:
            raise BaselineContractError(f"{sample_id}: RF-DETR result fields differ from contract")
        for field, expected in {
            "schema_version": 1,
            "job_id": job["job_id"],
            "baseline_id": job["baseline_id"],
            "fold_id": job["fold_id"],
            "image_sha256": job["image_sha256"],
        }.items():
            if result.get(field) != expected:
                raise BaselineContractError(f"{sample_id}: result {field} differs from job")
        latency = result["latency_ms"]
        if not _is_finite_number(latency) or latency < 0:
            raise BaselineContractError(f"{sample_id}: invalid RF-DETR latency")
        latencies.append(float(latency))
        status = result["status"]
        status_counts[status] += 1
        if status == "error":
            if result["predictions"] != [] or not isinstance(result["error"], str) or not result["error"]:
                raise BaselineContractError(f"{sample_id}: inconsistent RF-DETR error record")
            predictions = []
        elif status == "ok":
            if result["error"] is not None or not isinstance(result["predictions"], list):
                raise BaselineContractError(f"{sample_id}: inconsistent RF-DETR ok record")
            predictions = result["predictions"]
        else:
            raise BaselineContractError(f"{sample_id}: unknown RF-DETR status")

        for index, prediction in enumerate(predictions):
            if (
                set(prediction) != PREDICTION_FIELDS
                or type(prediction["prediction_index"]) is not int
                or prediction["prediction_index"] != index
            ):
                raise BaselineContractError(f"{sample_id}: prediction schema or index differs")
            class_id = prediction["class_id"]
            if type(class_id) is not int or class_id not in class_id_to_name:
                raise BaselineContractError(f"{sample_id}: invalid RF-DETR class_id")
            raw_name = class_id_to_name[class_id]
            if (
                prediction["raw_class_name"] != raw_name
                or prediction["canonical_label"] != baseline["class_aliases"][raw_name]
                or prediction["canonical_label"] not in allowed_classes
            ):
                raise BaselineContractError(f"{sample_id}: RF-DETR class mapping differs")
            confidence = prediction["confidence"]
            if not _is_finite_number(confidence) or not baseline["confidence_threshold"] <= confidence <= 1:
                raise BaselineContractError(f"{sample_id}: invalid RF-DETR confidence")
            bbox = prediction["bbox_xyxy_pixels"]
            if not isinstance(bbox, list) or len(bbox) != 4 or not all(_is_finite_number(value) for value in bbox):
                raise BaselineContractError(f"{sample_id}: invalid RF-DETR bbox")
            if not (0 <= bbox[0] < bbox[2] <= job["width"] and 0 <= bbox[1] < bbox[3] <= job["height"]):
                raise BaselineContractError(f"{sample_id}: RF-DETR bbox outside image")
            for field in ("bbox_area_fraction", "mask_area_fraction", "mask_coverage_in_bbox"):
                value = prediction[field]
                if not _is_finite_number(value) or not 0 <= value <= 1:
                    raise BaselineContractError(f"{sample_id}: invalid {field}")
            mask_areas.append(float(prediction["mask_area_fraction"]))
            mask_coverages.append(float(prediction["mask_coverage_in_bbox"]))

        truths = truth_by_sample.get(sample_id, [])
        if not truths:
            background_image_count += 1
            background_target_prediction_count += sum(
                prediction["canonical_label"] in target_classes for prediction in predictions
            )
        fold_id = fold_by_sample[sample_id]
        for label in target_classes:
            gt_boxes = [item["bbox_xyxy_pixels"] for item in truths if item["target_label"] == label]
            class_predictions = [item for item in predictions if item["canonical_label"] == label]
            matched = match_class(gt_boxes, class_predictions, iou_threshold)
            for key in ("true_positive", "false_positive", "false_negative"):
                totals[label][key] += matched[key]
                per_fold[fold_id][label][key] += matched[key]
            match_ious.extend(item["iou"] for item in matched["matches"])
        for prediction in predictions:
            if prediction["canonical_label"] in auxiliary_classes:
                auxiliary_counts[prediction["canonical_label"]] += 1

        truth_danger = any(item["target_label"] == "danger_red_sphere" for item in truths)
        predicted_danger = any(item["canonical_label"] == "danger_red_sphere" for item in predictions)
        if truth_danger and predicted_danger:
            danger_presence["true_positive"] += 1
        elif not truth_danger and predicted_danger:
            danger_presence["false_positive"] += 1
        elif truth_danger and not predicted_danger:
            danger_presence["false_negative"] += 1
        else:
            danger_presence["true_negative"] += 1

    per_class = {}
    for label in target_classes:
        counts = totals[label]
        per_class[label] = _prf(counts["true_positive"], counts["false_positive"], counts["false_negative"])
        per_class[label]["support"] = counts["true_positive"] + counts["false_negative"]
        per_class[label]["predicted"] = counts["true_positive"] + counts["false_positive"]
    macro_f1 = sum(per_class[label]["f1"] for label in target_classes) / len(target_classes)
    fold_reports = {}
    for fold_id, classes in per_fold.items():
        fold_reports[fold_id] = {
            label: _prf(values["true_positive"], values["false_positive"], values["false_negative"])
            for label, values in classes.items()
        }
    danger_image_level = _prf(
        danger_presence["true_positive"], danger_presence["false_positive"], danger_presence["false_negative"]
    )
    danger_image_level["true_negative"] = danger_presence["true_negative"]

    return {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "scope_label": protocol["scope_label"],
        "formal_generalization_claim_allowed": False,
        "source_files": {
            "evaluation_protocol": {"path": _report_path(protocol_path), "sha256": sha256_file(protocol_path)},
            "protocol_lock": {"path": _report_path(lock_path), "sha256": sha256_file(lock_path)},
            "baseline_config": {"path": _report_path(config_path), "sha256": sha256_file(config_path)},
            "baseline_jobs_protocol": {"path": _report_path(jobs_protocol_path), "sha256": sha256_file(jobs_protocol_path)},
            "jobs": job_sources,
            "ground_truth_candidates": truth_sources,
            "results": result_sources,
        },
        "metrics": {
            "image_count": len(jobs_by_sample),
            "status_counts": dict(sorted(status_counts.items())),
            "bbox_iou_threshold": iou_threshold,
            "per_target_class": per_class,
            "three_target_class_macro_f1": macro_f1,
            "danger_object_detection": per_class["danger_red_sphere"],
            "danger_image_presence": danger_image_level,
            "per_fold_counts": fold_reports,
            "background_convention_image_count": background_image_count,
            "target_predictions_on_background_convention_images": background_target_prediction_count,
            "auxiliary_unscored_prediction_counts": dict(sorted(auxiliary_counts.items())),
            "matched_bbox_iou": _distribution(match_ious),
            "mask_area_fraction": _distribution(mask_areas),
            "mask_coverage_in_bbox": _distribution(mask_coverages),
            "latency_ms": _distribution(latencies),
        },
        "limitations": [
            "five-seed exploratory pilot; no formal generalization claim",
            "legacy missing-label frames are an uncertain background convention, not audited negatives",
            "green-wall predictions are auxiliary because the pilot has no sealed green-wall boxes",
            "no ground-truth masks are available, so mask IoU is not reportable",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "configs" / "rfdetr_evaluation_protocol.json")
    parser.add_argument("--lock", type=Path, default=PROJECT_ROOT / "experiments" / "pilot_v1" / "rfdetr_evaluation_protocol.lock.json")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "baselines.json")
    parser.add_argument("--jobs-root", type=Path, default=PROJECT_ROOT / "experiments" / "pilot_v1" / "baseline_jobs")
    parser.add_argument("--split-root", type=Path, default=PROJECT_ROOT / "data" / "splits" / "pilot_v1")
    args = parser.parse_args()
    if args.output.exists():
        print(f"ERROR: refusing to overwrite existing output: {args.output}", file=sys.stderr)
        return 2
    try:
        report = evaluate(
            [path.resolve() for path in args.results], args.protocol.resolve(), args.lock.resolve(),
            args.config.resolve(), args.jobs_root.resolve(), args.split_root.resolve(),
        )
    except (BaselineContractError, OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    write_json(args.output.resolve(), report)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
