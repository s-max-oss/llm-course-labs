#!/usr/bin/env python3
"""Independently fail closed on pilot split leakage or metadata corruption."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def exact_duplicate_ids(images: list[dict[str, Any]]) -> set[str]:
    by_hash: dict[str, list[str]] = defaultdict(list)
    for image in images:
        by_hash[image["image_sha256"]].append(image["sample_id"])
    return {
        sample_id
        for sample_ids in by_hash.values()
        if len(sample_ids) > 1
        for sample_id in sample_ids
    }


def cross_seed_dhash_pairs(
    images: list[dict[str, Any]], threshold: int
) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for left_index, left in enumerate(images):
        left_hash = int(left["dhash64"], 16)
        for right in images[left_index + 1 :]:
            if left["scene_seed"] == right["scene_seed"]:
                continue
            if left["image_sha256"] == right["image_sha256"]:
                continue
            distance = (left_hash ^ int(right["dhash64"], 16)).bit_count()
            if distance <= threshold:
                pairs.add(tuple(sorted((left["sample_id"], right["sample_id"]))))
    return pairs


def assigned_record(
    source: dict[str, Any], protocol_id: str, fold_id: str, split: str
) -> dict[str, Any]:
    expected = dict(source)
    expected["source_split"] = source.get("split", "unassigned")
    expected["split"] = split
    expected["pilot_protocol_id"] = protocol_id
    expected["fold_id"] = fold_id
    return expected


def records_by_id(
    records: list[dict[str, Any]], id_field: str, label: str, errors: list[str]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        record_id = record.get(id_field)
        if not record_id or record_id in result:
            errors.append(f"{label}: missing or duplicate {id_field}: {record_id!r}")
            continue
        result[record_id] = record
    return result


def candidate_label_counts(candidates: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(candidate["target_label"] for candidate in candidates).items()))


def validate(
    split_root: Path,
    config_path: Path | None = None,
    image_manifest_path: Path | None = None,
    candidate_manifest_path: Path | None = None,
    near_duplicates_path: Path | None = None,
) -> tuple[list[str], list[str]]:
    project_root = Path(__file__).resolve().parents[1]
    config_path = config_path or project_root / "configs" / "pilot_cv.json"
    image_manifest_path = (
        image_manifest_path or project_root / "data" / "manifests" / "batch_500.jsonl"
    )
    candidate_manifest_path = (
        candidate_manifest_path
        or project_root / "data" / "manifests" / "batch_500.candidates.jsonl"
    )
    near_duplicates_path = (
        near_duplicates_path
        or project_root / "data" / "manifests" / "batch_500.near_duplicates.json"
    )
    errors: list[str] = []
    warnings: list[str] = []
    protocol_path = split_root / "protocol.json"
    required_paths = (
        config_path,
        image_manifest_path,
        candidate_manifest_path,
        near_duplicates_path,
        protocol_path,
        split_root / "quarantine.jsonl",
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        return [f"missing required file: {path}" for path in missing], warnings

    config = json.loads(config_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    images = load_jsonl(image_manifest_path)
    candidates = load_jsonl(candidate_manifest_path)
    near_data = json.loads(near_duplicates_path.read_text(encoding="utf-8"))
    quarantine = load_jsonl(split_root / "quarantine.jsonl")
    image_by_id = records_by_id(images, "sample_id", "source images", errors)
    records_by_id(candidates, "candidate_id", "source candidates", errors)

    source_paths = {
        "config": config_path,
        "image_manifest": image_manifest_path,
        "candidate_manifest": candidate_manifest_path,
        "near_duplicates": near_duplicates_path,
    }
    for key, path in source_paths.items():
        recorded = protocol.get("source_files", {}).get(key, {})
        if recorded.get("sha256") != sha256_file(path):
            errors.append(f"protocol source hash mismatch: {key}")
        if recorded.get("path") != path.name:
            errors.append(f"protocol source filename mismatch: {key}")

    expected_classes = set(config["expected_classes"])
    if set(protocol.get("expected_classes", [])) != expected_classes:
        errors.append("protocol expected_classes do not match config")
    quarantine_policy = config.get("quarantine_policy", {})
    for field in (
        "exclude_all_images_in_exact_duplicate_groups",
        "exclude_all_images_in_cross_seed_dhash_pairs",
        "near_duplicates_are_candidates_not_proven_duplicates",
    ):
        if quarantine_policy.get(field) is not True:
            errors.append(f"mandatory quarantine policy must remain {field}=True")
    threshold_value = quarantine_policy.get("dhash_hamming_threshold")
    if type(threshold_value) is not int or threshold_value != 4:
        errors.append("mandatory quarantine policy must remain dhash_hamming_threshold=4")
    config_reporting = config.get("reporting_policy", {})
    required_reporting = {
        "primary_metric_aggregation": "pooled_out_of_fold_predictions",
        "per_fold_metrics_require_nonzero_class_support": True,
        "average_of_per_fold_macro_f1_is_primary": False,
        "formal_generalization_claim_allowed": False,
        "scope_label": "five-seed leakage-controlled pilot study",
    }
    for field, required in required_reporting.items():
        actual = config_reporting.get(field)
        valid = actual is required if isinstance(required, bool) else actual == required
        if not valid:
            errors.append(f"mandatory reporting policy must remain {field}={required!r}")
    threshold = threshold_value if type(threshold_value) is int else 4
    if int(near_data.get("hamming_threshold", -1)) != threshold:
        errors.append("near-duplicate threshold does not match config")

    computed_near_pairs = cross_seed_dhash_pairs(images, threshold)
    reported_near_pairs: set[tuple[str, str]] = set()
    for pair in near_data.get("pairs", []):
        left = image_by_id.get(pair.get("left"))
        right = image_by_id.get(pair.get("right"))
        if left is None or right is None:
            errors.append("near-duplicate report references an unknown sample")
            continue
        derived_cross_seed = left["scene_seed"] != right["scene_seed"]
        distance = (int(left["dhash64"], 16) ^ int(right["dhash64"], 16)).bit_count()
        if bool(pair.get("cross_seed")) != derived_cross_seed:
            errors.append("near-duplicate cross_seed flag is incorrect")
        if int(pair.get("hamming_distance", -1)) != distance:
            errors.append("near-duplicate Hamming distance is incorrect")
        if derived_cross_seed:
            reported_near_pairs.add(tuple(sorted((left["sample_id"], right["sample_id"]))))
    if reported_near_pairs != computed_near_pairs:
        errors.append("near-duplicate report differs from independent dHash recomputation")

    expected_quarantine_reasons: dict[str, set[str]] = defaultdict(set)
    for sample_id in exact_duplicate_ids(images):
        expected_quarantine_reasons[sample_id].add("exact_duplicate_group")
    for left_id, right_id in computed_near_pairs:
        expected_quarantine_reasons[left_id].add("cross_seed_dhash_candidate")
        expected_quarantine_reasons[right_id].add("cross_seed_dhash_candidate")
    quarantine_by_id = records_by_id(quarantine, "sample_id", "quarantine", errors)
    if set(quarantine_by_id) != set(expected_quarantine_reasons):
        errors.append("quarantine sample IDs do not equal independent source recomputation")
    for sample_id, reasons in expected_quarantine_reasons.items():
        record = quarantine_by_id.get(sample_id)
        source = image_by_id.get(sample_id)
        if record is None or source is None:
            continue
        if set(record.get("reasons", [])) != reasons:
            errors.append(f"{sample_id}: quarantine reasons are incorrect")
        for field in ("scene_seed", "image", "image_sha256", "exact_duplicate_group"):
            if record.get(field) != source.get(field):
                errors.append(f"{sample_id}: quarantine {field} does not match source")
        if record.get("excluded_from_all_folds") is not True:
            errors.append(f"{sample_id}: quarantine exclusion flag is not true")
    quarantine_ids = set(expected_quarantine_reasons)
    eligible_ids = set(image_by_id) - quarantine_ids

    linked_fields = (
        "image",
        "image_sha256",
        "scene_seed",
        "image_root_id",
        "width",
        "height",
        "frame_id",
        "view_id",
    )
    source_candidates_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        source_id = candidate.get("source_sample_id")
        source = image_by_id.get(source_id)
        if source is None:
            errors.append(f"{candidate.get('candidate_id')}: unknown source image {source_id}")
            continue
        for field in linked_fields:
            if candidate.get(field) != source.get(field):
                errors.append(
                    f"{candidate.get('candidate_id')}: {field} does not match source image"
                )
        source_candidates_by_image[source_id].append(candidate)

    configured_seeds = set(config["scene_seeds"])
    fold_configs = {fold["fold_id"]: fold for fold in config["folds"]}
    if len(fold_configs) != len(config["folds"]):
        errors.append("config contains duplicate fold_id values")
    protocol_folds = protocol.get("folds", [])
    protocol_fold_ids = [fold.get("fold_id") for fold in protocol_folds]
    if len(protocol_folds) != len(config["folds"]):
        errors.append("protocol fold count does not match config")
    if len(protocol_fold_ids) != len(set(protocol_fold_ids)):
        errors.append("protocol contains duplicate fold_id values")
    if set(fold_configs) != set(protocol_fold_ids):
        errors.append("protocol folds do not match config folds")
    protocol_fold_by_id: dict[str, dict[str, Any]] = {}
    for fold in protocol_folds:
        fold_id = fold.get("fold_id")
        if fold_id not in protocol_fold_by_id:
            protocol_fold_by_id[fold_id] = fold
    test_occurrences: Counter[str] = Counter()
    validation_occurrences: Counter[str] = Counter()
    train_occurrences: Counter[str] = Counter()
    pooled_test_labels: Counter[str] = Counter()
    for fold_id, fold_config in fold_configs.items():
        fold_root = split_root / fold_id
        test_seed = int(fold_config["test_seed"])
        validation_seed = int(fold_config["validation_seed"])
        train_seeds = sorted(configured_seeds - {test_seed, validation_seed})
        seed_assignment = {
            **{seed: "train" for seed in train_seeds},
            validation_seed: "validation",
            test_seed: "test",
        }
        actual_images_by_split: dict[str, list[dict[str, Any]]] = {}
        actual_candidates_by_split: dict[str, list[dict[str, Any]]] = {}
        for split in ("train", "validation", "test"):
            image_path = fold_root / f"{split}_images.jsonl"
            candidate_path = fold_root / f"{split}_candidates.jsonl"
            if not image_path.is_file() or not candidate_path.is_file():
                errors.append(f"{fold_id}/{split}: missing split file")
                actual_images_by_split[split] = []
                actual_candidates_by_split[split] = []
                continue
            actual_images = load_jsonl(image_path)
            actual_candidates = load_jsonl(candidate_path)
            actual_images_by_split[split] = actual_images
            actual_candidates_by_split[split] = actual_candidates
            actual_images_by_id = records_by_id(
                actual_images, "sample_id", f"{fold_id}/{split} images", errors
            )
            actual_candidates_by_id = records_by_id(
                actual_candidates,
                "candidate_id",
                f"{fold_id}/{split} candidates",
                errors,
            )
            expected_source_images = {
                sample_id: source
                for sample_id, source in image_by_id.items()
                if sample_id in eligible_ids and seed_assignment[source["scene_seed"]] == split
            }
            expected_images = {
                sample_id: assigned_record(source, config["protocol_id"], fold_id, split)
                for sample_id, source in expected_source_images.items()
            }
            expected_candidates = {
                candidate["candidate_id"]: assigned_record(
                    candidate, config["protocol_id"], fold_id, split
                )
                for sample_id in expected_source_images
                for candidate in source_candidates_by_image[sample_id]
            }
            if actual_images_by_id != expected_images:
                errors.append(f"{fold_id}/{split}: image records differ from source assignment")
            if actual_candidates_by_id != expected_candidates:
                errors.append(f"{fold_id}/{split}: candidate records differ from source assignment")
            if set(actual_images_by_id) & quarantine_ids:
                errors.append(f"{fold_id}/{split}: quarantined image included")
            if split == "test":
                test_occurrences.update(actual_images_by_id.keys())
                pooled_test_labels.update(
                    candidate["target_label"] for candidate in actual_candidates
                )
            elif split == "validation":
                validation_occurrences.update(actual_images_by_id.keys())
            else:
                train_occurrences.update(actual_images_by_id.keys())

        expected_support = {
            split: candidate_label_counts(actual_candidates_by_split[split])
            for split in ("train", "validation", "test")
        }
        missing_test_classes = sorted(expected_classes - set(expected_support["test"]))
        recomputed_metadata = {
            "schema_version": 1,
            "protocol_id": config["protocol_id"],
            "fold_id": fold_id,
            "train_seeds": train_seeds,
            "validation_seed": validation_seed,
            "test_seed": test_seed,
            "image_counts": {
                split: len(actual_images_by_split[split])
                for split in ("train", "validation", "test")
            },
            "candidate_counts": {
                split: len(actual_candidates_by_split[split])
                for split in ("train", "validation", "test")
            },
            "candidate_label_counts": expected_support,
            "missing_test_classes": missing_test_classes,
            "per_fold_four_class_macro_f1_valid": not missing_test_classes,
            "primary_aggregation": "pooled_out_of_fold_predictions",
        }
        metadata_path = fold_root / "metadata.json"
        if not metadata_path.is_file():
            errors.append(f"{fold_id}: missing metadata.json")
        else:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata != recomputed_metadata:
                errors.append(f"{fold_id}: metadata differs from recomputed support and counts")
        protocol_fold = protocol_fold_by_id.get(fold_id)
        if protocol_fold != recomputed_metadata:
            errors.append(f"{fold_id}: protocol fold summary differs from recomputation")
        if missing_test_classes:
            warnings.append(
                f"{fold_id}: test support is missing {missing_test_classes}; do not report "
                "a four-class fold Macro-F1"
            )

    for sample_id in sorted(eligible_ids):
        if test_occurrences[sample_id] != 1:
            errors.append(f"{sample_id}: expected 1 test occurrence, got {test_occurrences[sample_id]}")
        if validation_occurrences[sample_id] != 1:
            errors.append(
                f"{sample_id}: expected 1 validation occurrence, got {validation_occurrences[sample_id]}"
            )
        if train_occurrences[sample_id] != 3:
            errors.append(f"{sample_id}: expected 3 train occurrences, got {train_occurrences[sample_id]}")

    eligible_candidates = [
        candidate
        for candidate in candidates
        if candidate.get("source_sample_id") in eligible_ids
    ]
    expected_pooled_support = candidate_label_counts(eligible_candidates)
    if dict(sorted(pooled_test_labels.items())) != expected_pooled_support:
        errors.append("pooled test candidates do not equal eligible source candidates")
    if protocol.get("pooled_out_of_fold_class_support") != expected_pooled_support:
        errors.append("protocol pooled class support differs from source recomputation")
    if set(expected_pooled_support) != expected_classes:
        errors.append("pooled out-of-fold support does not contain all configured classes")
    if protocol.get("pooled_out_of_fold_all_expected_classes_present") is not True:
        errors.append("protocol incorrectly marks pooled expected-class support")

    expected_reason_counts = dict(
        sorted(
            Counter(
                reason
                for reasons in expected_quarantine_reasons.values()
                for reason in reasons
            ).items()
        )
    )
    scalar_checks = {
        "total_image_count": len(images),
        "eligible_image_count": len(eligible_ids),
        "quarantined_image_count": len(quarantine_ids),
        "eligible_candidate_count": len(eligible_candidates),
        "quarantine_reason_counts": expected_reason_counts,
    }
    for field, expected in scalar_checks.items():
        if protocol.get(field) != expected:
            errors.append(f"protocol {field} differs from source recomputation")

    reporting = protocol.get("reporting_policy", {})
    if reporting != config["reporting_policy"]:
        errors.append("protocol reporting_policy differs from config")
    if reporting.get("primary_metric_aggregation") != "pooled_out_of_fold_predictions":
        errors.append("primary metric aggregation must be pooled out-of-fold predictions")
    if reporting.get("per_fold_metrics_require_nonzero_class_support") is not True:
        errors.append("mandatory per-fold metrics must require nonzero class support")
    if reporting.get("average_of_per_fold_macro_f1_is_primary") is not False:
        errors.append("average per-fold Macro-F1 must not be primary")
    if reporting.get("formal_generalization_claim_allowed") is not False:
        errors.append("formal generalization claims must remain disallowed")
    if protocol.get("scope_label") != "five-seed leakage-controlled pilot study":
        errors.append("protocol scope label is not the required pilot label")

    warnings.append("pilot protocol only: formal generalization claims remain disallowed")
    return errors, warnings


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-root",
        type=Path,
        default=project_root / "data" / "splits" / "pilot_v1",
    )
    parser.add_argument("--config", type=Path, default=project_root / "configs" / "pilot_cv.json")
    parser.add_argument(
        "--image-manifest",
        type=Path,
        default=project_root / "data" / "manifests" / "batch_500.jsonl",
    )
    parser.add_argument(
        "--candidate-manifest",
        type=Path,
        default=project_root / "data" / "manifests" / "batch_500.candidates.jsonl",
    )
    parser.add_argument(
        "--near-duplicates",
        type=Path,
        default=project_root / "data" / "manifests" / "batch_500.near_duplicates.json",
    )
    args = parser.parse_args()
    errors, warnings = validate(
        args.split_root.resolve(),
        args.config.resolve(),
        args.image_manifest.resolve(),
        args.candidate_manifest.resolve(),
        args.near_duplicates.resolve(),
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
