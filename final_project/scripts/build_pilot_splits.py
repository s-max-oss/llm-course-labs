#!/usr/bin/env python3
"""Build deterministic five-fold, seed-disjoint pilot splits."""

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


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    atomic_write(
        path,
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
    )


def label_counts(candidates: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(candidate["target_label"] for candidate in candidates)
    return dict(sorted(counts.items()))


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


def validate_candidate_links(
    images: list[dict[str, Any]], candidates: list[dict[str, Any]]
) -> None:
    image_by_id = {image["sample_id"]: image for image in images}
    if len(image_by_id) != len(images):
        raise ValueError("image manifest contains duplicate sample_id values")
    candidate_ids: set[str] = set()
    linked_fields = (
        ("image", "image"),
        ("image_sha256", "image_sha256"),
        ("scene_seed", "scene_seed"),
        ("image_root_id", "image_root_id"),
        ("width", "width"),
        ("height", "height"),
        ("frame_id", "frame_id"),
        ("view_id", "view_id"),
    )
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        if candidate_id in candidate_ids:
            raise ValueError(f"duplicate candidate_id: {candidate_id}")
        candidate_ids.add(candidate_id)
        source_id = candidate["source_sample_id"]
        source = image_by_id.get(source_id)
        if source is None:
            raise ValueError(f"{candidate_id}: unknown source_sample_id {source_id}")
        for candidate_field, image_field in linked_fields:
            if candidate.get(candidate_field) != source.get(image_field):
                raise ValueError(
                    f"{candidate_id}: {candidate_field} does not match source image {source_id}"
                )


def validate_fold_plan(config: dict[str, Any]) -> None:
    seeds = set(config["scene_seeds"])
    folds = config["folds"]
    fold_ids = [fold["fold_id"] for fold in folds]
    test_seeds = [int(fold["test_seed"]) for fold in folds]
    validation_seeds = [int(fold["validation_seed"]) for fold in folds]
    if len(folds) != len(seeds):
        raise ValueError("fold count must equal scene seed count")
    if len(set(fold_ids)) != len(fold_ids):
        raise ValueError("fold_id values must be unique")
    if Counter(test_seeds) != Counter({seed: 1 for seed in seeds}):
        raise ValueError("every scene seed must be test exactly once")
    if Counter(validation_seeds) != Counter({seed: 1 for seed in seeds}):
        raise ValueError("every scene seed must be validation exactly once")
    for fold in folds:
        if int(fold["test_seed"]) == int(fold["validation_seed"]):
            raise ValueError(f"{fold['fold_id']}: test and validation seeds are identical")


def validate_mandatory_policies(config: dict[str, Any]) -> None:
    quarantine = config.get("quarantine_policy", {})
    required_quarantine = {
        "exclude_all_images_in_exact_duplicate_groups": True,
        "exclude_all_images_in_cross_seed_dhash_pairs": True,
        "near_duplicates_are_candidates_not_proven_duplicates": True,
    }
    for field, required in required_quarantine.items():
        if quarantine.get(field) is not required:
            raise ValueError(f"mandatory quarantine policy must remain {field}={required}")
    if type(quarantine.get("dhash_hamming_threshold")) is not int or quarantine.get(
        "dhash_hamming_threshold"
    ) != 4:
        raise ValueError("mandatory quarantine policy must remain dhash_hamming_threshold=4")

    reporting = config.get("reporting_policy", {})
    required_reporting = {
        "primary_metric_aggregation": "pooled_out_of_fold_predictions",
        "per_fold_metrics_require_nonzero_class_support": True,
        "average_of_per_fold_macro_f1_is_primary": False,
        "formal_generalization_claim_allowed": False,
        "scope_label": "five-seed leakage-controlled pilot study",
    }
    for field, required in required_reporting.items():
        actual = reporting.get(field)
        valid = actual is required if isinstance(required, bool) else actual == required
        if not valid:
            raise ValueError(f"mandatory reporting policy must remain {field}={required!r}")


def with_assignment(record: dict[str, Any], protocol_id: str, fold_id: str, split: str):
    assigned = dict(record)
    assigned["source_split"] = record.get("split", "unassigned")
    assigned["split"] = split
    assigned["pilot_protocol_id"] = protocol_id
    assigned["fold_id"] = fold_id
    return assigned


def build(
    config_path: Path,
    image_manifest_path: Path,
    candidate_manifest_path: Path,
    near_duplicates_path: Path,
    output_root: Path,
) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    images = load_jsonl(image_manifest_path)
    candidates = load_jsonl(candidate_manifest_path)
    near_data = json.loads(near_duplicates_path.read_text(encoding="utf-8"))
    validate_fold_plan(config)
    validate_mandatory_policies(config)
    validate_candidate_links(images, candidates)
    configured_seeds = set(config["scene_seeds"])
    observed_seeds = {image["scene_seed"] for image in images}
    if observed_seeds != configured_seeds:
        raise ValueError(
            f"configured seeds {sorted(configured_seeds)} do not match manifest seeds "
            f"{sorted(observed_seeds)}"
        )

    threshold = int(config["quarantine_policy"]["dhash_hamming_threshold"])
    if int(near_data.get("hamming_threshold", -1)) != threshold:
        raise ValueError("near-duplicate report threshold does not match pilot config")
    computed_near_pairs = cross_seed_dhash_pairs(images, threshold)
    image_by_id = {image["sample_id"]: image for image in images}
    reported_near_pairs: set[tuple[str, str]] = set()
    for pair in near_data.get("pairs", []):
        left = image_by_id.get(pair["left"])
        right = image_by_id.get(pair["right"])
        if left is None or right is None:
            raise ValueError("near-duplicate report references an unknown sample")
        derived_cross_seed = left["scene_seed"] != right["scene_seed"]
        if bool(pair.get("cross_seed")) != derived_cross_seed:
            raise ValueError("near-duplicate report has an incorrect cross_seed flag")
        distance = (int(left["dhash64"], 16) ^ int(right["dhash64"], 16)).bit_count()
        if distance != int(pair["hamming_distance"]):
            raise ValueError("near-duplicate report has an incorrect Hamming distance")
        if derived_cross_seed:
            reported_near_pairs.add(tuple(sorted((pair["left"], pair["right"]))))
    if reported_near_pairs != computed_near_pairs:
        raise ValueError("near-duplicate report does not match independent manifest recomputation")

    quarantine_reasons: dict[str, set[str]] = defaultdict(set)
    for sample_id in exact_duplicate_ids(images):
        quarantine_reasons[sample_id].add("exact_duplicate_group")
    for left_id, right_id in computed_near_pairs:
        quarantine_reasons[left_id].add("cross_seed_dhash_candidate")
        quarantine_reasons[right_id].add("cross_seed_dhash_candidate")

    quarantine_ids = set(quarantine_reasons)
    eligible_images = [image for image in images if image["sample_id"] not in quarantine_ids]
    eligible_ids = {image["sample_id"] for image in eligible_images}
    candidates_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        candidates_by_source[candidate["source_sample_id"]].append(candidate)

    quarantine_records = []
    for sample_id in sorted(quarantine_ids):
        image = image_by_id[sample_id]
        quarantine_records.append(
            {
                "sample_id": sample_id,
                "scene_seed": image["scene_seed"],
                "image": image["image"],
                "image_sha256": image["image_sha256"],
                "exact_duplicate_group": image.get("exact_duplicate_group"),
                "reasons": sorted(quarantine_reasons[sample_id]),
                "excluded_from_all_folds": True,
            }
        )
    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_root / "quarantine.jsonl", quarantine_records)

    fold_summaries = []
    expected_classes = set(config["expected_classes"])
    for fold in config["folds"]:
        fold_id = fold["fold_id"]
        test_seed = int(fold["test_seed"])
        validation_seed = int(fold["validation_seed"])
        train_seeds = sorted(configured_seeds - {test_seed, validation_seed})
        seed_assignment = {
            **{seed: "train" for seed in train_seeds},
            validation_seed: "validation",
            test_seed: "test",
        }
        images_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
        candidates_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for image in eligible_images:
            split = seed_assignment[image["scene_seed"]]
            images_by_split[split].append(
                with_assignment(image, config["protocol_id"], fold_id, split)
            )
            for candidate in candidates_by_source[image["sample_id"]]:
                candidates_by_split[split].append(
                    with_assignment(candidate, config["protocol_id"], fold_id, split)
                )
        fold_root = output_root / fold_id
        for split in ("train", "validation", "test"):
            write_jsonl(fold_root / f"{split}_images.jsonl", images_by_split[split])
            write_jsonl(
                fold_root / f"{split}_candidates.jsonl", candidates_by_split[split]
            )
        test_support = label_counts(candidates_by_split["test"])
        missing_test_classes = sorted(expected_classes - set(test_support))
        metadata = {
            "schema_version": 1,
            "protocol_id": config["protocol_id"],
            "fold_id": fold_id,
            "train_seeds": train_seeds,
            "validation_seed": validation_seed,
            "test_seed": test_seed,
            "image_counts": {
                split: len(images_by_split[split])
                for split in ("train", "validation", "test")
            },
            "candidate_counts": {
                split: len(candidates_by_split[split])
                for split in ("train", "validation", "test")
            },
            "candidate_label_counts": {
                split: label_counts(candidates_by_split[split])
                for split in ("train", "validation", "test")
            },
            "missing_test_classes": missing_test_classes,
            "per_fold_four_class_macro_f1_valid": not missing_test_classes,
            "primary_aggregation": "pooled_out_of_fold_predictions",
        }
        write_json(fold_root / "metadata.json", metadata)
        fold_summaries.append(metadata)

    eligible_candidates = [
        candidate
        for candidate in candidates
        if candidate["source_sample_id"] in eligible_ids
    ]
    pooled_test_support = label_counts(eligible_candidates)
    protocol = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "scope_label": config["reporting_policy"]["scope_label"],
        "expected_classes": config["expected_classes"],
        "source_files": {
            "config": {"path": config_path.name, "sha256": sha256_file(config_path)},
            "image_manifest": {
                "path": image_manifest_path.name,
                "sha256": sha256_file(image_manifest_path),
            },
            "candidate_manifest": {
                "path": candidate_manifest_path.name,
                "sha256": sha256_file(candidate_manifest_path),
            },
            "near_duplicates": {
                "path": near_duplicates_path.name,
                "sha256": sha256_file(near_duplicates_path),
            },
        },
        "total_image_count": len(images),
        "eligible_image_count": len(eligible_images),
        "quarantined_image_count": len(quarantine_records),
        "quarantine_reason_counts": dict(
            sorted(
                Counter(
                    reason
                    for reasons in quarantine_reasons.values()
                    for reason in reasons
                ).items()
            )
        ),
        "eligible_candidate_count": len(eligible_candidates),
        "pooled_out_of_fold_class_support": pooled_test_support,
        "pooled_out_of_fold_all_expected_classes_present": (
            expected_classes == set(pooled_test_support)
        ),
        "reporting_policy": config["reporting_policy"],
        "folds": fold_summaries,
        "limitations": [
            f"{len(configured_seeds)} simulator scene seeds only",
            f"{len(quarantine_records)} images conservatively quarantined for exact or cross-seed dHash similarity",
            "dHash pairs are candidates and may include false positives",
            *[
                f"{fold['fold_id']} (seed {fold['test_seed']}) has no eligible test support for: "
                + ", ".join(fold["missing_test_classes"])
                for fold in fold_summaries
                if fold["missing_test_classes"]
            ],
            "results support a pilot study only, not a formal generalization claim",
        ],
    }
    protocol_path = output_root / "protocol.json"
    write_json(protocol_path, protocol)
    return protocol_path


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "data" / "splits" / "pilot_v1",
    )
    args = parser.parse_args()
    protocol = build(
        args.config.resolve(),
        args.image_manifest.resolve(),
        args.candidate_manifest.resolve(),
        args.near_duplicates.resolve(),
        args.output_root.resolve(),
    )
    print(protocol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
