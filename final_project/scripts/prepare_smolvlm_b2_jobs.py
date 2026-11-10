#!/usr/bin/env python3
"""Build deterministic, label-free SmolVLM B2 inference queues."""

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

from robot_vlm_baselines.b2 import (  # noqa: E402
    B2_JOBS_MANIFEST_SHA256,
    B2_QUEUE_ARTIFACTS,
    b2_job,
    expected_partition,
    load_frozen_b2_protocol,
    validate_b2_jobs,
)
from robot_vlm_baselines.contracts import (  # noqa: E402
    BaselineContractError,
    load_jsonl,
    sha256_file,
    validate_baseline_policy,
    write_json,
    write_jsonl,
)


def load_unique_test_candidates(split_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pilot_path = split_root / "protocol.json"
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for fold in pilot.get("folds", []):
        path = split_root / fold["fold_id"] / "test_candidates.jsonl"
        for candidate in load_jsonl(path):
            candidate_id = candidate["candidate_id"]
            if candidate_id in seen:
                raise BaselineContractError(f"candidate appears in multiple test folds: {candidate_id}")
            seen.add(candidate_id)
            candidates.append(candidate)
    if len(pilot.get("folds", [])) != 5 or len(candidates) != 444:
        raise BaselineContractError("B2 requires the sealed 5-fold, 444-candidate pilot")
    return candidates, pilot


def build(
    protocol_path: Path,
    lock_path: Path,
    config_path: Path,
    split_root: Path,
    output_root: Path,
) -> Path:
    protocol, _ = load_frozen_b2_protocol(PROJECT_ROOT, protocol_path, lock_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_baseline_policy(config)
    candidates, pilot = load_unique_test_candidates(split_root)
    if protocol["parent_artifacts"]["pilot_protocol"]["sha256"] != sha256_file(
        split_root / "protocol.json"
    ):
        raise BaselineContractError("B2 pilot protocol differs from its frozen parent")

    by_partition: dict[str, list[dict[str, Any]]] = {
        "development": [],
        "locked_evaluation": [],
    }
    truth_support: dict[str, Counter[str]] = {
        "development": Counter(),
        "locked_evaluation": Counter(),
    }
    for candidate in candidates:
        partition = expected_partition(protocol, int(candidate["scene_seed"]))
        if partition is None:
            raise BaselineContractError(f"candidate has an unassigned seed: {candidate['candidate_id']}")
        by_partition[partition].append(b2_job(candidate, protocol))
        truth_support[partition][candidate["target_label"]] += 1

    forbidden = set(config["inference_forbidden_fields"])
    for partition, jobs in by_partition.items():
        jobs.sort(key=lambda item: item["candidate_id"])
        validate_b2_jobs(jobs, protocol, forbidden, partition)
        expected = protocol["partition"][partition]
        if len(jobs) != expected["candidate_count"]:
            raise BaselineContractError(f"{partition}: candidate count differs from protocol")
        if dict(truth_support[partition]) != expected["class_support"]:
            raise BaselineContractError(f"{partition}: offline class support differs from protocol")

    development_truth = {
        candidate["candidate_id"]: candidate["target_label"]
        for candidate in candidates
        if expected_partition(protocol, int(candidate["scene_seed"])) == "development"
    }
    smoke_ids: list[str] = []
    for label in config["expected_labels"]:
        ids = sorted(
            candidate_id
            for candidate_id, target in development_truth.items()
            if target == label
        )
        if len(ids) < 2:
            raise BaselineContractError(f"development partition lacks two smoke examples: {label}")
        smoke_ids.extend(ids[:2])
    smoke_set = set(smoke_ids)
    smoke = [job for job in by_partition["development"] if job["candidate_id"] in smoke_set]
    smoke.sort(key=lambda item: smoke_ids.index(item["candidate_id"]))
    if len(smoke) != protocol["smoke_test"]["candidate_count"]:
        raise BaselineContractError("B2 smoke selection count differs from protocol")

    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "development": output_root / "development.jsonl",
        "development_smoke": output_root / "development_smoke.jsonl",
        "locked_evaluation": output_root / "locked_evaluation.jsonl",
    }
    write_jsonl(paths["development"], by_partition["development"])
    write_jsonl(paths["development_smoke"], smoke)
    write_jsonl(paths["locked_evaluation"], by_partition["locked_evaluation"])

    source_files = {
        "machine_protocol": {
            "path": protocol_path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(protocol_path),
        },
        "protocol_lock": {
            "path": lock_path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(lock_path),
        },
        "baseline_config": {
            "path": config_path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(config_path),
        },
        "pilot_protocol": {
            "path": (split_root / "protocol.json").relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(split_root / "protocol.json"),
        },
    }
    for fold in pilot["folds"]:
        fold_id = fold["fold_id"]
        path = split_root / fold_id / "test_candidates.jsonl"
        source_files[f"{fold_id}_test_candidates"] = {
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(path),
        }
    queues = {
        name: {
            "path": path.name,
            "sha256": sha256_file(path),
            "count": len(load_jsonl(path)),
        }
        for name, path in paths.items()
    }
    manifest = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "artifact": "label_free_smolvlm_b2_inference_jobs",
        "ground_truth_available_to_inference": False,
        "deployment_eligible": False,
        "source_files": source_files,
        "queues": queues,
        "smoke_candidate_ids": smoke_ids,
    }
    manifest_path = output_root / "protocol.json"
    write_json(manifest_path, manifest)
    for name, expected in B2_QUEUE_ARTIFACTS.items():
        actual = queues[name]
        if actual != expected:
            raise BaselineContractError(f"{name}: generated queue differs from frozen implementation artifact")
    if sha256_file(manifest_path) != B2_JOBS_MANIFEST_SHA256:
        raise BaselineContractError("generated B2 jobs manifest differs from frozen implementation artifact")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", type=Path, default=PROJECT_ROOT / "configs" / "smolvlm_b2_protocol.json"
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "pilot_v1" / "smolvlm_b2_protocol.lock.json",
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "baselines.json"
    )
    parser.add_argument(
        "--split-root", type=Path, default=PROJECT_ROOT / "data" / "splits" / "pilot_v1"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "pilot_v1" / "b2_jobs",
    )
    args = parser.parse_args()
    try:
        path = build(
            args.protocol.resolve(),
            args.lock.resolve(),
            args.config.resolve(),
            args.split_root.resolve(),
            args.output_root.resolve(),
        )
    except (BaselineContractError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
