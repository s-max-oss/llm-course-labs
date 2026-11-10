#!/usr/bin/env python3
"""Independently reconstruct and validate frozen SmolVLM B2 queues."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


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
)


def validate(
    protocol_path: Path,
    lock_path: Path,
    config_path: Path,
    split_root: Path,
    jobs_root: Path,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings = ["B2 uses offline oracle crops and is not deployable end-to-end"]
    try:
        protocol, _ = load_frozen_b2_protocol(PROJECT_ROOT, protocol_path, lock_path)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        validate_baseline_policy(config)
        manifest_path = jobs_root / "protocol.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        pilot = json.loads((split_root / "protocol.json").read_text(encoding="utf-8"))
    except (BaselineContractError, OSError, KeyError, json.JSONDecodeError) as exc:
        return [str(exc)], warnings

    if manifest.get("protocol_id") != protocol["protocol_id"]:
        errors.append("B2 jobs manifest protocol ID differs from frozen protocol")
    if manifest.get("ground_truth_available_to_inference") is not False:
        errors.append("B2 jobs manifest violates the ground-truth boundary")
    if manifest.get("deployment_eligible") is not False:
        errors.append("B2 jobs manifest incorrectly claims deployment eligibility")
    if sha256_file(jobs_root / "protocol.json") != B2_JOBS_MANIFEST_SHA256:
        errors.append("B2 jobs manifest differs from frozen implementation artifact")

    sources = {
        "machine_protocol": protocol_path,
        "protocol_lock": lock_path,
        "baseline_config": config_path,
        "pilot_protocol": split_root / "protocol.json",
    }
    for fold in pilot.get("folds", []):
        fold_id = fold["fold_id"]
        sources[f"{fold_id}_test_candidates"] = split_root / fold_id / "test_candidates.jsonl"
    for name, path in sources.items():
        recorded = manifest.get("source_files", {}).get(name, {})
        if not path.is_file() or recorded.get("sha256") != sha256_file(path):
            errors.append(f"B2 source hash mismatch: {name}")

    candidates = []
    seen: set[str] = set()
    for fold in pilot.get("folds", []):
        for candidate in load_jsonl(split_root / fold["fold_id"] / "test_candidates.jsonl"):
            if candidate["candidate_id"] in seen:
                errors.append(f"duplicate B2 source candidate: {candidate['candidate_id']}")
            seen.add(candidate["candidate_id"])
            candidates.append(candidate)
    expected = {"development": [], "locked_evaluation": []}
    for candidate in candidates:
        partition = expected_partition(protocol, int(candidate["scene_seed"]))
        if partition is None:
            errors.append(f"unassigned B2 candidate: {candidate['candidate_id']}")
            continue
        expected[partition].append(b2_job(candidate, protocol))
    for jobs in expected.values():
        jobs.sort(key=lambda item: item["candidate_id"])

    expected_smoke_ids = []
    for label in config["expected_labels"]:
        ids = sorted(
            item["candidate_id"]
            for item in candidates
            if item["target_label"] == label
            and expected_partition(protocol, int(item["scene_seed"])) == "development"
        )
        expected_smoke_ids.extend(ids[:2])
    expected_smoke_by_id = {item["candidate_id"]: item for item in expected["development"]}
    expected_smoke = [expected_smoke_by_id[item] for item in expected_smoke_ids]
    expected_queues = {
        "development": expected["development"],
        "development_smoke": expected_smoke,
        "locked_evaluation": expected["locked_evaluation"],
    }
    forbidden = set(config["inference_forbidden_fields"])
    for name, expected_jobs in expected_queues.items():
        path = jobs_root / f"{name}.jsonl"
        if not path.is_file():
            errors.append(f"missing B2 queue: {name}")
            continue
        actual = load_jsonl(path)
        partition = "development" if name == "development_smoke" else name
        try:
            validate_b2_jobs(actual, protocol, forbidden, partition)
        except BaselineContractError as exc:
            errors.append(str(exc))
        if actual != expected_jobs:
            errors.append(f"{name}: B2 jobs differ from sealed sources")
        queue = manifest.get("queues", {}).get(name, {})
        if queue.get("count") != len(actual) or queue.get("sha256") != sha256_file(path):
            errors.append(f"{name}: B2 queue manifest mismatch")
        if queue != B2_QUEUE_ARTIFACTS[name]:
            errors.append(f"{name}: queue differs from frozen implementation artifact")
    if manifest.get("smoke_candidate_ids") != expected_smoke_ids:
        errors.append("B2 smoke selection differs from frozen rule")
    return errors, warnings


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
        "--jobs-root",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "pilot_v1" / "b2_jobs",
    )
    args = parser.parse_args()
    errors, warnings = validate(
        args.protocol.resolve(),
        args.lock.resolve(),
        args.config.resolve(),
        args.split_root.resolve(),
        args.jobs_root.resolve(),
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
