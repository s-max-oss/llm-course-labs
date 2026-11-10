#!/usr/bin/env python3
"""Offline evaluator for frozen exploratory SmolVLM B2 outputs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from robot_vlm_baselines.b2 import (  # noqa: E402
    B2_A_RESULT_FIELDS,
    B2_B_RESULT_FIELDS,
    B2_JOBS_MANIFEST_SHA256,
    B2_QUEUE_ARTIFACTS,
    B2_TRUSTED_DEVELOPMENT_GATE_SHA256,
    b2_job,
    expected_partition,
    load_frozen_b2_protocol,
    parse_b2_a_label,
    rank_candidate_scores,
    validate_b2_jobs,
    validate_development_gate_report,
)
from robot_vlm_baselines.contracts import (  # noqa: E402
    BaselineContractError,
    EXPECTED_LABELS,
    load_jsonl,
    sha256_file,
    validate_baseline_policy,
    write_json,
)
from robot_vlm_baselines.metrics import classification_metrics  # noqa: E402


VARIANT_KEYS = {
    "b2_a": "b2_a_strict_natural_generation",
    "b2_b": "b2_b_fixed_candidate_scoring",
}


def _report_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _records_by_id(records: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        identifier = record.get(field)
        if not isinstance(identifier, str) or not identifier or identifier in result:
            raise BaselineContractError(f"missing or duplicate {field}: {identifier!r}")
        result[identifier] = record
    return result


def _load_partition_truth(
    split_root: Path,
    protocol: dict[str, Any],
    partition: str,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    pilot = json.loads((split_root / "protocol.json").read_text(encoding="utf-8"))
    if sha256_file(split_root / "protocol.json") != protocol["parent_artifacts"]["pilot_protocol"]["sha256"]:
        raise BaselineContractError("B2 truth pilot protocol differs from frozen parent")
    truth: dict[str, dict[str, Any]] = {}
    sources: list[dict[str, str]] = []
    parent_jobs_path = PROJECT_ROOT / protocol["parent_artifacts"]["baseline_jobs_protocol"]["path"]
    parent_jobs = json.loads(parent_jobs_path.read_text(encoding="utf-8"))
    for fold in pilot.get("folds", []):
        path = split_root / fold["fold_id"] / "test_candidates.jsonl"
        source_key = f"{fold['fold_id']}_test_candidates"
        if parent_jobs.get("source_files", {}).get(source_key, {}).get("sha256") != sha256_file(path):
            raise BaselineContractError(f"sealed B2 truth hash mismatch: {source_key}")
        sources.append(
            {
                "fold_id": fold["fold_id"],
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(path),
            }
        )
        for candidate in load_jsonl(path):
            if expected_partition(protocol, int(candidate["scene_seed"])) != partition:
                continue
            candidate_id = candidate["candidate_id"]
            if candidate_id in truth:
                raise BaselineContractError(f"duplicate truth candidate: {candidate_id}")
            truth[candidate_id] = candidate
    expected = protocol["partition"][partition]
    support = dict(Counter(item["target_label"] for item in truth.values()))
    if len(truth) != expected["candidate_count"] or support != expected["class_support"]:
        raise BaselineContractError(f"{partition}: truth differs from frozen B2 protocol")
    return truth, sources


def evaluate(
    jobs_path: Path,
    result_path: Path,
    variant_short: str,
    partition: str,
    protocol_path: Path,
    lock_path: Path,
    config_path: Path,
    split_root: Path,
    development_gate: Path | None = None,
) -> dict[str, Any]:
    protocol, lock = load_frozen_b2_protocol(PROJECT_ROOT, protocol_path, lock_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_baseline_policy(config)
    variant = VARIANT_KEYS[variant_short]
    if partition == "locked_evaluation":
        if variant_short == "b2_a":
            raise BaselineContractError("frozen protocol prohibits B2-A on locked evaluation")
        if development_gate is None:
            raise BaselineContractError("locked evaluation requires a passing development gate report")
        if B2_TRUSTED_DEVELOPMENT_GATE_SHA256 is None:
            raise BaselineContractError(
                "locked evaluation is not authorized: no trusted development gate digest"
            )
        validate_development_gate_report(
            development_gate,
            protocol,
            lock["sealed_files"]["machine_protocol"]["sha256"],
            PROJECT_ROOT,
            B2_TRUSTED_DEVELOPMENT_GATE_SHA256,
        )

    jobs = load_jsonl(jobs_path)
    validate_b2_jobs(jobs, protocol, set(config["inference_forbidden_fields"]), partition)
    jobs_manifest_path = jobs_path.parent / "protocol.json"
    if not jobs_manifest_path.is_file():
        raise BaselineContractError(f"missing B2 jobs manifest: {jobs_manifest_path}")
    jobs_manifest = json.loads(jobs_manifest_path.read_text(encoding="utf-8"))
    if sha256_file(jobs_manifest_path) != B2_JOBS_MANIFEST_SHA256:
        raise BaselineContractError("B2 jobs manifest differs from frozen implementation artifact")
    if jobs_manifest.get("protocol_id") != protocol["protocol_id"]:
        raise BaselineContractError("B2 jobs manifest uses another protocol")
    queue = jobs_manifest.get("queues", {}).get(
        "development" if partition == "development" else "locked_evaluation", {}
    )
    if queue.get("path") != jobs_path.name or queue.get("sha256") != sha256_file(jobs_path):
        raise BaselineContractError("B2 jobs file differs from its registered queue")
    if queue.get("count") != len(jobs):
        raise BaselineContractError("B2 jobs count differs from its registered queue")
    if queue != B2_QUEUE_ARTIFACTS[partition]:
        raise BaselineContractError("B2 jobs differ from frozen implementation artifact")
    for name, path in {
        "machine_protocol": protocol_path,
        "protocol_lock": lock_path,
        "baseline_config": config_path,
        "pilot_protocol": split_root / "protocol.json",
    }.items():
        if jobs_manifest.get("source_files", {}).get(name, {}).get("sha256") != sha256_file(path):
            raise BaselineContractError(f"B2 jobs manifest source hash mismatch: {name}")
    job_by_candidate = _records_by_id(jobs, "candidate_id")
    truth, truth_sources = _load_partition_truth(split_root, protocol, partition)
    if set(job_by_candidate) != set(truth):
        raise BaselineContractError("B2 jobs do not exactly cover the offline truth partition")

    results = load_jsonl(result_path)
    result_by_candidate = _records_by_id(results, "candidate_id")
    if set(result_by_candidate) != set(truth):
        missing = len(set(truth) - set(result_by_candidate))
        extra = len(set(result_by_candidate) - set(truth))
        raise BaselineContractError(f"incomplete B2 results: missing={missing} extra={extra}")

    result_fields = B2_A_RESULT_FIELDS if variant_short == "b2_a" else B2_B_RESULT_FIELDS
    natural_order = [
        value.strip()
        for value in protocol["variants"]["b2_b_fixed_candidate_scoring"][
            "candidate_completions_in_tie_order"
        ]
    ]
    pairs: list[tuple[str, str | None]] = []
    latencies: list[float] = []
    for candidate_id, truth_record in truth.items():
        job = job_by_candidate[candidate_id]
        if job != b2_job(truth_record, protocol):
            raise BaselineContractError(f"{candidate_id}: B2 job differs from sealed truth source")
        result = result_by_candidate[candidate_id]
        if set(result) != result_fields:
            raise BaselineContractError(f"{candidate_id}: B2 result fields differ from contract")
        checks = {
            "schema_version": 1,
            "b2_protocol_id": protocol["protocol_id"],
            "variant": variant,
            "job_id": job["job_id"],
            "partition": partition,
            "source_sample_id": job["source_sample_id"],
            "image_sha256": job["image_sha256"],
            "input_variant": job["input_variant"],
            "proposal_source": job["proposal_source"],
            "deployment_eligible": False,
        }
        for field, expected in checks.items():
            if result.get(field) != expected:
                raise BaselineContractError(f"{candidate_id}: result {field} differs from job")
        latency = result["latency_ms"]
        if (
            isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or not math.isfinite(float(latency))
            or latency < 0
        ):
            raise BaselineContractError(f"{candidate_id}: invalid latency")
        latencies.append(float(latency))
        status = result["status"]
        prediction = result["predicted_label"]
        natural = result["predicted_natural_label"]
        error = result["error"]
        if status == "error":
            if prediction is not None or natural is not None or not isinstance(error, str) or not error:
                raise BaselineContractError(f"{candidate_id}: inconsistent B2 error result")
            if variant_short == "b2_a":
                if result["raw_response"] != "":
                    raise BaselineContractError(f"{candidate_id}: error response must be empty")
            elif any(
                (
                    result["candidate_scores"] != {},
                    result["top1_natural_label"] is not None,
                    result["top2_natural_label"] is not None,
                    result["margin"] is not None,
                )
            ):
                raise BaselineContractError(f"{candidate_id}: inconsistent B2-B error scores")
            scored_prediction = None
        elif variant_short == "b2_a":
            parsed = parse_b2_a_label(result["raw_response"], protocol)
            if status == "ok":
                if parsed is None or parsed != (natural, prediction) or error is not None:
                    raise BaselineContractError(f"{candidate_id}: inconsistent B2-A ok result")
                scored_prediction = prediction
            elif status == "invalid_output":
                if parsed is not None or prediction is not None or natural is not None or error is not None:
                    raise BaselineContractError(f"{candidate_id}: inconsistent B2-A invalid output")
                scored_prediction = None
            else:
                raise BaselineContractError(f"{candidate_id}: invalid B2-A status")
        else:
            if status != "ok" or error is not None:
                raise BaselineContractError(f"{candidate_id}: invalid B2-B status")
            scores = result["candidate_scores"]
            if not isinstance(scores, dict):
                raise BaselineContractError(f"{candidate_id}: candidate_scores must be an object")
            try:
                numeric_scores = {key: float(value) for key, value in scores.items()}
                top1, top2, margin = rank_candidate_scores(numeric_scores, natural_order)
            except (TypeError, ValueError) as exc:
                raise BaselineContractError(f"{candidate_id}: invalid candidate scores") from exc
            if (
                natural != top1
                or prediction != protocol["label_mapping"][top1]
                or result["top1_natural_label"] != top1
                or result["top2_natural_label"] != top2
                or not math.isclose(float(result["margin"]), margin, rel_tol=1e-12, abs_tol=1e-12)
            ):
                raise BaselineContractError(f"{candidate_id}: inconsistent B2-B ranking")
            scored_prediction = prediction
        pairs.append((truth_record["target_label"], scored_prediction))

    metrics = classification_metrics(pairs, latencies)
    if metrics["macro_f1"] is None:
        raise BaselineContractError("B2 partition unexpectedly lacks a frozen class")
    report: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "variant": variant,
        "partition": partition,
        "scope_label": protocol["scope_label"],
        "confirmatory_claim_allowed": False,
        "deployment_eligible": False,
        "source_files": {
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
            "jobs": {
                "path": _report_path(jobs_path),
                "sha256": sha256_file(jobs_path),
            },
            "jobs_manifest": {
                "path": _report_path(jobs_manifest_path),
                "sha256": sha256_file(jobs_manifest_path),
            },
            "results": {
                "path": _report_path(result_path),
                "sha256": sha256_file(result_path),
            },
            "ground_truth_candidate_files": truth_sources,
        },
        "metrics": metrics,
        "limitations": [
            "offline oracle-proposal crop diagnostic; not deployable end-to-end",
            "legacy five-seed pilot; exploratory only, with no formal generalization claim",
        ],
    }
    if partition == "development" and variant_short == "b2_b":
        gates = protocol["development_gates_for_b2_b"]
        red_box_recall = metrics["per_class"]["distractor_red_box"]["recall"]
        checks = {
            "prediction_coverage": {
                "value": metrics["prediction_coverage"],
                "minimum": gates["prediction_coverage_min"],
                "passed": metrics["prediction_coverage"] >= gates["prediction_coverage_min"],
            },
            "macro_f1": {
                "value": metrics["macro_f1"],
                "minimum": gates["macro_f1_min"],
                "passed": metrics["macro_f1"] >= gates["macro_f1_min"],
            },
            "danger_red_sphere_recall": {
                "value": metrics["danger_binary"]["recall"],
                "minimum": gates["danger_red_sphere_recall_min"],
                "passed": metrics["danger_binary"]["recall"] >= gates["danger_red_sphere_recall_min"],
            },
            "distractor_red_box_recall": {
                "value": red_box_recall,
                "minimum": gates["distractor_red_box_recall_min"],
                "passed": red_box_recall >= gates["distractor_red_box_recall_min"],
            },
        }
        report["development_gates"] = {
            "checks": checks,
            "passed": all(item["passed"] for item in checks.values()),
            "on_failure": gates["on_failure"],
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--variant", choices=tuple(VARIANT_KEYS), required=True)
    parser.add_argument("--partition", choices=("development", "locked_evaluation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "configs" / "smolvlm_b2_protocol.json")
    parser.add_argument(
        "--lock", type=Path,
        default=PROJECT_ROOT / "experiments" / "pilot_v1" / "smolvlm_b2_protocol.lock.json",
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "baselines.json")
    parser.add_argument("--split-root", type=Path, default=PROJECT_ROOT / "data" / "splits" / "pilot_v1")
    parser.add_argument("--development-gate", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        print(f"ERROR: refusing to overwrite existing output: {args.output}", file=sys.stderr)
        return 2
    try:
        report = evaluate(
            args.jobs.resolve(), args.results.resolve(), args.variant, args.partition,
            args.protocol.resolve(), args.lock.resolve(), args.config.resolve(),
            args.split_root.resolve(),
            args.development_gate.resolve() if args.development_gate else None,
        )
    except (BaselineContractError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    write_json(args.output.resolve(), report)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
