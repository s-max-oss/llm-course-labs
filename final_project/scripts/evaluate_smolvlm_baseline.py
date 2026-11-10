#!/usr/bin/env python3
"""Offline-only evaluator for complete SmolVLM out-of-fold predictions."""

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

from robot_vlm_baselines.contracts import (  # noqa: E402
    BaselineContractError,
    EXPECTED_LABELS,
    find_forbidden_fields,
    load_jsonl,
    parse_strict_label,
    sha256_file,
    validate_baseline_policy,
    write_json,
)
from robot_vlm_baselines.metrics import classification_metrics  # noqa: E402


RESULT_FIELDS = {
    "schema_version",
    "job_id",
    "baseline_id",
    "fold_id",
    "candidate_id",
    "source_sample_id",
    "image_sha256",
    "input_variant",
    "proposal_source",
    "deployment_eligible",
    "status",
    "error",
    "latency_ms",
    "raw_response",
    "predicted_label",
}


def records_by_id(
    records: list[dict[str, Any]], field: str, label: str
) -> dict[str, dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    for record in records:
        identifier = record.get(field)
        if not identifier or identifier in mapped:
            raise BaselineContractError(f"{label}: missing or duplicate {field}: {identifier!r}")
        mapped[identifier] = record
    return mapped


def evaluate(
    result_paths: list[Path],
    split_root: Path,
    baseline_config_path: Path,
    jobs_protocol_path: Path | None = None,
) -> dict[str, Any]:
    config = json.loads(baseline_config_path.read_text(encoding="utf-8"))
    validate_baseline_policy(config)
    pilot_path = split_root / "protocol.json"
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    jobs_protocol_path = jobs_protocol_path or (
        PROJECT_ROOT / "experiments" / "pilot_v1" / "baseline_jobs" / "protocol.json"
    )
    if not jobs_protocol_path.is_file():
        raise BaselineContractError(f"missing sealed jobs protocol: {jobs_protocol_path}")
    jobs_protocol = json.loads(jobs_protocol_path.read_text(encoding="utf-8"))
    if tuple(config["expected_labels"]) != EXPECTED_LABELS:
        raise BaselineContractError("baseline label order differs from metric contract")
    if jobs_protocol.get("protocol_id") != pilot.get("protocol_id"):
        raise BaselineContractError("jobs protocol does not match pilot protocol")
    if jobs_protocol.get("ground_truth_available_to_inference") is not False:
        raise BaselineContractError("jobs protocol violates the inference truth boundary")
    sealed_sources = jobs_protocol.get("source_files", {})
    for key, path in {
        "baseline_config": baseline_config_path,
        "pilot_protocol": pilot_path,
    }.items():
        if sealed_sources.get(key, {}).get("sha256") != sha256_file(path):
            raise BaselineContractError(f"sealed source hash mismatch: {key}")
    results: list[dict[str, Any]] = []
    result_sources = []
    for path in result_paths:
        if not path.is_file():
            raise BaselineContractError(f"missing result file: {path}")
        result_sources.append({"path": path.name, "sha256": sha256_file(path)})
        results.extend(load_jsonl(path))
    by_candidate = records_by_id(results, "candidate_id", "SmolVLM results")
    if any(find_forbidden_fields(record, {"target_label"}) for record in results):
        raise BaselineContractError("prediction file contains target_label")

    expected: dict[str, dict[str, Any]] = {}
    fold_candidates: dict[str, list[dict[str, Any]]] = {}
    truth_sources = []
    folds = pilot.get("folds", [])
    fold_ids = [fold.get("fold_id") for fold in folds]
    if len(fold_ids) != 5 or len(set(fold_ids)) != 5:
        raise BaselineContractError("pilot protocol must contain five unique folds")
    job_fold_summaries = jobs_protocol.get("folds", [])
    job_fold_ids = [fold.get("fold_id") for fold in job_fold_summaries]
    if len(job_fold_ids) != 5 or set(job_fold_ids) != set(fold_ids):
        raise BaselineContractError("jobs protocol fold summaries differ from pilot")
    job_fold_by_id = {fold["fold_id"]: fold for fold in job_fold_summaries}
    for fold in pilot["folds"]:
        fold_id = fold["fold_id"]
        candidate_path = split_root / fold_id / "test_candidates.jsonl"
        source_key = f"{fold_id}_test_candidates"
        candidate_hash = sha256_file(candidate_path)
        sealed = sealed_sources.get(source_key, {})
        if sealed.get("path") != f"{fold_id}/test_candidates.jsonl":
            raise BaselineContractError(f"invalid sealed truth path: {source_key}")
        if sealed.get("sha256") != candidate_hash:
            raise BaselineContractError(f"sealed truth hash mismatch: {source_key}")
        candidates = load_jsonl(candidate_path)
        expected_count = fold.get("candidate_counts", {}).get("test")
        if len(candidates) != expected_count:
            raise BaselineContractError(f"{fold_id}: truth count differs from pilot protocol")
        if job_fold_by_id[fold_id].get("smolvlm_job_count") != len(candidates):
            raise BaselineContractError(
                f"{fold_id}: truth count differs from jobs protocol"
            )
        support = dict(
            sorted(Counter(item.get("target_label") for item in candidates).items())
        )
        if support != fold.get("candidate_label_counts", {}).get("test"):
            raise BaselineContractError(f"{fold_id}: truth support differs from pilot protocol")
        truth_sources.append(
            {"fold_id": fold_id, "path": sealed["path"], "sha256": candidate_hash}
        )
        fold_candidates[fold_id] = candidates
        for candidate in candidates:
            candidate_id = candidate["candidate_id"]
            if candidate_id in expected:
                raise BaselineContractError(
                    f"candidate appears in multiple test folds: {candidate_id}"
                )
            expected[candidate_id] = candidate
    pooled_support = dict(
        sorted(
            Counter(
                item["target_label"]
                for items in fold_candidates.values()
                for item in items
            ).items()
        )
    )
    if pooled_support != pilot.get("pooled_out_of_fold_class_support"):
        raise BaselineContractError("pooled truth support differs from pilot protocol")
    if len(expected) != pilot.get("eligible_candidate_count"):
        raise BaselineContractError("pooled truth count differs from pilot protocol")
    if set(by_candidate) != set(expected):
        missing = sorted(set(expected) - set(by_candidate))
        extra = sorted(set(by_candidate) - set(expected))
        raise BaselineContractError(
            f"incomplete out-of-fold predictions: missing={len(missing)} extra={len(extra)}"
        )

    baseline_id = config["smolvlm"]["baseline_id"]
    for candidate_id, result in by_candidate.items():
        source = expected[candidate_id]
        fields = set(result)
        if fields != RESULT_FIELDS:
            missing = sorted(RESULT_FIELDS - fields)
            unexpected = sorted(fields - RESULT_FIELDS)
            raise BaselineContractError(
                f"{candidate_id}: result schema differs from runner contract: "
                f"missing={missing} unexpected={unexpected}"
            )
        proposal_source = (
            "offline_annotation_oracle"
            if source["padded_crop_xyxy_pixels"] is not None
            else "offline_background_convention"
        )
        checks = {
            "schema_version": 1,
            "job_id": f"{baseline_id}::{source['fold_id']}::{candidate_id}",
            "baseline_id": baseline_id,
            "fold_id": source["fold_id"],
            "source_sample_id": source["source_sample_id"],
            "image_sha256": source["image_sha256"],
            "input_variant": config["smolvlm"]["input_variant"],
            "proposal_source": proposal_source,
            "deployment_eligible": False,
        }
        for field, value in checks.items():
            if result.get(field) != value:
                raise BaselineContractError(f"{candidate_id}: result {field} does not match job")
        latency = result["latency_ms"]
        if (
            isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or not math.isfinite(float(latency))
            or latency < 0
        ):
            raise BaselineContractError(f"{candidate_id}: invalid latency_ms")
        status = result["status"]
        prediction = result["predicted_label"]
        raw_response = result["raw_response"]
        error = result["error"]
        if not isinstance(raw_response, str):
            raise BaselineContractError(f"{candidate_id}: raw_response must be a string")
        if status == "ok":
            if prediction not in EXPECTED_LABELS:
                raise BaselineContractError(
                    f"{candidate_id}: ok result has invalid prediction {prediction!r}"
                )
            if error is not None or parse_strict_label(raw_response) != prediction:
                raise BaselineContractError(
                    f"{candidate_id}: ok result has inconsistent response/error fields"
                )
        elif status == "invalid_output":
            if prediction is not None or error is not None or parse_strict_label(raw_response):
                raise BaselineContractError(
                    f"{candidate_id}: invalid_output result is internally inconsistent"
                )
        elif status == "error":
            if (
                prediction is not None
                or raw_response != ""
                or not isinstance(error, str)
                or not error
            ):
                raise BaselineContractError(
                    f"{candidate_id}: error result is internally inconsistent"
                )
        else:
            raise BaselineContractError(f"{candidate_id}: unknown result status {status!r}")

    def score(candidates: list[dict[str, Any]]) -> dict[str, Any]:
        pairs = []
        latencies = []
        for candidate in candidates:
            result = by_candidate[candidate["candidate_id"]]
            prediction = (
                result["predicted_label"] if result["status"] == "ok" else None
            )
            pairs.append((candidate["target_label"], prediction))
            if isinstance(result.get("latency_ms"), (int, float)):
                latencies.append(float(result["latency_ms"]))
        return classification_metrics(pairs, latencies)

    folds = []
    for fold in pilot["folds"]:
        fold_id = fold["fold_id"]
        metrics = score(fold_candidates[fold_id])
        folds.append(
            {
                "fold_id": fold_id,
                "metrics": metrics,
                "four_class_macro_f1_reportable": metrics["macro_f1"] is not None,
            }
        )
    pooled_candidates = [
        candidate
        for fold in pilot["folds"]
        for candidate in fold_candidates[fold["fold_id"]]
    ]
    pooled = score(pooled_candidates)
    if pooled["macro_f1"] is None:
        raise BaselineContractError("pooled test support unexpectedly lacks an expected class")
    return {
        "schema_version": 1,
        "protocol_id": pilot["protocol_id"],
        "baseline_id": baseline_id,
        "scope_label": pilot["scope_label"],
        "formal_generalization_claim_allowed": False,
        "deployment_eligible": False,
        "primary_aggregation": "pooled_out_of_fold_predictions",
        "source_files": {
            "pilot_protocol": {"path": pilot_path.name, "sha256": sha256_file(pilot_path)},
            "baseline_config": {
                "path": baseline_config_path.name,
                "sha256": sha256_file(baseline_config_path),
            },
            "baseline_jobs_protocol": {
                "path": jobs_protocol_path.name,
                "sha256": sha256_file(jobs_protocol_path),
            },
            "ground_truth_candidate_files": truth_sources,
            "prediction_files": result_sources,
        },
        "pooled_out_of_fold": pooled,
        "folds": folds,
        "limitations": [
            "offline oracle-proposal crop diagnostic; not deployable end-to-end",
            "legacy missing-label frames use an uncertain background convention",
            "five-seed pilot only; no formal generalization claim",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--split-root", type=Path, default=PROJECT_ROOT / "data" / "splits" / "pilot_v1"
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "baselines.json"
    )
    parser.add_argument(
        "--jobs-protocol",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiments"
            / "pilot_v1"
            / "baseline_jobs"
            / "protocol.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        print(f"ERROR: refusing to overwrite existing output: {args.output}", file=sys.stderr)
        return 2
    try:
        report = evaluate(
            [path.resolve() for path in args.results],
            args.split_root.resolve(),
            args.config.resolve(),
            args.jobs_protocol.resolve(),
        )
    except BaselineContractError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    write_json(args.output.resolve(), report)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
