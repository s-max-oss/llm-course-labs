#!/usr/bin/env python3
"""Offline-only RF-only versus RF+B2 development feasibility evaluator."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for root in (SRC_ROOT, SCRIPT_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from evaluate_rfdetr_baseline import match_class  # noqa: E402
from robot_vlm_baselines.b2 import load_frozen_b2_protocol, rank_candidate_scores  # noqa: E402
from robot_vlm_baselines.cascade import (  # noqa: E402
    CASCADE_RESULT_FIELDS,
    build_cascade_jobs,
    load_cascade_protocol,
)
from robot_vlm_baselines.contracts import (  # noqa: E402
    BaselineContractError,
    load_jsonl,
    percentile,
    sha256_file,
    write_json,
)


TARGET_LABELS = (
    "danger_red_sphere",
    "distractor_red_box",
    "distractor_green_sphere",
)


def _report_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _prf(counts: Counter[str]) -> dict[str, Any]:
    tp, fp, fn = counts["true_positive"], counts["false_positive"], counts["false_negative"]
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _metrics(records: list[dict[str, Any]], truths: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    totals = {label: Counter() for label in TARGET_LABELS}
    for record in records:
        for label in TARGET_LABELS:
            gt = [item["bbox_xyxy_pixels"] for item in truths[record["sample_id"]] if item["target_label"] == label]
            predictions = [item for item in record["predictions"] if item["canonical_label"] == label]
            matched = match_class(gt, predictions, 0.5)
            for key in ("true_positive", "false_positive", "false_negative"):
                totals[label][key] += matched[key]
    per_class = {label: _prf(totals[label]) for label in TARGET_LABELS}
    return {
        "per_target_class": per_class,
        "three_target_class_macro_f1": sum(per_class[label]["f1"] for label in TARGET_LABELS) / 3,
        "target_false_positive_total": sum(per_class[label]["false_positive"] for label in TARGET_LABELS),
        "danger_recall": per_class["danger_red_sphere"]["recall"],
    }


def _load_truths() -> dict[str, list[dict[str, Any]]]:
    truths: dict[str, list[dict[str, Any]]] = defaultdict(list)
    jobs_protocol = json.loads(
        (PROJECT_ROOT / "experiments" / "pilot_v1" / "baseline_jobs" / "protocol.json").read_text(
            encoding="utf-8"
        )
    )
    for fold_id in ("fold_01", "fold_02"):
        path = PROJECT_ROOT / "data" / "splits" / "pilot_v1" / fold_id / "test_candidates.jsonl"
        key = f"{fold_id}_test_candidates"
        if jobs_protocol.get("source_files", {}).get(key, {}).get("sha256") != sha256_file(path):
            raise BaselineContractError(f"cascade evaluator truth hash mismatch: {key}")
        for item in load_jsonl(path):
            if item["bbox_xyxy_pixels"] is not None:
                truths[item["source_sample_id"]].append(item)
    return truths


def _apply(
    source_records: list[dict[str, Any]],
    jobs_by_id: dict[str, dict[str, Any]],
    results_by_id: dict[str, dict[str, Any]],
    policy: str,
) -> list[dict[str, Any]]:
    records = copy.deepcopy(source_records)
    by_key = {(record["fold_id"], record["sample_id"]): record for record in records}
    for job_id, job in jobs_by_id.items():
        result = results_by_id[job_id]
        if set(result) != CASCADE_RESULT_FIELDS:
            raise BaselineContractError(f"cascade B2 result fields differ from contract: {job_id}")
        if result["status"] != "ok":
            continue
        record = by_key[(job["fold_id"], job["sample_id"])]
        matches = [p for p in record["predictions"] if p["prediction_index"] == job["prediction_index"]]
        if len(matches) != 1:
            raise BaselineContractError(f"routed RF prediction is missing or duplicated: {job_id}")
        prediction = matches[0]
        vlm = result["predicted_label"]
        if policy == "primary":
            if prediction["canonical_label"] == "danger_red_sphere":
                continue
            if vlm in {"distractor_red_box", "distractor_green_sphere"}:
                prediction["canonical_label"] = vlm
        elif policy == "diagnostic":
            if vlm == "background_or_unknown":
                record["predictions"].remove(prediction)
            else:
                prediction["canonical_label"] = vlm
        else:
            raise BaselineContractError(f"unknown cascade policy: {policy}")
    return records


def evaluate(results_path: Path) -> dict[str, Any]:
    cascade_path = PROJECT_ROOT / "configs" / "cascade_feasibility_v1.json"
    b2_path = PROJECT_ROOT / "configs" / "smolvlm_b2_protocol.json"
    b2_lock = PROJECT_ROOT / "experiments" / "pilot_v1" / "smolvlm_b2_protocol.lock.json"
    protocol = load_cascade_protocol(PROJECT_ROOT, cascade_path)
    b2_protocol, _ = load_frozen_b2_protocol(PROJECT_ROOT, b2_path, b2_lock)
    jobs = build_cascade_jobs(PROJECT_ROOT, protocol)
    jobs_by_id = {item["job_id"]: item for item in jobs}
    results = load_jsonl(results_path)
    if len(results) != len(jobs) or len({item.get("job_id") for item in results}) != len(jobs):
        raise BaselineContractError("cascade B2 results do not exactly cover the frozen queue")
    results_by_id = {item["job_id"]: item for item in results}
    if set(results_by_id) != set(jobs_by_id):
        raise BaselineContractError("cascade B2 result identities differ from the frozen queue")
    completions = b2_protocol["variants"]["b2_b_fixed_candidate_scoring"][
        "candidate_completions_in_tie_order"
    ]
    order = [item.strip() for item in completions]
    latencies: list[float] = []
    for job_id, job in jobs_by_id.items():
        result = results_by_id[job_id]
        identity = (
            result.get("schema_version") == 1
            and result.get("protocol_id") == protocol["protocol_id"]
            and result.get("b2_protocol_id") == b2_protocol["protocol_id"]
            and result.get("variant") == "b2_b_fixed_candidate_scoring"
            and result.get("fold_id") == job["fold_id"]
            and result.get("sample_id") == job["sample_id"]
            and result.get("prediction_index") == job["prediction_index"]
            and result.get("image_sha256") == job["image_sha256"]
            and result.get("detector_label") == job["detector_label"]
            and result.get("detector_confidence") == job["detector_confidence"]
            and result.get("proposal_xyxy_pixels") == job["proposal_xyxy_pixels"]
            and result.get("input_variant") == job["input_variant"]
            and result.get("proposal_source") == job["proposal_source"]
        )
        if not identity or result.get("status") not in {"ok", "error"}:
            raise BaselineContractError(f"cascade B2 result contract mismatch: {job_id}")
        latency = result.get("latency_ms")
        if type(latency) not in {int, float} or not math.isfinite(latency) or latency < 0:
            raise BaselineContractError(f"invalid cascade B2 latency: {job_id}")
        latencies.append(float(latency))
        if result["status"] == "ok":
            top1, top2, margin = rank_candidate_scores(result["candidate_scores"], order)
            if (
                result.get("top1_natural_label") != top1
                or result.get("top2_natural_label") != top2
                or result.get("predicted_natural_label") != top1
                or result.get("margin") != margin
                or result.get("predicted_label") != b2_protocol["label_mapping"][top1]
                or result.get("error") is not None
            ):
                raise BaselineContractError(f"cascade B2 score ranking mismatch: {job_id}")
        elif (
            not isinstance(result.get("error"), str)
            or not result["error"]
            or result.get("predicted_natural_label") is not None
            or result.get("predicted_label") is not None
            or result.get("candidate_scores") != {}
            or result.get("top1_natural_label") is not None
            or result.get("top2_natural_label") is not None
            or result.get("margin") is not None
        ):
            raise BaselineContractError(f"errored cascade B2 result payload is inconsistent: {job_id}")

    source_records: list[dict[str, Any]] = []
    for key in ("fold_01_rfdetr_results", "fold_02_rfdetr_results"):
        source_records.extend(load_jsonl(PROJECT_ROOT / protocol["parent_artifacts"][key]["path"]))
    truths = _load_truths()
    baseline = _metrics(source_records, truths)
    frozen = protocol["development_baseline"]
    if (
        baseline["three_target_class_macro_f1"] != frozen["three_target_macro_f1"]
        or baseline["target_false_positive_total"] != frozen["target_false_positive_total"]
        or baseline["per_target_class"]["danger_red_sphere"]["true_positive"]
        != frozen["danger_counts"]["true_positive"]
    ):
        raise BaselineContractError("recomputed RF-only development baseline differs from protocol")
    primary = _metrics(_apply(source_records, jobs_by_id, results_by_id, "primary"), truths)
    diagnostic = _metrics(_apply(source_records, jobs_by_id, results_by_id, "diagnostic"), truths)
    route_fraction = len(jobs) / sum(
        1
        for record in source_records
        for prediction in record["predictions"]
        if prediction["canonical_label"] in TARGET_LABELS
    )
    gates = {
        "danger_recall_not_decreased": primary["danger_recall"] >= baseline["danger_recall"],
        "three_target_macro_f1_strictly_increased": primary["three_target_class_macro_f1"]
        > baseline["three_target_class_macro_f1"],
        "target_false_positive_total_not_increased": primary["target_false_positive_total"]
        <= baseline["target_false_positive_total"],
        "routed_target_proposal_fraction_at_most_0_15": route_fraction
        <= protocol["continuation_gates"]["routed_target_proposal_fraction_max"],
    }
    gates["all_passed"] = all(gates.values())
    return {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "scope_label": protocol["scope_label"],
        "locked_evaluation_authorized": False,
        "source_files": {
            "cascade_protocol": {"path": "configs/cascade_feasibility_v1.json", "sha256": sha256_file(cascade_path)},
            "b2_results": {"path": _report_path(results_path), "sha256": sha256_file(results_path)},
        },
        "routing": {
            "job_count": len(jobs),
            "routed_image_count": len({item["sample_id"] for item in jobs}),
            "development_target_proposal_count": 172,
            "routed_target_proposal_fraction": route_fraction,
        },
        "rf_only": baseline,
        "primary_safety_preserving_cascade": primary,
        "diagnostic_direct_top1_cascade": diagnostic,
        "continuation_gates": gates,
        "b2_status_counts": dict(Counter(item["status"] for item in results)),
        "b2_latency_ms": {
            "count": len(latencies),
            "mean": sum(latencies) / len(latencies),
            "p50": percentile(latencies, 0.5),
            "p95": percentile(latencies, 0.95),
        },
        "decision": (
            "continue_only_with_separate_authorization_and_new_locked_protocol"
            if gates["all_passed"]
            else "stop_cascade_and_recommend_rfdetr_only"
        ),
        "limitations": protocol["reporting_boundaries"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = evaluate(args.results.resolve())
        if args.output.exists():
            raise BaselineContractError(f"refusing to overwrite existing output: {args.output}")
        write_json(args.output.resolve(), report)
    except (BaselineContractError, OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
