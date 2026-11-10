"""Deterministic label-free Level-A replay construction."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .contracts import ReplayContractError, load_replay_contract, validate_detection_frame


EXPECTED_PROTOCOL_ID = "robot_replay_level_a_v1"
EXPECTED_FOLDS = ("fold_01", "fold_02")
FORBIDDEN_FOLDS = {"fold_03", "fold_04", "fold_05"}
EXPECTED_EVENT_CONTRACT = {
    "path": "configs/replay_event_contract_v1.json",
    "sha256": "698de5fcd3be87cdd5dfa0f4d31c71bc61885fc5502a85761b3b295bcd5707c6",
}
EXPECTED_INPUTS = [
    {
        "fold_id": "fold_01",
        "scene_group": "development_group_1",
        "job_path": "experiments/pilot_v1/baseline_jobs/fold_01/rfdetr_images.jsonl",
        "job_sha256": "c443bd05d76d87dec40eb947b56342ee87668b8ea1ce1813390a36bf229113fa",
        "result_path": "experiments/pilot_v1/results/rfdetr_fold_01.jsonl",
        "result_sha256": "4731adfabfbcf486708189f2a0a97f5b4b7230875569e5b7c215c4dcdcbf0d7a",
        "expected_records": 98,
    },
    {
        "fold_id": "fold_02",
        "scene_group": "development_group_2",
        "job_path": "experiments/pilot_v1/baseline_jobs/fold_02/rfdetr_images.jsonl",
        "job_sha256": "415e874b8753dbb8366f60ed6e7f40b77bec54c2a44b20e028217e81af16a14c",
        "result_path": "experiments/pilot_v1/results/rfdetr_fold_02.jsonl",
        "result_sha256": "6d234f8b7a41bd193d6e450a859fda86e568d9ca7dbcd658ec91e5f6d7d67457",
        "expected_records": 74,
    },
]
EXPECTED_OUTPUTS = {
    "manifest": "experiments/replay_v1/outputs/replay_manifest.jsonl",
    "events": "experiments/replay_v1/outputs/danger_detection_events.jsonl",
    "decision_trace": "experiments/replay_v1/outputs/decision_trace.jsonl",
    "summary": "experiments/replay_v1/outputs/replay_summary.json",
}
EXPECTED_STORYBOARD_POLICY = {
    "states_in_order": ["HELD_UNLOCALIZED", "REJECTED", "SEARCHING"],
    "cases_per_state": 4,
    "selection_rule": "first_by_sequence_index_per_state",
    "truth_used_for_selection": False,
    "artifact": "demo/replay_v1_level_a.gif",
    "manifest": "demo/replay_v1_storyboard_manifest.json",
}
FORBIDDEN_KEYS = {
    "target_label", "record_labels", "labels", "ground_truth", "danger_truth",
    "truth_label", "label_source", "label_status", "label_reliability",
    "scene_seed", "room_id", "source_split", "exact_duplicate_group",
    "oracle_bbox", "layout", "world_layout",
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl_with_line_hash(path: Path) -> list[tuple[dict[str, Any], str]]:
    output = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            content = raw_line.rstrip("\r\n")
            if not content:
                raise ReplayContractError(f"blank JSONL line at {path}:{line_number}")
            value = json.loads(content)
            if not isinstance(value, dict):
                raise ReplayContractError(f"JSONL record must be an object at {path}:{line_number}")
            output.append((value, hashlib.sha256(content.encode("utf-8")).hexdigest()))
    return output


def _walk_keys(value: Any):
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def load_replay_protocol(project_root: Path, path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "protocol_id", "status", "replay_level", "event_contract",
        "development_inputs", "ordering", "decision_policy", "locked_folds_prohibited",
        "truth_join_policy", "storyboard_policy", "outputs",
    }
    if not isinstance(protocol, dict) or set(protocol) != required:
        raise ReplayContractError("replay protocol top-level schema changed")
    if protocol["schema_version"] != 1 or protocol["protocol_id"] != EXPECTED_PROTOCOL_ID:
        raise ReplayContractError("unsupported replay protocol")
    if protocol["status"] != "frozen_before_replay" or protocol["replay_level"] != "A":
        raise ReplayContractError("replay protocol must remain frozen at Level A")
    event_contract = protocol["event_contract"]
    if set(event_contract) != {"path", "sha256"}:
        raise ReplayContractError("event contract binding changed")
    if event_contract != EXPECTED_EVENT_CONTRACT:
        raise ReplayContractError("frozen event contract binding changed")
    contract_path = project_root / event_contract["path"]
    if sha256_path(contract_path) != event_contract["sha256"]:
        raise ReplayContractError("event contract hash mismatch")
    load_replay_contract(contract_path)
    inputs = protocol["development_inputs"]
    if inputs != EXPECTED_INPUTS:
        raise ReplayContractError("frozen development input bindings changed")
    if set(protocol["locked_folds_prohibited"]) != FORBIDDEN_FOLDS:
        raise ReplayContractError("locked fold prohibition changed")
    ordering = protocol["ordering"]
    if ordering != {
        "method": "fold_order_then_sample_id_lexicographic",
        "semantics": "seed_grouped_storyboard_non_temporal",
        "trusted_time_available": False,
    }:
        raise ReplayContractError("ordering semantics changed")
    policy = protocol["decision_policy"]
    if policy != {
        "detector_threshold": 0.5,
        "danger_class": "danger_red_sphere",
        "no_detection_state": "SEARCHING",
        "danger_state_path": ["SEARCHING", "CANDIDATE", "HELD_UNLOCALIZED"],
        "distractor_state_path": ["SEARCHING", "REJECTED"],
        "confirmation_allowed": False,
        "reporting_allowed": False,
        "timeout_policy": "disabled_no_trusted_time",
        "cross_frame_state_carry": False,
    }:
        raise ReplayContractError("decision policy changed")
    if protocol["truth_join_policy"] != "forbidden_during_replay_allowed_only_in_separate_posthoc_evaluation":
        raise ReplayContractError("truth join policy changed")
    if protocol["storyboard_policy"] != EXPECTED_STORYBOARD_POLICY:
        raise ReplayContractError("frozen storyboard policy changed")
    if protocol["outputs"] != EXPECTED_OUTPUTS:
        raise ReplayContractError("frozen replay output paths changed")
    return protocol


def build_replay_records(project_root: Path, protocol: Mapping[str, Any]):
    contract = load_replay_contract(project_root / protocol["event_contract"]["path"])
    manifest_records = []
    frame_records = []
    trace_records = []
    sequence_index = 0
    for input_spec in protocol["development_inputs"]:
        expected_keys = {
            "fold_id", "scene_group", "job_path", "job_sha256", "result_path",
            "result_sha256", "expected_records",
        }
        if set(input_spec) != expected_keys:
            raise ReplayContractError("development input schema changed")
        job_path = project_root / input_spec["job_path"]
        result_path = project_root / input_spec["result_path"]
        if sha256_path(job_path) != input_spec["job_sha256"]:
            raise ReplayContractError(f"job hash mismatch for {input_spec['fold_id']}")
        if sha256_path(result_path) != input_spec["result_sha256"]:
            raise ReplayContractError(f"result hash mismatch for {input_spec['fold_id']}")
        jobs = {record["sample_id"]: record for record, _ in _load_jsonl_with_line_hash(job_path)}
        result_pairs = _load_jsonl_with_line_hash(result_path)
        if len(jobs) != input_spec["expected_records"] or len(result_pairs) != input_spec["expected_records"]:
            raise ReplayContractError(f"record count mismatch for {input_spec['fold_id']}")
        for result, line_hash in sorted(result_pairs, key=lambda item: item[0]["sample_id"]):
            leaked = sorted(set(_walk_keys(result)) & FORBIDDEN_KEYS)
            if leaked:
                raise ReplayContractError(f"truth fields in inference result: {leaked}")
            sample_id = result.get("sample_id")
            job = jobs.get(sample_id)
            if job is None or result.get("job_id") != job.get("job_id"):
                raise ReplayContractError(f"job/result identity mismatch for {sample_id}")
            if job.get("fold_id") != input_spec["fold_id"] or result.get("fold_id") != input_spec["fold_id"]:
                raise ReplayContractError(f"job/result fold mismatch for {sample_id}")
            if result.get("status") != "ok" or result.get("error") is not None:
                raise ReplayContractError(f"non-ok result is not replayable: {sample_id}")
            if result.get("image_sha256") != job.get("image_sha256"):
                raise ReplayContractError(f"image hash mismatch for {sample_id}")
            predictions = result.get("predictions")
            if not isinstance(predictions, list):
                raise ReplayContractError(f"predictions must be a list for {sample_id}")
            detections = []
            for prediction in predictions:
                detections.append({
                    "prediction_index": int(prediction["prediction_index"]),
                    "track_id": None,
                    "raw_class_name": str(prediction["raw_class_name"]),
                    "class_name": str(prediction["canonical_label"]),
                    "confidence": float(prediction["confidence"]),
                    "bbox_xyxy": [float(value) for value in prediction["bbox_xyxy_pixels"]],
                    "temporal_hits": 0,
                    "temporal_confirmed": False,
                    "localized": False,
                    "position": None,
                    "position_frame": None,
                    "quality_accepted": False,
                    "quality_rejection_reasons": ["level_a_missing_rgbd_pose"],
                })
            frame = {
                "schema_version": 1,
                "contract_id": contract["contract_id"],
                "record_id": f"replay_v1::{sample_id}",
                "sample_id": sample_id,
                "sequence_index": sequence_index,
                "sequence_semantics": "seed_grouped_storyboard_non_temporal",
                "stamp": None,
                "stamp_source": "unavailable_legacy_rgb",
                "frame_id": "replay_camera",
                "frame_id_source": "offline_namespace_not_ros_tf",
                "input_sha256": job["image_sha256"],
                "inference_record_sha256": line_hash,
                "source_model": "RF-DETR-Seg Medium",
                "source_model_sha256": contract["source_model"]["sha256"],
                "inference_run_id": contract["source_model"]["run_id"],
                "processing_ms": float(result["latency_ms"]),
                "detections": detections,
            }
            validate_detection_frame(frame, contract)
            danger_count = sum(item["class_name"] == "danger_red_sphere" for item in detections)
            if danger_count:
                state_path = ["SEARCHING", "CANDIDATE", "HELD_UNLOCALIZED"]
                state = "HELD_UNLOCALIZED"
                reason = "danger_proposal_without_rgbd_pose"
            elif detections:
                state_path = ["SEARCHING", "REJECTED"]
                state = "REJECTED"
                reason = "non_danger_semantic_predictions"
            else:
                state_path = ["SEARCHING"]
                state = "SEARCHING"
                reason = "no_danger_proposal"
            manifest_records.append({
                "schema_version": 1,
                "protocol_id": protocol["protocol_id"],
                "record_id": frame["record_id"],
                "sample_id": sample_id,
                "sequence_index": sequence_index,
                "sequence_semantics": frame["sequence_semantics"],
                "scene_group": input_spec["scene_group"],
                "image": job["image"],
                "image_root_id": job["image_root_id"],
                "input_sha256": job["image_sha256"],
                "inference_result_path": input_spec["result_path"],
                "inference_record_sha256": line_hash,
            })
            frame_records.append(frame)
            trace_records.append({
                "schema_version": 1,
                "protocol_id": protocol["protocol_id"],
                "record_id": f"decision::{sample_id}",
                "event_record_id": frame["record_id"],
                "sample_id": sample_id,
                "sequence_index": sequence_index,
                "state_path": state_path,
                "final_state": state,
                "reason": reason,
                "danger_candidate_count": danger_count,
                "distractor_prediction_count": len(detections) - danger_count,
                "confirmation_allowed": False,
                "reporting_allowed": False,
                "cross_frame_state_carried": False,
            })
            sequence_index += 1
    summary = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "replay_level": "A",
        "frame_count": len(frame_records),
        "detection_count": sum(len(item["detections"]) for item in frame_records),
        "state_counts": {
            state: sum(item["final_state"] == state for item in trace_records)
            for state in ("SEARCHING", "REJECTED", "HELD_UNLOCALIZED")
        },
        "confirmed_count": 0,
        "reported_count": 0,
        "world_coordinate_count": 0,
        "truth_joined": False,
        "official_result_generated": False,
    }
    return manifest_records, frame_records, trace_records, summary


def _jsonl_bytes(records) -> bytes:
    return b"".join(
        (json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for record in records
    )


def _json_bytes(record) -> bytes:
    return (json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def render_outputs(project_root: Path, protocol: Mapping[str, Any]) -> dict[str, bytes]:
    manifest, events, trace, summary = build_replay_records(project_root, protocol)
    outputs = protocol["outputs"]
    return {
        outputs["manifest"]: _jsonl_bytes(manifest),
        outputs["events"]: _jsonl_bytes(events),
        outputs["decision_trace"]: _jsonl_bytes(trace),
        outputs["summary"]: _json_bytes(summary),
    }


def write_outputs(project_root: Path, protocol: Mapping[str, Any]) -> None:
    allowed_root = (project_root / "experiments" / "replay_v1" / "outputs").resolve()
    for relative_path, content in render_outputs(project_root, protocol).items():
        target = (project_root / relative_path).resolve()
        if allowed_root not in target.parents:
            raise ReplayContractError(f"replay output escapes dedicated directory: {relative_path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, target)
        except Exception:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise
