"""Fail-closed Level-A replay contracts with no robot-repository imports."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

EXPECTED_ROBOT_COMMIT = "e26b48ebd6871eb5753ac4b7750726c36b05a65a"
EXPECTED_ROBOT_FILES = {
    "docs/INTERFACE_REGISTRY.md": "551c4272e0f3eb3edaf60b4a329d836e1cd49d7d",
    "docs/WINDYSKY5_ROOM_ARCHITECTURE.md": "ce9b9cf45d311c1a4360840cff26d2f06b0157d2",
    "scripts/exploration/vision_v2_perception.py": "4251d5a84ec88b0db68fbafaa7c2a4c17f9a830f",
    "src/danger_vision/config/detector.yaml": "9c906760fae839ee63251283eba2c634e8e58f75",
    "src/danger_vision/config/model_manifest.yaml": "a1fdfb043d3dcac4cd96d294e5c446094f59bec6",
    "src/danger_vision/msg/DangerDetection.msg": "a00d2fd9445857d7d5e5f5c699f861f235b7d0f1",
    "src/danger_vision/msg/DangerDetectionArray.msg": "047f37f29775b8a7062859d04afa1e61412094f7",
    "src/danger_vision/src/danger_vision/result_writer.py": "a9a1761f6a29fbf48ed42d82463958bdec580b96",
}
EXPECTED_INTERFACES = {
    "rgb_input": {
        "topic": "/real_sense/rgb/image_raw",
        "ros_type": "sensor_msgs/Image",
    },
    "detection_output_equivalent": {
        "topic": "/danger_vision/detection_array",
        "ros_type": "danger_vision/DangerDetectionArray",
        "empty_array_is_heartbeat": True,
    },
    "processed_pose_level_b_only": {
        "topic": "/danger_vision/processed_pose",
        "ros_type": "geometry_msgs/PoseStamped",
    },
    "world_alignment_level_b_only": {
        "topic": "/danger_vision/world_alignment",
        "ros_type": "std_msgs/String",
    },
    "official_result_level_b_only": "results/detected_danger.json",
}
EXPECTED_CLASSES = [
    "danger_red_sphere",
    "distractor_red_box",
    "distractor_green_sphere",
    "distractor_green_wall",
]
EXPECTED_ALIASES = {
    "Red_ball": "danger_red_sphere",
    "red_ball": "danger_red_sphere",
    "danger_red_sphere": "danger_red_sphere",
    "red_cube": "distractor_red_box",
    "distractor_red_box": "distractor_red_box",
    "Green_ball": "distractor_green_sphere",
    "green_ball": "distractor_green_sphere",
    "distractor_green_sphere": "distractor_green_sphere",
    "Green_wall": "distractor_green_wall",
    "green_wall": "distractor_green_wall",
    "distractor_green_wall": "distractor_green_wall",
}
EXPECTED_FRAME_FIELDS = [
    "schema_version",
    "contract_id",
    "record_id",
    "sample_id",
    "sequence_index",
    "sequence_semantics",
    "stamp",
    "stamp_source",
    "frame_id",
    "frame_id_source",
    "input_sha256",
    "inference_record_sha256",
    "source_model",
    "source_model_sha256",
    "inference_run_id",
    "processing_ms",
    "detections",
]
EXPECTED_DETECTION_FIELDS = [
    "prediction_index",
    "track_id",
    "raw_class_name",
    "class_name",
    "confidence",
    "bbox_xyxy",
    "temporal_hits",
    "temporal_confirmed",
    "localized",
    "position",
    "position_frame",
    "quality_accepted",
    "quality_rejection_reasons",
]
EXPECTED_FORBIDDEN_FIELDS = [
    "target_label",
    "record_labels",
    "labels",
    "ground_truth",
    "danger_truth",
    "truth_label",
    "label_source",
    "label_status",
    "label_reliability",
    "scene_seed",
    "room_id",
    "source_split",
    "exact_duplicate_group",
    "oracle_bbox",
    "layout",
    "world_layout",
]
EXPECTED_SOURCE_MODEL = {
    "name": "RF-DETR-Seg Medium",
    "sha256": "ec2de357f402901cf624835f11d5c54bb0eab8980d1268d7ee8a9122a446fa50",
    "run_id": "rfdetr_pilot_cpu_full_v1",
}
EXPECTED_MISSING_RESOURCES = [
    "trusted_capture_timestamps_or_frame_intervals",
    "synchronized_depth",
    "rgb_and_depth_camera_info",
    "autonomous_pose_or_odometry",
    "tf_or_permitted_world_alignment",
    "ros_bag_mcap_or_db3",
]


class ReplayContractError(ValueError):
    """Raised when a replay contract or event violates a frozen boundary."""


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], context: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ReplayContractError(
            f"{context} keys differ: missing={missing}, extra={extra}"
        )


def _walk_keys(value: Any):
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def _require_sha256(value: Any, context: str) -> None:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ReplayContractError(f"{context} must be a lowercase SHA-256")


def validate_replay_contract(contract: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "contract_id",
        "status",
        "replay_level",
        "authoritative_robot_source",
        "interfaces",
        "canonical_classes",
        "class_aliases",
        "frame_required_fields",
        "detection_required_fields",
        "forbidden_runtime_fields",
        "level_a_constraints",
        "source_model",
        "g0_decision",
    }
    _require_exact_keys(contract, required, "contract")
    if contract["schema_version"] != 1:
        raise ReplayContractError("unsupported contract schema_version")
    if contract["status"] != "frozen_before_replay":
        raise ReplayContractError("contract must be frozen before replay")
    if contract["replay_level"] != "A":
        raise ReplayContractError("this validator only permits Level A")

    source = contract["authoritative_robot_source"]
    if not isinstance(source, Mapping):
        raise ReplayContractError("authoritative_robot_source must be an object")
    _require_exact_keys(
        source,
        {"branch", "commit", "tree_access", "files"},
        "authoritative_robot_source",
    )
    if source["branch"] != "WindySky-Final":
        raise ReplayContractError("robot source must be WindySky-Final")
    if not isinstance(source["commit"], str) or not re.fullmatch(
        r"[0-9a-f]{40}", source["commit"]
    ):
        raise ReplayContractError("robot source commit must be a full Git object ID")
    if source["tree_access"] != "committed_tree_read_only":
        raise ReplayContractError("robot source must remain committed-tree read-only")
    if not isinstance(source["files"], Mapping) or not source["files"]:
        raise ReplayContractError("robot source files must be pinned")
    for path, blob_oid in source["files"].items():
        if not isinstance(path, str) or not path:
            raise ReplayContractError("robot source path must be non-empty")
        if not isinstance(blob_oid, str) or not re.fullmatch(r"[0-9a-f]{40}", blob_oid):
            raise ReplayContractError(f"invalid Git blob ID for {path}")
    if source["commit"] != EXPECTED_ROBOT_COMMIT or source["files"] != EXPECTED_ROBOT_FILES:
        raise ReplayContractError("frozen robot source pins changed")

    if contract["interfaces"] != EXPECTED_INTERFACES:
        raise ReplayContractError("frozen robot interface mapping changed")

    classes = contract["canonical_classes"]
    aliases = contract["class_aliases"]
    if not isinstance(classes, list) or len(classes) != len(set(classes)):
        raise ReplayContractError("canonical_classes must be a unique list")
    if "danger_red_sphere" not in classes:
        raise ReplayContractError("danger canonical class is missing")
    if not isinstance(aliases, Mapping) or not aliases:
        raise ReplayContractError("class_aliases must be a non-empty object")
    if any(value not in classes for value in aliases.values()):
        raise ReplayContractError("every class alias must map to a canonical class")
    if classes != EXPECTED_CLASSES or aliases != EXPECTED_ALIASES:
        raise ReplayContractError("frozen class mapping changed")

    frame_fields = contract["frame_required_fields"]
    detection_fields = contract["detection_required_fields"]
    if not isinstance(frame_fields, list) or len(frame_fields) != len(set(frame_fields)):
        raise ReplayContractError("frame_required_fields must be unique")
    if not isinstance(detection_fields, list) or len(detection_fields) != len(
        set(detection_fields)
    ):
        raise ReplayContractError("detection_required_fields must be unique")
    if frame_fields != EXPECTED_FRAME_FIELDS or detection_fields != EXPECTED_DETECTION_FIELDS:
        raise ReplayContractError("frozen event field set changed")

    forbidden = contract["forbidden_runtime_fields"]
    if not isinstance(forbidden, list) or not forbidden:
        raise ReplayContractError("forbidden_runtime_fields must be non-empty")
    if set(frame_fields) & set(forbidden) or set(detection_fields) & set(forbidden):
        raise ReplayContractError("required and forbidden fields overlap")
    if forbidden != EXPECTED_FORBIDDEN_FIELDS:
        raise ReplayContractError("frozen forbidden-field boundary changed")

    constraints = contract["level_a_constraints"]
    expected_constraints = {
        "stamp",
        "stamp_source",
        "frame_id",
        "frame_id_source",
        "sequence_semantics",
        "localized",
        "position",
        "position_frame",
        "temporal_hits",
        "temporal_confirmed",
        "quality_accepted",
        "official_result_output_allowed",
    }
    _require_exact_keys(constraints, expected_constraints, "level_a_constraints")
    if constraints != {
        "stamp": None,
        "stamp_source": "unavailable_legacy_rgb",
        "frame_id": "replay_camera",
        "frame_id_source": "offline_namespace_not_ros_tf",
        "sequence_semantics": "seed_grouped_storyboard_non_temporal",
        "localized": False,
        "position": None,
        "position_frame": None,
        "temporal_hits": 0,
        "temporal_confirmed": False,
        "quality_accepted": False,
        "official_result_output_allowed": False,
    }:
        raise ReplayContractError("Level A constraints were weakened")

    model = contract["source_model"]
    _require_exact_keys(model, {"name", "sha256", "run_id"}, "source_model")
    _require_sha256(model["sha256"], "source_model.sha256")
    if model != EXPECTED_SOURCE_MODEL:
        raise ReplayContractError("frozen source model changed")
    gate = contract["g0_decision"]
    _require_exact_keys(
        gate,
        {"decision", "missing_resources", "world_coordinate_output_allowed"},
        "g0_decision",
    )
    if (
        gate["decision"] != "level_a_only"
        or gate["world_coordinate_output_allowed"]
        or gate["missing_resources"] != EXPECTED_MISSING_RESOURCES
    ):
        raise ReplayContractError("G0 must fail closed to Level A")


def load_replay_contract(path: Path | str) -> dict[str, Any]:
    contract_path = Path(path)
    with contract_path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ReplayContractError("replay contract must be a JSON object")
    validate_replay_contract(value)
    return value


def _validate_detection(
    detection: Mapping[str, Any], contract: Mapping[str, Any], index: int
) -> None:
    if not isinstance(detection, Mapping):
        raise ReplayContractError(f"detection[{index}] must be an object")
    _require_exact_keys(
        detection,
        set(contract["detection_required_fields"]),
        f"detection[{index}]",
    )
    if not isinstance(detection["prediction_index"], int) or isinstance(
        detection["prediction_index"], bool
    ) or detection["prediction_index"] < 0:
        raise ReplayContractError("prediction_index must be a non-negative integer")
    if detection["track_id"] is not None:
        raise ReplayContractError("Level A cannot claim a temporal track_id")

    raw_name = detection["raw_class_name"]
    class_name = detection["class_name"]
    mapped = contract["class_aliases"].get(raw_name)
    if mapped != class_name:
        raise ReplayContractError(
            f"class alias mismatch: {raw_name!r} does not map to {class_name!r}"
        )
    if not _is_number(detection["confidence"]) or not math.isfinite(
        float(detection["confidence"])
    ) or not 0.0 <= float(detection["confidence"]) <= 1.0:
        raise ReplayContractError("confidence must be finite and within [0, 1]")

    bbox = detection["bbox_xyxy"]
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ReplayContractError("bbox_xyxy must contain four values")
    if any(not _is_number(value) or not math.isfinite(float(value)) for value in bbox):
        raise ReplayContractError("bbox_xyxy values must be finite numbers")
    x1, y1, x2, y2 = [float(value) for value in bbox]
    if x2 <= x1 or y2 <= y1 or x1 < 0.0 or y1 < 0.0:
        raise ReplayContractError("bbox_xyxy must be positive and ordered")

    constraints = contract["level_a_constraints"]
    for key in (
        "localized",
        "position",
        "position_frame",
        "temporal_hits",
        "temporal_confirmed",
        "quality_accepted",
    ):
        if detection[key] != constraints[key]:
            raise ReplayContractError(f"Level A requires {key}={constraints[key]!r}")
    reasons = detection["quality_rejection_reasons"]
    if not isinstance(reasons, list) or "level_a_missing_rgbd_pose" not in reasons:
        raise ReplayContractError("Level A quality rejection reason is required")
    if any(not isinstance(reason, str) or not reason for reason in reasons):
        raise ReplayContractError("quality rejection reasons must be strings")


def validate_detection_frame(
    frame: Mapping[str, Any], contract: Mapping[str, Any]
) -> None:
    validate_replay_contract(contract)
    if not isinstance(frame, Mapping):
        raise ReplayContractError("frame event must be an object")
    _require_exact_keys(frame, set(contract["frame_required_fields"]), "frame")

    forbidden = set(contract["forbidden_runtime_fields"])
    leaked = sorted({key for key in _walk_keys(frame) if key in forbidden})
    if leaked:
        raise ReplayContractError(f"forbidden runtime fields present: {leaked}")

    if frame["schema_version"] != contract["schema_version"]:
        raise ReplayContractError("frame schema_version mismatch")
    if frame["contract_id"] != contract["contract_id"]:
        raise ReplayContractError("frame contract_id mismatch")
    if not isinstance(frame["record_id"], str) or not frame["record_id"]:
        raise ReplayContractError("record_id must be non-empty")
    if not isinstance(frame["sample_id"], str) or not frame["sample_id"]:
        raise ReplayContractError("sample_id must be non-empty")
    if not isinstance(frame["sequence_index"], int) or isinstance(
        frame["sequence_index"], bool
    ) or frame["sequence_index"] < 0:
        raise ReplayContractError("sequence_index must be a non-negative integer")

    constraints = contract["level_a_constraints"]
    for key in (
        "stamp",
        "stamp_source",
        "frame_id",
        "frame_id_source",
        "sequence_semantics",
    ):
        if frame[key] != constraints[key]:
            raise ReplayContractError(f"Level A requires {key}={constraints[key]!r}")
    _require_sha256(frame["input_sha256"], "input_sha256")
    _require_sha256(frame["inference_record_sha256"], "inference_record_sha256")
    model = contract["source_model"]
    if frame["source_model"] != model["name"]:
        raise ReplayContractError("source_model mismatch")
    if frame["source_model_sha256"] != model["sha256"]:
        raise ReplayContractError("source_model_sha256 mismatch")
    if frame["inference_run_id"] != model["run_id"]:
        raise ReplayContractError("inference_run_id mismatch")
    if not _is_number(frame["processing_ms"]) or not math.isfinite(
        float(frame["processing_ms"])
    ) or float(frame["processing_ms"]) < 0.0:
        raise ReplayContractError("processing_ms must be finite and non-negative")

    detections = frame["detections"]
    if not isinstance(detections, list):
        raise ReplayContractError("detections must be a list, including when empty")
    for index, detection in enumerate(detections):
        _validate_detection(detection, contract, index)
    prediction_indices = [item["prediction_index"] for item in detections]
    if prediction_indices != list(range(len(detections))):
        raise ReplayContractError("prediction_index must be contiguous from zero")
    confidences = [float(item["confidence"]) for item in detections]
    if confidences != sorted(confidences, reverse=True):
        raise ReplayContractError("detections must be confidence-sorted")
