#!/usr/bin/env python3
"""Build deterministic Level-B localization, tracks, decisions, and result JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_KEYS = {
    "message_counts", "protocol_id", "records", "schema_version", "source_bag",
    "static_transforms", "synchronization_policy",
}
SOURCE_BAG_KEYS = {"filename", "ground_truth_topics_present", "sha256", "topics"}
CAMERA_INFO_KEYS = {"D", "K", "P", "R", "distortion_model", "frame_id", "height", "width"}
IMAGE_KEYS = {"encoding", "frame_id", "height", "path", "sha256", "stamp_ns", "width"}
ODOMETRY_KEYS = {"child_frame_id", "frame_id", "orientation_xyzw", "position", "stamp_ns"}
SYNCHRONIZATION_KEYS = {"image_odom_delta_ms", "rgb_depth_delta_ms"}
STATIC_TRANSFORM_KEYS = {"child_frame", "parent_frame", "rotation_xyzw", "translation"}
SYNCHRONIZATION_POLICY_KEYS = {
    "max_camera_info_delta_ms", "max_image_odometry_delta_ms",
    "max_rgb_depth_delta_ms", "sample_stride",
}
RECORD_KEYS = {
    "camera_info", "depth", "frame_id", "odometry", "rgb", "schema_version",
    "sequence_index", "synchronization",
}
PREDICTION_ROW_KEYS = {
    "confidence_threshold", "frame_id", "latency_ms", "predictions", "protocol_id",
    "rgb_sha256", "rgb_stamp_ns", "schema_version", "sequence_index", "source_model",
    "source_model_sha256",
}
PREDICTION_KEYS = {
    "bbox_area_fraction", "bbox_xyxy_pixels", "canonical_label", "class_id",
    "confidence", "mask_area_fraction", "mask_coverage_in_bbox", "prediction_index",
    "raw_class_name",
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contained_file(root: Path, raw_path: str) -> Path:
    root = root.resolve()
    path = (root / raw_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"input path escapes manifest root: {raw_path}") from exc
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"input is not a regular contained file: {raw_path}")
    return path


def reject_forbidden_manifest_fields(value: Any, path: str = "manifest") -> None:
    forbidden_tokens = ("truth", "label", "oracle", "target_coordinate", "scene_manifest", "world_file")
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = key.lower()
            if key != "ground_truth_topics_present" and any(token in lowered for token in forbidden_tokens):
                raise ValueError(f"forbidden inference field: {path}.{key}")
            reject_forbidden_manifest_fields(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_forbidden_manifest_fields(child, f"{path}[{index}]")


def validate_input_manifest(
    manifest: dict[str, Any],
    protocol: dict[str, Any],
    manifest_root: Path,
) -> None:
    """Validate the historical target-label-free, pose-oracle replay manifest."""
    if set(manifest) != MANIFEST_KEYS:
        raise ValueError("unexpected input manifest schema")
    reject_forbidden_manifest_fields(manifest)
    if protocol.get("status") not in {
        "frozen_before_accepted_inference",
        "frozen_before_inference",
        "frozen_before_v2_acquisition_and_inference",
    }:
        raise ValueError("protocol is not frozen")
    if manifest.get("protocol_id") != protocol.get("protocol_id"):
        raise ValueError("input manifest/protocol mismatch")
    source_bag = manifest.get("source_bag", {})
    if set(source_bag) != SOURCE_BAG_KEYS:
        raise ValueError("unexpected source bag schema")
    if source_bag.get("ground_truth_topics_present") is not False:
        raise ValueError("declared danger-target truth input is forbidden")
    if set(source_bag.get("topics", [])) != set(protocol["acquisition"]["required_topics"]):
        raise ValueError("bag topic set differs from the frozen allowlist")
    bag_path = contained_file(manifest_root, source_bag["filename"])
    if sha256_path(bag_path) != source_bag["sha256"]:
        raise ValueError("source bag hash mismatch")
    if set(manifest.get("synchronization_policy", {})) != SYNCHRONIZATION_POLICY_KEYS:
        raise ValueError("unexpected synchronization policy schema")
    for transform in manifest.get("static_transforms", []):
        if set(transform) != STATIC_TRANSFORM_KEYS:
            raise ValueError("unexpected static transform schema")
    for expected_index, frame in enumerate(manifest.get("records", [])):
        if set(frame) != RECORD_KEYS:
            raise ValueError("unexpected frame record schema")
        if set(frame.get("camera_info", {})) != CAMERA_INFO_KEYS:
            raise ValueError("unexpected camera info schema")
        if set(frame.get("rgb", {})) != IMAGE_KEYS or set(frame.get("depth", {})) != IMAGE_KEYS:
            raise ValueError("unexpected image record schema")
        if set(frame.get("odometry", {})) != ODOMETRY_KEYS:
            raise ValueError("unexpected odometry schema")
        if set(frame.get("synchronization", {})) != SYNCHRONIZATION_KEYS:
            raise ValueError("unexpected frame synchronization schema")
        if frame.get("sequence_index") != expected_index or frame.get("schema_version") != 1:
            raise ValueError("invalid frame sequence/schema")
        for channel in ("rgb", "depth"):
            path = contained_file(manifest_root, frame[channel]["path"])
            if sha256_path(path) != frame[channel]["sha256"]:
                raise ValueError(f"{channel} frame hash mismatch: {frame['frame_id']}")


def validate_manifest_and_predictions(
    manifest: dict[str, Any],
    prediction_rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    manifest_root: Path,
) -> None:
    validate_input_manifest(manifest, protocol, manifest_root)
    frames = manifest.get("records", [])
    if len(frames) != len(prediction_rows):
        raise ValueError("frame/prediction count mismatch")
    allowed_classes = set(protocol["detector"]["allowed_classes"])
    expected_model = protocol["detector"]["name"]
    expected_model_hash = protocol["detector"]["checkpoint_sha256"]
    expected_threshold = float(protocol["detector"]["confidence_threshold"])
    baseline = json.loads(
        (PROJECT_ROOT / "configs" / "baselines.json").read_text(encoding="utf-8")
    )["rfdetr"]
    class_names = {int(key): value for key, value in baseline["class_names"].items()}
    class_aliases = baseline["class_aliases"]
    last_stamp = -1
    for expected_index, (frame, inference) in enumerate(zip(frames, prediction_rows)):
        if set(inference) != PREDICTION_ROW_KEYS:
            raise ValueError("unexpected prediction row schema")
        if frame["sequence_index"] != expected_index or inference["sequence_index"] != expected_index:
            raise ValueError("non-contiguous sequence index")
        stamp = frame["rgb"]["stamp_ns"]
        if stamp <= last_stamp or inference["rgb_stamp_ns"] != stamp:
            raise ValueError("prediction timestamp/order mismatch")
        last_stamp = stamp
        if (
            frame["frame_id"] != inference["frame_id"]
            or frame["rgb"]["sha256"] != inference["rgb_sha256"]
            or inference["protocol_id"] != protocol["protocol_id"]
        ):
            raise ValueError("frame/prediction identity mismatch")
        if inference["schema_version"] != 1 or frame["schema_version"] != 1:
            raise ValueError("unexpected Level-B schema version")
        if inference["source_model"] != expected_model or inference["source_model_sha256"] != expected_model_hash:
            raise ValueError("prediction model binding mismatch")
        if float(inference["confidence_threshold"]) != expected_threshold:
            raise ValueError("prediction threshold mismatch")
        latency = inference["latency_ms"]
        if isinstance(latency, bool) or not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency < 0:
            raise ValueError("invalid prediction latency")
        seen_indexes = set()
        for prediction in inference["predictions"]:
            if set(prediction) != PREDICTION_KEYS:
                raise ValueError("unexpected prediction schema")
            index = prediction["prediction_index"]
            if isinstance(index, bool) or not isinstance(index, int) or index in seen_indexes:
                raise ValueError("invalid or duplicate prediction index")
            seen_indexes.add(index)
            class_id = prediction["class_id"]
            if isinstance(class_id, bool) or not isinstance(class_id, int) or class_id not in class_names:
                raise ValueError("invalid prediction class ID")
            raw_name = prediction["raw_class_name"]
            if raw_name != class_names[class_id] or class_aliases.get(raw_name) != prediction["canonical_label"]:
                raise ValueError("prediction class mapping mismatch")
            if prediction["canonical_label"] not in allowed_classes:
                raise ValueError("prediction class is not allowed")
            confidence = float(prediction["confidence"])
            bbox = [float(value) for value in prediction["bbox_xyxy_pixels"]]
            if not math.isfinite(confidence) or not expected_threshold <= confidence <= 1.0:
                raise ValueError("prediction confidence violates protocol")
            if len(bbox) != 4 or not all(math.isfinite(value) for value in bbox):
                raise ValueError("prediction bbox is malformed")
            if not (0 <= bbox[0] < bbox[2] <= 640 and 0 <= bbox[1] < bbox[3] <= 480):
                raise ValueError("prediction bbox is out of bounds")
            metrics = []
            for field in ("bbox_area_fraction", "mask_area_fraction", "mask_coverage_in_bbox"):
                value = prediction[field]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"invalid prediction {field}")
                metrics.append(float(value))
            bbox_fraction, mask_fraction, mask_coverage = metrics
            expected_bbox_fraction = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / (640 * 480)
            if abs(bbox_fraction - expected_bbox_fraction) > 5e-8:
                raise ValueError("prediction bbox area fraction mismatch")
            if not (0 <= mask_fraction <= 1 and 0 <= mask_coverage <= 1):
                raise ValueError("prediction mask metrics are out of range")
        if seen_indexes != set(range(len(inference["predictions"]))):
            raise ValueError("prediction indexes are not contiguous")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def rotate(q: np.ndarray, vector: np.ndarray) -> np.ndarray:
    xyz = q[:3]
    t = 2.0 * np.cross(xyz, vector)
    return vector + q[3] * t + np.cross(xyz, t)


def apply_transform(point: np.ndarray, transform: dict[str, Any]) -> np.ndarray:
    translation = np.asarray(transform["translation"], dtype=np.float64)
    quaternion = np.asarray(transform["rotation_xyzw"], dtype=np.float64)
    quaternion = quaternion / np.linalg.norm(quaternion)
    return rotate(quaternion, point) + translation


def localize_detection(
    frame: dict[str, Any],
    prediction: dict[str, Any],
    depth: np.ndarray,
    policy: dict[str, Any],
    static_by_edge: dict[tuple[str, str], dict[str, Any]],
) -> tuple[list[float] | None, dict[str, Any]]:
    bbox = [float(value) for value in prediction["bbox_xyxy_pixels"]]
    inner_fraction = float(policy["bbox_depth_region"]["inner_fraction"])
    center_x = (bbox[0] + bbox[2]) / 2.0
    center_y = (bbox[1] + bbox[3]) / 2.0
    half_width = (bbox[2] - bbox[0]) * inner_fraction / 2.0
    half_height = (bbox[3] - bbox[1]) * inner_fraction / 2.0
    x0 = max(0, int(math.floor(center_x - half_width)))
    x1 = min(depth.shape[1], int(math.ceil(center_x + half_width)))
    y0 = max(0, int(math.floor(center_y - half_height)))
    y1 = min(depth.shape[0], int(math.ceil(center_y + half_height)))
    region = depth[y0:y1, x0:x1]
    minimum = float(policy["bbox_depth_region"]["minimum_depth_m"])
    maximum = float(policy["bbox_depth_region"]["maximum_depth_m"])
    valid = np.isfinite(region) & (region >= minimum) & (region <= maximum)
    valid_values = region[valid]
    valid_fraction = float(valid.mean()) if valid.size else 0.0
    quality = {
        "region_xyxy": [x0, y0, x1, y1],
        "valid_pixels": int(valid_values.size),
        "valid_fraction": valid_fraction,
        "median_depth_m": None,
        "mad_depth_m": None,
        "accepted": False,
        "reasons": [],
    }
    if valid_values.size < int(policy["bbox_depth_region"]["minimum_valid_pixels"]):
        quality["reasons"].append("insufficient_valid_depth_pixels")
    if valid_fraction < float(policy["bbox_depth_region"]["minimum_valid_fraction"]):
        quality["reasons"].append("insufficient_valid_depth_fraction")
    if quality["reasons"]:
        return None, quality
    median = float(np.median(valid_values))
    mad = float(np.median(np.abs(valid_values - median)))
    quality["median_depth_m"] = median
    quality["mad_depth_m"] = mad
    if mad > float(policy["bbox_depth_region"]["maximum_mad_m"]):
        quality["reasons"].append("depth_mad_too_large")
        return None, quality

    intrinsic = frame["camera_info"]["K"]
    fx, fy = float(intrinsic[0]), float(intrinsic[4])
    cx, cy = float(intrinsic[2]), float(intrinsic[5])
    optical = np.asarray(
        [(center_x - cx) * median / fx, (center_y - cy) * median / fy, median],
        dtype=np.float64,
    )
    real_sense = apply_transform(
        optical, static_by_edge[("real_sense", "real_sense_optical_frame")]
    )
    base = apply_transform(real_sense, static_by_edge[("base", "real_sense")])
    odometry = frame["odometry"]
    odom_transform = {
        "translation": odometry["position"],
        "rotation_xyzw": odometry["orientation_xyzw"],
    }
    world = apply_transform(base, odom_transform)
    quality["accepted"] = True
    return [float(value) for value in world], quality


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(path)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            stream.write("\n")


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output_dir}")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    prediction_rows = [
        json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines()
    ]
    frames = manifest["records"]
    validate_manifest_and_predictions(manifest, prediction_rows, protocol, args.manifest.parent)
    static_by_edge = {
        (item["parent_frame"], item["child_frame"]): item
        for item in manifest["static_transforms"]
    }
    association = protocol["association_and_confirmation"]
    localization_policy = protocol["localization"]

    tracks: list[dict[str, Any]] = []
    events = []
    traces = []
    for frame, inference in zip(frames, prediction_rows):
        depth_path = contained_file(args.manifest.parent, frame["depth"]["path"])
        with np.load(depth_path) as archive:
            depth = np.asarray(archive["depth"], dtype=np.float32)
        detections = []
        tracks_used_this_frame: set[str] = set()
        for prediction in sorted(
            inference["predictions"], key=lambda item: -float(item["confidence"])
        ):
            position, quality = localize_detection(
                frame, prediction, depth, localization_policy, static_by_edge
            )
            track = None
            if position is not None:
                point = np.asarray(position, dtype=np.float64)
                candidates = []
                for candidate in tracks:
                    if candidate["track_id"] in tracks_used_this_frame:
                        continue
                    if candidate["class_name"] != prediction["canonical_label"]:
                        continue
                    gap = (
                        frame["rgb"]["stamp_ns"] - candidate["last_stamp_ns"]
                    ) / 1_000_000_000.0
                    if gap < 0 or gap > float(association["maximum_time_gap_s"]):
                        continue
                    distance = float(np.linalg.norm(point - candidate["positions"][-1]))
                    if distance <= float(association["maximum_world_distance_m"]):
                        candidates.append((distance, candidate))
                if candidates:
                    track = min(candidates, key=lambda item: item[0])[1]
                else:
                    track = {
                        "track_id": f"track_{len(tracks):03d}",
                        "class_name": prediction["canonical_label"],
                        "positions": [],
                        "frame_ids": [],
                        "last_stamp_ns": frame["rgb"]["stamp_ns"],
                        "confirmed_at_sequence": None,
                    }
                    tracks.append(track)
                track["positions"].append(point)
                track["frame_ids"].append(frame["frame_id"])
                track["last_stamp_ns"] = frame["rgb"]["stamp_ns"]
                tracks_used_this_frame.add(track["track_id"])
                positions = np.asarray(track["positions"])
                maximum_std = float(np.std(positions, axis=0).max())
                if (
                    track["confirmed_at_sequence"] is None
                    and len(track["positions"])
                    >= int(association["minimum_observations"])
                    and len(set(track["frame_ids"]))
                    >= int(association["minimum_distinct_frames"])
                    and maximum_std <= float(association["maximum_position_std_m"])
                ):
                    track["confirmed_at_sequence"] = frame["sequence_index"]
            temporal_hits = len(track["positions"]) if track is not None else 0
            confirmed = bool(
                track is not None and track["confirmed_at_sequence"] is not None
            )
            detections.append(
                {
                    "prediction_index": prediction["prediction_index"],
                    "class_name": prediction["canonical_label"],
                    "confidence": prediction["confidence"],
                    "bbox_xyxy": prediction["bbox_xyxy_pixels"],
                    "localized": position is not None,
                    "position": position,
                    "position_frame": "odom" if position is not None else None,
                    "depth_quality": quality,
                    "track_id": track["track_id"] if track is not None else None,
                    "temporal_hits": temporal_hits,
                    "temporal_confirmed": confirmed,
                }
            )
        danger = [item for item in detections if item["class_name"] == "danger_red_sphere"]
        if any(item["temporal_confirmed"] for item in danger):
            state_path, final_state, reason = ["SEARCHING", "CANDIDATE", "CONFIRMED", "REPORTED"], "REPORTED", "confirmed_localized_danger"
        elif any(item["localized"] for item in danger):
            state_path, final_state, reason = ["SEARCHING", "CANDIDATE"], "CANDIDATE", "localized_pending_temporal_confirmation"
        elif danger:
            state_path, final_state, reason = ["SEARCHING", "CANDIDATE", "HELD_UNLOCALIZED"], "HELD_UNLOCALIZED", "danger_depth_quality_rejected"
        elif detections:
            state_path, final_state, reason = ["SEARCHING", "REJECTED"], "REJECTED", "distractor_only"
        else:
            state_path, final_state, reason = ["SEARCHING"], "SEARCHING", "no_proposal"
        event = {
            "schema_version": 1,
            "protocol_id": protocol["protocol_id"],
            "record_id": f"event::{frame['frame_id']}",
            "frame_id": frame["frame_id"],
            "sequence_index": frame["sequence_index"],
            "stamp_ns": frame["rgb"]["stamp_ns"],
            "camera_frame": frame["rgb"]["frame_id"],
            "world_frame": "odom",
            "input_sha256": frame["rgb"]["sha256"],
            "inference_latency_ms": inference["latency_ms"],
            "detections": detections,
        }
        events.append(event)
        traces.append(
            {
                "schema_version": 1,
                "protocol_id": protocol["protocol_id"],
                "event_record_id": event["record_id"],
                "sequence_index": frame["sequence_index"],
                "state_path": state_path,
                "final_state": final_state,
                "reason": reason,
            }
        )

    confirmed_tracks = []
    for track in tracks:
        if track["confirmed_at_sequence"] is None:
            continue
        positions = np.asarray(track["positions"])
        confirmed_tracks.append(
            {
                "track_id": track["track_id"],
                "class_name": track["class_name"],
                "observation_count": len(track["positions"]),
                "confirmed_at_sequence": track["confirmed_at_sequence"],
                "mean_position": [float(value) for value in np.mean(positions, axis=0)],
                "position_std": [float(value) for value in np.std(positions, axis=0)],
            }
        )
    dangers = [item for item in confirmed_tracks if item["class_name"] == "danger_red_sphere"]
    exploration_time = (
        frames[-1]["rgb"]["stamp_ns"] - frames[0]["rgb"]["stamp_ns"]
    ) / 1_000_000_000.0
    result = {
        "detected_danger_sources": [
            {"position": [round(float(value), 2) for value in item["mean_position"]]}
            for item in dangers
        ],
        "exploration_time": round(exploration_time, 2),
    }
    summary = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "frames": len(frames),
        "detections": sum(len(item["detections"]) for item in events),
        "localized_detections": sum(
            item["localized"] for event in events for item in event["detections"]
        ),
        "tracks": len(tracks),
        "confirmed_tracks": confirmed_tracks,
        "reported_danger_count": len(dangers),
        "spatial_quality_gate_passed": bool(dangers),
        "official_result_generated": bool(dangers),
    }
    args.output_dir.mkdir(parents=True)
    write_jsonl(args.output_dir / "danger_detection_events.jsonl", events)
    write_jsonl(args.output_dir / "decision_trace.jsonl", traces)
    result_name = (
        "detected_danger.json"
        if dangers
        else "detected_danger.rejected_candidate.json"
    )
    for name, payload in ((result_name, result), ("replay_summary.json", summary)):
        path = args.output_dir / name
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
            stream.write("\n")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
