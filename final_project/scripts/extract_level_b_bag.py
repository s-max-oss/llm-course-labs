#!/usr/bin/env python3
"""Export synchronized RGB-D frames from the historical Gazebo-pose-oracle bag."""

from __future__ import annotations

import argparse
from bisect import bisect_left
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


REQUIRED_TOPICS = (
    "/clock",
    "/real_sense/rgb/image_raw",
    "/real_sense/rgb/camera_info",
    "/real_sense/depth/image_raw",
    "/real_sense/depth/camera_info",
    "/Odometry_gazebo",
    "/tf",
    "/tf_static",
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stamp_ns(message: Any) -> int:
    stamp = message.header.stamp
    return int(stamp.secs) * 1_000_000_000 + int(stamp.nsecs)


def nearest_by_stamp(
    ordered: Sequence[tuple[int, Any]], target_ns: int
) -> tuple[int, Any, int]:
    if not ordered:
        raise ValueError("cannot match against an empty message sequence")
    stamps = [item[0] for item in ordered]
    index = bisect_left(stamps, target_ns)
    candidates = []
    if index < len(ordered):
        candidates.append(ordered[index])
    if index:
        candidates.append(ordered[index - 1])
    stamp, message = min(candidates, key=lambda item: abs(item[0] - target_ns))
    return stamp, message, abs(stamp - target_ns)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol-id",
        choices=("level_b_replay_v1", "level_b_replay_v2"),
        default="level_b_replay_v1",
    )
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--sample-stride", type=int, default=7)
    parser.add_argument("--max-rgb-depth-delta-ms", type=float, default=20.0)
    parser.add_argument("--max-image-odom-delta-ms", type=float, default=20.0)
    parser.add_argument("--max-camera-info-delta-ms", type=float, default=1000.0)
    return parser.parse_args()


def transform_record(transform: Any) -> dict[str, Any]:
    return {
        "parent_frame": transform.header.frame_id,
        "child_frame": transform.child_frame_id,
        "translation": [
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z,
        ],
        "rotation_xyzw": [
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w,
        ],
    }


def main() -> int:
    args = parse_args()
    if not args.bag.is_file():
        raise FileNotFoundError(args.bag)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_dir}")
    if args.sample_stride <= 0:
        raise ValueError("sample stride must be positive")
    if args.protocol_id == "level_b_replay_v2":
        if args.protocol is None or not args.protocol.is_file():
            raise ValueError("Level-B v2 extraction requires --protocol")
        protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
        if (
            protocol.get("protocol_id") != args.protocol_id
            or protocol.get("status") != "frozen_before_v2_acquisition_and_inference"
        ):
            raise ValueError("Level-B v2 protocol is not the frozen acquisition protocol")
        synchronization = protocol["input_acceptance"]["synchronization"]
        args.max_rgb_depth_delta_ms = float(synchronization["max_rgb_depth_delta_ms"])
        args.max_image_odom_delta_ms = float(synchronization["max_image_odometry_delta_ms"])
        camera_info_limits = {
            float(synchronization["max_rgb_camera_info_delta_ms"]),
            float(synchronization["max_depth_camera_info_delta_ms"]),
        }
        if len(camera_info_limits) != 1:
            raise ValueError("extractor requires equal frozen RGB/depth CameraInfo limits")
        args.max_camera_info_delta_ms = camera_info_limits.pop()
        if tuple(protocol["acquisition"]["required_topics"]) != REQUIRED_TOPICS:
            raise ValueError("extractor topic contract differs from frozen protocol")

    import cv2
    import numpy as np
    import rosbag
    from cv_bridge import CvBridge

    with rosbag.Bag(str(args.bag), "r") as bag:
        topic_info = bag.get_type_and_topic_info().topics
        topics = set(topic_info)
        if topics != set(REQUIRED_TOPICS):
            raise RuntimeError(
                f"bag topic set differs from frozen allowlist: {sorted(topics)}"
            )
        forbidden = sorted(name for name in topics if name.startswith("/ground_truth"))
        if forbidden:
            raise RuntimeError(f"ground-truth topics are present in the bag: {forbidden}")

        messages: dict[str, list[tuple[int, Any]]] = {
            name: [] for name in REQUIRED_TOPICS
        }
        static_transforms = []
        for topic, message, _ in bag.read_messages(topics=list(REQUIRED_TOPICS)):
            if topic == "/tf_static":
                static_transforms.extend(message.transforms)
                continue
            if hasattr(message, "header"):
                messages[topic].append((stamp_ns(message), message))
        for sequence in messages.values():
            sequence.sort(key=lambda item: item[0])

        rgb_messages = messages["/real_sense/rgb/image_raw"]
        depth_messages = messages["/real_sense/depth/image_raw"]
        rgb_info_messages = messages["/real_sense/rgb/camera_info"]
        depth_info_messages = messages["/real_sense/depth/camera_info"]
        odometry_messages = messages["/Odometry_gazebo"]

        accepted = []
        rejected_counts = {
            "rgb_depth_delta": 0,
            "rgb_info_delta": 0,
            "depth_info_delta": 0,
            "image_odom_delta": 0,
        }
        for rgb_stamp, rgb_message in rgb_messages:
            depth_stamp, depth_message, rgb_depth_delta = nearest_by_stamp(
                depth_messages, rgb_stamp
            )
            _, rgb_info, rgb_info_delta = nearest_by_stamp(
                rgb_info_messages, rgb_stamp
            )
            _, depth_info, depth_info_delta = nearest_by_stamp(
                depth_info_messages, depth_stamp
            )
            _, odometry, image_odom_delta = nearest_by_stamp(
                odometry_messages, rgb_stamp
            )
            if rgb_depth_delta > args.max_rgb_depth_delta_ms * 1_000_000:
                rejected_counts["rgb_depth_delta"] += 1
                continue
            if rgb_info_delta > args.max_camera_info_delta_ms * 1_000_000:
                rejected_counts["rgb_info_delta"] += 1
                continue
            if depth_info_delta > args.max_camera_info_delta_ms * 1_000_000:
                rejected_counts["depth_info_delta"] += 1
                continue
            if image_odom_delta > args.max_image_odom_delta_ms * 1_000_000:
                rejected_counts["image_odom_delta"] += 1
                continue
            accepted.append(
                {
                    "rgb_stamp_ns": rgb_stamp,
                    "rgb": rgb_message,
                    "depth_stamp_ns": depth_stamp,
                    "depth": depth_message,
                    "rgb_info": rgb_info,
                    "depth_info": depth_info,
                    "odometry": odometry,
                    "rgb_depth_delta_ms": rgb_depth_delta / 1_000_000.0,
                    "image_odom_delta_ms": image_odom_delta / 1_000_000.0,
                }
            )

        if not accepted:
            raise RuntimeError("no RGB frame passed the frozen synchronization gates")
        selected = accepted[:: args.sample_stride]
        if selected[-1] is not accepted[-1]:
            selected.append(accepted[-1])

        required_static_edges = {
            ("base", "real_sense"),
            ("real_sense", "real_sense_optical_frame"),
        }
        static_by_edge = {
            (item.header.frame_id, item.child_frame_id): item
            for item in static_transforms
        }
        missing_edges = sorted(required_static_edges - set(static_by_edge))
        if missing_edges:
            raise RuntimeError(f"required static TF edges are absent: {missing_edges}")

        args.output_dir.mkdir(parents=True)
        frames_dir = args.output_dir / "frames"
        frames_dir.mkdir()
        bridge = CvBridge()
        records = []
        for index, pair in enumerate(selected):
            rgb_message = pair["rgb"]
            depth_message = pair["depth"]
            if (
                rgb_message.encoding != "rgb8"
                or rgb_message.width != 640
                or rgb_message.height != 480
            ):
                raise RuntimeError("RGB message violates the frozen image contract")
            if (
                depth_message.encoding != "32FC1"
                or depth_message.width != 640
                or depth_message.height != 480
            ):
                raise RuntimeError("depth message violates the frozen image contract")

            rgb = bridge.imgmsg_to_cv2(rgb_message, desired_encoding="rgb8")
            depth = bridge.imgmsg_to_cv2(depth_message, desired_encoding="32FC1")
            rgb_path = frames_dir / f"frame_{index:03d}_rgb.png"
            depth_path = frames_dir / f"frame_{index:03d}_depth.npz"
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(rgb_path), bgr):
                raise RuntimeError(f"failed to write {rgb_path}")
            np.savez_compressed(depth_path, depth=np.asarray(depth, dtype=np.float32))

            odometry = pair["odometry"]
            rgb_info = pair["rgb_info"]
            if (
                rgb_info.header.frame_id != "real_sense_optical_frame"
                or rgb_info.width != 640
                or rgb_info.height != 480
                or not np.isfinite([rgb_info.K[0], rgb_info.K[4]]).all()
                or rgb_info.K[0] <= 0
                or rgb_info.K[4] <= 0
            ):
                raise RuntimeError("RGB CameraInfo violates the frozen calibration contract")
            records.append(
                {
                    "schema_version": 1,
                    "frame_id": f"level_b_frame_{index:03d}",
                    "sequence_index": index,
                    "rgb": {
                        "path": str(Path("frames") / rgb_path.name),
                        "sha256": sha256_path(rgb_path),
                        "stamp_ns": pair["rgb_stamp_ns"],
                        "frame_id": rgb_message.header.frame_id,
                        "encoding": rgb_message.encoding,
                        "width": rgb_message.width,
                        "height": rgb_message.height,
                    },
                    "depth": {
                        "path": str(Path("frames") / depth_path.name),
                        "sha256": sha256_path(depth_path),
                        "stamp_ns": pair["depth_stamp_ns"],
                        "frame_id": depth_message.header.frame_id,
                        "encoding": depth_message.encoding,
                        "width": depth_message.width,
                        "height": depth_message.height,
                    },
                    "camera_info": {
                        "frame_id": rgb_info.header.frame_id,
                        "width": rgb_info.width,
                        "height": rgb_info.height,
                        "distortion_model": rgb_info.distortion_model,
                        "D": list(rgb_info.D),
                        "K": list(rgb_info.K),
                        "R": list(rgb_info.R),
                        "P": list(rgb_info.P),
                    },
                    "odometry": {
                        "stamp_ns": stamp_ns(odometry),
                        "frame_id": odometry.header.frame_id,
                        "child_frame_id": odometry.child_frame_id,
                        "position": [
                            odometry.pose.pose.position.x,
                            odometry.pose.pose.position.y,
                            odometry.pose.pose.position.z,
                        ],
                        "orientation_xyzw": [
                            odometry.pose.pose.orientation.x,
                            odometry.pose.pose.orientation.y,
                            odometry.pose.pose.orientation.z,
                            odometry.pose.pose.orientation.w,
                        ],
                    },
                    "synchronization": {
                        "rgb_depth_delta_ms": pair["rgb_depth_delta_ms"],
                        "image_odom_delta_ms": pair["image_odom_delta_ms"],
                    },
                }
            )

        manifest = {
            "schema_version": 1,
            "protocol_id": args.protocol_id,
            "source_bag": {
                "filename": args.bag.name,
                "sha256": sha256_path(args.bag),
                "topics": sorted(topics),
                "ground_truth_topics_present": False,
            },
            "synchronization_policy": {
                "max_rgb_depth_delta_ms": args.max_rgb_depth_delta_ms,
                "max_image_odometry_delta_ms": args.max_image_odom_delta_ms,
                "max_camera_info_delta_ms": args.max_camera_info_delta_ms,
                "sample_stride": args.sample_stride,
            },
            "message_counts": {
                "rgb_total": len(rgb_messages),
                "depth_total": len(depth_messages),
                "accepted_pairs": len(accepted),
                "exported_frames": len(records),
                "rejected": rejected_counts,
            },
            "static_transforms": [
                transform_record(static_by_edge[edge])
                for edge in sorted(required_static_edges)
            ],
            "records": records,
        }
        manifest_path = args.output_dir / "input_manifest.json"
        with manifest_path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
            stream.write("\n")
        print(manifest_path)
        print(json.dumps(manifest["message_counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
